import torch
import torch.nn as nn
import numpy as np
import logging
from tqdm import tqdm
from src.evaluation import evaluate_perplexity, evaluate_kldiv
from src.widthmerge_utils.utils_aux_old import prune_attention_ppl
from src.widthmerge_utils.attentionpruner import AttnPruner


class Catcher(nn.Module):
    def __init__(self, module, inps, cache, seqlen=None):
        super().__init__()
        self.module = module
        self.inps = inps
        self.cache = cache
        self.seqlen = seqlen

        if hasattr(module, "attention_type"):
            self.attention_type = module.attention_type

    def forward(self, inp, **kwargs):
        inp = inp.cpu()
        if self.seqlen is None or self.seqlen == inp.shape[1]:
            self.inps[self.cache["i"]] = inp.cpu()
            self.cache["i"] += 1
            # Move everything to cpu otherwise it will stay in cuda later on
            for key, value in kwargs.items():
                if isinstance(value, torch.Tensor):
                    kwargs[key] = value.cpu()
            if "attention_mask" in kwargs:
                self.cache["catcher_attention_mask"] = kwargs["attention_mask"]
            else:
                self.cache["catcher_attention_mask"] = None
            self.cache["catcher_position_ids"] = kwargs["position_ids"]
        raise ValueError


@torch.no_grad()
def second_stage_attention_merge(
    model,
    num_prune,
    calibration_input_ids,
    calibration_data_reconstruction,
    attn_prune_method="ppl",
    update_parameters=False,
):
    if attn_prune_method == "ppl_head":
        return prune_attention_ppl_head(
            model=model,
            num_prune=num_prune,
            calibration_input_ids=calibration_input_ids,
        )
    elif attn_prune_method == "ppl_head_rec":
        return prune_attention_ppl_head_rec(
            model=model,
            num_prune=num_prune,
            calibration_input_ids=calibration_input_ids,
            calibration_data_reconstruction=calibration_data_reconstruction,
        )
    if attn_prune_method == "kldiv_head":
        return prune_attention_kldiv_head(
            model=model,
            num_prune=num_prune,
            calibration_input_ids=calibration_input_ids,
        )
    elif attn_prune_method == "ppl":
        return prune_attention_ppl(
            model=model,
            num_prune=num_prune,
            calibration_input_ids=calibration_input_ids,
        )
    else:
        raise NotImplementedError()


"""
Second-Stage Attention Pruning Merge

Analogous to Second-Stage Attention Pruning but updates output matrix.

Args:
  model (torch.nn.Module): The transformer model to prune.
  num_prune (int): The number of attention submodules to prune.
  calibration_input_ids (torch.Tensor): A calibration input tensor used to 
                          calculate perplexity and evaluate pruning decisions.

Returns:
  tuple[list[int], list[int]]: Two binary masks representing the pruning 
                               decisions for attention and MLP submodules 
                               for each block. For MLP the mask is all 0.
"""


@torch.no_grad()
def prune_attention_ppl_head(
    model,
    num_prune,
    calibration_input_ids,
):
    num_blocks = len(model.model.layers)
    if model.config.model_type in ["qwen3", "llama"]:
        num_kv_heads = model.config.num_key_value_heads
    else:
        raise NotImplementedError(
            f"Not Implemented model: {model.config.model_type}"
        )
    logging.debug(f"{num_kv_heads} heads per attention layer")
    attnHeadMask = [[0] * num_kv_heads for _ in range(num_blocks)]
    mlpMask = [0] * num_blocks

    ppl = evaluate_perplexity(
        model, calibration_input_ids, seq_len=2048, enable_tqdm=False
    )
    logging.debug(f"Original perplexity: {ppl}")

    model.cuda()
    for _ in tqdm(range(num_prune), desc="Second stage (attention pruning)"):
        # Find the best attention to prune
        ppl_results = np.inf * np.ones([num_blocks, num_kv_heads])

        for layer_idx in range(num_blocks):
            for head_idx in range(num_kv_heads):
                # Cannot prune a block twice
                if attnHeadMask[layer_idx][head_idx] == 1:
                    logging.debug(
                        f"Layer {layer_idx} Head {head_idx} already removed"
                    )
                    continue

                # mask the model
                attnHeadMask[layer_idx][head_idx] = 1
                params = mask_attention_head(
                    model=model,
                    layer_idx=layer_idx,
                    attn_head=head_idx,
                )

                # Evaluate
                model.cuda()
                ppl = evaluate_perplexity(
                    model,
                    calibration_input_ids,
                    seq_len=2048,
                    enable_tqdm=False,
                )

                logging.debug(
                    f"[Attention] When pruning layer {layer_idx} head {head_idx}, ppl is {ppl}"
                )
                ppl_results[layer_idx][head_idx] = ppl

                # Unmask the model
                unmask_attention_head(
                    model=model,
                    layer_idx=layer_idx,
                    attn_head=head_idx,
                    params=params,
                )
                attnHeadMask[layer_idx][head_idx] = 0
        ppl_shape = ppl_results.shape
        for _ in range(num_kv_heads):
            best_idx = np.argmin(ppl_results)
            best_layer_to_prune, best_head_to_prune = np.unravel_index(
                best_idx, ppl_shape
            )
            logging.debug(
                f"[Attention] Best to prune: layer {best_layer_to_prune} head {best_head_to_prune} (ppl: {ppl_results[best_layer_to_prune][best_head_to_prune]})\n========================"
            )
            attnHeadMask[best_layer_to_prune][best_head_to_prune] = 1
            _ = mask_attention_head(
                model=model,
                layer_idx=best_layer_to_prune,
                attn_head=best_head_to_prune,
            )
            ppl_results[best_layer_to_prune][best_head_to_prune] = np.inf

    delete_attn_heads(model, attnHeadMask)
    attnMask = [0] * num_blocks
    return attnMask, mlpMask


@torch.no_grad()
def prune_attention_ppl_head_rec(
    model,
    num_prune,
    calibration_input_ids,
    calibration_data_reconstruction,
):
    num_blocks = len(model.model.layers)
    if model.config.model_type in ["qwen3", "llama"]:
        num_kv_heads = model.config.num_key_value_heads
    else:
        raise NotImplementedError(
            f"Not Implemented model: {model.config.model_type}"
        )
    logging.debug(f"{num_kv_heads} heads per attention layer")
    attnHeadMask = [[0] * num_kv_heads for _ in range(num_blocks)]
    mlpMask = [0] * num_blocks
    dev = model.device

    # Calculate reupdate weight related parameters
    nsamples = (
        len(calibration_data_reconstruction)
        * calibration_data_reconstruction[0].shape[0]
    )
    inps, outs, attention_mask, position_ids, nsamples = (
        prepare_calibration_input(
            model=model,
            dataloader=calibration_data_reconstruction,
            device=dev,
            nsamples=nsamples,
        )
    )
    XY_list = {}
    H_list = {}

    layer_wrapper = AttnPruner

    inps = inps.cpu()
    outs = outs.cpu()

    if model.config.model_type in ["llama"]:
        layers = model.model.layers

        # Get position embeddings
        model.model.rotary_emb.to(dev)
        inps = inps.to(dev)
        position_embeddings = model.model.rotary_emb(
            inps[0].unsqueeze(0), position_ids
        )
        inps = inps.cpu()
        model.model.rotary_emb.cpu()

    elif model.config.model_type in ["opt"]:
        layers = model.model.decoder.layers

    elif model.config.model_type in ["qwen3"]:
        layers = model.model.layers
        model.model.rotary_emb.to(dev)
        inps = inps.to(dev)
        position_embeddings = model.model.rotary_emb(inps, position_ids)
        inps = inps.cpu()
        model.model.rotary_emb.cpu()

    else:
        raise NotImplementedError(f"Invalid model: {model.config.model_type}")

    for i in tqdm(
        range(len(layers)),
        total=len(layers),
        desc="Calculate H and XY for attn output",
    ):
        layer = layers[i]
        layer.to(dev)
        gpts = layer_wrapper(
            layer=layer, model_config=model.config, layer_idx=i
        )

        def add_batch(_, inp, out):
            gpts.add_batch(inp=inp[0].data, out=out.data)

        handles = [gpts.linear_output.register_forward_hook(add_batch)]
        old_outs = [0.0 for _ in range(len(outs))]

        for j in range(nsamples):
            with torch.no_grad():
                inp_j = inps[j].to(dev)
                if model.config.model_type in ["llama"]:
                    outs_j = layer(
                        inp_j.unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                    )[0]
                elif model.config.model_type in ["opt"]:
                    outs_j = layer(
                        inp_j.unsqueeze(0),
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                    )[0]
                elif model.config.model_type in ["qwen3"]:
                    outs_j = layer(
                        inp_j.unsqueeze(0),
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                    )[0]
                else:
                    raise NotImplementedError(
                        f"Invalid model: {model.config.model_type}"
                    )
                inp_j = inp_j.cpu()
                outs_j = outs_j.cpu()
                old_outs[j] = outs_j
        handles[0].remove()

        H, XY = gpts.free()
        XY_list[i] = XY
        H_list[i] = H
        inps, outs = old_outs, inps

        layer.cpu()
        layers[i] = layer
        torch.cuda.empty_cache()

    model.cuda()
    ppl = evaluate_perplexity(
        model, calibration_input_ids, seq_len=2048, enable_tqdm=False
    )
    logging.debug(f"Original perplexity: {ppl}")

    for _ in tqdm(range(num_prune), desc="Second stage (attention pruning)"):
        # Find the best attention to prune
        ppl_results = np.inf * np.ones([num_blocks, num_kv_heads])

        for layer_idx in range(num_blocks):
            for head_idx in range(num_kv_heads):
                # Cannot prune a block twice
                if attnHeadMask[layer_idx][head_idx] == 1:
                    logging.debug(
                        f"Layer {layer_idx} Head {head_idx} already removed"
                    )
                    continue

                # mask the model
                attnHeadMask[layer_idx][head_idx] = 1
                params = mask_attention_head(
                    model=model,
                    layer_idx=layer_idx,
                    attn_head=head_idx,
                    attnHeadMask=attnHeadMask,
                    XY_list=XY_list,
                    H_list=H_list,
                )

                # Evaluate
                model.cuda()
                ppl = evaluate_perplexity(
                    model,
                    calibration_input_ids,
                    seq_len=2048,
                    enable_tqdm=False,
                )

                logging.debug(
                    f"[Attention] When pruning layer {layer_idx} head {head_idx}, ppl is {ppl}"
                )
                ppl_results[layer_idx][head_idx] = ppl

                # Unmask the model
                unmask_attention_head(
                    model=model,
                    layer_idx=layer_idx,
                    attn_head=head_idx,
                    params=params,
                )
                attnHeadMask[layer_idx][head_idx] = 0
        ppl_shape = ppl_results.shape
        for _ in range(num_kv_heads):
            best_idx = np.argmin(ppl_results)
            best_layer_to_prune, best_head_to_prune = np.unravel_index(
                best_idx, ppl_shape
            )
            logging.debug(
                f"[Attention] Best to prune: layer {best_layer_to_prune} head {best_head_to_prune} (ppl: {ppl_results[best_layer_to_prune][best_head_to_prune]})\n========================"
            )
            attnHeadMask[best_layer_to_prune][best_head_to_prune] = 1
            _ = mask_attention_head(
                model=model,
                layer_idx=best_layer_to_prune,
                attn_head=best_head_to_prune,
            )
            ppl_results[best_layer_to_prune][best_head_to_prune] = np.inf

    delete_attn_heads(model, attnHeadMask, XY_list=XY_list, H_list=H_list)
    attnMask = [0] * num_blocks
    return attnMask, mlpMask


@torch.no_grad()
def prune_attention_kldiv_head(
    model,
    num_prune,
    calibration_input_ids,
    update_parameters=False,
):
    num_blocks = len(model.model.layers)
    if model.config.model_type in ["qwen3", "llama"]:
        num_kv_heads = model.config.num_key_value_heads
    else:
        raise NotImplementedError(
            f"Not Implemented model: {model.config.model_type}"
        )
    logging.debug(f"{num_kv_heads} heads per attention layer")
    attnHeadMask = [[0] * num_kv_heads for _ in range(num_blocks)]
    mlpMask = [0] * num_blocks

    # ppl = evaluate_perplexity(
    #     model, calibration_input_ids, seq_len=2048, enable_tqdm=False
    # )
    model.cuda()
    for _ in tqdm(range(num_prune), desc="Second stage (attention pruning)"):
        # Find the best attention to prune
        kldiv_results = np.inf * np.ones([num_blocks, num_kv_heads])

        calibration_input_ids = calibration_input_ids.to(model.device)
        logits = model(calibration_input_ids).logits.squeeze()
        probs = torch.nn.functional.softmax(logits, dim=1)
        calibration_input_ids = calibration_input_ids.cpu()
        logging.debug(f"Original output calculated")

        for layer_idx in range(num_blocks):
            for head_idx in range(num_kv_heads):
                # Cannot prune a block twice
                if attnHeadMask[layer_idx][head_idx] == 1:
                    logging.debug(
                        f"Layer {layer_idx} Head {head_idx} already removed"
                    )
                    continue

                # mask the model
                attnHeadMask[layer_idx][head_idx] = 1
                params = mask_attention_head(
                    model=model,
                    layer_idx=layer_idx,
                    attn_head=head_idx,
                )

                # Evaluate
                model.cuda()
                calibration_input_ids = calibration_input_ids.cuda()
                kl_div = evaluate_kldiv(
                    model,
                    calibration_input_ids,
                    probs,
                    seq_len=2048,
                    enable_tqdm=False,
                )
                calibration_input_ids = calibration_input_ids.cpu()

                logging.debug(
                    f"[Attention] When pruning layer {layer_idx} head {head_idx}, kl_div is {kl_div}"
                )
                kldiv_results[layer_idx][head_idx] = kl_div

                # Unmask the model
                unmask_attention_head(
                    model=model,
                    layer_idx=layer_idx,
                    attn_head=head_idx,
                    params=params,
                )
                attnHeadMask[layer_idx][head_idx] = 0
        kldiv_shape = kldiv_results.shape
        for _ in range(num_kv_heads):
            best_idx = np.argmin(kldiv_results)
            best_layer_to_prune, best_head_to_prune = np.unravel_index(
                best_idx, kldiv_shape
            )
            logging.debug(
                f"[Attention] Best to prune: layer {best_layer_to_prune} head {best_head_to_prune} (kldiv: {kldiv_results[best_layer_to_prune][best_head_to_prune]})\n========================"
            )
            attnHeadMask[best_layer_to_prune][best_head_to_prune] = 1
            _ = mask_attention_head(
                model=model,
                layer_idx=best_layer_to_prune,
                attn_head=best_head_to_prune,
            )
            kldiv_results[best_layer_to_prune][best_head_to_prune] = np.inf

    delete_attn_heads(model, attnHeadMask)
    attnMask = [0] * num_blocks
    return attnMask, mlpMask


def delete_attn_heads(model, attnHeadMask, XY_list=None, H_list=None):

    if model.config.model_type in (
        "llama",
        "mistral",
        "qwen2",
        "qwen3",
        "ministral",
        "gemma3_text",
    ):
        head_size = model.model.layers[0].self_attn.head_dim
        q_kv_ratio = model.model.layers[0].self_attn.num_key_value_groups

        for i, layer in enumerate(model.model.layers):
            layer_mask = attnHeadMask[i]
            if sum(layer_mask):
                q_columns_kept = []
                kv_columns_kept = []
                for idx, mask in enumerate(layer_mask):
                    if not mask:
                        q_columns_kept.extend(
                            list(
                                range(
                                    idx * head_size * q_kv_ratio,
                                    (idx + 1) * head_size * q_kv_ratio,
                                )
                            )
                        )
                        kv_columns_kept.extend(
                            list(
                                range(
                                    idx * head_size,
                                    (idx + 1) * head_size,
                                )
                            )
                        )

                attn_layer = layer.self_attn

                attn_layer.q_proj.weight.data = attn_layer.q_proj.weight.data[
                    q_columns_kept, :
                ]
                attn_layer.k_proj.weight.data = attn_layer.k_proj.weight.data[
                    kv_columns_kept, :
                ]
                attn_layer.v_proj.weight.data = attn_layer.v_proj.weight.data[
                    kv_columns_kept, :
                ]
                attn_layer.o_proj.weight.data = attn_layer.o_proj.weight.data[
                    :, q_columns_kept
                ]

                if XY_list is not None:
                    masked_channels = []
                    dtype = attn_layer.o_proj.weight.data.dtype
                    device = model.device
                    for idx, mask_head in enumerate(layer_mask):
                        if not mask_head:
                            masked_channels.extend(
                                list(
                                    range(
                                        idx * head_size * q_kv_ratio,
                                        (idx + 1) * head_size * q_kv_ratio,
                                    )
                                )
                            )
                    H_p = H_list[i][masked_channels][:, masked_channels]
                    XY_p = XY_list[i][masked_channels]
                    try:
                        H_p = torch.linalg.cholesky(H_p)
                        H_p = torch.cholesky_inverse(H_p)
                        updated_weights = H_p @ XY_p
                    except:
                        logging.warning(
                            "Cholesky decomposition did not work. Using standard inverse."
                        )
                        updated_weights = torch.linalg.solve(H_p, XY_p)
                    attn_layer.o_proj.weight.data = updated_weights.T.to(
                        dtype
                    ).to(device)

    elif model.config.model_type in [
        "llama",
        "mistral",
        "qwen2",
        "qwen3",
        "ministral",
        "gemma3_text",
    ]:
        raise NotImplementedError()

    elif model.config.model_type == "phi3":
        raise NotImplementedError()

    elif model.config.model_type == "phi":
        raise NotImplementedError()
    else:
        raise NotImplementedError(
            f"invalid model config: {model.config.model_type}"
        )


def mask_attention_head(
    model, layer_idx, attn_head, attnHeadMask=None, XY_list=None, H_list=None
):

    if model.config.model_type in (
        "llama",
        "mistral",
        "qwen2",
        "qwen3",
        "ministral",
        "gemma3_text",
    ):
        device = model.device
        attn_layer = model.model.layers[layer_idx].self_attn
        dtype = attn_layer.q_proj.weight.data.dtype

        q = attn_layer.q_proj.weight.data.detach().clone()
        k = attn_layer.k_proj.weight.data.detach().clone()
        v = attn_layer.v_proj.weight.data.detach().clone()
        o = attn_layer.o_proj.weight.data.detach().clone()

        hidden_size = model.config.hidden_size
        head_size = attn_layer.head_dim
        q_kv_ratio = attn_layer.num_key_value_groups

        attn_layer.q_proj.weight.data[
            head_size * q_kv_ratio * attn_head : head_size
            * q_kv_ratio
            * (attn_head + 1),
            :,
        ] = torch.zeros([head_size * q_kv_ratio, hidden_size], dtype=dtype).to(
            device
        )
        attn_layer.k_proj.weight.data[
            head_size * attn_head : head_size * (attn_head + 1),
            :,
        ] = torch.zeros([head_size, hidden_size], dtype=dtype).to(device)
        attn_layer.v_proj.weight.data[
            head_size * attn_head : head_size * (attn_head + 1),
            :,
        ] = torch.zeros([head_size, hidden_size], dtype=dtype).to(device)
        attn_layer.o_proj.weight.data[
            :,
            head_size * q_kv_ratio * attn_head : head_size
            * q_kv_ratio
            * (attn_head + 1),
        ] = torch.zeros([hidden_size, head_size * q_kv_ratio], dtype=dtype).to(
            device
        )

        if XY_list is not None:
            masked_heads = attnHeadMask[layer_idx]
            masked_channels = []
            for idx, mask_head in enumerate(masked_heads):
                if not mask_head:
                    masked_channels.extend(
                        list(
                            range(
                                idx * head_size * q_kv_ratio,
                                (idx + 1) * head_size * q_kv_ratio,
                            )
                        )
                    )
            H_p = H_list[layer_idx][masked_channels][:, masked_channels]
            XY_p = XY_list[layer_idx][masked_channels]
            try:
                H_p = torch.linalg.cholesky(H_p)
                H_p = torch.cholesky_inverse(H_p)
                updated_weights = H_p @ XY_p
            except:
                logging.warning(
                    "Cholesky decomposition did not work. Using standard inverse."
                )
                updated_weights = torch.linalg.solve(H_p, XY_p)
            attn_layer.o_proj.weight.data[:, masked_channels] = (
                updated_weights.T.to(dtype).to(device)
            )

        return {"q": q, "k": k, "v": v, "o": o}

    elif model.config.model_type == "phi3":
        qkv = (
            model.model
            .layers[layer_idx]
            .self_attn.qkv_proj.weight.data.detach()
            .clone()
        )
        o = (
            model.model
            .layers[layer_idx]
            .self_attn.o_proj.weight.data.detach()
            .clone()
        )

        raise NotImplementedError()

        return {"qkv": qkv, "o": o}
    elif model.config.model_type == "phi":
        q = (
            model.model
            .layers[layer_idx]
            .self_attn.q_proj.weight.data.detach()
            .clone()
        )
        k = (
            model.model
            .layers[layer_idx]
            .self_attn.k_proj.weight.data.detach()
            .clone()
        )
        v = (
            model.model
            .layers[layer_idx]
            .self_attn.v_proj.weight.data.detach()
            .clone()
        )
        o = (
            model.model
            .layers[layer_idx]
            .self_attn.dense.weight.data.detach()
            .clone()
        )

        raise NotImplementedError()

        return {"q": q, "k": k, "v": v, "dense": o}
    else:
        raise NotImplementedError(
            f"invalid model config: {model.config.model_type}"
        )


def unmask_attention_head(model, layer_idx, attn_head, params):

    if model.config.model_type in (
        "llama",
        "mistral",
        "qwen2",
        "qwen3",
        "ministral",
        "gemma3_text",
    ):
        model.model.layers[layer_idx].self_attn.q_proj.weight.data = params["q"]
        model.model.layers[layer_idx].self_attn.k_proj.weight.data = params["k"]
        model.model.layers[layer_idx].self_attn.v_proj.weight.data = params["v"]
        model.model.layers[layer_idx].self_attn.o_proj.weight.data = params["o"]

    elif model.config.model_type == "phi3":
        model.model.layers[layer_idx].self_attn.qkv_proj.weight.data = params[
            "qkv"
        ]
        model.model.layers[layer_idx].self_attn.o_proj.weight.data = params["o"]

    elif model.config.model_type == "phi":
        model.model.layers[layer_idx].self_attn.q_proj.weight.data = params["q"]
        model.model.layers[layer_idx].self_attn.k_proj.weight.data = params["k"]
        model.model.layers[layer_idx].self_attn.v_proj.weight.data = params["v"]
        model.model.layers[layer_idx].self_attn.dense.weight.data = params[
            "dense"
        ]
    else:
        raise NotImplementedError(
            f"invalid model config: {model.config.model_type}"
        )


@torch.no_grad()
def prepare_calibration_input(model, dataloader, device, nsamples):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    if model.config.model_type in (
        "llama",
        "mistral",
        "qwen2",
        "qwen3",
        "ministral",
        "gemma3_text",
        "phi3",
        "phi",
    ):
        layers = model.model.layers
    elif model.config.model_type in ["opt"]:
        layers = model.model.decoder.layers
    else:
        raise Exception(f"Invalid model: {model.config.model_type}")

    dtype = next(iter(model.parameters())).dtype
    seqlen = dataloader[0].shape[1]
    print(f"Device: {device}")
    inps = torch.zeros(
        (nsamples, seqlen, model.config.hidden_size),
        dtype=dtype,
        device=device,
    )
    inps.requires_grad = False
    cache = {
        "i": 0,
        "catcher_attention_mask": None,
        "catcher_position_ids": None,
    }

    layers[0] = Catcher(module=layers[0], inps=inps, cache=cache, seqlen=seqlen)
    for batch in dataloader:
        try:
            if batch.shape[1]:
                batch_i = batch.to(device)  # batch[0].to(device)
                model(batch_i)
        except ValueError:
            pass

    # Remove unused inputs
    if nsamples > cache["i"]:
        logging.warning(
            f"Less inputs obtained as expected: {cache['i']} of {nsamples}"
        )
        inps = inps[: cache["i"]]
        nsamples = cache["i"]

    outs = torch.zeros_like(layers[0].inps)
    attention_mask = layers[0].cache["catcher_attention_mask"]
    position_ids = layers[0].cache["catcher_position_ids"]
    model.config.use_cache = use_cache

    # Remove model from cuda
    batch_i.cpu()
    model.cpu()

    # Move to cuda
    if isinstance(attention_mask, torch.Tensor):
        attention_mask = attention_mask.to(device)
    if isinstance(position_ids, torch.Tensor):
        position_ids = position_ids.to(device)

    layers[0] = layers[0].module

    return inps, outs, attention_mask, position_ids, nsamples
