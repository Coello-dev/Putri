from tqdm import tqdm
import torch
import time
import logging
from src.widthmerge_utils.sparsegpt import SparseGPTPruner
from src.widthmerge_utils.twossppruner import TwoSSPPruner
from src.widthmerge_utils.utils_aux import (
    Catcher,
    second_stage_attention_merge,
    prepare_calibration_input,
)
from .utilities import maskModel

"""
Widthmerge

Implements the complete framework. In the first stage, MLP submodules are pruned.
In the second stage, attention submodules are pruned.

Args:
  model (torch.nn.Module): The transformer model to prune. calibration_dataset
  (Iterable[torch.Tensor]): A dataset of samples used for calculating norms and
                        perplexity during pruning.
  pruning_rate (float): The overall proportion of parameters to prune, 
                        relative to the total number of parameters in the
                        model's main layers.
  num_attn_submodules_to_prune (int, optional): The number of attention
                        submodules to prune. If None, this is calculated based
                        on Equation 5 in the paper
  alpha (int, optional): balancing between the number of parameters pruned in
                        the first and the second stages. See "Choice of alpha
                        parameter" in Section "4.3. Hyperparameter Tuning" of
                        the paper (Default: 1.5)
  num_calibration_second_stage (int, optional): the number of calibration
                        samples to use for the second stage (Default: 1)

Returns:
  torch.nn.Module: The pruned transformer model, with both MLP and attention 
                   submodules pruned.
"""


@torch.no_grad()
def widthmerge_aux(
    model,
    calibration_dataset,
    pruning_rate,
    merge_method="min_mse",
    score_method="l2_norm",
    attn_prune_method="ppl",
    num_attn_submodules_to_prune=None,
    alpha=1.5,
    num_calibration_second_stage=1,
    use_new_output=True,
    skip_stage_one=False,
    # dtype=torch.float32,  # torch.bfloat16 torch.float32
):

    layers = model.model.layers
    num_blocks = len(layers)

    logging.debug("======================")
    main_model_total_params = sum(
        p.numel() for p in model.model.layers.parameters()
    )
    attn_total_params = sum(
        p.numel() for p in model.model.layers[0].self_attn.parameters()
    )
    mlp_total_params = sum(
        p.numel() for p in model.model.layers[0].mlp.parameters()
    )
    logging.debug(f"Attention parameters (one block): {attn_total_params}")
    logging.debug(f"MLP parameters (one block): {mlp_total_params}")
    logging.debug("======================")

    if num_attn_submodules_to_prune is None:
        num_attn_submodules_to_prune = round(
            num_blocks
            * pow(pruning_rate, (mlp_total_params / attn_total_params) / alpha)
        )
    logging.info(f"Pruning {num_attn_submodules_to_prune} attention submodules")

    if (
        num_attn_submodules_to_prune * attn_total_params
    ) / main_model_total_params > pruning_rate:
        logging.error("Exceeded pruning parameters number")
        return False

    if (
        num_attn_submodules_to_prune * attn_total_params
        + num_blocks * mlp_total_params
    ) / main_model_total_params < pruning_rate:
        logging.error(
            f"Unable to reach the target sparsity rate with only {num_attn_submodules_to_prune} pruned attention submodules"
        )
        return False

    # 1. First stage: pruning the FFN neurons
    parameters_pruned_for_attention = (
        num_attn_submodules_to_prune * attn_total_params
    )
    target_parameters_to_prune = round(pruning_rate * main_model_total_params)

    mlp_params_to_prune = round(
        (target_parameters_to_prune - parameters_pruned_for_attention)
        / num_blocks
    )  # number of remaining parameters to prune in each single mlp block
    mlp_pruning_rate = mlp_params_to_prune / mlp_total_params

    mlp_hidden_size = model.config.intermediate_size
    num_preserve_mlp = round(mlp_hidden_size * (1 - mlp_pruning_rate))

    layer_wrapper_dict = {
        # STADE
        "l2_norm": TwoSSPPruner,
        # SparseGPT
        "sparsegpt_norm": SparseGPTPruner,
    }
    layer_wrapper = layer_wrapper_dict[score_method]

    dev = model.device
    nsamples = len(calibration_dataset) * calibration_dataset[0].shape[0]
    inps, outs, attention_mask, position_ids, nsamples = (
        prepare_calibration_input(
            model=model,
            dataloader=calibration_dataset,
            device=dev,
            nsamples=nsamples,
        )
    )
    logging.info("Calibration dataset prepared")
    # print_gpu_info(dev)

    if "parallel" in merge_method:
        use_new_output = False

    if not skip_stage_one:
        # Make it gpu memory bound efficient
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
            raise NotImplementedError(
                f"Invalid model: {model.config.model_type}"
            )

        # if not args.filter_layer is None:
        #     print(f"Layers filtered by {args.filter_layer}")
        old_outs = [0.0 for _ in range(len(outs))]

        # print_gpu_info(dev)

        for i in tqdm(
            range(len(layers)), total=len(layers), desc="Pruning layers"
        ):
            layer = layers[i]
            layer.to(dev)
            gpts = layer_wrapper(
                layer=layer, model_config=model.config, layer_idx=i, merge_method=merge_method
            )

            def add_batch(_, inp, out):
                gpts.add_batch(inp=inp[0].data, out=out.data)

            handles = [gpts.linear_output.register_forward_hook(add_batch)]

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

            gpts.prune(
                sparsity=pruning_rate,
                new_intermediate_size=num_preserve_mlp,
            )
            gpts.free()

            if use_new_output:
                for j in range(nsamples):
                    inp_j = inps[j]
                    inp_j = inp_j.to(dev)
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
                            attention_mask=attention_mask,
                            position_embeddings=position_embeddings,
                        )[0]
                    else:
                        raise NotImplementedError(
                            f"Invalid model: {model.config.model_type}"
                        )
                    inp_j = inp_j.cpu()
                    outs_j = outs_j.cpu()
                    outs[j] = outs_j

                inps, outs = outs, inps
                torch.cuda.empty_cache()

            else:
                inps, outs = old_outs, inps

            layer.cpu()
            layers[i] = layer
            torch.cuda.empty_cache()

        model.config.intermediate_size = num_preserve_mlp

    else:
        logging.warning("MLP pruning step has been skipped")

    # 2. Second stage: pruning the Attention submodules
    calibration_input_ids = torch.cat(
        calibration_dataset[:num_calibration_second_stage], dim=1
    )
    calibration_dataset_reconstruction = calibration_dataset[
        num_calibration_second_stage:
    ]

    model.cuda()
    attnMask, mlpMask = second_stage_attention_merge(
        model,
        num_prune=num_attn_submodules_to_prune,
        calibration_input_ids=calibration_input_ids,
        calibration_data_reconstruction=calibration_dataset_reconstruction,
        attn_prune_method=attn_prune_method,
    )
    maskModel(model, attnMask=attnMask, mlpMask=mlpMask)

    return model
