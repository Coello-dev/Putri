import torch
import torch.nn as nn
import logging
from tqdm import tqdm
from src.evaluation import evaluate_perplexity
from src.utilities import (
    unmaskModel,
    maskModel,
    get_mlp_hidden_state_and_attention_output,
)


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
def prune_attention_ppl(model, num_prune, calibration_input_ids):
    num_blocks = len(model.model.layers)
    attnMask = [0] * num_blocks
    mlpMask = [0] * num_blocks

    ppl = evaluate_perplexity(
        model, calibration_input_ids, seq_len=2048, enable_tqdm=False
    )
    logging.debug(f"Original perplexity: {ppl}")

    for _ in tqdm(range(num_prune), desc="Second stage"):
        # Find the best attention to prune
        best_to_prune = None
        best_ppl = float("inf")

        for to_prune in range(num_blocks):
            # Cannot prune a block twice
            if attnMask[to_prune] == 1:
                continue

            # mask the model
            attnMask[to_prune] = 1
            maskModel(model, attnMask=attnMask, mlpMask=mlpMask)

            # Evaluate
            ppl = evaluate_perplexity(
                model, calibration_input_ids, seq_len=2048, enable_tqdm=False
            )

            logging.debug(
                f"[Attention] When pruning {to_prune} perplexity is {ppl}"
            )
            if ppl < best_ppl:
                best_ppl = ppl
                best_to_prune = to_prune

            # Unmask the model
            unmaskModel(model, attnMask=attnMask, mlpMask=mlpMask)
            attnMask[to_prune] = 0

        logging.debug(
            f"[Attention] Best to prune: {best_to_prune} ({best_ppl})"
        )
        logging.debug("========================")

        attnMask[best_to_prune] = 1

        if model.config.model_type in (
            "llama",
            "mistral",
            "qwen2",
            "qwen3",
            "ministral",
            "gemma3_text",
        ):
            del model.model.layers[best_to_prune].self_attn.q_proj
            del model.model.layers[best_to_prune].self_attn.k_proj
            del model.model.layers[best_to_prune].self_attn.v_proj
            del model.model.layers[best_to_prune].self_attn.o_proj
        elif model.config.model_type == "phi3":
            del model.model.layers[best_to_prune].self_attn.qkv_proj
            del model.model.layers[best_to_prune].self_attn.o_proj
        elif model.config.model_type == "phi":
            del model.model.layers[best_to_prune].self_attn.q_proj
            del model.model.layers[best_to_prune].self_attn.k_proj
            del model.model.layers[best_to_prune].self_attn.v_proj
            del model.model.layers[best_to_prune].self_attn.dense
        else:
            logging.error(f"Error: {model.config.model_type} is not supported")
            exit(0)

    return attnMask, mlpMask


@torch.no_grad()
def prune_attention_ppl_reconstruction(
    model,
    num_prune,
    calibration_input_ids,
    calibration_data_reconstruction,
    update_parameters=False,
):
    num_blocks = len(model.model.layers)
    attnMask = [0] * num_blocks
    mlpMask = [0] * num_blocks

    ppl = evaluate_perplexity(
        model, calibration_input_ids, seq_len=2048, enable_tqdm=False
    )
    logging.debug(f"Original perplexity: {ppl}")

    model.cuda()
    for _ in tqdm(range(num_prune), desc="Second stage (attention pruning)"):
        # Find the best attention to prune
        best_to_prune = None
        best_ppl = float("inf")

        if update_parameters:
            maskModel(model, attnMask=attnMask, mlpMask=mlpMask)
            w_increase = calculate_mlp_update_when_attn_pruned(
                model, calibration_data_reconstruction, attnMask
            )
            unmaskModel(model, attnMask=attnMask, mlpMask=mlpMask)

        for to_prune in range(num_blocks):
            # Cannot prune a block twice
            if attnMask[to_prune] == 1:
                continue

            # mask the model
            attnMask[to_prune] = 1
            if to_prune != 0 and update_parameters:
                w_old = get_mlp_down_weight(model, to_prune)
                dtype = w_old.dtype
                w_new = (w_increase[to_prune].cuda().T + w_old).to(dtype)
                set_mlp_down_weight(model, to_prune, w_new)
            maskModel(model, attnMask=attnMask, mlpMask=mlpMask)

            # Evaluate
            model.cuda()
            ppl = evaluate_perplexity(
                model, calibration_input_ids, seq_len=2048, enable_tqdm=False
            )

            logging.debug(
                f"[Attention] When pruning {to_prune} perplexity is {ppl}"
            )
            if ppl < best_ppl:
                best_ppl = ppl
                best_to_prune = to_prune

            # Unmask the model
            if to_prune != 0 and update_parameters:
                set_mlp_down_weight(model, to_prune, w_old)
            unmaskModel(model, attnMask=attnMask, mlpMask=mlpMask)
            attnMask[to_prune] = 0

        logging.debug(
            f"[Attention] Best to prune: {best_to_prune} ({best_ppl})"
        )
        logging.debug("========================")

        attnMask[best_to_prune] = 1
        if best_to_prune != 0 and update_parameters:
            w_old = get_mlp_down_weight(model, best_to_prune)
            dtype = w_old.dtype
            w_new = (w_increase[best_to_prune].cuda().T + w_old).to(dtype)
            set_mlp_down_weight(model, to_prune, w_new)

        if model.config.model_type in (
            "llama",
            "mistral",
            "qwen2",
            "qwen3",
            "ministral",
            "gemma3_text",
        ):
            del model.model.layers[best_to_prune].self_attn.q_proj
            del model.model.layers[best_to_prune].self_attn.k_proj
            del model.model.layers[best_to_prune].self_attn.v_proj
            del model.model.layers[best_to_prune].self_attn.o_proj
        elif model.config.model_type == "phi3":
            del model.model.layers[best_to_prune].self_attn.qkv_proj
            del model.model.layers[best_to_prune].self_attn.o_proj
        elif model.config.model_type == "phi":
            del model.model.layers[best_to_prune].self_attn.q_proj
            del model.model.layers[best_to_prune].self_attn.k_proj
            del model.model.layers[best_to_prune].self_attn.v_proj
            del model.model.layers[best_to_prune].self_attn.dense
        else:
            logging.error(f"Error: {model.config.model_type} is not supported")
            exit(0)

    return attnMask, mlpMask


def calculate_mlp_update_when_attn_pruned(
    model, calibration_data_reconstruction, attn_mask
):

    num_blocks = len(model.model.layers)
    xtx = {i: None for i in range(1, num_blocks)}
    xy = {i: None for i in range(1, num_blocks)}
    for sample in tqdm(
        calibration_data_reconstruction,
        total=len(calibration_data_reconstruction),
        desc="Calculating weight update",
    ):
        if sample.shape[1]:
            model.cuda()
            mlp_states, attention_states = (
                get_mlp_hidden_state_and_attention_output(
                    model, sample, attn_mask
                )
            )
            model.cpu()

            for li in range(1, num_blocks):
                if attn_mask[li] == 0:
                    x_i = mlp_states[li - 1].float().cuda()
                    y_i = attention_states[li].float().cuda()
                    if xtx[li] is None:
                        xtx[li] = x_i.T @ x_i
                        xy[li] = x_i.T @ y_i
                    else:
                        xtx[li] += x_i.T @ x_i
                        xy[li] += x_i.T @ y_i

            x_i = x_i.cpu()
            y_i = y_i.cpu()

    # Calculate reconstruction values
    w_increase = {i: None for i in range(1, num_blocks)}
    for li in range(1, num_blocks):
        if attn_mask[li] == 0:
            xtx_i = xtx[li]
            xy_i = xy[li]
            xtx_i = torch.linalg.cholesky(xtx_i)
            xtx_i = torch.cholesky_inverse(xtx_i)
            wi = xtx_i @ xy_i
            w_increase[li] = wi.cpu()
            xtx[li] = None
            xy[li] = None

    # Return reconstruction
    return w_increase


def get_mlp_down_weight(model, layer_idx):
    if model.config.model_type in (
        "llama",
        "mistral",
        "phi3",
        "qwen2",
        "qwen3",
        "ministral",
        "gemma3_text",
    ):
        return model.model.layers[layer_idx].mlp.down_proj.weight.data
    else:
        return model.model.layers[layer_idx].mlp.fc2.weight.data


def set_mlp_down_weight(model, layer_idx, weight_data):
    if model.config.model_type in (
        "llama",
        "mistral",
        "phi3",
        "qwen2",
        "qwen3",
        "ministral",
        "gemma3_text",
    ):
        model.model.layers[layer_idx].mlp.down_proj.weight.data = weight_data
    else:
        model.model.layers[layer_idx].mlp.fc2.weight.data = weight_data
