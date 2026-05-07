from .utilities import compute_intermediate_outputs
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging

"""
Args:
  model (torch.nn.Module): The transformer model to prune
  num_prune (int): The number of blocks to prune.
  calibration_dataset (Iterable): A dataset used to compute intermediate outputs 
                                   for similarity comparisons.

Returns:
  list[int]: A binary mask representing the pruning decision for each block.
.
"""


@torch.no_grad()
def merge(model, num_prune, calibration_dataset, temperature=2.0):

    intermediate_cosSim = compute_intermediate_cossim_outputs(
        model, calibration_dataset, last_token=False
    )
    num_blocks = intermediate_cosSim[0].size(0)

    block_similarity = [0.0] * num_blocks

    # average over all the calibration samples
    for ci in range(len(intermediate_cosSim)):
        for li in range(num_blocks):
            # the influence of a block is based on the cosine similarity between the
            # input and the output of the block
            block_similarity[li] += intermediate_cosSim[ci][li].mean()

    # average cosine "distance"
    block_influence = [
        1 - bi.item() / len(intermediate_cosSim) for bi in block_similarity
    ]

    # prune the layers with the lowest influence
    to_prune = sorted(
        range(len(block_influence)), key=lambda i: block_influence[i]
    )[:num_prune]
    mask_layers = [0] * num_blocks
    for i in to_prune:
        mask_layers[i] = 1
    logging.debug(
        f"Mask calculated: {mask_layers} - Layer importance: {torch.Tensor(block_influence)}"
    )

    # Merge layers
    previous_layer_masked = 0
    for i, mask in enumerate(mask_layers):
        if mask:
            previous_layer_masked += 1
        elif previous_layer_masked:
            start_layer = i - previous_layer_masked
            end_layer = i
            layer_importance = F.softmax(
                torch.Tensor(block_influence[start_layer : end_layer + 1])
                * temperature,
                dim=0,
            )
            logging.debug(
                f"Merging layers: [{start_layer}/{end_layer}] - Masks: {mask_layers[start_layer : end_layer + 1]} - Importance: {torch.Tensor(block_influence[start_layer : end_layer + 1])} - Ratio: {layer_importance}"
            )
            merge_layers(
                model=model,
                start_layer=start_layer,
                end_layer=end_layer,
                layer_importance=layer_importance,
            )
            previous_layer_masked = 0

    return mask_layers


@torch.no_grad()
def mergeshort(model, num_prune, calibration_dataset, temperature=2.0):

    intermediate_cosSim = compute_intermediate_cossim_outputs_full(
        model, calibration_dataset, last_token=False
    )
    num_blocks = intermediate_cosSim[0].size(0)

    block_similarity = [0.0] * num_blocks

    # average over all the calibration samples
    for ci in range(len(intermediate_cosSim)):
        for li in range(num_blocks):
            # the influence of a block is based on the cosine similarity between the
            # input and the output of the block
            block_similarity[li] += intermediate_cosSim[ci][li].mean()

    # average cosine "distance"
    block_influence = [
        1 - bi.item() / len(intermediate_cosSim) for bi in block_similarity
    ]

    # Get parameters per layer
    l = model.model.layers[0]
    params_per_layer = sum([t.numel() for t in l.parameters()])
    params_attn = sum([t.numel() for t in l.self_attn.parameters()])
    params_mlp = sum([t.numel() for t in l.mlp.parameters()])
    non_attn_or_mlp_params = params_per_layer - params_attn - params_mlp
    params_attn += non_attn_or_mlp_params / 2
    params_mlp += non_attn_or_mlp_params / 2
    params_to_prune = params_per_layer * num_prune

    # prune the layers with the lowest influence
    pruned_order = sorted(
        range(len(block_influence)), key=lambda i: block_influence[i]
    )
    params_pruned = 0
    count = 0
    enough_params = False
    masked_attn = [0] * num_blocks
    masked_mlp = [0] * num_blocks
    mask_layers = [0] * 2 * num_blocks
    while not enough_params:
        mask_sublayer = pruned_order[count]
        count += 1
        if mask_sublayer % 2 == 0:  # Attn layer
            params_pruned += params_attn
            masked_attn[mask_sublayer // 2] = 1
        else:
            params_pruned += params_mlp
            masked_mlp[(mask_sublayer - 1) // 2] = 1
        mask_layers[mask_sublayer] = 1
        if params_pruned >= params_to_prune:
            enough_params = True
    logging.debug(
        f"Mask calculated:\n - Sublayers masked ({count}/{2 * num_blocks}): {mask_layers} \n - Attn: {masked_attn} - {torch.Tensor([x for i, x in enumerate(block_influence) if i % 2 == 0])}\n - MLP: {masked_mlp} - {torch.Tensor([x for i, x in enumerate(block_influence) if i % 2 == 1])}"
    )

    # Merge layers
    previous_layer_masked = 0
    for i, mask in enumerate(mask_layers):
        if mask:
            previous_layer_masked += 1
        elif previous_layer_masked:
            start_layer = i - previous_layer_masked
            end_layer = i
            layer_importance = F.softmax(
                torch.Tensor(block_influence[start_layer : end_layer + 1])
                * temperature,
                dim=0,
            )
            logging.debug(
                f"Merging layers: [{start_layer}/{end_layer}] - Masks: {mask_layers[start_layer : end_layer + 1]} - Importance: {torch.Tensor(block_influence[start_layer : end_layer + 1])} - Ratio: {layer_importance}"
            )
            merge_sublayers(
                model=model,
                start_layer=start_layer,
                end_layer=end_layer,
                layer_importance=layer_importance,
            )
            previous_layer_masked = 0

    return masked_attn, masked_mlp


def merge_layers(model, start_layer, end_layer, layer_importance):
    assert (
        0 <= start_layer
        and start_layer < end_layer
        and end_layer <= len(model.model.layers)
    )

    # Get all weights
    weights_dict = {}
    for name, tensor in model.model.layers[start_layer].named_parameters():
        weights_dict[name] = [tensor]
    for i in range(start_layer + 1, end_layer + 1):
        for name, tensor in model.model.layers[i].named_parameters():
            weights_dict[name] += [tensor]

    # Merge in one tensor
    for key, value in weights_dict.items():
        # Get all tensors to merge
        u_full = []
        s_full = []
        v_full = []
        valid_matrix = True
        total_dims = min(value[0].shape)
        layers_dims = torch.round(
            torch.Tensor([total_dims * ratio for ratio in layer_importance])
        ).to(torch.int)
        layers_dims[-1] += total_dims - torch.sum(layers_dims)
        for matrix, num_dims in zip(value, layers_dims):
            dtype = matrix.dtype
            matrix = matrix.to(torch.float32)  # SVD does not work with BF16
            if len(matrix.shape) == 1:  # Normalization layers
                logging.debug(f"Layer {key} skipped due to being a vector.")
                valid_matrix = False
                if "norm" in key:
                    layer_start = model.model.layers[start_layer]
                    for attr in key.split("."):
                        layer_start = getattr(layer_start, attr)
                    layer_end = model.model.layers[end_layer]
                    for attr in key.split("."):
                        layer_end = getattr(layer_end, attr)
                    layer_end.data = layer_start.data
                break
            u, s, v = torch.linalg.svd(matrix.cuda())
            num_dims = num_dims.to(torch.int)
            u_full.append(u[:, :num_dims])
            s_full.append(s[:num_dims])
            v_full.append(v[:num_dims, :])

        if valid_matrix:
            u_full = torch.cat(u_full, axis=1)
            s_full = torch.cat(s_full)
            v_full = torch.cat(v_full)

            # Merge tensors
            uu, _, uv = torch.linalg.svd(u_full)
            vu, _, vv = torch.linalg.svd(v_full)
            ku = min(uu.shape[1], uv.shape[0])
            kv = min(vu.shape[1], vv.shape[0])
            u_orth = uu[:, :ku] @ uv[:ku, :]
            v_orth = vu[:, :kv] @ vv[:kv, :]
            k = min(u_orth.shape[1], v_orth.shape[0])
            w = (
                u_orth[:, :k].to(dtype).cuda()
                @ torch.diag(s).to(dtype).cuda()
                @ v_orth[:k, :].to(dtype).cuda()
            )

            # Update layer
            layer = model.model.layers[start_layer]
            for attr in key.split("."):
                layer = getattr(layer, attr)
            layer.data = w

    # Remove stuff from cuda
    u = u.cpu()
    s = s.cpu()
    v = v.cpu()
    if isinstance(u_full, torch.Tensor):
        u_full = u_full.cpu()
        v_full = v_full.cpu()
        s_full = s_full.cpu()
    uu = uu.cpu()
    uv = uv.cpu()
    vu = vu.cpu()
    vv = vv.cpu()
    u_orth = u_orth.cpu()
    v_orth = v_orth.cpu()
    w = w.cpu()


def merge_sublayers(model, start_layer, end_layer, layer_importance):
    assert (
        0 <= start_layer
        and start_layer < end_layer
        and end_layer <= 2 * len(model.model.layers)
    )

    # Get all weights
    weights_dict = {}
    for name, tensor in model.model.layers[start_layer].named_parameters():
        weights_dict[name] = [tensor]
    for i in range(start_layer + 1, end_layer + 1):
        for name, tensor in model.model.layers[i].named_parameters():
            weights_dict[name] += [tensor]

    # Merge in one tensor
    for key, value in weights_dict.items():
        # Get all tensors to merge
        u_full = []
        s_full = []
        v_full = []
        valid_matrix = True
        total_dims = min(value[0].shape)
        layers_dims = torch.round(
            torch.Tensor([total_dims * ratio for ratio in layer_importance])
        ).to(torch.int)
        layers_dims[-1] += total_dims - torch.sum(layers_dims)
        for matrix, num_dims in zip(value, layers_dims):
            dtype = matrix.dtype
            matrix = matrix.to(torch.float32)  # SVD does not work with BF16
            if len(matrix.shape) == 1:  # Normalization layers
                logging.debug(f"Layer {key} skipped due to being a vector.")
                valid_matrix = False
                if "norm" in key:
                    layer_start = model.model.layers[start_layer]
                    for attr in key.split("."):
                        layer_start = getattr(layer_start, attr)
                    layer_end = model.model.layers[end_layer]
                    for attr in key.split("."):
                        layer_end = getattr(layer_end, attr)
                    layer_end.data = layer_start.data
                break
            u, s, v = torch.linalg.svd(matrix.cuda())
            num_dims = num_dims.to(torch.int)
            u_full.append(u[:, :num_dims])
            s_full.append(s[:num_dims])
            v_full.append(v[:num_dims, :])

        if valid_matrix:
            u_full = torch.cat(u_full, axis=1)
            s_full = torch.cat(s_full)
            v_full = torch.cat(v_full)

            # Merge tensors
            uu, _, uv = torch.linalg.svd(u_full)
            vu, _, vv = torch.linalg.svd(v_full)
            ku = min(uu.shape[1], uv.shape[0])
            kv = min(vu.shape[1], vv.shape[0])
            u_orth = uu[:, :ku] @ uv[:ku, :]
            v_orth = vu[:, :kv] @ vv[:kv, :]
            k = min(u_orth.shape[1], v_orth.shape[0])
            w = (
                u_orth[:, :k].to(dtype).cuda()
                @ torch.diag(s).to(dtype).cuda()
                @ v_orth[:k, :].to(dtype).cuda()
            )

            # Update layer
            layer = model.model.layers[start_layer]
            for attr in key.split("."):
                layer = getattr(layer, attr)
            layer.data = w

    # Remove stuff from cuda
    u = u.cpu()
    s = s.cpu()
    v = v.cpu()
    if isinstance(u_full, torch.Tensor):
        u_full = u_full.cpu()
        v_full = v_full.cpu()
        s_full = s_full.cpu()
    uu = uu.cpu()
    uv = uv.cpu()
    vu = vu.cpu()
    vv = vv.cpu()
    u_orth = u_orth.cpu()
    v_orth = v_orth.cpu()
    w = w.cpu()


def compute_intermediate_cossim_outputs(model, calibration_set, last_token):
    intermediate_outputs = []
    for ci in tqdm(range(len(calibration_set)), desc="Intermediate outputs"):
        model.intermediate_outputs_ci = []

        # Set a forward hook to extract the intermediate output
        def hook(module, input, output):
            if last_token:
                # focus on the similarity on the last token in each sequence
                cosSim = F.cosine_similarity(
                    output[0][0, -1], input[0][0, -1], dim=0
                )
            else:
                # focus on the similarity on the entire sequence
                cosSim = F.cosine_similarity(output[0][0], input[0][0], dim=1)
            model.intermediate_outputs_ci.append(cosSim.to("cpu"))

        hooks = []
        for layer in model.model.layers:
            hooks.append(
                layer.register_forward_hook(
                    lambda module, input, output: hook(module, input, output)
                )
            )

        # Move input_ids to GPU + forward
        input_ids = calibration_set[ci].to(model.device)
        if input_ids.shape[1] > 0:  # Avoid passing empty inputs
            with torch.no_grad():
                _ = model(input_ids)
            intermediate_outputs.append(
                torch.stack(model.intermediate_outputs_ci)
            )

        # Remove all hooks
        for hook in hooks:
            hook.remove()

    return intermediate_outputs


def compute_intermediate_cossim_outputs_full(
    model, calibration_set, last_token
):
    intermediate_outputs = []
    for ci in tqdm(range(len(calibration_set)), desc="Intermediate outputs"):
        model.intermediate_outputs_ci = []
        model.intermediate_outputs_ci = []
        model.intermediate_attentions_ci = []

        # Set a forward hook to extract the intermediate output
        def hook(module, input, output):
            middle_output = (
                input[0]
                + model.intermediate_attentions_ci[module.layer_index].cuda()
            )
            if last_token:
                # focus on the similarity on the last token in each sequence
                cosSimMlp = F.cosine_similarity(
                    output[0][0, -1], middle_output[0, -1], dim=0
                )
                cosSimAttn = F.cosine_similarity(
                    middle_output[0, -1], input[0][0, -1], dim=0
                )
            else:
                # focus on the similarity on the entire sequence
                cosSimAttn = F.cosine_similarity(
                    middle_output[0], input[0][0], dim=1
                )
                cosSimMlp = F.cosine_similarity(
                    output[0][0], middle_output[0], dim=1
                )
            model.intermediate_outputs_ci.append(cosSimAttn.to("cpu"))
            model.intermediate_outputs_ci.append(cosSimMlp.to("cpu"))

        def hook_attention(module, input, output):
            model.intermediate_attentions_ci.append(output[0].to("cpu"))

        hooks = []
        for idx, layer in enumerate(model.model.layers):
            layer.layer_index = idx
            hooks.append(
                layer.register_forward_hook(
                    lambda module, input, output: hook(module, input, output)
                )
            )
            hooks.append(
                layer.self_attn.register_forward_hook(
                    lambda module, input, output: hook_attention(
                        module, input, output
                    )
                )
            )

        # Move input_ids to GPU + forward
        input_ids = calibration_set[ci].to(model.device)
        if input_ids.shape[1] > 0:  # Avoid passing empty inputs
            with torch.no_grad():
                _ = model(input_ids)
            intermediate_outputs.append(
                torch.stack(model.intermediate_outputs_ci)
            )

        # Remove all hooks
        for hook in hooks:
            hook.remove()

    return intermediate_outputs
