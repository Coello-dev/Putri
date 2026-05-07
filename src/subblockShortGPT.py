import torch
import logging
from src.merge import compute_intermediate_cossim_outputs_full


@torch.no_grad()
def subblockShortGPT(model, num_prune, calibration_dataset):

    intermediate_cosSim = compute_intermediate_cossim_outputs_full(
        model, calibration_dataset, last_token=False
    )
    num_subblocks = intermediate_cosSim[0].size(0)
    num_layers = len(model.model.layers)
    block_similarity = [0.0] * num_subblocks

    # average over all the calibration samples
    for ci in range(len(intermediate_cosSim)):
        for li in range(num_subblocks):
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
    masked_attn = [0] * num_layers
    masked_mlp = [0] * num_layers
    mask_layers = [0] * num_subblocks
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
        f"Mask calculated:\n - Sublayers masked ({count}/{num_subblocks}): {mask_layers} \n - Attn: {masked_attn} - {torch.Tensor([x for i, x in enumerate(block_influence) if i % 2 == 0])}\n - MLP: {masked_mlp} - {torch.Tensor([x for i, x in enumerate(block_influence) if i % 2 == 1])}"
    )

    return masked_attn, masked_mlp
