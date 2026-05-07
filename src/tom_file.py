import torch
from typing import Optional, Tuple


def feat_norm(z, num_preserve, prev_computation=None):
    """
    Select features with the lowest L2 norm across the dataset.

    Args:
        z (torch.Tensor): Input tensor of shape (B, M).
        num_preserve (int): Number of features to preserve (must be <= M).
        prev_computation (torch.Tensor, optional): Running sum of norms from previous batches (shape: M).

    Returns:
        selected_indices (torch.Tensor): Indices of preserved features (lowest norms).
        updated_norms (torch.Tensor): Updated accumulated norms for future use.
    """

    # Compute L2 norm per feature (column-wise)
    average_norms = z.norm(dim=0, p=2) ** 2

    # Accumulate with previous norms if provided
    if prev_computation is not None:
        average_norms += prev_computation

    # Select features with smallest norms
    _, top_indices = torch.topk(average_norms, num_preserve)

    return top_indices, average_norms


def twoNN(z, num_preserve, prev_computation=None):
    raise NotImplementedError()


if __name__ == "__main__":
    # Hyperparameter set-up (dummy values)
    input_size = 5
    output_size = 2
    batch_size = 10
    num_batches = 5
    dataset_size = batch_size * num_batches
    num_preserve = 3

    # Hyperparameter set-up (real values)
    input_size = 3072
    output_size = 1024
    batch_size = 2048
    num_batches = 5
    dataset_size = batch_size * num_batches
    num_preserve = 2779

    # Set seed
    seed = 0
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # Generate example
    x = torch.rand(dataset_size, input_size)
    y = torch.rand(dataset_size, output_size)

    # Run example
    if True:
        prev_computation = None
        for i in range(num_batches):
            z = x[i * batch_size : (i + 1) * batch_size]
            top_idx, prev_computation = feat_norm(
                z, num_preserve=num_preserve, prev_computation=prev_computation
            )
        print(
            f"Batching:\n - Top idxs: {top_idx}\n - Prev comp: {prev_computation}"
        )

        top_idx_full, prev_computation_full = feat_norm(
            x, num_preserve=num_preserve, prev_computation=None
        )
        print(
            f"Full data:\n - Top idxs: {top_idx_full}\n - Prev comp: {prev_computation_full}"
        )

        assert torch.all(torch.eq(top_idx, top_idx_full)), (
            "Incorrect Idxs match between batching and full data"
        )

        assert torch.allclose(
            prev_computation, prev_computation_full, atol=1e-6
        ), f"Max diff: {(prev_computation - prev_computation_full).abs().max()}"

    # Run twoNN
    prev_computation = None
    for i in range(num_batches):
        z = x[i * batch_size : (i + 1) * batch_size]
        top_idx, prev_computation = twoNN(
            z, num_preserve=num_preserve, prev_computation=prev_computation
        )
    print(
        f"Batching:\n - Top idxs: {top_idx}\n - Prev comp: {prev_computation}"
    )

    top_idx_full, prev_computation_full = twoNN(
        x, num_preserve=num_preserve, prev_computation=None
    )
    print(
        f"Full data:\n - Top idxs: {top_idx_full}\n - Prev comp: {prev_computation_full}"
    )

    assert torch.all(torch.eq(top_idx, top_idx_full)), (
        "Incorrect Idxs match between batching and full data"
    )

    assert torch.allclose(prev_computation, prev_computation_full, atol=1e-6), (
        f"Max diff: {(prev_computation - prev_computation_full).abs().max()}"
    )
