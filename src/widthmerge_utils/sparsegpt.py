import torch
import math
from src.widthmerge_utils.basepruner import BasePruner


class SparseGPTPruner(BasePruner):
    def __init__(self, norm=True, **kwargs):
        super().__init__(**kwargs)

        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.XY = torch.zeros((self.columns, self.rows), device=self.dev)
        self.nsamples = 0
        self.norm = norm
        if self.norm:
            self.channel_norm = torch.zeros((self.rows), device=self.dev)

    def add_batch(self, inp, out, **kwargs):

        # Get batch size
        inp = inp.squeeze()
        out = out.squeeze()
        if len(inp.shape) != 2:
            inp = inp.reshape(-1, self.columns)
            out = out.reshape(-1, self.rows)
        tmp = inp.shape[0]
        new_nsamples = self.nsamples + tmp

        inp = inp.float()
        out = out.float()
        self.XY += inp.T @ out

        # Update H (or XTX)
        # self.H *= self.nsamples / new_nsamples
        self.H += inp.T @ inp

        # Update channel output mean
        if self.norm:
            self.channel_norm *= self.nsamples / new_nsamples
            self.channel_norm += (
                out.float().norm(dim=0, p=2) ** 2
            ) / new_nsamples

        # Update number of samples
        self.nsamples = new_nsamples

    def get_prune_mask(self, new_intermediate_size, percdamp=0.0, **kwargs):

        # Calculate H inverse
        H = self.H
        # del self.H
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        Hinv = torch.linalg.cholesky(H, upper=True)

        # self.H = Hinv

        # Calculate score
        W = self.linear_output.weight.data.clone()
        score = W**2 / (torch.diag(Hinv).reshape((1, -1))) ** 2
        if self.norm:
            score_column = self.channel_norm @ score
        else:
            score_column = torch.sum(score, dim=0)
        _, top_idxs = torch.topk(score_column, new_intermediate_size)
        binary_mask = torch.ones_like(score_column)
        binary_mask[top_idxs] = 0
        mask = torch.where(binary_mask == 0)[0]

        # Return score
        return mask, binary_mask

    def get_linear_output_updated_weight(
        self, preserve_mask, binary_mask, blocksize=128, **kwargs
    ):
        H_pruned = self.H[preserve_mask][:, preserve_mask]
        H_pruned = torch.linalg.cholesky(H_pruned)
        H_pruned = torch.cholesky_inverse(H_pruned)
        updated_weights = H_pruned @ self.XY[preserve_mask]
        # W = self.linear_output.weight.data.clone()
        # W = W.to(self.dev)
        # Hinv = self.H
        # tmp = W / (torch.diag(Hinv).reshape((1, -1)))
        # mask_remove = torch.where(binary_mask == 1)[0]
        # increase_weight = tmp[:, mask_remove] @ Hinv[mask_remove]
        # W -= increase_weight
        # # assert torch.allclose(
        # #     W[:, mask_remove],
        # #     torch.zeros_like(W)[:, mask_remove],
        # #     atol=1e-6,
        # # ), (
        # #     f"Max diff: {W[:, mask_remove].abs().max()}"
        # # )
        # updated_weights = W[:, preserve_mask]
        torch.cuda.synchronize()

        return updated_weights.T

    def free(self):
        self.H = None
        self.nsamples = None
        if self.norm:
            self.channel_norm = None
            self.norm = None
        else:
            self.norm = None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
