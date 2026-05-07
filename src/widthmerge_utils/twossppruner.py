import torch
import logging
from src.widthmerge_utils.basepruner import BasePruner
import os


class TwoSSPPruner(BasePruner):
    def __init__(self, merge_method, **kwargs):
        super().__init__(**kwargs)

        self.merge_method = merge_method
        self.channel_norm = torch.zeros((self.columns), device=self.dev)
        # self.channelout_norm = torch.zeros((self.rows), device=self.dev)

        if merge_method in ["min_mse", "min_mse_parallel"]:
            self.H = torch.zeros((self.columns, self.columns), device=self.dev)
            self.XY = torch.zeros((self.columns, self.rows), device=self.dev)

        self.nsamples = 0

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
        if hasattr(self, "XY"):
            self.XY += inp.T @ out
            self.H += inp.T @ inp

        # Update channel_norm
        self.channel_norm *= self.nsamples / new_nsamples
        self.channel_norm += inp.float().norm(dim=0, p=2) / new_nsamples
        # self.channelout_norm *= self.nsamples / new_nsamples
        # self.channelout_norm += out.float().norm(dim=0, p=2) ** 2 / new_nsamples

        # Update number of samples
        self.nsamples = new_nsamples

    def get_prune_mask(self, new_intermediate_size, **kwargs):

        # Calculate score
        score = self.channel_norm
        # W2 = (
        #     self.linear_output.weight.data.clone().T ** 2 / self.channelout_norm
        # )
        # W2 /= W2.max()
        # score = torch.sum(score * W2.T, dim=0)
        _, top_idxs = torch.topk(score, new_intermediate_size)
        # logging.debug(top_idxs)

        # Obtain binary mask
        binary_mask = torch.ones_like(self.channel_norm)
        binary_mask[top_idxs] = 0
        mask = torch.where(binary_mask == 0)[0]

        # Return score
        return mask, binary_mask

    def get_linear_output_updated_weight(
        self, preserve_mask, sparsity, damp_value=0.01, big_damp=0.1, **kwargs
    ):

        if self.merge_method in ["min_mse","min_mse_parallel"]:
            # Cholesky standard
            self.H = self.H.cpu()
            self.XY = self.XY.cpu()
            num_inputs = len(preserve_mask)
            try:
                # Apparently this gives worse than calculating the inverse
                # H_pruned = torch.linalg.cholesky(H_pruned)
                # updated_weights = torch.cholesky_solve(
                #     self.XY[preserve_mask], H_pruned
                # )
                H_pruned = self.H.clone().to(self.dev)[preserve_mask][
                    :, preserve_mask
                ]
                XY = self.XY.clone().to(self.dev)[preserve_mask]
                H_pruned = torch.linalg.cholesky(H_pruned)
                H_pruned = torch.cholesky_inverse(H_pruned)
                updated_weights = H_pruned @ XY
                calculation_error = False
            except Exception as e:
                logging.warning(f"Cholesky decomposition did not work. MSG: {e}")
                calculation_error = True

            # Standard
            if calculation_error:
                try:
                    torch.cuda.empty_cache()
                    H_pruned = self.H.clone().to(self.dev)[preserve_mask][
                        :, preserve_mask
                    ]
                    XY = self.XY.clone().to(self.dev)[preserve_mask]
                    damp = damp_value * torch.mean(torch.diag(H_pruned))
                    diag = torch.arange(num_inputs, device=self.dev)
                    H_pruned[diag, diag] += damp
                    H_pruned = torch.linalg.inv(H_pruned)
                    updated_weights = H_pruned @ XY
                    calculation_error = False
                except Exception as e:
                    logging.critical(f"Standard inverse did not work. MSG: {e}")

            # Cholesky standard + damp
            if calculation_error:
                try:
                    torch.cuda.empty_cache()
                    H_pruned = self.H.clone().to(self.dev)[preserve_mask][
                        :, preserve_mask
                    ]
                    XY = self.XY.clone().to(self.dev)[preserve_mask]
                    damp = damp_value * torch.mean(torch.diag(H_pruned))
                    diag = torch.arange(num_inputs, device=self.dev)
                    H_pruned[diag, diag] += damp
                    H_pruned = torch.linalg.cholesky(H_pruned)
                    H_pruned = torch.cholesky_inverse(H_pruned)
                    updated_weights = H_pruned @ XY
                    calculation_error = False
                except Exception as e:
                    logging.warning(
                        f"Cholesky decomposition (with damp) did not work. MSG: {e}"
                    )

            # Cholesky solve
            if calculation_error:
                try:
                    torch.cuda.empty_cache()
                    H_pruned = self.H.clone().to(self.dev)[preserve_mask][
                        :, preserve_mask
                    ]
                    XY = self.XY.clone().to(self.dev)[preserve_mask]
                    H_pruned = torch.linalg.cholesky(H_pruned)
                    updated_weights = torch.cholesky_solve(XY, H_pruned)
                    calculation_error = False
                except Exception as e:
                    logging.warning(f"Cholesky solve did not work. MSG: {e}")

            # Cholesky solve + damp
            if calculation_error:
                try:
                    torch.cuda.empty_cache()
                    H_pruned = self.H.clone().to(self.dev)[preserve_mask][
                        :, preserve_mask
                    ]
                    XY = self.XY.clone().to(self.dev)[preserve_mask]
                    damp = damp_value * torch.mean(torch.diag(H_pruned))
                    diag = torch.arange(num_inputs, device=self.dev)
                    H_pruned[diag, diag] += damp
                    H_pruned = torch.linalg.cholesky(H_pruned)
                    updated_weights = torch.cholesky_solve(XY, H_pruned)
                    calculation_error = False
                except Exception as e:
                    logging.warning(
                        f"Cholesky solve (with damp) did not work. MSG: {e}"
                    )

            # # Standard solve
            if calculation_error:
                try:
                    torch.cuda.empty_cache()
                    H_pruned = self.H.clone().to(self.dev)[preserve_mask][
                        :, preserve_mask
                    ]
                    XY = self.XY.clone().to(self.dev)[preserve_mask]
                    updated_weights = torch.linalg.solve(H_pruned, XY)
                    calculation_error = False
                except Exception as e:
                    logging.critical(f"Standard Solve did not work. MSG: {e}")

            # Standard solve + damp
            if calculation_error:
                try:
                    torch.cuda.empty_cache()
                    H_pruned = self.H.clone().to(self.dev)[preserve_mask][
                        :, preserve_mask
                    ]
                    XY = self.XY.clone().to(self.dev)[preserve_mask]
                    damp = damp_value * torch.mean(torch.diag(H_pruned))
                    diag = torch.arange(num_inputs, device=self.dev)
                    H_pruned[diag, diag] += damp
                    updated_weights = torch.linalg.solve(H_pruned, XY)
                    calculation_error = False
                except Exception as e:
                    logging.critical(
                        f"Standard Solve with damp did not work. MSG: {e}"
                    )

            # Cholesky standard + big damp
            if calculation_error:
                try:
                    torch.cuda.empty_cache()
                    H_pruned = self.H.clone().to(self.dev)[preserve_mask][
                        :, preserve_mask
                    ]
                    XY = self.XY.clone().to(self.dev)[preserve_mask]
                    damp = big_damp * torch.mean(torch.diag(H_pruned))
                    diag = torch.arange(num_inputs, device=self.dev)
                    H_pruned[diag, diag] += damp
                    H_pruned = torch.linalg.cholesky(H_pruned)
                    H_pruned = torch.cholesky_inverse(H_pruned)
                    updated_weights = H_pruned @ XY
                    calculation_error = False
                except Exception as e:
                    logging.warning(
                        f"Cholesky decomposition (with BIG damp) did not work. MSG: {e}"
                    )

            # Standard solve + damp
            if calculation_error:
                try:
                    torch.cuda.empty_cache()
                    H_pruned = self.H.clone().to(self.dev)[preserve_mask][
                        :, preserve_mask
                    ]
                    XY = self.XY.clone().to(self.dev)[preserve_mask]
                    damp = big_damp * torch.mean(torch.diag(H_pruned))
                    diag = torch.arange(num_inputs, device=self.dev)
                    H_pruned[diag, diag] += damp
                    updated_weights = torch.linalg.solve(H_pruned, XY)
                    calculation_error = False
                except Exception as e:
                    logging.critical(
                        f"Standard Solve with big damp did not work. MSG: {e}"
                    )
            
            while calculation_error:
                big_damp *= 10
                try:
                    torch.cuda.empty_cache()
                    H_pruned = self.H.clone().to(self.dev)[preserve_mask][
                        :, preserve_mask
                    ]
                    XY = self.XY.clone().to(self.dev)[preserve_mask]
                    damp = big_damp * torch.mean(torch.diag(H_pruned))
                    diag = torch.arange(num_inputs, device=self.dev)
                    H_pruned[diag, diag] += damp
                    updated_weights = torch.linalg.solve(H_pruned, XY)
                    calculation_error = False
                except Exception as e:
                    logging.critical(
                        f"Standard Solve with VERY big damp ({big_damp}) did not work. MSG: {e}"
                    )

                if big_damp > 10**10:
                    logging.critical(
                        f"Stopping the execution"
                    )
                    calculation_error = False

            return updated_weights.T

        else:
            updated_weights = self.linear_output.weight.data.clone()[:,preserve_mask]
            return updated_weights

    def free(self):
        self.channel_norm = None
        self.nsamples = None
        if hasattr(self, "XY"):
            self.XY = None
        if hasattr(self, "H"):
            self.H = None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
