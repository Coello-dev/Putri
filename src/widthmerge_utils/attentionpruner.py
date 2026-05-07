import torch
import logging


class AttnPruner:
    def __init__(self, layer, model_config, layer_idx, **kwargs):
        self.layer = layer
        self.model_config = model_config
        self.linear_output = self.find_linear_output()
        self.dev = self.linear_output.weight.device
        W = self.linear_output.weight.data.clone()
        self.layer_idx = layer_idx
        self.rows = W.shape[0]
        self.columns = W.shape[1]

        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.XY = torch.zeros((self.columns, self.rows), device=self.dev)

    def add_batch(self, inp, out, **kwargs):

        # Get batch size
        inp = inp.squeeze()
        out = out.squeeze()
        if len(inp.shape) != 2:
            inp = inp.reshape(-1, self.columns)
            out = out.reshape(-1, self.rows)

        inp = inp.float()
        out = out.float()
        self.XY += inp.T @ out
        self.H += inp.T @ inp

    def free(self):
        XY = self.XY.cpu()
        H = self.H.cpu()
        self.XY = None
        self.H = None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        return H, XY

    def find_linear_output(self):

        if self.model_config.model_type in (
            "llama",
            "mistral",
            "qwen2",
            "qwen3",
            "ministral",
            "gemma3_text",
            "phi3",
        ):
            return self.layer.self_attn.o_proj

        # elif self.model_config.config.model_type in ["phi"]:
        #     return self.layer.self_attn.o_proj

        logging.error(
            f"Error: {self.model_config.config.model_type} is not supported"
        )
        exit(0)
