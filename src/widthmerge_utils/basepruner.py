import torch
import logging


class BasePruner:
    def __init__(self, layer, model_config, layer_idx, **kwargs):
        self.layer = layer
        self.model_config = model_config
        self.linear_output = self.find_linear_output()
        self.dev = self.linear_output.weight.device
        W = self.linear_output.weight.data.clone()
        self.layer_idx = layer_idx
        self.rows = W.shape[0]
        self.columns = W.shape[1]

    def add_batch(self, inp, out, **kwargs):
        raise NotImplementedError

    def get_prune_mask(self, sparsity, **kwargs):
        raise NotImplementedError

    def prune(self, new_intermediate_size, **kwargs):
        preserve_mask, binary_mask = self.get_prune_mask(
            new_intermediate_size=new_intermediate_size, **kwargs
        )
        updated_weights = self.get_linear_output_updated_weight(
            preserve_mask=preserve_mask,
            binary_mask=binary_mask,
            new_intermediate_size=new_intermediate_size,
            **kwargs,
        )
        self.update_layer(
            updated_weights=updated_weights,
            preserve_mask=preserve_mask,
            new_intermediate_size=new_intermediate_size,
            **kwargs,
        )
        logging.debug(
            f"Pruned MLP layer {self.layer_idx} - Mask: {preserve_mask}"
        )

    def free(self):
        torch.cuda.empty_cache()

    def update_layer(
        self, updated_weights, preserve_mask, new_intermediate_size, **kwargs
    ):

        assert new_intermediate_size == len(preserve_mask), (
            f"Invalid size match: mask ({len(preserve_mask)}) - new size ({new_intermediate_size})"
        )

        # Update weights
        if self.model_config.model_type in (
            "llama",
            "mistral",
            "qwen2",
            "qwen3",
            "ministral",
            "gemma3_text",
        ):
            self.layer.mlp.gate_proj.weight.data = (
                self.layer.mlp.gate_proj.weight.data[preserve_mask]
            )
            self.layer.mlp.up_proj.weight.data = (
                self.layer.mlp.up_proj.weight.data[preserve_mask]
            )
            self.layer.mlp.down_proj.weight.data = (
                self.layer.mlp.down_proj.weight.data[:, preserve_mask]
            )

            self.layer.mlp.gate_proj.weight.out_features = new_intermediate_size
            self.layer.mlp.gate_proj.out_features = new_intermediate_size
            self.layer.mlp.up_proj.weight.out_features = new_intermediate_size
            self.layer.mlp.up_proj.out_features = new_intermediate_size
            self.layer.mlp.down_proj.weight.in_features = new_intermediate_size
            self.layer.mlp.down_proj.in_features = new_intermediate_size

            assert (
                updated_weights.shape == self.layer.mlp.down_proj.weight.shape
            )
            updated_weights = updated_weights.to(
                self.layer.mlp.down_proj.weight.dtype
            )
            self.layer.mlp.down_proj.weight = torch.nn.Parameter(
                updated_weights
            )

        elif self.model_config.model_type == "phi3":
            # phi3 adopts a joint gate and up projection
            gate_up_weights = self.layer.mlp.gate_up_proj.weight.data
            gate_weights, up_weights = gate_up_weights.chunk(2, dim=0)

            gate_weights = gate_weights[preserve_mask]
            up_weights = up_weights[preserve_mask]

            self.layer.mlp.gate_up_proj.weight.data = torch.cat(
                [gate_weights, up_weights], dim=0
            )
            self.layer.mlp.down_proj.weight.data = (
                self.layer.mlp.down_proj.weight.data[:, preserve_mask]
            )

            assert (
                updated_weights.shape == self.layer.mlp.down_proj.weight.shape
            )
            updated_weights = updated_weights.to(
                self.layer.mlp.down_proj.weight.dtype
            )
            self.layer.mlp.down_proj.weight = torch.nn.Parameter(
                updated_weights
            )

        elif self.model_config.model_type == "phi":
            self.layer.mlp.fc1.weight.data = self.layer.mlp.fc1.weight.data[
                preserve_mask
            ]
            self.layer.mlp.fc1.bias.data = self.layer.mlp.fc1.bias.data[
                preserve_mask
            ]
            self.layer.mlp.fc2.weight.data = self.layer.mlp.fc2.weight.data[
                :, preserve_mask
            ]

            assert updated_weights.shape == self.layer.mlp.fc2.weight.shape
            updated_weights = updated_weights.to(
                self.layer.mlp.fc2.weight.dtype
            )
            self.layer.mlp.fc2.weight = torch.nn.Parameter(updated_weights)

        else:
            logging.error(
                f"Error: {self.model_config.model_type} is not supported"
            )
            exit(0)

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
            return self.layer.mlp.down_proj

        elif self.model_config.config.model_type in ["phi"]:
            return self.layer.mlp.fc2

        logging.error(
            f"Error: {self.model_config.config.model_type} is not supported"
        )
        exit(0)

    def get_linear_output_updated_weight(
        self, preserve_mask, sparsity, **kwargs
    ):
        raise NotImplementedError
