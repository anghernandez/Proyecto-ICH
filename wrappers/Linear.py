import math

import numpy as np
import torch
from torch import nn

import cpp_kernels


class CppLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True
    ) -> None:
        super().__init__()

        if in_features <= 0:
            raise ValueError(
                "in_features debe ser mayor que cero"
            )

        if out_features <= 0:
            raise ValueError(
                "out_features debe ser mayor que cero"
            )

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                dtype=torch.float32
            ),
            requires_grad=False
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(
                    out_features,
                    dtype=torch.float32
                ),
                requires_grad=False
            )
        else:
            self.register_parameter(
                "bias",
                None
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(
            self.in_features
        )

        with torch.no_grad():
            self.weight.uniform_(
                -bound,
                bound
            )

            if self.bias is not None:
                self.bias.uniform_(
                    -bound,
                    bound
                )

    def forward(
        self,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:

        if input_tensor.ndim != 2:
            raise ValueError(
                "input debe tener forma "
                "[batch_size, in_features]"
            )

        if input_tensor.shape[1] != self.in_features:
            raise ValueError(
                "La última dimensión de input "
                "no coincide con in_features"
            )

        original_device = input_tensor.device

        input_numpy = np.ascontiguousarray(
            input_tensor.detach().cpu().numpy(),
            dtype=np.float32
        )

        weight_numpy = np.ascontiguousarray(
            self.weight.detach().cpu().numpy(),
            dtype=np.float32
        )

        bias_numpy = None

        if self.bias is not None:
            bias_numpy = np.ascontiguousarray(
                self.bias.detach().cpu().numpy(),
                dtype=np.float32
            )

        output_numpy = cpp_kernels.linear_forward(
            input_numpy,
            weight_numpy,
            bias_numpy
        )

        return torch.from_numpy(
            output_numpy
        ).to(original_device)
