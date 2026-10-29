import math

import numpy as np
import torch
from torch import nn

import cpp_kernels


class CppPointwiseConv2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = False
    ) -> None:

        super().__init__()

        if in_channels <= 0:
            raise ValueError(
                "in_channels debe ser mayor que cero"
            )

        if out_channels <= 0:
            raise ValueError(
                "out_channels debe ser mayor que cero"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Compatible con PyTorch Conv2d 1x1:
        # [out_channels, in_channels, 1, 1]
        self.weight = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels,
                1,
                1,
                dtype=torch.float32
            ),
            requires_grad=False
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(
                    out_channels,
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

        nn.init.kaiming_uniform_(
            self.weight,
            a=math.sqrt(5)
        )

        if self.bias is not None:

            bound = 1.0 / math.sqrt(
                self.in_channels
            )

            nn.init.uniform_(
                self.bias,
                -bound,
                bound
            )

    def forward(
        self,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:

        if input_tensor.ndim != 4:
            raise ValueError(
                "input debe tener forma "
                "[batch_size, channels, height, width]"
            )

        if input_tensor.shape[1] != self.in_channels:
            raise ValueError(
                "Los canales de entrada no coinciden "
                "con in_channels"
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

        output_numpy = (
            cpp_kernels.pointwise_conv2d_forward(
                input_numpy,
                weight_numpy,
                bias_numpy
            )
        )

        return torch.from_numpy(
            output_numpy
        ).to(original_device)