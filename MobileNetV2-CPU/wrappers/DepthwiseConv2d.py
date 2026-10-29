import math

import numpy as np
import torch
from torch import nn

import cpp_kernels


class CppDepthwiseConv2d(nn.Module):

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False
    ) -> None:

        super().__init__()

        if channels <= 0:
            raise ValueError(
                "channels debe ser mayor que cero"
            )

        if kernel_size <= 0:
            raise ValueError(
                "kernel_size debe ser mayor que cero"
            )

        if stride <= 0:
            raise ValueError(
                "stride debe ser mayor que cero"
            )

        if padding < 0:
            raise ValueError(
                "padding no puede ser negativo"
            )

        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Forma compatible con:
        #
        # nn.Conv2d(
        #     channels,
        #     channels,
        #     kernel_size,
        #     groups=channels
        # )
        #
        # [channels, 1, kernel_height, kernel_width]

        self.weight = nn.Parameter(
            torch.empty(
                channels,
                1,
                kernel_size,
                kernel_size,
                dtype=torch.float32
            ),
            requires_grad=False
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(
                    channels,
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

            fan_in = (
                self.kernel_size
                * self.kernel_size
            )

            bound = 1.0 / math.sqrt(
                fan_in
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

        if input_tensor.shape[1] != self.channels:
            raise ValueError(
                "Los canales de entrada no coinciden "
                "con channels"
            )

        input_height = input_tensor.shape[2]
        input_width = input_tensor.shape[3]

        output_height = (
            (
                input_height
                + 2 * self.padding
                - self.kernel_size
            )
            // self.stride
            + 1
        )

        output_width = (
            (
                input_width
                + 2 * self.padding
                - self.kernel_size
            )
            // self.stride
            + 1
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
            cpp_kernels.depthwise_conv2d_forward(
                input_numpy,
                weight_numpy,
                bias_numpy,
                output_height,
                output_width,
                self.stride,
                self.stride,
                self.padding,
                self.padding
            )
        )

        return torch.from_numpy(
            output_numpy
        ).to(original_device)