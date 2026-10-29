import math

import numpy as np
import torch
from torch import nn

import cpp_kernels


def _to_pair(
    value: int | tuple[int, int],
    name: str
) -> tuple[int, int]:

    if isinstance(value, int):
        return value, value

    if isinstance(value, tuple) and len(value) == 2:
        return value

    raise ValueError(
        f"{name} debe ser un entero "
        "o una tupla de dos enteros"
    )


class CppConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True
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

        if groups != 1:
            raise NotImplementedError(
                "El kernel C++ actual solo soporta groups=1"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = _to_pair(
            kernel_size,
            "kernel_size"
        )

        self.stride = _to_pair(
            stride,
            "stride"
        )

        self.padding = _to_pair(
            padding,
            "padding"
        )

        self.dilation = _to_pair(
            dilation,
            "dilation"
        )

        self.groups = groups

        if self.dilation != (1, 1):
            raise NotImplementedError(
                "El kernel C++ actual solo soporta dilation=1"
            )

        if min(self.kernel_size) <= 0:
            raise ValueError(
                "kernel_size debe ser mayor que cero"
            )

        if min(self.stride) <= 0:
            raise ValueError(
                "stride debe ser mayor que cero"
            )

        if min(self.padding) < 0:
            raise ValueError(
                "padding no puede ser negativo"
            )

        kernel_height, kernel_width = (
            self.kernel_size
        )

        self.weight = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels,
                kernel_height,
                kernel_width,
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
        kernel_height, kernel_width = (
            self.kernel_size
        )

        fan_in = (
            self.in_channels
            * kernel_height
            * kernel_width
        )

        bound = 1.0 / math.sqrt(fan_in)

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

        if input_tensor.ndim != 4:
            raise ValueError(
                "input debe tener forma "
                "[batch, channels, height, width]"
            )

        if input_tensor.shape[1] != self.in_channels:
            raise ValueError(
                "Los canales de entrada no coinciden "
                "con in_channels"
            )

        input_height = input_tensor.shape[2]
        input_width = input_tensor.shape[3]

        kernel_height, kernel_width = (
            self.kernel_size
        )

        stride_height, stride_width = (
            self.stride
        )

        padding_height, padding_width = (
            self.padding
        )

        dilation_height, dilation_width = (
            self.dilation
        )

        output_height = (
            (
                input_height
                + 2 * padding_height
                - dilation_height
                * (kernel_height - 1)
                - 1
            )
            // stride_height
            + 1
        )

        output_width = (
            (
                input_width
                + 2 * padding_width
                - dilation_width
                * (kernel_width - 1)
                - 1
            )
            // stride_width
            + 1
        )

        if output_height <= 0 or output_width <= 0:
            raise ValueError(
                "La configuración genera una "
                "salida con dimensiones inválidas"
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

        output_numpy = cpp_kernels.conv2d_forward(
            input_numpy,
            weight_numpy,
            bias_numpy,
            output_height,
            output_width,
            stride_height,
            stride_width,
            padding_height,
            padding_width
        )

        return torch.from_numpy(
            output_numpy
        ).to(original_device)
