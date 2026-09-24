import numpy as np
import torch
from torch import nn

import cpp_kernels


class CppBatchNorm2d(nn.Module):

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True
    ) -> None:

        super().__init__()

        if num_features <= 0:
            raise ValueError(
                "num_features debe ser mayor que cero"
            )

        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if affine:

            self.weight = nn.Parameter(
                torch.ones(
                    num_features,
                    dtype=torch.float32
                ),
                requires_grad=False
            )

            self.bias = nn.Parameter(
                torch.zeros(
                    num_features,
                    dtype=torch.float32
                ),
                requires_grad=False
            )

        else:

            self.register_parameter(
                "weight",
                None
            )

            self.register_parameter(
                "bias",
                None
            )

        self.register_buffer(
            "running_mean",
            torch.zeros(
                num_features,
                dtype=torch.float32
            )
        )

        self.register_buffer(
            "running_var",
            torch.ones(
                num_features,
                dtype=torch.float32
            )
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

        if input_tensor.shape[1] != self.num_features:
            raise ValueError(
                "Los canales de entrada no coinciden "
                "con num_features"
            )

        original_device = input_tensor.device

        input_numpy = np.ascontiguousarray(
            input_tensor.detach().cpu().numpy(),
            dtype=np.float32
        )

        running_mean_numpy = np.ascontiguousarray(
            self.running_mean.detach().cpu().numpy(),
            dtype=np.float32
        )

        running_var_numpy = np.ascontiguousarray(
            self.running_var.detach().cpu().numpy(),
            dtype=np.float32
        )

        weight_numpy = None
        bias_numpy = None

        if self.affine:

            weight_numpy = np.ascontiguousarray(
                self.weight.detach().cpu().numpy(),
                dtype=np.float32
            )

            bias_numpy = np.ascontiguousarray(
                self.bias.detach().cpu().numpy(),
                dtype=np.float32
            )

        output_numpy = cpp_kernels.batchnorm2d_forward(
            input_numpy,
            weight_numpy,
            bias_numpy,
            running_mean_numpy,
            running_var_numpy,
            self.eps,
            self.affine
        )

        return torch.from_numpy(
            output_numpy
        ).to(original_device)