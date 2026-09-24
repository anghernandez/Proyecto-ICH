import numpy as np
import torch
from torch import nn

import cpp_kernels


class CppLayerAdd(nn.Module):

    def __init__(
        self,
        alpha: float = 1.0
    ) -> None:

        super().__init__()

        self.alpha = alpha

    def forward(
        self,
        input1: torch.Tensor,
        input2: torch.Tensor
    ) -> torch.Tensor:

        if input1.shape != input2.shape:
            raise ValueError(
                "input1 e input2 deben tener "
                "la misma forma"
            )

        original_device = input1.device

        input1_numpy = np.ascontiguousarray(
            input1.detach().cpu().numpy(),
            dtype=np.float32
        )

        input2_numpy = np.ascontiguousarray(
            input2.detach().cpu().numpy(),
            dtype=np.float32
        )

        output_numpy = cpp_kernels.layer_add_forward(
            input1_numpy,
            input2_numpy,
            self.alpha
        )

        return torch.from_numpy(
            output_numpy
        ).to(original_device)