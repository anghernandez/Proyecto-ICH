import numpy as np
import torch
from torch import nn

import cpp_kernels


class CppReLU6(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:

        original_device = input_tensor.device

        input_numpy = np.ascontiguousarray(
            input_tensor.detach().cpu().numpy(),
            dtype=np.float32
        )

        output_numpy = cpp_kernels.relu6_forward(
            input_numpy
        )

        return torch.from_numpy(
            output_numpy
        ).to(original_device)