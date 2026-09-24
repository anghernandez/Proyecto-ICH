import numpy as np
import torch
from torch import nn

import cpp_kernels


class CppGlobalAvgPool2d(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:

        if input_tensor.ndim != 4:
            raise ValueError(
                "input debe tener forma "
                "[batch_size, channels, height, width]"
            )

        original_device = input_tensor.device

        input_numpy = np.ascontiguousarray(
            input_tensor.detach().cpu().numpy(),
            dtype=np.float32
        )

        output_numpy = (
            cpp_kernels.global_avgpool2d_forward(
                input_numpy
            )
        )

        return torch.from_numpy(
            output_numpy
        ).to(original_device)