import torch
from torch import nn

from wrappers import (
    CppConv2d,
    CppPointwiseConv2d,
    CppDepthwiseConv2d,
    CppBatchNorm2d,
    CppReLU6,
    CppLayerAdd,
    CppGlobalAvgPool2d,
    CppLinear,
)


# ============================================================
# Conv2D + BatchNorm2D + ReLU6
# ============================================================

class CppConvBNReLU6(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
    ) -> None:

        super().__init__()

        # ----------------------------------------------------
        # Selección del tipo de convolución
        # ----------------------------------------------------

        # Pointwise 1x1
        if kernel_size == 1 and groups == 1:

            self.conv = CppPointwiseConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                bias=False,
            )

        # Depthwise
        elif (
            groups == in_channels
            and out_channels == in_channels
        ):

            self.conv = CppDepthwiseConv2d(
                channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )

        # Conv2D general
        else:

            self.conv = CppConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )

        self.bn = CppBatchNorm2d(
            num_features=out_channels
        )

        self.relu6 = CppReLU6()

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:

        output_tensor = self.conv(
            input_tensor
        )

        output_tensor = self.bn(
            output_tensor
        )

        output_tensor = self.relu6(
            output_tensor
        )

        return output_tensor

    # ========================================================
    # Carga de pesos desde PyTorch
    # ========================================================

    def load_pytorch_layer(
        self,
        pytorch_layer
    ) -> None:

        pytorch_conv = pytorch_layer[0]
        pytorch_bn = pytorch_layer[1]

        with torch.no_grad():

            # Conv2D
            self.conv.weight.copy_(
                pytorch_conv.weight
            )

            if (
                self.conv.bias is not None
                and pytorch_conv.bias is not None
            ):
                self.conv.bias.copy_(
                    pytorch_conv.bias
                )

            # BatchNorm2D
            self.bn.weight.copy_(
                pytorch_bn.weight
            )

            self.bn.bias.copy_(
                pytorch_bn.bias
            )

            self.bn.running_mean.copy_(
                pytorch_bn.running_mean
            )

            self.bn.running_var.copy_(
                pytorch_bn.running_var
            )

            self.bn.eps = pytorch_bn.eps


# ============================================================
# Inverted Residual
# ============================================================

class CppInvertedResidual(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: int,
    ) -> None:

        super().__init__()

        if stride not in (1, 2):
            raise ValueError(
                "stride debe ser 1 o 2"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.expand_ratio = expand_ratio

        hidden_channels = int(
            round(
                in_channels
                * expand_ratio
            )
        )

        self.hidden_channels = hidden_channels

        # ----------------------------------------------------
        # Expansion 1x1
        # ----------------------------------------------------

        if expand_ratio != 1:

            self.expand = CppConvBNReLU6(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
            )

        else:

            self.expand = None

        # ----------------------------------------------------
        # Depthwise 3x3
        # ----------------------------------------------------

        self.depthwise = CppConvBNReLU6(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=hidden_channels,
        )

        # ----------------------------------------------------
        # Pointwise projection 1x1
        #
        # IMPORTANTE:
        # No hay ReLU6 después de la proyección.
        # ----------------------------------------------------

        self.project_conv = CppPointwiseConv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            bias=False,
        )

        self.project_bn = CppBatchNorm2d(
            num_features=out_channels
        )

        # ----------------------------------------------------
        # Residual
        # ----------------------------------------------------

        self.use_residual = (
            stride == 1
            and in_channels == out_channels
        )

        self.add = CppLayerAdd(
            alpha=1.0
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:

        identity = input_tensor

        output_tensor = input_tensor

        # Expansion
        if self.expand is not None:

            output_tensor = self.expand(
                output_tensor
            )

        # Depthwise
        output_tensor = self.depthwise(
            output_tensor
        )

        # Projection
        output_tensor = self.project_conv(
            output_tensor
        )

        output_tensor = self.project_bn(
            output_tensor
        )

        # Residual
        if self.use_residual:

            output_tensor = self.add(
                identity,
                output_tensor
            )

        return output_tensor

    # ========================================================
    # Carga de pesos desde InvertedResidual de PyTorch
    # ========================================================

    def load_pytorch_layer(
        self,
        pytorch_block
    ) -> None:

        # ----------------------------------------------------
        # Caso expand_ratio != 1
        #
        # conv[0] -> expansión ConvBNReLU6
        # conv[1] -> depthwise ConvBNReLU6
        # conv[2] -> projection Conv2D
        # conv[3] -> projection BatchNorm
        # ----------------------------------------------------

        if self.expand is not None:

            self.expand.load_pytorch_layer(
                pytorch_block.conv[0]
            )

            self.depthwise.load_pytorch_layer(
                pytorch_block.conv[1]
            )

            pytorch_project_conv = (
                pytorch_block.conv[2]
            )

            pytorch_project_bn = (
                pytorch_block.conv[3]
            )

        # ----------------------------------------------------
        # Caso expand_ratio == 1
        #
        # No existe expansión.
        #
        # conv[0] -> depthwise ConvBNReLU6
        # conv[1] -> projection Conv2D
        # conv[2] -> projection BatchNorm
        # ----------------------------------------------------

        else:

            self.depthwise.load_pytorch_layer(
                pytorch_block.conv[0]
            )

            pytorch_project_conv = (
                pytorch_block.conv[1]
            )

            pytorch_project_bn = (
                pytorch_block.conv[2]
            )

        # ----------------------------------------------------
        # Projection
        # ----------------------------------------------------

        with torch.no_grad():

            self.project_conv.weight.copy_(
                pytorch_project_conv.weight
            )

            if (
                self.project_conv.bias is not None
                and pytorch_project_conv.bias is not None
            ):
                self.project_conv.bias.copy_(
                    pytorch_project_conv.bias
                )

            self.project_bn.weight.copy_(
                pytorch_project_bn.weight
            )

            self.project_bn.bias.copy_(
                pytorch_project_bn.bias
            )

            self.project_bn.running_mean.copy_(
                pytorch_project_bn.running_mean
            )

            self.project_bn.running_var.copy_(
                pytorch_project_bn.running_var
            )

            self.project_bn.eps = (
                pytorch_project_bn.eps
            )


# ============================================================
# MobileNetV2 completa
# ============================================================

class CppMobileNetV2(nn.Module):

    def __init__(
        self,
        num_classes: int = 1000
    ) -> None:

        super().__init__()

        self.num_classes = num_classes

        # ====================================================
        # Primera capa
        #
        # [N, 3, 224, 224]
        # ->
        # [N, 32, 112, 112]
        # ====================================================

        self.first_layer = CppConvBNReLU6(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
        )

        # ====================================================
        # MobileNetV2
        #
        # t = expansion ratio
        # c = output channels
        # n = número de bloques
        # s = stride del primer bloque
        # ====================================================

        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        blocks = []

        input_channels = 32

        for (
            expand_ratio,
            output_channels,
            number_of_blocks,
            first_stride,
        ) in inverted_residual_setting:

            for block_index in range(
                number_of_blocks
            ):

                if block_index == 0:
                    stride = first_stride
                else:
                    stride = 1

                block = CppInvertedResidual(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    stride=stride,
                    expand_ratio=expand_ratio,
                )

                blocks.append(
                    block
                )

                input_channels = (
                    output_channels
                )

        self.blocks = nn.ModuleList(
            blocks
        )

        # ====================================================
        # Última convolución
        #
        # [N, 320, 7, 7]
        # ->
        # [N, 1280, 7, 7]
        # ====================================================

        self.last_layer = CppConvBNReLU6(
            in_channels=320,
            out_channels=1280,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
        )

        # ====================================================
        # Global Average Pool
        #
        # [N, 1280, 7, 7]
        # ->
        # [N, 1280, 1, 1]
        # ====================================================

        self.global_avg_pool = (
            CppGlobalAvgPool2d()
        )

        # ====================================================
        # Classifier
        #
        # 1280 -> 1000
        # ====================================================

        self.classifier = CppLinear(
            in_features=1280,
            out_features=num_classes,
            bias=True,
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        input_tensor: torch.Tensor,
        verbose: bool = False,
    ) -> torch.Tensor:

        # ----------------------------------------------------
        # Primera capa
        # ----------------------------------------------------

        output_tensor = self.first_layer(
            input_tensor
        )

        if verbose:
            print(
                "First layer:",
                tuple(output_tensor.shape)
            )

        # ----------------------------------------------------
        # 17 Inverted Residual
        # ----------------------------------------------------

        for block_index, block in enumerate(
            self.blocks
        ):

            output_tensor = block(
                output_tensor
            )

            if verbose:

                print(
                    f"Block {block_index + 1}:",
                    tuple(output_tensor.shape)
                )

        # ----------------------------------------------------
        # Última Conv + BN + ReLU6
        # ----------------------------------------------------

        output_tensor = self.last_layer(
            output_tensor
        )

        if verbose:
            print(
                "Last layer:",
                tuple(output_tensor.shape)
            )

        # ----------------------------------------------------
        # Global Average Pool
        # ----------------------------------------------------

        output_tensor = self.global_avg_pool(
            output_tensor
        )

        if verbose:
            print(
                "GlobalAvgPool:",
                tuple(output_tensor.shape)
            )

        # ----------------------------------------------------
        # Flatten
        # ----------------------------------------------------

        output_tensor = torch.flatten(
            output_tensor,
            1
        )

        if verbose:
            print(
                "Flatten:",
                tuple(output_tensor.shape)
            )

        # ----------------------------------------------------
        # Linear
        # ----------------------------------------------------

        output_tensor = self.classifier(
            output_tensor
        )

        if verbose:
            print(
                "Classifier:",
                tuple(output_tensor.shape)
            )

        return output_tensor

    # ========================================================
    # Carga de todos los pesos desde MobileNetV2 PyTorch
    # ========================================================

    def load_pytorch_weights(
        self,
        pytorch_model
    ) -> None:

        print(
            "Cargando pesos de MobileNetV2..."
        )

        # ----------------------------------------------------
        # features[0]
        # Primera Conv + BN + ReLU6
        # ----------------------------------------------------

        self.first_layer.load_pytorch_layer(
            pytorch_model.features[0]
        )

        print(
            "Primera capa cargada."
        )

        # ----------------------------------------------------
        # features[1] ... features[17]
        # 17 Inverted Residual
        # ----------------------------------------------------

        if len(self.blocks) != 17:

            raise RuntimeError(
                "CppMobileNetV2 debe contener "
                "17 bloques InvertedResidual"
            )

        for block_index in range(17):

            pytorch_block = (
                pytorch_model.features[
                    block_index + 1
                ]
            )

            self.blocks[
                block_index
            ].load_pytorch_layer(
                pytorch_block
            )

            print(
                f"Bloque "
                f"{block_index + 1}/17 "
                f"cargado."
            )

        # ----------------------------------------------------
        # features[18]
        # Última Conv + BN + ReLU6
        # ----------------------------------------------------

        self.last_layer.load_pytorch_layer(
            pytorch_model.features[18]
        )

        print(
            "Última capa convolucional cargada."
        )

        # ----------------------------------------------------
        # classifier[1]
        #
        # classifier[0] = Dropout
        # classifier[1] = Linear
        # ----------------------------------------------------

        pytorch_linear = (
            pytorch_model.classifier[1]
        )

        with torch.no_grad():

            self.classifier.weight.copy_(
                pytorch_linear.weight
            )

            if (
                self.classifier.bias is not None
                and pytorch_linear.bias is not None
            ):

                self.classifier.bias.copy_(
                    pytorch_linear.bias
                )

        print(
            "Clasificador cargado."
        )

        print(
            "Pesos de MobileNetV2 cargados "
            "correctamente."
        )