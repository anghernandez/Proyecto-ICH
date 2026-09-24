from python.layers.Conv2d import ManualConv2d
from python.layers.BatchNorm2d import ManualBatchNorm2d
from python.layers.ReLU6 import ManualReLU6
from python.layers.LayerAdd import ManualLayerAdd
from python.layers.GlobalAvgPool2d import ManualGlobalAvgPool2d
from python.layers.Linear import ManualLinear


# ============================================================
# Conv2D + BatchNorm2D + ReLU6
# ============================================================

class ManualConvBNReLU6:

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
    ):

        self.conv = ManualConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )

        self.bn = ManualBatchNorm2d(
            num_features=out_channels,
        )

        self.relu6 = ManualReLU6()

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu6(x)

        return x

    def __call__(self, x):
        return self.forward(x)

    def load_pytorch_layer(self, pytorch_layer):
        """
        pytorch_layer corresponde a un
        Conv2dNormActivation de torchvision:

            [0] Conv2d
            [1] BatchNorm2d
            [2] ReLU6
        """

        pytorch_conv = pytorch_layer[0]
        pytorch_bn = pytorch_layer[1]

        # Conv2D
        self.conv.weight = (
            pytorch_conv.weight
            .detach()
            .cpu()
            .tolist()
        )

        if pytorch_conv.bias is not None:

            self.conv.bias = (
                pytorch_conv.bias
                .detach()
                .cpu()
                .tolist()
            )

        else:
            self.conv.bias = None

        # BatchNorm2D
        weight = (
            pytorch_bn.weight
            .detach()
            .cpu()
            .tolist()
        )

        bias = (
            pytorch_bn.bias
            .detach()
            .cpu()
            .tolist()
        )

        running_mean = (
            pytorch_bn.running_mean
            .detach()
            .cpu()
            .tolist()
        )

        running_var = (
            pytorch_bn.running_var
            .detach()
            .cpu()
            .tolist()
        )

        self.bn.eps = pytorch_bn.eps

        self.bn.load_parameters(
            weight=weight,
            bias=bias,
            running_mean=running_mean,
            running_var=running_var,
        )


# ============================================================
# Inverted Residual
# ============================================================

class ManualInvertedResidual:

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        expand_ratio,
    ):

        if stride not in (1, 2):
            raise ValueError(
                "stride debe ser 1 o 2"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.expand_ratio = expand_ratio

        hidden_channels = int(
            round(in_channels * expand_ratio)
        )

        self.hidden_channels = hidden_channels

        # ----------------------------------------------------
        # Expansion 1x1
        # Solo existe cuando expand_ratio != 1
        # ----------------------------------------------------

        if expand_ratio != 1:

            self.expand = ManualConvBNReLU6(
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

        self.depthwise = ManualConvBNReLU6(
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
        # aquí NO hay ReLU6.
        # Linear bottleneck.
        # ----------------------------------------------------

        self.project_conv = ManualConv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False,
        )

        self.project_bn = ManualBatchNorm2d(
            num_features=out_channels,
        )

        # ----------------------------------------------------
        # Residual
        # ----------------------------------------------------

        self.use_residual = (
            stride == 1
            and in_channels == out_channels
        )

        self.add = ManualLayerAdd(
            alpha=1.0
        )

    def forward(self, x):

        identity = x

        # Expansion
        if self.expand is not None:
            x = self.expand(x)

        # Depthwise
        x = self.depthwise(x)

        # Projection
        x = self.project_conv(x)
        x = self.project_bn(x)

        # Residual
        if self.use_residual:

            x = self.add(
                identity,
                x,
            )

        return x

    def __call__(self, x):
        return self.forward(x)

    def _load_projection(
        self,
        pytorch_conv,
        pytorch_bn,
    ):

        self.project_conv.weight = (
            pytorch_conv.weight
            .detach()
            .cpu()
            .tolist()
        )

        if pytorch_conv.bias is not None:

            self.project_conv.bias = (
                pytorch_conv.bias
                .detach()
                .cpu()
                .tolist()
            )

        else:
            self.project_conv.bias = None

        self.project_bn.eps = pytorch_bn.eps

        self.project_bn.load_parameters(
            weight=(
                pytorch_bn.weight
                .detach()
                .cpu()
                .tolist()
            ),
            bias=(
                pytorch_bn.bias
                .detach()
                .cpu()
                .tolist()
            ),
            running_mean=(
                pytorch_bn.running_mean
                .detach()
                .cpu()
                .tolist()
            ),
            running_var=(
                pytorch_bn.running_var
                .detach()
                .cpu()
                .tolist()
            ),
        )

    def load_pytorch_layer(
        self,
        pytorch_block,
    ):

        # ====================================================
        # Caso expand_ratio != 1
        #
        # conv[0] -> expansión
        # conv[1] -> depthwise
        # conv[2] -> projection Conv2D
        # conv[3] -> projection BatchNorm
        # ====================================================

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

        # ====================================================
        # Caso expand_ratio == 1
        #
        # No existe expansión.
        #
        # conv[0] -> depthwise
        # conv[1] -> projection Conv2D
        # conv[2] -> projection BatchNorm
        # ====================================================

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

        self._load_projection(
            pytorch_project_conv,
            pytorch_project_bn,
        )


# ============================================================
# MobileNetV2 completa
# ============================================================

class ManualMobileNetV2:

    def __init__(
        self,
        num_classes=1000,
    ):

        self.num_classes = num_classes

        # ====================================================
        # Entrada
        #
        # RGB:
        # [N, 3, 224, 224]
        #
        # ->
        #
        # [N, 32, 112, 112]
        # ====================================================

        self.first_layer = ManualConvBNReLU6(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # ====================================================
        # Configuración oficial MobileNetV2
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

        self.blocks = []

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

                block = ManualInvertedResidual(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    stride=stride,
                    expand_ratio=expand_ratio,
                )

                self.blocks.append(
                    block
                )

                input_channels = (
                    output_channels
                )

        # ====================================================
        # Última convolución
        #
        # [N, 320, 7, 7]
        #
        # ->
        #
        # [N, 1280, 7, 7]
        # ====================================================

        self.last_layer = ManualConvBNReLU6(
            in_channels=320,
            out_channels=1280,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # ====================================================
        # Global Average Pool
        # ====================================================

        self.global_avg_pool = (
            ManualGlobalAvgPool2d()
        )

        # ====================================================
        # Clasificador
        #
        # 1280 -> 1000
        # ====================================================

        self.classifier = ManualLinear(
            in_features=1280,
            out_features=num_classes,
            bias=True,
        )

    # ========================================================
    # Flatten
    # ========================================================

    def _flatten(self, x):
        """
        Después del GlobalAvgPool:

            [N, 1280, 1, 1]

        se convierte en:

            [N, 1280]
        """

        flattened = []

        for batch_index in range(
            len(x)
        ):

            sample = []

            for channel_index in range(
                len(x[batch_index])
            ):

                value = (
                    x[batch_index]
                    [channel_index]
                    [0]
                    [0]
                )

                sample.append(
                    value
                )

            flattened.append(
                sample
            )

        return flattened

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        x,
        verbose=False,
    ):

        # Primera Conv + BN + ReLU6
        x = self.first_layer(x)

        if verbose:
            print(
                "First layer:",
                self._shape(x)
            )

        # 17 Inverted Residual
        for block_index in range(
            len(self.blocks)
        ):

            x = self.blocks[
                block_index
            ](x)

            if verbose:

                print(
                    f"Block {block_index + 1}:",
                    self._shape(x)
                )

        # Conv final
        x = self.last_layer(x)

        if verbose:
            print(
                "Last layer:",
                self._shape(x)
            )

        # Global Average Pooling
        x = self.global_avg_pool(x)

        if verbose:
            print(
                "GlobalAvgPool:",
                self._shape(x)
            )

        # Flatten
        x = self._flatten(x)

        if verbose:
            print(
                "Flatten:",
                (
                    len(x),
                    len(x[0]),
                )
            )

        # Linear
        x = self.classifier(x)

        if verbose:
            print(
                "Classifier:",
                (
                    len(x),
                    len(x[0]),
                )
            )

        return x

    def __call__(
        self,
        x,
        verbose=False,
    ):

        return self.forward(
            x,
            verbose=verbose,
        )

    # ========================================================
    # Utilidad para mostrar shape
    # ========================================================

    @staticmethod
    def _shape(x):

        return (
            len(x),
            len(x[0]),
            len(x[0][0]),
            len(x[0][0][0]),
        )

    # ========================================================
    # Cargar MobileNetV2 PyTorch
    # ========================================================

    def load_pytorch_weights(
        self,
        pytorch_model,
    ):

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

        # ----------------------------------------------------
        # features[1] ... features[17]
        #
        # 17 InvertedResidual
        # ----------------------------------------------------

        if len(self.blocks) != 17:

            raise RuntimeError(
                "La MobileNetV2 manual debe "
                "contener 17 bloques"
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
                f"Bloque {block_index + 1}/17 "
                "cargado"
            )

        # ----------------------------------------------------
        # features[18]
        # Última Conv + BN + ReLU6
        # ----------------------------------------------------

        self.last_layer.load_pytorch_layer(
            pytorch_model.features[18]
        )

        # ----------------------------------------------------
        # classifier[1]
        # Linear 1280 -> 1000
        #
        # classifier[0] es Dropout.
        # En eval() no modifica la entrada.
        # ----------------------------------------------------

        pytorch_linear = (
            pytorch_model.classifier[1]
        )

        self.classifier.load_parameters(
            weight=(
                pytorch_linear.weight
                .detach()
                .cpu()
                .tolist()
            ),
            bias=(
                pytorch_linear.bias
                .detach()
                .cpu()
                .tolist()
            ),
        )

        print(
            "Pesos cargados correctamente."
        )