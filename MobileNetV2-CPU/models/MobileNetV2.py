import torch

from torchvision.models import (
    mobilenet_v2,
    MobileNet_V2_Weights
)


# ============================================================
# MobileNetV2 de referencia
# ============================================================

def get_mobilenet_v2():
    """
    Carga MobileNetV2 oficial de TorchVision
    preentrenada con ImageNet-1K.
    """

    weights = MobileNet_V2_Weights.IMAGENET1K_V1

    model = mobilenet_v2(
        weights=weights
    )

    model.eval()

    return model, weights


# ============================================================
# Mostrar dimensiones internas
# ============================================================

def print_feature_shapes(model, x):
    """
    Ejecuta manualmente la sección 'features' de MobileNetV2
    para observar la forma del tensor después de cada bloque.
    """

    print("\n====================================================")
    print("DIMENSIONES INTERNAS DE MOBILENETV2")
    print("====================================================")

    print(f"Input:       {tuple(x.shape)}")

    with torch.no_grad():

        for index, layer in enumerate(model.features):

            x = layer(x)

            print(
                f"features[{index:2d}]: "
                f"{tuple(x.shape)} "
                f"-> {layer.__class__.__name__}"
            )

        # Global Average Pooling
        x = torch.nn.functional.adaptive_avg_pool2d(
            x,
            (1, 1)
        )

        print(
            f"AvgPool:     {tuple(x.shape)}"
        )

        # Flatten
        x = torch.flatten(
            x,
            1
        )

        print(
            f"Flatten:     {tuple(x.shape)}"
        )

        # Classifier
        x = model.classifier(x)

        print(
            f"Classifier:  {tuple(x.shape)}"
        )

    print("====================================================\n")

    return x


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Dispositivo
    # --------------------------------------------------------

    device = torch.device("cpu")

    print(f"\nUsing device: {device}")


    # --------------------------------------------------------
    # Modelo
    # --------------------------------------------------------

    model, weights = get_mobilenet_v2()

    model = model.to(device)


    # --------------------------------------------------------
    # Información de ImageNet
    # --------------------------------------------------------

    categories = weights.meta["categories"]

    print(
        f"ImageNet classes: {len(categories)}"
    )


    # --------------------------------------------------------
    # Entrada de prueba
    #
    # B = 1
    # C = 3 canales RGB
    # H = 224
    # W = 224
    # --------------------------------------------------------

    x = torch.randn(
        1,
        3,
        224,
        224,
        device=device
    )


    # --------------------------------------------------------
    # Forward normal
    # --------------------------------------------------------

    with torch.no_grad():

        y = model(x)


    print("\nForward completo:")

    print(
        f"Input shape:  {tuple(x.shape)}"
    )

    print(
        f"Output shape: {tuple(y.shape)}"
    )


    # --------------------------------------------------------
    # Ver dimensiones internas
    # --------------------------------------------------------

    y_manual = print_feature_shapes(
        model,
        x
    )


    # --------------------------------------------------------
    # Comparar forward normal contra recorrido manual
    # --------------------------------------------------------

    max_error = torch.max(
        torch.abs(y - y_manual)
    ).item()

    print(
        f"Max error entre forward normal "
        f"y recorrido manual: {max_error:.10e}"
    )

    if max_error < 1e-6:
        print("Resultado: PASS")
    else:
        print("Resultado: FAIL")