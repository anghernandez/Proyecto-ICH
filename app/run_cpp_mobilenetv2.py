import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import (
    mobilenet_v2,
    MobileNet_V2_Weights,
)


# ============================================================
# Rutas del proyecto
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = PROJECT_ROOT / "python"
CPP_DIR = PROJECT_ROOT / "cpp"

sys.path.insert(0, str(PYTHON_DIR))
sys.path.insert(0, str(CPP_DIR))


from models import CppMobileNetV2


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("\n" + "=" * 60)
    print("MOBILENETV2 - PYTORCH vs C++")
    print("=" * 60)

    # ========================================================
    # MobileNetV2 oficial de TorchVision
    # ========================================================

    weights = MobileNet_V2_Weights.IMAGENET1K_V1

    pytorch_model = mobilenet_v2(
        weights=weights
    )

    pytorch_model.eval()

    # ========================================================
    # MobileNetV2 C++
    # ========================================================

    cpp_model = CppMobileNetV2(
        num_classes=1000
    )

    cpp_model.load_pytorch_weights(
        pytorch_model
    )

    cpp_model.eval()

    # ========================================================
    # Imagen de entrada
    # ========================================================

    if len(sys.argv) != 2:

        print(
            "\nUso:"
            "\npython3 python/apps/run_cpp_mobilenetv2.py "
            "<imagen>"
        )

        raise SystemExit(1)

    image_path = sys.argv[1]

    image = Image.open(
        image_path
    ).convert("RGB")

    # Preprocesamiento oficial correspondiente
    # a los pesos IMAGENET1K_V1.

    preprocess = weights.transforms()

    input_tensor = preprocess(
        image
    ).unsqueeze(0)

    print(
        "\nForma de entrada:",
        tuple(input_tensor.shape)
    )

    # ========================================================
    # Inferencia PyTorch
    # ========================================================

    print(
        "\nEjecutando MobileNetV2 PyTorch..."
    )

    with torch.inference_mode():

        pytorch_output = pytorch_model(
            input_tensor
        )

    # ========================================================
    # Inferencia C++
    # ========================================================

    print(
        "Ejecutando MobileNetV2 C++..."
    )

    with torch.inference_mode():

        cpp_output = cpp_model(
            input_tensor
        )

    # ========================================================
    # Comparación numérica
    # ========================================================

    difference = (
        pytorch_output
        - cpp_output
    )

    rmse = torch.sqrt(
        torch.mean(
            difference ** 2
        )
    ).item()

    max_error = torch.max(
        torch.abs(difference)
    ).item()

    # ========================================================
    # Predicciones
    # ========================================================

    pytorch_prediction = (
        pytorch_output.argmax(
            dim=1
        ).item()
    )

    cpp_prediction = (
        cpp_output.argmax(
            dim=1
        ).item()
    )

    categories = weights.meta[
        "categories"
    ]

    pytorch_class = categories[
        pytorch_prediction
    ]

    cpp_class = categories[
        cpp_prediction
    ]

    # ========================================================
    # Resultados
    # ========================================================

    print("\n" + "=" * 60)
    print("RESULTADOS")
    print("=" * 60)

    print(
        "Forma salida PyTorch:",
        tuple(pytorch_output.shape)
    )

    print(
        "Forma salida C++:    ",
        tuple(cpp_output.shape)
    )

    print(
        f"\nRMSE logits:          "
        f"{rmse:.10e}"
    )

    print(
        f"Error máximo logits: "
        f"{max_error:.10e}"
    )

    print(
        "\nPredicción PyTorch:"
    )

    print(
        f"  [{pytorch_prediction}] "
        f"{pytorch_class}"
    )

    print(
        "\nPredicción C++:"
    )

    print(
        f"  [{cpp_prediction}] "
        f"{cpp_class}"
    )

    # ========================================================
    # Validación
    # ========================================================

    rmse_limit = 1e-3

    if (
        rmse <= rmse_limit
        and pytorch_prediction
        == cpp_prediction
    ):

        print(
            "\nRESULTADO: PASS"
        )

    else:

        print(
            "\nRESULTADO: FAIL"
        )


if __name__ == "__main__":
    main()