import argparse
import csv
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.models import (
    mobilenet_v2,
    MobileNet_V2_Weights,
)

import cpp_kernels
from models import CppMobileNetV2


# ============================================================
# Preprocesamiento
#
# Esto corre en Python/PyTorch y NO se mide -- el perfilado
# se enfoca solo en las capas implementadas en C++.
# ============================================================

def build_preprocess(resolution: int):

    return T.Compose([
        T.Resize((resolution, resolution)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_tensor(image_path: Path, preprocess) -> torch.Tensor:

    image = Image.open(image_path).convert("RGB")

    return preprocess(image).unsqueeze(0)


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Corre inferencia N veces sobre el mismo tensor "
            "(o un set chico de tensores reales) y guarda el "
            "perfilado por capa C++ a un CSV."
        )
    )

    parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        default=["test_images/imagen.jpeg"],
        help=(
            "Una o mas imagenes reales a usar como tensor(es) "
            "de entrada (default: test_images/imagen.jpeg)"
        ),
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=224,
        help="Resolucion de entrada (default: 224)",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=8,
        help="Corridas de warm-up descartadas (default: 8)",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=120,
        help="Cantidad de corridas medidas (default: 120)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="resultados/perfilado.csv",
        help="Ruta del CSV de salida",
    )

    parser.add_argument(
        "--overhead-only",
        action="store_true",
        help=(
            "No genera CSV ni llama al profiler por iteracion. "
            "Solo mide el tiempo total de las N corridas, para "
            "comparar contra un build sin instrumentacion "
            "(make build vs make build PROFILE=1)."
        ),
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # Cargar modelos
    # ========================================================

    print("\nCargando MobileNetV2 (PyTorch + C++)...")

    weights = MobileNet_V2_Weights.IMAGENET1K_V1

    pytorch_model = mobilenet_v2(weights=weights)
    pytorch_model.eval()

    cpp_model = CppMobileNetV2(num_classes=1000)
    cpp_model.load_pytorch_weights(pytorch_model)
    cpp_model.eval()

    preprocess = build_preprocess(args.resolution)

    # Tensores de entrada ya preprocesados (esto no se mide).
    input_tensors = [
        (Path(image_path).name, load_tensor(Path(image_path), preprocess))
        for image_path in args.images
    ]

    # ========================================================
    # Warm-up (descartado, no se mide ni se guarda)
    # ========================================================

    print(f"\nWarm-up: {args.warmup} corridas descartadas...")

    _, warmup_tensor = input_tensors[0]

    for _ in range(args.warmup):

        cpp_kernels.clear_profile_records()

        with torch.inference_mode():
            cpp_model(warmup_tensor)

    cpp_kernels.clear_profile_records()

    # ========================================================
    # Modo overhead: solo tiempo total, sin CSV ni llamadas
    # al profiler (para comparar contra un build sin
    # instrumentacion, ver PERFILADO.md paso 4).
    # ========================================================

    if args.overhead_only:

        print(
            f"\nMidiendo tiempo total de {args.iterations} "
            "corridas (modo overhead-only, sin profiler)..."
        )

        start_time = time.perf_counter()

        for iteration in range(1, args.iterations + 1):

            _, input_tensor = (
                input_tensors[(iteration - 1) % len(input_tensors)]
            )

            with torch.inference_mode():
                cpp_model(input_tensor)

        total_ms = (time.perf_counter() - start_time) * 1000.0

        print(
            f"\nTiempo total: {total_ms:.2f} ms "
            f"({args.iterations} corridas)"
        )

        print(
            f"Promedio por corrida: "
            f"{total_ms / args.iterations:.4f} ms"
        )

        return

    # ========================================================
    # Muestreo: >100 corridas sobre el mismo tensor
    # (o ciclando por el set chico de tensores reales)
    # ========================================================

    rows = []

    print(f"\nMidiendo {args.iterations} corridas...")

    for iteration in range(1, args.iterations + 1):

        image_name, input_tensor = (
            input_tensors[(iteration - 1) % len(input_tensors)]
        )

        cpp_kernels.clear_profile_records()

        with torch.inference_mode():
            cpp_model(input_tensor)

        records = cpp_kernels.get_and_clear_profile_records()

        for call_index, record in enumerate(records):

            rows.append({
                "etapa": record["etapa"],
                "indice_llamada": call_index,
                "forma_tensor": record["forma_tensor"],
                "iteracion": iteration,
                "imagen": image_name,
                "tiempo_ms": record["tiempo_ms"],
            })

        if iteration % 20 == 0:
            print(f"  {iteration}/{args.iterations}")

    # ========================================================
    # Guardar CSV
    # ========================================================

    fieldnames = [
        "etapa",
        "indice_llamada",
        "forma_tensor",
        "iteracion",
        "imagen",
        "tiempo_ms",
    ]

    with open(output_path, "w", newline="") as csv_file:

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\nListo. {len(rows)} filas guardadas en {output_path}"
    )


if __name__ == "__main__":
    main()
