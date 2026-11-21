"""Benchmark FPGA: una imagen de MobileNetV2, 8 warmups y 120 mediciones.

Replica las condiciones de las pruebas CPU/GPU del Avance C del Proyecto ICH.
Lee una imagen individual desde --image (por defecto test_images/test_1.jpeg).
La exactitud frente a etiquetas reales solo se calcula si se proporciona --label.
No modifica el runtime ni los wrappers; utiliza su perfilado existente.
Mide el tiempo host de carga, ejecución, descarga y conversión; los campos
'upload' y 'download' NO se deben interpretar como latencia pura H2D/D2H.
Los tiempos 'wrapper_ms' incluyen costes host/XRT, no son latencia IP pura.
Se mantiene una sola imagen durante todas las repeticiones.
"""

import argparse
import csv
import math
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

import fpga_dag
from fpga_graph import FpgaGraph
from models.FpgaMobileNetV2 import FpgaMobileNetV2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=Path("test_images/test_1.jpeg"),
                        help="Ruta a la imagen única que se evaluará")
    parser.add_argument("--label", type=int, default=None,
                        help="Etiqueta ImageNet-1K opcional para calcular accuracy")
    parser.add_argument(
        "--weights", type=Path, default=None,
        help="state_dict local; si se omite, usa pesos oficiales de torchvision",
    )
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--inflight-batches", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--rest", type=float, default=0.0)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--mode", choices=("sequential", "pipeline", "both"), default="sequential",
    )
    args = parser.parse_args()

    if min(args.limit, args.batch_size, args.inflight_batches, args.repeats) <= 0:
        parser.error("Imágenes, batch, inflight y repeticiones deben ser positivos")
    if args.limit != 1 or args.batch_size != 1 or args.inflight_batches != 1:
        parser.error("Este protocolo utiliza exactamente una imagen, batch=1 e inflight=1")
    if args.warmup < 0:
        parser.error("--warmup no puede ser negativo")
    if not math.isfinite(args.rest) or args.rest < 0:
        parser.error("--rest debe ser finito y no negativo")
    if not math.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("--tolerance debe ser finita y positiva")

    if not args.image.is_file():
        parser.error(f"No existe la imagen: {args.image}")
    if args.label is not None and not (0 <= args.label < 1000):
        parser.error("--label debe estar entre 0 y 999")
    if args.weights is not None and not args.weights.is_file():
        parser.error(f"No existe el archivo de pesos: {args.weights}")

    # Resultados: misma estructura que LeNet-5.
    bank_dir = Path(__file__).resolve().parents[1]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = bank_dir / "results" / "mobilenetv2_fpga_paper_b1" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    timings_csv = run_dir / "timings.csv"
    python_profile_csv = run_dir / "python_profile.csv"
    with timings_csv.open("x", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow([
            "mode", "phase", "repetition", "images",
            "batch_size", "inflight_batches", "elapsed_ms",
        ])
    with python_profile_csv.open("x", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow([
            "mode", "phase", "repetition", "images",
            "batch_size", "inflight_batches",
            "preparation_ms", "upload_ms", "execution_ms",
            "download_ms", "conversion_ms", "cleanup_ms",
            "total_profile_ms", "unaccounted_ms",
        ])

    # Imagen única y preprocesamiento (fuera del cronómetro).
    weights = MobileNet_V2_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    with Image.open(args.image) as image:
        input_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(torch.float32)

    # La etiqueta es opcional; sin etiqueta se valida contra PyTorch,
    # pero no es posible calcular accuracy respecto al conjunto ImageNet.
    labels = (torch.tensor([args.label], dtype=torch.long)
              if args.label is not None else None)
    total_images = 1
    input_batches = [input_tensor]
    valid_counts = [1]

    # Referencia PyTorch y pesos FPGA (fuera del cronómetro).
    if args.weights is None:
        reference = mobilenet_v2(weights=weights).eval()
    else:
        reference = mobilenet_v2(weights=None).eval()
        state = torch.load(args.weights, map_location="cpu", weights_only=True)
        reference.load_state_dict(state, strict=True)

    with torch.inference_mode():
        expected = torch.cat([
            reference(batch)[:valid]
            for batch, valid in zip(input_batches, valid_counts)
        ])

    if not torch.isfinite(expected).all().item():
        raise RuntimeError("La referencia PyTorch contiene NaN o infinito")
    reference_predictions = expected.argmax(dim=1)
    reference_accuracy = (
        100.0 * reference_predictions.eq(labels).sum().item() / total_images
        if labels is not None else None
    )

    fpga_model = FpgaMobileNetV2(num_classes=1000).eval()
    fpga_model.load_pytorch_weights(reference)
    del reference

    modes = ["sequential", "pipeline"] if args.mode == "both" else [args.mode]
    timings = {mode: [] for mode in modes}

    print("\n" + "-" * 60)
    print("MobileNetV2 FPGA: ensayo del paper (una imagen)")
    print(f"Imagen seleccionada: {args.image}")
    print(f"Calentamientos: {args.warmup} | Repeticiones medidas: {args.repeats}")
    print(f"Imágenes reales: {total_images}")
    print(f"Muestras por batch: {args.batch_size}")
    print(f"Batches totales: {len(input_batches)}")
    print(f"Límite de batches en vuelo: {args.inflight_batches}")
    print(f"Repeticiones medidas por modo: {args.repeats}")
    if reference_accuracy is not None:
        print(f"Accuracy PyTorch: {reference_accuracy:.2f}%")
    else:
        print("Accuracy PyTorch: N/D (no se proporcionó etiqueta real)")
    print(f"Clase predicha por PyTorch: {reference_predictions[0].item()}")
    print(f"Muestras de relleno: {len(input_batches) * args.batch_size - total_images}")
    print("Tiempo: carga + ejecución + descarga y conversión")
    print(f"Resultados: {run_dir}")
    print("-" * 60, flush=True)

    try:
        # Construcción y parámetros persistentes: una sola vez.
        with torch.inference_mode():
            graph = FpgaGraph().build(
                fpga_model, torch.zeros_like(input_batches[0]),
            )
        persistent_count = fpga_dag.buffer_count()

        def evaluate(mode):
            with torch.inference_mode():
                begin = time.perf_counter()
                outputs = graph.run_batches(
                    input_batches,
                    mode=mode,
                    max_inflight=args.inflight_batches,
                )
                elapsed = time.perf_counter() - begin

                if len(outputs) != len(input_batches):
                    raise RuntimeError("Cantidad incorrecta de salidas")

                collected = []
                for output, valid in zip(outputs, valid_counts):
                    if tuple(output.shape) != (args.batch_size, 1000):
                        raise RuntimeError(f"Forma incorrecta: {tuple(output.shape)}")
                    collected.append(output[:valid])

                if fpga_dag.buffer_count() != persistent_count:
                    raise RuntimeError("Quedaron activaciones activas tras la ejecución")

                actual = torch.cat(collected)
                if actual.shape != expected.shape:
                    raise RuntimeError("La forma final difiere de PyTorch")
                if not torch.isfinite(actual).all().item():
                    raise RuntimeError("La salida FPGA contiene NaN o infinito")

                difference = actual.double() - expected.double()
                per_image_error = difference.abs().amax(dim=1)
                maximum = per_image_error.max().item()
                rmse = difference.square().mean().sqrt().item()
                outside = (per_image_error > args.tolerance).sum().item()

                predictions = actual.argmax(dim=1)
                accuracy = (
                    100.0 * predictions.eq(labels).sum().item() / total_images
                    if labels is not None else None
                )
                different = predictions.ne(reference_predictions).sum().item()

            if outside:
                raise RuntimeError(
                    f"{mode}: {outside} imágenes fuera de tolerancia; "
                    f"error máximo={maximum:.6e}"
                )
            return elapsed, accuracy, rmse, maximum, different

        # Calentamiento y repeticiones; alternar el orden entre modos.
        for round_index in range(args.warmup + args.repeats):
            warmup = round_index < args.warmup
            ordered_modes = modes if round_index % 2 == 0 else modes[::-1]

            for mode in ordered_modes:
                if args.rest and (round_index > 0 or mode != ordered_modes[0]):
                    time.sleep(args.rest)

                elapsed, accuracy, rmse, maximum, different = evaluate(mode)
                phase = "warmup" if warmup else "measured"
                repetition = (
                    round_index + 1 if warmup else round_index - args.warmup + 1
                )

                # Exportar el perfil completo de C++/XRT, por ejecución.
                profile_name = (
                    f"{mode}_profile_warmup{repetition}.csv" if warmup
                    else f"{mode}_profile_rep{repetition}.csv"
                )
                profile_path = run_dir / profile_name
                if mode == "sequential":
                    fpga_dag.export_sequential_profiles(str(profile_path))
                else:
                    fpga_dag.export_pipeline_profiles(str(profile_path))

                # Perfil Python: etapas de preparación/carga/ejecución/descarga.
                profile = graph._profile.copy()
                profile_total = sum(profile.values())
                unaccounted = elapsed - profile_total
                with python_profile_csv.open("a", newline="", encoding="utf-8") as file:
                    csv.writer(file).writerow([
                        mode, phase, repetition, total_images,
                        args.batch_size, args.inflight_batches,
                        f"{profile.get('preparation', 0.0) * 1000:.6f}",
                        f"{profile.get('upload', 0.0) * 1000:.6f}",
                        f"{profile.get('execution', 0.0) * 1000:.6f}",
                        f"{profile.get('download', 0.0) * 1000:.6f}",
                        f"{profile.get('conversion', 0.0) * 1000:.6f}",
                        f"{profile.get('cleanup', 0.0) * 1000:.6f}",
                        f"{profile_total * 1000:.6f}",
                        f"{unaccounted * 1000:.6f}",
                    ])

                with timings_csv.open("a", newline="", encoding="utf-8") as file:
                    csv.writer(file).writerow([
                        mode, phase, repetition, total_images,
                        args.batch_size, args.inflight_batches,
                        f"{elapsed * 1000:.6f}",
                    ])

                if not warmup:
                    timings[mode].append(elapsed)
                label = (
                    f"Calentamiento {repetition}" if warmup
                    else f"Repetición {repetition}"
                )
                accuracy_text = (
                    f"{accuracy:.2f}%" if accuracy is not None else "N/D"
                )
                print(
                    f"{label} | {mode} | {elapsed * 1000:.3f} ms | "
                    f"Accuracy: {accuracy_text} | RMSE: {rmse:.3e} | "
                    f"Máximo: {maximum:.3e} | "
                    f"Predicciones diferentes: {different}",
                    flush=True,
                )

        # Resumen estadístico de las repeticiones medidas (sin warmups).
        def stats_ms(values):
            ordered = sorted(v * 1000.0 for v in values)
            return (
                statistics.mean(ordered), statistics.median(ordered),
                statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
                ordered[math.ceil(0.95 * len(ordered)) - 1],
                ordered[0], ordered[-1],
            )

        with (run_dir / "summary.csv").open("x", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["mode", "image", "warmup", "repeats", "mean_ms",
                             "median_ms", "std_ms", "p95_ms", "min_ms", "max_ms"])
            for mode, values in timings.items():
                writer.writerow([mode, str(args.image), args.warmup, args.repeats,
                                 *[f"{v:.6f}" for v in stats_ms(values)]])

        # Agrupa nodos del mismo tipo por inferencia (p. ej. todas las Pointwise).
        # Solo para modo secuencial: wrapper_ms no es latencia IP pura.
        if "sequential" in modes:
            by_operation = {}
            for repetition in range(1, args.repeats + 1):
                csv_path = run_dir / f"sequential_profile_rep{repetition}.csv"
                with csv_path.open(newline="", encoding="utf-8") as file:
                    totals = {}
                    for row in csv.DictReader(file):
                        name = row["operation"]
                        totals[name] = totals.get(name, 0.0) + float(row["wrapper_ms"])
                for name, value in totals.items():
                    by_operation.setdefault(name, []).append(value / 1000.0)
            with (run_dir / "operation_summary.csv").open(
                "x", newline="", encoding="utf-8"
            ) as file:
                writer = csv.writer(file)
                writer.writerow(["operation", "metric", "repeats", "mean_ms", "median_ms",
                                 "std_ms", "p95_ms", "min_ms", "max_ms"])
                for name, values in sorted(by_operation.items()):
                    writer.writerow([name, "sum_wrapper_ms_per_inference", len(values),
                                     *[f"{v:.6f}" for v in stats_ms(values)]])

        averages = {
            mode: sum(values) / len(values) for mode, values in timings.items()
        }
        print("\n" + "-" * 60)
        for mode, average in averages.items():
            mean, median, std, p95, *_ = stats_ms(timings[mode])
            print(f"{mode}: media={mean:.3f}, mediana={median:.3f}, "
                  f"desviación={std:.3f}, P95={p95:.3f} ms")
            print(
                f"{mode}: {average * 1000:.3f} ms | "
                f"{average * 1000 / total_images:.3f} ms/imagen | "
                f"{total_images / average:.2f} imágenes/s"
            )
        if args.mode == "both":
            sequential = averages["sequential"]
            pipeline = averages["pipeline"]
            print(f"Aceleración: {sequential / pipeline:.3f}x")
            print(f"Reducción del tiempo: {100.0 * (1.0 - pipeline / sequential):.2f}%")
        print(f"Resultados guardados en: {run_dir}")

    finally:
        fpga_dag.shutdown()

    print("PASS: validación contra PyTorch y limpieza completadas.")


if __name__ == "__main__":
    main()