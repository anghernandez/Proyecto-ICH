"""Perfilado de MobileNetV2 desde imágenes guardadas, en CPU."""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def rss_mib():
    """Memoria residente del proceso en este instante, en MiB."""
    try:
        pages = int(
            Path("/proc/self/statm").read_text().split()[1]
        )

        return pages * os.sysconf("SC_PAGE_SIZE") / (2 ** 20)

    except (OSError, ValueError, IndexError):
        return None


def measured(stage, function):
    """Ejecuta una función y devuelve su resultado y sus mediciones."""

    memory_before = rss_mib()

    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    start = time.perf_counter_ns()

    result = function()

    end = time.perf_counter_ns()
    usage_after = resource.getrusage(resource.RUSAGE_SELF)

    elapsed_ms = (end - start) / 1e6

    cpu_ms = (
        (usage_after.ru_utime - usage_before.ru_utime)
        + (usage_after.ru_stime - usage_before.ru_stime)
    ) * 1000

    record = {
        "etapa": stage,
        "tiempo_ms": elapsed_ms,
        "cpu_ms": cpu_ms,
        "cpu_pct_un_nucleo": (
            100 * cpu_ms / elapsed_ms if elapsed_ms else 0
        ),
        "rss_antes_mib": memory_before,
        "rss_despues_mib": rss_mib(),
    }

    return result, record

def sha256(path):
    """Calcula una huella para identificar el contenido de un archivo."""
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def command(*args):
    """Consulta información del equipo o repositorio."""
    try:
        return subprocess.check_output(
            args,
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    except (OSError, subprocess.CalledProcessError):
        return None


def write_json(path, data):
    """Guarda configuración o resultados estructurados."""
    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )


def stats(values):
    """Calcula estadísticas de una serie de mediciones."""
    values = sorted(values)

    position = (len(values) - 1) * 0.95
    lower = math.floor(position)
    upper = math.ceil(position)

    percentile_95 = (
        values[lower]
        + (values[upper] - values[lower]) * (position - lower)
    )

    return {
        "n": len(values),
        "media": statistics.mean(values),
        "mediana": statistics.median(values),
        "desviacion": (
            statistics.stdev(values) if len(values) > 1 else 0
        ),
        "p95": percentile_95,
        "minimo": values[0],
        "maximo": values[-1],
    }


def summarize(source, destination, keys, metrics):
    """Agrupa las muestras de un CSV y escribe sus estadísticas."""
    groups = defaultdict(list)

    with source.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            group_key = tuple(row[key] for key in keys)
            groups[group_key].append(row)

    columns = keys + [
        "metrica",
        "n",
        "media",
        "mediana",
        "desviacion",
        "p95",
        "minimo",
        "maximo",
    ]

    with destination.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for group_key, rows in groups.items():
            for metric in metrics:
                values = [
                    float(row[metric])
                    for row in rows
                    if row[metric] != ""
                ]

                if values:
                    record = dict(zip(keys, group_key))
                    record["metrica"] = metric
                    record.update(stats(values))
                    writer.writerow(record)

    return {
        group_key: len(rows)
        for group_key, rows in groups.items()
    }

def main():
    parser = argparse.ArgumentParser(
        description="Perfilado de MobileNetV2 desde archivos, en CPU."
    )

    parser.add_argument(
        "--images",
        nargs="+",
        default=["test_images/imagen.jpeg"],
    )
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument(
        "--preprocess",
        choices=["legacy", "official"],
        default="legacy",
    )
    parser.add_argument(
        "--kernel-profile",
        choices=["on", "off"],
        default="on",
    )
    parser.add_argument(
        "--weights-file",
        help="Archivo local de pesos de MobileNetV2.",
    )
    parser.add_argument(
        "--startup-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()

    if args.iterations < 1 or args.warmup < 0 or args.threads < 1:
        parser.error(
            "iterations y threads deben ser positivos; warmup >= 0."
        )

    images = [Path(name).resolve() for name in args.images]

    if not args.startup_only:
        if any(not path.is_file() for path in images):
            parser.error("No existe una de las imágenes indicadas.")

    if args.weights_file:
        args.weights_file = str(Path(args.weights_file).resolve())

        if not Path(args.weights_file).is_file():
            parser.error("No existe el archivo de pesos indicado.")

    output = Path(args.output_dir).resolve()

    if output.exists():
        parser.error(
            "La carpeta de salida ya existe. Usa otro nombre."
        )

    output.mkdir(parents=True, exist_ok=False)

    # --------------------------------------------------------
    # Configuración y trazabilidad
    # --------------------------------------------------------

    metadata = {
        "status": "iniciado",
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "argumentos": vars(args),
        "plataforma": platform.platform(),
        "arquitectura": platform.machine(),
        "hostname": platform.node(),
        "python": sys.version,
        "cpu_logicos": os.cpu_count(),
        "afinidad": (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
        "git_commit": command("git", "rev-parse", "HEAD"),
        "git_estado": command("git", "status", "--porcelain"),
        "compilador": command("g++", "--version"),
        "entorno_hilos": {
            name: os.environ.get(name)
            for name in [
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            ]
        },
        "alcance": "Archivos y CPU; sin cámara ni energía.",
        "memoria": (
            "RSS del proceso antes/después. "
            "No es pico ni memoria exclusiva de la etapa."
        ),
        "lectura": "Caché del sistema operativo sin vaciar.",
    }

    metadata["fuentes_sha256"] = {
        str(path.relative_to(ROOT)): sha256(path)
        for folder in [
            "app", "bindings", "kernels", "models", "wrappers"
        ]
        for path in sorted((ROOT / folder).glob("*"))
        if path.suffix in [".py", ".cpp", ".hpp"]
    }

    metadata["makefile_sha256"] = sha256(ROOT / "Makefile")

    metadata["imagenes"] = (
        [
            {
                "id": index,
                "nombre": path.name,
                "sha256": sha256(path),
            }
            for index, path in enumerate(images)
        ]
        if not args.startup_only
        else []
    )

    for name in ["cpuinfo", "meminfo"]:
        source = Path("/proc") / name

        if source.exists():
            (output / f"{name}.txt").write_text(
                source.read_text(),
                encoding="utf-8",
            )

    write_json(output / "metadata.json", metadata)

    try:
        # ----------------------------------------------------
        # Importaciones medidas
        # ----------------------------------------------------

        def import_dependencies():
            import torch
            import torchvision
            import numpy
            from PIL import Image
            from torchvision import transforms as T
            from torchvision.models import (
                mobilenet_v2,
                MobileNet_V2_Weights,
            )

            import cpp_kernels
            from models import CppMobileNetV2

            return (
                torch,
                torchvision,
                numpy,
                Image,
                T,
                mobilenet_v2,
                MobileNet_V2_Weights,
                cpp_kernels,
                CppMobileNetV2,
            )

        modules, import_record = measured(
            "importaciones",
            import_dependencies,
        )

        (
            torch,
            torchvision,
            numpy,
            Image,
            T,
            mobilenet_v2,
            MobileNet_V2_Weights,
            kernels,
            CppMobileNetV2,
        ) = modules

        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)

        if not hasattr(kernels, "profiling_compiled"):
            raise RuntimeError(
                "Módulo C++ antiguo. Ejecuta make profile-build."
            )

        enabled = args.kernel_profile == "on"

        if enabled and not kernels.profiling_compiled:
            raise RuntimeError(
                "Perfilado no compilado. Ejecuta make profile-build."
            )

        kernels.set_profile_enabled(False)

        # ----------------------------------------------------
        # Creación de modelos y carga de pesos
        # ----------------------------------------------------

        weights = MobileNet_V2_Weights.IMAGENET1K_V1

        def load_models():
            if args.weights_file:
                reference = mobilenet_v2(weights=None)

                state_dict = torch.load(
                    args.weights_file,
                    map_location="cpu",
                    weights_only=True,
                )

                reference.load_state_dict(state_dict)
            else:
                reference = mobilenet_v2(weights=weights)

            reference.eval()

            model = CppMobileNetV2(num_classes=1000)
            model.load_pytorch_weights(reference)
            model.eval()

            return reference, model

        (reference, model), load_record = measured(
            "carga_modelos_y_pesos",
            load_models,
        )

        write_json(
            output / "inicializacion.json",
            [import_record, load_record],
        )

        metadata.update({
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": numpy.__version__,
            "torch_hilos": torch.get_num_threads(),
            "torch_hilos_interop": torch.get_num_interop_threads(),
            "perfilado_compilado": bool(kernels.profiling_compiled),
            "modulo_sha256": sha256(kernels.__file__),
            "pesos": (
                "archivo_local"
                if args.weights_file
                else "IMAGENET1K_V1"
            ),
            "pesos_archivo_sha256": (
                sha256(args.weights_file)
                if args.weights_file
                else None
            ),
        })

        build_info = Path(kernels.__file__ + ".build.json")

        metadata["compilacion"] = (
            json.loads(build_info.read_text())
            if build_info.exists()
            else None
        )

        # Identifica los parámetros realmente cargados.
        digest = hashlib.sha256()

        for name, tensor in reference.state_dict().items():
            digest.update(name.encode())
            digest.update(
                tensor.detach().cpu().contiguous().numpy().tobytes()
            )

        metadata["parametros_sha256"] = digest.hexdigest()

        if args.startup_only:
            metadata["status"] = "completo_inicializacion"
            write_json(output / "metadata.json", metadata)
            return

        # ----------------------------------------------------
        # Entrada y posprocesamiento
        # ----------------------------------------------------

        if args.preprocess == "official":
            preprocess = weights.transforms()
        else:
            # Conserva el procedimiento de la rama Perfilado.
            preprocess = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        metadata["transformaciones"] = repr(preprocess)

        def read_image(path):
            with Image.open(path) as source:
                # La conversión fuerza la decodificación aquí.
                return source.convert("RGB")

        def postprocess(logits):
            probabilities, ids = torch.topk(
                torch.softmax(logits, dim=1)[0],
                k=5,
            )

            return [
                (
                    int(index),
                    weights.meta["categories"][int(index)],
                    float(probability),
                )
                for probability, index in zip(
                    probabilities.tolist(),
                    ids.tolist(),
                )
            ]

        # ----------------------------------------------------
        # Validación fuera de las mediciones principales
        # ----------------------------------------------------

        print("\nValidando C++ contra PyTorch...")
        validation = []

        with torch.inference_mode():
            for path in images:
                tensor = preprocess(read_image(path)).unsqueeze(0)

                expected = reference(tensor)
                actual = model(tensor)

                difference = actual - expected
                finite = bool(torch.isfinite(difference).all())

                rmse = float(
                    torch.sqrt(torch.mean(difference ** 2))
                )
                maximum = float(difference.abs().max())

                passed = (
                    finite
                    and rmse <= 1e-3
                    and int(actual.argmax()) == int(expected.argmax())
                )

                validation.append({
                    "imagen": path.name,
                    "rmse": rmse if finite else None,
                    "error_maximo": maximum if finite else None,
                    "pass_test": passed,
                })

                print(
                    f"{path.name}: "
                    f"{'PASS' if passed else 'FAIL'} "
                    f"| RMSE={rmse:.6e}"
                )

        write_json(output / "validacion.json", validation)

        if not all(row["pass_test"] for row in validation):
            raise RuntimeError(
                "La validación falló. Revisa validacion.json."
            )

        # No mantener el modelo de referencia durante el perfilado.
        del reference

        import gc
        gc.collect()

        # ----------------------------------------------------
        # Calentamiento: sin guardar muestras
        # ----------------------------------------------------

        kernels.set_profile_enabled(enabled)

        print(f"\nCalentamiento: {args.warmup} repeticiones.")

        with torch.inference_mode():
            for index in range(args.warmup):
                path = images[index % len(images)]
                tensor = preprocess(read_image(path)).unsqueeze(0)
                postprocess(model(tensor))
                kernels.clear_profile_records()

        # ----------------------------------------------------
        # Archivos de muestras
        # ----------------------------------------------------

        stage_columns = [
            "iteracion",
            "imagen_id",
            "imagen",
            "etapa",
            "tiempo_ms",
            "cpu_ms",
            "cpu_pct_un_nucleo",
            "rss_antes_mib",
            "rss_despues_mib",
        ]

        kernel_columns = [
            "iteracion",
            "imagen_id",
            "imagen",
            "indice_llamada",
            "etapa",
            "forma_tensor",
            "tiempo_ms",
        ]

        prediction_columns = [
            "iteracion",
            "imagen_id",
            "clase",
            "categoria",
            "probabilidad",
        ]

        signature = None

        with (
            (output / "etapas.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stage_file,
            (output / "kernels.csv").open(
                "w", newline="", encoding="utf-8"
            ) as kernel_file,
            (output / "predicciones.csv").open(
                "w", newline="", encoding="utf-8"
            ) as prediction_file,
        ):
            stage_writer = csv.DictWriter(
                stage_file, fieldnames=stage_columns
            )
            kernel_writer = csv.DictWriter(
                kernel_file, fieldnames=kernel_columns
            )
            prediction_writer = csv.DictWriter(
                prediction_file, fieldnames=prediction_columns
            )

            stage_writer.writeheader()
            kernel_writer.writeheader()
            prediction_writer.writeheader()

            print(f"\nMidiendo {args.iterations} repeticiones...")

            with torch.inference_mode():
                for iteration in range(1, args.iterations + 1):
                    image_id = (iteration - 1) % len(images)
                    path = images[image_id]

                    common = {
                        "iteracion": iteration,
                        "imagen_id": image_id,
                        "imagen": path.name,
                    }

                    kernels.clear_profile_records()
                    rows = []

                    def pipeline():
                        image, row = measured(
                            "lectura_decodificacion",
                            lambda: read_image(path),
                        )
                        rows.append(row)

                        tensor, row = measured(
                            "preprocesamiento",
                            lambda: preprocess(image).unsqueeze(0),
                        )
                        rows.append(row)

                        logits, row = measured(
                            "inferencia_cpp",
                            lambda: model(tensor),
                        )
                        rows.append(row)

                        predictions, row = measured(
                            "posprocesamiento",
                            lambda: postprocess(logits),
                        )
                        rows.append(row)

                        return predictions

                    predictions, total_record = measured(
                        "total_flujo",
                        pipeline,
                    )
                    rows.append(total_record)

                    # Extracción y escritura fuera del total medido.
                    records = kernels.get_and_clear_profile_records()

                    current_signature = [
                        (record["etapa"], record["forma_tensor"])
                        for record in records
                    ]

                    if enabled:
                        if not records:
                            raise RuntimeError(
                                "No se generaron registros de kernels."
                            )

                        if (
                            signature is not None
                            and signature != current_signature
                        ):
                            raise RuntimeError(
                                "Cambió la secuencia de kernels."
                            )

                    signature = current_signature

                    for row in rows:
                        stage_writer.writerow(common | row)

                    for call_index, record in enumerate(records):
                        kernel_writer.writerow(
                            common
                            | {"indice_llamada": call_index}
                            | record
                        )

                    prediction_writer.writerow({
                        "iteracion": iteration,
                        "imagen_id": image_id,
                        "clase": predictions[0][0],
                        "categoria": predictions[0][1],
                        "probabilidad": predictions[0][2],
                    })

                    stage_file.flush()
                    kernel_file.flush()
                    prediction_file.flush()

                    if (
                        iteration % 20 == 0
                        or iteration == args.iterations
                    ):
                        print(
                            f"{iteration}/{args.iterations}",
                            flush=True,
                        )

        # ----------------------------------------------------
        # Estadísticas y comprobación del número de muestras
        # ----------------------------------------------------

        counts = summarize(
            output / "etapas.csv",
            output / "resumen_etapas.csv",
            ["etapa"],
            [
                "tiempo_ms",
                "cpu_ms",
                "cpu_pct_un_nucleo",
                "rss_antes_mib",
                "rss_despues_mib",
            ],
        )

        kernel_counts = summarize(
            output / "kernels.csv",
            output / "resumen_kernels.csv",
            ["indice_llamada", "etapa", "forma_tensor"],
            ["tiempo_ms"],
        )

        if (
            len(counts) != 5
            or any(count != args.iterations for count in counts.values())
        ):
            raise RuntimeError(
                "Cantidad inesperada de muestras por etapa."
            )

        if enabled and any(
            count != args.iterations
            for count in kernel_counts.values()
        ):
            raise RuntimeError(
                "Cantidad inesperada de muestras por kernel."
            )

        metadata.update({
            "status": "completo",
            "muestras_por_etapa": args.iterations,
            "llamadas_por_inferencia": len(signature or []),
            "supera_100_muestras": args.iterations > 100,
            "nota_total": (
                "Incluye instrumentación interna. "
                "No sumar total con etapas ni inferencia con kernels."
            ),
        })

        write_json(output / "metadata.json", metadata)

        print(f"\nResultados guardados en: {output}")

        if args.iterations <= 100:
            print(
                "PRUEBA CORTA: no cumple el mínimo "
                "de más de 100 muestras."
            )

    except Exception as error:
        metadata["status"] = "error"
        metadata["error"] = str(error)

        write_json(output / "metadata.json", metadata)
        raise


if __name__ == "__main__":
    main()