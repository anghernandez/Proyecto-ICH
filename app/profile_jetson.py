"""Perfilado de MobileNetV2 en Jetson (CPU y GPU). Python 3.6+.

Basado en app/profile_complete.py (perfilado de la Kria). Conserva
las mismas etapas, columnas y estadísticas para poder comparar
directo, y agrega:

  - Compatibilidad con Python 3.6 / torch 1.10 / torchvision 0.11
    (JetPack 4.6). También corre en versiones nuevas.
  - Marca de tiempo de cada etapa (t_inicio_ns, t_fin_ns), con el
    mismo reloj CLOCK_MONOTONIC que usa el monitor.
  - Monitor externo de GPU, potencia y RAM (app/jetson_monitor.py):
    potencia en reposo, energía por etapa e iteración.
  - Si el módulo C++ trae el perfilado GPU (GpuProfilerBindings):
      * costo de crear el contexto CUDA,
      * desglose h2d / kernel / d2h por llamada (cudaEvent),
      * memoria GPU propia (pico) y del driver en la inferencia,
      * región para nvprof (--nvprof-region on).

Con el módulo solo-CPU funciona igual (línea base jetson_cpu); las
partes GPU simplemente no se generan.
"""

import argparse
import contextlib
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
sys.path.insert(0, str(ROOT / "app"))

import jetson_monitor  # noqa: E402

now_ns = jetson_monitor.monotonic_ns

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ------------------------------------------------------------
# Utilidades (iguales a profile_complete.py salvo lo indicado)
# ------------------------------------------------------------

def rss_mib():
    """Memoria residente del proceso en este instante, en MiB."""
    try:
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (2 ** 20)
    except (OSError, ValueError, IndexError):
        return None


def mem_available_mib():
    """MemAvailable del sistema (CPU y GPU comparten la DRAM)."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024.0
    except (OSError, ValueError):
        pass
    return None


def gpu_memory(kernels):
    if kernels is None or not hasattr(kernels, "gpu_memory"):
        return None
    return kernels.gpu_memory()


def measured(stage, function, kernels=None):
    """Ejecuta una función y devuelve su resultado y sus mediciones.

    Igual que en profile_complete.py, más t_inicio_ns / t_fin_ns.
    Si se pasa `kernels` (solo en la inferencia), registra también
    la memoria GPU antes/después y el pico propio; esas lecturas
    quedan fuera del intervalo cronometrado.
    """
    gpu_before = gpu_memory(kernels)
    if gpu_before is not None:
        kernels.reset_gpu_memory_peak()

    memory_before = rss_mib()
    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    start = now_ns()

    result = function()

    end = now_ns()
    usage_after = resource.getrusage(resource.RUSAGE_SELF)
    memory_after = rss_mib()

    gpu_after = gpu_memory(kernels)

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
        "rss_despues_mib": memory_after,
        "t_inicio_ns": start,
        "t_fin_ns": end,
        "gpu_propia_pico_mib": "",
        "gpu_driver_usada_antes_mib": "",
        "gpu_driver_usada_despues_mib": "",
    }

    if gpu_before is not None and gpu_after is not None:
        record["gpu_propia_pico_mib"] = gpu_after["propia_pico_mib"]
        record["gpu_driver_usada_antes_mib"] = (
            gpu_before["driver_usada_mib"]
        )
        record["gpu_driver_usada_despues_mib"] = (
            gpu_after["driver_usada_mib"]
        )

    return result, record


def sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(*args):
    """Consulta información del equipo o repositorio."""
    try:
        return subprocess.check_output(
            args,
            cwd=str(ROOT),
            universal_newlines=True,   # 'text=' no existe en 3.6
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def write_json(path, data):
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False,
                   allow_nan=False) + "\n",
        encoding="utf-8",
    )


def stats(values):
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
        "desviacion": statistics.stdev(values) if len(values) > 1 else 0,
        "p95": percentile_95,
        "minimo": values[0],
        "maximo": values[-1],
    }


def summarize(source, destination, keys, metrics):
    """Agrupa las muestras de un CSV y escribe sus estadísticas."""
    groups = defaultdict(list)
    order = []

    with source.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            group_key = tuple(row[key] for key in keys)
            if group_key not in groups:
                order.append(group_key)
            groups[group_key].append(row)

    columns = keys + ["metrica", "n", "media", "mediana",
                      "desviacion", "p95", "minimo", "maximo"]

    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for group_key in order:
            rows = groups[group_key]
            for metric in metrics:
                values = [
                    float(row[metric])
                    for row in rows
                    if row.get(metric, "") not in ("", None)
                ]
                if values:
                    record = dict(zip(keys, group_key))
                    record["metrica"] = metric
                    record.update(stats(values))
                    writer.writerow(record)

    return {group_key: len(groups[group_key]) for group_key in order}


def jetson_info():
    """Estado de relojes y modo de potencia (para trazabilidad)."""
    info = {
        "nv_tegra_release": read_text("/etc/nv_tegra_release"),
        "nvpmodel": command("nvpmodel", "-q"),
        "nvcc": command("/usr/local/cuda/bin/nvcc", "--version"),
        "gpu_freq_hz": {},
        "cpu_freq_khz": {},
    }

    gpu_dir = jetson_monitor._first([
        "/sys/devices/gpu.0/devfreq/*",
        "/sys/class/devfreq/*.gpu",
    ])
    if gpu_dir:
        for name in ["cur_freq", "min_freq", "max_freq", "governor"]:
            info["gpu_freq_hz"][name] = read_text(gpu_dir + "/" + name)

    for cpu in range(os.cpu_count() or 0):
        base = "/sys/devices/system/cpu/cpu{}/cpufreq/".format(cpu)
        info["cpu_freq_khz"]["cpu{}".format(cpu)] = {
            name: read_text(base + name)
            for name in ["scaling_cur_freq", "scaling_min_freq",
                         "scaling_max_freq", "scaling_governor"]
        }

    return info


# ------------------------------------------------------------
# Programa principal
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Perfilado de MobileNetV2 en Jetson (CPU/GPU)."
    )
    parser.add_argument("--images", nargs="+",
                        default=["test_images/imagen.jpeg"])
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preprocess", choices=["legacy", "official"],
                        default="official")
    parser.add_argument("--kernel-profile", choices=["on", "off"],
                        default="on")
    parser.add_argument("--gpu-profile", choices=["on", "off"],
                        default="on",
                        help="Desglose cudaEvent (si el módulo lo trae).")
    parser.add_argument("--monitor", choices=["on", "off"], default="on",
                        help="Monitor externo de GPU/potencia/RAM.")
    parser.add_argument("--monitor-interval-ms", type=float, default=20)
    parser.add_argument("--idle-seconds", type=float, default=10,
                        help="Ventana en reposo para potencia base.")
    parser.add_argument("--nvprof-region", choices=["on", "off"],
                        default="off",
                        help="cudaProfilerStart/Stop alrededor de las "
                             "medidas (usar con nvprof "
                             "--profile-from-start off).")
    parser.add_argument("--weights-file",
                        help="Archivo local de pesos de MobileNetV2.")
    parser.add_argument("--categories-file",
                        help="Nombres de clases ImageNet, uno por línea.")
    args = parser.parse_args()

    if args.iterations < 1 or args.warmup < 0 or args.threads < 1:
        parser.error("iterations y threads deben ser positivos; "
                     "warmup >= 0.")

    images = [Path(name).resolve() for name in args.images]
    if any(not path.is_file() for path in images):
        parser.error("No existe una de las imágenes indicadas.")

    if args.weights_file:
        args.weights_file = str(Path(args.weights_file).resolve())
        if not Path(args.weights_file).is_file():
            parser.error("No existe el archivo de pesos indicado.")

    categories_path = (
        Path(args.categories_file) if args.categories_file
        else ROOT / "app" / "imagenet_classes.txt"
    )
    categories = None
    if categories_path.is_file():
        categories = [
            line.strip()
            for line in categories_path.read_text().splitlines()
            if line.strip()
        ]
        if len(categories) != 1000:
            parser.error("El archivo de clases debe tener 1000 líneas.")

    output = Path(args.output_dir).resolve()
    if output.exists():
        parser.error("La carpeta de salida ya existe. Usa otro nombre.")
    output.mkdir(parents=True)

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
        "afinidad": (sorted(os.sched_getaffinity(0))
                     if hasattr(os, "sched_getaffinity") else None),
        "git_commit": command("git", "rev-parse", "HEAD"),
        "git_estado": command("git", "status", "--porcelain"),
        "compilador": command("g++", "--version"),
        "entorno_hilos": {
            name: os.environ.get(name)
            for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS",
                         "OPENBLAS_NUM_THREADS"]
        },
        "jetson": jetson_info(),
        "reloj": "CLOCK_MONOTONIC (mismo que el monitor)",
        "memoria": (
            "RSS del proceso antes/después. En la inferencia, además, "
            "pico de memoria GPU reservada por los kernels y "
            "cudaMemGetInfo (en Jetson es memoria del sistema)."
        ),
        "lectura": "Caché del sistema operativo sin vaciar.",
    }

    metadata["fuentes_sha256"] = {
        str(path.relative_to(ROOT)): sha256(path)
        for folder in ["app", "bindings", "kernels", "models", "wrappers"]
        for path in sorted((ROOT / folder).glob("*"))
        if path.suffix in [".py", ".cpp", ".hpp", ".cu", ".cuh"]
    }
    if (ROOT / "Makefile").exists():
        metadata["makefile_sha256"] = sha256(ROOT / "Makefile")
    metadata["imagenes"] = [
        {"id": index, "nombre": path.name, "sha256": sha256(path)}
        for index, path in enumerate(images)
    ]

    for name in ["cpuinfo", "meminfo"]:
        source = Path("/proc") / name
        if source.exists():
            (output / (name + ".txt")).write_text(
                source.read_text(), encoding="utf-8")

    monitor_on = args.monitor == "on"
    if monitor_on:
        sources = jetson_monitor.verificar_fuentes()
        metadata["monitor_fuentes"] = sources
        if not sources["rieles"] or not all(sources["rieles"].values()):
            print(
                "\nAVISO: no se pueden leer los rieles de potencia. "
                "Ejecuta antes:\n  sudo chmod a+r "
                "/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/"
                "rail_name_* /sys/bus/i2c/drivers/ina3221x/6-0040/"
                "iio:device0/in_power*_input\n"
            )

    write_json(output / "metadata.json", metadata)

    monitor = None

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
            from torchvision.models import mobilenet_v2

            import cpp_kernels
            from models import CppMobileNetV2

            return (torch, torchvision, numpy, Image, T, mobilenet_v2,
                    cpp_kernels, CppMobileNetV2)

        modules, import_record = measured("importaciones",
                                          import_dependencies)
        (torch, torchvision, numpy, Image, T, mobilenet_v2,
         kernels, CppMobileNetV2) = modules

        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)

        if not hasattr(kernels, "profiling_compiled"):
            raise RuntimeError("Módulo C++ antiguo. Ejecuta make "
                               "profile-build.")

        enabled = args.kernel_profile == "on"
        if enabled and not kernels.profiling_compiled:
            raise RuntimeError("Perfilado no compilado. Ejecuta make "
                               "profile-build.")

        kernels.set_profile_enabled(False)

        has_gpu = hasattr(kernels, "get_and_clear_gpu_records")
        gpu_enabled = has_gpu and enabled and args.gpu_profile == "on"
        if has_gpu:
            kernels.set_gpu_profile_enabled(False)

        init_records = [import_record]

        # ----------------------------------------------------
        # Contexto CUDA (solo si el módulo usa GPU)
        # ----------------------------------------------------

        if hasattr(kernels, "cuda_init"):
            available_before = mem_available_mib()
            _, cuda_record = measured("inicializacion_cuda",
                                      kernels.cuda_init)
            available_after = mem_available_mib()
            cuda_record["mem_disponible_antes_mib"] = available_before
            cuda_record["mem_disponible_despues_mib"] = available_after
            init_records.append(cuda_record)

        # ----------------------------------------------------
        # Modelos y pesos (IMAGENET1K_V1 = mobilenet_v2-b0353104)
        # ----------------------------------------------------

        def load_models():
            if args.weights_file:
                reference = mobilenet_v2()
                state_dict = torch.load(args.weights_file,
                                        map_location="cpu")
                reference.load_state_dict(state_dict)
            else:
                try:
                    from torchvision.models import MobileNet_V2_Weights
                    reference = mobilenet_v2(
                        weights=MobileNet_V2_Weights.IMAGENET1K_V1)
                except ImportError:
                    # torchvision 0.11: mismos pesos IMAGENET1K_V1.
                    reference = mobilenet_v2(pretrained=True)

            reference.eval()

            model = CppMobileNetV2(num_classes=1000)
            model.load_pytorch_weights(reference)
            model.eval()
            return reference, model

        (reference, model), load_record = measured(
            "carga_modelos_y_pesos", load_models)
        init_records.append(load_record)
        write_json(output / "inicializacion.json", init_records)

        metadata.update({
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": numpy.__version__,
            "torch_hilos": torch.get_num_threads(),
            "torch_hilos_interop": torch.get_num_interop_threads(),
            "perfilado_compilado": bool(kernels.profiling_compiled),
            "perfilado_gpu_disponible": has_gpu,
            "usa_cuda": bool(getattr(kernels, "usa_cuda", False)),
            "modulo_sha256": sha256(kernels.__file__),
            "pesos": "archivo_local" if args.weights_file
                     else "IMAGENET1K_V1",
            "pesos_archivo_sha256": (sha256(args.weights_file)
                                     if args.weights_file else None),
        })

        # Debe coincidir con el de la Kria si los pesos son iguales.
        digest = hashlib.sha256()
        for name, tensor in reference.state_dict().items():
            digest.update(name.encode())
            digest.update(
                tensor.detach().cpu().contiguous().numpy().tobytes())
        metadata["parametros_sha256"] = digest.hexdigest()

        # ----------------------------------------------------
        # Entrada y posprocesamiento
        # ----------------------------------------------------

        if args.preprocess == "official":
            # Equivale a MobileNet_V2_Weights.IMAGENET1K_V1.transforms()
            # (resize 256 bilineal, recorte 224, normalización),
            # que no existe en torchvision 0.11.
            preprocess = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ])
        else:
            preprocess = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ])
        metadata["transformaciones"] = repr(preprocess)

        def read_image(path):
            with Image.open(str(path)) as source:
                return source.convert("RGB")

        def postprocess(logits):
            probabilities, ids = torch.topk(
                torch.softmax(logits, dim=1)[0], k=5)
            return [
                (int(index),
                 categories[int(index)] if categories
                 else "clase_{}".format(int(index)),
                 float(probability))
                for probability, index in zip(probabilities.tolist(),
                                              ids.tolist())
            ]

        # ----------------------------------------------------
        # Validación fuera de las mediciones
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
                rmse = float(torch.sqrt(torch.mean(difference ** 2)))
                maximum = float(difference.abs().max())
                passed = (finite and rmse <= 1e-3
                          and int(actual.argmax()) == int(expected.argmax()))

                validation.append({
                    "imagen": path.name,
                    "rmse": rmse if finite else None,
                    "error_maximo": maximum if finite else None,
                    "pass_test": passed,
                })
                print("{}: {} | RMSE={:.6e}".format(
                    path.name, "PASS" if passed else "FAIL", rmse))

        write_json(output / "validacion.json", validation)
        if not all(row["pass_test"] for row in validation):
            raise RuntimeError("La validación falló. Revisa "
                               "validacion.json.")

        del reference
        import gc
        gc.collect()

        # ----------------------------------------------------
        # Monitor y potencia en reposo
        # ----------------------------------------------------

        idle_window = None
        if monitor_on:
            print("\nIniciando monitor ({} ms) y midiendo reposo "
                  "{} s...".format(args.monitor_interval_ms,
                                   args.idle_seconds))
            monitor = jetson_monitor.iniciar_monitor(
                output / "monitoreo.csv", args.monitor_interval_ms)
            idle_start = now_ns()
            time.sleep(args.idle_seconds)
            idle_window = (idle_start, now_ns())

        # ----------------------------------------------------
        # Calentamiento: sin guardar muestras
        # ----------------------------------------------------

        kernels.set_profile_enabled(enabled)
        if has_gpu:
            kernels.set_gpu_profile_enabled(gpu_enabled)

        print("\nCalentamiento: {} repeticiones.".format(args.warmup))
        with torch.inference_mode():
            for index in range(args.warmup):
                path = images[index % len(images)]
                tensor = preprocess(read_image(path)).unsqueeze(0)
                postprocess(model(tensor))
                kernels.clear_profile_records()
                if has_gpu:
                    kernels.clear_gpu_records()

        # ----------------------------------------------------
        # Medición
        # ----------------------------------------------------

        stage_columns = [
            "iteracion", "imagen_id", "imagen", "etapa", "tiempo_ms",
            "cpu_ms", "cpu_pct_un_nucleo", "rss_antes_mib",
            "rss_despues_mib", "t_inicio_ns", "t_fin_ns",
            "gpu_propia_pico_mib", "gpu_driver_usada_antes_mib",
            "gpu_driver_usada_despues_mib",
        ]
        kernel_columns = ["iteracion", "imagen_id", "imagen",
                          "indice_llamada", "etapa", "forma_tensor",
                          "tiempo_ms"]
        gpu_columns = ["iteracion", "imagen_id", "imagen",
                       "indice_registro", "etapa", "forma_tensor",
                       "fase", "tiempo_ms"]
        prediction_columns = ["iteracion", "imagen_id", "clase",
                              "categoria", "probabilidad"]

        signature = None
        gpu_signature = None
        windows = []   # (iteracion, etapa, t_inicio, t_fin)

        with contextlib.ExitStack() as stack:
            def open_csv(name, columns):
                handle = stack.enter_context(
                    (output / name).open("w", newline="",
                                         encoding="utf-8"))
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                return handle, writer

            stage_file, stage_writer = open_csv("etapas.csv",
                                                stage_columns)
            kernel_file, kernel_writer = open_csv("kernels.csv",
                                                  kernel_columns)
            pred_file, prediction_writer = open_csv("predicciones.csv",
                                                    prediction_columns)
            if gpu_enabled:
                gpu_file, gpu_writer = open_csv("kernels_gpu.csv",
                                                gpu_columns)

            if args.nvprof_region == "on" and hasattr(
                    kernels, "cuda_profiler_start"):
                kernels.cuda_profiler_start()

            print("\nMidiendo {} repeticiones...".format(args.iterations))
            block_start = now_ns()

            with torch.inference_mode():
                for iteration in range(1, args.iterations + 1):
                    image_id = (iteration - 1) % len(images)
                    path = images[image_id]
                    common = {"iteracion": iteration,
                              "imagen_id": image_id,
                              "imagen": path.name}

                    kernels.clear_profile_records()
                    if has_gpu:
                        kernels.clear_gpu_records()
                    rows = []

                    def pipeline():
                        image, row = measured(
                            "lectura_decodificacion",
                            lambda: read_image(path))
                        rows.append(row)

                        tensor, row = measured(
                            "preprocesamiento",
                            lambda: preprocess(image).unsqueeze(0))
                        rows.append(row)

                        logits, row = measured(
                            "inferencia_cpp",
                            lambda: model(tensor),
                            kernels=kernels if has_gpu else None)
                        rows.append(row)

                        predictions, row = measured(
                            "posprocesamiento",
                            lambda: postprocess(logits))
                        rows.append(row)
                        return predictions

                    predictions, total_record = measured("total_flujo",
                                                         pipeline)
                    rows.append(total_record)

                    # Extracción y escritura fuera del total medido.
                    records = kernels.get_and_clear_profile_records()
                    current = [(r["etapa"], r["forma_tensor"])
                               for r in records]
                    if enabled:
                        if not records:
                            raise RuntimeError("No se generaron "
                                               "registros de kernels.")
                        if signature is not None and signature != current:
                            raise RuntimeError("Cambió la secuencia de "
                                               "kernels.")
                    signature = current

                    for row in rows:
                        stage_writer.writerow(dict(common, **row))
                        windows.append((iteration, row["etapa"],
                                        row["t_inicio_ns"],
                                        row["t_fin_ns"]))

                    for call_index, record in enumerate(records):
                        kernel_writer.writerow(dict(
                            common, indice_llamada=call_index, **record))

                    if gpu_enabled:
                        gpu_records = kernels.get_and_clear_gpu_records()
                        current_gpu = [(r["etapa"], r["forma_tensor"],
                                        r["fase"]) for r in gpu_records]
                        if (gpu_signature is not None
                                and gpu_signature != current_gpu):
                            raise RuntimeError("Cambió la secuencia de "
                                               "registros GPU.")
                        gpu_signature = current_gpu
                        for index, record in enumerate(gpu_records):
                            gpu_writer.writerow(dict(
                                common, indice_registro=index, **record))
                        gpu_file.flush()

                    prediction_writer.writerow({
                        "iteracion": iteration,
                        "imagen_id": image_id,
                        "clase": predictions[0][0],
                        "categoria": predictions[0][1],
                        "probabilidad": predictions[0][2],
                    })

                    stage_file.flush()
                    kernel_file.flush()
                    pred_file.flush()

                    if iteration % 20 == 0 or iteration == args.iterations:
                        print("{}/{}".format(iteration, args.iterations),
                              flush=True)

            block_end = now_ns()

            if args.nvprof_region == "on" and hasattr(
                    kernels, "cuda_profiler_stop"):
                kernels.cuda_profiler_stop()

        if monitor is not None:
            time.sleep(0.5)   # que el monitor cubra el final del bloque
            jetson_monitor.detener_monitor(monitor)
            monitor = None

        # ----------------------------------------------------
        # Estadísticas de tiempo (mismo formato que la Kria)
        # ----------------------------------------------------

        counts = summarize(
            output / "etapas.csv", output / "resumen_etapas.csv",
            ["etapa"],
            ["tiempo_ms", "cpu_ms", "cpu_pct_un_nucleo",
             "rss_antes_mib", "rss_despues_mib", "gpu_propia_pico_mib",
             "gpu_driver_usada_antes_mib",
             "gpu_driver_usada_despues_mib"],
        )
        kernel_counts = summarize(
            output / "kernels.csv", output / "resumen_kernels.csv",
            ["indice_llamada", "etapa", "forma_tensor"], ["tiempo_ms"],
        )
        if gpu_enabled:
            summarize(
                output / "kernels_gpu.csv",
                output / "resumen_kernels_gpu.csv",
                ["indice_registro", "etapa", "forma_tensor", "fase"],
                ["tiempo_ms"],
            )

        if (len(counts) != 5
                or any(c != args.iterations for c in counts.values())):
            raise RuntimeError("Cantidad inesperada de muestras por "
                               "etapa.")
        if enabled and any(c != args.iterations
                           for c in kernel_counts.values()):
            raise RuntimeError("Cantidad inesperada de muestras por "
                               "kernel.")

        # ----------------------------------------------------
        # Energía y uso de GPU por etapa
        # ----------------------------------------------------

        if monitor_on and (output / "monitoreo.csv").exists():
            monitor_rows = jetson_monitor.cargar_monitoreo(
                output / "monitoreo.csv")
            times = jetson_monitor.tiempos(monitor_rows)

            idle = jetson_monitor.integrar(monitor_rows, idle_window[0],
                                           idle_window[1], times)
            idle_power = {
                key[len("potencia_media_mw_"):]: value
                for key, value in idle.items()
                if key.startswith("potencia_media_mw_")
            }

            energy_rows = []
            for iteration, stage, start, end in windows:
                result = jetson_monitor.integrar(monitor_rows, start, end,
                                                 times)
                if not result:
                    continue
                row = {"iteracion": iteration, "etapa": stage}
                row.update(result)
                for rail, power in idle_power.items():
                    key = "energia_mj_" + rail
                    if key in result:
                        row["energia_dinamica_mj_" + rail] = (
                            result[key] - power * result["duracion_ms"]
                            / 1000.0)
                energy_rows.append(row)

            if energy_rows:
                energy_columns = []
                for row in energy_rows:
                    for key in row:
                        if key not in energy_columns:
                            energy_columns.append(key)

                with (output / "energia.csv").open(
                        "w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file,
                                            fieldnames=energy_columns,
                                            restval="")
                    writer.writeheader()
                    writer.writerows(energy_rows)

                summarize(
                    output / "energia.csv",
                    output / "resumen_energia.csv",
                    ["etapa"],
                    [c for c in energy_columns
                     if c not in ("iteracion", "etapa")],
                )

            block = jetson_monitor.integrar(monitor_rows, block_start,
                                            block_end, times)
            in_block = [r for r in monitor_rows
                        if block_start <= r["t_ns"] <= block_end]
            ram_values = [r["ram_usada_mib"] for r in in_block
                          if r["ram_usada_mib"] is not None]
            load_values = [r["gpu_carga_pct"] for r in in_block
                           if r["gpu_carga_pct"] is not None]

            energy_per_inference = {
                key[len("energia_mj_"):]: value / args.iterations
                for key, value in block.items()
                if key.startswith("energia_mj_")
            }

            metadata["energia"] = {
                "reposo": idle,
                "bloque_medido": block,
                "energia_por_iteracion_mj": energy_per_inference,
                "ram_sistema_pico_mib": max(ram_values) if ram_values
                                        else None,
                "gpu_carga_max_pct": max(load_values) if load_values
                                     else None,
                "muestras_monitor_en_bloque": len(in_block),
                "nota": (
                    "Energía por iteración = energía del bloque de "
                    "medición / iteraciones (incluye escritura de CSV "
                    "entre iteraciones). energia_dinamica = energía - "
                    "potencia en reposo x duración."
                ),
            }

        metadata.update({
            "status": "completo",
            "muestras_por_etapa": args.iterations,
            "llamadas_por_inferencia": len(signature or []),
            "registros_gpu_por_inferencia": len(gpu_signature or []),
            "supera_100_muestras": args.iterations > 100,
            "nota_total": (
                "Incluye instrumentación interna. No sumar total con "
                "etapas ni inferencia con kernels."
            ),
        })
        write_json(output / "metadata.json", metadata)

        print("\nResultados guardados en: {}".format(output))
        if args.iterations <= 100:
            print("PRUEBA CORTA: no cumple el mínimo de más de 100 "
                  "muestras.")

    except Exception as error:
        metadata["status"] = "error"
        metadata["error"] = str(error)
        write_json(output / "metadata.json", metadata)
        raise

    finally:
        if monitor is not None:
            jetson_monitor.detener_monitor(monitor)


if __name__ == "__main__":
    main()
