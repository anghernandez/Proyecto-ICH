"""Monitor de uso de GPU, potencia y memoria para Jetson.

Corre como PROCESO SEPARADO del perfilado. Motivo: las funciones
pybind11 retienen el GIL mientras ejecutan C++, así que un hilo de
muestreo dentro del mismo proceso se quedaría congelado durante
cada capa larga. Como proceso aparte, cae en otro núcleo (la
inferencia usa 1 hilo) y no altera las mediciones.

Lee directamente de sysfs (lo mismo que usan tegrastats y jtop):
  - Carga de GPU:  /sys/devices/gpu.0/load  (0-1000 = 0-100 %)
  - Frecuencia GPU: devfreq de la GPU
  - Potencia:      sensores INA3221 (rieles POM_5V_IN, _GPU, _CPU
                   en la Nano; los nombres se leen del propio sysfs)
  - RAM sistema:   /proc/meminfo (CPU y GPU comparten la DRAM)
  - CPU:           /proc/stat

Cada muestra lleva una marca de tiempo CLOCK_MONOTONIC en ns, el
mismo reloj de time.perf_counter_ns() en Linux. Así las ventanas
de cada etapa del perfilado se pueden cruzar con estas muestras
para obtener energía y uso de GPU por etapa (ver integrar()).

Compatible con Python 3.6 (JetPack 4.x).

Uso directo:
    python3 app/jetson_monitor.py --out monitoreo.csv --interval-ms 20

Uso desde el script de perfilado:
    from app.jetson_monitor import iniciar_monitor, detener_monitor
"""

import argparse
import bisect
import csv
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time


# ------------------------------------------------------------
# Reloj común
# ------------------------------------------------------------

def monotonic_ns():
    """CLOCK_MONOTONIC en ns (igual a perf_counter_ns en Linux)."""
    if hasattr(time, "clock_gettime_ns"):
        return time.clock_gettime_ns(time.CLOCK_MONOTONIC)

    return int(time.clock_gettime(time.CLOCK_MONOTONIC) * 1e9)


# ------------------------------------------------------------
# Descubrimiento de rutas sysfs
# ------------------------------------------------------------

def _first(patterns):
    for pattern in patterns:
        found = sorted(glob.glob(pattern))
        if found:
            return found[0]
    return None


def _read(path):
    try:
        with open(path) as file:
            return file.read().strip()
    except (OSError, IOError):
        return None


def descubrir_fuentes():
    """Busca las rutas disponibles en esta Jetson."""
    sources = {
        "gpu_load": _first([
            "/sys/devices/gpu.0/load",
            "/sys/devices/platform/gpu.0/load",
            "/sys/devices/platform/*.gpu/load",
            "/sys/devices/platform/*/gpu.0/load",
        ]),
        "gpu_freq": _first([
            "/sys/devices/gpu.0/devfreq/*/cur_freq",
            "/sys/class/devfreq/*.gpu/cur_freq",
            "/sys/class/devfreq/*gpu*/cur_freq",
        ]),
        "rieles": [],
    }

    # JetPack 4.x (Nano, TX2): driver ina3221x con interfaz iio.
    # Solo in_powerN_input (medición continua). El driver también
    # expone in_powerN_trigger_input, que dispara una conversión
    # aparte: daría una columna duplicada con otro valor.
    for device in sorted(glob.glob(
        "/sys/bus/i2c/drivers/ina3221x/*/iio:device*"
    )):
        for power in sorted(glob.glob(device + "/in_power*_input")):
            match = re.match(r"in_power(\d+)_input$",
                             os.path.basename(power))
            if not match:
                continue
            channel = match.group(1)
            name = _read("{}/rail_name_{}".format(device, channel))
            sources["rieles"].append({
                "nombre": name or "canal{}".format(channel),
                "tipo": "iio",
                "potencia": power,
            })

    # JetPack 5/6 (Xavier/Orin): driver ina3221 con hwmon.
    for hwmon in sorted(glob.glob(
        "/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*"
    )):
        for label in sorted(glob.glob(hwmon + "/in*_label")):
            channel = label.rsplit("/in", 1)[1].split("_")[0]
            name = _read(label)
            current = "{}/curr{}_input".format(hwmon, channel)
            voltage = "{}/in{}_input".format(hwmon, channel)
            if name and os.path.exists(current):
                sources["rieles"].append({
                    "nombre": name,
                    "tipo": "hwmon",
                    "corriente": current,
                    "voltaje": voltage,
                })

    # Un riel por nombre (se conserva el primero encontrado).
    unique = []
    seen = set()
    for rail in sources["rieles"]:
        if rail["nombre"] not in seen:
            seen.add(rail["nombre"])
            unique.append(rail)
    sources["rieles"] = unique

    return sources


# ------------------------------------------------------------
# Lecturas individuales
# ------------------------------------------------------------

def _power_mw(rail):
    if rail["tipo"] == "iio":
        value = _read(rail["potencia"])
        return float(value) if value is not None else None

    current = _read(rail["corriente"])   # mA
    voltage = _read(rail["voltaje"])     # mV
    if current is None or voltage is None:
        return None
    return float(current) * float(voltage) / 1000.0


def verificar_fuentes():
    """Indica qué fuentes se pueden leer con el usuario actual.

    Los rieles INA3221 suelen ser solo de root. Para habilitarlos
    (se pierde al reiniciar):
        sudo chmod a+r .../iio:device0/rail_name_* \
                       .../iio:device0/in_power*_input
    """
    sources = descubrir_fuentes()

    rails = {}
    for rail in sources["rieles"]:
        rails[rail["nombre"]] = _power_mw(rail) is not None

    return {
        "gpu_carga": bool(sources["gpu_load"])
        and _read(sources["gpu_load"]) is not None,
        "gpu_freq": bool(sources["gpu_freq"])
        and _read(sources["gpu_freq"]) is not None,
        "rieles": rails,
        "rieles_encontrados": len(sources["rieles"]),
    }


def _meminfo():
    values = {}
    try:
        with open("/proc/meminfo") as file:
            for line in file:
                key, rest = line.split(":", 1)
                values[key] = int(rest.split()[0])  # kB
    except (OSError, IOError, ValueError):
        return None, None

    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if total is None or available is None:
        return None, None

    used_mib = (total - available) / 1024.0
    swap_mib = (
        values.get("SwapTotal", 0) - values.get("SwapFree", 0)
    ) / 1024.0
    return used_mib, swap_mib


def _cpu_times():
    line = _read("/proc/stat")
    if line is None:
        return None
    fields = [int(x) for x in line.splitlines()[0].split()[1:]]
    idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
    return sum(fields), idle


# ------------------------------------------------------------
# Bucle de muestreo
# ------------------------------------------------------------

_stop = False


def _handle_stop(signum, frame):
    global _stop
    _stop = True


def muestrear(out_path, interval_ms, duration_s=None):
    sources = descubrir_fuentes()
    rail_names = [rail["nombre"] for rail in sources["rieles"]]

    meta_path = os.path.splitext(out_path)[0] + "_fuentes.json"
    with open(meta_path, "w") as file:
        json.dump(
            {"intervalo_ms": interval_ms, "fuentes": sources},
            file, indent=2,
        )

    columns = (
        ["t_ns", "gpu_carga_pct", "gpu_freq_mhz",
         "ram_usada_mib", "swap_usada_mib", "cpu_total_pct"]
        + ["p_{}_mw".format(name) for name in rail_names]
    )

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    interval_ns = int(interval_ms * 1e6)
    start = monotonic_ns()
    next_tick = start
    previous_cpu = _cpu_times()

    with open(out_path, "w") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        file.flush()

        count = 0
        while not _stop:
            now = monotonic_ns()

            load = _read(sources["gpu_load"]) if sources["gpu_load"] else None
            freq = _read(sources["gpu_freq"]) if sources["gpu_freq"] else None
            ram, swap = _meminfo()

            cpu_pct = None
            current_cpu = _cpu_times()
            if previous_cpu and current_cpu:
                total = current_cpu[0] - previous_cpu[0]
                idle = current_cpu[1] - previous_cpu[1]
                if total > 0:
                    cpu_pct = 100.0 * (total - idle) / total
            previous_cpu = current_cpu

            row = [
                now,
                float(load) / 10.0 if load is not None else "",
                float(freq) / 1e6 if freq is not None else "",
                ram if ram is not None else "",
                swap if swap is not None else "",
                cpu_pct if cpu_pct is not None else "",
            ]
            for rail in sources["rieles"]:
                value = _power_mw(rail)
                row.append(value if value is not None else "")

            writer.writerow(row)
            count += 1
            if count % 50 == 0:
                file.flush()

            if duration_s is not None and now - start >= duration_s * 1e9:
                break

            # Programación sin deriva acumulada.
            next_tick += interval_ns
            delay = next_tick - monotonic_ns()
            if delay > 0:
                time.sleep(delay / 1e9)
            else:
                next_tick = monotonic_ns()

        file.flush()


# ------------------------------------------------------------
# Control desde otro script
# ------------------------------------------------------------

def iniciar_monitor(out_path, interval_ms=20, espera_s=1.0):
    """Lanza el monitor como proceso aparte y espera su encabezado."""
    process = subprocess.Popen([
        sys.executable, os.path.abspath(__file__),
        "--out", str(out_path),
        "--interval-ms", str(interval_ms),
    ])

    deadline = time.time() + 10
    while time.time() < deadline:
        if os.path.exists(str(out_path)) and os.path.getsize(str(out_path)) > 0:
            break
        time.sleep(0.05)

    time.sleep(espera_s)
    return process


def detener_monitor(process, timeout_s=5):
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()


# ------------------------------------------------------------
# Análisis: energía y uso por ventana de tiempo
# ------------------------------------------------------------

def cargar_monitoreo(path):
    with open(str(path)) as file:
        reader = csv.DictReader(file)
        rows = []
        for row in reader:
            parsed = {}
            for key, value in row.items():
                parsed[key] = float(value) if value != "" else None
            rows.append(parsed)
    return rows


def tiempos(rows):
    """Lista de marcas de tiempo, para reutilizar en integrar()."""
    return [row["t_ns"] for row in rows]


def _value_at(rows, times, key, t):
    """Valor de una columna en el instante t (interpolación lineal)."""
    index = bisect.bisect_left(times, t)

    if index <= 0:
        return rows[0][key]
    if index >= len(rows):
        return rows[-1][key]

    before = rows[index - 1]
    after = rows[index]

    if before[key] is None or after[key] is None:
        return after[key] if before[key] is None else before[key]

    span = after["t_ns"] - before["t_ns"]
    weight = (t - before["t_ns"]) / span if span > 0 else 0.0
    return before[key] + weight * (after[key] - before[key])


def integrar(rows, t0_ns, t1_ns, times=None):
    """Energía (mJ) por riel y promedios dentro de [t0, t1].

    Integra por trapecios usando las muestras que caen en la
    ventana más una interpolación en los bordes. Si la ventana es
    más corta que el intervalo de muestreo, la energía se estima
    con la potencia interpolada (útil, pero menos precisa: para
    etapas cortas conviene mirar la media de las 120 corridas).

    Devuelve {} si la ventana queda fuera del rango monitoreado.
    """
    if not rows or t1_ns <= t0_ns:
        return {}

    if times is None:
        times = tiempos(rows)

    if t0_ns < times[0] or t1_ns > times[-1]:
        return {}

    power_keys = [k for k in rows[0] if k.startswith("p_")]
    average_keys = ["gpu_carga_pct", "gpu_freq_mhz",
                    "ram_usada_mib", "cpu_total_pct"]

    low = bisect.bisect_right(times, t0_ns)
    high = bisect.bisect_left(times, t1_ns)
    inner = rows[low:high]

    result = {"duracion_ms": (t1_ns - t0_ns) / 1e6}

    for key in power_keys + average_keys:
        if key not in rows[0]:
            continue

        points = [(t0_ns, _value_at(rows, times, key, t0_ns))]
        points += [
            (row["t_ns"], row[key])
            for row in inner
            if row[key] is not None
        ]
        points.append((t1_ns, _value_at(rows, times, key, t1_ns)))
        points = [(t, v) for t, v in points if v is not None]

        if len(points) < 2:
            continue

        area = 0.0  # unidad * ns
        for (ta, va), (tb, vb) in zip(points, points[1:]):
            area += (va + vb) / 2.0 * (tb - ta)

        mean = area / (t1_ns - t0_ns)
        name = key[2:-3] if key.startswith("p_") else key

        if key in power_keys:
            result["potencia_media_mw_" + name] = mean
            result["energia_mj_" + name] = area / 1e9  # mW*s = mJ
        else:
            result[name + "_media"] = mean

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out",
                        help="CSV de salida (obligatorio salvo con "
                             "--fuentes).")
    parser.add_argument("--interval-ms", type=float, default=20)
    parser.add_argument("--duration", type=float, default=None,
                        help="Segundos; sin valor corre hasta SIGTERM.")
    parser.add_argument("--fuentes", action="store_true",
                        help="Solo muestra las rutas encontradas.")
    args = parser.parse_args()

    if args.fuentes:
        print(json.dumps(descubrir_fuentes(), indent=2))
        return

    if not args.out:
        parser.error("falta --out")

    muestrear(args.out, args.interval_ms, args.duration)


if __name__ == "__main__":
    main()
