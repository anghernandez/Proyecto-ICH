"""Compara dos perfilados (p. ej. jetson_cpu vs jetson_gpu). Python 3.6+.

Lee resumen_etapas.csv y resumen_kernels.csv de dos carpetas de
resultados y genera, en --out:

  comparacion_etapas.csv   tiempo por etapa y speedup (base / nuevo)
  comparacion_capas.csv    tiempo por llamada (capa) y speedup
  comparacion_por_tipo.csv suma por tipo de operación, % del total
                           y speedup (pointwise, depthwise, conv2d...)
  comparacion.json         resumen, incluida la cota de Amdahl

La cota de Amdahl indica el speedup máximo de la inferencia si las
operaciones aceleradas tardaran cero: 1 / (1 - f), con f la fracción
del tiempo de kernels que ocupaban en la base.

Uso:
    python3 app/comparar_resultados.py \
        --base resultados/jetson_cpu --nuevo resultados/jetson_gpu \
        --out resultados/comparacion_jetson
"""

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path


ACELERADAS = ["pointwise_conv", "depthwise_conv", "conv2d"]


def load_summary(folder, name, keys, metric="tiempo_ms"):
    """Devuelve {clave: fila} para la métrica indicada."""
    path = Path(folder) / name
    if not path.exists():
        candidates = sorted(Path(folder).glob(name.replace(".csv", "*.csv")))
        if not candidates:
            raise SystemExit("No existe {} en {}".format(name, folder))
        path = candidates[0]

    result = OrderedDict()
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["metrica"] == metric:
                result[tuple(row[key] for key in keys)] = row
    return result


def speedup(base, new):
    return base / new if new > 0 else None


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True,
                        help="Carpeta de referencia (p. ej. jetson_cpu).")
    parser.add_argument("--nuevo", required=True,
                        help="Carpeta a comparar (p. ej. jetson_gpu).")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- Etapas ----------------
    base_stages = load_summary(args.base, "resumen_etapas.csv", ["etapa"])
    new_stages = load_summary(args.nuevo, "resumen_etapas.csv", ["etapa"])

    stage_rows = []
    for key, base in base_stages.items():
        new = new_stages.get(key)
        if new is None:
            continue
        b, n = float(base["media"]), float(new["media"])
        stage_rows.append(OrderedDict([
            ("etapa", key[0]),
            ("media_base_ms", b),
            ("media_nuevo_ms", n),
            ("mediana_base_ms", float(base["mediana"])),
            ("mediana_nuevo_ms", float(new["mediana"])),
            ("p95_base_ms", float(base["p95"])),
            ("p95_nuevo_ms", float(new["p95"])),
            ("speedup_media", speedup(b, n)),
        ]))
    write_csv(out / "comparacion_etapas.csv", stage_rows)

    # ---------------- Capas ----------------
    keys = ["indice_llamada", "etapa", "forma_tensor"]
    base_layers = load_summary(args.base, "resumen_kernels.csv", keys)
    new_layers = load_summary(args.nuevo, "resumen_kernels.csv", keys)

    if list(base_layers.keys()) != list(new_layers.keys()):
        raise SystemExit(
            "Las secuencias de capas no coinciden (índice, tipo, forma). "
            "Revisa que ambos perfilados usen la misma red y entrada.")

    layer_rows = []
    by_type = OrderedDict()
    for key, base in base_layers.items():
        new = new_layers[key]
        b, n = float(base["media"]), float(new["media"])
        layer_rows.append(OrderedDict([
            ("indice_llamada", int(key[0])),
            ("etapa", key[1]),
            ("forma_tensor", key[2]),
            ("media_base_ms", b),
            ("media_nuevo_ms", n),
            ("desviacion_base_ms", float(base["desviacion"])),
            ("desviacion_nuevo_ms", float(new["desviacion"])),
            ("speedup", speedup(b, n)),
        ]))
        totals = by_type.setdefault(key[1], [0, 0.0, 0.0])
        totals[0] += 1
        totals[1] += b
        totals[2] += n
    write_csv(out / "comparacion_capas.csv", layer_rows)

    total_base = sum(t[1] for t in by_type.values())
    total_new = sum(t[2] for t in by_type.values())

    type_rows = []
    for name, (count, b, n) in sorted(by_type.items(),
                                      key=lambda item: -item[1][1]):
        type_rows.append(OrderedDict([
            ("etapa", name),
            ("llamadas", count),
            ("suma_base_ms", b),
            ("pct_base", 100 * b / total_base if total_base else 0),
            ("suma_nuevo_ms", n),
            ("pct_nuevo", 100 * n / total_new if total_new else 0),
            ("speedup", speedup(b, n)),
        ]))
    write_csv(out / "comparacion_por_tipo.csv", type_rows)

    # ---------------- Resumen ----------------
    accelerated_base = sum(by_type[k][1] for k in ACELERADAS
                           if k in by_type)
    fraction = accelerated_base / total_base if total_base else 0
    inference = next((r for r in stage_rows
                      if r["etapa"] == "inferencia_cpp"), None)

    summary = OrderedDict([
        ("base", str(args.base)),
        ("nuevo", str(args.nuevo)),
        ("operaciones_aceleradas", ACELERADAS),
        ("fraccion_aceleradas_en_base_pct", 100 * fraction),
        ("cumple_50_pct", fraction >= 0.5),
        ("speedup_max_amdahl", (1 / (1 - fraction))
         if fraction < 1 else None),
        ("speedup_inferencia", inference["speedup_media"]
         if inference else None),
        ("suma_kernels_base_ms", total_base),
        ("suma_kernels_nuevo_ms", total_new),
    ])
    (out / "comparacion.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    print("Operaciones aceleradas en la base: {:.1f} % del tiempo de "
          "kernels".format(100 * fraction))
    if summary["speedup_max_amdahl"]:
        print("Speedup máximo posible (Amdahl): {:.2f}x".format(
            summary["speedup_max_amdahl"]))
    if inference:
        print("Speedup de la inferencia: {:.2f}x".format(
            inference["speedup_media"]))
    for row in type_rows:
        if row["etapa"] in ACELERADAS:
            print("  {:<16} {:>9.2f} ms -> {:>9.2f} ms  ({:.2f}x)".format(
                row["etapa"], row["suma_base_ms"], row["suma_nuevo_ms"],
                row["speedup"] or 0))
    print("Resultados en: {}".format(out))


if __name__ == "__main__":
    main()
