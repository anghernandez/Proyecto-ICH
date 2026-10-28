import json
import csv
import re
from collections import defaultdict

import onnx
from onnx import helper

ONNX_PATH = "mobilenetv2_fusion.onnx"

PROFILE_PATH = (
    "resultados_tensorrt/perfilado/perfil_capas.json"
)

OUTPUT_PATH = (
    "resultados_tensorrt/perfilado/resumen_operaciones.csv"
)

model = onnx.load(ONNX_PATH)

# Clasificar convoluciones por sus atributos.
conv_types = {}

for index, node in enumerate(model.graph.node):

    if node.op_type != "Conv":
        continue

    attrs = {
        attr.name: helper.get_attribute_value(attr)
        for attr in node.attribute
    }

    kernel = list(attrs.get("kernel_shape", []))
    groups = int(attrs.get("group", 1))

    if kernel == [1, 1]:
        tipo = "Pointwise Conv2D"

    elif kernel == [3, 3] and groups > 1:
        tipo = "Depthwise Conv2D"

    elif kernel == [3, 3] and groups == 1:
        tipo = "Conv2D convencional"

    else:
        tipo = "Otra convolucion"

    # TensorRT utiliza nombres como Conv_13.
    conv_types["Conv_{}".format(index)] = tipo


with open(PROFILE_PATH) as f:
    profile = json.load(f)

results = defaultdict(
    lambda: {"count": 0, "time_ms": 0.0}
)

unmatched = []

for layer in profile:

    if "name" not in layer:
        continue

    name = layer["name"]
    time_ms = float(layer["averageMs"])

    match = re.search(r"\bConv_\d+\b", name)

    if match:
        conv_name = match.group(0)

        tipo = conv_types.get(conv_name)

        if tipo is None:
            unmatched.append(name)
            tipo = "Convolucion no identificada"

    elif "Gemm" in name:
        tipo = "Linear"

    elif "GlobalAveragePool" in name:
        tipo = "GlobalAvgPool2D"

    elif "Shuffle" in name:
        tipo = "Shuffle"

    else:
        tipo = "Otras operaciones"

    results[tipo]["count"] += 1
    results[tipo]["time_ms"] += time_ms


total = sum(
    value["time_ms"]
    for value in results.values()
)

print("\nPERFILADO TENSORRT POR TIPO")
print("-" * 75)

for tipo, value in sorted(
    results.items(),
    key=lambda x: x[1]["time_ms"],
    reverse=True
):

    ms = value["time_ms"]
    percentage = ms / total * 100

    print(
        "{:<28} {:>3} grupos  {:>8.4f} ms  {:>6.2f}%".format(
            tipo,
            value["count"],
            ms,
            percentage
        )
    )

print("\nTiempo acumulado: {:.4f} ms".format(total))

if unmatched:
    print("\nADVERTENCIA: capas sin correspondencia:")
    for name in unmatched:
        print(name)

with open(OUTPUT_PATH, "w") as f:

    writer = csv.writer(f)

    writer.writerow([
        "operacion",
        "grupos",
        "tiempo_acumulado_ms",
        "porcentaje"
    ])

    for tipo, value in sorted(
        results.items(),
        key=lambda x: x[1]["time_ms"],
        reverse=True
    ):
        writer.writerow([
            tipo,
            value["count"],
            value["time_ms"],
            100 * value["time_ms"] / total
        ])

print("\nArchivo:", OUTPUT_PATH)
