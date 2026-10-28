import csv
from collections import defaultdict

import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v2
from torch.nn.utils.fusion import fuse_conv_bn_eval



# CONFIGURACION
WARMUP = 8
REPEATS = 120

IMAGE_PATH = "test_images/imagen.jpeg"

device = torch.device("cuda")

torch.backends.cudnn.benchmark = True



# CARGAR MOBILENETV2
model = mobilenet_v2(pretrained=True)
model.eval()



# OPTIMIZACION: FUSION CONV2D + BATCHNORM2D
def fusionar_conv_bn(model):

    fusiones = 0

    for module in model.modules():

        if not isinstance(module, nn.Sequential):
            continue

        hijos = list(module.named_children())

        for i in range(len(hijos) - 1):

            nombre_conv, conv = hijos[i]
            nombre_bn, bn = hijos[i + 1]

            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):

                conv_fusionada = fuse_conv_bn_eval(conv, bn)

                module._modules[nombre_conv] = conv_fusionada
                module._modules[nombre_bn] = nn.Identity()

                fusiones += 1

    return fusiones


fusiones = fusionar_conv_bn(model)

batchnorm_restantes = sum(
    isinstance(module, nn.BatchNorm2d)
    for module in model.modules()
)

print("\n========== MODELO OPTIMIZADO ==========")

print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("cuDNN:", torch.backends.cudnn.version())
print("cuDNN Benchmark:", torch.backends.cudnn.benchmark)
print("Fusiones realizadas:", fusiones)
print("BatchNorm restantes:", batchnorm_restantes)

if fusiones != 52:
    raise RuntimeError(
        "Cantidad inesperada de fusiones: {}".format(fusiones)
    )

if batchnorm_restantes != 0:
    raise RuntimeError(
        "Existen operaciones BatchNorm sin fusionar"
    )

model = model.to(device)



# PREPROCESAMIENTO
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image = Image.open(IMAGE_PATH).convert("RGB")

input_tensor = preprocess(image).unsqueeze(0).to(device)



# CLASIFICACION DE OPERACIONES
def clasificar_operacion(evento):

    nombre = evento.key
    shapes = evento.input_shapes

    if nombre == "aten::_conv_depthwise2d":
        return "Depthwise Conv2D"

    if nombre == "aten::cudnn_convolution":

        if len(shapes) >= 2 and len(shapes[1]) == 4:

            pesos = shapes[1]

            if pesos[2] == 1 and pesos[3] == 1:
                return "Pointwise Conv2D"

            if pesos[2] == 3 and pesos[3] == 3:
                return "Conv2D"

    if nombre == "aten::cudnn_batch_norm":
        return "BatchNorm2D"

    if nombre == "aten::clamp_":
        return "ReLU6"

    if nombre == "aten::add":
        return "Add"

    if nombre == "aten::mean":
        return "GlobalAvgPool2D"

    if nombre == "aten::addmm":
        return "Linear"

    return None



# CALENTAMIENTO
print("\nCalentamiento...")

with torch.no_grad():

    for i in range(WARMUP):
        model(input_tensor)

torch.cuda.synchronize()

print("Calentamientos completados:", WARMUP)



# PERFILADO CON PYTORCH
print("\nPerfilando operaciones GPU...")

with torch.no_grad():

    with torch.autograd.profiler.profile(
        use_cuda=True,
        record_shapes=True
    ) as prof:

        for i in range(REPEATS):
            output = model(input_tensor)

        torch.cuda.synchronize()


# AGRUPAR RESULTADOS
resultados = defaultdict(
    lambda: {
        "tiempo_us": 0.0,
        "llamadas": 0
    }
)

detalles = []

batchnorm_perfiladas = 0

for evento in prof.key_averages(
    group_by_input_shape=True
):

    if "batch_norm" in evento.key.lower():
        batchnorm_perfiladas += evento.count

    etapa = clasificar_operacion(evento)

    if etapa is None:
        continue

    tiempo_us = evento.self_cuda_time_total

    resultados[etapa]["tiempo_us"] += tiempo_us
    resultados[etapa]["llamadas"] += evento.count

    detalles.append({
        "etapa": etapa,
        "operacion_pytorch": evento.key,
        "forma_entrada": str(evento.input_shapes),
        "llamadas": evento.count,
        "tiempo_total_ms": tiempo_us / 1000.0,
        "tiempo_promedio_ms": tiempo_us / 1000.0 / REPEATS
    })



# RESUMEN
resumen = []

for etapa, datos in resultados.items():

    media_ms = datos["tiempo_us"] / 1000.0 / REPEATS

    llamadas = datos["llamadas"] / float(REPEATS)

    resumen.append({
        "operacion": etapa,
        "media_ms": media_ms,
        "llamadas_por_inferencia": llamadas
    })

resumen.sort(
    key=lambda fila: fila["media_ms"],
    reverse=True
)


print("\n========== PERFILADO OPTIMIZADO ==========")

print(
    "{:<23} {:>15} {:>12}".format(
        "Operacion",
        "Media (ms)",
        "Llamadas"
    )
)

print("-" * 53)

for fila in resumen:

    print(
        "{:<23} {:>15.4f} {:>12.1f}".format(
            fila["operacion"],
            fila["media_ms"],
            fila["llamadas_por_inferencia"]
        )
    )



# VERIFICACION DE BATCHNORM
print("\n========== VERIFICACION ==========")

print("BatchNorm en el modelo:", batchnorm_restantes)
print("Llamadas BatchNorm perfiladas:", batchnorm_perfiladas)

if batchnorm_perfiladas != 0:
    print("ADVERTENCIA: Se detectaron operaciones BatchNorm")



# GUARDAR CSV DE RESUMEN
with open(
    "resumen_operaciones_gpu_optimized.csv",
    "w",
    newline=""
) as archivo:

    writer = csv.DictWriter(
        archivo,
        fieldnames=[
            "operacion",
            "media_ms",
            "llamadas_por_inferencia"
        ]
    )

    writer.writeheader()
    writer.writerows(resumen)



# GUARDAR CSV DETALLADO
with open(
    "detalle_operaciones_gpu_optimized.csv",
    "w",
    newline=""
) as archivo:

    writer = csv.DictWriter(
        archivo,
        fieldnames=[
            "etapa",
            "operacion_pytorch",
            "forma_entrada",
            "llamadas",
            "tiempo_total_ms",
            "tiempo_promedio_ms"
        ]
    )

    writer.writeheader()
    writer.writerows(detalles)



# REPORTE ORIGINAL PYTORCH
with open(
    "perfilado_pytorch_gpu_optimized.txt",
    "w"
) as archivo:

    archivo.write(
        prof.key_averages(
            group_by_input_shape=True
        ).table(
            sort_by="self_cuda_time_total",
            row_limit=200
        )
    )



# RESULTADO FINAL
print("\n========== RESULTADO ==========")

print("Inferencias perfiladas:", REPEATS)
print("Fusiones Conv+BN:", fusiones)

print(
    "Prediccion:",
    output.argmax(dim=1).item()
)

print("\nArchivos generados:")
print("resumen_operaciones_gpu_optimized.csv")
print("detalle_operaciones_gpu_optimized.csv")
print("perfilado_pytorch_gpu_optimized.txt")
