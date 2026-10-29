import csv
import torch
from collections import defaultdict

from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v2


# Configuracion
WARMUP = 8
REPEATS = 120

device = torch.device("cuda")

model = mobilenet_v2(pretrained=True).to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image = Image.open("test_images/imagen.jpeg").convert("RGB")
input_tensor = preprocess(image).unsqueeze(0).to(device)



# Clasificar operaciones
def clasificar(evento):

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


# Calentamiento
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)

print("\nCalentamiento...")

with torch.no_grad():
    for _ in range(WARMUP):
        model(input_tensor)

torch.cuda.synchronize()



# Perfilado
print("Perfilando MobileNetV2...")

with torch.no_grad():

    with torch.autograd.profiler.profile(
        use_cuda=True,
        record_shapes=True
    ) as prof:

        for _ in range(REPEATS):
            model(input_tensor)

        torch.cuda.synchronize()



# Agrupar operaciones
resultados = defaultdict(
    lambda: {
        "tiempo_us": 0.0,
        "llamadas": 0
    }
)

detalles = []

for evento in prof.key_averages(
    group_by_input_shape=True
):

    etapa = clasificar(evento)

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
        "tiempo_promedio_ms": (
            tiempo_us / 1000.0 / REPEATS
        )
    })



# Resumen
print("\n========== PERFILADO POR OPERACION ==========")

print(
    "{:<23} {:>15} {:>12}".format(
        "Operacion",
        "Media (ms)",
        "Llamadas"
    )
)

print("-" * 53)

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
    key=lambda x: x["media_ms"],
    reverse=True
)

for fila in resumen:

    print(
        "{:<23} {:>15.4f} {:>12.1f}".format(
            fila["operacion"],
            fila["media_ms"],
            fila["llamadas_por_inferencia"]
        )
    )



# Guardar CSV resumen
with open(
    "resumen_operaciones_gpu.csv",
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



# Guardar detalles
with open(
    "detalle_operaciones_gpu.csv",
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



# Reporte original
with open(
    "perfilado_pytorch_gpu.txt",
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

print("\nArchivos generados:")
print("resumen_operaciones_gpu.csv")
print("detalle_operaciones_gpu.csv")
print("perfilado_pytorch_gpu.txt")
