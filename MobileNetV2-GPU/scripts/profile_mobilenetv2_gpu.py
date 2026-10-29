import argparse
import csv
import time
import statistics

import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v2


WARMUP = 8
REPEATS = 120

device = torch.device("cuda")

# Modelo
model = mobilenet_v2(pretrained=True)
model = model.to(device)
model.eval()


# Preprocesamiento
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

parser = argparse.ArgumentParser()
parser.add_argument("image")
args = parser.parse_args()

image = Image.open(args.image).convert("RGB")
input_tensor = preprocess(image).unsqueeze(0)
input_tensor = input_tensor.to(device)

# Clasificacion de operaciones
def tipo_operacion(module):

    if isinstance(module, nn.Conv2d):

        if module.groups == module.in_channels and module.groups > 1:
            return "Depthwise Conv2D"

        if module.kernel_size == (1, 1):
            return "Pointwise Conv2D"

        return "Conv2D"

    if isinstance(module, nn.BatchNorm2d):
        return "BatchNorm2D"

    if isinstance(module, nn.ReLU6):
        return "ReLU6"

    if isinstance(module, nn.AdaptiveAvgPool2d):
        return "GlobalAvgPool2D"

    if isinstance(module, nn.Linear):
        return "Linear"

    return None


def guardar_csv(nombre, columnas, registros):
    with open(nombre, "w", newline="") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=columnas)
        writer.writeheader()
        writer.writerows(registros)



# Calentamiento
print("Calentamiento...")

with torch.no_grad():
    for i in range(WARMUP):
        model(input_tensor)

torch.cuda.synchronize()


# Perfilado de inferencia completa
print("Midiendo inferencia completa...")

latencias = []
registros_inferencia = []

with torch.no_grad():

    for i in range(REPEATS):

        torch.cuda.synchronize()
        inicio = time.perf_counter()

        output = model(input_tensor)

        torch.cuda.synchronize()
        fin = time.perf_counter()

        tiempo_ms = (fin - inicio) * 1000.0

        latencias.append(tiempo_ms)

        registros_inferencia.append({
            "iteracion": i + 1,
            "latencia_ms": tiempo_ms
        })

guardar_csv(
    "inferencias_gpu.csv",
    ["iteracion", "latencia_ms"],
    registros_inferencia
)


# Perfilado por operaciones
print("Midiendo operaciones...")

eventos = []
handles = []


def pre_hook(nombre, etapa):

    def hook(module, inputs):

        inicio = torch.cuda.Event(enable_timing=True)
        fin = torch.cuda.Event(enable_timing=True)

        inicio.record()

        eventos.append({
            "nombre": nombre,
            "etapa": etapa,
            "forma_tensor": str(tuple(inputs[0].shape)),
            "inicio": inicio,
            "fin": fin
        })

    return hook


def post_hook(module, inputs, output):
    eventos[-1]["fin"].record()


for nombre, module in model.named_modules():

    etapa = tipo_operacion(module)

    if etapa is not None:

        handles.append(
            module.register_forward_pre_hook(
                pre_hook(nombre, etapa)
            )
        )

        handles.append(
            module.register_forward_hook(post_hook)
        )


registros_capas = []

with torch.no_grad():

    for i in range(REPEATS):

        eventos.clear()

        model(input_tensor)

        torch.cuda.synchronize()

        for evento in eventos:

            tiempo_ms = evento["inicio"].elapsed_time(
                evento["fin"]
            )

            registros_capas.append({
                "iteracion": i + 1,
                "nombre": evento["nombre"],
                "etapa": evento["etapa"],
                "forma_tensor": evento["forma_tensor"],
                "tiempo_ms": tiempo_ms
            })


for handle in handles:
    handle.remove()


guardar_csv(
    "capas_gpu.csv",
    [
        "iteracion",
        "nombre",
        "etapa",
        "forma_tensor",
        "tiempo_ms"
    ],
    registros_capas
)


# Estadisticas
ordenadas = sorted(latencias)
p95 = ordenadas[int(0.95 * (len(ordenadas) - 1))]

media = statistics.mean(latencias)
mediana = statistics.median(latencias)
desviacion = statistics.stdev(latencias)

throughput = 1000.0 / media

print("\n========== RESULTADOS GPU ==========")

print("GPU:", torch.cuda.get_device_name(0))
print("Calentamiento:", WARMUP)
print("Mediciones:", REPEATS)

print("Media: {:.4f} ms".format(media))
print("Mediana: {:.4f} ms".format(mediana))
print("Desviacion: {:.4f} ms".format(desviacion))
print("P95: {:.4f} ms".format(p95))
print("Throughput: {:.4f} imagenes/s".format(throughput))

print("Prediccion:", output.argmax(dim=1).item())

print("\nArchivos generados:")
print("inferencias_gpu.csv")
print("capas_gpu.csv")
