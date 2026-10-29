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


# Modelo e imagen
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



# Seleccionar operaciones
TIPOS = (
    nn.Conv2d,
    nn.BatchNorm2d,
    nn.ReLU6,
    nn.AdaptiveAvgPool2d,
    nn.Linear
)

modulos = [
    module
    for module in model.modules()
    if isinstance(module, TIPOS)
]



# Hooks
handles = []
eventos = []


def pre_hook_simple(module, inputs):
    pass


def post_hook_simple(module, inputs, output):
    pass


def pre_hook_evento(module, inputs):

    inicio = torch.cuda.Event(enable_timing=True)
    fin = torch.cuda.Event(enable_timing=True)

    inicio.record()

    eventos.append([inicio, fin])


def post_hook_evento(module, inputs, output):
    eventos[-1][1].record()


def registrar_hooks(modo):

    global handles

    handles = []

    if modo == "A":
        return

    for module in modulos:

        if modo == "B":

            handles.append(
                module.register_forward_pre_hook(
                    pre_hook_simple
                )
            )

            handles.append(
                module.register_forward_hook(
                    post_hook_simple
                )
            )

        elif modo == "C":

            handles.append(
                module.register_forward_pre_hook(
                    pre_hook_evento
                )
            )

            handles.append(
                module.register_forward_hook(
                    post_hook_evento
                )
            )


def eliminar_hooks():

    for handle in handles:
        handle.remove()

    handles.clear()



# Ejecutar prueba
def ejecutar_prueba(modo):

    registrar_hooks(modo)

    print("\nPrueba:", modo)

    # Calentamiento
    with torch.no_grad():

        for _ in range(WARMUP):

            if modo == "C":
                eventos.clear()

            model(input_tensor)

            if modo == "C":
                torch.cuda.synchronize()

    torch.cuda.synchronize()

    latencias = []

    # Mediciones
    with torch.no_grad():

        for i in range(REPEATS):

            if modo == "C":
                eventos.clear()

            torch.cuda.synchronize()
            inicio = time.perf_counter()

            model(input_tensor)

            torch.cuda.synchronize()
            fin = time.perf_counter()

            latencia_ms = (fin - inicio) * 1000

            latencias.append(latencia_ms)

    eliminar_hooks()

    media = statistics.mean(latencias)
    mediana = statistics.median(latencias)
    desviacion = statistics.stdev(latencias)

    print("Media: {:.4f} ms".format(media))
    print("Mediana: {:.4f} ms".format(mediana))
    print("Desviacion: {:.4f} ms".format(desviacion))

    return {
        "prueba": modo,
        "media_ms": media,
        "mediana_ms": mediana,
        "desviacion_ms": desviacion
    }



# Comparación
resultados = []

for modo in ["A", "B", "C"]:
    resultados.append(ejecutar_prueba(modo))

baseline = resultados[0]["media_ms"]

print("\n========== COMPARACION ==========")

for resultado in resultados:

    overhead = resultado["media_ms"] - baseline

    print(
        "{} | Latencia: {:.4f} ms | Diferencia: {:+.4f} ms".format(
            resultado["prueba"],
            resultado["media_ms"],
            overhead
        )
    )



# Guardar CSV
with open("overhead_gpu.csv", "w", newline="") as archivo:

    writer = csv.DictWriter(
        archivo,
        fieldnames=[
            "prueba",
            "media_ms",
            "mediana_ms",
            "desviacion_ms"
        ]
    )

    writer.writeheader()
    writer.writerows(resultados)

print("\nResultados guardados en overhead_gpu.csv")
