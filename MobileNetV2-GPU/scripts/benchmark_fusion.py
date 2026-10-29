import csv
import time
import copy
import statistics

import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v2
from torch.nn.utils.fusion import fuse_conv_bn_eval



# Configuracion
WARMUP = 8
REPEATS = 120

device = torch.device("cuda")

# Mantenemos la optimizacion anterior
torch.backends.cudnn.benchmark = True



# Modelo baseline
model_original = mobilenet_v2(pretrained=True)
model_original.eval()



# Fusion Conv2D + BatchNorm2D
def fusionar_modelo(model):

    fusiones = 0

    for module in model.modules():

        if not isinstance(module, nn.Sequential):
            continue

        hijos = list(module._modules.items())

        for i in range(len(hijos) - 1):

            nombre_conv, conv = hijos[i]
            nombre_bn, bn = hijos[i + 1]

            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):

                module._modules[nombre_conv] = fuse_conv_bn_eval(
                    conv, bn
                )

                module._modules[nombre_bn] = nn.Identity()

                fusiones += 1

    return fusiones


# Copiar modelo para no modificar baseline
model_fusionado = copy.deepcopy(model_original)

fusiones = fusionar_modelo(model_fusionado)

print("Fusiones realizadas:", fusiones)

if fusiones != 52:
    raise RuntimeError(
        "Se esperaban 52 fusiones Conv2D + BatchNorm2D"
    )

# Transferencia a GPU
model_original = model_original.to(device)
model_fusionado = model_fusionado.to(device)



# Imagen de prueba
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image = Image.open(
    "test_images/imagen.jpeg"
).convert("RGB")

input_tensor = preprocess(image).unsqueeze(0).to(device)


# Validacion numerica
print("\nValidando resultados...")

with torch.no_grad():

    salida_original = model_original(input_tensor)

    salida_fusionada = model_fusionado(input_tensor)

torch.cuda.synchronize()

rmse = torch.sqrt(
    torch.mean(
        (salida_original - salida_fusionada) ** 2
    )
).item()

error_max = torch.max(
    torch.abs(salida_original - salida_fusionada)
).item()

pred_original = salida_original.argmax(dim=1).item()
pred_fusionada = salida_fusionada.argmax(dim=1).item()

print("RMSE:", rmse)
print("Error maximo:", error_max)
print("Prediccion original:", pred_original)
print("Prediccion fusionada:", pred_fusionada)

if rmse >= 1e-3 or pred_original != pred_fusionada:
    raise RuntimeError(
        "La validacion numerica no fue satisfactoria"
    )



# Funcion de benchmark
def medir_modelo(model, nombre):

    print("\nEjecutando:", nombre)

    # Calentamiento
    with torch.no_grad():

        for _ in range(WARMUP):
            model(input_tensor)

    torch.cuda.synchronize()

    latencias = []

    # Inferencias medidas
    with torch.no_grad():

        for i in range(REPEATS):

            torch.cuda.synchronize()

            inicio = time.perf_counter()

            output = model(input_tensor)

            torch.cuda.synchronize()

            fin = time.perf_counter()

            latencia = (fin - inicio) * 1000.0

            latencias.append(latencia)

    media = statistics.mean(latencias)
    mediana = statistics.median(latencias)
    desviacion = statistics.stdev(latencias)

    ordenadas = sorted(latencias)

    p95 = ordenadas[
        int(0.95 * (len(ordenadas) - 1))
    ]

    throughput = 1000.0 / media

    print("Media: {:.4f} ms".format(media))
    print("Mediana: {:.4f} ms".format(mediana))
    print("Desviacion: {:.4f} ms".format(desviacion))
    print("P95: {:.4f} ms".format(p95))
    print("Throughput: {:.4f} imagenes/s".format(throughput))

    with open(
        "fusion_{}.csv".format(nombre),
        "w",
        newline=""
    ) as archivo:

        writer = csv.writer(archivo)

        writer.writerow([
            "iteracion",
            "latencia_ms"
        ])

        for i, latencia in enumerate(latencias):
            writer.writerow([i + 1, latencia])

    return media



# Comparacion
media_original = medir_modelo(
    model_original,
    "original"
)

media_fusionada = medir_modelo(
    model_fusionado,
    "optimizado"
)

speedup = media_original / media_fusionada

mejora = (
    (media_original - media_fusionada)
    / media_original
) * 100.0

print("\n========== COMPARACION ==========")

print("Fusiones:", fusiones)

print(
    "Baseline: {:.4f} ms".format(media_original)
)

print(
    "Fusionado: {:.4f} ms".format(media_fusionada)
)

print(
    "Speedup: {:.4f}x".format(speedup)
)

print(
    "Mejora: {:.2f}%".format(mejora)
)
