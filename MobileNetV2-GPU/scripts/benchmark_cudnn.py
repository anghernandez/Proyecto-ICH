import argparse
import csv
import time
import statistics

import torch

from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v2



# Configuración
parser = argparse.ArgumentParser()

parser.add_argument(
    "--benchmark",
    type=int,
    choices=[0, 1],
    required=True
)

args = parser.parse_args()

WARMUP = 8
REPEATS = 120

device = torch.device("cuda")

torch.backends.cudnn.benchmark = bool(args.benchmark)

print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("cuDNN:", torch.backends.cudnn.version())
print("Benchmark:", torch.backends.cudnn.benchmark)



# Modelo
model = mobilenet_v2(pretrained=True)
model = model.to(device)
model.eval()


# Imagen
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

input_tensor = preprocess(image).unsqueeze(0)
input_tensor = input_tensor.to(device)



# Calentamiento
print("\nCalentamiento...")

with torch.no_grad():

    for _ in range(WARMUP):
        model(input_tensor)

torch.cuda.synchronize()


# Inferencias medidas
print("Midiendo inferencias...")

latencias = []

with torch.no_grad():

    for i in range(REPEATS):

        torch.cuda.synchronize()

        inicio = time.perf_counter()

        output = model(input_tensor)

        torch.cuda.synchronize()

        fin = time.perf_counter()

        latencia_ms = (fin - inicio) * 1000

        latencias.append(latencia_ms)



# Estadisticas
media = statistics.mean(latencias)
mediana = statistics.median(latencias)
desviacion = statistics.stdev(latencias)

ordenadas = sorted(latencias)

p95 = ordenadas[
    int(0.95 * (len(ordenadas) - 1))
]

throughput = 1000.0 / media


print("\n========== RESULTADOS ==========")

print("Benchmark:", args.benchmark)
print("Media: {:.4f} ms".format(media))
print("Mediana: {:.4f} ms".format(mediana))
print("Desviacion: {:.4f} ms".format(desviacion))
print("P95: {:.4f} ms".format(p95))
print("Throughput: {:.4f} imagenes/s".format(throughput))
print("Prediccion:", output.argmax(dim=1).item())


# Guardar resultados
nombre = (
    "cudnn_benchmark_{}.csv".format(args.benchmark)
)

with open(nombre, "w", newline="") as archivo:

    writer = csv.writer(archivo)

    writer.writerow([
        "iteracion",
        "latencia_ms"
    ])

    for i, latencia in enumerate(latencias):

        writer.writerow([
            i + 1,
            latencia
        ])


# Guardar logits para comparar resultados
torch.save(
    output.detach().cpu(),
    "cudnn_output_{}.pt".format(args.benchmark)
)

print("\nArchivos guardados:")
print(nombre)
print("cudnn_output_{}.pt".format(args.benchmark))
