import os
import csv
import time
import statistics

import numpy as np
import torch
import tensorrt as trt

from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v2
from torch.nn.utils.fusion import fuse_conv_bn_eval



# Configuración
ENGINE_PATH = "mobilenetv2_fp32.engine"
IMAGE_PATH = "test_images/imagen.jpeg"

WARMUPS = 8
REPEATS = 120

torch.backends.cudnn.benchmark = True

device = torch.device("cuda")

os.makedirs("resultados_tensorrt", exist_ok=True)



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

image = Image.open(IMAGE_PATH).convert("RGB")

input_tensor = preprocess(image).unsqueeze(0)
input_tensor = input_tensor.to(device).contiguous()

print("Entrada:", tuple(input_tensor.shape))


# PyTorch: modelo de referencia fusionado
model = mobilenet_v2(pretrained=True)
model.eval()


def fusionar_modelo(model):
    fusiones = 0

    for module in model.modules():

        if not isinstance(module, torch.nn.Sequential):
            continue

        hijos = list(module.named_children())

        for i in range(len(hijos) - 1):

            nombre_conv, conv = hijos[i]
            nombre_bn, bn = hijos[i + 1]

            if isinstance(conv, torch.nn.Conv2d) and \
               isinstance(bn, torch.nn.BatchNorm2d):

                module._modules[nombre_conv] = fuse_conv_bn_eval(
                    conv, bn
                )

                module._modules[nombre_bn] = torch.nn.Identity()

                fusiones += 1

    return fusiones


fusiones = fusionar_modelo(model)

if fusiones != 52:
    raise RuntimeError("Número incorrecto de fusiones")

model = model.to(device)

with torch.no_grad():
    output_pytorch = model(input_tensor)

torch.cuda.synchronize()

print("Fusiones:", fusiones)
print("Predicción PyTorch:", output_pytorch.argmax(dim=1).item())



# Cargar TensorRT
logger = trt.Logger(trt.Logger.WARNING)

runtime = trt.Runtime(logger)

with open(ENGINE_PATH, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

if engine is None:
    raise RuntimeError("No se pudo cargar TensorRT")

context = engine.create_execution_context()

if context is None:
    raise RuntimeError("No se pudo crear el contexto TensorRT")



# Buffers GPU
output_tensor = torch.empty(
    (1, 1000),
    device=device,
    dtype=torch.float32
)

bindings = [0] * engine.num_bindings

input_index = engine.get_binding_index("input")
output_index = engine.get_binding_index("output")

bindings[input_index] = input_tensor.data_ptr()
bindings[output_index] = output_tensor.data_ptr()



# Ejecución TensorRT
def inferencia_tensorrt():

    success = context.execute_v2(bindings)

    if not success:
        raise RuntimeError("Error durante inferencia TensorRT")



# Validación numérica
inferencia_tensorrt()
torch.cuda.synchronize()

ref = output_pytorch.detach().cpu().numpy()
test = output_tensor.detach().cpu().numpy()

rmse = float(np.sqrt(np.mean((ref - test) ** 2)))
mae = float(np.mean(np.abs(ref - test)))
max_error = float(np.max(np.abs(ref - test)))

prediction_trt = int(np.argmax(test))
prediction_torch = int(np.argmax(ref))

print("\n========== VALIDACIÓN ==========")

print("Predicción PyTorch:", prediction_torch)
print("Predicción TensorRT:", prediction_trt)
print("Coinciden:", prediction_torch == prediction_trt)

print("RMSE:", rmse)
print("MAE:", mae)
print("Error máximo:", max_error)

if not np.isfinite(rmse):
    raise RuntimeError("Error numérico no finito")

if prediction_torch != prediction_trt:
    raise RuntimeError("Las predicciones no coinciden")



# Calentamiento
print("\nCalentamientos:", WARMUPS)

for i in range(WARMUPS):
    inferencia_tensorrt()

torch.cuda.synchronize()



# Benchmark
print("Inferencias medidas:", REPEATS)

latencias = []

for i in range(REPEATS):

    torch.cuda.synchronize()

    inicio = time.perf_counter()

    inferencia_tensorrt()

    torch.cuda.synchronize()

    fin = time.perf_counter()

    latencia_ms = (fin - inicio) * 1000.0

    latencias.append(latencia_ms)



# Estadísticas
mean_ms = statistics.mean(latencias)
median_ms = statistics.median(latencias)
std_ms = statistics.pstdev(latencias)
p95_ms = float(np.percentile(latencias, 95))

throughput = 1000.0 / mean_ms

print("\n========== TENSORRT ==========")

print("Latencia media: {:.4f} ms".format(mean_ms))
print("Mediana: {:.4f} ms".format(median_ms))
print("Desviación estándar: {:.4f} ms".format(std_ms))
print("P95: {:.4f} ms".format(p95_ms))
print("Throughput: {:.4f} imágenes/s".format(throughput))



# Comparación
baseline_ms = 26.3283
optimized_ms = 23.1743

mejora_baseline = (
    (baseline_ms - mean_ms) / baseline_ms
) * 100.0

mejora_optimized = (
    (optimized_ms - mean_ms) / optimized_ms
) * 100.0

print("\n========== COMPARACIÓN ==========")

print("PyTorch baseline: {:.4f} ms".format(baseline_ms))
print("PyTorch optimizado: {:.4f} ms".format(optimized_ms))
print("TensorRT: {:.4f} ms".format(mean_ms))

print("Mejora vs baseline: {:.2f}%".format(mejora_baseline))
print("Mejora vs optimizado: {:.2f}%".format(mejora_optimized))



# Guardar CSV
csv_path = "resultados_tensorrt/inferencias.csv"

with open(csv_path, "w") as f:

    writer = csv.writer(f)

    writer.writerow(["iteracion", "latencia_ms"])

    for i, latencia in enumerate(latencias):
        writer.writerow([i + 1, latencia])


summary_path = "resultados_tensorrt/resumen.csv"

with open(summary_path, "w") as f:

    writer = csv.writer(f)

    writer.writerow([
        "latencia_media_ms",
        "mediana_ms",
        "std_ms",
        "p95_ms",
        "throughput_imagenes_s",
        "rmse",
        "mae",
        "error_maximo",
        "prediccion_pytorch",
        "prediccion_tensorrt"
    ])

    writer.writerow([
        mean_ms,
        median_ms,
        std_ms,
        p95_ms,
        throughput,
        rmse,
        mae,
        max_error,
        prediction_torch,
        prediction_trt
    ])

print("\nResultados guardados en:", summary_path)
