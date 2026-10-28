# MobileNetV2 en NVIDIA Jetson Nano: implementación, perfilado y optimización con CUDA

## Objetivo

Evaluar la inferencia de **MobileNetV2** en la GPU de una **NVIDIA Jetson Nano** y comparar cuatro configuraciones: (1) PyTorch + CUDA, (2) PyTorch + CUDA con selección de algoritmos cuDNN, (3) fusión Conv2D–BatchNorm2D junto con cuDNN y (4) ejecución con TensorRT. Se utiliza la misma arquitectura y los pesos preentrenados **ImageNet-1K V1**.

## Plataforma y metodología

| Elemento | Configuración |
|---|---|
| Equipo | NVIDIA Jetson Nano DevKit (Tegra X1, GPU Maxwell de 128 núcleos CUDA) |
| Sistema | Ubuntu 18.04.6, JetPack/L4T R32.7.1, ARM64 |
| Python | 3.6.9 |
| PyTorch / torchvision | 1.10.0 / 0.11.1 |
| CUDA / cuDNN | 10.2 / 8.2.1 |
| ONNX / TensorRT | 1.11.0 / 8.2.1.8 |
| Modelo | `torchvision.models.mobilenet_v2(pretrained=True)`; modo `eval()` |
| Entrada | Imagen `test_images/imagen.jpeg`, RGB, batch de 1, FP32 |
| Preprocesamiento | `Resize(256)`, `CenterCrop(224)`, `ToTensor()`, normalización ImageNet |
| Forma de entrada / salida | `(1, 3, 224, 224)` / `(1, 1000)` |
| Muestras | 8 calentamientos y 120 inferencias medidas por configuración |
| Métrica | Latencia media, mediana, desviación estándar, P95 y throughput |

Las mediciones de inferencia utilizan una entrada ya procesada y residente en GPU; **no incluyen** la lectura del JPEG, el preprocesamiento, la carga de pesos ni la construcción del motor. Se sincroniza CUDA para que los tiempos incluyan el trabajo GPU iniciado por cada inferencia. El throughput indicado se calcula como `1000 / latencia_media_ms` para batch 1; no es una medición de varias solicitudes concurrentes.



## 1. PyTorch + CUDA: Baseline

Se carga MobileNetV2 preentrenada desde `torchvision`, se coloca el modelo y el tensor de entrada en `device='cuda'` y se ejecuta `model(input_tensor)` bajo `torch.no_grad()`. PyTorch delega las operaciones a implementaciones GPU (incluidas las de cuDNN cuando corresponden), sin reescribir manualmente las capas en CUDA. Esta versión no aplica transformaciones adicionales a la red.

**Script:** `profile_mobilenetv2_gpu.py`.

| Métrica | Resultado |
|---|---:|
| Latencia media | **26.3283 ms** |
| Mediana | 26.2100 ms |
| Desviación estándar | 0.6242 ms |
| P95 | 26.6730 ms |
| Throughput | **37.9820 imágenes/s** |

**Perfil de operaciones.** Se realizó un perfil separado con `profile_torch_gpu.py`, mediante `torch.autograd.profiler` (120 ejecuciones). Estos son **tiempos acumulados instrumentados**, útiles para localizar cuellos de botella, pero **no** son fracciones de la latencia real de 26.3283 ms, ya que el perfilado añade una sobrecarga:

| Grupo de operaciones | Llamadas por inferencia | Tiempo medio acumulado bajo perfilado |
|---|---:|---:|
| Pointwise Conv2D (1×1) | 34 | 19.9106 ms |
| BatchNorm2D | 52 | 19.3681 ms |
| Depthwise Conv2D | 17 | 8.3157 ms |
| ReLU6 | 35 | 4.3949 ms |
| Add | 10 | 1.0726 ms |
| Conv2D convencional | 1 | 0.8329 ms |
| Linear | 1 | 0.5793 ms |
| GlobalAvgPool2D | 1 | 0.1938 ms |

Pointwise y BatchNorm sobresalen en ese perfil.
**Archivos de perfilado:** `resumen_operaciones_gpu.csv`, `detalle_operaciones_gpu.csv` y `perfilado_pytorch_gpu.txt`.

## 2. PyTorch + CUDA con `cuDNN benchmark`

Con `torch.backends.cudnn.benchmark = True`, cuDNN puede evaluar y seleccionar algoritmos de convolución más adecuados para las dimensiones observadas. Es útil cuando la forma de entrada permanece fija, como aquí `(1, 3, 224, 224)`. **No cambia la arquitectura ni fusiona BatchNorm por sí solo.** Puede introducir tiempo adicional en las primeras ejecuciones, por lo que se utiliza calentamiento.

**Script:** `benchmark_cudnn.py`. Se comparó la misma red con la opción desactivada y activada.

| Métrica | `benchmark=False` | `benchmark=True` |
|---|---:|---:|
| Latencia media | 28.5077 ms | **26.4305 ms** |
| Mediana | 26.9952 ms | 25.4597 ms |
| Desviación estándar | 2.9497 ms | 2.3666 ms |
| P95 | 31.9882 ms | 30.4344 ms |
| Throughput | 35.0782 imágenes/s | **37.8351 imágenes/s** |

En **este ensayo específico**, activar la búsqueda de algoritmos redujo la media **7.29 %** respecto a `benchmark=False`. No obstante, la ejecución baseline independiente registró 26.3283 ms, ligeramente menos que los 26.4305 ms de esta prueba con `benchmark=True`; por eso no se debe atribuir una mejora universal únicamente a este ajuste. La comparación de logits entre las dos opciones dio **RMSE = 0**, error máximo = 0 y predicción de clase **239**.

## 3. PyTorch + cuDNN + fusión Conv2D–BatchNorm2D

Durante inferencia, una capa BatchNorm con parámetros fijos puede incorporarse a los pesos y sesgos de la convolución precedente. Se aplica `torch.nn.utils.fusion.fuse_conv_bn_eval` a **52 pares** `Conv2D + BatchNorm2D`, reemplazando cada BatchNorm fusionada por `nn.Identity()`. Así se evita ejecutarla como operación independiente. El modelo se mantiene en `eval()` y `torch.backends.cudnn.benchmark=True`.

**Scripts:** `benchmark_fusion.py` (medición) y `profile_mobilenetv2_optimized.py` (perfilado).

| Métrica | Sin fusión, misma prueba | Con 52 fusiones |
|---|---:|---:|
| Latencia media | 25.8027 ms | **23.1743 ms** |
| Mediana | 25.1648 ms | 23.1684 ms |
| Desviación estándar | 1.6615 ms | 0.0485 ms |
| P95 | 29.8139 ms | 23.2259 ms |
| Throughput | 38.7557 imágenes/s | **43.1512 imágenes/s** |

La reducción de latencia en esta comparación controlada fue de **10.19 %** (aceleración **1.1134×**). Se confirmó que quedaban **0 módulos BatchNorm** y **0 llamadas al operador `batch_norm`**. Frente a la red no fusionada de la misma prueba: **RMSE = 4.0305 × 10⁻⁶**, error máximo = **1.4305 × 10⁻⁵** y la predicción siguió siendo **239**.

**Costo de operaciones después de la fusión.** El perfilado separado mediante `profile_mobilenetv2_optimized.py` (120 inferencias instrumentadas) produjo los siguientes tiempos medios acumulados por grupo:

| Grupo de operaciones | Llamadas por inferencia | Tiempo medio acumulado bajo perfilado |
|---|---:|---:|
| Pointwise Conv2D (1×1) | 34 | 18.8598 ms |
| BatchNorm2D | **0** | **0 ms** |
| Depthwise Conv2D | 17 | 8.2603 ms |
| ReLU6 | 35 | 3.8693 ms |
| Add | 10 | 0.9706 ms |
| Conv2D convencional | 1 | 0.8323 ms |
| Linear | 1 | 0.5850 ms |
| GlobalAvgPool2D | 1 | 0.1893 ms |

**Comparación de costos por operación (perfiles instrumentados):**

| Grupo | Sin fusión | Con fusión | Variación observada |
|---|---:|---:|---:|
| Pointwise Conv2D | 19.9106 ms | 18.8598 ms | −1.0508 ms |
| BatchNorm2D | 19.3681 ms | 0 ms | Eliminada como operación independiente |
| Depthwise Conv2D | 8.3157 ms | 8.2603 ms | −0.0554 ms |
| ReLU6 | 4.3949 ms | 3.8693 ms | −0.5256 ms |
| Add | 1.0726 ms | 0.9706 ms | −0.1020 ms |
| Conv2D convencional | 0.8329 ms | 0.8323 ms | −0.0006 ms |
| Linear | 0.5793 ms | 0.5850 ms | +0.0057 ms |
| GlobalAvgPool2D | 0.1938 ms | 0.1893 ms | −0.0045 ms |

El perfilado permitió identificar que las operaciones Pointwise Conv2D y BatchNorm2D concentraban una parte importante del tiempo registrado durante la ejecución instrumentada de MobileNetV2. Sin embargo, la instrumentación introduce sobrecarga, por lo que estos tiempos no representan directamente la latencia de inferencia medida sin perfilado.

Para reducir este costo, se aplicó la fusión de 52 pares Conv2D–BatchNorm2D, incorporando los parámetros de normalización a los pesos y sesgos de las convoluciones. Esto permitió eliminar la ejecución independiente de las 52 operaciones BatchNorm2D durante la inferencia, conservando su efecto matemático.

La efectividad de esta optimización se verificó mediante un benchmark sin perfilado por operaciones, donde la latencia media disminuyó de **25.8027 ms a 23.1743 ms**, equivalente a una **reducción del 10.19 %**.


**Archivos de perfilado:** `resumen_operaciones_gpu_optimized.csv`, `detalle_operaciones_gpu_optimized.csv` y `perfilado_pytorch_gpu_optimized.txt`.

## 4. PyTorch → ONNX → TensorRT (FP32)

Se parte de la misma MobileNetV2 preentrenada, se fusionan los **52 pares Conv2D–BatchNorm2D**, se exporta el grafo a **ONNX opset 11** y se genera en la propia Jetson un motor **TensorRT 8.2.1.8** con entrada y salida FP32. TensorRT optimiza el grafo para el hardware de destino mediante selección de implementaciones y posibles fusiones internas. El motor se ejecuta usando tensores CUDA de PyTorch y la API `execute_v2()` de TensorRT 8.x: **no fue necesario instalar PyCUDA**.

**Scripts:** `export_mobilenetv2_onnx.py` y `benchmark_tensorrt.py`.

**Exportación y creación del motor:**

```bash
cd ~/grupo-5/proyecto
export OPENBLAS_CORETYPE=ARMV8
python3 export_mobilenetv2_onnx.py

/usr/src/tensorrt/bin/trtexec \
  --onnx=mobilenetv2_fusion.onnx \
  --saveEngine=mobilenetv2_fp32.engine \
  --workspace=512 \
  --buildOnly
```

El archivo ONNX superó `onnx.checker.check_model()`. Sus nodos fueron: **52 Conv, 35 Clip (ReLU6), 10 Add, 1 GlobalAveragePool, 1 Flatten, 1 Gemm, 70 Constant y 0 BatchNormalization**. El motor generado ocupa aproximadamente **14 MB** y expone los bindings `input: (1,3,224,224) FLOAT` y `output: (1,1000) FLOAT`.

**Validación y benchmark:**

```bash
export OPENBLAS_CORETYPE=ARMV8
python3 benchmark_tensorrt.py
```

| Métrica | TensorRT |
|---|---:|
| Latencia media | **13.5285 ms** |
| Mediana | 13.4985 ms |
| Desviación estándar | 0.1079 ms |
| P95 | 13.7942 ms |
| Throughput | **73.9178 imágenes/s** |
| Calentamientos / mediciones | 8 / 120 |

La salida se comparó con PyTorch fusionado para la misma imagen:

| Validación | Resultado |
|---|---:|
| Clase PyTorch / TensorRT | **239 / 239** |
| RMSE de 1000 logits | **2.9589 × 10⁻⁶** |
| MAE | 2.3853 × 10⁻⁶ |
| Error máximo absoluto | 1.0014 × 10⁻⁵ |

La coincidencia de clase y los errores bajos verifican este **caso de prueba**, pero no constituyen una validación de exactitud sobre un conjunto completo de imágenes. Tampoco garantizan que TensorRT haya ejecutado todas las operaciones internamente en FP32; la entrada/salida y la configuración solicitada fueron FP32 y no se habilitó `--fp16` ni `--int8`.

**Resultados guardados:** `resultados_tensorrt/inferencias.csv` y `resultados_tensorrt/resumen.csv`.

### 4.1. Perfilado por capas del motor TensorRT

Para medir el costo de los grupos de operaciones **realmente ejecutados por TensorRT**, se perfiló el motor ya generado mediante `trtexec` (120 iteraciones), sin reconstruirlo:

```bash
mkdir -p resultados_tensorrt/perfilado
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=mobilenetv2_fp32.engine \
  --dumpProfile \
  --exportProfile=resultados_tensorrt/perfilado/perfil_capas.json \
  --separateProfileRun \
  --warmUp=0 --duration=0 --iterations=120 \
  2>&1 | tee resultados_tensorrt/perfilado/perfilado.log
```

El JSON exportado contiene un encabezado `{"count": 120}` seguido de **55 grupos perfilados**. El tiempo medio acumulado de esos grupos fue **13.3596 ms por inferencia**; es una medición distinta de los **13.5285 ms** del benchmark sin perfilado. En particular, `trtexec` emplea entradas sintéticas, mientras que `benchmark_tensorrt.py` usa la imagen de prueba.

TensorRT fusionó operaciones que PyTorch ejecutaba por separado, por ejemplo `Conv_13 + PWN(Clip_16)` (convolución + activación ReLU6) y `Conv_26 + Add_27` (convolución + suma residual). Por eso **no es posible separar el tiempo de ReLU6 o Add** del tiempo del grupo fusionado a partir de este perfil.

**Costo agregado de los grupos TensorRT, clasificados por convolución de origen:**

| Grupo atribuido | Cantidad de grupos | Tiempo acumulado medio | Fracción del tiempo perfilado |
|---|---:|---:|---:|
| Pointwise Conv2D (1×1), incluidas las fusiones | 34 | **7.1504 ms** | **53.52 %** |
| Depthwise Conv2D (3×3), incluidas las fusiones | 17 | **5.3542 ms** | **40.08 %** |
| Conv2D convencional (3×3), incluida la fusión | 1 | 0.4154 ms | 3.11 % |
| Linear (`Gemm`) | 1 | 0.3941 ms | 2.95 % |
| GlobalAvgPool2D | 1 | 0.0449 ms | 0.34 % |
| Shuffle | 1 | 0.0006 ms | <0.01 % |
| **Total** | **55** | **13.3596 ms** | **≈100 %** |

Pointwise y Depthwise, junto con sus activaciones o sumas fusionadas, representan **93.60 %** del tiempo de capas perfilado. El porcentaje **no** es una fracción demostrada de los 13.5285 ms de latencia total medidos con Python.

**Grupos individuales más costosos:**

| Grupo del motor | Tiempo medio | Proporción del perfil |
|---|---:|---:|
| `Conv_13 + PWN(Clip_16)` | 0.8625 ms | 6.46 % |
| `Conv_9 + PWN(Clip_12)` | 0.7083 ms | 5.30 % |
| `Conv_163 + PWN(Clip_166)` | 0.5821 ms | 4.36 % |
| `Conv_22 + PWN(Clip_25)` | 0.5220 ms | 3.91 % |
| `Conv_4 + PWN(Clip_7)` | 0.5003 ms | 3.74 % |
| `Conv_0 + PWN(Clip_3)` | 0.4154 ms | 3.11 % |

La clasificación Pointwise/Depthwise se obtuvo con `analizar_perfil_tensorrt.py` al leer los atributos `kernel_shape` y `group` del grafo ONNX y relacionar `Conv_N` del perfil con el índice del nodo ONNX. **Es una correspondencia inferida por nombres:** las cantidades finales (34 + 17 + 1) concuerdan con la arquitectura, pero conviene verificar los identificadores mediante información detallada del motor antes de presentarla como correspondencia garantizada. Los tiempos asignados a Conv incluyen las activaciones `Clip` y las sumas `Add` fusionadas cuando corresponde.

**Archivos del perfilado:** `resultados_tensorrt/perfilado/perfilado.log`, `perfil_capas.json`, `resumen_capas.csv` y `resumen_operaciones.csv`. El primero registra la ejecución; los CSV ordenan las 55 capas por costo y las agrupan por tipo, respectivamente.

### 4.2. Contraste de los perfiles de operaciones

| Operación/grupo | PyTorch CUDA sin fusión¹ | PyTorch Conv–BN fusionadas¹ | TensorRT² |
|---|---:|---:|---:|
| Pointwise Conv2D | 19.9106 ms | 18.8598 ms | **7.1504 ms** (incluye fusiones) |
| Depthwise Conv2D | 8.3157 ms | 8.2603 ms | **5.3542 ms** (incluye fusiones) |
| Conv2D convencional | 0.8329 ms | 0.8323 ms | 0.4154 ms (incluye fusión) |
| BatchNorm2D independiente | 19.3681 ms | 0 ms | 0 ms (incorporada previamente a Conv) |
| ReLU6 independiente | 4.3949 ms | 3.8693 ms | No separable de Conv fusionada |
| Add independiente | 1.0726 ms | 0.9706 ms | No separable de Conv fusionada |
| Linear | 0.5793 ms | 0.5850 ms | 0.3941 ms |
| GlobalAvgPool2D | 0.1938 ms | 0.1893 ms | 0.0449 ms |

¹ `torch.autograd.profiler` sobre PyTorch, con una sobrecarga considerable. ² Perfil por grupos del motor de TensorRT mediante `trtexec`; utiliza una instrumentación y agrupación distintas. **No calcular reducciones porcentuales por fila entre PyTorch y TensorRT** como si los valores fueran equivalentes: las fusiones y el perfilado cambian qué trabajo se asigna a cada fila. Para medir la ganancia global, usar las latencias sin perfilado de la sección 5.

## 5. Resumen 

| Implementación | Latencia media | Throughput | Reducción frente al baseline |
|---|---:|---:|---:|
| PyTorch + CUDA (baseline) | **26.3283 ms** | 37.9820 img/s | — |
| PyTorch + CUDA + `cuDNN benchmark=True` | **26.4305 ms** | 37.8351 img/s | −0.39 % |
| PyTorch + cuDNN + 52 fusiones Conv–BN | **23.1743 ms** | 43.1512 img/s | 11.98 % |
| PyTorch → ONNX → TensorRT | **13.5285 ms** | 73.9178 img/s | **48.62 %** |

**Lectura de la tabla:** el ensayo con cuDNN compara directamente `benchmark=False` (28.5077 ms) contra `benchmark=True` (26.4305 ms): mejora **7.29 %** dentro de ese ensayo. La cifra de **−0.39 %** de la tabla surge de compararlo con otro baseline independiente de 26.3283 ms y **no prueba un empeoramiento causado por cuDNN**. La fusión mejora **10.19 %** respecto al modelo sin fusionar medido en su **misma ejecución** (25.8027 → 23.1743 ms). TensorRT mejora **41.62 %** respecto a la configuración fusionada y es **1.95×** más rápido que el baseline en la comparación entre sesiones.


## 6. Reproducción de los experimentos

Esta sección describe el entorno de ejecución, las dependencias necesarias y el procedimiento utilizado para reproducir las cuatro implementaciones de MobileNetV2 sobre la NVIDIA Jetson Nano.

### 6.1. Plataforma de ejecución

Los experimentos se realizaron sobre una NVIDIA Jetson Nano Developer Kit, utilizando su GPU Maxwell de 128 núcleos CUDA.

| Componente | Especificación |
|---|---|
| Plataforma | NVIDIA Jetson Nano |
| CPU | ARM Cortex-A57, 4 núcleos |
| GPU | NVIDIA Maxwell, 128 núcleos CUDA |
| Arquitectura | ARM64 (aarch64) |
| Sistema operativo | Ubuntu 18.04.6 LTS |
| JetPack / L4T | L4T R32.7.1 |
| Python | 3.6.9 |
| CUDA | 10.2 |
| PyTorch | 1.10.0 |
| torchvision | 0.11.1 |
| cuDNN | 8.2.1 |
| TensorRT | 8.2.1.8 |
| ONNX | 1.11.0 |
| pybind11 | 2.9.1 |
| Protobuf C++ | 3.16.0 |

Para verificar el hardware:

```bash
uname -a
cat /etc/nv_tegra_release
lscpu
nvcc --version
```

Para verificar PyTorch:

```bash
python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.cuda.is_available())"
```

La última instrucción debe devolver `True`.

### 6.2. Dependencias

La implementación requiere:

- PyTorch con soporte CUDA.
- torchvision.
- NumPy.
- Pillow.
- TensorRT.
- ONNX, para reconstruir el modelo exportado.

Los paquetes PyTorch y torchvision deben ser compatibles con la versión de JetPack y la arquitectura ARM64 de la Jetson Nano.

Para comprobar las dependencias instaladas:

```bash
python3 -c "import torch; print(torch.__version__)"
python3 -c "import torchvision; print(torchvision.__version__)"
python3 -c "import numpy; print(numpy.__version__)"
python3 -c "import tensorrt; print(tensorrt.__version__)"
python3 -c "import onnx; print(onnx.__version__)"
```

En la plataforma utilizada fue necesario configurar:

```bash
export OPENBLAS_CORETYPE=ARMV8
```

Esta variable evita el error `Illegal instruction` observado al importar NumPy en el entorno experimental.

### 6.3. Organización del proyecto

```text
entrega_gpu_mobilenetv2/
├── Makefile
├── README.md
│
├── scripts/
│   ├── profile_mobilenetv2_gpu.py
│   ├── profile_torch_gpu.py
│   ├── test_overhead_gpu.py
│   ├── benchmark_cudnn.py
│   ├── benchmark_fusion.py
│   ├── profile_mobilenetv2_optimized.py
│   ├── export_mobilenetv2_onnx.py
│   ├── benchmark_tensorrt.py
│   └── analizar_perfil_tensorrt.py
│
├── modelos/
│   ├── mobilenetv2_fusion.onnx
│   └── mobilenetv2_fp32.engine
│
├── test_images/
│   └── imagen.jpeg
│
└── resultados/
    ├── baseline/
    ├── cudnn/
    ├── fusion/
    └── tensorrt/
        └── perfilado/
```

Los scripts contienen la implementación y los procedimientos experimentales. Los archivos CSV y JSON conservan los resultados obtenidos.

### 6.4. Metodología experimental

Las cuatro implementaciones utilizan MobileNetV2 con pesos preentrenados de ImageNet-1K V1.

Se utiliza la misma imagen y el mismo preprocesamiento:

1. Redimensionamiento a 256 píxeles.
2. Recorte central de 224 × 224 píxeles.
3. Conversión a tensor.
4. Normalización mediante las medias y desviaciones estándar de ImageNet.

La entrada del modelo tiene dimensiones:

```text
(1, 3, 224, 224)
```

La salida corresponde a 1000 clases:

```text
(1, 1000)
```

Para las mediciones de latencia se realizaron:

| Parámetro | Configuración |
|---|---|
| Batch size | 1 |
| Calentamientos | 8 |
| Inferencias medidas | 120 |
| Precisión | FP32 |
| Imagen | `test_images/imagen.jpeg` |

Las mediciones excluyen el tiempo de carga del modelo y el preprocesamiento inicial de la imagen.

### 6.5. PyTorch + CUDA

La primera implementación utiliza MobileNetV2 directamente desde torchvision y ejecuta sus operaciones sobre la GPU mediante CUDA.

Para realizar la prueba:

```bash
python3 scripts/profile_mobilenetv2_gpu.py
```

Para obtener el perfilado por operaciones:

```bash
python3 scripts/profile_torch_gpu.py
```

El perfilado permite identificar las operaciones con mayor costo computacional, especialmente Pointwise Conv2D y BatchNorm2D.

**Resultado:** latencia media de 26.3283 ms y throughput de 37.9820 imágenes/s.

### 6.6. PyTorch + cuDNN Benchmark

La segunda implementación estudia el efecto de la selección automática de algoritmos mediante:

```python
torch.backends.cudnn.benchmark = True
```

Esta configuración permite que cuDNN evalúe diferentes implementaciones y seleccione algoritmos adecuados para las dimensiones de los tensores.

Para ejecutar la comparación:

```bash
python3 scripts/benchmark_cudnn.py
```


### 6.7. Fusión Conv2D + BatchNorm2D

La tercera implementación incorpora la fusión de las operaciones Conv2D y BatchNorm2D durante la inferencia.

La fusión se realiza mediante:

```python
fuse_conv_bn_eval(conv, bn)
```

Se fusionaron 52 pares Conv2D–BatchNorm2D, eliminando la ejecución independiente de BatchNorm2D y conservando su efecto matemático mediante los parámetros de las convoluciones.

Para realizar el benchmark:

```bash
python3 scripts/benchmark_fusion.py
```

Para perfilar la implementación optimizada:

```bash
python3 scripts/profile_mobilenetv2_optimized.py
```


### 6.8. Exportación a ONNX

Para utilizar TensorRT, la red fusionada se exportó a ONNX.

El modelo se exportó utilizando ONNX opset 11, con una entrada de dimensiones `(1, 3, 224, 224)`.

Para generar el archivo:

```bash
python3 scripts/export_mobilenetv2_onnx.py
```

El archivo generado es:

```text
mobilenetv2_fusion.onnx
```

Durante la exportación se verificó que las 52 operaciones BatchNorm2D quedaran fusionadas.

### 6.9. Construcción del motor TensorRT

Para construir el motor TensorRT en la Jetson Nano:

```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=modelos/mobilenetv2_fusion.onnx \
    --saveEngine=modelos/mobilenetv2_fp32.engine \
    --workspace=512 \
    --buildOnly
```

Este procedimiento construye un motor optimizado para la plataforma de destino.

El motor debe reconstruirse cuando cambian las características relevantes del dispositivo o del entorno TensorRT.

### 6.10. Benchmark TensorRT

El motor se ejecuta mediante la API de TensorRT 8.2.

Para realizar las mediciones:

```bash
python3 scripts/benchmark_tensorrt.py
```

Se realizan 8 calentamientos y 120 inferencias.



### 6.11. Perfilado de TensorRT

Para identificar el costo de las capas ejecutadas por TensorRT:

```bash
mkdir -p resultados/tensorrt/perfilado
```

Ejecutar:

```bash
/usr/src/tensorrt/bin/trtexec \
    --loadEngine=modelos/mobilenetv2_fp32.engine \
    --dumpProfile \
    --exportProfile=resultados/tensorrt/perfilado/perfil_capas.json \
    --separateProfileRun \
    --warmUp=0 \
    --duration=0 \
    --iterations=120
```

Para clasificar las convoluciones:

```bash
python3 scripts/analizar_perfil_tensorrt.py
```

El perfilado obtuvo 55 grupos de operaciones y un tiempo acumulado de 13.3596 ms.

Los resultados indican que Pointwise Conv2D y Depthwise Conv2D concentran el 93.60 % del tiempo perfilado.

Estos tiempos incluyen las operaciones fusionadas por TensorRT, como ReLU6 y sumas residuales.

### 6.12. Automatización mediante Makefile

Se incorporó un Makefile con objetivos para preparar el entorno y ejecutar los experimentos.

Para consultar las opciones:

```bash
make help
```

Para verificar las dependencias:

```bash
make check
```

Para ejecutar los benchmarks:

```bash
make baseline
make cudnn
make fusion
make tensorrt
```

Para realizar el perfilado:

```bash
make profile-baseline
make profile-fusion
make profile-tensorrt
make analyze-tensorrt
```

Para reconstruir el motor:

```bash
make export
make engine
```

**Nota:** antes de utilizar estos objetivos, deben verificarse las rutas de los archivos en los scripts originales, ya que fueron desarrollados antes de reorganizar el proyecto en las carpetas `scripts/`, `modelos/` y `resultados/`.

