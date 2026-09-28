# Perfilado de MobileNetV2 C++

## 1. Entrar al proyecto

Desde una terminal, ubicarse en la raíz del proyecto:

```bash
cd MobileNetV2
```

Activar el entorno virtual:

```bash
source venv/bin/activate
```

---

## 2. Compilar con el perfilador habilitado

Antes de realizar el perfilado se debe recompilar el módulo C++:

```bash
make clean
make profile-build
```

`profile-build` habilita el perfilado interno de los kernels C++.

---

## 3. Validar la implementación

Antes de realizar las mediciones se recomienda comprobar que MobileNetV2 C++
continúa generando resultados equivalentes a PyTorch:

```bash
PYTHONPATH="$PWD" python app/run_cpp_mobilenetv2.py \
    test_images/imagen.jpeg
```

La ejecución debe finalizar con:

```text
RESULTADO: PASS
```

Si el resultado es `FAIL`, no se debe utilizar esa versión para realizar el
perfilado hasta revisar la implementación.

---

## 4. Ejecutar el perfilado

El perfilado debe realizarse con las mismas condiciones en todas las máquinas:

- 120 iteraciones medidas.
- 8 iteraciones de warm-up.
- 1 thread.
- Preprocesamiento `official`.
- Perfilado interno de kernels habilitado.

Ejecutar:

```bash
PYTHONPATH="$PWD" python app/profile_complete.py \
    --images test_images/imagen.jpeg \
    --iterations 120 \
    --warmup 8 \
    --threads 1 \
    --output-dir resultados/NOMBRE_RESULTADO \
    --preprocess official \
    --kernel-profile on
```

Cambiar `NOMBRE_RESULTADO` según la máquina.

### CPU-1 — Milagro

```bash
PYTHONPATH="$PWD" python app/profile_complete.py \
    --images test_images/imagen.jpeg \
    --iterations 120 \
    --warmup 8 \
    --threads 1 \
    --output-dir resultados/milagro_cpu \
    --preprocess official \
    --kernel-profile on
```

### CPU-2 — Angie

```bash
PYTHONPATH="$PWD" python app/profile_complete.py \
    --images test_images/imagen.jpeg \
    --iterations 120 \
    --warmup 8 \
    --threads 1 \
    --output-dir resultados/angie_cpu \
    --preprocess official \
    --kernel-profile on
```

### Kria

```bash
PYTHONPATH="$PWD" python app/profile_complete.py \
    --images test_images/imagen.jpeg \
    --iterations 120 \
    --warmup 8 \
    --threads 1 \
    --output-dir resultados/kria_cpu \
    --preprocess official \
    --kernel-profile on
```

---

## 5. Resultados generados

Al terminar el perfilado se crea automáticamente el directorio indicado con
`--output-dir`.

Por ejemplo:

```text
resultados/milagro_cpu/
```

Dentro se generan:

```text
cpuinfo.txt
etapas.csv
inicializacion.json
kernels.csv
meminfo.txt
metadata.json
predicciones.csv
resumen_etapas.csv
resumen_kernels.csv
validacion.json
```

Los archivos principales para analizar los resultados son:

### `resumen_etapas.csv`

Contiene el resumen del tiempo de:

```text
lectura/decodificación
preprocesamiento
inferencia C++
posprocesamiento
flujo total
```

Incluye estadísticas como media, mediana, desviación estándar, p95, mínimo y
máximo.

### `resumen_kernels.csv`

Contiene el resumen del tiempo utilizado por cada operación interna de
MobileNetV2, incluyendo:

```text
Pointwise Conv2D
Depthwise Conv2D
Conv2D
BatchNorm2D
ReLU6
LayerAdd
GlobalAvgPool2D
Linear
```

Este archivo permite identificar cuáles kernels consumen más tiempo.

### `validacion.json`

Guarda el resultado de la comparación entre PyTorch y C++.

### `metadata.json`

Guarda la configuración utilizada para realizar el experimento.

### `cpuinfo.txt`

Guarda información del CPU donde se realizó el perfilado.

---

## Importante

Para que los resultados entre CPU-1, CPU-2 y Kria sean comparables, no cambiar:

```text
Imagen:       test_images/imagen.jpeg
Iteraciones:  120
Warm-up:      8
Threads:      1
Preprocess:   official
Kernel profile: on
```


## Justificación de la optimización

La selección de los kernels a optimizar se realizó a partir de los resultados
obtenidos durante el perfilado de la implementación original de MobileNetV2.

El objetivo fue identificar las operaciones que representaban la mayor parte
del tiempo de inferencia y concentrar la optimización en ellas, en lugar de
modificar todas las operaciones de la red.

### Identificación del cuello de botella

En CPU-2 — Angie, el perfilado original obtuvo un tiempo medio de inferencia
C++ de aproximadamente:

**345.05 ms**

Al analizar individualmente las operaciones internas de MobileNetV2 se
obtuvieron los siguientes tiempos acumulados:

| Operación | Tiempo Baseline |
|---|---:|
| Pointwise Conv2D | 252.77 ms |
| Depthwise Conv2D | 34.46 ms |
| ReLU6 | 18.88 ms |
| Conv2D | 17.01 ms |
| BatchNorm2D | 11.28 ms |
| Linear | 1.52 ms |

El resultado más importante fue el tiempo correspondiente a
**Pointwise Conv2D**, que representó aproximadamente el **75 % del tiempo
interno registrado para los kernels**.
Además, las operaciones convolucionales Pointwise Conv2D, Depthwise Conv2D
y Conv2D concentraban en conjunto la mayor parte del costo computacional de
la implementación.
Por esta razón, estas tres operaciones fueron seleccionadas como los
principales objetivos de la primera etapa de optimización.

### Estrategia de Optimization V1

Para esta primera versión se decidió mantener:

- Un único thread.
- La misma arquitectura de MobileNetV2.
- Los mismos pesos.
- El mismo formato de los tensores.
- Las mismas operaciones matemáticas.
- La misma interfaz entre Python y C++.

No se incorporaron técnicas como OpenMP, ejecución multinúcleo o SIMD
explícito.
El objetivo de Optimization V1 fue determinar cuánto podía reducirse el
tiempo de ejecución únicamente mediante mejoras en la implementación C++.
Las modificaciones se concentraron principalmente en:

- mejorar el patrón de acceso a memoria;
- favorecer accesos contiguos cuando fuera posible;
- reducir cálculos repetitivos de índices;
- precalcular posiciones base utilizadas repetidamente dentro de los loops;
- reducir trabajo innecesario dentro de los loops más internos.

### Kernels seleccionados

#### Pointwise Conv2D

Fue seleccionado como la principal prioridad debido a que era el kernel con
mayor tiempo acumulado del perfilado.

La implementación original presentaba un patrón de acceso a los datos que no
favorecía el recorrido contiguo de memoria.

Optimization V1 reorganizó los loops para recorrer las posiciones espaciales
de forma contigua, buscando mejorar la localidad de memoria y reducir el
costo asociado al acceso a los datos.

#### Depthwise Conv2D

Depthwise Conv2D fue la segunda operación convolucional con mayor tiempo
acumulado.

En la implementación original se realizaban repetidamente cálculos de índices
dentro de los loops internos.

Optimization V1 precalculó diferentes posiciones base utilizadas durante la
convolución, reduciendo la cantidad de operaciones necesarias para determinar
las posiciones de entrada, salida y pesos.

#### Conv2D

La Conv2D inicial también presentó un tiempo relevante dentro del perfilado.

Se aplicó una estrategia similar a Depthwise Conv2D, precalculando índices y
posiciones base fuera de los loops más internos para evitar repetir estos
cálculos para cada operación de multiplicación y acumulación.

### Operaciones no modificadas

Operaciones como:

- ReLU6;
- BatchNorm2D;
- Linear;
- LayerAdd;
- GlobalAvgPool2D;

no fueron seleccionadas para Optimization V1.

Aunque algunas de ellas presentan un costo medible, su contribución al tiempo
total era considerablemente menor que la observada en las operaciones
convolucionales seleccionadas.

Esto permitió concentrar la primera etapa de optimización en los kernels donde
el perfilado indicaba un mayor potencial de reducción del tiempo total de
inferencia.


## Resultados de Optimization V1

Para evaluar el efecto de Optimization V1 se mantuvieron las mismas
condiciones utilizadas durante el perfilado inicial:

- 120 iteraciones medidas.
- 8 iteraciones de calentamiento.
- Ejecución con 1 hilo.
- Misma imagen de entrada.
- Mismos pesos de MobileNetV2.
- Compilación con `-O2`.
- Perfilado interno de los kernels habilitado.

Las modificaciones de Optimization V1 se aplicaron únicamente a los tres
kernels seleccionados a partir del perfilado inicial:

- Pointwise Conv2D.
- Depthwise Conv2D.
- Conv2D.

El objetivo fue reducir cálculos repetitivos de índices y mejorar el patrón
de acceso a memoria sin modificar la arquitectura de MobileNetV2, sus pesos
ni el resultado esperado de la inferencia.


### Métricas utilizadas

No se utilizó únicamente el tiempo promedio. Se consideraron varias métricas
para evaluar tanto el rendimiento general como la estabilidad de las
ejecuciones.

| Métrica | ¿Qué permite evaluar? |
|---|---|
| Inferencia C++ media | Tiempo promedio de ejecución de MobileNetV2 C++ |
| Inferencia mediana | Tiempo típico de ejecución, con menor influencia de valores extremos |
| Inferencia p95 | Tiempo por debajo del cual se encuentra el 95 % de las ejecuciones |
| Flujo total medio | Tiempo promedio del procesamiento completo, incluyendo lectura, preprocesamiento, inferencia y posprocesamiento |


### CPU-1 — Mila

> Pendiente completar con los resultados de Optimization V1 ejecutados en
> CPU-1 bajo las mismas condiciones experimentales.

| Métrica / Operación | Baseline | Optimization V1 | Cambio |
|---|---:|---:|---:|
| Inferencia C++ media | Pendiente | Pendiente | Pendiente |
| Inferencia mediana | Pendiente | Pendiente | Pendiente |
| Inferencia p95 | Pendiente | Pendiente | Pendiente |
| Flujo total medio | Pendiente | Pendiente | Pendiente |
| Pointwise Conv2D | ≈199.95 ms | Pendiente | Pendiente |
| Depthwise Conv2D | ≈28.67 ms | Pendiente | Pendiente |
| Conv2D | ≈15.91 ms | Pendiente | Pendiente |
| BatchNorm2D | ≈21.68 ms | Pendiente | Pendiente |
| ReLU6 | ≈14.86 ms | Pendiente | Pendiente |
| Linear | ≈1.10 ms | Pendiente | Pendiente |


### CPU-2 — Angie

Optimization V1 produjo una reducción clara del tiempo de ejecución en
CPU-2.

| Métrica / Operación | Baseline | Optimization V1 | Mejora |
|---|---:|---:|---:|
| Inferencia C++ media | 345.05 ms | 236.98 ms | **31.32 %** |
| Inferencia mediana | 338.62 ms | 232.31 ms | **31.39 %** |
| Inferencia p95 | 379.04 ms | 269.03 ms | **29.02 %** |
| Flujo total medio | 350.30 ms | 242.03 ms | **30.91 %** |
| Pointwise Conv2D | 252.77 ms | 160.33 ms | **36.57 %** |
| Depthwise Conv2D | 34.46 ms | 24.66 ms | **28.43 %** |
| Conv2D | 17.01 ms | 13.29 ms | **21.85 %** |
| ReLU6 | 18.88 ms | 18.59 ms | 1.52 % |
| BatchNorm2D | 11.28 ms | 11.18 ms | 0.84 % |
| Linear | 1.52 ms | 1.47 ms | 2.88 % |

El tiempo medio de inferencia disminuyó de **345.05 ms a 236.98 ms**, lo
que representa una reducción del **31.32 %**.

El factor de aceleración obtenido fue aproximadamente:

**Speedup = 1.46×**

Los tres kernels modificados redujeron su tiempo acumulado. La mayor
reducción se obtuvo en **Pointwise Conv2D**, con un **36.57 %**, seguido de
**Depthwise Conv2D**, con **28.43 %**, y **Conv2D**, con **21.85 %**.

Pointwise Conv2D continúa siendo el kernel con mayor tiempo acumulado, pero
su reducción de **252.77 ms a 160.33 ms** explica gran parte de la mejora
observada en el tiempo total de inferencia.


### Kria KV260 — Cortex-A53

El comportamiento de Optimization V1 fue diferente en la CPU ARM
Cortex-A53 de la Kria KV260.

| Métrica / Operación | Baseline | Optimization V1 | Cambio |
|---|---:|---:|---:|
| Inferencia C++ media | 2779.90 ms | 3076.21 ms | **+10.66 % tiempo** |
| Inferencia mediana | 2779.11 ms | 3077.01 ms | **+10.72 % tiempo** |
| Inferencia p95 | 2786.51 ms | 3084.06 ms | **+10.68 % tiempo** |
| Flujo total medio | 2811.95 ms | 3109.29 ms | **+10.57 % tiempo** |
| Pointwise Conv2D | 2188.59 ms | 2503.50 ms | **+14.39 % tiempo** |
| Depthwise Conv2D | 216.62 ms | 195.37 ms | **9.81 % mejora** |
| Conv2D | 115.27 ms | 109.99 ms | **4.58 % mejora** |
| BatchNorm2D | 154.38 ms | 160.02 ms | +3.65 % tiempo |
| ReLU6 | 49.62 ms | 49.70 ms | +0.17 % tiempo |
| Linear | 5.12 ms | 5.17 ms | +0.89 % tiempo |

En la Kria, el tiempo medio de inferencia aumentó de **2779.90 ms a
3076.21 ms**, correspondiente a un incremento del **10.66 %**.

El comportamiento de los kernels modificados no fue uniforme.
**Depthwise Conv2D mejoró un 9.81 %** y **Conv2D un 4.58 %**. Sin embargo,
**Pointwise Conv2D aumentó su tiempo en 14.39 %**, pasando de
**2188.59 ms a 2503.50 ms**.

Debido a que Pointwise Conv2D representa la mayor parte del tiempo de
inferencia, este aumento fue suficiente para superar las mejoras obtenidas
en Depthwise Conv2D y Conv2D, provocando que el tiempo total de MobileNetV2
aumentara.


### Conclusión

Optimization V1 demuestra que una misma modificación del código no produce
necesariamente el mismo resultado en diferentes arquitecturas.

En CPU-2, las modificaciones realizadas sobre Pointwise Conv2D, Depthwise
Conv2D y Conv2D redujeron el tiempo medio de inferencia en **31.32 %**,
alcanzando un speedup aproximado de **1.46×**.

En la Kria KV260, Depthwise Conv2D y Conv2D también mejoraron, pero
Pointwise Conv2D presentó una regresión del **14.39 %**. Debido al peso de
este kernel dentro de MobileNetV2, el tiempo medio de inferencia terminó
aumentando un **10.66 %**.

Por lo tanto, Optimization V1 resulta efectiva en CPU-2, pero no puede
considerarse una optimización general para todas las plataformas evaluadas.
El resultado evidencia la necesidad de medir las optimizaciones directamente
sobre cada arquitectura objetivo.

El proceso utilizado fue:

**Perfilado → identificación de cuellos de botella → optimización →
validación → nuevo perfilado**

Esto permitió comprobar experimentalmente tanto la mejora obtenida en CPU-2
como la regresión observada en la Kria, en lugar de asumir que una
modificación del código produciría el mismo efecto en ambas plataformas.