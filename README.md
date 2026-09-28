# MobileNetV2 C++ — Optimization V1

Esta rama contiene la **primera etapa de optimización** de la implementación
manual de MobileNetV2 en C++.

Las optimizaciones realizadas en esta etapa se concentran en tres operaciones:

- Pointwise Conv2D
- Depthwise Conv2D
- Conv2D

La arquitectura de MobileNetV2, los pesos de la red y el resultado esperado
de la inferencia se mantienen sin cambios.

---

# 1. Reproducibilidad

## 1.1 Entrar al proyecto

Ubicarse en la raíz del proyecto:

```bash
cd MobileNetV2
```

---

## 1.2 Activar el entorno virtual

Si el entorno `venv` ya se encuentra creado:

```bash
source venv/bin/activate
```

Si es la primera ejecución y el entorno todavía no existe:

```bash
make setup
source venv/bin/activate
```

---

## 1.3 Limpiar compilaciones anteriores

Antes de compilar:

```bash
make clean
```

---

## 1.4 Compilar con soporte para perfilado

Para evaluar Optimization V1 se debe compilar habilitando el perfilado interno
de los kernels:

```bash
make profile-build
```

La compilación utiliza `-O2` y genera el módulo C++ correspondiente a la
arquitectura de la máquina.

---

## 1.5 Validar la implementación

Antes de realizar el perfilado se debe comprobar que la implementación C++
continúa produciendo resultados correctos después de las optimizaciones.

Ejecutar:

```bash
PYTHONPATH="$PWD" python app/run_cpp_mobilenetv2.py \
    test_images/imagen.jpeg
```

La ejecución debe finalizar con:

```text
RESULTADO: PASS
```

Si la validación falla, no se debe continuar con el perfilado.

---

## 1.6 Ejecutar el perfilado

Después de validar la implementación:

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

`NOMBRE_RESULTADO` debe identificar la plataforma utilizada.

Por ejemplo:

```bash
--output-dir resultados/milagro_cpu_optimization_v1
```

Para que los resultados sean comparables con el baseline y entre plataformas,
se deben mantener las mismas condiciones:

| Parámetro | Valor |
|---|---|
| Imagen | `test_images/imagen.jpeg` |
| Iteraciones medidas | 120 |
| Warm-up | 8 |
| Hilos | 1 |
| Preprocesamiento | `official` |
| Perfilado interno de kernels | habilitado |
| Optimización del compilador | `-O2` |

No se utiliza OpenMP, ejecución multihilo ni SIMD explícito en esta primera
etapa de optimización.

---

## 1.7 Resultados generados

El perfilado genera una carpeta dentro de `resultados/` con los archivos:

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

Los principales archivos utilizados para la comparación son:

### `resumen_etapas.csv`

Contiene las métricas correspondientes a:

- lectura y decodificación;
- preprocesamiento;
- inferencia C++;
- posprocesamiento;
- flujo total.

### `resumen_kernels.csv`

Contiene los tiempos individuales de las operaciones internas de MobileNetV2.

Estos archivos permiten realizar la comparación:

```text
Baseline → Optimization V1
```

---

# 2. Resultados

Los resultados de Optimization V1 se comparan contra el perfilado baseline
obtenido antes de modificar los kernels.

La reducción porcentual del tiempo se calcula como:

```text
Mejora (%) = (Baseline - Optimization V1) / Baseline × 100
```

Un porcentaje positivo representa una reducción del tiempo de ejecución.

---

## 2.1 CPU-1 — Mila

| Métrica / Operación | Baseline | Optimization V1 | Mejora |
|---|---:|---:|---:|
| Inferencia C++ media | 286.33 ms | 216.28 ms | **24.46 %** |
| Inferencia mediana | 285.46 ms | 216.14 ms | **24.28 %** |
| Inferencia p95 | 290.20 ms | 217.41 ms | **25.08 %** |
| Flujo total medio | 291.44 ms | 221.58 ms | **23.97 %** |
| Pointwise Conv2D | 199.56 ms | 129.57 ms | **35.08 %** |
| Depthwise Conv2D | 28.74 ms | 27.99 ms | **2.59 %** |
| Conv2D | 15.89 ms | 17.24 ms | −8.46 % |
| ReLU6 | 15.05 ms | 14.89 ms | 1.05 % |
| BatchNorm2D | 19.91 ms | 19.85 ms | 0.32 % |
| Linear | 1.11 ms | 1.11 ms | 0.43 % |

El tiempo medio de inferencia disminuyó de **286.33 ms a 216.28 ms**, lo que
representa una reducción del **24.46 %**.

El factor de aceleración obtenido fue aproximadamente:

```text
Speedup = 286.33 / 216.28 ≈ 1.32×
```

De los tres kernels modificados, dos presentaron mejoras: Pointwise Conv2D
(**35.08 %**) y Depthwise Conv2D (**2.59 %**). Conv2D, en cambio, resultó
**más lento (~8.46 %)** que el baseline en esta máquina.

Pointwise Conv2D continúa siendo la operación con mayor tiempo acumulado,
y su reducción de **199.56 ms a 129.57 ms** explica la mayor parte de la
mejora obtenida en el tiempo total de inferencia, a pesar del retroceso en
Conv2D.
---

## 2.2 CPU-2 — Angie

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

El tiempo medio de inferencia disminuyó de **345.05 ms a 236.98 ms**, lo que
representa una reducción del **31.32 %**.

El factor de aceleración obtenido fue aproximadamente:

```text
Speedup = 345.05 / 236.98 ≈ 1.46×
```

Los tres kernels modificados presentaron una reducción en su tiempo acumulado:

- Pointwise Conv2D: **36.57 %**
- Depthwise Conv2D: **28.43 %**
- Conv2D: **21.85 %**

Pointwise Conv2D continúa siendo la operación con mayor tiempo acumulado, pero
su reducción de **252.77 ms a 160.33 ms** explica una parte importante de la
reducción obtenida en el tiempo total de inferencia.

---

## 2.3 Kria KV260 — Cortex-A53

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

En la Kria KV260 se obtuvo un comportamiento diferente al observado en
CPU-2.

El tiempo medio de inferencia aumentó de **2779.90 ms a 3076.21 ms**, lo que
representa un incremento aproximado del **10.66 %**.

Los tres kernels modificados no presentaron el mismo comportamiento:

- Depthwise Conv2D redujo su tiempo aproximadamente **9.81 %**.
- Conv2D redujo su tiempo aproximadamente **4.58 %**.
- Pointwise Conv2D aumentó su tiempo aproximadamente **14.39 %**.

Debido a que Pointwise Conv2D representa la mayor parte del tiempo de
inferencia, el incremento observado en esta operación supera las mejoras
obtenidas en Depthwise Conv2D y Conv2D.

Por esta razón, Optimization V1 reduce el rendimiento global de MobileNetV2
en el Cortex-A53 de la Kria, aunque dos de los tres kernels modificados sí
presenten reducciones individuales de tiempo.

---

# 3. Primera etapa de optimización

Optimization V1 corresponde a la **primera etapa de optimización** de la
implementación C++.

El objetivo de esta etapa no es obtener todavía la implementación final más
rápida posible, sino realizar una primera optimización dirigida por los
resultados obtenidos durante el perfilado baseline.

El procedimiento utilizado fue:

```text
Perfilado baseline
        ↓
Identificación de cuellos de botella
        ↓
Selección de operaciones
        ↓
Optimization V1
        ↓
Validación
        ↓
Nuevo perfilado
        ↓
Comparación de resultados
```

---

## 3.1 ¿Por qué se optimizaron estas operaciones?

El perfilado baseline permitió determinar cuánto tiempo consumía cada
operación interna de MobileNetV2.

En CPU-2 se obtuvieron los siguientes tiempos acumulados:

| Operación | Tiempo baseline |
|---|---:|
| Pointwise Conv2D | 252.77 ms |
| Depthwise Conv2D | 34.46 ms |
| ReLU6 | 18.88 ms |
| Conv2D | 17.01 ms |
| BatchNorm2D | 11.28 ms |
| Linear | 1.52 ms |

El resultado más importante fue **Pointwise Conv2D**, con aproximadamente
**252.77 ms** acumulados.

Esta operación representaba cerca del **75 % del tiempo acumulado de los
kernels perfilados**, por lo que constituía el principal cuello de botella de
la implementación.

Por esta razón, Pointwise Conv2D fue la principal operación seleccionada para
Optimization V1.

También se seleccionaron:

- **Depthwise Conv2D**, debido a su costo acumulado y a su uso repetido dentro
  de los bloques de MobileNetV2.
- **Conv2D**, para aplicar una primera optimización sobre las principales
  operaciones convolucionales utilizadas por la implementación.

Las demás operaciones se mantuvieron sin modificaciones durante esta etapa
para limitar el alcance de Optimization V1 y poder observar con mayor claridad
el efecto de los cambios realizados.

---

# 4. Cambios realizados en Optimization V1

Los archivos modificados son:

```text
kernels/Pointwise_Conv2d.cpp
kernels/Depthwise_Conv2d.cpp
kernels/Conv2d.cpp
```

Las modificaciones se concentran en la estructura interna de los bucles, el
cálculo de índices y los patrones de acceso a memoria.

No se modifica la operación matemática que debe realizar cada kernel.

---

## 4.1 Pointwise Conv2D

Pointwise Conv2D utiliza convoluciones de tamaño `1×1`.

En la implementación baseline, el orden de los bucles provocaba que el acceso
a los datos de entrada al recorrer los canales no siempre siguiera posiciones
contiguas en memoria.

Optimization V1 reorganiza el cálculo para recorrer de forma más conveniente
la dimensión espacial del tensor.

Además:

- se inicializa previamente el mapa de salida con el bias;
- se reutilizan valores de los pesos;
- se reducen cálculos repetitivos de índices;
- se favorece el recorrido consecutivo de posiciones espaciales.

El objetivo es reducir trabajo dentro de los bucles internos y mejorar el
patrón de acceso a memoria.

---

## 4.2 Depthwise Conv2D

En Depthwise Conv2D cada canal de entrada se procesa de forma independiente.

Optimization V1 precalcula valores utilizados repetidamente durante la
convolución, entre ellos:

- tamaño de los canales de entrada;
- tamaño de los canales de salida;
- tamaño del kernel;
- posiciones base de entrada;
- posiciones base de salida;
- posiciones base de los pesos;
- valor del bias;
- posiciones base correspondientes a las filas.

También se evita realizar trabajo innecesario cuando una posición del kernel
queda fuera de los límites válidos de la entrada.

El objetivo es reducir la cantidad de operaciones auxiliares realizadas
dentro de los bucles más internos.

---

## 4.3 Conv2D

La optimización de Conv2D sigue un principio similar.

Se precalculan posiciones base asociadas a:

- batch;
- canal de entrada;
- canal de salida;
- pesos;
- filas de entrada;
- filas de salida.

Esto reduce la cantidad de expresiones de indexación que deben evaluarse
repetidamente dentro de los bucles internos de la convolución.

El orden de acumulación utilizado para calcular cada salida se mantiene, de
forma que la operación matemática del kernel no cambia.

---

# 5. Conclusiones de Optimization V1

Los resultados muestran que Optimization V1 produce efectos diferentes según
la plataforma utilizada.

En CPU-1, la optimización redujo el tiempo medio de inferencia en **24.46 %**,
pasando de **286.33 ms a 216.28 ms**, con un speedup aproximado de **1.32×**.
Dos de los tres kernels modificados mejoraron —Pointwise Conv2D (**35.08 %**)
y Depthwise Conv2D (**2.59 %**)—, mientras que Conv2D resultó **~8.46 % más
lento** que el baseline en esta máquina.

En CPU-2, la optimización redujo el tiempo medio de inferencia en **31.32 %**,
pasando de **345.05 ms a 236.98 ms**, con un speedup aproximado de **1.46×**.
Los tres kernels modificados presentaron mejoras individuales, siendo
Pointwise Conv2D el cambio con mayor impacto.

En la Kria KV260, el comportamiento fue diferente. Depthwise Conv2D y Conv2D
redujeron sus tiempos, pero Pointwise Conv2D aumentó aproximadamente un
**14.39 %**. Como Pointwise Conv2D continúa siendo la operación dominante,
el tiempo medio de inferencia aumentó aproximadamente un **10.66 %**.

Por lo tanto, los resultados de esta primera etapa muestran que una
modificación que mejora el rendimiento en una plataforma no necesariamente
produce el mismo efecto en otra arquitectura: incluso entre las dos
computadoras x86 (CPU-1 y CPU-2), Conv2D tuvo comportamientos opuestos
—mejoró en una y empeoró en la otra—, mientras que Pointwise Conv2D fue la
única optimización consistentemente positiva en ambas.

No se determina únicamente a partir de este perfilado cuál es la causa
microarquitectónica de esta diferencia. Para establecerla sería necesario
realizar análisis adicionales sobre aspectos como el código generado por el
compilador, la vectorización y el comportamiento de la jerarquía de memoria.

Optimization V1 debe entenderse como la **primera etapa del proceso de
optimización**. Los resultados obtenidos permiten identificar qué
transformaciones fueron efectivas en cada plataforma y proporcionan una base
experimental para decidir las siguientes optimizaciones.
