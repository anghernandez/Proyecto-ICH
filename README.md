# MobileNetV2 C++ — Perfilado en CPU

Esta rama contiene la implementación de **MobileNetV2 en C++** utilizada para
realizar el perfilado inicial de rendimiento en CPU.

El objetivo de esta rama es obtener una medición **baseline**, antes de aplicar
optimizaciones sobre los kernels C++. Esto permite ejecutar la misma
implementación bajo condiciones controladas en diferentes plataformas y
comparar posteriormente el efecto de las optimizaciones.

---

## Condiciones del experimento

Para mantener la comparabilidad entre las diferentes plataformas, el perfilado
debe realizarse utilizando las mismas condiciones:

| Parámetro | Valor |
|---|---|
| Imagen de entrada | `test_images/imagen.jpeg` |
| Iteraciones medidas | 120 |
| Warm-up | 8 iteraciones |
| Número de hilos | 1 |
| Preprocesamiento | `official` |
| Perfilado interno de kernels | habilitado |
| Optimización de compilación | `-O2` |

Se deben mantener la misma implementación C++, los mismos pesos de MobileNetV2
y la misma imagen de entrada.

---

# Reproducibilidad del perfilado

## 1. Entrar al proyecto

Ubicarse en la carpeta raíz de `MobileNetV2`:

```bash
cd MobileNetV2
```

---

## 2. Activar el entorno virtual

Si el entorno virtual `venv` ya se encuentra creado:

```bash
source venv/bin/activate
```

Si es la primera ejecución y el entorno todavía no existe:

```bash
make setup
source venv/bin/activate
```

---

## 3. Limpiar compilaciones anteriores

Antes de realizar las pruebas se recomienda eliminar cualquier módulo C++
generado anteriormente:

```bash
make clean
```

---

## 4. Compilar con soporte para perfilado

Compilar la implementación C++ habilitando el perfilado interno de los kernels:

```bash
make profile-build
```

Este comando compila el módulo C++ con `-O2` y habilita la instrumentación
utilizada para medir individualmente las operaciones internas de MobileNetV2.

El módulo generado dependerá de la arquitectura y versión de Python de la
máquina donde se realice la prueba.

---

## 5. Validar la implementación

Antes de ejecutar las mediciones se debe comprobar que la implementación C++
produce resultados correctos.

Ejecutar:

```bash
PYTHONPATH="$PWD" python app/run_cpp_mobilenetv2.py \
    test_images/imagen.jpeg
```

Al finalizar se realiza una comparación entre la salida de referencia y la
implementación C++.

La validación debe terminar con:

```text
RESULTADO: PASS
```

Si la validación falla, **no se debe continuar con el perfilado** hasta
identificar la causa.

---

## 6. Ejecutar el perfilado

Una vez validada la implementación, ejecutar:

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

Se debe reemplazar:

```text
NOMBRE_RESULTADO
```

por un nombre que identifique claramente la plataforma utilizada.

Por ejemplo:

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

Para que los resultados sean comparables con los obtenidos en las otras
plataformas, **no modificar** los siguientes parámetros:

```text
--iterations 120
--warmup 8
--threads 1
--preprocess official
--kernel-profile on
```

---

# Resultados generados

Al finalizar el perfilado se crea una carpeta dentro de `resultados/`.

Por ejemplo:

```text
resultados/milagro_cpu/
```

La carpeta contiene archivos similares a los siguientes:

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

Cada archivo conserva información necesaria para analizar o reproducir el
experimento.

---

## `resumen_etapas.csv`

Este archivo contiene el resumen de los tiempos correspondientes a las
principales etapas del procesamiento:

- lectura y decodificación;
- preprocesamiento;
- inferencia C++;
- posprocesamiento;
- flujo total.

Para cada etapa se almacenan métricas como:

- media;
- mediana;
- desviación estándar;
- percentil 95 (`p95`);
- mínimo;
- máximo.

Las métricas principales para comparar el rendimiento general de MobileNetV2
son:

| Métrica | Descripción |
|---|---|
| Inferencia C++ media | Tiempo promedio de ejecución de MobileNetV2 C++ |
| Inferencia mediana | Tiempo típico de ejecución con menor influencia de valores extremos |
| Inferencia p95 | Tiempo por debajo del cual se encuentra el 95 % de las ejecuciones |
| Flujo total medio | Tiempo promedio del procesamiento completo |

---

## `resumen_kernels.csv`

Este archivo contiene el perfilado de las operaciones internas utilizadas
durante la inferencia de MobileNetV2.

Entre las operaciones medidas se encuentran:

- Conv2D;
- Pointwise Conv2D;
- Depthwise Conv2D;
- BatchNorm2D;
- ReLU6;
- LayerAdd;
- Global Average Pooling;
- Linear.

El objetivo de este archivo es determinar cuánto tiempo consume cada operación
y permitir identificar los principales **cuellos de botella** de la
implementación C++.

---

## Otros archivos

### `etapas.csv`

Contiene las mediciones individuales de las diferentes etapas para cada
iteración ejecutada.

### `kernels.csv`

Contiene las mediciones individuales de las llamadas a los kernels internos
de MobileNetV2.

### `validacion.json`

Almacena información relacionada con la validación de la implementación.

### `metadata.json`

Contiene información sobre la configuración utilizada durante el experimento.

### `cpuinfo.txt`

Guarda información sobre el procesador utilizado durante la prueba.

### `meminfo.txt`

Guarda información relacionada con la memoria del sistema.

### `predicciones.csv`

Contiene las predicciones registradas durante las ejecuciones.

### `inicializacion.json`

Contiene las mediciones relacionadas con la inicialización previa a la
ejecución del perfilado.

---

# Archivos que se deben compartir

Después de completar el perfilado se debe conservar la **carpeta completa**
generada dentro de:

```text
resultados/NOMBRE_RESULTADO/
```

Como mínimo, para realizar la comparación de rendimiento se utilizarán:

```text
resumen_etapas.csv
resumen_kernels.csv
validacion.json
metadata.json
cpuinfo.txt
```

Sin embargo, se recomienda conservar y compartir **todos los archivos
generados** para mantener la trazabilidad del experimento.

---

# Flujo completo

El procedimiento completo para reproducir el perfilado es:

```text
Activar entorno
      ↓
Limpiar compilación anterior
      ↓
Compilar con perfilado
      ↓
Validar MobileNetV2 C++
      ↓
RESULTADO: PASS
      ↓
Ejecutar 8 warm-ups
      ↓
Ejecutar 120 iteraciones medidas
      ↓
Guardar resultados
      ↓
Analizar resumen_etapas.csv
      ↓
Analizar resumen_kernels.csv
```

En comandos:

```bash
cd MobileNetV2

source venv/bin/activate

make clean
make profile-build

PYTHONPATH="$PWD" python app/run_cpp_mobilenetv2.py \
    test_images/imagen.jpeg

PYTHONPATH="$PWD" python app/profile_complete.py \
    --images test_images/imagen.jpeg \
    --iterations 120 \
    --warmup 8 \
    --threads 1 \
    --output-dir resultados/NOMBRE_RESULTADO \
    --preprocess official \
    --kernel-profile on
```

---

# Objetivo del baseline

Los resultados obtenidos mediante este procedimiento representan el
**rendimiento inicial o baseline** de la implementación C++.

Este baseline se utiliza posteriormente como referencia para determinar qué
operaciones consumen la mayor parte del tiempo de inferencia y evaluar
cuantitativamente el efecto de futuras optimizaciones.

Por esta razón, en esta rama **no se aplican optimizaciones específicas a los
kernels**. Su propósito es mantener una implementación de referencia y un
procedimiento reproducible de medición.
