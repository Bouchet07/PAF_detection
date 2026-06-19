# Contexto y Guía del Proyecto (TFG) - Predicción de PAF

Este documento sirve como referencia centralizada para el Trabajo de Fin de Grado (TFG) en la Universidad de Zaragoza, enfocado en la **predicción inminente de la Fibrilación Auricular Paroxística (PAF)** a partir de señales de electrocardiograma (ECG) mediante aprendizaje profundo.

---

## 1. Objetivo del Proyecto

El objetivo principal es predecir episodios de fibrilación auricular paroxística (PAF) **dentro de una ventana de 5 minutos antes de que ocurran** (Pre-PAF). Se trata de un problema de clasificación binaria:
- **Clase 1 (Pre-PAF):** Segmento de ECG de 5 minutos inmediatamente anterior al inicio de un episodio de PAF.
- **Clase 0 (Control/Normal):** Segmento de ECG de 5 minutos en ritmo sinusal normal, alejado de episodios de PAF.

---

## 2. Bases de Datos Utilizadas

El pipeline procesa y unifica **5 bases de datos** procedentes de PhysioNet, cuyas descripciones originales y artículos de referencia se encuentran en [sources/](file:///C:/Users/diego/Desktop/Programming/python/PAF_detection/sources):

1. **PAF Prediction Challenge Database (`afpdb`):**
   - **Ubicación:** `data/paf-prediction-challenge-database/`
   - **Registros:** 100 registros de 30 minutos (50 normales `n`, 50 pre-PAF `p` que anteceden inmediatamente a un episodio) procedentes de 48 sujetos. Adicionalmente, cuenta con un conjunto de test blindado de otros 100 registros.
   - **Frecuencia de muestreo original:** 128 Hz.
   - **Canales:** 2.

2. **China Physiological Signal Challenge 2021 (`cpsc2021`):**
   - **Ubicación:** `data/cpsc2021/`
   - **Registros:** 1425 registros de duración variable procedentes de 105 sujetos. Contiene anotaciones sobre el inicio y final exacto de episodios de PAF.
   - **Frecuencia de muestreo original:** 200 Hz.
   - **Canales:** 2.

3. **Long-Term Atrial Fibrillation Database (`ltafdb`):**
   - **Ubicación:** `data/ltafdb/`
   - **Registros:** 84 registros de ECG ambulatorio de larga duración (~24 horas por registro) procedentes de 84 sujetos con fibrilación auricular paroxística o sostenida.
   - **Frecuencia de muestreo original:** 128 Hz.
   - **Canales:** 2.

4. **Smart Health Devices - Atrial Fibrillation Database (`shdb-af`):**
   - **Ubicación:** `data/shdb-af/`
   - **Registros:** 128 registros de Holter de larga duración (~24 horas) grabados mediante dispositivos inteligentes de 122 sujetos. Usa el archivo `AdditionalData.csv` para mapear los identificadores de registros a sujetos.
   - **Frecuencia de muestreo original:** 125 Hz (resampleados a 200 Hz).
   - **Canales:** 2.

5. **MIT-BIH Atrial Fibrillation Database (`afdb`):**
   - **Ubicación:** `data/afdb/`
   - **Registros:** 25 registros de 10 horas de duración (23 con señales de ECG utilizables), con anotaciones manuales detalladas de ritmo.
   - **Frecuencia de muestreo original:** 250 Hz.
   - **Canales:** 2.


### Estadísticas Reales de las Bases de Datos (Salida de `stats.py`)
Al analizar los archivos locales del proyecto, obtenemos las siguientes estadísticas de registros e inicios de episodios de fibrilación auricular:
- **AFPDB:** 50 registros pre-PAF (`p`), 50 normales (`n`).
- **CPSC2021:** 1425 registros totales (1425 anotados). Se detectaron 968 inicios de episodios de AFIB.
- **LTAFDB:** 84 registros totales (84 anotados). Se detectaron 7358 inicios de episodios de AFIB.
- **SHDB-AF:** 128 registros totales (98 anotados). Se detectaron 794 inicios de episodios de AFIB.
- **AFDB (MIT-BIH):** 25 registros totales de 10 horas de duración (23 con señales de ECG utilizables).

---

## 3. Pipeline de Preprocesamiento (`src/data/preprocess.py`)

El script unifica y procesa los datos crudos para evitar sesgos y filtraciones de información:
- **Armonización:** Todas las señales se resamplean a **128 Hz** y se fuerza una estructura de **2 canales**.
- **Extracción de Clase 1 (Pre-PAF):** Se aíslan ventanas de hasta 5 minutos (`WINDOW_PRE_PAF_SEC = 300`, con un mínimo de 10 segundos definido en `MIN_PRE_PAF_SEC`) previas al inicio de un episodio de PAF. La ventana retrocede hasta el final de la arritmia anterior o el inicio del registro, garantizando que el segmento contenga solo ritmo sinusal.
- **Extracción de Clase 0 (Control):** Se aíslan ventanas de 5 minutos en ritmo sinusal, alejadas al menos 30 minutos (`label_0_gap_sec = 1800`) de cualquier episodio de PAF.
- **Extracción de HRV (Variabilidad del Ritmo Cardíaco):** A partir de los picos R (QRS) detectados en las anotaciones, se calculan 9 variables HRV: `mean_rr`, `std_rr`, `rmssd`, `pnn50`, `mean_hr`, `std_hr`, `lf`, `hf`, y la ratio `lf_hf_ratio`.

Los segmentos procesados se guardan como archivos `.npy` en `processed_data/`, y los metadatos correspondientes (archivo, sujeto, clase y HRV) en `metadata.csv`.

---

## 4. Arquitectura de Modelos (`src/models/`)

El proyecto cuenta con tres alternativas de redes neuronales 1D que reciben una entrada de dimensiones `(Batch, 2, Samples)`:
- **ResNet 1D (`resnet1d.py`):** Implementa el baseline con bloques residuales convolucionales 1D.
- **SEResNet 1D (`senet1d.py`):** Añade atención de canales a la ResNet base mediante bloques Squeeze-and-Excitation (SE).
- **CNN-Transformer Híbrido (`transformer1d.py`):** Utiliza capas convolucionales iniciales para extraer rasgos morfológicos seguidos de codificadores Transformer para capturar dependencias temporales de largo alcance.

---

## 5. Estrategia de Entrenamiento y Evaluación

### Prevención de Data Leakage (Filtración de Datos)
El reparto de conjuntos de entrenamiento, validación y prueba se realiza **agrupando estrictamente por sujeto/paciente** ([dataloader.py](file:///C:/Users/diego/Desktop/Programming/python/PAF_detection/src/data/dataloader.py)). Esto garantiza que las señales de un mismo sujeto nunca se compartan entre distintos splits.

### Carga Dinámica de Ventanas (`dataset.py`)
Dado que la red se entrena típicamente con ventanas más pequeñas (por ejemplo, de 10 segundos, configurables por el usuario):
- **Modo train:** Se extrae una rodaja aleatoria de la duración deseada de la ventana de 5 minutos para aumentar la robustez.
- **Modo val / test:** Se extrae una rodaja determinista al final de la ventana (la más cercana al inicio del episodio arritmogénico en Clase 1).
- Se aplica normalización Z-score por canal a cada rodaja de forma independiente.

### Ejecución de Comandos con `uv`
Para preparar el entorno y ejecutar el entrenamiento/evaluación, se utilizan los siguientes comandos:

```bash
# 1. Sincronizar dependencias (y GPU si está disponible)
uv sync
uv sync --extra gpu

# 2. Descargar y preprocesar
uv run python -m src.utils.download
uv run python -m src.data.preprocess

# 3. Entrenar el modelo (con soporte para HRV y validación cruzada k-fold)
uv run python train.py --model_type senet --window_seconds 10 --num_epochs 30 --use_hrv --k_fold 5

# 4. Evaluar en el conjunto de test blindado
uv run python test.py --metadata_path metadata.csv --data_dir processed_data
```

---

## 6. Estructura de la Memoria del TFG (`report/`)

La memoria escrita está configurada en LaTeX bajo las normativas de formato de la **Universidad de Zaragoza** (fuente tipo Carlito/Calibri, interlineado 1.5, márgenes de 2.5cm/3cm, etc.).

### Archivo Raíz
- [report/main.tex](file:///C:/Users/diego/Desktop/Programming/python/PAF_detection/report/main.tex): Configura paquetes básicos, estilos, glosarios y estructura de capítulos. Se compila ejecutando `latexmk -pdf main.tex`.

### Estructura de Capítulos ([report/chapters/](file:///C:/Users/diego/Desktop/Programming/python/PAF_detection/report/chapters)):
- **`00_abstract.tex`:** Resumen (castellano) y Abstract (inglés).
- **`01_introduccion.tex`:** Introducción, objetivos y alcance del proyecto.
- **`02_datos.tex`:** Detalle exhaustivo y tabla comparativa de las 5 bases de datos empleadas en el estudio (ya actualizada con las últimas incorporaciones).
- **`03_metodologia.tex`:** Pipeline de preprocesamiento, extracción de ventanas y HRV, arquitecturas de redes y diseño experimental (Grouped Split).
- **`04_resultados.tex`:** Métricas cuantitativas, análisis comparativo y discusión.
- **`05_conclusiones.tex`:** Cumplimiento de objetivos, limitaciones y líneas futuras (Transformers de señales, afinamiento de hiperparámetros).
- **`99_ejemplos.tex`:** Guía técnica interna de ejemplos y sintaxis útil de LaTeX.
