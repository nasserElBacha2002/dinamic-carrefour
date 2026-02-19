# Sistema de Inventario de Góndolas — Roboflow

Sistema de visión artificial para **detectar, clasificar y contar** productos en góndolas de supermercado a partir de videos, usando **Roboflow** como motor de inferencia en la nube.

---

## Tabla de Contenidos

1. [Qué hace el sistema](#qué-hace-el-sistema)
2. [Arquitectura general](#arquitectura-general)
3. [Estructura del proyecto](#estructura-del-proyecto)
4. [Requisitos previos](#requisitos-previos)
5. [Instalación](#instalación)
6. [Configuración](#configuración)
7. [Uso principal — procesar un video](#uso-principal--procesar-un-video)
8. [Argumentos CLI de `run.py`](#argumentos-cli-de-runpy)
9. [Catálogo de productos (`eans.txt`)](#catálogo-de-productos-eanstxt)
10. [Archivos de mapeo](#archivos-de-mapeo)
11. [Agregar un producto nuevo (flujo completo)](#agregar-un-producto-nuevo-flujo-completo)
12. [Subir imágenes a Roboflow](#subir-imágenes-a-roboflow)
13. [Sincronizar label map](#sincronizar-label-map)
14. [Reentrenar el modelo en Roboflow](#reentrenar-el-modelo-en-roboflow)
15. [Confianza del modelo](#confianza-del-modelo)
16. [Diferencia entre ROBOFLOW_PROJECT y ROBOFLOW_WORKFLOW](#diferencia-entre-roboflow_project-y-roboflow_workflow)
17. [Output del sistema](#output-del-sistema)
18. [Troubleshooting](#troubleshooting)
19. [Productos actuales en el modelo](#productos-actuales-en-el-modelo)
20. [Versión](#versión)

---

## Qué hace el sistema

1. **Recibe un video** grabado frente a una góndola de supermercado.
2. **Extrae frames** a un FPS configurable, descartando los borrosos.
3. **Envía cada frame a Roboflow** que detecta y clasifica productos.
4. **Mapea cada clase** detectada a un EAN (código de barras) usando `roboflow_label_map.json`.
5. **Genera un reporte** (`inventario_sku.csv`) con el conteo por EAN y fecha.
6. **Genera imágenes anotadas** con bounding boxes sobre los frames originales.

---

## Arquitectura general

```text
Video (.MOV)
   │
   ▼
analizar_video.py   →  Extrae frames (1 fps por defecto)
   │
   ▼
detectar_roboflow.py →  Envía frame a Roboflow Workflows API
   │                     Recibe: clase + bounding box + confianza
   ▼
roboflow_label_map.json →  Traduce clase (ej: "3") a EAN (ej: "7790895000997")
   │
   ▼
inventario_sku.csv   →  EAN, Cantidad, Fecha
```

El sistema usa **Roboflow** exclusivamente — no hay modelo local. Toda la inferencia se hace vía API serverless.

---

## Estructura del proyecto

```text
dinamic-carrefour/
├── run.py                          # Punto de entrada principal
├── .env                            # Variables de entorno (API keys, workflow)
├── .env.example                    # Template de .env
├── eans.txt                        # Catálogo: EAN → descripción
├── ean_class_map.json              # Mapeo EAN → nombre de clase visual
├── roboflow_label_map.json         # Mapeo clase Roboflow → EAN (para inferencia)
├── requirements.txt                # Dependencias Python
├── data/                           # Videos de entrada
│   ├── IMG_2195.MOV
│   ├── IMG_2196.MOV
│   └── ...
├── imagenes/                       # Imágenes de referencia por EAN
│   ├── 7750496/                    # Pepsi 2.25L
│   ├── 7791813421719/              # Pepsi 1.5L
│   ├── 7791813423775/              # Pepsi Black 1.5L
│   ├── 7790895000997/              # Coca-Cola 2.25L
│   ├── 7790895000430/              # Coca-Cola 1.5L
│   ├── 7790895001130/              # Coca-Cola Zero 1.5L
│   └── 7790315058201/              # Villavicencio Sport 750ml
├── src/
│   ├── main.py                     # Orquestador del pipeline
│   ├── analizar_video.py           # Análisis y extracción de frames
│   ├── detectar_roboflow.py        # Detector vía Roboflow API
│   ├── factory.py                  # Factory para inyección de dependencias
│   ├── protocols.py                # Protocolos/interfaces (DIP)
│   ├── exporters.py                # Exportadores de reportes
│   └── utils/
│       └── image_utils.py          # Utilidades de imagen
├── scripts/
│   ├── agregar_producto_auto.py    # Alta automática de producto nuevo
│   ├── buscarimagenes.py           # Descarga imágenes de Bing
│   ├── upload_to_roboflow.py       # Sube imágenes/anotaciones al dataset
│   ├── sync_eans_to_roboflow.py    # Sincronización incremental EANs → Roboflow
│   └── sync_roboflow_label_map.py  # Regenera roboflow_label_map.json
├── output/                         # Resultados de cada ejecución
│   └── VIDEO_TIMESTAMP/
│       ├── analisis_video.json
│       ├── frames_extraidos/
│       └── reporte_deteccion/
│           ├── inventario_sku.csv
│           └── *_detectado.jpg
└── tests/
    └── test_solid_improvements.py
```

---

## Requisitos previos

- **Python 3.8+** (probado con 3.13)
- **Cuenta de Roboflow** con un proyecto y modelo entrenado
- **API Key de Roboflow**
- **Conexión a internet** (la inferencia se ejecuta en la nube)

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone <URL_DEL_REPO>
cd dinamic-carrefour

# 2. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate   # macOS / Linux

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env
```

Editá `.env` con tus datos reales (ver siguiente sección).

---

## Configuración

### Archivo `.env`

Creá un archivo `.env` en la raíz del proyecto con estas variables:

```env
ROBOFLOW_API_KEY=TU_API_KEY_AQUI
ROBOFLOW_WORKSPACE=gondolacarrefour
ROBOFLOW_WORKFLOW=custom-workflow-2
ROBOFLOW_PROJECT=gondolacarrefour/gondola-dataset
```

| Variable | Qué es | Ejemplo |
|---|---|---|
| `ROBOFLOW_API_KEY` | Tu API Key de Roboflow | `UvOpfuykQC2paoNWmaOa` |
| `ROBOFLOW_WORKSPACE` | Nombre del workspace | `gondolacarrefour` |
| `ROBOFLOW_WORKFLOW` | ID del workflow de **inferencia** | `custom-workflow-2` |
| `ROBOFLOW_PROJECT` | Slug del **proyecto/dataset** | `gondolacarrefour/gondola-dataset` |

> ⚠️ **`ROBOFLOW_WORKFLOW` y `ROBOFLOW_PROJECT` NO son lo mismo.** Ver sección [Diferencia entre ROBOFLOW_PROJECT y ROBOFLOW_WORKFLOW](#diferencia-entre-roboflow_project-y-roboflow_workflow).

---

## Uso principal — procesar un video

```bash
python run.py data/IMG_2196.MOV
```

Eso es todo. El sistema:

1. Lee la API key desde `.env`
2. Extrae frames del video a 1 fps
3. Envía cada frame a Roboflow para detección
4. Genera `inventario_sku.csv` con el conteo por EAN
5. Genera imágenes anotadas con bounding boxes

### Ejemplo con opciones

```bash
python run.py data/IMG_2196.MOV \
  --fps 2.0 \
  --confianza 0.2 \
  --guardar-crops \
  --label-map roboflow_label_map.json
```

---

## Argumentos CLI de `run.py`

| Argumento | Tipo | Default | Descripción |
|---|---|---|---|
| `video` | posicional | — | Ruta al archivo de video |
| `--roboflow-api-key` | str | `.env` | API Key (si no se define en `.env`) |
| `--roboflow-workspace` | str | `gondolacarrefour` | Workspace de Roboflow |
| `--roboflow-workflow` | str | `.env` | Workflow ID de inferencia |
| `--label-map` | str | `roboflow_label_map.json` | Archivo de mapeo clase → EAN |
| `--confianza` | float | `0.25` | Confianza mínima para filtrar detecciones |
| `--fps` | float | `1.0` | Frames por segundo a extraer del video |
| `--guardar-crops` | flag | `false` | Guardar recorte individual de cada detección |
| `--sin-deteccion` | flag | `false` | Solo extraer frames, sin correr detección |
| `--sin-anotaciones` | flag | `false` | No generar imágenes anotadas (más rápido) |
| `--rotar` | flag | `false` | Rotar frames 90° (videos verticales) |
| `--output` | str | `output` | Directorio base para resultados |

### Ejemplo mínimo

```bash
python run.py data/MI_VIDEO.MOV
```

### Ejemplo completo

```bash
python run.py data/IMG_2196.MOV \
  --fps 1.0 \
  --confianza 0.2 \
  --guardar-crops \
  --roboflow-workflow custom-workflow-2 \
  --label-map roboflow_label_map.json \
  --output output
```

---

## Catálogo de productos (`eans.txt`)

Archivo de texto que define los productos conocidos. Formato: `EAN<TAB>DESCRIPCION`, una línea por producto.

```text
7750496	GASEOSA COLA REGULAR PEPSI PET X 2.25 LT
7791813421719	GASEOSA COLA REGULAR PEPSI PET X 1.5 LT
7791813423775	GASEOSA PEPSI BLACK PET X 1.5 LT
7790895000997	GASEOSA COLA REGULAR COCA COLA PET X 2.25 LT
7790895000430	GASEOSA COLA REGULAR COCA COLA PET X 1.5 LT
7790895001130	GASEOSA COCA COLA ZERO PET X 1.5 LT
7790315058201	AGUA MINERAL VILLAVICENCIO SPORT PET X 750 ML
```

**Regla**: un EAN por cada producto visualmente distinto. Si dos productos se ven diferente, necesitan EANs separados.

---

## Archivos de mapeo

El sistema usa dos archivos JSON que se generan automáticamente:

### `ean_class_map.json`

Mapea cada EAN a un nombre de clase visual para Roboflow:

```json
{
  "7750496": "pepsi_225",
  "7791813421719": "pepsi_15",
  "7791813423775": "pepsi_black_15",
  "7790895000997": "cocacola_225",
  "7790895000430": "cocacola_15",
  "7790895001130": "cocacola_zero_15",
  "7790315058201": "ean_7790315058201"
}
```

### `roboflow_label_map.json`

Mapea las clases que devuelve el modelo (numéricas: `"0"`, `"1"`, ...) al EAN y descripción correspondiente:

```json
{
  "0": { "ean": "7750496", "descripcion": "GASEOSA COLA REGULAR PEPSI PET X 2.25 LT" },
  "1": { "ean": "7791813421719", "descripcion": "GASEOSA COLA REGULAR PEPSI PET X 1.5 LT" },
  "2": { "ean": "7791813423775", "descripcion": "GASEOSA PEPSI BLACK PET X 1.5 LT" },
  "3": { "ean": "7790895000997", "descripcion": "GASEOSA COLA REGULAR COCA COLA PET X 2.25 LT" },
  "4": { "ean": "7790895000430", "descripcion": "GASEOSA COLA REGULAR COCA COLA PET X 1.5 LT" },
  "5": { "ean": "7790895001130", "descripcion": "GASEOSA COCA COLA ZERO PET X 1.5 LT" },
  "6": { "ean": null, "descripcion": "Botella genérica (sin EAN asignado)" },
  "7": { "ean": "7790315058201", "descripcion": "AGUA MINERAL VILLAVICENCIO SPORT PET X 750 ML" }
}
```

> Las clases `"6"` que tengan `ean: null` aparecen como `SIN_EAN_6` en el inventario.

Ambos archivos se regeneran automáticamente con:

```bash
python scripts/sync_roboflow_label_map.py --write
```

---

## Agregar un producto nuevo (flujo completo)

### Opción A — Script automático (recomendado)

Un solo comando que hace todo:

```bash
python scripts/agregar_producto_auto.py \
  --ean 7790315058201 \
  --descripcion "AGUA MINERAL VILLAVICENCIO SPORT PET X 750 ML"
```

Esto ejecuta:

1. Agrega el EAN a `eans.txt` (si no existe)
2. Descarga imágenes de referencia desde Bing
3. Sube las imágenes al dataset de Roboflow con anotaciones
4. Actualiza `roboflow_label_map.json`

Si además querés subir pre-anotaciones desde un video:

```bash
python scripts/agregar_producto_auto.py \
  --ean 7790315058201 \
  --descripcion "AGUA MINERAL VILLAVICENCIO SPORT PET X 750 ML" \
  --video data/IMG_2197.MOV
```

### Opción B — Paso a paso manual

```bash
# 1. Agregar línea a eans.txt (manualmente o con el script)
echo "7790315058201\tAGUA MINERAL VILLAVICENCIO SPORT PET X 750 ML" >> eans.txt

# 2. Sincronizar nuevos EANs al dataset Roboflow
python scripts/sync_eans_to_roboflow.py --per-ean 8

# 3. Regenerar el label map local
python scripts/sync_roboflow_label_map.py --write

# 4. En Roboflow: revisar anotaciones → generar nueva versión → reentrenar
# 5. Probar inferencia
python run.py data/MI_VIDEO.MOV --confianza 0.2
```

### Opción C — Sincronización incremental

Si agregaste varios EANs a `eans.txt` de una vez:

```bash
python scripts/sync_eans_to_roboflow.py --per-ean 8
```

Este comando detecta automáticamente los EANs nuevos comparando `eans.txt` con el estado guardado en `scripts/.eans_sync_state.json`.

Para ver qué haría sin ejecutar nada:

```bash
python scripts/sync_eans_to_roboflow.py --dry-run
```

---

## Subir imágenes a Roboflow

El script `scripts/upload_to_roboflow.py` soporta tres modos:

### Modo catálogo — subir imágenes de referencia

```bash
python scripts/upload_to_roboflow.py \
  --modo catalogo \
  --proyecto gondolacarrefour/gondola-dataset \
  --solo-eans 7790895000997,7790895000430
```

Sube las imágenes de `imagenes/<EAN>/` con anotaciones full-frame automáticas.

### Modo frames — pre-anotar desde video

```bash
python scripts/upload_to_roboflow.py \
  --modo frames \
  --video data/IMG_2196.MOV \
  --proyecto gondolacarrefour/gondola-dataset \
  --fps 1.0 \
  --confianza 0.2
```

Extrae frames del video, corre inferencia con el modelo actual, y sube frames + predicciones como pre-anotaciones para revisión en Roboflow.

### Modo imágenes — subir desde reportes existentes

```bash
python scripts/upload_to_roboflow.py \
  --modo imagenes \
  --proyecto gondolacarrefour/gondola-dataset \
  --imagenes-dir output
```

Sube imágenes de `output/*/reporte_deteccion` con las detecciones ya hechas como pre-anotaciones.

> **Nota**: Roboflow deduplica imágenes por **contenido** (hash). Si subís la misma imagen dos veces, no se crea un duplicado aunque el nombre sea distinto.

---

## Sincronizar label map

Cuando cambian las clases en el modelo (por ejemplo, después de reentrenar con nuevos productos):

```bash
# Ver preview de los cambios
python scripts/sync_roboflow_label_map.py

# Escribir los cambios
python scripts/sync_roboflow_label_map.py --write
```

Esto regenera `roboflow_label_map.json` a partir de `eans.txt` y `ean_class_map.json`.

---

## Reentrenar el modelo en Roboflow

Después de subir imágenes nuevas:

1. Ir a [https://app.roboflow.com/gondolacarrefour](https://app.roboflow.com/gondolacarrefour)
2. Entrar al proyecto `gondola-dataset`
3. En **Annotate**: revisar y corregir anotaciones de las imágenes nuevas
4. En **Generate**: crear una nueva versión del dataset
5. En **Train**: lanzar entrenamiento (o usar Roboflow Train)
6. Una vez entrenado, **publicar el workflow** actualizado
7. Probar localmente:

```bash
python run.py data/IMG_2196.MOV --confianza 0.2
```

---

## Confianza del modelo

La **confianza** (`--confianza`) es un valor entre 0 y 1 que filtra las detecciones del modelo.

| Valor | Efecto |
|---|---|
| `0.5` - `1.0` | Solo detecciones muy seguras. Puede perder productos reales (falsos negativos). |
| `0.2` - `0.4` | Balance entre precisión y cobertura. **Recomendado para producción.** |
| `0.05` - `0.2` | Detecta más productos, pero puede incluir falsos positivos. |

### Valor actual recomendado: `0.2`

Usamos `--confianza 0.2` porque el modelo todavía está en fase de entrenamiento y con un threshold bajo captura más detecciones reales.

**Importante**: la confianza también se configura **dentro del workflow de Roboflow**. Si en el workflow el nodo "Object Detection Model" tiene un `Confidence` alto (ej: 0.4), las predicciones que estén por debajo de ese umbral **nunca llegan** al programa, sin importar qué valor pongas en `--confianza`. Asegurate de que el threshold en el workflow sea **igual o menor** que el que usás en `--confianza`.

Para configurar en Roboflow:

1. Ir a **Workflows** → seleccionar tu workflow
2. Click en el nodo **"Object Detection Model"**
3. Bajar **Confidence** a `0.2` (o al valor deseado)
4. **Publicar** el workflow (botón "Deploy" o "Publish")

---

## Diferencia entre ROBOFLOW_PROJECT y ROBOFLOW_WORKFLOW

Estos dos valores se confunden frecuentemente pero son **cosas distintas**:

| | `ROBOFLOW_PROJECT` | `ROBOFLOW_WORKFLOW` |
|---|---|---|
| **Qué es** | El dataset/proyecto donde se guardan imágenes y anotaciones | El pipeline de inferencia que procesa imágenes |
| **Para qué se usa** | Subir imágenes (`upload_to_roboflow.py`) | Correr detecciones (`run.py`) |
| **Formato** | `workspace/project-slug` | Solo el `workflow_id` |
| **Ejemplo** | `gondolacarrefour/gondola-dataset` | `custom-workflow-2` |
| **Dónde se encuentra** | URL del proyecto en Roboflow | Roboflow → Workflows → nombre del workflow en la URL |

### Cómo encontrar tu `workflow_id`

1. Ir a [https://app.roboflow.com](https://app.roboflow.com)
2. Ir a **Workflows** (menú lateral)
3. Abrir tu workflow
4. En la URL del navegador verás algo como: `https://app.roboflow.com/gondolacarrefour/workflows/custom-workflow-2`
5. El `workflow_id` es la última parte: **`custom-workflow-2`**

### Error común: HTTP 404 en inferencia

Si ves este error:

```
Error HTTP 404: 404 Client Error: Not Found for url:
https://serverless.roboflow.com/infer/workflows/gondolacarrefour/gondola-dataset
```

Significa que `ROBOFLOW_WORKFLOW` tiene el valor del **proyecto** en vez del **workflow**. Corregí `.env`:

```env
# ❌ Incorrecto
ROBOFLOW_WORKFLOW=gondola-dataset

# ✅ Correcto
ROBOFLOW_WORKFLOW=custom-workflow-2
```

---

## Output del sistema

Cada ejecución crea una carpeta en `output/` con esta estructura:

```text
output/IMG_2196_20260218_215836/
├── analisis_video.json              # Metadata del video (fps, resolución, duración)
├── frames_extraidos/                # Frames crudos extraídos
│   ├── frame_0001_t0.00s.jpg
│   ├── frame_0002_t1.00s.jpg
│   └── ...
├── crops/                           # (si --guardar-crops) Recorte por detección
└── reporte_deteccion/
    ├── inventario_sku.csv           # Conteo final por EAN
    ├── frame_0001_t0.00s_detectado.jpg   # Frame con bounding boxes dibujados
    └── ...
```

### Formato de `inventario_sku.csv`

```csv
EAN,Cantidad,Fecha
7790895000430,10,2026-02-18 21:59:08
7790895000997,5,2026-02-18 21:59:08
7790895001130,58,2026-02-18 21:59:08
SIN_EAN_6,20,2026-02-18 21:59:08
```

- **EAN**: Código del producto (o `SIN_EAN_X` si la clase no tiene EAN asignado)
- **Cantidad**: Número de veces que se detectó en todos los frames
- **Fecha**: Timestamp de la ejecución

---

## Troubleshooting

### `Falta API key`

Definila en `.env`:

```env
ROBOFLOW_API_KEY=TU_API_KEY
```

O pasala por CLI:

```bash
python run.py data/VIDEO.MOV --roboflow-api-key TU_API_KEY
```

### HTTP 404 en inferencia

Ver sección [Diferencia entre ROBOFLOW_PROJECT y ROBOFLOW_WORKFLOW](#diferencia-entre-roboflow_project-y-roboflow_workflow).

### 0 detecciones (el modelo no detecta nada)

Causas posibles:

1. **Confidence muy alto en el workflow de Roboflow**: bajalo a 0.2 desde la UI de Workflows y republicá.
2. **Modelo no entrenado** con los productos del video.
3. **Workflow no publicado**: después de cambiar parámetros, hacé click en "Deploy/Publish".

### `SIN_EAN_X` en el inventario

Significa que el modelo detectó una clase (ej: `6`) que no tiene EAN asignado en `roboflow_label_map.json`.

Solución:

```bash
python scripts/sync_roboflow_label_map.py --write
```

Si la clase es nueva (agregaste un producto), primero hay que:

1. Agregar el EAN a `eans.txt`
2. Correr `python scripts/sync_eans_to_roboflow.py`
3. Reentrenar el modelo en Roboflow

### `Endpoint not found` al subir dataset

Los scripts ya normalizan automáticamente el slug del proyecto. Si persiste, verificá que el nombre del proyecto en Roboflow coincida con `ROBOFLOW_PROJECT` en `.env`.

### Imágenes no se agregan a Roboflow (duplicados)

Roboflow deduplica imágenes por **contenido** (hash del archivo). Si subís la misma imagen con distinto nombre, Roboflow la detecta como duplicada y no la agrega.

Para agregar variantes nuevas, podés:

- Usar imágenes de referencia diferentes
- Generar variantes augmentadas (brillo, contraste, blur) que tengan contenido distinto

### `ModuleNotFoundError: No module named 'cv2'`

```bash
source venv/bin/activate
pip install opencv-python
```

---

## Productos actuales en el modelo

| Clase | EAN | Producto |
|---|---|---|
| 0 | 7750496 | Pepsi Regular 2.25L |
| 1 | 7791813421719 | Pepsi Regular 1.5L |
| 2 | 7791813423775 | Pepsi Black 1.5L |
| 3 | 7790895000997 | Coca-Cola Regular 2.25L |
| 4 | 7790895000430 | Coca-Cola Regular 1.5L |
| 5 | 7790895001130 | Coca-Cola Zero 1.5L |
| 6 | — | Botella genérica (sin EAN) |
| 7 | 7790315058201 | Villavicencio Sport 750ml |

---

## Versión

- **Versión**: 5.0 (Roboflow Only)
- **Última actualización**: Febrero 2026
- **Confianza recomendada**: `0.2`
- **Workflow activo**: `custom-workflow-2`
