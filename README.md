Abajo te dejo el **README actualizado para la nueva versión (v5.4 / Sprint 2.3)**, alineado con tu **YOLO retail-ready (sin COCO mapping)** + **categorización por packaging (CLIP)** + **identificación SKU (CLIP embeddings)** + **SQL Server opcional**.
Además agrego una sección completa de **entrenamiento del detector** (YOLO 1-clase `product`) y una sección de **“Nuevos pasos a hacer”** (qué cambia y qué tenés que ejecutar/ajustar).

> Nota: Mantengo el documento “copiable” como README.md.

---

# Sistema de Inventario de Góndolas — Sprint 2.1 (v5.5)

> **⭐ NUEVO**: Sistema de Aprendizaje Continuo implementado. El sistema ahora mejora automáticamente con cada ejecución.  
> Ver [LEARNING_SYSTEM.md](LEARNING_SYSTEM.md) para documentación completa del sistema de aprendizaje.

Sistema de visión artificial para **detectar, identificar y contar** productos en góndolas de supermercado a partir de videos.

**Arquitectura (2 capas + categorización):**

* **Capa A — Detección genérica (YOLOv8, local)**: detecta **instancias de producto** (no SKUs, no COCO mapping).
* **Categorización por packaging (CLIP zero-shot)**: clasifica el tipo de envase (botella, lata, bolsa, caja, etc.) para **filtrar el espacio de búsqueda**.
* **Capa B — Identificación SKU (CLIP embeddings)**: compara el embedding del crop contra el catálogo (filtrado por categoría) y devuelve **EAN + confianza**.

✅ **Objetivo retail real:** separar “dónde hay un producto” (estable) de “qué SKU es” (dinámico).
✅ **Escala:** agregar SKUs no requiere reentrenar (solo sumar imágenes + embeddings).
✅ **Offline:** todo corre local (SQL Server es opcional).

---

## Tabla de Contenidos

1. [Cambio de arquitectura](#cambio-de-arquitectura)
2. [Cómo funciona](#cómo-funciona)
3. [Estructura del proyecto](#estructura-del-proyecto)
4. [Requisitos previos](#requisitos-previos)
5. [Instalación](#instalación)
6. [Uso principal](#uso-principal--procesar-un-video)
7. [Agregar un SKU nuevo](#agregar-un-sku-nuevo-5-minutos)
8. [Categorización por packaging](#categorización-por-packaging)
9. [Detector YOLO retail-ready (Capa A)](#detector-yolo-retail-ready-capa-a)
10. [Identificador SKU (Capa B)](#identificador-sku-capa-b)
11. [Pipeline + deduplicación](#pipeline--deduplicación-por-frame)
12. [Sistema de Aprendizaje Continuo](#sistema-de-aprendizaje-continuo-dataset-evolutivo--v21) ⭐ **NUEVO**
13. [SQL Server (opcional)](#sql-server-opcional)
14. [Entrenar el detector YOLO (1 clase product)](#entrenar-el-detector-yolo-1-clase-product)
15. [Migración a v5.4](#migración-a-v54-qué-cambió-y-qué-hacer)
16. [Roadmap](#roadmap-sprints-37)
17. [Troubleshooting](#troubleshooting)
18. [Versión](#versión)

---

## Cambio de arquitectura

### Sprint 1 (anterior)

```
Video → Roboflow API (detección + clasificación) → EAN → Reporte
```

* Cada SKU nuevo requería: subir imágenes → anotar → reentrenar → esperar → probar.
* **Tiempo por SKU: 3–6 horas.**
* Dependencia total de API externa.

### Sprint 2 (actual)

```
Video → YOLO local (detección genérica) → Crops → CLIP (embeddings)
     → (packaging) → Búsqueda vectorial → EAN → Reporte
```

* Agregar un SKU = agregar imágenes + recalcular embeddings.
* **Tiempo por SKU: 5 minutos.**
* Sin reentrenamiento por SKU.
* El detector YOLO se reentrena **solo si** querés mejorar la detección.

---

## Cómo funciona

1. **Extraer frames** del video (`--fps` configurable)
2. **Detectar instancias de producto** (YOLOv8 local, genérico)
3. **Recortar crops** (padding dinámico, ROI opcional)
4. **Clasificar packaging** (CLIP zero-shot) → filtrar candidatos
5. **Identificar SKU** (CLIP embedding + búsqueda por similitud)
6. **Guardar review** (unknown/ambiguous) para auto-mejora
7. **Reporte**: CSV + frames anotados (+ DB opcional)

---

## Estructura del proyecto

```
dinamic-carrefour/
├── run.py
├── eans.txt
├── requirements.txt
├── .env (opcional)
│
├── src/
│   ├── analizar_video.py
│   ├── protocols.py
│   ├── detector/
│   │   └── yolo_detector.py          # YOLO retail-ready (sin COCO mapping)
│   ├── sku_identifier/
│   │   ├── embedder.py               # CLIP embeddings
│   │   ├── categorizer.py            # packaging zero-shot (CLIP)
│   │   ├── vector_store.py           # búsqueda vectorial (filtra por categoría)
│   │   └── identifier.py             # decisión (matched/unknown/ambiguous)
│   ├── pipeline/
│   │   └── engine.py                 # video → frames → detección → identificación → reporte
│   └── database/
│       ├── schema.sql
│       ├── connection.py
│       └── repository.py
│
├── scripts/
│   ├── buscarimagenes.py
│   ├── agregar_sku.py
│   └── init_db.py
│
├── imagenes/<EAN>/
├── catalog/embeddings/<EAN>.npy
├── review/
└── output/
```

---

## Requisitos previos

* **Python 3.10+**
* **8 GB RAM** mínimo
* GPU (CUDA) opcional (recomendado)
* **Ultralytics YOLOv8**
* **OpenAI CLIP** (`openai-clip`)

---

## Instalación

```bash
git clone <URL_DEL_REPO>
cd dinamic-carrefour

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Primera ejecución descarga y cachea:

* YOLOv8n (~6 MB)
* CLIP ViT-B/32 (~340 MB)

---

## Uso principal — procesar un video

```bash
python run.py data/IMG_2196.MOV
```

Recomendado retail:

```bash
python run.py data/IMG_2196.MOV \
  --fps 1.0 \
  --confianza 0.15 \
  --det-iou 0.60 \
  --imgsz 960 \
  --max-det 300
```

Con ROI (muy útil para ignorar piso/techo/reflejos):

```bash
python run.py data/IMG_2196.MOV \
  --roi 0.05,0.10,0.95,0.98
```

Verbose (top-3 candidatos por crop):

```bash
python run.py data/IMG_2196.MOV --verbose
```

---

## Agregar un SKU nuevo (5 minutos)

Formato `eans.txt`:

```
EAN<TAB>DESCRIPCION<TAB>CATEGORIA
```

Ejemplo:

```bash
echo "7790315058201	AGUA MINERAL VILLAVICENCIO SPORT PET X 750 ML	botella" >> eans.txt
```

Luego:

```bash
python scripts/agregar_sku.py --ean 7790315058201
```

Si querés descargar imágenes automáticamente:

```bash
python scripts/agregar_sku.py \
  --ean 7790315058201 \
  --descripcion "AGUA MINERAL VILLAVICENCIO SPORT PET X 750 ML" \
  --descargar-imagenes
```

✅ Resultado:

* `imagenes/<EAN>/...`
* `catalog/embeddings/<EAN>.npy`

---

## Categorización por packaging

El sistema primero predice packaging (CLIP zero-shot).
Luego busca el SKU **solo** dentro de la misma categoría, con fallback automático:

* Si detecta `lata` pero no hay latas en el catálogo → busca en todo el catálogo.

Desactivar categorización:

```bash
python run.py data/IMG_2196.MOV --sin-categorias
```

---

## Detector YOLO retail-ready (Capa A)

**Ubicación:** `src/detector/yolo_detector.py`

### Cambio clave (v5.4)

✅ **Ya NO se usa COCO mapping** ni `categoria_coco_mapeo`.
✅ YOLO corre **sin filtrar por clases** y devuelve detecciones normalizadas como `clase="product"`.

**MVP (hoy):**

* Podés usar `yolov8n.pt` como “detector genérico”
* Ajustás `--confianza`, `--imgsz`, `--roi`

**Evolución recomendada:**

* entrenar un modelo YOLO con **1 clase**: `product`
* el pipeline no cambia: solo apuntás `--modelo-yolo runs/.../best.pt`

### Heurísticas baratas (retail real)

* filtro por **área relativa**
* filtro por **aspect ratio**
* **ROI** opcional
* **padding dinámico** para mejorar crops de CLIP

---

## Identificador SKU (Capa B)

* **CLIP (ViT-B/32)** genera embedding del crop (512D).
* **VectorStore** busca por similitud coseno.
* **Max-Similarity**: compara contra **todas** las imágenes del SKU (no solo centroide).

Decisión:

* `top1 ≥ sku_threshold` → MATCHED
* `top1 < unknown_threshold` → UNKNOWN (va a review)
* `top1 - top2 < margen_ambiguedad` → AMBIGUOUS (va a review)

---

## Pipeline + deduplicación por frame

**Deduplicación (v5.1+)**
En vez de sumar detecciones por todos los frames, el conteo final por SKU es:

> **máximo conteo observado en un frame**

Esto representa la “mejor vista” de la góndola y evita inflar conteos.

---

## Sistema de Aprendizaje Continuo (Dataset Evolutivo) — v2.1

> **📚 Documentación completa**: Ver [LEARNING_SYSTEM.md](LEARNING_SYSTEM.md)

El sistema ahora implementa un **loop de mejora automática** que permite que el sistema mejore con cada ejecución, sin necesidad de reentrenamientos.

### ¿Cómo funciona?

1. **Cada ejecución genera un dataset estructurado**:
   - Crops dudosos (UNKNOWN, AMBIGUOUS) se guardan automáticamente
   - Metadata completa de cada decisión (detection, packaging, SKU identification)
   - Información lista para revisión humana

2. **Revisión rápida** (5-10 minutos):
   ```bash
   python scripts/revisar_crops.py output/VIDEO_TIMESTAMP/learning
   ```
   - Asignar EAN correcto a cada crop
   - Guardar cambios en metadata

3. **Absorción automática**:
   ```bash
   python scripts/absorber_crops.py output/VIDEO_TIMESTAMP/learning
   ```
   - Copia crops al catálogo del SKU correcto
   - Recalcula embeddings automáticamente
   - Actualiza el vector store

4. **Siguiente ejecución mejora automáticamente**:
   - Más imágenes de referencia = embeddings más representativos
   - Menos UNKNOWN, menos confusiones

### Estructura generada

```
output/<video_timestamp>/
    learning/
        unknown/              # Crops no identificados
        ambiguous/             # Crops con top1≈top2
        metadata/
            execution_meta.json
            crops_index.jsonl   # Metadata completa
```

### Beneficios

✅ **Sin reentrenamientos**: El sistema mejora solo agregando crops al catálogo  
✅ **Escalable**: Cada ejecución genera datos valiosos  
✅ **Rápido**: Revisión típica: 5-10 minutos  
✅ **Medible**: Métricas de evolución (UNKNOWN%, similitud promedio, etc.)

### Uso rápido

```bash
# 1. Ejecutar pipeline (guarda crops automáticamente)
python run.py data/video.MOV

# 2. Revisar crops dudosos
python scripts/revisar_crops.py output/VIDEO_TIMESTAMP/learning

# 3. Absorber crops al catálogo
python scripts/absorber_crops.py output/VIDEO_TIMESTAMP/learning
```

**Resultado**: El sistema mejora progresivamente sin intervención técnica.

---

## Sistema de auto-mejora (review) — Legacy

Los crops con:

* baja similitud → `UNKNOWN`
* top1≈top2 → `AMBIGUOUS`

se guardan en `review/` con `_meta.json` (top-k + scores).
Luego:

1. los etiquetás
2. los movés a `imagenes/<EAN>/`
3. recalculás embeddings

---

## SQL Server (opcional)

Se activa con:

```bash
python run.py data/IMG_2196.MOV --db
```

Setup:

```bash
python scripts/init_db.py --test
python scripts/init_db.py --crear
python scripts/init_db.py --sync
python scripts/init_db.py --status
```

---

# Entrenar el detector YOLO (1 clase `product`)

## Objetivo del entrenamiento

Pasar de “YOLO genérico (COCO)” a un detector específico de góndolas:

✅ 1 clase: `product`
✅ mejor recall en góndola
✅ menos falsos positivos (manos/reflejos/carteles)

---

## Paso 1 — Crear dataset

### Recomendado (rápido): Roboflow / CVAT / Label Studio

* Tomá frames reales del video (o fotos de góndola)
* Anotá **cada producto visible** con bounding box
* Exportá formato **YOLOv8** (o YOLO)

Estructura esperada:

```
datasets/shelf-products/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

`data.yaml`:

```yaml
path: datasets/shelf-products
train: train/images
val: valid/images
test: test/images

names:
  0: product
```

---

## Paso 2 — Entrenar con Ultralytics

### Entrenamiento base (recomendado)

```bash
yolo detect train \
  model=yolov8n.pt \
  data=datasets/shelf-products/data.yaml \
  imgsz=960 \
  epochs=80 \
  batch=8 \
  device=auto
```

### Tips de entrenamiento (retail real)

* `imgsz=960` o `1280` mejora detección de productos chicos
* si tenés GPU: subí batch
* si tenés pocos datos: empezá con `yolov8n.pt`, luego `yolov8s.pt`

---

## Paso 3 — Validar resultados

El entrenamiento genera:

```
runs/detect/train/
├── weights/best.pt
└── results.png
```

Probá inferencia rápida:

```bash
yolo detect predict \
  model=runs/detect/train/weights/best.pt \
  source=data/frames_test/ \
  imgsz=960 \
  conf=0.15
```

---

## Paso 4 — Usar el modelo entrenado en el pipeline

```bash
python run.py data/IMG_2196.MOV \
  --modelo-yolo runs/detect/train/weights/best.pt \
  --confianza 0.15 \
  --det-iou 0.60 \
  --imgsz 960
```

✅ El resto del sistema no cambia.

---

# Migración a v5.4: qué cambió y qué hacer

## Cambios principales

1. **Se elimina COCO mapping**

   * ya no existe `categoria_coco_mapeo`
   * el detector NO carga clases desde DB
   * YOLO detecta “todo lo que parezca producto” y normaliza `clase="product"`

2. **El detector ahora tiene parámetros retail**

   * `--det-iou`, `--imgsz`, `--max-det`, `--roi`, `--device`, `--half`

3. **El pipeline prioriza procesar y descartar crops**

   * evita acumular crops en RAM
   * (si `--guardar-crops`) guarda en disco para debug/review

## Pasos para actualizar tu repo

1. Reemplazar `src/detector/yolo_detector.py` por la versión retail-ready (v5.4)
2. Asegurarte que **no exista ninguna referencia** a:

   * `categoria_coco_mapeo`
   * `obtener_mapeo_coco()`
   * `cargar_clases_desde_bd()`
3. Actualizar `run.py` para incluir flags retail (si no los tenías):

   * `--det-iou`, `--imgsz`, `--max-det`, `--roi`, `--device`, `--half`
4. Si usás DB:

   * mantener catálogo/categorías/ejecuciones/detecciones
   * **no** crear tabla de mapping COCO (ya no aplica)

## “Nuevos pasos a hacer” (operativo)

1. **Elegir ROI estándar por tipo de video** (reduce falsos positivos muchísimo)
2. **Ajustar defaults retail**

   * `conf=0.15`, `imgsz=960`, `iou=0.60`
3. **Cargar 50–200 frames y anotar** para dataset `product` (Sprint 2.4)
4. Entrenar YOLO 1-clase y usar `best.pt`
5. Implementar revisión rápida de `review/` (script simple) para absorber crops reales al catálogo
6. (Opcional) empezar a preparar Sprint 3 (tracking real para conteo por unidad)

---

## Roadmap (Sprints 3–7)

* **Sprint 3**: Tracking (ByteTrack/SORT) → conteo real por unidad (IDs persistentes)
* **Sprint 4**: Conteo por estantes/zonas + sampling inteligente
* **Sprint 5**: Keyframes + consolidación espacial (IoU entre keyframes)
* **Sprint 6**: Planograma + faltantes + productos fuera de lugar
* **Sprint 7**: Escala (batch workers + drones + sucursal)

---

## Troubleshooting

### YOLO no detecta productos

* bajá `--confianza 0.15`
* subí `--imgsz 960`
* usá `--roi ...`
* probá `yolov8s.pt`
* si sigue flojo → entrenar 1-clase `product`

### Todo da UNKNOWN

* faltan embeddings `.npy`
* corré: `python scripts/agregar_sku.py --todos` o por EAN
* bajá `--sku-threshold` para diagnosticar (`0.65`)

---

## Versión

* **Versión**: **5.5** (Sprint 2.1 — Sistema de Aprendizaje Continuo)
* **Última actualización**: **19 Febrero 2026**
* **Inferencia externa**: ninguna
* **Catálogo actual**: 9 SKUs
* **Packaging**: botella, lata, bolsa, caja, paquete, tubo, frasco

### Changelog v5.5 (Sistema de Aprendizaje Continuo)

* **Learning Manager**: Captura automática de crops dudosos por ejecución
* **Dataset evolutivo**: Cada ejecución genera metadata estructurada en `learning/`
* **Script de revisión**: `scripts/revisar_crops.py` para revisión rápida CLI
* **Script de absorción**: `scripts/absorber_crops.py` para absorber crops al catálogo
* **Loop de mejora**: Sistema mejora automáticamente sin reentrenamientos
* **Metadata completa**: JSONL con toda la información de cada decisión (detection, packaging, SKU identification)
* **Integración automática**: Learning Manager se activa automáticamente en el pipeline

### Changelog v5.4

* **YOLO retail-ready**: detección genérica sin COCO mapping
* **ROI + heurísticas**: filtros baratos (área/aspect) + padding dinámico
* **CLI extendida**: `--det-iou`, `--imgsz`, `--max-det`, `--roi`, `--device`, `--half`
* **Preparado para YOLO 1-clase**: mismo pipeline, solo cambia el modelo

---

Si querés, pegame tu `README.md` actual como archivo o texto y te lo devuelvo **en formato “diff”** (qué líneas borrar/agregar) para que lo puedas mergear sin perder nada de lo que ya tenías.
