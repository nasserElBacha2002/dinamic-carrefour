# Sistema de Inventario Inteligente — v6.0

> **🎯 Nuevo Enfoque**: Sistema de identificación y conteo de productos para **depósitos y almacenes**, con soporte para unidades logísticas (pack/box/pallet) e identificación híbrida (barcode/OCR/visual).

Sistema de visión artificial para **detectar, identificar y contar** productos en videos de depósitos y almacenes, con tracking temporal y aprendizaje continuo.

**Arquitectura (3 capas + tracking + aprendizaje):**

* **Capa A — Detección genérica (YOLOv8, local)**: detecta **instancias de producto** (no SKUs específicos).
* **Categorización por packaging (CLIP zero-shot)**: clasifica el tipo de envase (botella, lata, bolsa, caja, etc.) para **filtrar el espacio de búsqueda**.
* **Capa B — Identificación SKU (CLIP embeddings)**: compara el embedding del crop contra el catálogo (filtrado por categoría) y devuelve **EAN + confianza**.
* **Capa C — Tracking temporal (SORT-like)**: asigna IDs persistentes a productos a lo largo de frames, permitiendo **conteo por unidad** y **consolidación de decisiones**.
* **Aprendizaje continuo**: sistema evolutivo que mejora automáticamente con cada ejecución.

✅ **Objetivo warehouse/depot:** identificación y conteo preciso en entornos de depósito, con soporte para unidades logísticas.  
✅ **Escala:** agregar SKUs no requiere reentrenar (solo sumar imágenes + embeddings).  
✅ **Offline:** todo corre local (SQL Server es opcional).  
✅ **Tracking:** conteo por unidad con IDs persistentes, eliminando duplicados temporales.

---

## Tabla de Contenidos

1. [Objetivo y Enfoque](#objetivo-y-enfoque)
2. [Arquitectura](#arquitectura)
3. [Cómo funciona](#cómo-funciona)
4. [Estructura del proyecto](#estructura-del-proyecto)
5. [Requisitos previos](#requisitos-previos)
6. [Instalación](#instalación)
7. [Uso principal](#uso-principal--procesar-un-video)
8. [Tracking temporal (Sprint 3.1)](#tracking-temporal-sprint-31) ⭐ **NUEVO**
9. [Agregar un SKU nuevo](#agregar-un-sku-nuevo-5-minutos)
10. [Sistema de Aprendizaje Continuo](#sistema-de-aprendizaje-continuo-dataset-evolutivo) ⭐
11. [UI de Revisión y Gestión](#ui-de-revisión-y-gestión) ⭐ **NUEVO**
12. [SQL Server (opcional)](#sql-server-opcional)
13. [Troubleshooting](#troubleshooting)
14. [Roadmap](#roadmap-sprints-4-7)
15. [Versión](#versión)

---

## Objetivo y Enfoque

### Evolución del Proyecto

**Sprint 1-2 (Góndolas de supermercado):**
- Enfoque inicial en góndolas de retail
- Conteo por frame con deduplicación
- Identificación visual pura (CLIP)

**Sprint 3+ (Depósitos y almacenes):**
- **Pivot a escenario warehouse/depot**
- **Tracking temporal** para conteo por unidad
- **Unidades logísticas** (pack/box/pallet) - en desarrollo
- **Identificación híbrida** (barcode/OCR/visual) - en desarrollo
- **Consolidación de decisiones** por track (votación temporal)

### Casos de Uso Actuales

1. **Inventario de depósito**: Conteo preciso de productos en almacenes
2. **Tracking de unidades**: Seguimiento de productos individuales a lo largo del tiempo
3. **Aprendizaje continuo**: Mejora automática del sistema con cada ejecución
4. **Revisión asistida**: UI para revisar y corregir identificaciones dudosas

### Casos de Uso Futuros (Sprint 4+)

1. **Conteo por unidades logísticas**: Pack, box, pallet
2. **Identificación híbrida**: Barcode + OCR + visual
3. **Planograma de depósito**: Verificación de ubicación correcta
4. **Detección de faltantes**: Productos ausentes en ubicaciones esperadas

---

## Arquitectura

### Flujo Principal

```
Video → Frames → Detección (YOLO) → Tracking (SORT-like) → Identificación (CLIP)
  → Consolidación por Track → Conteo por Unidad → Reporte
```

### Componentes Clave

1. **Pipeline Engine** (`src/pipeline/core/engine.py`): Orquesta todo el flujo
2. **Tracking Runtime** (`src/pipeline/tracking/track_runtime.py`): Maneja tracking por frame
3. **Track Vote Accumulator** (`src/tracking/track_vote_accumulator.py`): Consolida decisiones por track
4. **Detection Processor** (`src/pipeline/processing/detection_processor.py`): Procesa detecciones con política de decisión
5. **Learning Integration** (`src/pipeline/output/learning_integration.py`): Captura crops dudosos para aprendizaje
6. **UI de Revisión** (`src/ui/app.py`): Interfaz web para revisar y corregir

---

## Cómo funciona

### Procesamiento con Tracking (Recomendado)

1. **Extraer frames** del video (`--fps` configurable)
2. **Detectar instancias de producto** (YOLOv8 local, genérico)
3. **Tracking temporal** (SORT-like): asigna IDs persistentes a productos
4. **Filtrar detecciones** por área, aspecto y bordes
5. **Identificar SKU** (CLIP embedding + búsqueda por similitud) - **una vez por detección**
6. **Acumular votos** por track (cada frame vota por un EAN)
7. **Finalizar tracks** y consolidar decisiones (mayoría + confianza)
8. **Conteo por unidad**: 1 track finalizado = 1 producto contado
9. **Guardar crops dudosos** (unknown/ambiguous) para auto-mejora
10. **Reporte**: CSV por track + CSV por frame + frames anotados

### Procesamiento sin Tracking (Legacy)

1. **Extraer frames** del video
2. **Detectar instancias de producto**
3. **Identificar SKU** por frame
4. **Deduplicación por frame**: máximo conteo observado
5. **Reporte**: CSV por frame

---

## Estructura del proyecto

```
dinamic-carrefour/
├── run.py                          # Script principal (legacy)
├── src/main.py                     # Punto de entrada principal
├── start_ui.py                     # Iniciar UI de revisión
├── eans.txt                        # Catálogo de productos
├── requirements.txt
├── .env (opcional)
│
├── src/
│   ├── main.py                     # CLI principal
│   ├── analizar_video.py
│   ├── protocols.py
│   │
│   ├── detector/
│   │   └── yolo_detector.py        # YOLO retail-ready
│   │
│   ├── sku_identifier/
│   │   ├── embedder.py             # CLIP embeddings
│   │   ├── categorizer.py          # packaging zero-shot (CLIP)
│   │   ├── vector_store.py         # búsqueda vectorial
│   │   └── identifier.py           # decisión (matched/unknown/ambiguous)
│   │
│   ├── tracking/
│   │   ├── tracker_base.py         # Interfaz base
│   │   ├── sort_like_tracker.py    # Tracker SORT-like
│   │   ├── track_types.py          # Tipos de datos
│   │   └── track_vote_accumulator.py # Consolidación de votos
│   │
│   ├── pipeline/
│   │   ├── core/
│   │   │   ├── engine.py           # Pipeline principal
│   │   │   └── video_reader.py    # Lectura de frames
│   │   ├── processing/
│   │   │   ├── detection_processor.py # Procesamiento de detecciones
│   │   │   ├── crop_processor.py   # Heurísticas de crops
│   │   │   ├── decision_policy.py  # Política de decisión genérica
│   │   │   └── bbox_quality.py     # Scorer de calidad de bbox
│   │   ├── tracking/
│   │   │   ├── track_setup.py      # Configuración de tracking
│   │   │   ├── track_runtime.py    # Runtime de tracking
│   │   │   ├── track_exporter.py   # Exportación de resultados
│   │   │   └── track_integration.py # Utilidades de integración
│   │   └── output/
│   │       ├── report_generator.py # Generación de reportes
│   │       ├── result_builder.py   # Construcción de resultados
│   │       └── learning_integration.py # Integración con aprendizaje
│   │
│   ├── database/
│   │   ├── schema.sql
│   │   ├── connection.py
│   │   └── repository.py
│   │
│   ├── learning/
│   │   └── manager.py              # Learning Manager
│   │
│   └── ui/
│       ├── app.py                  # FastAPI app
│       ├── services/
│       │   ├── report.py           # Servicios de reporte
│       │   ├── review_store.py    # Almacenamiento de revisión
│       │   └── db.py              # Servicios de DB
│       ├── templates/              # Templates HTML
│       └── static/                 # CSS/JS
│
├── scripts/
│   ├── buscarimagenes.py
│   ├── agregar_sku.py
│   ├── init_db.py
│   ├── revisar_crops.py
│   └── absorber_crops.py
│
├── imagenes/<EAN>/                 # Imágenes de referencia
├── catalog/embeddings/<EAN>.npy    # Embeddings CLIP
├── output/<run_id>/                # Resultados de ejecución
│   ├── reporte_deteccion/
│   │   ├── inventario_por_track.csv
│   │   ├── inventario_por_frame.csv
│   │   └── frame_*.jpg
│   ├── learning/                   # Crops dudosos
│   └── track_summary.json
└── data/                            # Videos (ignorado en git)
```

---

## Requisitos previos

* **Python 3.10+**
* **8 GB RAM** mínimo (16 GB recomendado)
* **GPU (CUDA)** opcional pero recomendado
* **Ultralytics YOLOv8**
* **OpenAI CLIP** (`openai-clip`)

---

## Instalación

```bash
git clone <URL_DEL_REPO>
cd dinamic-carrefour

python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Primera ejecución descarga y cachea:

* YOLOv8n (~6 MB)
* CLIP ViT-B/32 (~340 MB)

---

## Uso principal — procesar un video

### Con tracking (Recomendado)

```bash
python src/main.py data/video.MOV --use-tracks
```

Con parámetros personalizados:

```bash
python src/main.py data/video.MOV \
  --use-tracks \
  --track-iou 0.35 \
  --track-min-hits 3 \
  --track-max-age 15 \
  --fps 1.0 \
  --confianza 0.15
```

### Sin tracking (Legacy)

```bash
python src/main.py data/video.MOV
```

### Parámetros comunes

```bash
python src/main.py data/video.MOV \
  --fps 1.0 \                    # FPS de extracción
  --confianza 0.15 \             # Confianza mínima YOLO
  --det-iou 0.60 \               # IoU para NMS
  --roi 0.05,0.10,0.95,0.98 \   # ROI normalizado (opcional)
  --verbose                      # Output detallado
```

---

## Tracking temporal (Sprint 3.1) ⭐ **NUEVO**

### ¿Qué es?

El sistema ahora implementa **tracking temporal** que asigna IDs persistentes a productos a lo largo de frames, permitiendo:

1. **Conteo por unidad**: 1 track = 1 producto (elimina duplicados temporales)
2. **Consolidación de decisiones**: Votación por mayoría a lo largo del tiempo
3. **Mejor precisión**: Decisiones más estables que identificación frame-by-frame

### Cómo funciona

1. **Asignación de IDs**: SORT-like tracker asigna IDs persistentes basado en IoU
2. **Acumulación de votos**: Cada frame vota por un EAN para cada track
3. **Finalización de tracks**: Cuando un track termina (max_age o video termina)
4. **Decisión final**: Mayoría de votos + reglas de confianza (DecisionProfile)

### Decision Profiles

El sistema soporta diferentes perfiles de decisión según el escenario:

- **`WAREHOUSE_BALANCED`** (default): Balance entre precisión y recall para depósitos
- **`WAREHOUSE_LENIENT`**: Más permisivo, acepta tracks cortos
- **`SHELF_STRICT`**: Más estricto, requiere más frames y mayor confianza

### Parámetros de tracking

```bash
--use-tracks              # Activar tracking
--track-iou 0.35         # IoU mínimo para matching (default: 0.4)
--track-min-hits 3       # Mínimo de hits para track válido (default: 3)
--track-max-age 15       # Máximo de frames sin detección (default: 15)
```

### Output con tracking

Cuando se activa tracking, se generan:

- `inventario_por_track.csv`: Conteo por unidad (1 track = 1 producto)
- `inventario_por_frame.csv`: Conteo por frame (comparación)
- `tracks_debug.csv`: Debug de tracking (frame, det_idx, track_id, IoU)
- `track_summary.json`: Resumen de tracks finalizados

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

## Sistema de Aprendizaje Continuo (Dataset Evolutivo)

> **📚 Documentación completa**: Ver [LEARNING_SYSTEM.md](LEARNING_SYSTEM.md) (si existe)

El sistema implementa un **loop de mejora automática** que permite que el sistema mejore con cada ejecución, sin necesidad de reentrenamientos.

### ¿Cómo funciona?

1. **Cada ejecución genera un dataset estructurado**:
   - Crops dudosos (UNKNOWN, AMBIGUOUS) se guardan automáticamente
   - Metadata completa de cada decisión (detection, packaging, SKU identification)
   - Información lista para revisión humana

2. **Revisión rápida** (5-10 minutos):
   - Usar la UI web (recomendado) o CLI
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

---

## UI de Revisión y Gestión ⭐ **NUEVO**

### Iniciar la UI

```bash
python start_ui.py
```

O directamente:

```bash
uvicorn src.ui.app:app --reload --port 8000
```

Luego abrir: `http://localhost:8000`

### Funcionalidades

1. **Home**: Lista de videos y runs existentes
2. **Reporte del run**: 
   - Tabla del CSV de inventario
   - Descarga de CSV
   - Grid de frames anotados
   - Botón para revisar crops
3. **Revisión**:
   - Ver cada crop dudoso
   - Ver candidatos (top matches)
   - Asignar EAN o saltar
   - Autocomplete desde DB
   - Progreso de revisión
   - Botón para absorber cambios
4. **Absorción**: Ejecuta el script de absorción y muestra logs

### Workflow recomendado

1. Ejecutar pipeline: `python src/main.py data/video.MOV --use-tracks`
2. Abrir UI: `python start_ui.py`
3. Ir al run recién creado
4. Click en "Revisar" si hay crops dudosos
5. Revisar y asignar EANs
6. Click en "Absorber" cuando termines
7. El sistema mejora automáticamente

---

## SQL Server (opcional)

Se activa automáticamente si está disponible. Para configurar:

```bash
python scripts/init_db.py --test
python scripts/init_db.py --crear
python scripts/init_db.py --sync
python scripts/init_db.py --status
```

---

## Troubleshooting

### YOLO no detecta productos

* bajá `--confianza 0.15`
* subí `--imgsz 960`
* usá `--roi ...`
* probá `yolov8s.pt`
* si sigue flojo → entrenar 1-clase `product`

### Todo da UNKNOWN

**Causas comunes**:

1. **Faltan embeddings**:
   ```bash
   python scripts/agregar_sku.py --todos --forzar
   ```

2. **Mismatch de modelo CLIP**:
   ```bash
   export CLIP_MODEL="ViT-B/16"  # o el que uses
   python scripts/agregar_sku.py --todos --forzar
   python src/main.py data/video.MOV
   ```

3. **Thresholds demasiado altos**:
   - Defaults actuales: `match=0.28`, `unknown=0.20`
   - Ajustar según verbose output

### Tracking no funciona correctamente

1. **Tracks muy cortos**: Aumentar `--track-min-hits`
2. **Tracks no terminan**: Reducir `--track-max-age`
3. **Muchos tracks efímeros**: Aumentar `--confianza` o ajustar filtros de área

### Productos identificados incorrectamente

1. **Usar tracking**: Mejora la precisión con consolidación temporal
2. **Revisar crops dudosos**: Usar UI para identificar problemas
3. **Mejorar catálogo**: Agregar más imágenes de referencia

---

## Roadmap (Sprints 4-7)

* **Sprint 4**: 
  - Unidades logísticas (pack/box/pallet)
  - Identificación híbrida (barcode/OCR/visual)
  - Conteo por unidades logísticas
* **Sprint 5**: Planograma de depósito + detección de faltantes
* **Sprint 6**: Escala (batch workers + múltiples cámaras)
* **Sprint 7**: Optimizaciones y mejoras de performance

---

## Versión

* **Versión**: **6.0** (Tracking Temporal + UI + Warehouse Focus)
* **Última actualización**: **Febrero 2026**
* **Inferencia externa**: ninguna
* **Catálogo actual**: 9+ SKUs
* **Packaging**: botella, lata, bolsa, caja, paquete, tubo, frasco

### Changelog v6.0 (Tracking Temporal + UI + Warehouse Focus)

* **Tracking temporal**: SORT-like tracker con IDs persistentes
* **Consolidación por track**: TrackVoteAccumulator con DecisionProfiles
* **UI de revisión**: Interfaz web completa para revisar y gestionar crops
* **Pivot a warehouse/depot**: Enfoque en depósitos y almacenes
* **Modularización**: Pipeline dividido en módulos (core/processing/tracking/output)
* **Código en inglés**: Funciones y variables en inglés, comentarios en español
* **Mejoras de precisión**: Decisiones más estables con tracking

### Changelog v5.6 (Política de Decisión Genérica)

* **1 bbox = 1 decisión final**: Eliminado doble conteo por split
* **Decision Policy**: Módulo genérico y escalable
* **BBox Quality Scorer**: Métricas genéricas de calidad
* **Split como fallback controlado**: Solo si mejora significativamente
* **Packaging calculado una vez**: Reutilización de categoría en splits

### Changelog v5.5 (Sistema de Aprendizaje Continuo)

* **Learning Manager**: Captura automática de crops dudosos
* **Dataset evolutivo**: Metadata estructurada por ejecución
* **Scripts de revisión y absorción**: Workflow completo de mejora

---

## Contribuir

Este es un proyecto en desarrollo activo. Para contribuir:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

---

## Licencia

[Especificar licencia si aplica]

---

## Contacto

[Información de contacto si aplica]
