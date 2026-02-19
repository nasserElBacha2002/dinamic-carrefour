# Changelog v5.5 — Sistema de Aprendizaje Continuo

**Fecha**: 19 Febrero 2026  
**Versión**: 5.5 (Sprint 2.1)

---

## 🎯 Resumen

Se implementó un **sistema de aprendizaje continuo** que permite que el sistema mejore automáticamente con cada ejecución, sin necesidad de reentrenamientos.

---

## ✨ Nuevas Funcionalidades

### 1. Learning Manager (`src/learning/manager.py`)

**Qué hace**:
- Captura automáticamente crops dudosos (UNKNOWN, AMBIGUOUS) durante el procesamiento
- Organiza crops en carpetas separadas (`unknown/`, `ambiguous/`)
- Genera metadata completa en formato JSONL

**Integración**:
- Se activa automáticamente en el pipeline
- No requiere configuración adicional
- Se integra transparentemente con el flujo existente

### 2. Script de Revisión (`scripts/revisar_crops.py`)

**Qué hace**:
- Permite revisar crops pendientes de forma interactiva
- Muestra información completa de cada crop (detection, packaging, SKU identification)
- Permite asignar EAN correcto a cada crop
- Guarda cambios en metadata

**Uso**:
```bash
# Revisar todos los crops
python scripts/revisar_crops.py output/VIDEO_TIMESTAMP/learning

# Solo UNKNOWN
python scripts/revisar_crops.py output/VIDEO_TIMESTAMP/learning --solo-unknown

# Solo AMBIGUOUS
python scripts/revisar_crops.py output/VIDEO_TIMESTAMP/learning --solo-ambiguous
```

### 3. Script de Absorción (`scripts/absorber_crops.py`)

**Qué hace**:
- Lee crops que fueron revisados y tienen EAN asignado
- Copia crops a la carpeta de imágenes del SKU correspondiente
- Recalcula embeddings automáticamente
- Actualiza el vector store

**Uso**:
```bash
# Absorber todos los crops revisados
python scripts/absorber_crops.py output/VIDEO_TIMESTAMP/learning

# Solo un SKU específico
python scripts/absorber_crops.py output/VIDEO_TIMESTAMP/learning --solo-ean 7793890258288

# Simular sin hacer cambios
python scripts/absorber_crops.py output/VIDEO_TIMESTAMP/learning --dry-run
```

---

## 📁 Estructura de Archivos Nuevos

```
src/
  learning/
    __init__.py          # Módulo de aprendizaje
    manager.py            # Learning Manager (gestor de dataset evolutivo)

scripts/
  revisar_crops.py        # Script de revisión rápida CLI
  absorber_crops.py       # Script de absorción al catálogo
```

---

## 🔄 Flujo Completo

### Antes (v5.4)
```
Ejecutar pipeline → Generar reporte → Fin
```

### Ahora (v5.5)
```
Ejecutar pipeline
  ↓
Generar reporte + Capturar crops dudosos automáticamente
  ↓
Revisar crops (5-10 min)
  ↓
Absorber crops al catálogo
  ↓
Siguiente ejecución mejora automáticamente
```

---

## 📊 Estructura Generada por Ejecución

Cada ejecución ahora genera:

```
output/<video_timestamp>/
    learning/
        unknown/              # Crops no identificados
            frame_003_crop_001.jpg
            ...
        ambiguous/             # Crops con top1≈top2
            frame_002_crop_001.jpg
            ...
        metadata/
            execution_meta.json    # Parámetros de ejecución
            crops_index.jsonl      # Metadata completa por crop
    reporte_deteccion/
        inventario_sku.csv
        frame_XXXX.jpg
```

---

## 🔧 Cambios en Código Existente

### `src/pipeline/engine.py`

**Cambios**:
- Integración automática del `LearningManager`
- Captura de crops dudosos durante el procesamiento
- Guardado de metadata completa (detection, packaging, SKU identification)
- Resumen de crops guardados al final

**Líneas modificadas**:
- Importación de `LearningManager`
- Inicialización en `procesar_video()`
- Guardado de crops dudosos en el loop de procesamiento
- Resumen en el output final

### Sin cambios breaking

- El pipeline funciona igual que antes
- Los parámetros CLI no cambiaron
- La estructura de output es compatible (solo se agregó `learning/`)

---

## 📈 Beneficios

### 1. Mejora Automática
- Cada ejecución genera datos valiosos
- El sistema mejora sin intervención técnica
- Menos UNKNOWN, menos confusiones con el tiempo

### 2. Escalabilidad
- Agregar crops al catálogo es rápido (minutos)
- No requiere reentrenamientos
- El sistema se vuelve más robusto progresivamente

### 3. Auditabilidad
- Metadata completa de cada decisión
- Trazabilidad de crops dudosos
- Historial de mejoras

### 4. Eficiencia Operativa
- Revisión rápida (5-10 minutos típicamente)
- Absorción automática
- Sin trabajo manual repetitivo

---

## 🚀 Cómo Empezar

### 1. Primera Ejecución

```bash
# Ejecutar pipeline (guarda crops automáticamente)
python run.py data/video.MOV
```

**Output esperado**:
```
📚 Learning Manager activado (dataset evolutivo)
...
📚 Dataset evolutivo:
   Crops guardados: 12
   UNKNOWN: 8
   AMBIGUOUS: 4
   Metadata: output/.../learning/metadata/crops_index.jsonl
```

### 2. Revisar Crops

```bash
python scripts/revisar_crops.py output/VIDEO_TIMESTAMP/learning
```

### 3. Absorber Crops

```bash
python scripts/absorber_crops.py output/VIDEO_TIMESTAMP/learning
```

### 4. Siguiente Ejecución

Al ejecutar nuevamente, el sistema ya habrá mejorado automáticamente.

---

## 📝 Ejemplo de Metadata JSONL

Cada línea en `crops_index.jsonl` contiene:

```json
{
  "crop_id": "frame_003_crop_001",
  "timestamp": "2026-02-19T17:14:24Z",
  "execution_id": "IMG_2199_20260219_171424",
  "detection": {
    "bbox": [x1, y1, x2, y2],
    "bbox_padded": [x1p, y1p, x2p, y2p],
    "yolo_conf": 0.87,
    "class_id": 39,
    "raw_label": "product"
  },
  "packaging": {
    "predicted": "bolsa"
  },
  "sku_identification": {
    "decision": "unknown",
    "top1": {"ean": "UNKNOWN", "similitud": 0.38},
    "top2": {"ean": "7793890258288", "similitud": 0.35},
    "all_matches": [...],
    "threshold_used": 0.70,
    "unknown_threshold": 0.40,
    "margen_ambiguedad": 0.05
  },
  "decision": "unknown",
  "frame_path": "frames/frame_003.jpg",
  "frame_idx": 3,
  "paths": {
    "crop": "learning/unknown/frame_003_crop_001.jpg",
    "frame": "frames/frame_003.jpg"
  },
  "review": {
    "status": "pending",
    "assigned_ean": null,
    "reviewed_at": null,
    "reviewer": null
  }
}
```

---

## 🔍 Detalles Técnicos

### Learning Manager

**Clase**: `LearningManager`  
**Ubicación**: `src/learning/manager.py`

**Métodos principales**:
- `__init__()`: Inicializa estructura de directorios
- `guardar_crop_dudoso()`: Guarda crop con metadata completa
- `resumen()`: Retorna estadísticas de crops guardados

### Integración en Pipeline

El `PipelineEngine` ahora:
1. Inicializa `LearningManager` al inicio de `procesar_video()`
2. Captura crops cuando `status in ["unknown", "ambiguous"]`
3. Guarda metadata completa de cada decisión
4. Muestra resumen al final

### Scripts

**`revisar_crops.py`**:
- Lee metadata JSONL
- Filtra crops pendientes
- Interfaz CLI interactiva
- Guarda cambios en metadata

**`absorber_crops.py`**:
- Lee crops revisados desde metadata
- Copia a carpetas de SKU
- Recalcula embeddings
- Actualiza vector store

---

## ⚠️ Notas Importantes

1. **El Learning Manager se activa automáticamente**: No requiere configuración adicional
2. **Los crops se mantienen**: No se eliminan después de absorberlos, quedan para referencia
3. **Solo crops revisados se absorben**: Crops con `status: "reviewed"` y `assigned_ean` válido
4. **Compatible con versión anterior**: El pipeline funciona igual, solo se agregó funcionalidad

---

## 📚 Documentación Relacionada

- **Documentación completa**: [LEARNING_SYSTEM.md](LEARNING_SYSTEM.md)
- **README principal**: [README.md](README.md)

---

## 🎓 Conceptos Clave

### Dataset Evolutivo

Cada ejecución genera un **micro-dataset real de góndola**:
- No es dataset sintético
- Captura condiciones reales
- Incluye casos límite y errores

### Loop de Mejora

```
Ejecución → Captura errores → Revisión humana → Absorción → Mejora automática
```

**Sin reentrenamientos. Sin romper el sistema.**

---

## ✅ Checklist de Migración

Si estás actualizando desde v5.4:

- [x] Los nuevos archivos se crean automáticamente
- [x] No hay cambios breaking en el pipeline
- [x] Los scripts son opcionales (el sistema funciona sin ellos)
- [x] La estructura de output es compatible

**No se requiere acción adicional**. El sistema funciona igual que antes, con funcionalidad adicional disponible.

---

**Versión**: 5.5  
**Fecha**: 19 Febrero 2026  
**Autor**: Sistema Dinamic Carrefour
