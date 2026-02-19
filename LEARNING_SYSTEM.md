# Sistema de Aprendizaje Continuo — Dataset Evolutivo

## 📚 Versión 2.1 — Loop de Mejora Automática

Este documento describe el sistema de aprendizaje continuo implementado que permite que el sistema mejore automáticamente con cada ejecución, sin necesidad de reentrenamientos.

---

## 🎯 Objetivo

Transformar cada ejecución del pipeline en un **dataset estructurado** que capture:
- Casos dudosos (UNKNOWN, AMBIGUOUS)
- Metadata completa de cada decisión
- Información para revisión humana rápida
- Absorción automática al catálogo

**Resultado**: El sistema reduce errores progresivamente sin reentrenar modelos.

---

## 🏗️ Arquitectura

### Componentes Nuevos

1. **`LearningManager`** (`src/learning/manager.py`)
   - Gestiona la captura y estructuración de datos por ejecución
   - Organiza crops dudosos en carpetas separadas
   - Genera metadata JSONL completa

2. **Script de Revisión** (`scripts/revisar_crops.py`)
   - CLI simple para revisar crops pendientes
   - Asignar EAN correcto a cada crop
   - Guardar cambios en metadata

3. **Script de Absorción** (`scripts/absorber_crops.py`)
   - Lee crops revisados
   - Los copia al catálogo del SKU correcto
   - Recalcula embeddings automáticamente
   - Actualiza el vector store

### Integración en Pipeline

El `PipelineEngine` ahora:
- Inicializa automáticamente el `LearningManager`
- Captura crops dudosos durante el procesamiento
- Guarda metadata completa de cada decisión
- Muestra resumen de crops guardados

---

## 📁 Estructura Generada por Ejecución

Cada ejecución genera la siguiente estructura:

```
output/<video_timestamp>/
    learning/
        unknown/              # Crops no identificados
            frame_003_crop_001.jpg
            frame_005_crop_002.jpg
            ...
        ambiguous/            # Crops con top1≈top2
            frame_002_crop_001.jpg
            ...
        metadata/
            execution_meta.json    # Parámetros de ejecución
            crops_index.jsonl      # Metadata completa por crop
    reporte_deteccion/
        inventario_sku.csv
        frame_XXXX.jpg
```

### Metadata por Crop (JSONL)

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
    "top1": {"ean": "UNKNOWN", "similitud": 0.38, "descripcion": ""},
    "top2": {"ean": "7793890258288", "similitud": 0.35, "descripcion": "..."},
    "top3": [...],
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
    "reviewer": null,
    "notes": null
  }
}
```

---

## 🔄 Flujo Completo de Uso

### 1️⃣ Ejecutar Pipeline

```bash
python run.py data/video.MOV
```

**Resultado**:
- Procesa el video
- Detecta productos
- Identifica SKUs
- **Guarda automáticamente crops dudosos en `learning/`**

**Output**:
```
📚 Learning Manager activado (dataset evolutivo)
...
📚 Dataset evolutivo:
   Crops guardados: 12
   UNKNOWN: 8
   AMBIGUOUS: 4
   Metadata: output/.../learning/metadata/crops_index.jsonl
```

### 2️⃣ Revisar Crops Dudosos

```bash
# Revisar todos los crops pendientes
python scripts/revisar_crops.py output/IMG_2199_20260219_171424/learning

# Solo UNKNOWN
python scripts/revisar_crops.py output/IMG_2199_20260219_171424/learning --solo-unknown

# Solo AMBIGUOUS
python scripts/revisar_crops.py output/IMG_2199_20260219_171424/learning --solo-ambiguous
```

**Interfaz**:
```
Crop 1/12: frame_003_crop_001
======================================================================
🔍 Detección YOLO: conf=0.870
📦 Packaging: bolsa
🎯 Decisión: UNKNOWN

   Top candidatos:
   👈 1. UNKNOWN (sim=0.3800) — 
     2. 7793890258288 (sim=0.3500) — Rapiditas Integrales Bimbo...
     3. 7796989011894 (sim=0.3200) — Rapiditas Light Bimbo...

   ➜ EAN correcto (o s/q/d): 7793890258288
   ✅ Asignado: 7793890258288
```

**Comandos**:
- `EAN`: Asignar EAN correcto (ej: `7793890258288`)
- `s`: Saltar este crop
- `q`: Salir
- `d`: Descartar (no es producto)

### 3️⃣ Absorber Crops al Catálogo

```bash
# Absorber todos los crops revisados
python scripts/absorber_crops.py output/IMG_2199_20260219_171424/learning

# Solo un SKU específico
python scripts/absorber_crops.py output/IMG_2199_20260219_171424/learning --solo-ean 7793890258288

# Simular sin hacer cambios (dry-run)
python scripts/absorber_crops.py output/IMG_2199_20260219_171424/learning --dry-run
```

**Proceso**:
1. Lee crops revisados desde metadata
2. Copia crops a `imagenes/<EAN>/`
3. Recalcula embeddings para SKUs afectados
4. Actualiza el vector store

**Output**:
```
🔄 ABSORBIENDO CROPS REVISADOS AL CATÁLOGO
======================================================================

📋 8 crops revisados encontrados
   SKUs afectados: 3

   📦 7793890258288 (3 crops):
      ✅ frame-003-crop-001.jpg
      ✅ frame-005-crop-002.jpg
      ✅ frame-007-crop-001.jpg

🔄 Recalculando embeddings para 3 SKUs...
  📸 7793890258288 — 8 imágenes → calculando embeddings...
  ✅ 7793890258288 — 8 embeddings guardados

✅ Proceso completado:
   Crops copiados: 8
   SKUs actualizados: 3
   Errores: 0
```

### 4️⃣ Siguiente Ejecución

Al ejecutar el pipeline nuevamente:
- El sistema ya tiene más imágenes de referencia
- Los embeddings son más representativos
- **Menos UNKNOWN, menos confusiones**

---

## 📊 Cómo el Sistema Mejora

### 1. Embeddings Más Representativos

**Antes**: Embeddings solo de imágenes limpias de catálogo web

**Después**: Embeddings incluyen:
- Variaciones reales de góndola (reflejos, oclusiones)
- Ángulos y perspectivas reales
- Iluminación real del ambiente

**Resultado**: Mejor representación → Menos UNKNOWN

### 2. Reducción de Falsos Positivos

Cada crop absorbido mejora la "memoria" del SKU:
- Más ejemplos = mejor perfil del producto
- Variaciones reales = mejor cobertura
- Casos límite = mejor discriminación

### 3. Métricas de Evolución

El sistema puede medirse por:
- **% UNKNOWN**: Debe bajar con el tiempo
- **% AMBIGUOUS**: Debe bajar con el tiempo
- **Similitud promedio top1**: Debe subir
- **Tiempo de revisión**: Debe bajar (menos crops dudosos)

---

## 🔧 Configuración

### Activar/Desactivar Learning Manager

El Learning Manager se activa automáticamente si el módulo está disponible.

Para desactivarlo temporalmente, puedes modificar `src/pipeline/engine.py`:

```python
# Desactivar Learning Manager
_LEARNING_AVAILABLE = False
```

### Parámetros Guardados en Metadata

Cada ejecución guarda automáticamente:
- `fps_extraccion`
- `confianza_minima` (YOLO)
- `iou_nms` (YOLO)
- `sku_threshold`
- `unknown_threshold`
- `margen_ambiguedad`

Esto permite reproducir condiciones exactas de una ejecución.

---

## 📝 Ejemplos de Uso

### Ejemplo 1: Primera Ejecución

```bash
# 1. Ejecutar pipeline
python run.py data/IMG_2199.MOV

# Output muestra:
# 📚 Dataset evolutivo:
#    Crops guardados: 15
#    UNKNOWN: 10
#    AMBIGUOUS: 5

# 2. Revisar crops
python scripts/revisar_crops.py output/IMG_2199_20260219_171424/learning

# 3. Absorber crops revisados
python scripts/absorber_crops.py output/IMG_2199_20260219_171424/learning
```

### Ejemplo 2: Segunda Ejecución (Sistema Mejorado)

```bash
# 1. Ejecutar pipeline nuevamente
python run.py data/IMG_2200.MOV

# Output muestra:
# 📚 Dataset evolutivo:
#    Crops guardados: 8  ← Menos crops dudosos!
#    UNKNOWN: 5          ← Menos UNKNOWN!
#    AMBIGUOUS: 3

# El sistema ya mejoró automáticamente
```

### Ejemplo 3: Revisión Selectiva

```bash
# Solo revisar UNKNOWN (más críticos)
python scripts/revisar_crops.py output/.../learning --solo-unknown

# Absorber solo un SKU específico
python scripts/absorber_crops.py output/.../learning --solo-ean 7793890258288
```

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

### Absorción vs Reentrenamiento

| Aspecto | Reentrenamiento | Absorción |
|---------|----------------|-----------|
| Tiempo | Horas | Minutos |
| Riesgo | Puede romper modelo | No rompe nada |
| Escalabilidad | Limitada | Excelente |
| Costo | Alto | Bajo |

---

## 🚀 Próximos Pasos (Roadmap)

### Sprint 2.1 (Próximo)
- [ ] Thresholds dinámicos por SKU
- [ ] Hard negative mining
- [ ] Dashboard de métricas

### Sprint 3+
- [ ] Tracking de productos
- [ ] Frame deduplicación
- [ ] Vector DB (FAISS/Pinecone)

---

## 📚 Archivos Relacionados

- `src/learning/manager.py` - Gestor de dataset evolutivo
- `src/learning/__init__.py` - Módulo de aprendizaje
- `src/pipeline/engine.py` - Integración en pipeline
- `scripts/revisar_crops.py` - Script de revisión
- `scripts/absorber_crops.py` - Script de absorción

---

## ❓ Preguntas Frecuentes

### ¿Cuánto tiempo toma revisar crops?

**Respuesta**: Depende del volumen, pero típicamente:
- 10-20 crops: 5-10 minutos
- 50 crops: 15-20 minutos
- 100+ crops: 30-40 minutos

### ¿Qué pasa si no reviso los crops?

**Respuesta**: Nada. Los crops quedan guardados para revisión posterior. El sistema sigue funcionando normalmente.

### ¿Puedo absorber crops sin revisarlos?

**Respuesta**: No. Solo se absorben crops que tienen `status: "reviewed"` y `assigned_ean` válido.

### ¿Los crops se eliminan después de absorberlos?

**Respuesta**: No. Los crops se mantienen en `learning/` para referencia futura. Solo se copian al catálogo.

### ¿Cómo sé si el sistema está mejorando?

**Respuesta**: Compara métricas entre ejecuciones:
- Menos crops guardados = mejor
- Menos UNKNOWN = mejor
- Mayor similitud promedio = mejor

---

## 📈 Métricas de Éxito

Un sistema que evoluciona correctamente muestra:

✅ **% UNKNOWN baja** con el tiempo  
✅ **% AMBIGUOUS baja** con el tiempo  
✅ **Similitud promedio top1 sube**  
✅ **Tiempo de revisión baja** (menos crops)  
✅ **Falsos positivos disminuyen**

---

## 🔗 Integración con Sistema Principal

El Learning Manager se integra automáticamente en el pipeline principal. No requiere configuración adicional.

Para más detalles sobre el sistema completo, ver:
- `README.md` - Documentación general
- `src/pipeline/engine.py` - Implementación del pipeline

---

**Versión**: 2.1  
**Fecha**: Febrero 2026  
**Autor**: Sistema Dinamic Carrefour
