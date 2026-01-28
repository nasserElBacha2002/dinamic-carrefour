# Sistema de Inventario de Góndolas MVP - Carrefour

Sistema de visión artificial **enterprise-ready** para detectar productos en góndolas de supermercados e identificar SKUs (EANs) a partir de videos.

## 📋 Descripción

Este sistema procesa videos de góndolas de supermercado para:
- **Detectar productos** usando YOLOv8 (detección de objetos)
- **Generar crops** de cada producto detectado
- **Identificar SKU/EAN** usando retrieval visual (comparación con catálogo)
- **Contar cantidades** visibles de cada producto
- **Reconocer marcas** usando OCR configurable (EasyOCR/Tesseract)
- **Generar reportes** en múltiples formatos (CSV, JSON)

### Arquitectura del Sistema

```
Video → Análisis → Frames → Detección → Crops → Identificación SKU → Reporte
                                  ↓                    ↓
                              YOLOv8            Retrieval Visual
                                                (vs Catálogo)
```

### Estado Actual

- ✅ Pipeline completo funcional (análisis → detección → identificación SKU → reporte)
- ✅ Generación de crops de productos detectados
- ✅ Identificación SKU mediante retrieval visual
- ✅ Catálogo de 106 imágenes de referencia para 21 SKUs
- ✅ Reconocimiento de marcas con OCR configurable
- ✅ **Arquitectura SOLID** implementada (4.5/5 puntos)
- ✅ **Dependency Injection** y **Strategy Pattern**
- ✅ Exportación multi-formato (CSV, JSON)
- ✅ Sistema extensible y testeable
- ⚠️ Usa modelo pre-entrenado genérico (YOLOv8n COCO)
- 📝 Para producción, se requiere entrenar modelo personalizado

### 🏆 Calidad del Código

| Principio SOLID | Puntuación |
|-----------------|-----------|
| **S** - Single Responsibility | ★★★★★ (5/5) |
| **O** - Open/Closed | ★★★★★ (5/5) |
| **I** - Interface Segregation | ★★★☆☆ (3/5) |
| **D** - Dependency Inversion | ★★★★★ (5/5) |
| **TOTAL** | **★★★★½ (4.5/5)** |

---

## 🏗️ Arquitectura Completa

### Flujo del Sistema

```
┌──────────────────────────────────────────────────────────────┐
│ FASE 1: GENERACIÓN DE CATÁLOGO (una vez)                    │
└──────────────────────────────────────────────────────────────┘

eans.txt → buscarimagenes.py → imagenes/EAN/*.jpg
   (22 SKUs)                    (106 imágenes)


┌──────────────────────────────────────────────────────────────┐
│ FASE 2: PROCESAMIENTO DE VIDEO                              │
└──────────────────────────────────────────────────────────────┘

Video.MOV
   ↓
1. Análisis de video (metadata)
   ↓
2. Extracción de frames (1 FPS)
   ↓
3. Detección YOLOv8 + Generación de crops
   ↓
4. Identificación SKU (retrieval visual vs catálogo)
   ↓
5. Reporte CSV (EAN, cantidad, fecha)
```

### Componentes Principales

#### Módulos Core
1. **`main.py`**: Orquestador principal del pipeline completo
2. **`analizar_video.py`**: Extrae metadata y frames del video (con detección de nitidez)
3. **`detectar_productos.py`**: Detección YOLOv8 + generación de crops
4. **`identificar_sku_retrieval.py`**: Identificación SKU usando embeddings visuales
5. **`reconocer_marcas.py`**: OCR para identificar marcas

#### Arquitectura SOLID
6. **`protocols.py`**: Abstracciones (interfaces Protocol) para DIP
7. **`ocr_strategies.py`**: Strategy Pattern para OCR (Tesseract/EasyOCR/Dummy)
8. **`exporters.py`**: Strategy Pattern para exportación (CSV/JSON/Multi)
9. **`factory.py`**: Component Factory para Dependency Injection
10. **`utils/`**: Utilidades compartidas (procesamiento de imágenes)

#### Scripts Auxiliares
11. **`buscarimagenes.py`**: Descarga imágenes de referencia por EAN desde Bing

---

## 📁 Estructura del Proyecto

```
dinamic-carrefour/
├── src/                              # Código fuente principal
│   ├── main.py                       # Orquestador principal
│   ├── analizar_video.py             # Análisis y extracción de frames
│   ├── detectar_productos.py         # Detección YOLOv8 + crops
│   ├── identificar_sku_retrieval.py  # Identificación SKU (retrieval)
│   ├── reconocer_marcas.py           # OCR para marcas
│   ├── config.py                     # Configuración centralizada
│   │
│   ├── protocols.py                  # Abstracciones (interfaces)
│   ├── ocr_strategies.py             # Strategy: OCR engines
│   ├── exporters.py                  # Strategy: Export formats
│   ├── factory.py                    # Component Factory (DI)
│   │
│   └── utils/                        # Utilidades compartidas
│       ├── __init__.py
│       └── image_utils.py            # Procesamiento de imágenes
│
├── scripts/                          # Scripts utilitarios
│   ├── buscarimagenes.py             # Descargar imágenes de catálogo
│   ├── descargar_modelo.py           # Descargar modelo pre-entrenado
│   ├── entrenar_modelo.py            # Entrenar modelo personalizado
│   ├── probar_deteccion.py           # Probar detección
│   ├── probar_identificacion_sku.py  # Probar identificación SKU
│   ├── integrar_roboflow.py          # Integración con Roboflow
│   ├── organizar_dataset_descargado.py
│   └── pre_anotar_con_api.py
│
├── tests/                            # Tests y validación
│   └── test_solid_improvements.py    # Tests de arquitectura SOLID
│
├── data/                             # Videos de entrada
│   └── IMG_1838.MOV                  # Video de ejemplo
│
├── imagenes/                         # Catálogo de SKUs (generado)
│   ├── 7790895000997/                # EAN/SKU
│   │   ├── 7790895000997__000.jpg
│   │   └── ...
│   ├── metadata.jsonl                # Metadata de imágenes
│   └── errors.jsonl                  # Errores de descarga
│
├── modelos/                          # Modelos ML
│   ├── yolov8_gondola_mvp.pt         # Modelo por defecto
│   └── yolov8n.pt                    # Modelo base
│
├── output/                           # Resultados del procesamiento
│   └── [video_timestamp]/            # Carpeta por ejecución
│       ├── analisis_video.json       # Metadatos del video
│       ├── frames_extraidos/         # Frames extraídos
│       ├── crops/                    # Crops de productos
│       └── reporte_deteccion/        # Resultados
│           ├── inventario.csv        # Reporte por clase
│           ├── inventario_sku.csv    # Reporte por EAN
│           ├── metadata.json         # Metadatos
│           └── *.jpg                 # Imágenes anotadas
│
├── eans.txt                          # Catálogo de EANs + descripciones
├── embeddings.pkl                    # Embeddings del catálogo (generado)
├── requirements.txt                  # Dependencias Python
├── run.py                            # Wrapper para ejecutar desde raíz
└── README.md                         # Este archivo
```

---

## 🚀 Instalación

### Requisitos Previos

- Python 3.8+
- pip
- (Opcional) Tesseract OCR si prefieres usar pytesseract

### Pasos de Instalación

1. **Clonar/Descargar el proyecto**

2. **Crear entorno virtual** (recomendado):
```bash
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

4. **Descargar modelo pre-entrenado** (si no existe):
```bash
python scripts/descargar_modelo.py
```

5. **Generar catálogo de imágenes** (primera vez):
```bash
python scripts/buscarimagenes.py --input eans.txt --out imagenes --per-ean 5
```

6. **Probar identificación SKU**:
```bash
python scripts/probar_identificacion_sku.py
```

---

## 💻 Uso

### 1. Procesar Video Completo (con Identificación SKU)

```bash
# Pipeline completo: detección + crops + identificación SKU
python run.py data/IMG_1838.MOV --identificar-sku --catalogo imagenes/
```

**Output:**
- `inventario.csv`: Conteo por clase genérica
- `inventario_sku.csv`: Conteo por EAN específico ✨
- `crops/`: Imágenes individuales de cada producto
- `*_detectado.jpg`: Imágenes anotadas con bounding boxes

### 2. Solo Detección + Crops (sin identificar SKU)

```bash
# Generar crops pero sin identificar SKU
python run.py data/IMG_1838.MOV --guardar-crops
```

### 3. Opciones Avanzadas

```bash
# Especificar modelo personalizado
python run.py video.MOV --modelo modelos/mi_modelo.pt --identificar-sku --catalogo imagenes/

# Ajustar confianza de detección (0-1)
python run.py video.MOV --confianza 0.3 --identificar-sku --catalogo imagenes/

# Ajustar threshold de SKU (0-1)
python run.py video.MOV --identificar-sku --catalogo imagenes/ --sku-threshold 0.6

# Elegir método OCR (tesseract/easyocr/dummy)
python run.py video.MOV --ocr-metodo tesseract --identificar-sku --catalogo imagenes/

# Exportar en JSON o múltiples formatos (csv/json/multi)
python run.py video.MOV --export-formato json --identificar-sku --catalogo imagenes/
python run.py video.MOV --export-formato multi --identificar-sku --catalogo imagenes/

# Extraer más frames por segundo
python run.py video.MOV --fps 2.0 --identificar-sku --catalogo imagenes/

# Rotar frames (90/180/270 grados, para videos verticales)
python run.py video.MOV --rotar 90 --identificar-sku --catalogo imagenes/

# Filtrar frames borrosos automáticamente
python run.py video.MOV --filtrar-borrosos --umbral-nitidez 100

# Configurar formato y calidad de frames exportados
python run.py video.MOV --formato png --calidad 95

# Sin generar imágenes anotadas (más rápido)
python run.py video.MOV --sin-anotaciones --identificar-sku --catalogo imagenes/

# Desactivar reconocimiento de marcas (más rápido)
python run.py video.MOV --sin-marcas --identificar-sku --catalogo imagenes/

# Solo análisis y frames (sin detección)
python run.py video.MOV --sin-deteccion
```

### Ver Ayuda Completa

```bash
python run.py --help
```

---

## ⚙️ Configuración

### Archivo de Configuración

La configuración principal está en `src/config.py`:

- **MODELO_DEFAULT**: Ruta al modelo por defecto
- **CONFIANZA_MINIMA_DEFAULT**: Umbral de confianza (0.25)
- **FPS_EXTRACCION_DEFAULT**: Frames por segundo a extraer (1.0)

### Catálogo de EANs

Edita `eans.txt` con tus productos (formato: `EAN<TAB>DESCRIPCIÓN`):

```
7790895000997	GASEOSA COLA REGULAR COCA COLA PET X 2.25 LT
7790895000430	GASEOSA COLA REGULAR COCA COLA PET X 1.5 LT
```

### Generar Catálogo de Imágenes

```bash
# Descargar imágenes de referencia por EAN
python scripts/buscarimagenes.py --input eans.txt --out imagenes --per-ean 10

# Opciones:
#   --per-ean: Cantidad de imágenes por EAN (default: 10)
#   --shuffle-candidates: Mezclar para mayor diversidad
#   --dedupe-global: Evitar duplicados entre EANs
```

---

## 🏗️ Arquitectura SOLID y Extensibilidad

### Dependency Injection

El sistema usa **Dependency Injection** para facilitar testing y extensibilidad:

```python
from src.factory import ComponentFactory

# Crear componentes desde CLI args
factory = ComponentFactory.from_cli_args(args)
detector = factory.create_detector()
identificador_sku = factory.create_sku_identifier()

# O desde configuración personalizada
config = {
    'ocr_metodo': 'tesseract',
    'export_formato': 'json',
    'confianza_minima': 0.3
}
factory = ComponentFactory.desde_config(config)
```

### Strategy Pattern: OCR

Intercambia engines OCR sin modificar código:

```python
from src.ocr_strategies import TesseractOCRStrategy, EasyOCRStrategy
from src.reconocer_marcas import ReconocedorMarcas

# Usar Tesseract
ocr = TesseractOCRStrategy()
reconocedor = ReconocedorMarcas(ocr_strategy=ocr)

# Cambiar a EasyOCR
ocr = EasyOCRStrategy()
reconocedor = ReconocedorMarcas(ocr_strategy=ocr)
```

### Strategy Pattern: Exporters

Exporta reportes en diferentes formatos:

```python
from src.exporters import CSVExporter, JSONExporter, MultiFormatExporter

# Solo CSV
exporter = CSVExporter()
exporter.export_data(data, 'inventario.csv')

# Solo JSON
exporter = JSONExporter()
exporter.export_data(data, 'inventario.json')

# Ambos formatos
exporter = MultiFormatExporter()
exporter.export_data(data, 'inventario')  # → .csv y .json
```

### Testing con Mocks

```python
from src.ocr_strategies import DummyOCRStrategy
from src.reconocer_marcas import ReconocedorMarcas

# Mock OCR para tests sin dependencias externas
mock_ocr = DummyOCRStrategy()
reconocedor = ReconocedorMarcas(ocr_strategy=mock_ocr)

# Tests rápidos sin instalar Tesseract/EasyOCR
resultado = reconocedor.procesar_detecciones('test.jpg', detecciones)
```

---

## 📊 Formato de Salida

### CSV de Inventario SKU (`inventario_sku.csv`)

```csv
EAN,Cantidad,Fecha
7790895000997,4,2026-01-27 15:30:22
7790895000430,2,2026-01-27 15:30:22
UNKNOWN,3,2026-01-27 15:30:22
```

### CSV de Inventario por Clase (`inventario.csv`)

```csv
Producto/Marca,Cantidad Detectada,Fecha
bottle_Susante,4,2026-01-27 15:30:22
bottle_Levite,2,2026-01-27 15:30:22
bottle,3,2026-01-27 15:30:22
```

### JSON de Metadatos (`metadata.json`)

```json
{
  "fecha": "2026-01-27T15:30:22",
  "total_frames": 7,
  "total_skus": 3,
  "total_productos": 9,
  "sku_identificados": 6,
  "conteo": {
    "7790895000997": 4,
    "7790895000430": 2,
    "UNKNOWN": 3
  }
}
```

---

## 🎯 Identificación SKU - Cómo Funciona

### Retrieval Visual

El sistema usa **retrieval visual** para identificar SKUs:

1. **Generación de embeddings del catálogo** (una vez):
   - Extrae features de cada imagen del catálogo usando ResNet50
   - Guarda embeddings en `embeddings.pkl`
   - ~10 segundos para 100 imágenes

2. **Identificación de crops**:
   - Extrae features del crop detectado
   - Compara con embeddings del catálogo (cosine similarity)
   - Retorna EAN con mayor similitud (si > threshold)

### Ventajas del Retrieval Visual

- ✅ No requiere reentrenar modelos
- ✅ Agregar SKU = agregar imágenes al catálogo
- ✅ Funciona con variaciones de iluminación/ángulo
- ✅ Agrupa automáticamente EANs visualment similares
- ✅ Threshold configurable para precisión/recall

### Probar el Sistema

```bash
# Ver estadísticas del catálogo y hacer pruebas
python scripts/probar_identificacion_sku.py
```

**Output ejemplo:**
```
📊 ESTADÍSTICAS DEL CATÁLOGO:
   Total de SKUs (EANs): 21
   Total de imágenes: 106
   Promedio por SKU: 5.0

🔍 PRUEBA DE AUTO-IDENTIFICACIÓN:
✅ Test 1/5: EAN predicho correctamente (0.987)
✅ Test 2/5: EAN predicho correctamente (0.923)
...

📈 RESULTADOS:
   Correctos: 5/5
   Precisión: 100.0%
```

---

## 🎯 Modelos y Entrenamiento

### Modelo Actual

El sistema usa un modelo **pre-entrenado genérico** (YOLOv8n COCO) que detecta objetos comunes:
- `bottle`, `cup`, `bowl`, etc.

**Limitación**: No detecta productos específicos de góndola.

### Entrenar Modelo Personalizado

#### Opción 1: Usar Roboflow (⭐ Recomendado)

1. **Crear cuenta** en https://roboflow.com
2. **Subir imágenes** de tus videos
3. **Anotar productos** en la interfaz web
4. **Exportar dataset** en formato YOLOv8
5. **Entrenar**:
   ```bash
   python scripts/entrenar_modelo.py --dataset datos/dataset.yaml --epochs 100
   ```

#### Opción 2: Entrenamiento Manual

```bash
# 1. Preparar dataset (imágenes + anotaciones YOLO)
# 2. Crear configuración
python scripts/entrenar_modelo.py --crear-config datos/ --clases botella bidon

# 3. Entrenar
python scripts/entrenar_modelo.py --dataset datos/dataset.yaml --epochs 100

# 4. Usar modelo entrenado
python run.py video.MOV --modelo modelos/gondola_training/weights/best.pt
```

---

## 🧪 Testing y Validación

### Probar Arquitectura SOLID

```bash
python tests/test_solid_improvements.py
```

**Output esperado:**
```
✅ Test 1/4: Imports y sintaxis correctos
✅ Test 2/4: ComponentFactory funcional
✅ Test 3/4: Dependency Injection operativa
✅ Test 4/4: Exportadores validados

🎉 TODOS LOS TESTS PASARON (4/4)
```

### Probar Detección

```bash
python scripts/probar_deteccion.py
```

### Probar Identificación SKU

```bash
python scripts/probar_identificacion_sku.py
```

### Verificar Catálogo

```bash
# Ver imágenes descargadas
ls -la imagenes/*/

# Ver metadata
cat imagenes/metadata.jsonl | head -5

# Ver errores (si hubo)
cat imagenes/errors.jsonl | grep "error"
```

---

## 📝 Casos de Uso

### Caso 1: Inventario Rápido (sin SKU)

```bash
# Solo detección y conteo genérico
python run.py video.MOV --sin-marcas
```

### Caso 2: Inventario con Identificación SKU

```bash
# Identificación completa por EAN
python run.py video.MOV --identificar-sku --catalogo imagenes/
```

### Caso 3: Generar Dataset para Entrenamiento

```bash
# Generar crops para etiquetar manualmente
python run.py video.MOV --guardar-crops --sin-marcas
```

### Caso 4: Agregar Nuevos SKUs al Catálogo

```bash
# 1. Agregar EAN a eans.txt
echo "7790895999999\tPRODUCTO NUEVO X 1L" >> eans.txt

# 2. Descargar imágenes
python buscarimagenes.py --input eans.txt --out imagenes --per-ean 10

# 3. Regenerar embeddings
rm embeddings.pkl
python scripts/probar_identificacion_sku.py
```

---

## 📚 Referencias

- **YOLOv8**: https://github.com/ultralytics/ultralytics
- **EasyOCR**: https://github.com/JaidedAI/EasyOCR
- **PyTorch**: https://pytorch.org/
- **OpenCV**: https://opencv.org/
- **Roboflow**: https://roboflow.com/

---

## 🐛 Troubleshooting

### Error: "PyTorch no disponible"

```bash
pip install torch torchvision
```

### Error: "Catálogo no encontrado"

```bash
# Generar catálogo primero
python buscarimagenes.py --input eans.txt --out imagenes --per-ean 5
```

### Error: "Modelo no encontrado"

```bash
# Descargar modelo pre-entrenado
python scripts/descargar_modelo.py
```

### Baja Precisión en Identificación SKU

1. Verificar calidad de imágenes del catálogo
2. Aumentar cantidad de imágenes por SKU (10-15 recomendado)
3. Ajustar threshold: `threshold=0.4` (menos estricto)
4. Verificar que los EANs en `eans.txt` coincidan con carpetas en `imagenes/`

### Rendimiento Lento

- Usar `--sin-anotaciones` para no generar imágenes
- Usar `--sin-marcas` para desactivar OCR
- Reducir FPS de extracción: `--fps 0.5`
- Los embeddings se calculan una sola vez y se cachean

---

## 👥 Soporte

Para preguntas o problemas:
1. Revisar este README
2. Ejecutar scripts de prueba
3. Verificar logs en consola
4. Revisar archivos de salida en `output/`

---

## 📄 Licencia

MVP desarrollado para Carrefour - Uso interno

---

**Última actualización**: Enero 2026  
**Versión**: 2.0 (Enterprise-Ready con SOLID Architecture)
