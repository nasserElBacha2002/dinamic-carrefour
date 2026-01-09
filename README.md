# Sistema de Inventario de Góndolas MVP

Sistema de visión artificial para detectar y contar productos en góndolas de supermercados a partir de videos. Desarrollado como MVP para Carrefour con enfoque en precisión y escalabilidad.

## 📋 Descripción

Este sistema procesa videos de góndolas de supermercado para:
- **Detectar productos** usando YOLOv8 (detección de objetos)
- **Contar cantidades** visibles de cada producto
- **Identificar marcas** usando OCR (EasyOCR)
- **Generar reportes** en formato CSV con inventario

### Estado Actual

- ✅ Pipeline completo funcional (análisis → detección → reporte)
- ✅ Reconocimiento de marcas con OCR
- ⚠️ Usa modelo **pre-entrenado genérico** (YOLOv8n COCO)
- 📝 Para producción, se requiere entrenar modelo personalizado

---

## 🏗️ Arquitectura del Sistema

```
Video → Análisis → Extracción de Frames → Detección → Reconocimiento de Marcas → Reporte CSV
```

### Flujo Completo

1. **Análisis de Video** (`analizar_video.py`)
   - Extrae metadatos (duración, FPS, resolución)
   - Valida calidad del video

2. **Extracción de Frames** (`analizar_video.py`)
   - Extrae frames a intervalos configurables (default: 1 FPS)
   - Opción de rotación para videos verticales

3. **Detección de Productos** (`detectar_productos.py`)
   - Usa YOLOv8 para detectar objetos en cada frame
   - Filtra por confianza mínima
   - Genera imágenes anotadas con bounding boxes

4. **Reconocimiento de Marcas** (`reconocer_marcas.py`)
   - Extrae texto de cada producto detectado usando OCR
   - Identifica marcas usando fuzzy matching
   - Soporta marcas conocidas desde archivo

5. **Generación de Reportes** (`detectar_productos.py`)
   - Cuenta productos por clase/marca
   - Exporta CSV con inventario
   - Genera metadata JSON

---

## 📁 Estructura del Proyecto

```
Dinamic sistems/
├── src/                          # Código fuente principal
│   ├── main.py                  # Orquestador principal del pipeline
│   ├── analizar_video.py        # Análisis y extracción de frames
│   ├── detectar_productos.py    # Detección YOLOv8 y conteo
│   ├── reconocer_marcas.py      # OCR y reconocimiento de marcas
│   └── config.py                # Configuración centralizada
│
├── scripts/                      # Scripts utilitarios
│   ├── descargar_modelo.py      # Descargar modelo pre-entrenado
│   ├── entrenar_modelo.py       # Entrenar modelo personalizado
│   └── probar_deteccion.py       # Probar detección en imagen
│
├── data/                         # Videos de entrada
│   └── IMG_1838.MOV             # Video de ejemplo
│
├── modelos/                      # Modelos ML
│   ├── yolov8_gondola_mvp.pt    # Modelo por defecto (pre-entrenado)
│   └── yolov8n.pt               # Modelo base
│
├── output/                       # Resultados del procesamiento
│   └── [video_timestamp]/       # Carpeta por ejecución
│       ├── analisis_video.json  # Metadatos del video
│       ├── frames_extraidos/    # Frames extraídos
│       └── reporte_deteccion/    # Resultados de detección
│           ├── inventario.csv   # Reporte final
│           ├── metadata.json     # Metadatos de detección
│           └── *.jpg            # Imágenes anotadas
│
├── marcas_conocidas.txt          # Marcas conocidas (opcional)
├── requirements.txt              # Dependencias Python
├── run.py                        # Wrapper para ejecutar desde raíz
└── README.md                     # Este archivo
```

---

## 🚀 Instalación

### Requisitos Previos

- Python 3.8+
- pip
- (Opcional) Tesseract OCR si prefieres usar pytesseract en lugar de EasyOCR

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

5. **Configurar marcas conocidas** (opcional):
```bash
cp marcas_conocidas.txt.example marcas_conocidas.txt
# Editar marcas_conocidas.txt con tus marcas
```

---

## 💻 Uso

### Uso Básico

Procesar un video completo (análisis + detección + reporte):

```bash
python run.py data/IMG_1838.MOV
```

O desde el directorio raíz:

```bash
python -m src.main data/IMG_1838.MOV
```

### Opciones Avanzadas

```bash
# Solo análisis y frames (sin detección)
python run.py video.MOV --sin-deteccion

# Especificar modelo personalizado
python run.py video.MOV --modelo modelos/mi_modelo.pt

# Ajustar confianza de detección (0-1)
python run.py video.MOV --confianza 0.3

# Extraer más frames por segundo
python run.py video.MOV --fps 2.0

# Rotar frames (para videos verticales)
python run.py video.MOV --rotar

# Sin generar imágenes anotadas (más rápido)
python run.py video.MOV --sin-anotaciones

# Desactivar reconocimiento de marcas (más rápido)
python run.py video.MOV --sin-marcas

# Especificar directorio de salida
python run.py video.MOV --output mis_resultados/
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

### Marcas Conocidas

Crea `marcas_conocidas.txt` en la raíz del proyecto:

```
Susante
Levite
Agua
```

El sistema usará estas marcas para mejorar la identificación con fuzzy matching.

---

## 🔧 Componentes Principales

### 1. `SistemaInventarioGondola` (main.py)

Clase principal que orquesta todo el pipeline:

```python
from src.main import SistemaInventarioGondola

sistema = SistemaInventarioGondola(
    modelo_path="modelos/mi_modelo.pt",
    confianza_minima=0.3,
    reconocer_marcas=True
)

resultados = sistema.procesar_video(
    "data/video.MOV",
    fps_extraccion=1.0,
    detectar=True,
    generar_anotaciones=True
)
```

### 2. `DetectorProductos` (detectar_productos.py)

Maneja la detección YOLOv8 y conteo:

```python
from src.detectar_productos import DetectorProductos

detector = DetectorProductos(
    modelo_path="modelos/yolov8_gondola_mvp.pt",
    confianza_minima=0.25,
    reconocer_marcas=True
)

# Detectar en una imagen
detecciones = detector.detectar_en_imagen("frame.jpg")

# Procesar múltiples frames
resultados = detector.procesar_frames("frames_extraidos/")
conteo = detector.contar_productos(resultados)
detector.exportar_csv(conteo, "inventario.csv")
```

### 3. `ReconocedorMarcas` (reconocer_marcas.py)

Reconocimiento de marcas con OCR:

```python
from src.reconocer_marcas import ReconocedorMarcas

reconocedor = ReconocedorMarcas(metodo='easyocr')
marca = reconocedor.identificar_marca(
    textos_con_confianza=[("Susante", 0.9)],
    marcas_conocidas=["Susante", "Levite"]
)
```

---

## 📊 Formato de Salida

### CSV de Inventario

```csv
Producto/Marca,Cantidad Detectada,Fecha
bottle_Susante,4,2026-01-09 11:06:04
bottle_Levite,2,2026-01-09 11:06:04
bottle,3,2026-01-09 11:06:04
```

### JSON de Metadatos

```json
{
  "fecha": "2026-01-09T11:06:04",
  "total_frames": 7,
  "total_skus": 3,
  "total_productos": 9,
  "conteo": {
    "bottle_Susante": 4,
    "bottle_Levite": 2,
    "bottle": 3
  }
}
```

---

## 🎯 Modelos y Entrenamiento

### Modelo Actual

El sistema usa un modelo **pre-entrenado genérico** (YOLOv8n COCO) que detecta objetos comunes:
- `bottle`, `cup`, `bowl`, `spoon`, etc.

**Limitación**: No detecta productos específicos de góndola.

### Entrenar Modelo Personalizado

Para detectar productos específicos, necesitas entrenar tu propio modelo. **Recomendamos usar Roboflow** para facilitar el proceso:

#### Opción 1: Usar Roboflow (⭐ Recomendado)

1. **Crear cuenta** en https://roboflow.com (gratis)
2. **Subir imágenes** de tus videos
3. **Anotar productos** en la interfaz web (drag & drop)
4. **Exportar dataset** en formato YOLOv8
5. **Descargar** usando el script de integración:
   ```bash
   pip install roboflow
   python scripts/integrar_roboflow.py --api-key TU_KEY --workspace WORKSPACE --project PROYECTO
   ```
6. **Entrenar** con tu script existente:
   ```bash
   python scripts/entrenar_modelo.py --dataset datos/datasets/PROYECTO/data.yaml
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

#### Opción 3: Pre-anotación con APIs

Usar APIs (Google Vision, AWS Rekognition) para pre-anotar imágenes y acelerar el proceso:

```bash
python scripts/pre_anotar_con_api.py --imagenes data/frames/ --google-key key.json
```

**Ver documentación completa:** `docs/GUIA_APIS_ENTRENAMIENTO.md`

---

## 🧪 Testing y Validación

### Probar Detección en Imagen

```bash
python scripts/probar_deteccion.py
```

### Verificar Modelo

```bash
python scripts/probar_deteccion.py --imagen output/.../frames_extraidos/frame_0000.jpg
```

---

## 📝 Notas Importantes

### Limitaciones Actuales

1. **Modelo genérico**: Detecta objetos comunes, no productos específicos
2. **OCR**: Puede tener errores con texto borroso o iluminación pobre
3. **Conteo**: Cuenta productos visibles, no totales en góndola
4. **MVP**: Sistema diseñado para validación, no reemplazo completo de conteo manual

### Mejoras Futuras

- [ ] Entrenar modelo específico para productos de góndola
- [ ] Mejorar precisión de OCR con preprocesamiento
- [ ] Implementar tracking para evitar conteos duplicados
- [ ] Interfaz web para validación manual
- [ ] Integración con sistemas de inventario

---

## 🐛 Troubleshooting

### Error: "Modelo no encontrado"

```bash
# Descargar modelo pre-entrenado
python scripts/descargar_modelo.py
```

### Error: "EasyOCR no disponible"

```bash
pip install easyocr
```

### Error: "No se detectan marcas"

1. Verificar que `marcas_conocidas.txt` existe y tiene marcas
2. Verificar que el OCR está funcionando (revisar logs)
3. Ajustar confianza mínima: `--confianza 0.2`

### Rendimiento Lento

- Usar `--sin-anotaciones` para no generar imágenes
- Usar `--sin-marcas` para desactivar OCR
- Reducir FPS de extracción: `--fps 0.5`

---

## 📚 Referencias

- **YOLOv8**: https://github.com/ultralytics/ultralytics
- **EasyOCR**: https://github.com/JaidedAI/EasyOCR
- **OpenCV**: https://opencv.org/

---

## 👥 Soporte

Para preguntas o problemas:
1. Revisar este README
2. Verificar logs en consola
3. Revisar archivos de salida en `output/`

---

## 📄 Licencia

MVP desarrollado para Carrefour - Uso interno

---

**Última actualización**: Enero 2026

