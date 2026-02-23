# Comandos para agregar nuevos productos

## Productos agregados
- 7790387015539 - YERBA LA MERCED CAMPO Y MONTE 500G
- 7790387015645 - YERBA LA MERCED SUR 500G
- 7790387015522 - YERBA LA MERCED BARBACUA 500G
- 7790387015515 - YERBA LA MERCED DE MONTE 500G

---

## Paso 1: Sincronizar con la base de datos

Sincroniza los nuevos productos desde `eans.txt` a la base de datos SQL Server:

```bash
cd "/Users/nasserelbacha/Documents/Dinamic sistems/dinamic-carrefour"
python scripts/init_db.py --sync
```

Esto agregará los nuevos productos a la tabla `productos` en la base de datos.

---

## Paso 2: Descargar imágenes de referencia

Descarga imágenes de referencia para cada producto usando Bing Images Search:

```bash
python scripts/buscarimagenes.py --input eans.txt --out imagenes --per-ean 10
```

**Opciones:**
- `--input eans.txt`: Archivo con los EANs
- `--out imagenes`: Directorio donde se guardarán las imágenes
- `--per-ean 10`: Cantidad de imágenes a descargar por producto (ajustá según necesites)

Las imágenes se guardarán en:
```
imagenes/
  7790387015539/
    imagen_001.jpg
    imagen_002.jpg
    ...
  7790387015645/
    ...
```

---

## Paso 3: Generar embeddings (entrenar el modelo)

Genera los embeddings CLIP para los nuevos productos. Esto es lo que "entrena" al modelo de identificación:

### Opción A: Generar embeddings para TODOS los productos
```bash
python scripts/agregar_sku.py --todos
```

### Opción B: Generar embeddings solo para los nuevos productos
```bash
python scripts/agregar_sku.py --ean 7790387015539
python scripts/agregar_sku.py --ean 7790387015645
python scripts/agregar_sku.py --ean 7790387015522
python scripts/agregar_sku.py --ean 7790387015515
```

### Opción C: Forzar recálculo (si ya existían)
```bash
python scripts/agregar_sku.py --ean 7790387015539 --forzar
python scripts/agregar_sku.py --ean 7790387015645 --forzar
python scripts/agregar_sku.py --ean 7790387015522 --forzar
python scripts/agregar_sku.py --ean 7790387015515 --forzar
```

Los embeddings se guardarán en:
```
catalog/embeddings/
  7790387015539.npy
  7790387015645.npy
  7790387015522.npy
  7790387015515.npy
```

---

## Verificar estado

Para ver el estado actual del catálogo:

```bash
# Estado en la base de datos
python scripts/init_db.py --status

# Estado de embeddings
python scripts/agregar_sku.py --status
```

---

## Notas importantes

1. **Formato de eans.txt**: Debe usar **TABs** (no espacios) entre columnas:
   ```
   EAN<TAB>DESCRIPCION<TAB>CATEGORIA
   ```

2. **Categorías válidas**: `botella`, `lata`, `bolsa`, `caja`, `paquete`, `tubo`, `frasco`

3. **Imágenes mínimas**: Se recomienda tener al menos 3-5 imágenes por producto para mejor identificación

4. **Modelo CLIP**: Por defecto usa `ViT-B/16`. Podés cambiarlo con la variable de entorno:
   ```bash
   export CLIP_MODEL="ViT-L/14"  # Para mejor precisión (más lento)
   ```

---

## Flujo completo (copiar y pegar)

```bash
cd "/Users/nasserelbacha/Documents/Dinamic sistems/dinamic-carrefour"

# 1. Sincronizar DB
python scripts/init_db.py --sync

# 2. Descargar imágenes
python scripts/buscarimagenes.py --input eans.txt --out imagenes --per-ean 10

# 3. Generar embeddings para los nuevos productos
python scripts/agregar_sku.py --ean 7790387015539
python scripts/agregar_sku.py --ean 7790387015645
python scripts/agregar_sku.py --ean 7790387015522
python scripts/agregar_sku.py --ean 7790387015515

# 4. Verificar
python scripts/agregar_sku.py --status
```
