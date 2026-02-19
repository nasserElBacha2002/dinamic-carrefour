IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = 'dinamic_carrefour')
BEGIN
    CREATE DATABASE dinamic_carrefour;
END
GO

USE dinamic_carrefour;
GO

IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'categorias')
BEGIN
    CREATE TABLE categorias (
        id              VARCHAR(50)     PRIMARY KEY,
        nombre          NVARCHAR(100)   NOT NULL,
        descripcion     NVARCHAR(500)   NULL,
        prompts_clip    NVARCHAR(MAX)   NULL,
        fecha_creacion  DATETIME        DEFAULT GETDATE(),
        activo          BIT             DEFAULT 1
    );
END
GO

IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'productos')
BEGIN
    CREATE TABLE productos (
        ean                     VARCHAR(50)     PRIMARY KEY,
        descripcion             NVARCHAR(500)   NOT NULL,
        categoria_id            VARCHAR(50)     NOT NULL,
        n_imagenes              INT             DEFAULT 0,
        embeddings_calculados   BIT             DEFAULT 0,
        embeddings_path         NVARCHAR(500)   NULL,
        fecha_alta              DATETIME        DEFAULT GETDATE(),
        fecha_embeddings        DATETIME        NULL,
        activo                  BIT             DEFAULT 1,
        CONSTRAINT FK_productos_categorias
            FOREIGN KEY (categoria_id) REFERENCES categorias(id)
    );
END
GO

IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'ejecuciones')
BEGIN
    CREATE TABLE ejecuciones (
        id                      INT IDENTITY(1,1)   PRIMARY KEY,
        video_path              NVARCHAR(500)       NOT NULL,
        video_nombre            NVARCHAR(200)       NULL,
        fecha                   DATETIME            DEFAULT GETDATE(),
        frames_procesados       INT                 NULL,
        frames_con_producto     INT                 NULL,
        total_detecciones       INT                 NULL,
        skus_identificados      INT                 NULL,
        duracion_segundos       FLOAT               NULL,
        parametros              NVARCHAR(MAX)       NULL,
        output_dir              NVARCHAR(500)       NULL,
        csv_path                NVARCHAR(500)       NULL
    );
END
GO

IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'detecciones')
BEGIN
    CREATE TABLE detecciones (
        id                  INT IDENTITY(1,1)   PRIMARY KEY,
        ejecucion_id        INT                 NOT NULL,
        ean                 VARCHAR(50)         NOT NULL,
        cantidad_dedup      INT                 NOT NULL,
        cantidad_raw        INT                 NULL,
        confianza_promedio  FLOAT               NULL,
        CONSTRAINT FK_detecciones_ejecuciones
            FOREIGN KEY (ejecucion_id) REFERENCES ejecuciones(id)
            ON DELETE CASCADE,
        CONSTRAINT FK_detecciones_productos
            FOREIGN KEY (ean) REFERENCES productos(ean)
    );
END
GO

IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'categoria_coco_mapeo')
BEGIN
    CREATE TABLE categoria_coco_mapeo (
        id                  INT IDENTITY(1,1)   PRIMARY KEY,
        categoria_id        VARCHAR(50)         NOT NULL,
        coco_class_id        INT                 NOT NULL,
        fecha_creacion       DATETIME            DEFAULT GETDATE(),
        activo              BIT                 DEFAULT 1,
        CONSTRAINT FK_coco_mapeo_categorias
            FOREIGN KEY (categoria_id) REFERENCES categorias(id)
            ON DELETE CASCADE,
        CONSTRAINT UQ_categoria_coco UNIQUE (categoria_id, coco_class_id)
    );
END
GO

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_productos_categoria')
    CREATE INDEX IX_productos_categoria ON productos(categoria_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_productos_activo')
    CREATE INDEX IX_productos_activo ON productos(activo);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_detecciones_ejecucion')
    CREATE INDEX IX_detecciones_ejecucion ON detecciones(ejecucion_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_detecciones_ean')
    CREATE INDEX IX_detecciones_ean ON detecciones(ean);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_ejecuciones_fecha')
    CREATE INDEX IX_ejecuciones_fecha ON ejecuciones(fecha DESC);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_coco_mapeo_categoria')
    CREATE INDEX IX_coco_mapeo_categoria ON categoria_coco_mapeo(categoria_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_coco_mapeo_coco_id')
    CREATE INDEX IX_coco_mapeo_coco_id ON categoria_coco_mapeo(coco_class_id);
GO

MERGE INTO categorias AS target
USING (VALUES
    ('botella',  'Botella',  'Botellas PET o vidrio (gaseosas, agua, jugos)',
     '["a plastic bottle on a shelf","a PET bottle of soda","a water bottle","a beverage bottle","a plastic drink bottle"]'),

    ('lata',     'Lata',     'Latas metálicas (gaseosas, cerveza, conservas)',
     '["a metal can on a shelf","an aluminum soda can","a beer can","a canned drink","a tin can of food"]'),

    ('bolsa',    'Bolsa',    'Bolsas/paquetes flexibles (snacks, papas fritas, pan)',
     '["a bag of chips on a shelf","a plastic bag of snacks","a bag of bread","a flexible plastic package","a bag of food on a shelf"]'),

    ('caja',     'Caja',     'Cajas de cartón (cereales, galletitas, té)',
     '["a cardboard box on a shelf","a cereal box","a box of cookies","a rectangular cardboard package","a tea box"]'),

    ('paquete',  'Paquete',  'Paquetes cerrados/metalizados (yerba, café, arroz)',
     '["a sealed package of yerba mate","a foil package of coffee","a sealed bag of rice","a metallic sealed package","a vacuum sealed food package"]'),

    ('tubo',     'Tubo',     'Tubos plásticos/aluminio (pasta dental, cremas)',
     '["a tube of toothpaste","a plastic tube on a shelf","a squeezable tube","a cosmetic tube","a hygiene product tube"]'),

    ('frasco',   'Frasco',   'Frascos de vidrio (mermeladas, salsas)',
     '["a glass jar on a shelf","a jar of jam","a sauce bottle","a glass container with lid","a small glass jar of food"]')
) AS source (id, nombre, descripcion, prompts_clip)
ON target.id = source.id
WHEN NOT MATCHED THEN
    INSERT (id, nombre, descripcion, prompts_clip)
    VALUES (source.id, source.nombre, source.descripcion, source.prompts_clip);
GO

-- Mapeo de categorías a clases COCO
-- COCO class IDs: 39=bottle, 40=wine glass, 41=cup, 44=handbag, 27=backpack
MERGE INTO categoria_coco_mapeo AS target
USING (VALUES
    ('botella', 39),  -- bottle
    ('botella', 40),  -- wine glass
    ('botella', 41),  -- cup
    ('lata', 39),     -- bottle
    ('bolsa', 44),    -- handbag
    ('bolsa', 27),    -- backpack
    ('caja', 39),     -- bottle
    ('paquete', 44),  -- handbag
    ('paquete', 27),  -- backpack
    ('tubo', 39),     -- bottle
    ('frasco', 39),   -- bottle
    ('frasco', 40)    -- wine glass
) AS source (categoria_id, coco_class_id)
ON target.categoria_id = source.categoria_id AND target.coco_class_id = source.coco_class_id
WHEN NOT MATCHED THEN
    INSERT (categoria_id, coco_class_id)
    VALUES (source.categoria_id, source.coco_class_id);
GO

IF EXISTS (SELECT * FROM sys.views WHERE name = 'v_catalogo_resumen')
    DROP VIEW v_catalogo_resumen;
GO

CREATE VIEW v_catalogo_resumen AS
SELECT
    c.id                    AS categoria,
    c.nombre                AS categoria_nombre,
    COUNT(p.ean)            AS total_productos,
    SUM(p.n_imagenes)       AS total_imagenes,
    SUM(CAST(p.embeddings_calculados AS INT)) AS con_embeddings,
    COUNT(p.ean) - SUM(CAST(p.embeddings_calculados AS INT)) AS sin_embeddings
FROM categorias c
LEFT JOIN productos p ON p.categoria_id = c.id AND p.activo = 1
WHERE c.activo = 1
GROUP BY c.id, c.nombre;
GO

IF EXISTS (SELECT * FROM sys.views WHERE name = 'v_historial_ejecuciones')
    DROP VIEW v_historial_ejecuciones;
GO

CREATE VIEW v_historial_ejecuciones AS
SELECT
    e.id,
    e.video_nombre,
    e.fecha,
    e.frames_procesados,
    e.total_detecciones,
    e.skus_identificados,
    e.duracion_segundos,
    COUNT(d.id)                     AS productos_distintos,
    SUM(d.cantidad_dedup)           AS total_unidades_dedup
FROM ejecuciones e
LEFT JOIN detecciones d ON d.ejecucion_id = e.id
GROUP BY e.id, e.video_nombre, e.fecha, e.frames_procesados,
         e.total_detecciones, e.skus_identificados, e.duracion_segundos;
GO

PRINT 'Base de datos dinamic_carrefour creada correctamente.';
GO