#!/usr/bin/env python3
"""
Script de prueba para identificación SKU
Permite probar el sistema de retrieval visual con el catálogo de imágenes
"""

import sys
import os
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.identificar_sku_retrieval import IdentificadorSKURetrieval


def main():
    """Prueba del identificador SKU"""
    print("\n" + "=" * 70)
    print("🧪 PRUEBA DE IDENTIFICACIÓN SKU - RETRIEVAL VISUAL")
    print("=" * 70)
    
    # Configuración
    catalogo_dir = Path(__file__).parent.parent / "imagenes"
    embeddings_path = catalogo_dir.parent / "embeddings.pkl"
    
    # Verificar que el catálogo existe
    if not catalogo_dir.exists():
        print(f"\n❌ Error: Catálogo no encontrado en: {catalogo_dir}")
        print(f"\n💡 Ejecuta primero: python buscarimagenes.py --input eans.txt --out imagenes")
        return 1
    
    print(f"\n📂 Catálogo: {catalogo_dir}")
    
    # Inicializar identificador
    try:
        print("\n🔧 Inicializando identificador SKU...")
        identificador = IdentificadorSKURetrieval(
            catalogo_dir=str(catalogo_dir),
            embeddings_path=str(embeddings_path)
        )
    except Exception as e:
        print(f"\n❌ Error inicializando identificador: {e}")
        return 1
    
    # Mostrar estadísticas
    stats = identificador.get_estadisticas()
    print("\n📊 ESTADÍSTICAS DEL CATÁLOGO:")
    print("─" * 70)
    print(f"   Total de SKUs (EANs): {stats['total_skus']}")
    print(f"   Total de imágenes: {stats['total_imagenes']}")
    print(f"   Promedio por SKU: {stats['promedio_imagenes_por_sku']:.1f}")
    
    # Listar algunos SKUs
    print(f"\n📦 SKUs en el catálogo:")
    print("─" * 70)
    for i, (ean, embeddings) in enumerate(list(identificador.catalogo_embeddings.items())[:10], 1):
        print(f"   {i}. {ean} ({len(embeddings)} imágenes)")
    if len(identificador.catalogo_embeddings) > 10:
        print(f"   ... y {len(identificador.catalogo_embeddings) - 10} más")
    
    # Prueba de auto-identificación (test con imágenes del propio catálogo)
    print(f"\n🔍 PRUEBA DE AUTO-IDENTIFICACIÓN:")
    print("─" * 70)
    print("Probando identificación con imágenes del mismo catálogo...")
    
    # Tomar una imagen de cada SKU para probar
    test_cases = []
    for ean_dir in sorted(catalogo_dir.iterdir())[:5]:  # Solo los primeros 5
        if ean_dir.is_dir():
            imagenes = list(ean_dir.glob("*.jpg")) + \
                      list(ean_dir.glob("*.png")) + \
                      list(ean_dir.glob("*.webp"))
            if imagenes:
                test_cases.append({
                    'ean_real': ean_dir.name,
                    'imagen': imagenes[0]
                })
    
    if not test_cases:
        print("\n⚠️  No se encontraron imágenes para probar")
        return 0
    
    correctos = 0
    total = len(test_cases)
    
    print(f"\nProbando con {total} imágenes...\n")
    
    for i, test in enumerate(test_cases, 1):
        resultado = identificador.identificar(
            str(test['imagen']),
            threshold=0.5
        )
        
        ean_predicho = resultado['ean']
        confianza = resultado['confianza']
        correcto = ean_predicho == test['ean_real']
        
        if correcto:
            correctos += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"{status} Test {i}/{total}:")
        print(f"   Imagen: {test['imagen'].name}")
        print(f"   EAN real: {test['ean_real']}")
        print(f"   EAN predicho: {ean_predicho}")
        print(f"   Confianza: {confianza:.3f}")
        
        if not correcto and resultado['top_matches']:
            print(f"   Top 3 matches:")
            for ean, sim in resultado['top_matches'][:3]:
                print(f"      {ean}: {sim:.3f}")
        print()
    
    # Resumen
    precision = (correctos / total * 100) if total > 0 else 0
    
    print("─" * 70)
    print(f"📈 RESULTADOS:")
    print(f"   Correctos: {correctos}/{total}")
    print(f"   Precisión: {precision:.1f}%")
    print("=" * 70)
    
    # Recomendaciones
    if precision < 80:
        print("\n💡 RECOMENDACIONES:")
        print("   - El catálogo necesita más imágenes por SKU (mínimo 10)")
        print("   - Asegurar que las imágenes sean de buena calidad")
        print("   - Verificar que los EANs estén correctamente etiquetados")
    elif precision >= 80:
        print("\n✅ El sistema está listo para usar en producción")
        print("   El catálogo tiene buena cobertura y calidad")
    
    # Instrucciones de uso
    print("\n📝 CÓMO USAR EN PRODUCCIÓN:")
    print("─" * 70)
    print("1. Procesar video con detección + crops:")
    print("   python run.py video.MOV --guardar-crops")
    print()
    print("2. Procesar video con identificación SKU completa:")
    print("   python run.py video.MOV --identificar-sku --catalogo imagenes/")
    print()
    print("3. Ver resultados:")
    print("   cat output/.../reporte_deteccion/inventario_sku.csv")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
