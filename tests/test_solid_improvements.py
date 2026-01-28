#!/usr/bin/env python3
"""
Script de testing para validar mejoras SOLID
Verifica que los cambios funcionan correctamente
"""

import sys
from pathlib import Path

def test_imports():
    """Test 1: Verificar que todos los imports funcionan"""
    print("=" * 70)
    print("TEST 1: VERIFICANDO IMPORTS")
    print("=" * 70)
    
    try:
        from src.protocols import (
            DetectorProtocol,
            ReconocedorMarcasProtocol,
            OCRStrategy,
            ReporteExporter
        )
        print("✅ Protocolos importados correctamente")
    except Exception as e:
        print(f"❌ Error importando protocolos: {e}")
        return False
    
    try:
        from src.ocr_strategies import (
            OCRStrategyBase,
            TesseractOCRStrategy,
            EasyOCRStrategy,
            DummyOCRStrategy,
            crear_ocr_strategy
        )
        print("✅ Estrategias OCR importadas correctamente")
    except Exception as e:
        print(f"❌ Error importando OCR strategies: {e}")
        return False
    
    try:
        from src.exporters import (
            ReporteExporterBase,
            CSVExporter,
            JSONExporter,
            MultiFormatExporter,
            crear_exporter
        )
        print("✅ Exportadores importados correctamente")
    except Exception as e:
        print(f"❌ Error importando exporters: {e}")
        return False
    
    try:
        from src.factory import ComponentFactory
        print("✅ Factory importado correctamente")
    except Exception as e:
        print(f"❌ Error importando factory: {e}")
        return False
    
    try:
        from src.reconocer_marcas import ReconocedorMarcas
        print("✅ ReconocedorMarcas importado correctamente")
    except Exception as e:
        print(f"❌ Error importando ReconocedorMarcas: {e}")
        return False
    
    try:
        from src.detectar_productos import DetectorProductos
        print("✅ DetectorProductos importado correctamente")
    except Exception as e:
        print(f"❌ Error importando DetectorProductos: {e}")
        return False
    
    try:
        from src.main import SistemaInventarioGondola
        print("✅ SistemaInventarioGondola importado correctamente")
    except Exception as e:
        print(f"❌ Error importando main: {e}")
        return False
    
    print("\n✅ Todos los imports funcionan correctamente\n")
    return True


def test_factory_creation():
    """Test 2: Verificar creación de componentes con Factory"""
    print("=" * 70)
    print("TEST 2: VERIFICANDO COMPONENT FACTORY")
    print("=" * 70)
    
    try:
        from src.factory import ComponentFactory
        
        # Test: Crear reconocedor de marcas con dummy OCR
        print("\n📦 Creando reconocedor de marcas con DummyOCR...")
        reconocedor = ComponentFactory.crear_reconocedor_marcas(ocr_metodo='dummy')
        print(f"✅ Reconocedor creado: {type(reconocedor).__name__}")
        
        # Test: Crear detector con dependencias
        print("\n📦 Creando detector con dependencias...")
        detector = ComponentFactory.crear_detector(
            with_reconocedor_marcas=True,
            ocr_metodo='dummy',
            export_formato='csv'
        )
        print(f"✅ Detector creado: {type(detector).__name__}")
        print(f"   - Reconocedor: {type(detector.reconocedor_marcas).__name__ if detector.reconocedor_marcas else 'None'}")
        print(f"   - Exporter: {type(detector.exporter).__name__}")
        
        # Test: Crear desde configuración
        print("\n📦 Creando componentes desde configuración...")
        config = {
            'modelo_path': None,
            'confianza_minima': 0.25,
            'reconocer_marcas': True,
            'ocr_metodo': 'dummy',
            'export_formato': 'json'
        }
        componentes = ComponentFactory.desde_config(config)
        print(f"✅ Componentes creados desde config:")
        print(f"   - Detector: {type(componentes['detector']).__name__}")
        print(f"   - Identificador SKU: {type(componentes['identificador_sku']).__name__ if componentes['identificador_sku'] else 'None'}")
        
        print("\n✅ Factory funciona correctamente\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Error en factory: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dependency_injection():
    """Test 3: Verificar inyección de dependencias"""
    print("=" * 70)
    print("TEST 3: VERIFICANDO DEPENDENCY INJECTION")
    print("=" * 70)
    
    try:
        from src.ocr_strategies import DummyOCRStrategy
        from src.reconocer_marcas import ReconocedorMarcas
        from src.exporters import JSONExporter
        from src.detectar_productos import DetectorProductos
        
        # Test: Inyectar estrategia OCR
        print("\n🔧 Inyectando DummyOCRStrategy en ReconocedorMarcas...")
        ocr_strategy = DummyOCRStrategy()
        reconocedor = ReconocedorMarcas(ocr_strategy=ocr_strategy)
        print(f"✅ Estrategia inyectada: {type(reconocedor.ocr_strategy).__name__}")
        
        # Test: Inyectar reconocedor y exporter en detector
        print("\n🔧 Inyectando dependencias en DetectorProductos...")
        exporter = JSONExporter()
        detector = DetectorProductos(
            reconocedor_marcas=reconocedor,
            exporter=exporter
        )
        print(f"✅ Reconocedor inyectado: {type(detector.reconocedor_marcas).__name__ if detector.reconocedor_marcas else 'None'}")
        print(f"✅ Exporter inyectado: {type(detector.exporter).__name__}")
        
        print("\n✅ Dependency Injection funciona correctamente\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Error en DI: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_exporters():
    """Test 4: Verificar exportadores"""
    print("=" * 70)
    print("TEST 4: VERIFICANDO EXPORTADORES")
    print("=" * 70)
    
    try:
        from src.exporters import CSVExporter, JSONExporter, crear_exporter
        import tempfile
        import os
        
        # Datos de prueba
        conteo = {
            'bottle_Susante': 5,
            'bottle_Levite': 3,
            'cup': 2
        }
        
        # Test CSV Exporter
        print("\n📄 Testeando CSVExporter...")
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test.csv")
            exporter_csv = CSVExporter()
            result = exporter_csv.exportar(conteo, csv_path)
            assert os.path.exists(csv_path), "CSV no fue creado"
            print(f"✅ CSV creado: {result}")
        
        # Test JSON Exporter
        print("\n📄 Testeando JSONExporter...")
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "test.json")
            exporter_json = JSONExporter()
            result = exporter_json.exportar(conteo, json_path)
            assert os.path.exists(json_path), "JSON no fue creado"
            print(f"✅ JSON creado: {result}")
        
        # Test Factory
        print("\n📄 Testeando crear_exporter...")
        exp1 = crear_exporter('csv')
        print(f"✅ CSV Exporter: {type(exp1).__name__}")
        exp2 = crear_exporter('json')
        print(f"✅ JSON Exporter: {type(exp2).__name__}")
        
        print("\n✅ Exportadores funcionan correctamente\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Error en exportadores: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Ejecutar todos los tests"""
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║                                                                      ║")
    print("║           🧪 TESTING DE MEJORAS SOLID                                ║")
    print("║                                                                      ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print("\n")
    
    resultados = []
    
    # Ejecutar tests
    resultados.append(("Imports", test_imports()))
    resultados.append(("Factory", test_factory_creation()))
    resultados.append(("Dependency Injection", test_dependency_injection()))
    resultados.append(("Exporters", test_exporters()))
    
    # Resumen
    print("=" * 70)
    print("RESUMEN DE TESTS")
    print("=" * 70)
    
    total = len(resultados)
    exitosos = sum(1 for _, passed in resultados if passed)
    
    for nombre, passed in resultados:
        estado = "✅ PASS" if passed else "❌ FAIL"
        print(f"{estado} - {nombre}")
    
    print("\n" + "=" * 70)
    print(f"RESULTADO FINAL: {exitosos}/{total} tests exitosos")
    print("=" * 70)
    
    if exitosos == total:
        print("\n🎉 ¡Todos los tests pasaron! Sistema listo para uso.\n")
        return 0
    else:
        print(f"\n⚠️  {total - exitosos} test(s) fallaron. Revisar errores.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
