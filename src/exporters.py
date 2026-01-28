#!/usr/bin/env python3
"""
Exportadores de reportes - Strategy Pattern
Implementa Open/Closed Principle para diferentes formatos de exportación
"""

import csv
import json
from abc import ABC, abstractmethod
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path


class ReporteExporterBase(ABC):
    """Clase base abstracta para exportadores de reportes"""
    
    @abstractmethod
    def exportar(
        self,
        conteo: Dict[str, int],
        output_path: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """Exporta conteo de productos a archivo"""
        pass
    
    def _agregar_metadata_default(self, metadata: Optional[Dict]) -> Dict:
        """Agrega metadata por defecto si no se proporciona"""
        if metadata is None:
            metadata = {}
        
        if 'fecha' not in metadata:
            metadata['fecha'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return metadata


class CSVExporter(ReporteExporterBase):
    """Exportador a formato CSV"""
    
    def exportar(
        self,
        conteo: Dict[str, int],
        output_path: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """Exporta a CSV con formato estándar"""
        metadata = self._agregar_metadata_default(metadata)
        fecha = metadata['fecha']
        
        # Separar productos con marca y genéricos
        productos_marca = {}
        productos_genericos = {}
        
        for sku, cantidad in sorted(conteo.items()):
            # Heurística: '_' indica marca identificada
            if '_' in sku:
                productos_marca[sku] = cantidad
            else:
                productos_genericos[sku] = cantidad
        
        # Escribir CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Producto/Marca', 'Cantidad Detectada', 'Fecha'])
            
            # Productos con marca primero
            for sku, cantidad in sorted(productos_marca.items()):
                writer.writerow([sku, cantidad, fecha])
            
            # Luego productos genéricos
            for sku, cantidad in sorted(productos_genericos.items()):
                writer.writerow([sku, cantidad, fecha])
        
        print(f"✅ CSV exportado: {Path(output_path).name}")
        print(f"   Total de SKUs: {len(conteo)}")
        print(f"   Total de productos: {sum(conteo.values())}")
        if productos_marca:
            print(f"   Productos con marca: {len(productos_marca)}")
        
        return output_path


class JSONExporter(ReporteExporterBase):
    """Exportador a formato JSON"""
    
    def exportar(
        self,
        conteo: Dict[str, int],
        output_path: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """Exporta a JSON con estructura completa"""
        metadata = self._agregar_metadata_default(metadata)
        
        # Estructura JSON
        reporte = {
            'metadata': metadata,
            'resumen': {
                'total_skus': len(conteo),
                'total_productos': sum(conteo.values()),
                'productos_con_marca': sum(1 for sku in conteo if '_' in sku)
            },
            'inventario': conteo
        }
        
        # Escribir JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(reporte, f, indent=2, ensure_ascii=False)
        
        print(f"✅ JSON exportado: {Path(output_path).name}")
        print(f"   Total de SKUs: {reporte['resumen']['total_skus']}")
        print(f"   Total de productos: {reporte['resumen']['total_productos']}")
        
        return output_path


class MultiFormatExporter(ReporteExporterBase):
    """Exportador que genera múltiples formatos a la vez"""
    
    def __init__(self, formatos: list = None):
        """
        Args:
            formatos: Lista de formatos ('csv', 'json')
        """
        if formatos is None:
            formatos = ['csv', 'json']
        
        self.exportadores = []
        for formato in formatos:
            if formato == 'csv':
                self.exportadores.append(('csv', CSVExporter()))
            elif formato == 'json':
                self.exportadores.append(('json', JSONExporter()))
    
    def exportar(
        self,
        conteo: Dict[str, int],
        output_path: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """Exporta a múltiples formatos"""
        rutas_generadas = []
        base_path = Path(output_path)
        
        for formato, exportador in self.exportadores:
            # Cambiar extensión según formato
            ruta_formato = base_path.with_suffix(f'.{formato}')
            ruta = exportador.exportar(conteo, str(ruta_formato), metadata)
            rutas_generadas.append(ruta)
        
        print(f"\n✅ Exportación multi-formato completada: {len(rutas_generadas)} archivos")
        return rutas_generadas[0] if rutas_generadas else output_path


def crear_exporter(formato: str = 'csv') -> ReporteExporterBase:
    """
    Factory function para crear exportadores
    
    Args:
        formato: 'csv', 'json', o 'multi'
        
    Returns:
        Instancia de ReporteExporterBase
    """
    if formato == 'csv':
        return CSVExporter()
    elif formato == 'json':
        return JSONExporter()
    elif formato == 'multi':
        return MultiFormatExporter(['csv', 'json'])
    else:
        print(f"⚠️  Formato '{formato}' no reconocido, usando CSV")
        return CSVExporter()
