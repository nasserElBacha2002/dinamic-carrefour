#!/usr/bin/env python3
"""
Factory de componentes (Roboflow-only).
"""

from typing import Optional, Dict, Any

try:
    from src.detectar_roboflow import DetectorRoboflow
    ROBOFLOW_AVAILABLE = True
except ImportError:
    ROBOFLOW_AVAILABLE = False


class ComponentFactory:
    """Factory para crear componentes del sistema en modo Roboflow-only."""

    @staticmethod
    def crear_detector_roboflow(
        api_key: str,
        workspace_name: str = "gondolacarrefour",
        workflow_id: str = "find-bottles-pepsis-pepsi-1s-pepsi-blacks-and-5-lts",
        label_map_path: Optional[str] = None,
        confianza_minima: float = 0.25
    ):
        if not ROBOFLOW_AVAILABLE:
            print("⚠️  DetectorRoboflow no disponible (error de importación)")
            return None

        try:
            return DetectorRoboflow(
                api_key=api_key,
                workspace_name=workspace_name,
                workflow_id=workflow_id,
                label_map_path=label_map_path,
                confianza_minima=confianza_minima
            )
        except Exception as e:
            print(f"⚠️  Error creando detector Roboflow: {e}")
            return None

    @staticmethod
    def desde_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crea componentes desde diccionario de configuración (solo Roboflow).

        Args:
            config: Diccionario con:
                - roboflow_api_key: str (requerido)
                - roboflow_workspace: str
                - roboflow_workflow: str
                - label_map_path: str
                - confianza_minima: float

        Returns:
            Dict con:
                - detector: DetectorRoboflow
                - identificador_sku: None
        """
        api_key = config.get('roboflow_api_key', '')
        if not api_key:
            raise ValueError("roboflow_api_key es requerido en modo Roboflow-only")

        detector = ComponentFactory.crear_detector_roboflow(
            api_key=api_key,
            workspace_name=config.get('roboflow_workspace', 'gondolacarrefour'),
            workflow_id=config.get(
                'roboflow_workflow',
                'find-bottles-pepsis-pepsi-1s-pepsi-blacks-and-5-lts'
            ),
            label_map_path=config.get('label_map_path'),
            confianza_minima=config.get('confianza_minima') or 0.25
        )

        if detector is None:
            raise RuntimeError("No se pudo crear DetectorRoboflow")

        return {
            'detector': detector,
            'identificador_sku': None
        }
