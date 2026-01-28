#!/usr/bin/env python3
"""
Sistema de identificación de SKU usando retrieval visual
Compara crops de productos detectados con catálogo de imágenes de referencia
MVP - Sistema de Inventario de Góndolas para Carrefour
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# Imports para feature extraction
try:
    import torch
    from torchvision import models, transforms
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch no disponible. Instalar con: pip install torch torchvision")

# Sklearn para similitud
try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn no disponible. Instalar con: pip install scikit-learn")


class IdentificadorSKURetrieval:
    """
    Identifica SKU (EAN) de productos usando retrieval visual
    Compara crops contra catálogo de imágenes de referencia usando embeddings
    """
    
    def __init__(self, catalogo_dir: str, embeddings_path: Optional[str] = None,
                 modelo: str = 'resnet50'):
        """
        Inicializa el identificador SKU
        
        Args:
            catalogo_dir: Ruta al directorio con imágenes de referencia
                         Estructura: catalogo_dir/EAN/imagen_001.jpg
            embeddings_path: Ruta a archivo con embeddings pre-calculados (opcional)
            modelo: Modelo para feature extraction ('resnet50', 'mobilenet_v2')
        """
        if not TORCH_AVAILABLE or not SKLEARN_AVAILABLE:
            raise ImportError(
                "PyTorch y scikit-learn son requeridos. "
                "Instalar con: pip install torch torchvision scikit-learn"
            )
        
        self.catalogo_dir = Path(catalogo_dir)
        self.embeddings_path = embeddings_path
        self.modelo_nombre = modelo
        
        if not self.catalogo_dir.exists():
            raise FileNotFoundError(f"Catálogo no encontrado: {catalogo_dir}")
        
        # Inicializar modelo de feature extraction
        self._init_modelo(modelo)
        
        # Cargar o generar embeddings
        if embeddings_path and Path(embeddings_path).exists():
            print(f"📂 Cargando embeddings desde: {embeddings_path}")
            self.catalogo_embeddings = self._load_embeddings(embeddings_path)
        else:
            print(f"🔄 Generando embeddings del catálogo...")
            self.catalogo_embeddings = self._generar_embeddings_catalogo()
            if embeddings_path:
                self.save_embeddings(embeddings_path)
        
        print(f"✅ Identificador SKU listo con {len(self.catalogo_embeddings)} SKUs")
    
    def _init_modelo(self, modelo_nombre: str):
        """Inicializa el modelo de feature extraction"""
        print(f"🔧 Inicializando modelo: {modelo_nombre}")
        
        if modelo_nombre == 'resnet50':
            modelo_completo = models.resnet50(pretrained=True)
            # Remover última capa (clasificador)
            self.modelo = torch.nn.Sequential(*list(modelo_completo.children())[:-1])
        elif modelo_nombre == 'mobilenet_v2':
            modelo_completo = models.mobilenet_v2(pretrained=True)
            self.modelo = modelo_completo.features
        else:
            raise ValueError(f"Modelo no soportado: {modelo_nombre}")
        
        self.modelo.eval()
        
        # No usar GPU para MVP (más compatible)
        self.device = 'cpu'
        self.modelo.to(self.device)
        
        # Transformaciones estándar para ImageNet
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _extract_features(self, imagen_path: str) -> np.ndarray:
        """
        Extrae embedding de una imagen
        
        Args:
            imagen_path: Ruta a la imagen
        
        Returns:
            Vector de features (embedding)
        """
        try:
            img = Image.open(imagen_path).convert('RGB')
        except Exception as e:
            print(f"⚠️  Error leyendo imagen {imagen_path}: {e}")
            return None
        
        # Transformar imagen
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Extraer features
        with torch.no_grad():
            features = self.modelo(img_tensor)
        
        # Aplanar y convertir a numpy
        embedding = features.squeeze().cpu().numpy()
        
        # Normalizar (para cosine similarity)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return embedding
    
    def _generar_embeddings_catalogo(self) -> Dict[str, List[np.ndarray]]:
        """
        Genera embeddings para todas las imágenes del catálogo
        
        Returns:
            Diccionario {EAN: [embedding1, embedding2, ...]}
        """
        embeddings = {}
        
        # Listar carpetas de EANs
        ean_dirs = [d for d in self.catalogo_dir.iterdir() if d.is_dir()]
        
        if not ean_dirs:
            print(f"⚠️  No se encontraron carpetas de EANs en: {self.catalogo_dir}")
            return embeddings
        
        print(f"📦 Procesando {len(ean_dirs)} SKUs...")
        
        for ean_dir in sorted(ean_dirs):
            ean = ean_dir.name
            embeddings[ean] = []
            
            # Buscar todas las imágenes en la carpeta
            imagenes = list(ean_dir.glob("*.jpg")) + \
                      list(ean_dir.glob("*.png")) + \
                      list(ean_dir.glob("*.webp"))
            
            if not imagenes:
                print(f"   ⚠️  {ean}: Sin imágenes")
                continue
            
            for img_path in imagenes:
                emb = self._extract_features(str(img_path))
                if emb is not None:
                    embeddings[ean].append(emb)
            
            print(f"   ✓ {ean}: {len(embeddings[ean])} embeddings")
        
        return embeddings
    
    def identificar(self, crop_path: str, top_k: int = 3, 
                   threshold: float = 0.5) -> Dict:
        """
        Identifica el SKU (EAN) de un crop
        
        Args:
            crop_path: Ruta al crop del producto detectado
            top_k: Número de candidatos a retornar
            threshold: Umbral de similitud (0-1) para considerar match válido
        
        Returns:
            Dict con:
                - ean: EAN identificado (o 'UNKNOWN')
                - confianza: similitud máxima (0-1)
                - top_matches: Lista de (ean, similitud) para top K
        """
        # Extraer features del crop
        crop_emb = self._extract_features(crop_path)
        
        if crop_emb is None:
            return {
                'ean': 'UNKNOWN',
                'confianza': 0.0,
                'top_matches': []
            }
        
        # Comparar con todos los SKUs del catálogo
        similitudes = {}
        
        for ean, ref_embeddings in self.catalogo_embeddings.items():
            if not ref_embeddings:
                continue
            
            # Calcular similitud máxima contra todas las referencias de este SKU
            sims = []
            for ref_emb in ref_embeddings:
                # Similitud coseno (ya normalizados, es solo dot product)
                sim = float(np.dot(crop_emb, ref_emb))
                sims.append(sim)
            
            # Usar máxima similitud
            similitudes[ean] = max(sims) if sims else 0.0
        
        if not similitudes:
            return {
                'ean': 'UNKNOWN',
                'confianza': 0.0,
                'top_matches': []
            }
        
        # Ordenar por similitud
        ranked = sorted(similitudes.items(), key=lambda x: x[1], reverse=True)
        
        best_ean, best_sim = ranked[0]
        
        # Threshold de confianza
        if best_sim < threshold:
            return {
                'ean': 'UNKNOWN',
                'confianza': best_sim,
                'top_matches': ranked[:top_k]
            }
        
        return {
            'ean': best_ean,
            'confianza': best_sim,
            'top_matches': ranked[:top_k]
        }
    
    def identificar_batch(self, crops_paths: List[str], threshold: float = 0.5) -> List[Dict]:
        """
        Identifica múltiples crops de una vez
        
        Args:
            crops_paths: Lista de rutas a crops
            threshold: Umbral de similitud
        
        Returns:
            Lista de resultados de identificación
        """
        resultados = []
        
        for crop_path in crops_paths:
            resultado = self.identificar(crop_path, threshold=threshold)
            resultado['crop_path'] = crop_path
            resultados.append(resultado)
        
        return resultados
    
    def save_embeddings(self, output_path: str):
        """Guarda embeddings para reutilizar"""
        print(f"💾 Guardando embeddings en: {output_path}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(self.catalogo_embeddings, f)
        
        print(f"✅ Embeddings guardados")
    
    def _load_embeddings(self, path: str) -> Dict[str, List[np.ndarray]]:
        """Carga embeddings pre-calculados"""
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def get_estadisticas(self) -> Dict:
        """Retorna estadísticas del catálogo"""
        total_skus = len(self.catalogo_embeddings)
        total_imagenes = sum(len(embs) for embs in self.catalogo_embeddings.values())
        
        return {
            'total_skus': total_skus,
            'total_imagenes': total_imagenes,
            'promedio_imagenes_por_sku': total_imagenes / total_skus if total_skus > 0 else 0
        }


def main():
    """Función principal para pruebas"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Identificar SKU de productos usando retrieval visual'
    )
    parser.add_argument('--catalogo', required=True, 
                       help='Directorio con imágenes de referencia (EAN/imagen.jpg)')
    parser.add_argument('--crop', 
                       help='Crop de producto a identificar')
    parser.add_argument('--crops-dir',
                       help='Directorio con crops a identificar')
    parser.add_argument('--embeddings', default='embeddings.pkl',
                       help='Archivo para guardar/cargar embeddings')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Umbral de similitud (0-1)')
    parser.add_argument('--generar-embeddings', action='store_true',
                       help='Solo generar embeddings y salir')
    
    args = parser.parse_args()
    
    # Inicializar identificador
    identificador = IdentificadorSKURetrieval(
        catalogo_dir=args.catalogo,
        embeddings_path=args.embeddings
    )
    
    # Mostrar estadísticas
    stats = identificador.get_estadisticas()
    print("\n📊 Estadísticas del catálogo:")
    print(f"   Total SKUs: {stats['total_skus']}")
    print(f"   Total imágenes: {stats['total_imagenes']}")
    print(f"   Promedio por SKU: {stats['promedio_imagenes_por_sku']:.1f}")
    
    if args.generar_embeddings:
        print("\n✅ Embeddings generados y guardados")
        return
    
    # Identificar crop individual
    if args.crop:
        print(f"\n🔍 Identificando: {args.crop}")
        resultado = identificador.identificar(args.crop, threshold=args.threshold)
        
        print(f"\n📦 Resultado:")
        print(f"   EAN: {resultado['ean']}")
        print(f"   Confianza: {resultado['confianza']:.3f}")
        print(f"\n   Top 3 matches:")
        for ean, sim in resultado['top_matches'][:3]:
            print(f"      {ean}: {sim:.3f}")
    
    # Identificar directorio de crops
    if args.crops_dir:
        crops_dir = Path(args.crops_dir)
        if not crops_dir.exists():
            print(f"❌ Error: Directorio no encontrado: {crops_dir}")
            return
        
        crops = list(crops_dir.glob("*.jpg")) + list(crops_dir.glob("*.png"))
        
        if not crops:
            print(f"⚠️  No se encontraron crops en: {crops_dir}")
            return
        
        print(f"\n🔍 Identificando {len(crops)} crops...")
        resultados = identificador.identificar_batch(
            [str(c) for c in crops],
            threshold=args.threshold
        )
        
        # Resumen
        identificados = sum(1 for r in resultados if r['ean'] != 'UNKNOWN')
        print(f"\n✅ Identificados: {identificados}/{len(resultados)}")
        
        # Mostrar resultados
        for resultado in resultados:
            crop_name = Path(resultado['crop_path']).name
            print(f"   {crop_name}: {resultado['ean']} ({resultado['confianza']:.3f})")


if __name__ == "__main__":
    main()
