# comp_conexas.py
"""
Módulo para análisis de componentes conexas en imágenes.

Provee funcionalidades para detección de regiones, cálculo de propiedades geométricas
y generación de visualizaciones con anotaciones.
"""

import cv2
from skimage.measure import label, regionprops
from typing import Tuple, List


class ConnectedComponentsAnalyzer:
    """
    Analizador de componentes conexas para extracción de características de regiones.

    Atributos:
        image_path (str): Ruta de la imagen a procesar
        min_area (int): Área mínima para considerar una región válida
    """

    def __init__(self, image_path: str, min_area: int = 10):
        """
        Inicializa el analizador con parámetros de configuración.

        Args:
            image_path: Ruta a la imagen de entrada
            min_area: Área mínima en píxeles para considerar una región (default: 10)
        """
        self.image_path = image_path
        self.min_area = min_area
        self._binary_image = None
        self._labeled_image = None
        self.regions = []
        self._annotated_image = None