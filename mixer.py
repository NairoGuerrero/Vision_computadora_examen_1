"""
Módulo para combinar máscaras de detección de grietas y región de referencia.

Proporciona funcionalidad para fusionar resultados de diferentes detectores y visualizar
la combinación resultante.
"""

import matplotlib.pyplot as plt
import cv2
from grietas import CrackDetectionPipeline
from referencia import WhiteRegionDetector


class MaskCombiner:
    """
    Clase para combinar máscaras binarias de diferentes detectores.

    Atributos:
        image_path (str): Ruta de la imagen a procesar
        crack_mask (np.ndarray): Máscara binaria de grietas detectadas
        reference_mask (np.ndarray): Máscara binaria de región de referencia
    """

    def __init__(self, image_path: str):
        """
        Inicializa el combinador con la ruta de la imagen.

        Args:
            image_path: Ruta absoluta o relativa al archivo de imagen
        """
        self.image_path = image_path
        self.crack_mask = None
        self.reference_mask = None

    def _detect_cracks(self) -> None:
        """Ejecuta el detector de grietas y almacena su máscara resultante."""
        crack_detector = CrackDetectionPipeline(self.image_path)
        self.crack_mask = crack_detector.execute()

    def _detect_reference(self) -> None:
        """Ejecuta el detector de referencia y almacena su máscara resultante."""
        reference_extractor = WhiteRegionDetector(self.image_path)
        self.reference_mask = reference_extractor.process()