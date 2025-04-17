"""
Módulo para detección de región de referencia blanca en imágenes.

Implementa un pipeline completo para identificación y análisis de la región blanca
de mayor área en una imagen, útil para calibración de escalas.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple


class WhiteRegionDetector:
    """
    Detector de región blanca de referencia.

    Atributos:
        image_path (str): Ruta de la imagen a procesar
        threshold (int): Umbral de binarización (0-255)
        original_image (np.ndarray): Imagen original cargada
        binary_image (np.ndarray): Imagen binarizada
        largest_contour (np.ndarray): Contorno de la región detectada
        mask (np.ndarray): Máscara binaria de la región
    """

    def __init__(self, image_path: str, threshold: int = 150):
        """
        Inicializa el detector con parámetros de configuración.

        Args:
            image_path: Ruta a la imagen de entrada
            threshold: Valor de umbral para binarización (default: 150)
        """
        self.image_path = image_path
        self.threshold = threshold
        self.original_image = None
        self.gray_image = None
        self.binary_image = None
        self.largest_contour = None
        self.mask = None

    def _load_image(self) -> None:
        """Carga y convierte la imagen a escala de grises."""
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise FileNotFoundError(f"Imagen no encontrada: {self.image_path}")
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

    def _binarize_image(self) -> None:
        """Aplica umbralización para obtener imagen binaria."""
        _, self.binary_image = cv2.threshold(
            self.gray_image, self.threshold, 255, cv2.THRESH_BINARY
        )

    def _find_largest_contour(self) -> None:
        """Identifica el contorno de mayor área en la imagen binaria."""
        contours, _ = cv2.findContours(
            self.binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            raise ValueError("No se encontraron contornos en la imagen")

        self.largest_contour = max(contours, key=cv2.contourArea)
        self._create_mask()

    def _create_mask(self) -> None:
        """Genera máscara binaria del área detectada."""
        self.mask = np.zeros_like(self.binary_image)
        cv2.drawContours(
            self.mask, [self.largest_contour], -1, 255, thickness=cv2.FILLED
        )