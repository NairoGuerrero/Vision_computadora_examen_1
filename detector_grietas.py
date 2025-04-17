"""
Módulo para detección de grietas en superficies usando procesamiento de imágenes.

Combina técnicas de preprocesamiento, detección de bordes y extracción de características
para identificar posibles áreas dañadas.
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from typing import Tuple


class CrackDetectionPipeline:
    """
    Pipeline completo para detección y análisis de grietas.

    Atributos:
        image_path (str): Ruta de la imagen a analizar
        orb_features (int): Número de características ORB a detectar
    """

    def __init__(self, image_path: str, orb_features: int = 1500):
        """
        Inicializa el pipeline con parámetros de configuración.

        Args:
            image_path: Ruta a la imagen de entrada
            orb_features: Cantidad máxima de características ORB (default: 1500)
        """
        self.image_path = image_path
        self.orb_features = orb_features
        self.original_image = None
        self.gray_image = None
        self.edges = None
        self.keypoints = None
        self.descriptors = None
        self.features_image = None

    def _load_image(self) -> None:
        """Carga y convierte la imagen a escala de grises."""
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise FileNotFoundError(f"Imagen no encontrada: {self.image_path}")
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

    def _preprocess_image(self) -> None:
        """Aplica secuencia de preprocesamiento para realzar características."""
        blurred = cv2.blur(self.gray_image, (3, 3))
        log_transformed = self._apply_log_transform(blurred)
        self.edges = self._enhance_edges(log_transformed)

    def _apply_log_transform(self, image: np.ndarray) -> np.ndarray:
        """Aplica transformación logarítmica para mejorar el contraste."""
        normalized = (np.log(image + 1) / np.log(1 + np.max(image))) * 255
        return normalized.astype(np.uint8)

    def _enhance_edges(self, image: np.ndarray) -> np.ndarray:
        """Realza bordes usando filtrado bilateral y detección Canny."""
        bilateral = cv2.bilateralFilter(image, 4, 90, 90)
        return cv2.Canny(bilateral, 82, 172)

    def _apply_morphology(self) -> None:
        """Mejora la conectividad de bordes con operaciones morfológicas."""
        kernel = np.ones((8, 8), np.uint8)
        self.edges = cv2.dilate(self.edges, kernel, iterations=2)
        self.edges = cv2.morphologyEx(self.edges, cv2.MORPH_CLOSE, kernel)

    def _detect_orb_features(self) -> None:
        """Detecta características ORB en los bordes procesados."""
        orb = cv2.ORB_create(nfeatures=self.orb_features)
        self.keypoints, self.descriptors = orb.detectAndCompute(self.edges, None)
        self.features_image = cv2.drawKeypoints(
            self.edges, self.keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )