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

    def _display_results(self) -> None:
        """Muestra resultados intermedios usando matplotlib."""
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(self.gray_image, cmap='gray')
        plt.title('Escala de grises')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(self.binary_image, cmap='gray')
        plt.title('Binarización')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(self.mask, cmap='gray')
        plt.title('Región detectada')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    def process(self) -> np.ndarray:
        """
        Ejecuta el pipeline completo de procesamiento.

        Returns:
            Máscara binaria de la región detectada
        """
        self._load_image()
        self._binarize_image()
        self._find_largest_contour()
        self._display_results()
        return self.mask

    def get_bounding_box(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Obtiene las coordenadas del bounding box de la región detectada.

        Returns:
            Tupla (x, y, ancho, alto) o None si no hay detección
        """
        if self.largest_contour is not None:
            return cv2.boundingRect(self.largest_contour)
        return None


def main():
    """Función de demostración del módulo."""
    detector = WhiteRegionDetector('imagenes/img_1.jpg', threshold=150)
    detector.process()

    bbox = detector.get_bounding_box()
    if bbox:
        x, y, w, h = bbox
        print(f"Región de referencia detectada en: x={x}, y={y}, ancho={w}, alto={h}")


if __name__ == '_main_':
    main()