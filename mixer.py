"""
Módulo para combinar máscaras de detección de grietas y región de referencia.

Proporciona funcionalidad para fusionar resultados de diferentes detectores y visualizar
la combinación resultante.
"""

import matplotlib.pyplot as plt
import cv2
from detector_grietas import CrackDetectionPipeline
from detector_referencia import WhiteRegionDetector


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

    def combine_masks(self, show_combined: bool = False) -> cv2.Mat:
        """
        Combina las máscaras mediante operación OR y opcionalmente muestra el resultado.

        Args:
            show_combined: Bandera para mostrar visualización de la máscara combinada

        Returns:
            Matriz OpenCV con la máscara combinada
        """
        self._detect_cracks()
        self._detect_reference()

        combined = cv2.bitwise_or(self.crack_mask, self.reference_mask)

        if show_combined:
            self._display_combined_mask(combined)

        return combined

    @staticmethod
    def _display_combined_mask(mask: cv2.Mat) -> None:
        """
        Muestra la máscara combinada usando matplotlib.

        Args:
            mask: Máscara binaria a visualizar
        """
        plt.figure(figsize=(10, 5))
        plt.imshow(mask, cmap='gray')
        plt.title('Regiones combinadas: grietas + hoja')
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def main():
    """Función principal para demostración del módulo."""
    combiner = MaskCombiner('imagenes/img_1.jpg')
    _ = combiner.combine_masks(show_combined=True)


if __name__ == '_main_':
    main()