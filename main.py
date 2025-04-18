"""
Módulo principal para análisis estructural de paredes.

Calcula métricas de salud basado en detección de grietas y referencia de escala,
optimizado para evitar ejecuciones duplicadas.
"""

import cv2
from componentes_conexas import ConnectedComponentsAnalyzer
from skimage.measure import regionprops


class StructuralHealthAnalyzer:
    """
    Analizador de salud estructural con optimización de ejecuciones.

    Atributos:
        image_path (str): Ruta de la imagen a analizar
        reference_width (float): Ancho real de referencia en cm
        reference_height (float): Alto real de referencia en cm
    """

    def __init__(self, image_path: str, reference_width: float = 26.0, reference_height: float = 36.0):
        """
        Inicializa el analizador y componentes necesarios.

        Args:
            image_path: Ruta a la imagen de la pared
            reference_width: Ancho real de referencia en cm (default: 26)
            reference_height: Alto real de referencia en cm (default: 36)
        """
        self.image_path = image_path
        self.reference_width = reference_width
        self.reference_height = reference_height
        self._wall_image = None
        self._components_analyzer = ConnectedComponentsAnalyzer(image_path)
        self._regions = None
        self._reference_component = None
        self._conversion_factor = None
        self._total_wall_area = None
        self._damaged_areas = []

    def _load_wall_image(self) -> None:
        """Carga la imagen y valida su existencia."""
        self._wall_image = cv2.imread(self.image_path)
        if self._wall_image is None:
            raise FileNotFoundError(f"Imagen no encontrada: {self.image_path}")

    def _execute_single_analysis(self) -> None:
        """Ejecuta el análisis de componentes conexas una sola vez."""
        _, self._regions = self._components_analyzer.analyze()
        self._reference_component = max(self._regions, key=lambda r: r.area)

    def _calculate_conversion_factor(self) -> None:
        """Calcula factor de conversión píxeles a cm²."""
        reference_area_cm = self.reference_width * self.reference_height
        self._conversion_factor = reference_area_cm / self._reference_component.area

    def _calculate_total_areas(self) -> None:
        """Calcula áreas totales y dañadas."""
        height, width = self._wall_image.shape[:2]
        pixel_wall_area = height * width
        self._total_wall_area = pixel_wall_area * self._conversion_factor

        # Filtrar regiones excluyendo la referencia
        self._damaged_areas = [
            r.area * self._conversion_factor
            for r in self._regions
            if r != self._reference_component
        ]