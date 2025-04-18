# comp_conexas.py
"""
Módulo para análisis de componentes conexas en imágenes.

Provee funcionalidades para detección de regiones, cálculo de propiedades geométricas
y generación de visualizaciones con anotaciones.
"""

import cv2
from skimage.measure import label, regionprops
from typing import Tuple, List
from mixer import MaskCombiner


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

    def _load_binary_image(self) -> None:
        """Carga y procesa la imagen binaria usando el módulo mixer."""
        combiner = MaskCombiner(self.image_path)
        self._binary_image = combiner.combine_masks(True)

    def _label_regions(self) -> None:
        """Etiqueta las regiones en la imagen binaria."""
        self._labeled_image = label(self._binary_image)

    def _extract_region_properties(self) -> None:
        """Extrae y filtra propiedades de las regiones detectadas."""
        self.regions = [
            region for region in regionprops(self._labeled_image)
            if region.area >= self.min_area
        ]

    def _annotate_image(self) -> None:
        """Genera imagen anotada con bounding boxes y centroides."""
        self._annotated_image = cv2.imread(self.image_path)

        for idx, region in enumerate(self.regions):
            minr, minc, maxr, maxc = region.bbox
            cy, cx = region.centroid

            # Dibujar bounding box
            cv2.rectangle(
                self._annotated_image,
                (minc, minr), (maxc, maxr),
                (0, 255, 0), 2
            )

            # Dibujar centroide
            cv2.circle(
                self._annotated_image,
                (int(cx), int(cy)), 5,
                (0, 0, 255), -1
            )

            # Añadir etiqueta numérica
            cv2.putText(
                self._annotated_image,
                str(idx + 1),
                (int(cx + 4), int(cy + 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 0, 0), 2
            )

    def analyze(self) -> Tuple[cv2.Mat, List[regionprops]]:
        """
        Ejecuta el pipeline completo de análisis.

        Returns:
            Tuple con:
                - Imagen anotada con las regiones detectadas
                - Lista de propiedades de las regiones
        """
        self._load_binary_image()
        self._label_regions()
        self._extract_region_properties()
        self._annotate_image()
        self._print_region_stats()
        return self._annotated_image, self.regions

    def _print_region_stats(self) -> None:
        """Imprime estadísticas detalladas de cada región."""
        for idx, region in enumerate(self.regions):
            print(f"\n--- Objeto {idx + 1} ---")
            print(f"Área: {region.area}")
            print(f"Perímetro: {region.perimeter:.2f}")
            print(f"Centroide: {region.centroid}")
            print(f"BBox: {region.bbox}")
            print(f"Relación aspecto: {region.bbox_area / region.area:.2f}")
            print(f"Extensión: {region.extent:.2f}")
            print(f"Solidez: {region.solidity:.2f}")
            print(f"Orientación: {region.orientation:.2f} rad")
            print(f"Ejes elipse: Mayor={region.major_axis_length:.2f}, Menor={region.minor_axis_length:.2f}")