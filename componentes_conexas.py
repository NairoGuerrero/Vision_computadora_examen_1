"""
Script para detección y análisis de objetos geométricos en imágenes binarizadas.
Utiliza OpenCV y scikit-image para aplicar análisis de componentes conexas y
extraer múltiples características geométricas como área, perímetro, centroide,
orientación, relación de aspecto, entre otras.

Ideal para fines educativos o proyectos de visión por computador que requieran
clasificación o reconocimiento de formas básicas.

Autor: Juan Esteban Osorio Montoya, Andres Felipe Zambrano Torres, Fabian Esteban Quintero Arias
Fecha: 03/04/2025
"""

import cv2
import numpy as np
from skimage.measure import label, regionprops


class GeometricRecognition:
    """
    Clase para detectar y analizar objetos geométricos en una imagen
    binarizada utilizando análisis de componentes conexas y extraer
    características geométricas de cada objeto detectado.
    """

    def __init__(self, img_path: str):
        """
        Inicializa la clase con la ruta de la imagen y realiza el
        preprocesamiento binario.

        :param img_path: Ruta de la imagen de entrada.
        """
        self.img_path = img_path
        self.binary = self._process_image()
        self.img_color = self.img_color

    def _process_image(self):
        """
        Convierte la imagen a escala de grises, binariza usando un
        umbral automático basado en histograma, y aplica operación
        morfológica de cierre para eliminar huecos.

        :return: Imagen binaria procesada.
        """
        self.img_color = cv2.imread(self.img_path)
        img_gray = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2GRAY)

        frequency, _ = np.histogram(img_gray.ravel(), bins=256, range=(0, 256))
        top_index = np.argsort(frequency)[::-1][:2]
        threshold_value = int((top_index[0] + top_index[1]) / 2)

        _, binary = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY)

        if top_index[0] > 127:
            binary = cv2.bitwise_not(binary)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return closed

    def _connected_regions(self):
        """
        Detecta componentes conexas en la imagen binaria y extrae
        características geométricas como área, perímetro, centroide,
        bbox, orientación, etc. También dibuja cada bbox y centroide
        sobre la imagen a color.
        """
        label_image = label(self.binary)
        features = regionprops(label_image)
        self.figures = []
        for i, region in enumerate(features):
            if region.area >= 2500:
                self.figures.append(region)
                print(f"\n--- Objeto {i} ---")
                print(f"Área: {region.area}")
                print(f"Perímetro: {region.perimeter:.2f}")
                print(f"Centroide: {region.centroid}")
                print(f"BBox: {region.bbox}")
                print(f"Relación de aspecto: {region.bbox_area / region.area:.2f}")
                print(f"Extensión: {region.extent:.2f}")
                print(f"Solidez: {region.solidity:.2f}")
                print(f"Orientación: {region.orientation:.2f} rad")
                print(
                    f"Elipse ajustada - eje mayor: {region.major_axis_length:.2f}, eje menor: {region.minor_axis_length:.2f}")

                # Dibujar bounding box
                minr, minc, maxr, maxc = region.bbox
                cv2.rectangle(self.img_color, (minc, minr), (maxc, maxr), (0, 255, 0), 2)

                # Dibujar centroide
                cy, cx = region.centroid
                cv2.circle(self.img_color, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                cv2.putText(self.img_color, str(i), (int(cx + 4), int(cy + 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    def get_figures(self):
        """
        Devuelve la lista de regiones (figuras) detectadas para análisis externo.

        :return: Lista de objetos regionprops con propiedades geométricas.
        """
        return getattr(self, 'figures', [])

    def show_image(self):
        """
        Ejecuta el análisis de regiones y muestra la imagen resultante
        con los objetos etiquetados visualmente.
        """
        self._connected_regions()
        cv2.imshow('Figuras detectadas', self.img_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# EJemplo de uso
if __name__ == '__main__':
    # Instancia de la clase
    image = GeometricRecognition('caracteristicas.png') # Ruta a la imagen

    # Recnocimiento de las figuras y visualizacion
    image.show_image()

    # Extraccion de lista de las figuras para nuevos algoritmos
    figures = image.get_figures()
