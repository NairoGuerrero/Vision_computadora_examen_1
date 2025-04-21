# 🧠 Examen de Visión por Computadora

Este repositorio contiene la solución al examen práctico de la asignatura de Visión por Computadora. Se abordan técnicas fundamentales de procesamiento de imágenes para la detección de regiones conectadas, identificación de grietas y localización de marcadores de referencia.

## 📁 Estructura del Proyecto

- `componentes_conexas.py` – Detección de componentes conexos en imágenes binarias.
- `detector_grietas.py` – Identificación de grietas mediante técnicas de umbralización y filtrado.
- `detector_referencia.py` – Localización de marcadores de referencia en imágenes.
- `mixer.py` – Herramientas para la mezcla y combinación de imágenes o canales.
- `main.py` – Script principal que integra y ejecuta los módulos anteriores.
- `imagenes/` – Conjunto de imágenes utilizadas como casos de prueba.
- `requirements.txt` – Lista de dependencias necesarias para ejecutar el proyecto.

## ⚙️ Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/NairoGuerrero/Vision_computadora_examen_1.git
   cd Vision_computadora_examen_1
   ```

2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Uso
Asegúrate de que la carpeta `imagenes/` contenga las imágenes necesarias para cada módulo.

Puedes ejecutar el análisis completo con:

```bash
python main.py
```
Si se quiere ejecutar cada modulo por separado lo puede a hacer con:

```bash
python Nombre_Modulo.py
```


## 👥 Autores

- **Nairo Guerrero Márquez** - [nairo.guerrero@utp.edu.co](mailto:nairo.guerrero@utp.edu.co)
- **Juan David Perdomo Quintero** - [juandavid.perdomo@utp.edu.co](mailto:juandavid.perdomo@utp.edu.co)
- **Andres Felipe Zambrano Torres** - [a.zambrano1@utp.edu.co](mailto:a.zambrano1@utp.edu.co)
- **Fabian Esteban Quintero Arias** - [esteban.quintero@utp.edu.co](mailto:esteban.quintero@utp.edu.co)
- **Santiago Rojas Diez** - [santiago.rojas@utp.edu.co](mailto:santiago.rojas@utp.edu.co)
- **Juan Esteban Osorio Montoya** - [juanesteban.osorio@utp.edu.co](mailto:juanesteban.osorio@utp.edu.co)
