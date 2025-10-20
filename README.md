# clasificador_gastos

Proyecto: Clasificador Inteligente de Gastos desde Tickets
=========================================================

Este repositorio contiene código para tomar fotos o imágenes de tickets (boletas), extraer texto mediante OCR (Tesseract) y obtener datos estructurados: fecha, comercio y monto. También incluye heurísticas para mejorar el OCR en tickets reales y utilidades para depuración.

Estructura del repositorio
---------------------------

- data/
	- Contiene imágenes de tickets de prueba. Por ejemplo `ticket1.jpg`, `ticket2.png`.

- src/
	- Código fuente principal del proyecto.
	- `__init__.py` : convierte `src` en paquete Python.
	- `procesamiento.py` : funciones para preprocesado de imágenes (vacío/plantilla).
	- `ocr.py` : núcleo del OCR. Contiene `preprocesar_ticket` y `extraer_texto` con múltiples pipelines y heurísticas para extraer `texto`, `monto` y `fecha`.
	- `analisis.py` : funciones para extraer campos específicos (fecha, monto) y procesamiento posterior (vacío/plantilla).
	- `clasificador.py` : reglas o modelo para categorizar gastos (vacío/plantilla).
	- `almacenamiento.py` : guardar registros en CSV o SQLite (vacío/plantilla).

- tools/
	- Utilidades para desarrollar y depurar:
	- `dump_tokens.py` : script para volcar los tokens que Tesseract detecta (texto, confianza y coordenadas). Útil para localizar fechas/montos y ajustar ROIs.
	- `run_ticket1.py` : script de ejemplo que carga `src.ocr.extraer_texto` y ejecuta el pipeline sobre `data/ticket1.jpg`. Útil para tests rápidos.

- app.py
	- Interfaz web (por implementar, sugerencia: Streamlit). Archivo creado en blanco por ahora.

- requirements.txt
	- Dependencias del proyecto (OpenCV, pytesseract, numpy, pandas, streamlit, etc.).

Archivos agregados para desarrollo
---------------------------------
- `tools/dump_tokens.py` (herramienta de depuración)
	- Objetivo: ejecutar `pytesseract.image_to_data` sobre una imagen y volcar los tokens que contienen dígitos, barras, dos puntos, etc., junto con su confianza y coordenadas (left, top, width, height).
	- Uso típico: ayuda a encontrar dónde Tesseract detecta la hora y la fecha en el ticket para después recortar esa zona (ROI) y aplicar OCR más agresivo.

- `tools/run_ticket1.py` (script de prueba)
	- Objetivo: ejemplo de cómo importar `src.ocr.extraer_texto` y ejecutar el pipeline sobre `ticket1.jpg`. Muestra el texto extraído, confianza, monto y fecha.

Recomendaciones de uso (rápido)
------------------------------
1) Instalar dependencias (recomendado usar un venv):

	 python -m venv venv
	 .\.venv\Scripts\Activate.ps1  # PowerShell
	 python -m pip install -r requirements.txt

2) Instalar Tesseract (binario) en Windows
	 - Descargar el instalador desde las releases oficiales o usar Chocolatey:
		 choco install -y tesseract
	 - Asegurarse de que `tesseract.exe` esté en PATH o ajustar `pytesseract.pytesseract.tesseract_cmd` en `src/ocr.py`.

3) Probar el extractor en un ticket de ejemplo:

	 & "venv\Scripts\python.exe" tools\run_ticket1.py

4) Depuración: si la fecha o monto no aparecen, ejecutar:

	 & "venv\Scripts\python.exe" tools\dump_tokens.py

	 Revisar la salida para ver las coordenadas y confidencias de tokens que contienen dígitos o ':' o '/'.

Notas y próximos pasos
---------------------
- Integrar una UI con Streamlit para subir imágenes y mostrar ROI/intervenciones visuales.
- Añadir un paso de post-procesamiento (normalización de montos, corrección ortográfica y limpieza) antes de guardar.
- Crear tests unitarios para garantizar que la extracción funciona con varios formatos de tickets.

Contacto
--------
Si algo no funciona en tu entorno, revisa: la versión de Python, que `tesseract.exe` esté instalado y que el venv sea el que usa VS Code para que Pylance encuentre las dependencias.

INSTALAR!
TESSERACT : https://github.com/UB-Mannheim/tesseract/wiki

LUEGO  Añadir Tesseract al PATH de Windows

Si querés que cualquier programa (Python o cmd) lo reconozca:

Abre PowerShell como Administrador.

Ejecuta:

$old = [Environment]::GetEnvironmentVariable('Path', 'Machine')
$new = $old + ';C:\Program Files\Tesseract-OCR'
[Environment]::SetEnvironmentVariable('Path', $new, 'Machine')


Cierra PowerShell o VS Code y vuelve a abrirlo.

Prueba en PowerShell:

where.exe tesseract
tesseract --version

INSTALAR el idioma español si no está

IR a:

C:\Program Files\Tesseract-OCR\tessdata


Si no existe spa.traineddata, descárgalo de aquí:
Tesseract traineddata files

Busca spa.traineddata y ponlo en tessdata.