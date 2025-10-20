"""tools/run_ticket1.py
-----------------------
Script de ejemplo que ejecuta el pipeline OCR del proyecto sobre `data/ticket1.jpg`.

Objetivo:
 - Servir como prueba rápida para comprobar que `src.ocr.extraer_texto` funciona en el entorno.
 - Mostrar en consola el texto extraído, la confianza media, el monto y la fecha detectada.

Uso:
 - Activar el venv y ejecutar:
     & "venv/Scripts/python.exe" tools/run_ticket1.py

Notas:
 - El script ajusta `sys.path` para poder importar el paquete `src` desde la estructura del repo.
 - Está pensado para uso en desarrollo; la integración final debería ser a través de `app.py` o tests automatizados.
"""

import os
import sys

# añadir la raíz del proyecto al path para importar src.ocr
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from src.ocr import extraer_texto


def main():
    ruta = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ticket1.jpg'))
    print('Procesando:', ruta)
    try:
        resultado = extraer_texto(ruta)
        if isinstance(resultado, dict):
            print('\n--- RESULTADO ---')
            print('Texto extraído:\n', resultado.get('text'))
            print('Mean confidence:', resultado.get('mean_confidence'))
            print('Monto detectado:', resultado.get('monto'))
            print('Fecha detectada:', resultado.get('fecha'))
        else:
            print('Resultado inesperado:', resultado)
    except Exception as e:
        print('Error al procesar:', e)


if __name__ == '__main__':
    main()
