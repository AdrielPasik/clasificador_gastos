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
    # Editable path: set IMAGE_PATH below to test a specific image without using shell args
    IMAGE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'ticket1.jpg')
    ruta = os.path.abspath(IMAGE_PATH)
    print('Procesando:', ruta)
    import json
    try:
        resultado = extraer_texto(ruta)
        if isinstance(resultado, dict):
            # No imprimimos raw_data completo para mantener salida legible
            salida = dict(resultado)
            if 'raw_data' in salida:
                salida.pop('raw_data')
            print('\n--- RESULTADO (JSON) ---')
            print(json.dumps(salida, indent=2, ensure_ascii=False))
        else:
            print('Resultado inesperado:', resultado)
    except Exception as e:
        print('Error al procesar:', e)


if __name__ == '__main__':
    main()
