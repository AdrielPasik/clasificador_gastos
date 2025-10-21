"""tools/run_image.py
Generic runner: ejecuta el pipeline OCR sobre una imagen indicada por la variable
IMAGE_PATH en el script o por la variable de entorno TEST_IMAGE.

Edita la constante IMAGE_PATH abajo si preferís cambiarla desde el editor.
También soporta TEST_IMAGE (ruta relativa a data/ o absoluta).
"""
import os
import sys
import json

proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from src.ocr import extraer_texto


def main():
    # Editable default: cambia esto en el script si querés
    IMAGE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'ticket2.jpg')

    # Si se define TEST_IMAGE en el entorno, úsala (relativa a data/ o absoluta)
    test_env = os.environ.get('TEST_IMAGE')
    if test_env:
        if not os.path.isabs(test_env):
            IMAGE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', test_env)
        else:
            IMAGE_PATH = test_env

    ruta = os.path.abspath(IMAGE_PATH)
    print('Procesando:', ruta)
    try:
        resultado = extraer_texto(ruta)
        if isinstance(resultado, dict):
            # Por defecto imprimimos todo excepto raw_data para legibilidad
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
