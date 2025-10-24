import requests
url = 'http://127.0.0.1:8000/api/ocr?debug_tokens=true'
files = {'file': ('ticket2.jpg', open(r'c:/Users/Personal/Documents/Universidad/Inteligencia Artificial/clasificador_gastos/data/ticket2.jpg','rb'), 'image/jpeg')}
try:
    r = requests.post(url, files=files, timeout=60)
    print('STATUS', r.status_code)
    print(r.headers.get('content-type'))
    print(r.text)
except Exception as e:
    print('ERROR', e)
