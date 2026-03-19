import os

def api_config(request):
    return {
        'API_BASE_URL': os.environ.get('API_BASE_URL', 'http://127.0.0.1:8001')
    }
