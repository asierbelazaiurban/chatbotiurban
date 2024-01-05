
import os
import json
import re
from bs4 import BeautifulSoup  
import requests  



def save_dataset(dataset_entries, chatbot_id):
    dataset_folder = os.path.join('data', 'uploads', 'datasets', chatbot_id)
    os.makedirs(dataset_folder, exist_ok=True)
    dataset_file_path = os.path.join(dataset_folder, 'dataset.json')

    with open(dataset_file_path, 'w') as dataset_file:
        json.dump(dataset_entries, dataset_file, indent=4)


def read_urls(chatbot_folder, chatbot_id):
    urls_file_path = os.path.join(chatbot_folder, f'{chatbot_id}_urls.txt')
    
    try:
        with open(urls_file_path, 'r') as file:
            urls = file.read().splitlines()
        return urls
    except FileNotFoundError:
        app.logger.error(f"Archivo de URLs no encontrado para chatbot_id {chatbot_id}")
        return None
        
def process_single_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = clean_and_format_text(soup)
        return {
            "indice": None,  # El índice se asignará más tarde
            "url": url,
            "dialogue": text
        }
    except Exception as e:
        logging.error(f"Error al procesar la URL {url}: {e}")
        return {"url": url, "error": str(e)}

def clean_and_format_text(soup):

    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()

    # Ignorar imágenes y videos
    for media in soup.find_all(['img', 'video']):
        media.decompose()

    # Extraer el texto del HTML
    text = soup.get_text()
    # Convertir el texto a minúsculas
    text = text.lower()

    # Reemplazar saltos de línea y retornos de carro por espacios
    text = text.replace('\n', ' ').replace('\r', ' ')

    # Eliminar URLs
    text = re.sub(r'http\S+', '', text)

    # Eliminar menciones de usuario (ejemplo: @usuario)
    text = re.sub(r'@\w+', '', text)

    # Eliminar hashtags (ejemplo: #hashtag)
    text = re.sub(r'#\w+', '', text)

    # Opcional: Eliminar o reemplazar caracteres especiales, según sea necesario
    text = re.sub(r'[^\w\s,.]', '', text)

    # Eliminar caracteres unicode extraños y no imprimibles
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # Eliminar espacios múltiples
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


