##!/usr/bin/env python
# coding: utf-8


import faiss
import chardet  # Added for encoding detection  # Ensure faiss library is installed
import numpy as np
from flask import Flask, request, jsonify
import openai
import requests
from bs4 import BeautifulSoup
import chardet  # Added for encoding detection
import os
import shutil
import json
import time
from urllib.parse import urlparse, urljoin
import random
from time import sleep

import gensim.downloader as api
from gensim.models import Word2Vec


app = Flask(__name__)


######## Creación de bbddd FAISS para cada cliente ########

# Global variable to store the FAISS index
faiss_index = None

def initialize_faiss_index(dimension=128):
    global faiss_index
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss.write_index(faiss_index, 'data/faiss_index/faiss.idx')

def get_faiss_index():
    global faiss_index
    if faiss_index is None:
        raise ValueError("FAISS index has not been initialized.")
    return faiss_index

def create_database(chatbot_id, dimension=128):
    directory = os.path.join('data/faiss_index', chatbot_id)
    file_name = 'faiss.idx'
    file_path = os.path.join(directory, file_name)
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    if not os.path.exists(file_path):
        initialize_faiss_index(dimension)

    return file_path

@app.route('/create_new_bbdd', methods=['POST'])
def create_bbdd():
    data = request.json
    chatbot_id = data.get('chatbot_id')
    
    if not chatbot_id:
        return jsonify({"error": "No chatbot_id provided"}), 400

    index_file_path = create_database(chatbot_id)
    message = f"FAISS Index created or verified at: {index_file_path}"
    
    return jsonify({"message": message}), 200

# Example of how to initialize the index (adjust dimension as needed)
initialize_faiss_index(128)  # Assuming your embeddings are 128-dimensional


######## ########



def generate_embedding(text):
    """
    Genera un embedding para un texto dado utilizando OpenAI.
    """
    openai_api_key = os.environ.get('OPENAI_API_KEY')  # Establece la clave API de OpenAI aquí

    try:
        openai.api_key = openai_api_key
        response = openai.Embedding.create(
            input=[text],  # Ajuste para llamar a la función de embeddings de OpenAI
            engine="text-embedding-ada-002",  # Especifica el motor a utilizar
            max_tokens=1  
        )
    except Exception as e:
        raise ValueError(f"No se pudo obtener el embedding: {e}")

    # Ajuste para extraer el embedding según la nueva estructura de respuesta de la API
    
    # Asegurarse de que la respuesta de la API contiene 'data' y tiene al menos un elemento
    if 'data' in response and len(response['data']) > 0:
        embedding = np.array(response['data'][0]['embedding'])
    else:
        raise ValueError("Respuesta de la API de OpenAI no válida o sin datos de embedding.")

    
    # Asegurar que el embedding es un numpy array de tipo float32
    if embedding is not None:
        embedding = embedding.astype(np.float32)
    else:
        raise ValueError("No se pudo obtener el embedding del texto proporcionado.")
    
    return embedding



def generate_embedding_withou_openAI(text):
    """
    Genera un embedding para un texto dado utilizando un modelo Word2Vec de Gensim.
    """
    # Cargar un modelo Word2Vec preentrenado. Puedes elegir otro modelo si lo prefieres.
    model = api.load("word2vec-google-news-300")  # Este es un modelo grande y puede tardar en descargarse.

    # Preprocesar el texto: dividir en palabras y eliminar palabras que no están en el modelo.
    words = [word for word in text.split() if word in model.key_to_index]

    # Generar embeddings para cada palabra y promediarlos.
    if len(words) >= 1:
        embedding = np.mean(model[words], axis=0)
    else:
        raise ValueError("El texto proporcionado no contiene palabras reconocidas por el modelo.")

    return embedding

# Ejemplo de uso



# Importar las bibliotecas necesarias
# Suponiendo que los datos son embeddings de dimensión 768 (cambiar según sea necesario)
dim = 768
num_data_points = 10000  # Número de puntos de datos (cambiar según sea necesario)
# Crear datos de ejemplo (reemplazar con tus propios datos)
data = np.random.rand(num_data_points, dim).astype(np.float32)
# Crear y entrenar el índice Faiss para la búsqueda de vecinos más cercanos
index = faiss.IndexFlatL2(dim)  # Usar L2 para la distancia
# Milvus adds data to the collection in a different way  # Agregar los datos al índice
# Realizar una consulta de ejemplo
query = np.random.rand(dim).astype(np.float32)
k = 5  # Número de vecinos más cercanos a buscar
distances, neighbors = index.search(query.reshape(1, -1), k)
# Mostrar los resultados
print("Índices de los vecinos más cercanos:", neighbors)
print("Distancias de los vecinos más cercanos:", distances)



# Configuraciones
UPLOAD_FOLDER = 'data/uploads/'  # Ajusta esta ruta según sea necesario
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'csv', 'docx', 'xlsx', 'pptx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Esta ruta maneja la subida de archivos y almacena los embeddings en Milvus



def allowed_file(filename, chatbot_id):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

def dividir_en_segmentos(texto, max_tokens):
    # Tokenizar el texto usando NLTK
    tokens = word_tokenize(texto)

    segmentos = []
    segmento_actual = []

    for token in tokens:
        # Cambiar la condición para que divida los segmentos antes de alcanzar el límite exacto de max_tokens
        if len(segmento_actual) + len(token.split()) > max_tokens:
            segmentos.append(' '.join(segmento_actual))
            segmento_actual = [token]
        else:
            segmento_actual.append(token)

    # Añadir el último segmento si hay tokens restantes
    if segmento_actual:
        segmentos.append(' '.join(segmento_actual))

    return segmentos


    """
    Divide un texto en segmentos que no excedan el límite de tokens.
    Esta es una aproximación simple y debe ajustarse para usar un tokenizador específico.
    """
    palabras = texto.split()
    segmentos = []
    segmento_actual = []

    tokens_contados = 0
    for palabra in palabras:
        # Asumimos un promedio de 4 tokens por palabra como aproximación
        if tokens_contados + len(palabra.split()) * 4 > max_tokens:
            segmentos.append(' '.join(segmento_actual))
            segmento_actual = [palabra]
            tokens_contados = len(palabra.split()) * 4
        else:
            segmento_actual.append(palabra)
            tokens_contados += len(palabra.split()) * 4

    # Añadir el último segmento si hay alguno
    if segmento_actual:
        segmentos.append(' '.join(segmento_actual))

    return segmentos

#metodo param la subida de documentos

UPLOAD_FOLDER = 'uploads'  # Asegúrate de definir esta variable correctamente
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/uploads', methods=['POST'])
def upload_file():
    try:
        if 'documento' not in request.files:
            return jsonify({"respuesta": "No se encontró el archivo 'documento'", "codigo_error": 1})
        
        file = request.files['documento']
        chatbot_id = request.form.get('chatbot_id')

        if file.filename == '':
            return jsonify({"respuesta": "No se seleccionó ningún archivo", "codigo_error": 1})

        # Crear la carpeta del chatbot si no existe
        chatbot_folder = os.path.join(UPLOAD_FOLDER, str(chatbot_id))
        os.makedirs(chatbot_folder, exist_ok=True)

        # Determinar la extensión del archivo y crear una subcarpeta
        file_extension = os.path.splitext(file.filename)[1][1:]  # Obtiene la extensión sin el punto
        extension_folder = os.path.join(chatbot_folder, file_extension)
        os.makedirs(extension_folder, exist_ok=True)

        # Guardar el archivo en la subcarpeta correspondiente
        file_path = os.path.join(extension_folder, file.filename)
        file.save(file_path)

        # Procesamiento del archivo
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                encoding = chardet.detect(raw_data)['encoding'] or 'utf-8'
                if not encoding or chardet.detect(raw_data)['confidence'] < 0.7:
                    encoding = 'utf-8'
            contenido = raw_data.decode(encoding, errors='replace')

            # Dividir el contenido en segmentos si supera el límite de tokens
            MAX_TOKENS_PER_SEGMENT = 7000  # Establecer un límite seguro de tokens por segmento
            segmentos = dividir_en_segmentos(contenido, MAX_TOKENS_PER_SEGMENT)

            # Procesar cada segmento y almacenar los embeddings
            vector_embeddings = []
            for segmento in segmentos:
                try:
                    embedding = obtener_embeddings(segmento)
                    vector_embeddings.append(embedding)
                except Exception as e:
                    return jsonify({"respuesta": f"No se pudo procesar el segmento. Error: {e}", "codigo_error": 1})

            # Agregar todos los embeddings al índice de FAISS
            global faiss_index
            for embedding in vector_embeddings:
                faiss_index.add(np.array([embedding], dtype=np.float32))

            indexado_en_faiss = True
        except Exception as e:
            indexado_en_faiss = False
            return jsonify({"respuesta": f"No se pudo indexar en FAISS. Error: {e}", "codigo_error": 1})

        # Si todo salió bien, devolver una respuesta positiva
        return jsonify({
            "respuesta": "Archivo procesado e indexado con éxito.",
            "indexado_en_faiss": indexado_en_faiss,
            "codigo_error": 0
        })
    except Exception as e:
        return jsonify({"respuesta": f"Error durante el procesamiento. Error: {e}", "codigo_error": 1})






@app.route('/process_urls', methods=['POST'])
def process_urls():
    data = request.json
    chatbot_id = data.get('chatbot_id')
    if not chatbot_id:
        return jsonify({"status": "error", "index": "No chatbot_id provided"}), 400

    chatbot_folder = os.path.join('data/uploads/scraping', f'{chatbot_id}')
    if not os.path.exists(chatbot_folder):
        os.makedirs(chatbot_folder)

    try:
        with open(os.path.join(chatbot_folder, f'{chatbot_id}.txt'), 'r') as file:
            urls = file.readlines()
    except FileNotFoundError:
        return jsonify({"status": "error", "index": "URLs file not found for the provided chatbot_id"}), 404

    all_indexed = True
    error_message = ""

    FAISS_INDEX_DIMENSION = 128

    for url in urls:
        url = url.strip()
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text()

            segmentos = dividir_en_segmentos(text, MAX_TOKENS_PER_SEGMENT)

            for segmento in segmentos:
                embeddings = generate_embedding_withou_openAI(segmento)
                if embeddings.shape[1] != FAISS_INDEX_DIMENSION:
                    raise ValueError(f"Dimensión de embeddings incorrecta: esperada {FAISS_INDEX_DIMENSION}, obtenida {embeddings.shape[1]}")
                faiss_index.add(np.array([embeddings], dtype=np.float32))
        except Exception as e:
            all_indexed = False
            error_message = str(e)
            break

        sleep(0.2)  # Pausa de 0.2 segundos entre cada petición

    if all_indexed:
        return jsonify({"status": "success", "index": "Todo indexado en FAISS correctamente"})
    else:
        return jsonify({"status": "error", "index": f"Error al indexar: {error_message}"})



@app.route('/fine-tuning', methods=['POST'])
def fine_tuning():
    # Obtener los datos del cuerpo de la solicitud
    data = request.json
    training_text = data.get('training_text')
    chat_id = data.get('chat_id')

    # Validación de los datos recibidos
    if not training_text or not isinstance(chat_id, int):
        return jsonify({"status": "error", "message": "Invalid input"}), 400

    # Aquí puedes hacer algo con training_text y chat_id si es necesario

    # Datos para el proceso de fine-tuning
    training_data = {
        # Suponiendo que estos son los datos que OpenAI necesita para el fine-tuning
        "text": training_text,
        "chat_id": chat_id
    }

    # Endpoint y headers para la API de OpenAI
    
    openai_endpoint = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": "Bearer " + openai.api_key
    }

    # Realizar la solicitud POST a la API de OpenAI
    response = requests.post(openai_endpoint, json=training_data, headers=headers)

    # Manejar la respuesta
    if response.status_code == 200:
        return jsonify({"status": "fine-tuning started", "response": response.json()})
    else:
        return jsonify({"status": "error", "message": response.text}), response.status_code


@app.route('/save_urls', methods=['POST'])
def save_urls():
    data = request.json
    urls = data.get('urls', [])  # Asumimos que 'urls' es una lista de URLs
    chatbot_id = data.get('chatbot_id')

    if not urls or not chatbot_id:
        return jsonify({"error": "Missing 'urls' or 'chatbot_id'"}), 400

    # Crear o verificar la carpeta específica del chatbot_id
    chatbot_folder = os.path.join('data/uploads/scraping', f'{chatbot_id}')
    os.makedirs(chatbot_folder, exist_ok=True)

    # El nombre del archivo será el 'chatbot_id' con la extensión .txt
    file_path = os.path.join(chatbot_folder, f'{chatbot_id}.txt')

    # Si el archivo ya existe, se borra
    if os.path.exists(file_path):
        os.remove(file_path)

    # Crear el archivo con el nombre del chatbot_id y escribir las URLs
    with open(file_path, 'w') as file:
        for url in urls:
            file.write(url + '\n')

    return jsonify({"status": "success", "message": "URLs saved successfully"})



def safe_request(url, max_retries=3):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                return response
        except RequestException as e:
            print(f"Attempt {attempt + 1} failed for {url}: {e}")
    return None

@app.route('/url_for_scraping', methods=['POST'])
def url_for_scraping():
    try:
        # Obtener URL y chatbot_id del request
        data = request.get_json()
        base_url = data.get('url')
        chatbot_id = data.get('chatbot_id')

        if not base_url:
            return jsonify({'error': 'No URL provided'}), 400

        # Crear o verificar la carpeta específica del chatbot_id
        save_dir = os.path.join('data/uploads/scraping', f'{chatbot_id}')
        os.makedirs(save_dir, exist_ok=True)

        # Ruta del archivo a crear o sobrescribir
        file_path = os.path.join(save_dir, f'{chatbot_id}.txt')

        # Borrar el archivo existente si existe
        if os.path.exists(file_path):
            os.remove(file_path)

        # Función para determinar si la URL pertenece al mismo dominio
        def same_domain(url):
            return urlparse(url).netloc == urlparse(base_url).netloc

        # Hacer scraping y recoger URLs únicas
        urls = set()
        base_response = safe_request(base_url)
        if base_response:
            soup = BeautifulSoup(base_response.content, 'html.parser')
            for tag in soup.find_all('a'):
                url = urljoin(base_url, tag.get('href'))
                if same_domain(url):
                    urls.add(url)
        else:
            return jsonify({'error': 'Failed to fetch base URL'}), 500

        # Contar palabras en cada URL y preparar los datos para el JSON de salida
        urls_data = []
        for url in urls:
            response = safe_request(url)
            if response:
                soup = BeautifulSoup(response.content, 'html.parser')
                text = soup.get_text()
                word_count = len(text.split())
                urls_data.append({'url': url, 'word_count': word_count})
            else:
                urls_data.append({'url': url, 'message': 'Failed HTTP request after retries'})

        # Guardar solo las URLs en un archivo de texto con el nombre chatbot_id
        with open(file_path, 'w') as text_file:
            for url_data in urls_data:
                text_file.write(url_data['url'] + '\n')

        # Devolver al front las URLs y el conteo de palabras asociado a cada una
        return jsonify(urls_data)
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500



@app.route('/url_for_scraping_only_a_few', methods=['POST'])
def url_for_scraping_only_a_few():
    try:
        # Obtener URL, chatbot_id y número máximo de URLs del request
        data = request.get_json()
        base_url = data.get('url')
        chatbot_id = data.get('chatbot_id')
        max_urls = data.get('max_urls')  # No hay valor por defecto

        if not base_url or not max_urls:
            return jsonify({'error': 'No URL or max_urls provided'}), 400

        # Crear o verificar la carpeta específica del chatbot_id
        save_dir = os.path.join('data/uploads/scraping', f'{chatbot_id}')
        os.makedirs(save_dir, exist_ok=True)

        # Ruta del archivo a crear o sobrescribir
        file_path = os.path.join(save_dir, f'{chatbot_id}.txt')

        # Borrar el archivo existente si existe
        if os.path.exists(file_path):
            os.remove(file_path)

        # Función para determinar si la URL pertenece al mismo dominio
        def same_domain(url):
            return urlparse(url).netloc == urlparse(base_url).netloc

        # Hacer scraping y recoger hasta max_urls URLs únicas
        urls = set()
        base_response = safe_request(base_url)
        if base_response:
            soup = BeautifulSoup(base_response.content, 'html.parser')
            for tag in soup.find_all('a', href=True):
                if len(urls) >= max_urls:  # Limitar a max_urls URLs
                    break
                url = urljoin(base_url, tag.get('href'))
                if same_domain(url) and url not in urls:
                    urls.add(url)
        else:
            return jsonify({'error': 'Failed to fetch base URL'}), 500

        # Contar palabras en las URLs y preparar los datos para el JSON de salida
        urls_data = []
        for url in list(urls)[:max_urls]:  # Procesar solo las URLs especificadas
            response = safe_request(url)
            if response:
                soup = BeautifulSoup(response.content, 'html.parser')
                text = soup.get_text()
                word_count = len(text.split())
                urls_data.append({'url': url, 'word_count': word_count})
            else:
                urls_data.append({'url': url, 'message': 'Failed HTTP request after retries'})

        # Guardar las URLs en un archivo de texto con el nombre chatbot_id
        with open(file_path, 'w') as text_file:
            for url_data in urls_data:
                text_file.write(url_data['url'] + '\n')

        # Devolver las URLs y el conteo de palabras asociado a cada una
        return jsonify(urls_data)
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500


#Recibimos las urls no validas de front, de cicerone

@app.route('/delete_urls', methods=['POST'])
def delete_urls():
    data = request.json
    urls_to_delete = set(data.get('urls', []))  # Conjunto de URLs a eliminar
    chatbot_id = data.get('chatbot_id')  # Identificador del chatbot

    # Verifica si faltan datos necesarios
    if not urls_to_delete or not chatbot_id:
        return jsonify({"error": "Missing 'urls' or 'chatbot_id'"}), 400

    # Construir la ruta del archivo basado en chatbot_id
    chatbot_folder = os.path.join('data/uploads/scraping', str(chatbot_id))

    # Para depuración: imprime la ruta del directorio del chatbot
    print("Ruta del directorio del chatbot:", chatbot_folder)

    if not os.path.exists(chatbot_folder):
        return jsonify({"status": "error", "message": "Chatbot folder not found"}), 404

    file_name = f"{chatbot_id}.txt"
    file_path = os.path.join(chatbot_folder, file_name)

    # Para depuración: imprime la ruta del archivo de URLs
    print("Ruta del archivo de URLs:", file_path)

    if not os.path.exists(file_path):
        return jsonify({"status": "error", "message": f"File {file_name} not found in chatbot folder"}), 404

    try:
        # Leer y actualizar el archivo
        with open(file_path, 'r+') as file:
            existing_urls = set(file.read().splitlines())
            updated_urls = existing_urls - urls_to_delete

            # Volver al inicio del archivo y limpiarlo
            file.seek(0)
            file.truncate()

            # Guardar las URLs actualizadas, evitando líneas vacías
            for url in updated_urls:
                if url.strip():  # Asegura que la URL no sea una línea vacía
                    file.write(url + '\n')

        return jsonify({"status": "success", "message": "URLs deleted successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# Supongamos que ya tenemos un índice FAISS y funciones para generar embeddings y procesar resultados
# index = ... (índice FAISS ya creado)
# generate_embedding = ... (función para generar embeddings a partir de texto)
# process_results = ... (función para procesar los índices de FAISS y obtener información relevante)
@app.route('/ask', methods=['POST'])
def ask():
    try:
        # Recibir la consulta de texto
        query_text = request.json.get('query')

        # Generar un embedding para la consulta usando OpenAI
        query_embedding = generate_embedding(query_text)

        # Realizar la búsqueda en FAISS
        k = 5  # Número de resultados a devolver
        distances, indices = index.search(np.array([query_embedding]).astype(np.float32), k)

        # Procesar los índices para obtener la información correspondiente
        info = process_results(indices)

        # Utilizar OpenAI para generar una respuesta comprensible en español
        response = openai.Completion.create(
            model="text-embedding-ada-002",  # Especifica el modelo de OpenAI a utilizar
            prompt=info,
            max_tokens=150,  # Define el número máximo de tokens en la respuesta
            temperature=0.7,  # Ajusta la creatividad de la respuesta
            top_p=1,  # Controla la diversidad de la respuesta
            frequency_penalty=0,  # Penalización por frecuencia de uso de palabras
            presence_penalty=0,  # Penalización por presencia de palabras
            language="es"  # Especifica el idioma de la respuesta
        )

        # Extraer el texto de la respuesta
        response_text = response.choices[0].text.strip()

        # Devolver la respuesta
        return jsonify({"response": response_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def obtener_embeddings(texto):
    # Llamada a la API de OpenAI para obtener embeddings
    response = openai.Embedding.create(input=texto, engine="text-embedding-ada-002")
    # La respuesta incluye los embeddings, que puedes transformar en un array de numpy
    embedding = np.array(response['data'][0]['embedding'])
    return embedding



def update_faiss_index(embeddings, chatbot_id):
    # Esta función debe actualizar el índice de FAISS con nuevos embeddings
    index = faiss.IndexFlatL2(512)  # Suponiendo que usas un índice FlatL2
    index.add(np.array(embeddings).astype(np.float32))
    return index


# Código para diagnosticar problemas con FAISS


# Verificar la versión de FAISS
print('Versión de FAISS:', faiss.__version__)

# Listar los atributos y métodos de FAISS
print('Atributos y métodos en FAISS:', dir(faiss))

# Si necesitas reinstalar FAISS, descomenta las siguientes líneas:
# !pip uninstall -y faiss faiss-gpu faiss-cpu
# !pip install faiss-cpu  # Para CPU
# !pip install faiss-gpu  # Para GPU


# Suponiendo que tienes un diccionario para mapear IDs de documentos a índices en FAISS
doc_id_to_faiss_index = {}

def add_document_to_faiss(document, doc_id):
    # Verifica si el documento ya está indexado
    if doc_id in doc_id_to_faiss_index:
        print(f"Documento con ID {doc_id} ya indexado.")
        return

    # Genera el embedding para el nuevo documento
    embedding = generate_embedding(document)

    # Añade el embedding al índice de FAISS y actualiza el mapeo
    faiss_index = get_faiss_index()  # Obtén tu índice de FAISS actual
    faiss_index.add(np.array([embedding]))  # Añade el nuevo embedding
    new_index = faiss_index.ntotal - 1  # El nuevo índice en FAISS
    doc_id_to_faiss_index[doc_id] = new_index  # Actualiza el mapeo

    # No olvides guardar el índice de FAISS y el mapeo actualizado de forma persistente


@app.route('/filter_urls', methods=['POST'])
def filter_urls():
    data = request.json
    provided_urls = set(data.get('urls', []))
    chatbot = data.get('chatbot')

    if not chatbot:
        return jsonify({"status": "error", "message": "No chatbot provided"}), 400

    file_path = os.path.join('data/uploads/chatbot', 'urls.txt')
    if not os.path.exists(file_path):
        return jsonify({"status": "error", "message": "File not found"}), 404

    try:
        with open(file_path, 'r') as file:
            existing_urls = set(file.read().splitlines())

        urls_to_keep = existing_urls.intersection(provided_urls)

        with open(file_path, 'w') as file:
            for url in urls_to_keep:
                file.write(url + '\n')  # Corregido aquí

        return jsonify({
            "status": "success",
            "message": "URLs have been filtered",
            "kept_urls": list(urls_to_keep)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500



@app.route('/ask_prueba', methods=['POST'])
def ask_prueba():
    data = request.json

    # Comprobaciones de seguridad para asegurarse de que todos los campos están presentes
    if not data or 'pregunta' not in data or 'chatbot_id' not in data or 'token' not in data:
        return jsonify({'error': 'Faltan campos requeridos'}), 400

    # Extraer los datos de la solicitud
    pregunta = data['pregunta']
    chatbot_id = data['chatbot_id']
    token = data['token']

    # Respuestas predefinidas sobre el turismo en Málaga
    respuestas = [
        "Málaga es famosa por sus playas y su clima cálido.",
        "El Museo Picasso es uno de los lugares más visitados en Málaga.",
        "La Alcazaba, una fortaleza árabe, es una de las joyas históricas de la ciudad.",
        "Málaga es conocida por ser la ciudad natal de Pablo Picasso.",
        "El Caminito del Rey ofrece impresionantes vistas y senderos emocionantes.",
        "El Parque Natural Montes de Málaga es ideal para amantes de la naturaleza.",
        "Málaga celebra su feria anual en agosto con música, baile y comida tradicional.",
        "El Centro Pompidou Málaga alberga arte moderno y contemporáneo.",
        "La Catedral de Málaga es un impresionante ejemplo de arquitectura renacentista.",
        "El Mercado Central de Atarazanas es famoso por sus productos frescos y locales.",
        "El barrio de La Malagueta es conocido por su bulliciosa playa urbana.",
        "El Castillo de Gibralfaro ofrece vistas panorámicas de la ciudad.",
        "La gastronomía malagueña incluye espetos de sardinas y pescaíto frito.",
        "Málaga es una ciudad vibrante con una animada vida nocturna.",
        "El Jardín Botánico-Histórico La Concepción es un oasis tropical en Málaga.",
        "El Soho de Málaga es famoso por su arte callejero y ambiente bohemio.",
        "El Muelle Uno es un moderno espacio de ocio y comercio junto al mar.",
        "El Teatro Romano es un vestigio del pasado romano de Málaga.",
        "La Fiesta de los Verdiales celebra una antigua tradición musical local.",
        "El Museo Carmen Thyssen Málaga exhibe arte español y andaluz.",
        "Málaga es punto de partida para explorar la Costa del Sol.",
        "El barrio del Perchel conserva la esencia tradicional de Málaga.",
        "El vino dulce de Málaga es conocido internacionalmente.",
        "El Museo Automovilístico y de la Moda combina coches clásicos con alta costura.",
        "Málaga tiene una rica tradición en la producción de aceite de oliva.",
        "La Semana Santa en Málaga es famosa por sus procesiones y pasos.",
        "Los baños árabes Hammam Al Ándalus ofrecen una experiencia relajante.",
        "El CAC Málaga es un centro de arte contemporáneo de referencia.",
        "El Paseo del Parque es un agradable paseo lleno de vegetación tropical.",
        "La Casa Natal de Picasso alberga obras tempranas del artista.",
        "El Mercado de la Merced es un lugar popular para comer y socializar.",
        "Málaga cuenta con hermosas playas como Guadalmar y El Palo.",
        "La Térmica es un centro cultural y de creación contemporánea.",
        "El FESTIVAL DE MÁLAGA es importante en el panorama cinematográfico español.",
        "La Noria de Málaga ofrece vistas espectaculares desde sus cabinas.",
        "Málaga es conocida por sus festivales de música y cultura.",
        "El MUPAM es el Museo del Patrimonio Municipal de Málaga.",
        "El Museo Revello de Toro alberga pinturas de Félix Revello de Toro.",
        "El Barrio de Pedregalejo es famoso por sus chiringuitos y ambiente relajado."
    ]

    respuesta = random.choice(respuestas)  # Seleccionar una respuesta aleatoria
    return jsonify({'pregunta': pregunta, 'respuesta': respuesta})



def add_document_to_faiss(text, url):
    # Supongamos que 'faiss_index' es tu índice FAISS y 'doc_id_to_faiss_index' es un diccionario que mapea URLs a índices FAISS
    global faiss_index, doc_id_to_faiss_index

    # Genera el embedding para el nuevo documento
    embedding = generate_embedding(text)

    # Verifica si el documento ya está indexado
    if url in doc_id_to_faiss_index:
        print(f"Documento con URL {url} ya indexado.")
        return

    # Añade el embedding al índice de FAISS y actualiza el mapeo
    faiss_index.add(np.array([embedding]))  # Añade el nuevo embedding
    new_index = faiss_index.ntotal - 1  # El nuevo índice en FAISS
    doc_id_to_faiss_index[url] = new_index  # Actualiza el mapeo

    # No se añade información a la base de datos en esta versión




if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)