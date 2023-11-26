##!/usr/bin/env python
# coding: utf-8


import faiss
import chardet  # Added for encoding detection  # Ensure faiss library is installed
import numpy as np
from flask import Flask, request, jsonify
import logging
from logging.handlers import RotatingFileHandler
import os
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
import traceback
import gensim.downloader as api
from gensim.models import Word2Vec
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize


app = Flask(__name__)


# Configuración del registro de logs
if not os.path.exists('logs'):
    os.mkdir('logs')
file_handler = RotatingFileHandler('logs/chatbotiurban.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.DEBUG)  # Cambiado a DEBUG para capturar todos los registros
app.logger.addHandler(file_handler)

app.logger.setLevel(logging.DEBUG)  # Cambiado a DEBUG
app.logger.info('Registro de prueba inmediatamente después de la configuración de logging')



# Configuración del registro de logs
if not os.path.exists('logs'):
    os.mkdir('logs')
file_handler = RotatingFileHandler('logs/chatbotiurban.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)

app.logger.setLevel(logging.INFO)
app.logger.info('Inicio de la aplicación ChatbotIUrban')



######## Creación de bbddd FAISS para cada cliente ########

# Global variable to store the FAISS index
faiss_index = None

def initialize_faiss_index(dimension, chatbot_id):
    global faiss_index  # Usa la variable global 'faiss_index'
    start_time = time.time()  # Inicio del registro de tiempo
    app.logger.info('Iniciando initialize_faiss_index')

    # Crea un nuevo índice de FAISS con la dimensión especificada
    faiss_index = faiss.IndexFlatL2(dimension)

    # Reemplaza {chatbot_id} con el valor real del chatbot_id
    faiss_index_path = f'data/faiss_index/{chatbot_id}/faiss.idx'

    # Crea la carpeta si no existe
    os.makedirs(os.path.dirname(faiss_index_path), exist_ok=True)

    # Escribe el índice de FAISS
    faiss.write_index(faiss_index, faiss_index_path)
    app.logger.info(f'Tiempo total en initialize_faiss_index: {time.time() - start_time:.2f} segundos')


def get_faiss_index(chatbot_id):
    global faiss_index
    if faiss_index is None:
        faiss_index_path = f'data/faiss_index/{chatbot_id}/faiss.idx'
        if os.path.exists(faiss_index_path):
            faiss_index = faiss.read_index(faiss_index_path)
        else:
            raise ValueError("FAISS index has not been initialized.")
    return faiss_index


def almacenar_en_faiss(respuesta, faiss_index):

    respuesta_vector = convert_to_vector(respuesta)

    # Convertir el vector de respuesta en un array numpy, que es el formato requerido por FAISS.
    # El vector debe ser de tipo 'float32' y se debe añadir una dimensión extra para convertirlo en un array 2D.
    respuesta_vector_np = np.array([respuesta_vector]).astype('float32')

    # Añadir el vector al índice FAISS.
    faiss_index.add(respuesta_vector_np)



def obtener_incrustacion(texto):
    # Configura tu clave API de OpenAI aquí
    openai_api_key = os.environ.get('OPENAI_API_KEY')

    # Obtener la incrustación del texto
    response = openai.Embedding.create(
        input=texto,
        engine="text-embedding-ada-002"
    )
    
    # Extraer el vector de incrustación
    incrustacion = response['data'][0]['embedding']

    return incrustacion

def convert_to_vector(texto):
    # Utilizar la función 'obtener_incrustacion' para convertir texto en vector
    vector = obtener_incrustacion(texto)
    return vector


def obtener_lista_indices(chatbot_id):
    """
    Carga el índice FAISS para un chatbot específico.
    :param chatbot_id: ID del chatbot para el cual se cargará el índice FAISS.
    :return: Índice FAISS del chatbot especificado.
    """
    directorio_base = os.path.join('data/faiss_index', chatbot_id)
    ruta_faiss = os.path.join(directorio_base, 'faiss.idx')

    if os.path.exists(ruta_faiss):
        # Cargar el índice FAISS
        indice_faiss = faiss.read_index(ruta_faiss)
        return indice_faiss
    else:
        return None

    

def create_database(chatbot_id):
    start_time = time.time()
    app.logger.info('Iniciando create_database')

    directory = os.path.join('data/faiss_index', chatbot_id)
    file_name = 'faiss.idx'
    file_path = os.path.join(directory, file_name)
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    if not os.path.exists(file_path):
        initialize_faiss_index(1536, chatbot_id)  # Usa la dimensión 1536 directamente aquí

    return file_path

def create_bbdd(chatbot_id):
    data = request.json

    index_file_path = create_database(chatbot_id)
    message = f"FAISS Index created or verified at: {index_file_path}"
    
    return jsonify({"message": message}), 200
    app.logger.info(f'Tiempo total en {function_name}: {time.time() - start_time:.2f} segundos')


######## ########


######## Embedding, tokenizacion y add to FAISS ########



# Suponiendo que tienes un diccionario para mapear IDs de documentos a índices en FAISS
doc_id_to_faiss_index = {}

def add_document_to_faiss(document, doc_id):
    start_time = time.time()  # Inicio del registro de tiempo
    app.logger.info('Iniciando add_document_to_faiss')

    global doc_id_to_faiss_index  # Supone que este mapeo es una variable global

    # Verifica si el documento ya está indexado
    if doc_id in doc_id_to_faiss_index:
        print(f"Documento con ID {doc_id} ya indexado.")
        return

    # Genera el embedding para el nuevo documento
    # Asume que 'generate_embedding' puede manejar diferentes tipos de entrada
    embedding = generate_embedding(document)

    # Obtiene el índice de FAISS actual
    faiss_index = get_faiss_index()

    # Añade el embedding al índice de FAISS y actualiza el mapeo
    faiss_index.add(np.array([embedding]))  # Añade el nuevo embedding
    new_index = faiss_index.ntotal - 1  # El nuevo índice en FAISS
    doc_id_to_faiss_index[doc_id] = new_index  # Actualiza el mapeo



def generate_embedding(text):
    start_time = time.time()
    app.logger.info('Iniciando generate_embedding')

    openai_api_key = os.environ.get('OPENAI_API_KEY')

    if not openai_api_key:
        app.logger.error("La clave API de OpenAI no está configurada.")
        raise ValueError("La clave API de OpenAI no está configurada.")

    try:
        openai.api_key = openai_api_key
        response = openai.Embedding.create(
            input=text,
            engine='text-embedding-ada-002',
        )
    except Exception as e:
        app.logger.error(f'Error al generar embedding: {e}')
        raise

    if 'data' in response and len(response['data']) > 0 and 'embedding' in response['data'][0]:
        embedding = np.array(response['data'][0]['embedding'], dtype=np.float32)
    else:
        app.logger.error('Respuesta de la API de OpenAI no válida o sin datos de embedding.')
        raise ValueError('Respuesta de la API de OpenAI no válida o sin datos de embedding.')

    app.logger.info(f'Tiempo total en generate_embedding: {time.time() - start_time:.2f} segundos')
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
    app.logger.info(f'Tiempo total en {function_name}: {time.time() - start_time:.2f} segundos')



######## ########

######## Hiperparámetros y funciones generales ########

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

#metodo param la subida de documentos

MAX_TOKENS_PER_SEGMENT = 7000  # Establecer un límite seguro de tokens por segmento


# Configuraciones
UPLOAD_FOLDER = 'data/uploads/'  # Ajusta esta ruta según sea necesario
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'csv', 'docx', 'xlsx', 'pptx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename, chatbot_id):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    app.logger.info(f'Tiempo total en {function_name}: {time.time() - start_time:.2f} segundos')


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

    return [texto[:max_tokens]]
    app.logger.info(f'Tiempo total en {function_name}: {time.time() - start_time:.2f} segundos')


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
    app.logger.info(f'Tiempo total en {function_name}: {time.time() - start_time:.2f} segundos')

######## ########


########  Inicio de endpints hasta el final########

@app.route('/process_urls', methods=['POST'])
def process_urls():
    start_time = time.time()
    app.logger.info('Iniciando process_urls')

    data = request.json
    chatbot_id = data.get('chatbot_id')
    if not chatbot_id:
        return jsonify({"status": "error", "message": "No chatbot_id provided"}), 400

    # Asegúrate de que el índice FAISS existe o inicialízalo
    faiss_index_path = os.path.join('data/faiss_index', f'{chatbot_id}', 'faiss.idx')
    if not os.path.exists(faiss_index_path):
        create_bbdd(chatbot_id)

    chatbot_folder = os.path.join('data/uploads/scraping', f'{chatbot_id}')
    if not os.path.exists(chatbot_folder):
        os.makedirs(chatbot_folder)

    try:
        with open(os.path.join(chatbot_folder, f'{chatbot_id}.txt'), 'r') as file:
            urls = file.readlines()
    except FileNotFoundError:
        return jsonify({"status": "error", "message": "URLs file not found for the provided chatbot_id"}), 404

    all_indexed = True
    error_message = ""

    # Asumiendo que la dimensión del índice FAISS es 1536
    FAISS_INDEX_DIMENSION = 1536

    for url in urls:
        url = url.strip()
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text()

            segmentos = dividir_en_segmentos(text, MAX_TOKENS_PER_SEGMENT)

            for segmento in segmentos:
                try:
                    embeddings = generate_embedding(segmento)
                    if embeddings.shape[0] != FAISS_INDEX_DIMENSION:
                        app.logger.error(f"Dimensión de embeddings incorrecta: esperada {FAISS_INDEX_DIMENSION}, obtenida {embeddings.shape[0]}")
                        continue

                    faiss_index = get_faiss_index(chatbot_id)
                    faiss_index.add(np.array([embeddings], dtype=np.float32))
                except Exception as e:
                    app.logger.error(f"Error al procesar el segmento de la URL {url}: {e}")
                    all_indexed = False
                    error_message = str(e)
                    continue

        except Exception as e:
            app.logger.error(f"Error al procesar la URL {url}: {e}")
            all_indexed = False
            error_message = str(e)
            break

        sleep(0.2)

    if all_indexed:
        return jsonify({"status": "success", "message": "Todo indexado en FAISS correctamente"})
    else:
        return jsonify({"status": "error", "message": f"Error al indexar: {error_message}"})

    app.logger.info(f'Tiempo total en process_urls: {time.time() - start_time:.2f} segundos')




@app.route('/save_urls', methods=['POST'])
def save_urls():
    data = request.json
    urls = data.get('urls', [])  # Asumimos que 'urls' es una lista de URLs
    chatbot_id = data.get('chatbot_id')

    # Comprueba si existe el índice de FAISS para el chatbot_id
    faiss_index_path = os.path.join('data/faiss_index', f'{chatbot_id}', 'faiss.idx')
    if not os.path.exists(faiss_index_path):
        create_bbdd(chatbot_id)  # Esta función ya debe incluir initialize_faiss_index

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
@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        chatbot_id = data.get('chatbot_id')
        if not chatbot_id:
            app.logger.error("No chatbot_id provided in the request")
            return jsonify({"error": "No chatbot_id provided"}), 400

        # Llamar a obtener_lista_indices con el chatbot_id
        indice_faiss = obtener_lista_indices(chatbot_id)
        if indice_faiss is None:
            app.logger.error(f"FAISS index not found for chatbot_id: {chatbot_id}")
            return jsonify({"error": f"FAISS index not found for chatbot_id: {chatbot_id}"}), 404


        query_text = data.get('query')
        if not query_text:
            app.logger.error("No query provided in the request")
            return jsonify({"error": "No query provided"}), 400

        query_vector = convert_to_vector(query_text)

        # Buscar en el índice FAISS
        D, I = indice_faiss.search(np.array([query_vector]).astype(np.float32), k=1)

        umbral_distancia = 0.5  # Ajusta este valor según sea necesario
        if D[0][0] < umbral_distancia:
            mejor_respuesta = obtener_respuesta_faiss(I[0][0], indice_faiss)
            return jsonify({'respuesta': mejor_respuesta})

        # Si no hay coincidencia, generar una nueva respuesta usando OpenAI
        openai.api_key = openai_api_key
        response_openai = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "user", "content": query_text}])

        nueva_respuesta = response_openai['choices'][0]['message']['content']
        almacenar_en_faiss(nueva_respuesta, indice_faiss)

        return jsonify({'respuesta': nueva_respuesta})

    except Exception as e:
        app.logger.error(f"Unexpected error in ask function: {e}")
        return jsonify({"error": str(e)}), 500


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



# Método externo para generar respuestas con OpenAI
def generate_response_with_openai(info, model="text-embedding-ada-002", max_tokens=150, temperature=0.7, top_p=1, frequency_penalty=0, presence_penalty=0, language="es"):
    response = openai.Completion.create(
        model=model,
        prompt=info,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        language=language
    )
    return response.choices[0].text.strip()


@app.route('/list_chatbot_ids', methods=['GET'])
def list_folders():
    directory = 'data/uploads/scraping/'
    folders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    return jsonify(folders)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)