##!/usr/bin/env python
# coding: utf-8


import faiss
import chardet  # Added for encoding detection  # Ensure faiss library is installed
import numpy as np
from flask import Flask, request, jsonify
import logging
from logging.handlers import RotatingFileHandler
from logging import FileHandler 
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
from requests.exceptions import RequestException


app = Flask(__name__)

##### Configuración del registro de logs #####

if not os.path.exists('logs'):
    os.mkdir('logs')

# Configura un manejador de archivos de registro simple
file_handler = FileHandler('logs/chatbotiurban.log')
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.DEBUG)  # Usa DEBUG o INFO según necesites

# Añade el manejador al logger de la aplicación
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.DEBUG)  # Asegúrate de que este nivel sea consistente con file_handler.setLevel

# Mensaje de prueba para verificar que la configuración funciona
app.logger.info('Inicio de la aplicación ChatbotIUrban')


######## FAISS ########

# Global variable to store the FAISS index
faiss_index = None

def initialize_faiss_index(dimension, chatbot_id):
    app.logger.info('Called initialize_faiss_index')
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
    faiss_index_path = f'data/faiss_index/{chatbot_id}/faiss.idx'
    if faiss_index is None or not os.path.exists(faiss_index_path):
        initialize_faiss_index(1536, chatbot_id)  # Asegúrate de que la dimensión sea la correcta
        faiss_index = faiss.read_index(faiss_index_path)
    return faiss_index



def almacenar_en_faiss(respuesta, faiss_index):
    app.logger.info('Called almacenar_en_faiss')
    respuesta_vector = convert_to_vector(respuesta)

    # Convertir el vector de respuesta en un array numpy, que es el formato requerido por FAISS.
    # El vector debe ser de tipo 'float32' y se debe añadir una dimensión extra para convertirlo en un array 2D.
    respuesta_vector_np = np.array([respuesta_vector]).astype('float32')

    # Añadir el vector al índice FAISS.
    faiss_index.add(respuesta_vector_np)
    # Guardar el índice actualizado
    faiss.write_index(faiss_index, f'data/faiss_index/{chatbot_id}/faiss.idx')


def obtener_incrustacion(texto):
    app.logger.info('Called obtener_incrustacion')
    openai_api_key = os.environ.get('OPENAI_API_KEY')

    # Obtener la incrustación del texto
    response = openai.Embedding.create(
        input=texto,
        engine="text-embedding-ada-002"
    )
    
    # Extraer el vector de incrustación
    incrustacion = response['data'][0]['embedding']
    app.logger.info(incrustacion)

    return incrustacion


def convert_to_vector(texto):
    app.logger.info('Called convert_to_vector')
    vector = generate_embedding(texto)
    app.logger.info(vector)
    
    return vector


def obtener_lista_indices(chatbot_id):
    app.logger.info('Called obtener_lista_indices')
    directorio_base = os.path.join('data/faiss_index', chatbot_id)
    ruta_faiss = os.path.join(directorio_base, 'faiss.idx')

    if os.path.exists(ruta_faiss):
        # Cargar el índice FAISS
        indice_faiss = faiss.read_index(ruta_faiss)
        app.logger.info(f"Índice FAISS cargado para chatbot_id {chatbot_id}")
        return indice_faiss
    else:
        app.logger.info("Índice FAISS no encontrado")
        return None


def obtener_respuesta_faiss(indice, chatbot_id):
    mapping_file_path = os.path.join('data/faiss_index', f'{chatbot_id}', 'index_to_text.json')

    if not os.path.exists(mapping_file_path):
        raise FileNotFoundError("Mapping file not found.")

    with open(mapping_file_path, 'r') as file:
        index_to_text = json.load(file)

    texto = index_to_text.get(str(indice))
    if texto is None:
        raise ValueError(f"No text found for index {indice}")

    return texto



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


from nltk.tokenize import word_tokenize

def dividir_en_segmentos(texto, max_tokens):
    # Tokenizar el texto usando NLTK
    tokens = word_tokenize(texto)

    segmentos = []
    segmento_actual = []

    contador_tokens_actual = 0

    for token in tokens:
        # Si añadir el token actual excede el límite de tokens, guarda el segmento actual y comienza uno nuevo
        if contador_tokens_actual + len(token.split()) > max_tokens:
            if segmento_actual:  # Asegura que no se agreguen segmentos vacíos
                segmentos.append(' '.join(segmento_actual))
            segmento_actual = [token]
            contador_tokens_actual = len(token.split())
        else:
            segmento_actual.append(token)
            contador_tokens_actual += len(token.split())

    # Añadir el último segmento si hay tokens restantes
    if segmento_actual:
        segmentos.append(' '.join(segmento_actual))

    return segmentos


######## ########


def prepare_paths(chatbot_id):
    # Rutas de los directorios y archivos necesarios
    faiss_index_path = os.path.join('data/faiss_index', f'{chatbot_id}', 'faiss.idx')
    chatbot_folder = os.path.join('data/uploads/scraping', f'{chatbot_id}')
    mapping_file_path = os.path.join('data/faiss_index', f'{chatbot_id}', 'index_to_text.json')

    # Asegurarse de que el índice FAISS y los directorios existen o inicializarlos
    os.makedirs(os.path.dirname(faiss_index_path), exist_ok=True)
    if not os.path.exists(faiss_index_path):
        create_bbdd(chatbot_id)

    # Asegurarse de que el directorio del chatbot y el archivo de mapeo existen
    os.makedirs(chatbot_folder, exist_ok=True)
    if not os.path.exists(mapping_file_path):
        with open(mapping_file_path, 'w') as file:
            json.dump({}, file)

    return chatbot_folder, mapping_file_path

def read_urls(chatbot_folder, chatbot_id):
    urls_file_path = os.path.join(chatbot_folder, f'{chatbot_id}.txt')
    try:
        with open(urls_file_path, 'r') as file:
            urls = [url.strip() for url in file.readlines()]
        return urls
    except FileNotFoundError:
        app.logger.error(f"Archivo de URLs no encontrado para el chatbot_id {chatbot_id}")
        return None


def get_last_index(mapping_file_path):
    try:
        with open(mapping_file_path, 'r') as file:
            index_to_text = json.load(file)
            if not index_to_text:
                return 0
            return max(map(int, index_to_text.keys()))
    except (FileNotFoundError, json.JSONDecodeError):
        return 

def get_segment_position(segmento, texto_completo):
    """
    Encuentra la posición del primer carácter del segmento dentro del texto completo.

    :param segmento: El segmento de texto a buscar.
    :param texto_completo: El texto completo donde buscar el segmento.
    :return: Índice de la primera aparición del segmento en el texto completo. -1 si no se encuentra.
    """
    return texto_completo.find(segmento)

########  Inicio de endpints hasta el final########


@app.route('/process_urls', methods=['POST'])
def process_urls():
    start_time = time.time()
    app.logger.info('Iniciando process_urls')

    data = request.json
    chatbot_id = data.get('chatbot_id')
    if not chatbot_id:
        return jsonify({"status": "error", "message": "No chatbot_id provided"}), 400

    chatbot_folder, mapping_file_path = prepare_paths(chatbot_id)
    urls = read_urls(chatbot_folder, chatbot_id)
    if urls is None:
        return jsonify({"status": "error", "message": "URLs file not found"}), 404

    all_indexed = True
    error_message = ""
    indice_global = get_last_index(mapping_file_path)
    FAISS_INDEX_DIMENSION = 1536

    for url in urls:
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

                    with open(mapping_file_path, 'r+') as file:
                        index_to_text = json.load(file)
                        nuevo_indice = indice_global + 1
                        index_to_text[nuevo_indice] = {
                            "indice": nuevo_indice,
                            "url": url,
                            "segmento": segmento,
                            "posicion": get_segment_position(segmento, text)
                        }
                        file.seek(0)
                        json.dump(index_to_text, file, indent=4)
                        indice_global += 1

                except Exception as e:
                    app.logger.error(f"Error al procesar el segmento: {e}")
                    all_indexed = False
                    error_message = str(e)
                    continue

        except Exception as e:
            app.logger.error(f"Error al procesar la URL: {e}")
            all_indexed = False
            error_message = str(e)
            break

        time.sleep(0.2)

    if all_indexed:
        return jsonify({"status": "success", "message": "Todo indexado correctamente"})
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

        pregunta = data.get('pregunta')
    

        # Si no hay coincidencia, generar una nueva respuesta usando OpenAI
        openai.api_key = os.environ.get('OPENAI_API_KEY')
        response_openai = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "user", "content": pregunta}])

        nueva_respuesta = response_openai['choices'][0]['message']['content']
       

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


@app.route('/ask_pruebas_asier', methods=['POST'])
def ask_pruebas_asier():
    try:
        data = request.json
        chatbot_id = data.get('chatbot_id')
        token = data.get('token')  # Obtener el token

        if not chatbot_id:
            app.logger.error("No chatbot_id provided in the request")
            return jsonify({"error": "No chatbot_id provided"}), 400

        if not token:
            app.logger.error("No token provided in the request")
            return jsonify({"error": "No token provided"}), 400

        # Obtener el índice FAISS para el chatbot_id dado
        indice_faiss = obtener_lista_indices(chatbot_id)

        app.logger.info(indice_faiss)
        if indice_faiss is None:
            app.logger.error(f"FAISS index not found for chatbot_id: {chatbot_id}")
            return jsonify({"error": f"FAISS index not found for chatbot_id: {chatbot_id}"}), 404

        pregunta_text = data.get('pregunta')
        if not pregunta_text:
            app.logger.error("No pregunta provided in the request")
            return jsonify({"error": "No pregunta provided"}), 400

        # Convertir la consulta en un vector
        query_vector = generate_embedding(pregunta_text)
        app.logger.info(f"Query vector: {query_vector}")

        # Buscar en el índice FAISS
        D, I = indice_faiss.search(np.array([query_vector]).astype(np.float32), k=20)
        app.logger.info(f"Distancias FAISS: {D}")
        app.logger.info(f"Índices FAISS: {I}")

        # Registro de vectores (simulación, ya que no puedes acceder a los vectores directamente)
        for indice in I[0]:
            vector = "simulación de vector para el índice " + str(indice)
            app.logger.info(f"Vector en índice {indice}: {vector}")

        # Ajuste para manejar múltiples resultados y distancias extremas
        umbral_distancia = 1.0  # Aumento del umbral para ser más flexible
        respuestas = []
        for i, distancia in enumerate(D[0]):
            if distancia < umbral_distancia:
                respuesta = obtener_respuesta_faiss(I[0][i], chatbot_id)
                respuestas.append((respuesta, distancia))
                app.logger.info(f"Respuesta encontrada: {respuesta} con distancia {distancia}")

        if respuestas:
            # Ordenar respuestas por distancia
            respuestas.sort(key=lambda x: x[1])
            return jsonify({'respuesta': respuestas[0][0]})  # Devolver la respuesta más cercana
        else:
            return jsonify({'respuesta': 'Respuesta no encontrada'})

    except Exception as e:
        app.logger.error(f"Unexpected error in ask_pruebas_asier function: {e}")
        return jsonify({"error": str(e)}), 500




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