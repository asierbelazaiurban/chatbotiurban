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

#####  #####

######## FAISS ########

# Variables globales para la matriz PCA y el índice FAISS
pca_matrix = None
faiss_index = None


# Función para entrenar la matriz PCA con todos los datos
def entrenar_pca_con_datos(data, n_components=100):
    global pca_matrix
    if data.shape[0] == 0:
        raise ValueError("No data available for PCA training.")
    pca_matrix = faiss.PCAMatrix(data.shape[1], n_components, eigen_power=-0.5)
    pca_matrix.train(data)

def aplicar_pca_a_vector(vector):
    global pca_matrix
    if pca_matrix is None:
        raise Exception("PCA matrix has not been initialized. Please call initialize_and_train_pca first.")
    return pca_matrix.apply_py(np.array([vector]).astype(np.float32))


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
    app.logger.info('Called get_faiss_index')
    global faiss_index
    if faiss_index is None:
        faiss_index_path = f'data/faiss_index/{chatbot_id}/faiss.idx'
        if os.path.exists(faiss_index_path):
            faiss_index = faiss.read_index(faiss_index_path)
        else:
            raise ValueError("FAISS index has not been initialized.")
    return faiss_index


def almacenar_en_faiss(respuesta, faiss_index):
    app.logger.info('Called almacenar_en_faiss')
    respuesta_vector = convert_to_vector(respuesta)

    # Aplicar PCA al vector de respuesta
    respuesta_vector_reducido = aplicar_pca_a_vector(respuesta_vector)

    # Añadir el vector reducido al índice FAISS
    faiss_index.add(respuesta_vector_reducido)


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
    vector = obtener_incrustacion(texto)
    
    return vector


def obtener_lista_indices(chatbot_id):
    app.logger.info('Called obtener_lista_indices')
    directorio_base = os.path.join('data/faiss_index', chatbot_id)
    ruta_faiss = os.path.join(directorio_base, 'faiss.idx')

    if os.path.exists(ruta_faiss):
        # Cargar el índice FAISS
        indice_faiss = faiss.read_index(ruta_faiss)
        app.logger.info(indice_faiss)
        return indice_faiss
    else:
        app.logger.info("obtener_lista_indices "+ None)
        return None


def buscar_en_faiss(query_vector, faiss_index, k=5):
    query_vector_reducido = aplicar_pca_a_vector(query_vector)
    D, I = faiss_index.search(query_vector_reducido, k)
    return D, I

def obtener_respuesta_faiss(indice, chatbot_id):
    # Ruta al archivo JSON que mapea índices FAISS a textos
    mapping_file_path = os.path.join('data/faiss_index', f'{chatbot_id}', 'index_to_text.json')

    # Verificar si el archivo de mapeo existe
    if not os.path.exists(mapping_file_path):
        raise FileNotFoundError("Mapping file not found.")

    # Cargar el mapeo
    with open(mapping_file_path, 'r') as file:
        index_to_text = json.load(file)

    # Obtener el texto correspondiente al índice
    texto = index_to_text.get(str(indice))
    if texto is None:
        raise ValueError(f"No text found for index {indice}")

    return texto

def buscar_y_obtener_respuestas_faiss(query_vector, chatbot_id, k=5):
    global faiss_index
    # Aplicar PCA al vector de consulta
    query_vector_reducido = aplicar_pca_a_vector(query_vector)
    
    # Realizar la búsqueda en el índice FAISS
    D, I = faiss_index.search(query_vector_reducido, k)

    respuestas = []
    for indice in I[0]:
        if indice != -1:  # Verificar que el índice es válido
            respuesta = obtener_respuesta_faiss(indice, chatbot_id)
            respuestas.append(respuesta)
        else:
            respuestas.append("Respuesta no encontrada")

    return respuestas


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
        # Ajustar la dimensión del embedding a 1536
        if embedding.shape[0] != 1536:
            # Rellenar o truncar el embedding según sea necesario
            embedding = np.pad(embedding, (0, max(0, 1536 - embedding.shape[0])), 'constant')
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

def load_existing_embeddings(mapping_file_path):
    try:
        with open(mapping_file_path, 'r') as file:
            existing_embeddings = json.load(file)
            # Suponiendo que los embeddings están almacenados como listas de listas
            return np.array(existing_embeddings)
    except Exception as e:
        app.logger.error(f"Error al cargar embeddings existentes: {e}")
        return None

def validate_embeddings(embeddings, expected_dim):
    if embeddings is None or embeddings.ndim != 2 or embeddings.shape[1] != expected_dim:
        return False
    return True

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
            embeddings = []

            for entry in index_to_text.values():
                if 'embedding' in entry:
                    embedding = entry['embedding']
                    if isinstance(embedding, list) and len(embedding) > 0:
                        # Ajustar la longitud del embedding a 1536
                        adjusted_embedding = np.array(embedding, dtype=np.float32)
                        if adjusted_embedding.shape[0] != 1536:
                            adjusted_embedding = np.pad(adjusted_embedding, (0, 1536 - adjusted_embedding.shape[0]), 'constant', constant_values=0)
                        embeddings.append(adjusted_embedding)
            
            if not embeddings:
                return 0, np.empty((0, 1536), dtype=np.float32)

            embeddings_matrix = np.vstack(embeddings)
            return max(index_to_text.keys(), key=int), embeddings_matrix

    except Exception as e:
        app.logger.error(f"Error in get_last_index: {e}")
        return 0, np.empty((0, 1536), dtype=np.float32)

def entrenar_pca_con_datos(data, n_components=100):
    global pca_matrix
    if data.shape[0] == 0:
        raise ValueError("No data available for PCA training.")
    pca_matrix = faiss.PCAMatrix(data.shape[1], n_components, eigen_power=-0.5)
    pca_matrix.train(data)



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
    indice_global, existing_embeddings = get_last_index(mapping_file_path)

    # Verificación adicional para asegurarse de que los embeddings existentes tienen la dimensión correcta
    if existing_embeddings.size > 0 and (existing_embeddings.ndim != 2 or existing_embeddings.shape[1] != 1536):
        return jsonify({"status": "error", "message": "Existing embeddings are not in a valid format"}), 500

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
        chatbot_id = data.get('chatbot_id')
        token = data.get('token')  # Añadido para recibir un token del front-end

        if not chatbot_id:
            app.logger.error("No chatbot_id provided in the request")
            return jsonify({"error": "No chatbot_id provided"}), 400

        if not token:
            app.logger.error("No token provided in the request")
            return jsonify({"error": "No token provided"}), 400

        # Si no hay coincidencia, generar una nueva respuesta usando OpenAI
        openai.api_key = os.environ.get('OPENAI_API_KEY')
        response_openai = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "user", "content": pregunta}])

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


@app.route('/ask_pruebas_asier', methods=['POST'])
def ask_pruebas_asier():
    try:
        data = request.json
        chatbot_id = data.get('chatbot_id')
        token = data.get('token')  # Nuevo: obtener el token

        if not chatbot_id:
            app.logger.error("No chatbot_id provided in the request")
            return jsonify({"error": "No chatbot_id provided"}), 400

        if not token:  # Verificar la existencia del token
            app.logger.error("No token provided in the request")
            return jsonify({"error": "No token provided"}), 400

        pregunta_text = data.get('pregunta')  # Cambiado de query a pregunta
        if not pregunta_text:
            app.logger.error("No pregunta provided in the request")
            return jsonify({"error": "No pregunta provided"}), 400

        # Convertir la consulta en un vector
        query_vector = convert_to_vector(pregunta_text)  # Usar pregunta_text

        # Realizar la búsqueda en FAISS y obtener respuestas
        respuestas = buscar_y_obtener_respuestas_faiss(query_vector, chatbot_id, k=5)

        return jsonify({'respuestas': respuestas})

    except Exception as e:
        app.logger.error(f"Unexpected error in ask_pruebas_asier function: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/train', methods=['POST'])
def train_faiss():
    try:
        data = request.json
        chatbot_id = data.get('chatbot_id')

        if not chatbot_id:
            app.logger.error("No chatbot_id provided in the request")
            return jsonify({"error": "No chatbot_id provided"}), 400
        app.logger.info(f"Training FAISS index for chatbot_id: {chatbot_id}")

        # Ruta al índice FAISS existente
        faiss_index_path = os.path.join('data/faiss_index', chatbot_id, 'faiss.idx')
        app.logger.info(f"FAISS index path: {faiss_index_path}")

        # Asegurarse de que el archivo de índice FAISS exista
        if not os.path.exists(faiss_index_path):
            app.logger.error(f"FAISS index file not found: {faiss_index_path}")
            return jsonify({"error": "FAISS index file not found"}), 404

        # Cargar el índice FAISS existente
        app.logger.info("Loading existing FAISS index")
        faiss_index = faiss.read_index(faiss_index_path)
        app.logger.warning(faiss_index.ntotal)

        # Verificar si hay vectores en el índice FAISS
        if faiss_index.ntotal == 0:
            app.logger.warning("No vectors found in the FAISS index")
            raise ValueError("No valid vector data found in the FAISS index.")

        app.logger.info(f"Number of vectors in FAISS index: {faiss_index.ntotal}")

        # Extraer los vectores del índice FAISS
        nb_vectors = faiss_index.ntotal
        dimension = faiss_index.d
        app.logger.info(f"Extracting {nb_vectors} vectors of dimension {dimension}")
        vectors = np.zeros((nb_vectors, dimension), dtype=np.float32)
        faiss_index.reconstruct_n(0, nb_vectors, vectors)

        # Entrenar la matriz PCA y aplicarla a los vectores
        n_components = 128  # Número de componentes para PCA
        app.logger.info(f"Training PCA matrix with {n_components} components")
        pca_matrix = faiss.PCAMatrix(dimension, n_components, eigen_power=-0.5)
        pca_matrix.train(vectors)
        vectors_transformed = pca_matrix.apply_py(vectors)

        # Crear un nuevo índice FAISS con los vectores transformados
        app.logger.info("Creating new FAISS index with transformed vectors")
        new_faiss_index = faiss.IndexFlatL2(n_components)
        new_faiss_index.add(vectors_transformed)

        # Guardar el nuevo índice FAISS
        app.logger.info("Saving new FAISS index")
        faiss.write_index(new_faiss_index, faiss_index_path)

        app.logger.info("FAISS index trained/optimized successfully")
        return jsonify({"message": "FAISS index trained/optimized with PCA successfully"})

    except ValueError as ve:
        app.logger.error(f"Error during training: {str(ve)}")
        return jsonify({"error": f"Error during training: {str(ve)}"}), 400
    except Exception as e:
        app.logger.error(f"Unexpected error: {e}")
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