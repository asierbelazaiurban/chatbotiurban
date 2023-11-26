##!/usr/bin/env python
# coding: utf-8

import faiss
import numpy as np
from flask import Flask, request, jsonify
import logging
from logging.handlers import RotatingFileHandler
import os
import openai
import requests
from bs4 import BeautifulSoup
import json
import time
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# Configuración del registro de logs
if not os.path.exists('logs'):
    os.mkdir('logs')
file_handler = RotatingFileHandler('logs/chatbotiurban.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.DEBUG)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.DEBUG)
app.logger.info('Inicio de la aplicación ChatbotIUrban')

# Variables y funciones globales

faiss_index = None  # Variable global para almacenar el índice FAISS

def initialize_faiss_index(dimension, chatbot_id):
    global faiss_index
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index_path = f'data/faiss_index/{chatbot_id}/faiss.idx'
    os.makedirs(os.path.dirname(faiss_index_path), exist_ok=True)
    faiss.write_index(faiss_index, faiss_index_path)

def get_faiss_index():
    if faiss_index is None:
        raise ValueError("FAISS index has not been initialized.")
    return faiss_index

def create_database(chatbot_id, dimension=128):
    directory = os.path.join('data/faiss_index', chatbot_id)
    os.makedirs(directory, exist_ok=True)
    initialize_faiss_index(dimension, chatbot_id)

def create_bbdd(chatbot_id):
    create_database(chatbot_id)
    return jsonify({"message": f"FAISS Index created or verified for chatbot_id: {chatbot_id}"}), 200

def generate_embedding(text):
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("La clave API de OpenAI no está configurada.")

    try:
        openai.api_key = openai_api_key
        response = openai.Embedding.create(
            input=text,
            engine='text-embedding-ada-002',
        )
        if 'data' in response and len(response['data']) > 0 and 'embedding' in response['data'][0]:
            embedding = np.array(response['data'][0]['embedding'], dtype=np.float32)
        else:
            raise ValueError('Respuesta de la API de OpenAI no válida o sin datos de embedding.')
    except Exception as e:
        raise ValueError(f'No se pudo obtener el embedding: {e}')

    return embedding

def dividir_en_segmentos(texto, max_tokens):
    tokens = word_tokenize(texto)
    segmentos = []
    segmento_actual = []

    for token in tokens:
        if len(segmento_actual) + len(token.split()) > max_tokens:
            segmentos.append(' '.join(segmento_actual))
            segmento_actual = [token]
        else:
            segmento_actual.append(token)

    if segmento_actual:
        segmentos.append(' '.join(segmento_actual))

    return segmentos if segmentos else [""]

@app.route('/process_urls', methods=['POST'])
def process_urls():
    data = request.json
    chatbot_id = data.get('chatbot_id')
    if not chatbot_id:
        return jsonify({"status": "error", "message": "No chatbot_id provided"}), 400

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
    FAISS_INDEX_DIMENSION = 128

    for url in urls:
        url = url.strip()
        if not url:
            continue

        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text()
            segmentos = dividir_en_segmentos(text, MAX_TOKENS_PER_SEGMENT)

            for segmento in segmentos:
                if not segmento:
                    continue

                embeddings = generate_embedding(segmento)
                if embeddings.shape[1] != FAISS_INDEX_DIMENSION:
                    raise ValueError(f"Dimensión de embeddings incorrecta: esperada {FAISS_INDEX_DIMENSION}, obtenida {embeddings.shape[1]}")

                faiss_index = get_faiss_index()
                faiss_index.add(np.array([embeddings], dtype=np.float32))
        except Exception as e:
            all_indexed = False
            error_message = str(e)
            break

        sleep(0.2)

    if all_indexed:
        return jsonify({"status": "success", "message": "Todo indexado en FAISS correctamente"})
    else:
        return jsonify({"status": "error", "message": f"Error al indexar: {error_message}"})

if __name__ == '__main__':
    app.run(debug=True)


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
            return jsonify({"error": "No chatbot_id provided"}), 400

        # Ruta al índice de FAISS para el chatbot_id
        faiss_index_path = os.path.join('data/faiss_index', f'{chatbot_id}', 'faiss.idx')
        if not os.path.exists(faiss_index_path):
            return jsonify({"error": f"FAISS index not found for chatbot_id: {chatbot_id}"}), 404

        # Cargar el índice de FAISS
        index = faiss.read_index(faiss_index_path)

        # Recibir la consulta de texto
        query_text = data.get('query')

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


@app.route('/list_chatbot_ids', methods=['GET'])
def list_folders():
    directory = 'data/uploads/scraping/'
    folders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    return jsonify(folders)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)