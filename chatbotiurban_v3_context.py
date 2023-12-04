##!/usr/bin/env python
# coding: utf-8

import chardet
import numpy as np
from flask import Flask, request, jsonify
import logging
from logging.handlers import RotatingFileHandler
from logging import FileHandler
import os
import openai
import requests
from bs4 import BeautifulSoup
import json
import time
from urllib.parse import urlparse, urljoin
import random
from time import sleep
import traceback
import gensim.downloader as api
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, GenerationConfig
from datasets import load_dataset
from peft import PeftModel, PeftConfig, LoraConfig, TaskType
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead
from trl import create_reference_model
from trl.core import LengthSampler
import torch
import evaluate
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
import subprocess
import difflib
import re 
from werkzeug.datastructures import FileStorage 
from process_docs import process_file



nltk.download('punkt')
nltk.download('stopwords')

tqdm.pandas()

app = Flask(__name__)


####### Configuración logs #######

if not os.path.exists('logs'):
    os.mkdir('logs')

file_handler = FileHandler('logs/chatbotiurban.log')
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.DEBUG)  # Usa DEBUG o INFO según necesites

app.logger.addHandler(file_handler)
app.logger.setLevel(logging.DEBUG)  # Asegúrate de que este nivel sea consistente con file_handler.setLevel

app.logger.info('Inicio de la aplicación ChatbotIUrban')


#######  #######


MAX_TOKENS_PER_SEGMENT = 7000  # Establecer un límite seguro de tokens por segmento
BASE_DATASET_DIR = "data/uploads/datasets/"
BASE_DATASET_PROMPTS = "data/uploads/prompts/"
BASE_DIR_SCRAPING = "data/uploads/scraping/"
BASE_DIR_DOCS = "data/uploads/docs/"
UPLOAD_FOLDER = 'data/uploads/'  # Ajusta esta ruta según sea necesario
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename, chatbot_id):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    app.logger.info(f'Tiempo total en {function_name}: {time.time() - start_time:.2f} segundos')


def read_urls(chatbot_folder, chatbot_id):
    urls_file_path = os.path.join(chatbot_folder, f'{chatbot_id}.txt')
    try:
        with open(urls_file_path, 'r') as file:
            urls = [url.strip() for url in file.readlines()]
        return urls
    except FileNotFoundError:
        app.logger.error(f"Archivo de URLs no encontrado para el chatbot_id {chatbot_id}")
        return None


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

def procesar_pregunta(pregunta_usuario, preguntas_palabras_clave):
    palabras_pregunta_usuario = set(word_tokenize(pregunta_usuario.lower()))
    stopwords_ = set(stopwords.words('spanish'))
    palabras_relevantes_usuario = palabras_pregunta_usuario - stopwords_

    respuesta_mas_adeacuada = None
    max_coincidencias = 0

    for pregunta, datos in preguntas_palabras_clave.items():
        palabras_clave = set(datos['palabras_clave'])
        coincidencias = palabras_relevantes_usuario.intersection(palabras_clave)

        if len(coincidencias) > max_coincidencias:
            max_coincidencias = len(coincidencias)
            respuesta_mas_adeacuada = datos['respuesta']

    return respuesta_mas_adeacuada

def clean_and_transform_data(data):
    # Aquí puedes implementar tu lógica de limpieza y transformación
    cleaned_data = data.strip().replace("\r", "").replace("\n", " ")
    return cleaned_data

def mejorar_respuesta_con_openai(respuesta_original, pregunta):
    openai.api_key = os.environ.get('OPENAI_API_KEY')

    # Construyendo el prompt para un modelo de chat
    prompt = f"La pregunta es: {pregunta}\nLa respuesta original es: {respuesta_original}\n Responde como si fueras una guía de una oficina de turismo. Siempre responde en el mismo idioma de la pregunta y SIEMPRE contesta sobre el mismo idioma que te están realizando la pregunta. SE BREVE, entre 20 y 40 palabras"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Mejora las respuestas"},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        print(f"Error al interactuar con OpenAI: {e}")
        return None

def mejorar_respuesta_generales_con_openai(pregunta, respuesta, new_prompt="", contexto_adicional="", temperature="", model_gpt="", chatbot_id=""):
    openai.api_key = os.environ.get('OPENAI_API_KEY')

    # Comprobación y carga del dataset basado en chatbot_id
    if chatbot_id:
        try:
            dataset_file_path = os.path.join(BASE_DATASET_PROMPTS, str(chatbot_id), 'prompt.txt')
            with open(dataset_file_path, 'r') as file:
                dataset_content = json.load(file)
            new_prompt = dataset_content
            logging.info(f"Conjunto de datos cargado con éxito para chatbot_id {chatbot_id}.")
        except Exception as e:
            logging.info(f"Error al cargar el conjunto de datos para chatbot_id {chatbot_id}: {e}")

    # Construcción del prompt base
    prompt_base = f"{new_prompt} {contexto_adicional}\n\nPregunta reciente: {pregunta}\nRespuesta original: {respuesta}\n--\n"

    # Intento de generar la respuesta mejorada
    try:
        response = openai.ChatCompletion.create(
            model=model_gpt if model_gpt else "gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt_base},
                {"role": "user", "content": respuesta}
            ],
            temperature=float(temperature) if temperature else 0.5
        )
        improved_response = response.choices[0].message['content'].strip()
        logging.info("Respuesta generada con éxito.")
        return improved_response
    except Exception as e:
        logging.info(f"Error al interactuar con OpenAI: {e}")
        return None


def generar_contexto_con_openai(historial):
    openai.api_key = os.environ.get('OPENAI_API_KEY')

    try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo",
            prompt=f"Resumen del historial de conversación:\n{historial}\n--\n",
            max_tokens=100
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error al generar contexto con OpenAI: {e}")
        return ""


def extraer_palabras_clave(pregunta):
    # Tokenizar la pregunta
    palabras = word_tokenize(pregunta)

    # Filtrar las palabras de parada (stop words) y los signos de puntuación
    palabras_filtradas = [palabra for palabra in palabras if palabra.isalnum()]

    # Filtrar palabras comunes (stop words)
    stop_words = set(stopwords.words('spanish'))
    palabras_clave = [palabra for palabra in palabras_filtradas if palabra not in stop_words]

    return palabras_clave

 ######## Inicio Endpoints ########


####### Utils busqueda en Json #######

# Suponiendo que la función convertir_a_texto convierte cada item del dataset a un texto
def convertir_a_texto(item):
    """
    Convierte un elemento de dataset en una cadena de texto.
    Esta función asume que el 'item' puede ser un diccionario, una lista, o un texto simple.
    """
    if isinstance(item, dict):
        # Concatena los valores del diccionario si 'item' es un diccionario
        return ' '.join(str(value) for value in item.values())
    elif isinstance(item, list):
        # Concatena los elementos de la lista si 'item' es una lista
        return ' '.join(str(element) for element in item)
    elif isinstance(item, str):
        # Devuelve el string si 'item' ya es una cadena de texto
        return item
    else:
        # Convierte el 'item' a cadena si es de otro tipo de dato
        return str(item)


def cargar_dataset(chatbot_id, base_dataset_dir):
    dataset_file_path = os.path.join(base_dataset_dir, str(chatbot_id), 'dataset.json')
    app.logger.info(f"Dataset con ruta {dataset_file_path}")

    try:
        with open(dataset_file_path, 'r') as file:
            data = json.load(file)
            app.logger.info(f"Dataset cargado con éxito desde {dataset_file_path}")
            return [convertir_a_texto(item) for item in data.values()]
    except Exception as e:
        app.logger.error(f"Error al cargar el dataset: {e}")
        return []

def encode_data(data):
    vectorizer = TfidfVectorizer()
    encoded_data = vectorizer.fit_transform(data)
    return encoded_data, vectorizer

def preprocess_query(query):
    tokens = word_tokenize(query.lower())
    processed_query = ' '.join(tokens)
    return processed_query

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

def encontrar_respuesta(pregunta, datos, contexto, longitud_minima=200):
    try:
        # Preprocesar la pregunta y el contexto
        pregunta_procesada = preprocess_query(pregunta)
        contexto_procesado = preprocess_query(contexto)

        # Codificar los datos
        encoded_data, vectorizer = encode_data(datos)

        # Codificar la pregunta y el contexto
        encoded_query = vectorizer.transform([pregunta_procesada + " " + contexto_procesado])

        # Calcular la similitud
        similarity_scores = cosine_similarity(encoded_data, encoded_query).flatten()

        # Ordenar los índices de los documentos por similitud
        indices_ordenados = similarity_scores.argsort()[::-1]

        respuesta_amplia = ""
        for indice in indices_ordenados:
            if similarity_scores[indice] > 0:
                respuesta_amplia += " " + datos[indice]
                if len(word_tokenize(respuesta_amplia)) >= longitud_minima:
                    break

        if len(respuesta_amplia) > 0:
            app.logger.info("Respuesta ampliada encontrada.")
            return respuesta_amplia.strip()
        else:
            app.logger.info("No se encontró ninguna coincidencia.")
            return "No se encontró ninguna coincidencia."

    except Exception as e:
        app.logger.error(f"Error en encontrar_respuesta_amplia: {e}")
        raise e


####### FIN Utils busqueda en Json #######


####### Inicio Endpoints #######


@app.route('/ask_general_context', methods=['POST'])
def ask_general_context():
    contenido = request.json
    pares_pregunta_respuesta = contenido['pares_pregunta_respuesta']
    chatbot_id = contenido['chatbot_id']

    # Cargar datos
    datos = cargar_dataset(chatbot_id, BASE_DATASET_DIR)

    # Construir el historial de preguntas y respuestas
    historial = ""
    for par in pares_pregunta_respuesta:
        historial += f"Pregunta: {par['pregunta']} Respuesta: {par['respuesta_usuario']} "

    # Generar contexto utilizando OpenAI
    contexto = generar_contexto_con_openai(historial)

    respuesta_mejorada_final = ""

    # Procesar la última pregunta si la respuesta del usuario está vacía
    ultima_pregunta = pares_pregunta_respuesta[-1]['pregunta']
    if not pares_pregunta_respuesta[-1]['respuesta_usuario']:
        respuesta_original = encontrar_respuesta(ultima_pregunta, datos, contexto)
        try:
            respuesta_mejorada = mejorar_respuesta_generales_con_openai(
                ultima_pregunta, respuesta_original, new_prompt=contexto, chatbot_id=chatbot_id
            )
            respuesta_mejorada_final = respuesta_mejorada if respuesta_mejorada else respuesta_original
        except Exception as e:
            print(f"Error al mejorar respuesta con OpenAI: {e}")
            respuesta_mejorada_final = respuesta_original

    # Devolver la respuesta mejorada de la última pregunta
    return jsonify({'respuesta': respuesta_mejorada_final})


@app.route('/ask_general', methods=['POST'])
def ask_general():
    contenido = request.json
    pregunta = contenido['pregunta']
    chatbot_id = contenido['chatbot_id']

    # Cargar datos
    datos = cargar_dataset(chatbot_id, BASE_DATASET_DIR)

    # Encontrar respuesta
    respuesta_original = encontrar_respuesta(pregunta, datos)

    # Intentar mejorar la respuesta con OpenAI
    try:
        respuesta_mejorada = mejorar_respuesta_generales_con_openai(
            pregunta, respuesta_original, chatbot_id=chatbot_id
        )
        if respuesta_mejorada:
            return jsonify({'respuesta': respuesta_mejorada})
    except Exception as e:
        print(f"Error al mejorar respuesta con OpenAI: {e}")
        # Devolver la respuesta original en caso de error
        return jsonify({'respuesta': respuesta_original})


@app.route('/uploads', methods=['POST'])
def upload_file():
    try:
        logging.info("Procesando solicitud de carga de archivo")

        if 'documento' not in request.files:
            logging.warning("Archivo 'documento' no encontrado en la solicitud")
            return jsonify({"respuesta": "No se encontró el archivo 'documento'", "codigo_error": 1})
        
        uploaded_file = request.files['documento']
        chatbot_id = request.form.get('chatbot_id')
        logging.info(f"Archivo recibido: {uploaded_file.filename}, Chatbot ID: {chatbot_id}")

        if uploaded_file.filename == '':
            logging.warning("Nombre de archivo vacío")
            return jsonify({"respuesta": "No se seleccionó ningún archivo", "codigo_error": 1})

        docs_folder = os.path.join(BASE_DIR_DOCS, str(chatbot_id))
        os.makedirs(docs_folder, exist_ok=True)
        logging.info(f"Carpeta del chatbot creada o ya existente: {docs_folder}")

        file_extension = os.path.splitext(uploaded_file.filename)[1][1:].lower()
        file_path = os.path.join(docs_folder, uploaded_file.filename)
        uploaded_file.save(file_path)
        logging.info(f"Archivo guardado en: {file_path}")

        readable_content = process_file(file_path, file_extension)
        if readable_content is None:
            logging.error("No se pudo procesar el archivo")
            return jsonify({"respuesta": "Error al procesar el archivo", "codigo_error": 1})

        # Contar palabras en el contenido
        word_count = len(readable_content.split())

        dataset_file_path = os.path.join(BASE_DATASET_DIR, f"{chatbot_id}", "dataset.json")
        os.makedirs(os.path.dirname(dataset_file_path), exist_ok=True)

        dataset_entries = {}
        if os.path.exists(dataset_file_path):
            with open(dataset_file_path, 'r', encoding='utf-8') as json_file:
                dataset_entries = json.load(json_file)
                logging.info("Archivo JSON del dataset existente cargado")

        indice = uploaded_file.filename
        dataset_entries[indice] = {
            "indice": indice,
            "url": file_path,
            "dialogue": readable_content
        }

        with open(dataset_file_path, 'w', encoding='utf-8') as json_file_to_write:
            json.dump(dataset_entries, json_file_to_write, ensure_ascii=False, indent=4)
            logging.info("Archivo JSON del dataset actualizado y guardado")

        return jsonify({
            "respuesta": "Archivo procesado y añadido al dataset con éxito.",
            "word_count": word_count,  # Incluir el recuento de palabras en la respuesta
            "codigo_error": 0
        })

    except Exception as e:
        app.logger.error(f"Error durante el procesamiento general. Error: {e}")
        return jsonify({"respuesta": f"Error durante el procesamiento. Error: {e}", "codigo_error": 1})



@app.route('/save_text', methods=['POST'])
def save_text():
    try:
        logging.info("Procesando solicitud para guardar texto")

        # Obtener el JSON de la solicitud
        data = request.get_json()
        text = data.get('texto')
        chatbot_id = data.get('chatbot_id')

        if not text or not chatbot_id:
            return jsonify({"respuesta": "Falta texto o chatbot_id", "codigo_error": 1})

        # Contar palabras en el texto
        word_count = len(text.split())

        # Definir la ruta del dataset
        dataset_folder = os.path.join('data', 'uploads', 'datasets', chatbot_id)
        os.makedirs(dataset_folder, exist_ok=True)
        dataset_file_path = os.path.join(dataset_folder, 'dataset.json')

        # Cargar o crear el archivo del dataset
        dataset_entries = {}
        if os.path.exists(dataset_file_path):
            with open(dataset_file_path, 'r', encoding='utf-8') as json_file:
                dataset_entries = json.load(json_file)

        # Verificar si el texto ya existe en el dataset
        existing_entry = None
        for entry in dataset_entries.values():
            if entry.get("dialogue") == text:
                existing_entry = entry
                break

        # Si el texto ya existe, no añadirlo
        if existing_entry:
            return jsonify({
                "respuesta": "El texto ya existe en el dataset.",
                "word_count": word_count,
                "codigo_error": 0
            })

        # Agregar nueva entrada al dataset
        new_index = len(dataset_entries) + 1
        dataset_entries[new_index] = {
            "indice": new_index,
            "url": "",
            "dialogue": text
        }

        # Guardar el archivo del dataset actualizado
        with open(dataset_file_path, 'w', encoding='utf-8') as json_file_to_write:
            json.dump(dataset_entries, json_file_to_write, ensure_ascii=False, indent=4)

        return jsonify({
            "respuesta": "Texto guardado con éxito en el dataset.",
            "word_count": word_count,
            "codigo_error": 0
        })

    except Exception as e:
        logging.error(f"Error durante el procesamiento. Error: {e}")
        return jsonify({"respuesta": f"Error durante el procesamiento. Error: {e}", "codigo_error": 1})



@app.route('/process_urls', methods=['POST'])
def process_urls():
    start_time = time.time()
    app.logger.info('Iniciando process_urls')

    data = request.json
    chatbot_id = data.get('chatbot_id')
    if not chatbot_id:
        return jsonify({"status": "error", "message": "No chatbot_id provided"}), 400

    chatbot_folder = os.path.join('data/uploads/scraping', f'{chatbot_id}')
    urls = read_urls(chatbot_folder, chatbot_id)
    if urls is None:
        return jsonify({"status": "error", "message": "URLs file not found"}), 404
    
    all_processed = True
    error_message = ""
    dataset_entries = {}

    for url in urls:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text()
            dataset_entries[len(dataset_entries) + 1] = {
                "indice": len(dataset_entries) + 1,
                "url": url,
                "dialogue": text
            }

        except Exception as e:
            app.logger.error(f"Error al procesar la URL: {e}")
            all_processed = False
            error_message = str(e)
            break

    if all_processed:
        # Construye la ruta del archivo donde se guardará el dataset
        dataset_folder = os.path.join('data', 'uploads', 'datasets', chatbot_id)
        os.makedirs(dataset_folder, exist_ok=True)  # Crea la carpeta si no existe
        dataset_file_path = os.path.join(dataset_folder, 'dataset.json')

        # Guarda el dataset en un archivo JSON
        with open(dataset_file_path, 'w') as dataset_file:
            json.dump(dataset_entries, dataset_file, indent=4)


        return jsonify({"status": "success", "message": "Datos procesados y almacenados correctamente"})
    else:
        return jsonify({"status": "error", "message": f"Error al procesar datos: {error_message}"})

    app.logger.info(f'Tiempo total en process_urls: {time.time() - start_time:.2f} segundos')



@app.route('/save_urls', methods=['POST'])
def save_urls():
    data = request.json
    urls = data.get('urls', [])  # Asumimos que 'urls' es una lista de URLs
    chatbot_id = data.get('chatbot_id')

    if not urls or not chatbot_id:
        return jsonify({"error": "Missing 'urls' or 'chatbot_id'"}), 400

    chatbot_folder = os.path.join('data/uploads/scraping', f'{chatbot_id}')
    os.makedirs(chatbot_folder, exist_ok=True)

    file_path = os.path.join(chatbot_folder, f'{chatbot_id}.txt')

    if os.path.exists(file_path):
        os.remove(file_path)

    with open(file_path, 'w') as file:
        for url in urls:
            file.write(url + '\n')

    return jsonify({"status": "success", "message": "URLs saved successfully"})


@app.route('/url_for_scraping', methods=['POST'])
def url_for_scraping():
    try:
        data = request.get_json()
        base_url = data.get('url')
        chatbot_id = data.get('chatbot_id')

        if not base_url:
            return jsonify({'error': 'No URL provided'}), 400

        save_dir = os.path.join('data/uploads/scraping', f'{chatbot_id}')
        os.makedirs(save_dir, exist_ok=True)

        file_path = os.path.join(save_dir, f'{chatbot_id}.txt')

        if os.path.exists(file_path):
            os.remove(file_path)

        def same_domain(url):
            return urlparse(url).netloc == urlparse(base_url).netloc

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

        with open(file_path, 'w') as text_file:
            for url_data in urls_data:
                text_file.write(url_data['url'] + '\n')

        return jsonify(urls_data)
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500


@app.route('/url_for_scraping_by_sitemap', methods=['POST'])
def url_for_scraping_by_sitemap():
    try:
        data = request.get_json()
        sitemap_url = data.get('url')
        chatbot_id = data.get('chatbot_id')

        logging.info(f"Recibida solicitud para chatbot_id: {chatbot_id}, URL del sitemap: {sitemap_url}")

        if not sitemap_url:
            logging.error("No se proporcionó URL del sitemap")
            return jsonify({'error': 'No se proporcionó URL del sitemap'}), 400

        # Crear el directorio para guardar los datos
        save_dir = os.path.join('data/uploads/scraping', f'{chatbot_id}')
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'{chatbot_id}.txt')

        logging.info(f"Directorio creado: {save_dir}")

        # Función para solicitar el sitemap
        def request_sitemap(url):
            try:
                response = requests.get(url)
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                logging.error(f"Error al descargar el sitemap: {e}")
                return None

        # Obtener y procesar el sitemap
        sitemap_content = request_sitemap(sitemap_url)
        if not sitemap_content:
            return jsonify({'error': 'Error al descargar el sitemap'}), 500

        soup = BeautifulSoup(sitemap_content, 'xml')
        urls = [loc.text for loc in soup.find_all('loc')]

        logging.info(f"URLs encontradas en el sitemap: {len(urls)}")

        # Guardar las URLs en un archivo
        with open(file_path, 'w') as file:
            for url in urls:
                file.write(url + '\n')

        logging.info(f"URLs guardadas en {file_path}")

        return jsonify({'message': 'Sitemap procesado correctamente', 'urls_count': len(urls)})
    except Exception as e:
        logging.error(f"Error inesperado: {e}")
        return jsonify({'error': f'Error inesperado: {str(e)}'}), 500



@app.route('/delete_urls', methods=['POST'])
def delete_urls():
    data = request.json
    urls_to_delete = set(data.get('urls', []))  # Conjunto de URLs a eliminar
    chatbot_id = data.get('chatbot_id')  # Identificador del chatbot

    if not urls_to_delete or not chatbot_id:
        return jsonify({"error": "Missing 'urls' or 'chatbot_id'"}), 400

    chatbot_folder = os.path.join('data/uploads/scraping', str(chatbot_id))

    print("Ruta del directorio del chatbot:", chatbot_folder)

    if not os.path.exists(chatbot_folder):
        return jsonify({"status": "error", "message": "Chatbot folder not found"}), 404

    file_name = f"{chatbot_id}.txt"
    file_path = os.path.join(chatbot_folder, file_name)

    print("Ruta del archivo de URLs:", file_path)

    if not os.path.exists(file_path):
        return jsonify({"status": "error", "message": f"File {file_name} not found in chatbot folder"}), 404

    try:
        with open(file_path, 'r+') as file:
            existing_urls = set(file.read().splitlines())
            updated_urls = existing_urls - urls_to_delete

            file.seek(0)
            file.truncate()

            for url in updated_urls:
                if url.strip():  # Asegura que la URL no sea una línea vacía
                    file.write(url + '\n')

        return jsonify({"status": "success", "message": "URLs deleted successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/ask', methods=['POST'])
def ask():
    app.logger.info("Inicio de la función /ask")
    try:
        app.logger.info("Intentando obtener datos JSON de la solicitud")
        data = request.get_json()
        if not data:
            raise ValueError("No se recibieron datos JSON válidos.")

        app.logger.info(f"Datos recibidos: {data}")
        chatbot_id = data.get('chatbot_id')
        pregunta = data.get('pregunta')
        token = data.get('token')

        if not pregunta or not chatbot_id or not token:
            app.logger.error("Falta chatbot_id o o token o pregunta en la solicitud.")
            return jsonify({"error": "Falta chatbot_id o pregunta en la solicitud."}), 400

        json_file_path = f'data/uploads/pre_established_answers/{chatbot_id}/pre_established_answers.json'
        app.logger.info(f"Ruta del archivo JSON: {json_file_path}")

        if not os.path.exists(json_file_path):
            app.logger.error("Archivo de respuestas preestablecidas no encontrado.")
            return jsonify({'error': 'Archivo de respuestas preestablecidas no encontrado.'}), 404

        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            preguntas_respuestas = json.load(json_file)

        app.logger.info("Archivo JSON cargado correctamente")
        respuesta = None
        max_similarity = 0.0
        for entry in preguntas_respuestas.values():
            palabras_clave = ' '.join(entry["palabras_clave"])
            similarity = difflib.SequenceMatcher(None, pregunta, palabras_clave).ratio()
            if similarity > max_similarity:
                max_similarity = similarity
                respuesta = entry["respuesta"]
                app.logger.info(f"Similitud encontrada: {similarity} para la pregunta '{pregunta}'")

        if max_similarity <= 0.9:
            app.logger.info("No se encontró una coincidencia adecuada, llamando a /ask_general")
            try:
                contenido_general = {
                    'pregunta': pregunta,
                    'chatbot_id': chatbot_id,
                    'token': token,  
                }

                # Llamando a /ask_general internamente
                respuesta_general = ask_general(contenido_general)
                return respuesta_general

            except Exception as e:
                app.logger.error(f"Error al llamar a /ask_general: {e}")
                # Si hay un error, devolver la respuesta generada por OpenAI
                return jsonify({'respuesta': respuesta})

        # Intentar mejorar la respuesta con OpenAI
        try:
            app.logger.info("Intentando mejorar respuesta con OpenAI")
            respuesta_mejorada = mejorar_respuesta_con_openai(respuesta, pregunta)
            if respuesta_mejorada:
                return jsonify({'respuesta': respuesta_mejorada})
        except Exception as e:
            app.logger.error(f"Error al mejorar respuesta con OpenAI: {e}")
            # Devolver la respuesta original en caso de error
            return jsonify({'respuesta': respuesta})

    except Exception as e:
        app.logger.error(f"Error inesperado en /ask: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/pre_established_answers', methods=['POST'])
def pre_established_answers():
    data = request.json
    chatbot_id = data.get('chatbot_id')
    pregunta = data.get('pregunta')
    respuesta = data.get('respuesta')

    if not (chatbot_id and pregunta and respuesta):
        return jsonify({"error": "Faltan datos en la solicitud (chatbot_id, pregunta, respuesta)."}), 400

    # Extraer palabras clave de la pregunta
    palabras_clave = extraer_palabras_clave(pregunta)

    json_file_path = f'data/uploads/pre_established_answers/{chatbot_id}/pre_established_answers.json'

    # Verificar si el directorio existe, si no, crearlo
    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)

    # Intentar leer el archivo JSON existente o crear un nuevo diccionario si no existe
    if os.path.isfile(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            preguntas_respuestas = json.load(json_file)
    else:
        preguntas_respuestas = {}

    # Actualizar o añadir la nueva pregunta y respuesta
    preguntas_respuestas[pregunta] = {
        "Pregunta": [pregunta],
        "palabras_clave": palabras_clave,
        "respuesta": respuesta
    }

    # Guardar los cambios en el archivo JSON
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(preguntas_respuestas, json_file, ensure_ascii=False, indent=4)

    # Devolver la respuesta
    return jsonify({'mensaje': 'Pregunta y respuesta guardadas correctamente'})


@app.route('/change_params_prompt_temperature_and_model', methods=['POST'])
def change_params():
    data = request.json
    new_prompt = data.get('new_prompt')
    chatbot_id = data.get('chatbot_id')
    temperature = data.get('temperature', '')
    model_gpt = data.get('model_gpt', 'gpt-3.5-turbo')

    if not new_prompt or not chatbot_id:
        return jsonify({"error": "Los campos 'new_prompt' y 'chatbot_id' son requeridos"}), 400

    # Guardar el nuevo prompt en un archivo
    prompt_file_path = f"data/uploads/prompts/{chatbot_id}/prompt.txt"
    os.makedirs(os.path.dirname(prompt_file_path), exist_ok=True)

    with open(prompt_file_path, 'w') as file:
        file.write(new_prompt)

    # Leer el contenido del nuevo prompt del archivo
    with open(prompt_file_path, 'r') as file:
        saved_prompt = file.read()

    pregunta = ""  # 'pregunta' está vacía porque no se utiliza en este contexto
    respuesta = ""  # 'respuesta' está vacía porque no se utiliza en este contexto
    prompt_base = f"Cuando recibas una pregunta, comienza con: '{pregunta}'. Luego sigue con tu respuesta original: '{respuesta}'. {saved_prompt}"

    # Comprobación para ver si el nuevo prompt es igual al prompt base
    if new_prompt == prompt_base:
        return jsonify({"mensaje": "El nuevo prompt es igual al prompt base. No se realizó ninguna acción."})

    # Llamada a la función para actualizar la configuración del modelo
    mejorar_respuesta_generales_con_openai(saved_prompt, temperature, model_gpt)

    return jsonify({"mensaje": "Parámetros cambiados con éxito"})



@app.route('/list_chatbot_ids', methods=['GET'])
def list_folders():
    directory = 'data/uploads/scraping/'
    folders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    return jsonify(folders)

 ######## Fin Endpoints ######## 

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)