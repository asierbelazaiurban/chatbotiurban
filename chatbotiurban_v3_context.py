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
from sentence_transformers import SentenceTransformer, util



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
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                return response
        except RequestException as e:
            logging.error(f"Attempt {attempt + 1} failed for {url}: {e}")
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

    app.logger.info(f"Respuesta más adecuada encontrada: {respuesta_mas_adeacuada}")
    return respuesta_mas_adeacuada


def clean_and_transform_data(data):
    cleaned_data = data.strip().replace("\r", "").replace("\n", " ")
    app.logger.info("Datos limpiados y transformados")
    return cleaned_data


def mejorar_respuesta_con_openai(respuesta_original, pregunta):
    openai.api_key = os.environ.get('OPENAI_API_KEY')
    ("Mejorando respuesta con OpenAI")

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
        app.logger.info("Mejorando respuesta con OpenAI")(f"Error al interactuar con OpenAI: {e}")
        return None

def mejorar_respuesta_generales_con_openai(pregunta, respuesta, new_prompt="", contexto_adicional="", temperature="", model_gpt="", chatbot_id=""):
    # Configurar la clave API de OpenAI
    openai.api_key = os.environ.get('OPENAI_API_KEY')

    app.logger.info("Entrando en OpenAI")

    # Definir las rutas base para los prompts
    BASE_PROMPTS_DIR = "data/uploads/prompts/"

    # Inicializar la variable para almacenar el new_prompt obtenido por chatbot_id
    new_prompt_by_id = None

    # Intentar cargar el new_prompt desde los prompts, según chatbot_id
    if chatbot_id:
        prompt_file_path = os.path.join(BASE_PROMPTS_DIR, str(chatbot_id), 'prompt.txt')
        try:
            with open(prompt_file_path, 'r') as file:
                new_prompt_by_id = file.read()
            app.logger.info(f"Prompt cargado con éxito desde prompts para chatbot_id {chatbot_id}.")
        except Exception as e:
            app.logger.info(f"Error al cargar desde prompts para chatbot_id {chatbot_id}: {e}")

    # Utilizar new_prompt_by_id si no viene vacío, de lo contrario usar new_prompt proporcionado
    if new_prompt_by_id:
        new_prompt = new_prompt_by_id
    elif new_prompt:
        prompt_file_path_direct = os.path.join(BASE_PROMPTS_DIR, new_prompt)
        try:
            with open(prompt_file_path_direct, 'r') as file:
                new_prompt_direct = file.read()
            new_prompt = new_prompt_direct
            app.logger.info(f"Prompt cargado con éxito directamente desde {prompt_file_path_direct}.")
        except Exception as e:
            app.logger.info(f"Error al cargar prompt directamente desde {prompt_file_path_direct}: {e}")

    # Verificar si hay contexto adicional. Si no hay, detener el proceso y devolver un mensaje
    if not contexto_adicional:
        app.logger.error("Falta contexto adicional para continuar.")
        return "Se requiere contexto adicional para procesar esta solicitud."

    # Si no se ha proporcionado new_prompt, usar un prompt predeterminado
    if not new_prompt:
        new_prompt = ("Mantén la coherencia con la pregunta y, si la respuesta no se alinea, indica 'No tengo información "
                      "en este momento sobre este tema, ¿puedo ayudarte en algo más?'. Actúa como un guía turístico experto, "
                      "presentando tus respuestas en forma de listas para facilitar la planificación diaria de actividades. "
                      "Es crucial responder en el mismo idioma que la pregunta. Al finalizar tu respuesta, recuerda sugerir "
                      "'Si deseas más información, crea tu ruta con Cicerone o consulta las rutas de expertos locales'. "
                      "Si careces de la información solicitada, evita comenzar con 'Lo siento, no puedo darte información específica'. "
                      "En su lugar, aconseja planificar con Cicerone para una experiencia personalizada. Para cualquier duda, "
                      "proporciona el contacto: info@iurban.es.")

    # Construir el prompt base
    prompt_base = f"{contexto_adicional}\n\nPregunta reciente: {pregunta}\nRespuesta original: {respuesta}\n--\n {new_prompt}"
    app.logger.info(prompt_base)

    # Generar la respuesta mejorada
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
        app.logger.info("Respuesta generada con éxito.")
        return improved_response
    except Exception as e:
        app.logger.error(f"Error al interactuar con OpenAI: {e}")
        return None


def generar_contexto_con_openai(historial):
    openai.api_key = os.environ.get('OPENAI_API_KEY')

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Resumen del historial de conversación."},
                {"role": "user", "content": historial}
            ]
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        app.logger.info(f"Error al generar contexto con OpenAI: {e}")
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


def buscar_en_respuestas_preestablecidas_nlp(pregunta_usuario, chatbot_id, umbral_similitud=0.7):
    app.logger.info("Iniciando búsqueda en respuestas preestablecidas con NLP")

    modelo = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Un modelo preentrenado
    json_file_path = f'data/uploads/pre_established_answers/{chatbot_id}/pre_established_answers.json'

    if not os.path.exists(json_file_path):
        app.logger.warning(f"Archivo JSON no encontrado en la ruta: {json_file_path}")
        return None, False

    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        preguntas_respuestas = json.load(json_file)

    # Crear una lista de todas las palabras clave
    palabras_clave = [entry["palabras_clave"] for entry in preguntas_respuestas.values()]
    palabras_clave_flat = [' '.join(palabras) for palabras in palabras_clave]

    # Calcular los embeddings para las palabras clave y la pregunta del usuario
    embeddings_palabras_clave = modelo.encode(palabras_clave_flat, convert_to_tensor=True)
    embedding_pregunta_usuario = modelo.encode(pregunta_usuario, convert_to_tensor=True)

    # Calcular la similitud semántica
    similitudes = util.pytorch_cos_sim(embedding_pregunta_usuario, embeddings_palabras_clave)[0]

    # Encontrar la mejor coincidencia si supera el umbral
    mejor_coincidencia = similitudes.argmax()
    max_similitud = similitudes[mejor_coincidencia].item()

    if max_similitud >= umbral_similitud:
        respuesta_mejor_coincidencia = list(preguntas_respuestas.values())[mejor_coincidencia]["respuesta"]
        app.logger.info(f"Respuesta encontrada con una similitud de {max_similitud}")
        return respuesta_mejor_coincidencia, True
    else:
        app.logger.info("No se encontró una coincidencia adecuada")
        return None, False

####### FIN Utils busqueda en Json #######


####### Inicio Endpoints #######


@app.route('/ask', methods=['POST'])
def ask():
    app.logger.info("Solicitud recibida en /ask")

    try:
        data = request.get_json()
        chatbot_id = data.get('chatbot_id')
        app.logger.info(f"Datos recibidos: {data}")

        if 'pares_pregunta_respuesta' in data:
            pares_pregunta_respuesta = data['pares_pregunta_respuesta']
            contexto = ""

            for par in pares_pregunta_respuesta[:-1]:
                contexto += f"Pregunta: {par['pregunta']} Respuesta: {par['respuesta']} "

            ultima_pregunta = pares_pregunta_respuesta[-1]['pregunta']
            ultima_respuesta = pares_pregunta_respuesta[-1]['respuesta']

            if ultima_respuesta == "":
                respuesta_preestablecida, encontrada_en_json = buscar_en_respuestas_preestablecidas_nlp(ultima_pregunta, chatbot_id)
                if encontrada_en_json:
                    contexto_adicional = generar_contexto_con_openai(contexto)
                    ultima_respuesta = respuesta_preestablecida + " " + contexto_adicional
                    fuente_respuesta = "preestablecida"
                else:
                    contexto_generado = generar_contexto_con_openai(contexto)
                    ultima_respuesta = mejorar_respuesta_generales_con_openai(ultima_pregunta, contexto_generado, chatbot_id=chatbot_id)
                    fuente_respuesta = "mejorada"

                if ultima_respuesta:
                    app.logger.info("Respuesta generada con éxito")
                    return jsonify({'respuesta': ultima_respuesta, 'fuente': fuente_respuesta})
                else:
                    app.logger.info("No se encontró una respuesta adecuada")
                    return jsonify({'respuesta': 'No se encontró una respuesta adecuada.', 'fuente': 'ninguna'})
            else:
                return jsonify({'respuesta': ultima_respuesta, 'fuente': 'existente'})

        else:
            app.logger.warning("Formato de solicitud incorrecto")
            return jsonify({'error': 'Formato de solicitud incorrecto'}), 400

    except Exception as e:
        app.logger.error(f"Error en /ask: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500




@app.route('/uploads', methods=['POST'])
def upload_file():
    try:
        logging.info("Procesando solicitud de carga de archivo")

        if 'documento' not in request.files:
            app.logger.warning("Archivo 'documento' no encontrado en la solicitud")
            return jsonify({"respuesta": "No se encontró el archivo 'documento'", "codigo_error": 1})
        
        uploaded_file = request.files['documento']
        chatbot_id = request.form.get('chatbot_id')
        app.logger.info(f"Archivo recibido: {uploaded_file.filename}, Chatbot ID: {chatbot_id}")

        if uploaded_file.filename == '':
            app.logger.warning("Nombre de archivo vacío")
            return jsonify({"respuesta": "No se seleccionó ningún archivo", "codigo_error": 1})

        docs_folder = os.path.join(BASE_DIR_DOCS, str(chatbot_id))
        os.makedirs(docs_folder, exist_ok=True)
        app.logger.info(f"Carpeta del chatbot creada o ya existente: {docs_folder}")

        file_extension = os.path.splitext(uploaded_file.filename)[1][1:].lower()
        file_path = os.path.join(docs_folder, uploaded_file.filename)
        uploaded_file.save(file_path)
        app.logger.info(f"Archivo guardado en: {file_path}")

        readable_content = process_file(file_path, file_extension)
        if readable_content is None:
            app.logger.error("No se pudo procesar el archivo")
            return jsonify({"respuesta": "Error al procesar el archivo", "codigo_error": 1})

        # Contar palabras en el contenido
        word_count = len(readable_content.split())

        dataset_file_path = os.path.join(BASE_DATASET_DIR, f"{chatbot_id}", "dataset.json")
        os.makedirs(os.path.dirname(dataset_file_path), exist_ok=True)

        dataset_entries = {}
        if os.path.exists(dataset_file_path):
            with open(dataset_file_path, 'r', encoding='utf-8') as json_file:
                dataset_entries = json.load(json_file)
                app.logger.info("Archivo JSON del dataset existente cargado")

        indice = uploaded_file.filename
        dataset_entries[indice] = {
            "indice": indice,
            "url": file_path,
            "dialogue": readable_content
        }

        with open(dataset_file_path, 'w', encoding='utf-8') as json_file_to_write:
            json.dump(dataset_entries, json_file_to_write, ensure_ascii=False, indent=4)
            app.logger.info("Archivo JSON del dataset actualizado y guardado")

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
        app.logger.info("Procesando solicitud para guardar texto")

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
        app.logger.error(f"Error durante el procesamiento. Error: {e}")
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


def url_for_scraping():
    try:
        app.logger.info("Procesando solicitud de scraping de URL")

        data = request.get_json()
        base_url = data.get('url')
        chatbot_id = data.get('chatbot_id')

        if not base_url:
            app.logger.warning("No se proporcionó URL")
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
            app.logger.error("Error al obtener respuesta del URL base")
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

        app.logger.info(f"Scraping completado para {base_url}")
        return jsonify(urls_data)
    except Exception as e:
        app.logger.error(f"Error inesperado en url_for_scraping: {e}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/url_for_scraping_by_sitemap', methods=['POST'])
def url_for_scraping_by_sitemap():
    try:
        app.logger.info("Procesando solicitud de scraping por sitemap")

        data = request.get_json()
        sitemap_url = data.get('url')
        chatbot_id = data.get('chatbot_id')

        if not sitemap_url:
            app.logger.warning("No se proporcionó URL del sitemap")
            return jsonify({'error': 'No se proporcionó URL del sitemap'}), 400

        save_dir = os.path.join('data/uploads/scraping', f'{chatbot_id}')
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'{chatbot_id}.txt')

        def request_sitemap(url):
            try:
                response = requests.get(url)
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                app.logger.error(f"Error al descargar el sitemap: {e}")
                return None

        sitemap_content = request_sitemap(sitemap_url)
        if not sitemap_content:
            return jsonify({'error': 'Error al descargar el sitemap'}), 500

        soup = BeautifulSoup(sitemap_content, 'xml')
        urls = [loc.text for loc in soup.find_all('loc')]

        with open(file_path, 'w') as file:
            for url in urls:
                file.write(url + '\n')

        app.logger.info(f"Sitemap procesado correctamente para {sitemap_url}")
        return jsonify({'message': 'Sitemap procesado correctamente', 'urls_count': len(urls)})
    except Exception as e:
        app.logger.error(f"Error inesperado en url_for_scraping_by_sitemap: {e}")
        return jsonify({'error': f'Error inesperado: {str(e)}'}), 500



@app.route('/delete_urls', methods=['POST'])
def delete_urls():
    data = request.json
    urls_to_delete = set(data.get('urls', []))  # Conjunto de URLs a eliminar
    chatbot_id = data.get('chatbot_id')  # Identificador del chatbot

    if not urls_to_delete or not chatbot_id:
        app.logger.warning("Faltan 'urls' o 'chatbot_id' en la solicitud")
        return jsonify({"error": "Missing 'urls' or 'chatbot_id'"}), 400

    chatbot_folder = os.path.join('data/uploads/scraping', str(chatbot_id))

    app.logger.info(f"Ruta del directorio del chatbot: {chatbot_folder}")

    if not os.path.exists(chatbot_folder):
        app.logger.error("Carpeta del chatbot no encontrada")
        return jsonify({"status": "error", "message": "Chatbot folder not found"}), 404

    file_name = f"{chatbot_id}.txt"
    file_path = os.path.join(chatbot_folder, file_name)

    app.logger.info(f"Ruta del archivo de URLs: {file_path}")

    if not os.path.exists(file_path):
        app.logger.error(f"Archivo {file_name} no encontrado en la carpeta del chatbot")
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

        app.logger.info("URLs eliminadas con éxito")
        return jsonify({"status": "success", "message": "URLs deleted successfully"})
    except Exception as e:
        app.logger.error(f"Error al eliminar URLs: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500



@app.route('/pre_established_answers', methods=['POST'])
def pre_established_answers():
    data = request.json
    chatbot_id = data.get('chatbot_id')
    pregunta = data.get('pregunta')
    respuesta = data.get('respuesta')

    if not (chatbot_id and pregunta and respuesta):
        app.logger.warning("Faltan datos en la solicitud (chatbot_id, pregunta, respuesta).")
        return jsonify({"error": "Faltan datos en la solicitud (chatbot_id, pregunta, respuesta)."}), 400

    palabras_clave = extraer_palabras_clave(pregunta)
    json_file_path = f'data/uploads/pre_established_answers/{chatbot_id}/pre_established_answers.json'
    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)

    if os.path.isfile(json_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            preguntas_respuestas = json.load(json_file)
    else:
        preguntas_respuestas = {}

    preguntas_respuestas[pregunta] = {
        "Pregunta": [pregunta],
        "palabras_clave": palabras_clave,
        "respuesta": respuesta
    }

    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(preguntas_respuestas, json_file, ensure_ascii=False, indent=4)

    app.logger.info(f"Respuesta para la pregunta '{pregunta}' guardada con éxito.")
    return jsonify({'mensaje': 'Pregunta y respuesta guardadas correctamente'})

@app.route('/delete_pre_established_answers', methods=['POST'])
def delete_pre_established_answers():
    data = request.json
    chatbot_id = data.get('chatbot_id')
    preguntas_a_eliminar = data.get('preguntas')

    if not chatbot_id or not preguntas_a_eliminar:
        app.logger.warning("Faltan datos en la solicitud (chatbot_id, preguntas).")
        return jsonify({"error": "Faltan datos en la solicitud (chatbot_id, preguntas)."}), 400

    json_file_path = f'data/uploads/pre_established_answers/{chatbot_id}/pre_established_answers.json'

    if not os.path.isfile(json_file_path):
        app.logger.error("Archivo de preguntas y respuestas no encontrado.")
        return jsonify({"error": "No se encontró el archivo de preguntas y respuestas."}), 404

    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        preguntas_respuestas = json.load(json_file)

    preguntas_eliminadas = []
    preguntas_no_encontradas = []

    for pregunta in preguntas_a_eliminar:
        if pregunta in preguntas_respuestas:
            del preguntas_respuestas[pregunta]
            preguntas_eliminadas.append(pregunta)
        else:
            preguntas_no_encontradas.append(pregunta)

    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(preguntas_respuestas, json_file, ensure_ascii=False, indent=4)

    app.logger.info("Proceso de eliminación de preguntas completado.")
    return jsonify({
        'mensaje': 'Proceso de eliminación completado',
        'preguntas_eliminadas': preguntas_eliminadas,
        'preguntas_no_encontradas': preguntas_no_encontradas
    })

@app.route('/change_params_prompt_temperature_and_model', methods=['POST'])
def change_params():
    data = request.json
    new_prompt = data.get('new_prompt')
    chatbot_id = data.get('chatbot_id')
    temperature = data.get('temperature', '')
    model_gpt = data.get('model_gpt', 'gpt-3.5-turbo')

    if not new_prompt or not chatbot_id:
        app.logger.warning("Los campos 'new_prompt' y 'chatbot_id' son requeridos.")
        return jsonify({"error": "Los campos 'new_prompt' y 'chatbot_id' son requeridos"}), 400

    prompt_file_path = f"data/uploads/prompts/{chatbot_id}/prompt.txt"
    os.makedirs(os.path.dirname(prompt_file_path), exist_ok=True)

    with open(prompt_file_path, 'w') as file:
        file.write(new_prompt)

    with open(prompt_file_path, 'r') as file:
        saved_prompt = file.read()

    pregunta = ""
    respuesta = ""
    prompt_base = f"Cuando recibas una pregunta, comienza con: '{pregunta}'. Luego sigue con tu respuesta original: '{respuesta}'. {saved_prompt}"

    if new_prompt == prompt_base:
        app.logger.info("El nuevo prompt es igual al prompt base. No se realizó ninguna acción.")
        return jsonify({"mensaje": "El nuevo prompt es igual al prompt base. No se realizó ninguna acción."})

    mejorar_respuesta_generales_con_openai(saved_prompt, temperature, model_gpt)

    app.logger.info("Parámetros del chatbot cambiados con éxito.")
    return jsonify({"mensaje": "Parámetros cambiados con éxito"})



@app.route('/list_chatbot_ids', methods=['GET'])
def list_folders():
    directory = 'data/uploads/scraping/'
    folders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    return jsonify(folders)



@app.route('/run_tests', methods=['POST'])
def run_tests():
    import subprocess
    result = subprocess.run(['python', 'run_tests.py'], capture_output=True, text=True)
    return result.stdout
 ######## Fin Endpoints ######## 

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)