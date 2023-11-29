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


nltk.download('punkt')
nltk.download('stopwords')

tqdm.pandas()

app = Flask(__name__)


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

model_name="google/flan-t5-base"


MAX_TOKENS_PER_SEGMENT = 7000  # Establecer un límite seguro de tokens por segmento
BASE_DATASET_DIR = "data/uploads/datasets/"
BASE_DIR_SCRAPING = "data/uploads/scraping/"
UPLOAD_FOLDER = 'data/uploads/'  # Ajusta esta ruta según sea necesario
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'csv', 'docx', 'xlsx', 'pptx'}
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
                {"role": "system", "content": "Estás aquí para ayudar a mejorar las respuestas a las preguntas."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        print(f"Error al interactuar con OpenAI: {e}")
        return None

def mejorar_respuesta_generales_con_openai(respuesta_original):
    openai.api_key = os.environ.get('OPENAI_API_KEY')

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """Responde como si fueras una guía de una oficina de turismo. Siempre responde en el mismo idioma de la pregunta, y di las cosas en forma de listado para que se vea más claro que hacer por días.SIEMPRE contesta sobre el mismo idioma que te están realizando la pregunta.Si te piden cualquier cosa de programación, recomendaciones turísticas o algo que no esté relacionado con Sevilla diles lo siento que solo puedes responder con información de la página webSolo puedes dar recomendaciones turísticas de Sevilla si te preguntan por otra ciudad diles que acceden a https://ciceroneai.com/ para ver qué ciudades están disponiblesCuando respondas a una pregunta al final de la respuesta quiero que le digas que "si quieres más información hazte tu ruta con Cicerone o mira las rutas de los expertos locales para guiarte"Solo responde a preguntas cuyo contenido es de sevilla No respondas a nada que no esté relacionado con turismo y planes de la ciudad de Sevilla, si te preguntan por otra ciudad  diles "Si quieres más información de otras ciudades mira en nuestra web las ciudades disponibles"Siempre tienes que responder en el mismo idioma que la pregunta del usuario, si pregunta en inglés responde en inglés, si preguntan en valenciano responde en valenciano, si preguntan en castellano respondes en castellano.Si no tienes la información, nunca empieces la respuesta con la frase "Lo siento, no puedo darte información específica", y di que mejor hagan su ruta con Cicerone para vivir una experiencia personalizadaSi tienen cualquier duda déjales el contacto para resolver dudas info@iurban.es"""},
                {"role": "user", "content": respuesta_original}
            ]
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        print(f"Error al interactuar con OpenAI: {e}")
        return None

def convertir_a_texto(dato):
    if isinstance(dato, dict):
        # Suponiendo que 'texto' es la clave deseada
        return dato.get('texto', '')
    elif isinstance(dato, list):
        # Concatenar elementos de la lista
        return ' '.join(map(str, dato))
    elif isinstance(dato, str):
        return dato
    else:
        # Convertir cualquier otro tipo de dato a string
        return str(dato)



    return None  # o puedes devolver un valor predeterminado o lanzar una excepción

# Recuerda llamar a la función con los parámetros adecuados


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


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def encontrar_respuesta(pregunta, datos):
    try:
        # Obtener stopwords una sola vez
        spanish_stopwords = stopwords.words('spanish')
        app.logger.info("Stopwords cargadas correctamente.")

        # Tokenizar y limpiar la pregunta
        palabras_clave_pregunta = set([palabra for palabra in word_tokenize(pregunta.lower()) if palabra not in spanish_stopwords])
        app.logger.info("Pregunta tokenizada y limpiada.")

        respuestas = []
        for item in datos.values():
            # Convertir el item a texto y luego tokenizar y limpiar
            texto_tokenizado = [palabra for palabra in word_tokenize(convertir_a_texto(item).lower()) if palabra not in spanish_stopwords]
            for idx, palabra in enumerate(texto_tokenizado):
                if palabra in palabras_clave_pregunta:
                    # Calcular los índices de inicio y fin para extraer el fragmento
                    inicio = max(idx - 5, 0)
                    fin = min(idx + 6, len(texto_tokenizado))
                    fragmento = ' '.join(texto_tokenizado[inicio:fin])
                    respuestas.append(fragmento)

        if respuestas:
            app.logger.info("Coincidencias encontradas.")
            return respuestas
        else:
            app.logger.info("No se encontró ninguna coincidencia.")
            return "No se encontró ninguna coincidencia."

    except Exception as e:
        app.logger.error(f"Error en encontrar_respuesta: {e}")
        raise e



def cargar_dataset(chatbot_id, base_dataset_dir):
    dataset_file_path = os.path.join(BASE_DATASET_DIR, str(chatbot_id), 'dataset.json')
    app.logger.info(f"Dataset cin ruta {dataset_file_path}")

    try:
        with open(dataset_file_path, 'r') as file:
            data = json.load(file)
            app.logger.info(f"Dataset cargado con éxito desde {dataset_file_path}")
            return data
    except FileNotFoundError:
        app.logger.error(f"Archivo no encontrado: {dataset_file_path}")
    except json.JSONDecodeError:
        app.logger.error(f"Error al decodificar JSON en el archivo: {dataset_file_path}")
    except Exception as e:
        app.logger.error(f"Error al cargar el dataset: {e}")

@app.route('/ask_general', methods=['POST'])
def ask_general():
    contenido = request.json
    pregunta = contenido['pregunta']
    chatbot_id = contenido['chatbot_id']
    token = contenido.get('token', None)  # Añadido el token como un parámetro opcional
   

    # Aquí podrías validar o utilizar el token, si es necesario

    datos = cargar_dataset(chatbot_id, BASE_DATASET_DIR)
    respuesta_original = encontrar_respuesta(pregunta, datos)

    # Intentar mejorar la respuesta con OpenAI
    try:
        respuesta_mejorada = mejorar_respuesta_con_openai(respuesta_original)
        if respuesta_mejorada:
            return jsonify({'respuesta': respuesta_mejorada})
    except Exception as e:
        print(f"Error al mejorar respuesta con OpenAI: {e}")
        # Devolver la respuesta original en caso de error
        return jsonify({'respuesta': respuesta_original})



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

        time.sleep(0.2)

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

        if max_similarity <= 0.5:
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

 ######## Fin Endpoints ########

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)