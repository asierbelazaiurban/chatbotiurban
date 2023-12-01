##!/usr/bin/env python
# coding: utf-8

# Bibliotecas estándar de Python
import json
import logging
import os
import random
import re
import subprocess
import time
from logging import FileHandler
from logging.handlers import RotatingFileHandler
from urllib.parse import urlparse, urljoin

# Bibliotecas de terceros
import chardet
import gensim.downloader as api
import nltk
import numpy as np
import openai
import pandas as pd
import requests
import torch
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from peft import PeftModel, PeftConfig, LoraConfig, TaskType
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, GenerationConfig
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead, create_reference_model
from trl.core import LengthSampler
from werkzeug.datastructures import FileStorage

# Importaciones locales
from process_docs import process_file
from useful_methods import *



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
BASE_DIR_SCRAPING = "data/uploads/scraping/"
BASE_DIR_DOCS = "data/uploads/docs/"
UPLOAD_FOLDER = 'data/uploads/'  # Ajusta esta ruta según sea necesario
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

####### Endpoints #######

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

def encontrar_respuesta(pregunta, datos, longitud_minima=200):
    try:
        # Preprocesar la pregunta
        pregunta_procesada = preprocess_query(pregunta)

        # Codificar los datos
        encoded_data, vectorizer = encode_data(datos)

        # Codificar la pregunta
        encoded_query = vectorizer.transform([pregunta_procesada])

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

# Uso de la función
# respuesta_ampliada = encontrar_respuesta_amplia(la_pregunta, los_datos)


####### FIN Utils busqueda en Json #######


####### Inicio Endpoints #######

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
        respuesta_mejorada = mejorar_respuesta_generales_con_openai(respuesta_original, pregunta)
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
            "dialogue": readable_content,
            "word_count": word_count  # Agregar recuento de palabras
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


@app.route('/list_chatbot_ids', methods=['GET'])
def list_folders():
    directory = 'data/uploads/scraping/'
    folders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    return jsonify(folders)

 ######## Fin Endpoints ######## 

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)