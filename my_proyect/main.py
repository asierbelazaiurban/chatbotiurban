#!/usr/bin/env python
# coding: utf-8

# Flask y herramientas relacionadas para la aplicación web
from flask import Flask, request, jsonify

# Manejo de archivos y rutas
import os
import json

# Logging para registrar eventos y errores
import logging
from logging import FileHandler
from logging.handlers import RotatingFileHandler

# Procesamiento de texto y NLP
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer

# Solicitudes web y scraping
import requests
from bs4 import BeautifulSoup

# Manejo de datos y cálculos
import numpy as np

# Interacción con OpenAI GPT
import openai

# Herramientas adicionales para tareas específicas
import subprocess
import random
import re
import time
import traceback
from urllib.parse import urlparse, urljoin

# Configuración de Flask y Logging
app = Flask(__name__)

# Utilidades y Otras Librerías
import chardet
import evaluate
import unidecode
from peft import PeftConfig, PeftModel, TaskType, LoraConfig
from trl import PPOConfig, PPOTrainer, AutoModelForSeq2SeqLMWithValueHead, create_reference_model
from trl.core import LengthSampler
import gensim.downloader as api

# ---------------------------
# Módulos Locales
# ---------------------------
from utils.clean_data_for_scraping import *
from utils.date_management import *
from utils.process_docs import process_file
from utils.utils_and_nlp_for_preestablecidas import *
from utils.search_in_dataset import *
from utils.all_relative_to_openai import *
 
# ---------------------------
# Configuración Adicional
# ---------------------------

modelo = api.load("glove-wiki-gigaword-50")


nltk.download('stopwords')
nltk.download('punkt')
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




####### Inicio Endpoints #######

@app.route('/ask', methods=['POST'])
def ask():
    app.logger.info("Solicitud recibida en /ask")

    try:
        data = request.get_json()
        chatbot_id = data.get('chatbot_id', 'default_id')
        fuente_respuesta = "ninguna"

        if 'pares_pregunta_respuesta' in data:
            pares_pregunta_respuesta = data['pares_pregunta_respuesta']
            ultima_pregunta = pares_pregunta_respuesta[-1]['pregunta']
            ultima_respuesta = pares_pregunta_respuesta[-1]['respuesta']
            contexto = ' '.join([f"Pregunta: {par['pregunta']} Respuesta: {par['respuesta']}" for par in pares_pregunta_respuesta[:-1]])

            if ultima_respuesta == "":
                respuesta_preestablecida, encontrada_en_json = buscar_en_respuestas_preestablecidas_nlp(ultima_pregunta, chatbot_id)

                if encontrada_en_json:
                    ultima_respuesta = respuesta_preestablecida
                    fuente_respuesta = "preestablecida"
                elif buscar_en_openai_relacion_con_eventos(ultima_pregunta):
                    ultima_respuesta = obtener_eventos(ultima_pregunta, chatbot_id)
                    fuente_respuesta = "eventos"
                else:
                    app.logger.info("Entrando en la sección del dataset")
                    dataset_file_path = os.path.join(BASE_DATASET_DIR, str(chatbot_id), 'dataset.json')
                    if os.path.exists(dataset_file_path):
                        with open(dataset_file_path, 'r') as file:
                            datos_del_dataset = json.load(file)

                        # Crear y entrenar el vectorizer
                        vectorizer = TfidfVectorizer()
                        prepared_data = [convertir_a_texto(item['dialogue']) for item in datos_del_dataset.values()]
                        vectorizer.fit(prepared_data)

                        # Llamar a encontrar_respuesta con el vectorizer
                        respuesta_del_dataset = encontrar_respuesta(ultima_pregunta, datos_del_dataset, vectorizer, contexto)
                        app.logger.info(respuesta_del_dataset)

                        if respuesta_del_dataset:
                            ultima_respuesta = respuesta_del_dataset
                            fuente_respuesta = "dataset"
                        else:
                            ultima_respuesta = seleccionar_respuesta_por_defecto()
                            fuente_respuesta = "respuesta_por_defecto"


                # Mejora de la respuesta con OpenAI
                ultima_respuesta_mejorada = mejorar_respuesta_generales_con_openai(
                    pregunta=ultima_pregunta, 
                    respuesta=ultima_respuesta, 
                    new_prompt="", 
                    contexto_adicional=contexto, 
                    temperature="", 
                    model_gpt="", 
                    chatbot_id=chatbot_id
                )
                ultima_respuesta = ultima_respuesta_mejorada if ultima_respuesta_mejorada else ultima_respuesta
                fuente_respuesta = "mejorada"

                return jsonify({'respuesta': ultima_respuesta, 'fuente': fuente_respuesta})

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
            
            # Limpiar y formatear el texto
            text = clean_and_format_text(soup)

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
        # Guardar el dataset
        save_dataset(dataset_entries, chatbot_id)

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
    model_gpt = data.get('model_gpt', 'gpt-4')

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

    mejorar_respuesta_generales_con_openai(
        pregunta=pregunta,
        respuesta=respuesta,
        new_prompt=saved_prompt,
        contexto_adicional="",
        temperature=temperature,
        model_gpt=model_gpt,
        chatbot_id=chatbot_id
    )

    app.logger.info("Parámetros del chatbot cambiados con éxito.")
    return jsonify({"mensaje": "Parámetros cambiados con éxito"})

@app.route('/events', methods=['POST'])
def events():
    data = request.json
    chatbot_id = data.get('chatbot_id')
    eventos = data.get('events')  # 'events' es un array
    pregunta = data.get('pregunta')

    if not (chatbot_id and eventos):
        app.logger.warning("Faltan datos en la solicitud (chatbot_id, events).")
        return jsonify({"error": "Faltan datos en la solicitud (chatbot_id, events)."}), 400

    # Concatenar los eventos en una sola cadena
    eventos_concatenados = " ".join(eventos)

    # Llamar a mejorar_respuesta_con_openai con los eventos concatenados
    respuesta_mejorada = mejorar_respuesta_con_openai(eventos_concatenados, pregunta, chatbot_id)
    if respuesta_mejorada:
        app.logger.info("Respuesta mejorada generada con éxito.")
        return jsonify({'mensaje': 'Respuesta mejorada', 'respuesta_mejorada': respuesta_mejorada})
    else:
        app.logger.error("Error al mejorar la respuesta para el evento concatenado.")
        return jsonify({"error": "Error al mejorar la respuesta"}), 500


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