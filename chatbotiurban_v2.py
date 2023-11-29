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
from requests.exceptions import RequestException
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
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from requests.exceptions import RequestException
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
from transformers import AutoTokenizer
from datasets import Dataset
import subprocess


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



def build_dataset(model_name, chatbot_id, input_min_text_length, input_max_text_length):
    """
    Preprocess the dataset and split it into train and test parts.

    Parameters:
    - model_name (str): Tokenizer model name.
    - chatbot_id (str): ID of the chatbot to identify the dataset.
    - input_min_text_length (int): Minimum length of the dialogues.
    - input_max_text_length (int): Maximum length of the dialogues.
        
    Returns:
    - dataset_splits (datasets.dataset_dict.DatasetDict): Preprocessed dataset containing train and test parts.
    """

    # Path to the JSON file created by process_urls
    dataset_file_path = os.path.join(BASE_DATASET_DIR, str(chatbot_id), 'dataset.json')
    
    # Load dataset from JSON file
    df = pd.read_json(dataset_file_path)
    dataset = Dataset.from_pandas(df)

    # Filter dialogues based on length
    dataset = dataset.filter(lambda x: len(x["dialogue"]) > input_min_text_length and len(x["dialogue"]) <= input_max_text_length, batched=False)

    # Preparación del tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")

    # Función para tokenizar el texto
    def tokenize_function(examples):
        return tokenizer(examples["dialogue"], padding="max_length", truncation=True)

    # Aplicar la tokenización a todo el dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Dividir el dataset en partes de entrenamiento y prueba
    dataset_splits = tokenized_dataset.train_test_split(test_size=0.2)

    return dataset_splits


def generar_prompt(dataset_path, pregunta):
    # Construir la ruta completa al archivo dataset.json
    json_file_path = os.path.join(dataset_path, 'dataset.json')

    # Construir el prompt con la pregunta al principio
    prompt = f"pregunta: {pregunta} "

    # Cargar el contenido del archivo JSON
    with open(json_file_path, 'r') as file:
        dataset = json.load(file)

    # Extraer y añadir el texto del dataset al prompt
    for entry in dataset.values():
        text = entry.get("dialogue", "")  # Asumiendo que el texto está en la clave 'dialogue'
        prompt += text + " "

    return prompt.strip()

def clean_and_transform_data(data):
    # Aquí puedes implementar tu lógica de limpieza y transformación
    cleaned_data = data.strip().replace("\r", "").replace("\n", " ")
    return cleaned_data

 ######## Inicio Endpoints ########

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


@app.route('/ask_pruebas', methods=['POST'])
def ask_pruebas():
    try:
        data = request.json
        chatbot_id = data.get('chatbot_id')
        token = data.get('token')

        if not chatbot_id or not token:
            app.logger.error("No chatbot_id or token provided in the request")
            return jsonify({"error": "No chatbot_id or token provided"}), 400

        # Cargar el dataset
        dataset_file_path = os.path.join(BASE_DATASET_DIR, str(chatbot_id), 'dataset.json')
        with open(dataset_file_path, 'r') as file:
            dataset = json.load(file)

        pregunta = data.get('pregunta')
        if not pregunta:
            app.logger.error("No pregunta provided in the request")
            return jsonify({"error": "No pregunta provided"}), 400

        dataset_folder = os.path.join('data', 'uploads', 'datasets', chatbot_id)

        # Obtener la respuesta de OpenAI
        #client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        #api_key="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",
        #)

        #chat_completion = client.chat.completions.create(
        #    messages=[
        #        {
        #            "role": "user",
        #            "prompt":generar_prompt(dataset_folder, pregunta)
        #         }
        #    ],
        #    model="gpt-4",
        #)


        openai.api_key = os.environ.get('OPENAI_API_KEY')
        response_openai = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "user", "prompt": generar_prompt(dataset_folder, pregunta)}])


        # Devolver solo el texto de la respuesta
        return response.choices[0].text.strip()

    except Exception as e:
        app.logger.error(f"Unexpected error in ask function: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/pre_established_answers', methods=['POST'])
def pre_established_answers():
    # Obtener los datos del cuerpo de la solicitud POST
    data = request.json
    chatbot_id = data.get('chatbot_id')
    pregunta = data.get('pregunta')
    respuesta = data.get('respuesta')

    # Procesar la pregunta y generar una respuesta
    # Aquí puedes integrar un modelo de IA o cualquier lógica específica
    # Para este ejemplo, voy a simular una respuesta
   
    # Devolver la respuesta procesada
    return jsonify({'respuesta': respuesta})


@app.route('/list_chatbot_ids', methods=['GET'])
def list_folders():
    directory = 'data/uploads/scraping/'
    folders = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    return jsonify(folders)

 ######## Fin Endpoints ########

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)