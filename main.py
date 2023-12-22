#!/usr/bin/env python
# coding: utf-8

# ---------------------------
# Librerías Estándar de Python
# ---------------------------
# Utilidades básicas y manejo de archivos
import json
import logging
import os
import random
import re
import subprocess
import time
import traceback
from logging import FileHandler
from logging.handlers import RotatingFileHandler
from time import sleep
from urllib.parse import urlparse, urljoin
from openai import ChatCompletion
import sys

# ---------------------------
# Librerías de Terceros
# ---------------------------
# Procesamiento de Lenguaje Natural y Aprendizaje Automático
import nltk
import numpy as np
import torch
from nltk.corpus import stopwords
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer, GenerationConfig, pipeline)
from sentence_transformers import SentenceTransformer, util
import gensim.downloader as api
from deep_translator import GoogleTranslator
from elasticsearch import Elasticsearch, helpers
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTNeoForCausalLM, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Descarga de paquetes necesarios de NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Procesamiento de Datos y Modelos
import pandas as pd
from tqdm import tqdm  # Importación única de tqdm
from gensim.models import Word2Vec
from datasets import Dataset, load_dataset
import pdfplumber

# Web Scraping y Solicitudes HTTP
import requests
from bs4 import BeautifulSoup

# Marco de Trabajo Web
from flask import Flask, request, jsonify, current_app as app
from werkzeug.datastructures import FileStorage

# Utilidades y Otras Librerías
import chardet
import evaluate
import unidecode
from peft import PeftConfig, PeftModel, TaskType, LoraConfig
from trl import PPOConfig, PPOTrainer, AutoModelForSeq2SeqLMWithValueHead, create_reference_model
from trl.core import LengthSampler

# ---------------------------
# Módulos Locales
# ---------------------------
from clean_data_for_scraping import *
from date_management import *
from process_docs import process_file

# ---------------------------
# Configuración Adicional
# ---------------------------

modelo = api.load("glove-wiki-gigaword-50")

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY


nltk.download('stopwords')
nltk.download('punkt')
app = Flask(__name__)

#### Logger ####

# Crear el directorio de logs si no existe
if not os.path.exists('logs'):
    os.mkdir('logs')

# Configurar el manejador de logs para escribir en un archivo
log_file_path = 'logs/chatbotiurban.log'
file_handler = RotatingFileHandler(log_file_path, maxBytes=10240000, backupCount=1)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.DEBUG)  # Usa DEBUG o INFO según necesites

# Añadir el manejador de archivos al logger de la aplicación
app.logger.addHandler(file_handler)

# También añadir un manejador de consola para la salida estándar
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
console_handler.setLevel(logging.DEBUG)  # Asegúrate de que este nivel sea consistente con file_handler.setLevel
app.logger.addHandler(console_handler)

# Establecer el nivel del logger de la aplicación
app.logger.setLevel(logging.DEBUG)

app.logger.info('Inicio de la aplicación ChatbotIUrban')

#### Logger ####


MAX_TOKENS_PER_SEGMENT = 7000  # Establecer un límite seguro de tokens por segmento
BASE_DATASET_DIR = "data/uploads/datasets/"
BASE_PDFS_DIR = "data/uploads/pdfs/"
BASE_PDFS_DIR_JSON = "data/uploads/pdfs/json/"
BASE_CACHE_DIR =  "data/uploads/cache/"
BASE_DATASET_PROMPTS = "data/uploads/prompts/"
BASE_DIR_SCRAPING = "data/uploads/scraping/"
BASE_DIR_DOCS = "data/uploads/docs/"
UPLOAD_FOLDER = 'data/uploads/'  # Ajusta esta ruta según sea necesario
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'csv', 'docx', 'xlsx', 'pptx'}

# Configuración global
"""INDICE_ELASTICSEARCH = 'busca-asier-iurban'

es_client = Elasticsearch(
    cloud_id="1432c4b2cc52479b9a94f9544db4db49:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJDdlZGM5YTRhZDg5MTQ4ZTViMWE1NjkwYjYxMDE0OWEzJDk1NmI2MTRjODAxMzQ3MWU5NDY0ZDAxMTdjMTJkNjc5",
    api_key="aFlBbmtJd0JYOHBWMWlhY05IRDk6dlhlWVpIck1UTWE4Ti03NjM5QjdGdw=="
)
# Crear el índice en Elasticsearch si no existe
if not es_client.indices.exists(index=INDICE_ELASTICSEARCH):
    es_client.indices.create(index=INDICE_ELASTICSEARCH)"""

ELASTIC_PASSWORD = "@1Urb4n2022"

# Found in the 'Manage Deployment' page
CLOUD_ID = "1432c4b2cc52479b9a94f9544db4db49:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJDdlZGM5YTRhZDg5MTQ4ZTViMWE1NjkwYjYxMDE0OWEzJDk1NmI2MTRjODAxMzQ3MWU5NDY0ZDAxMTdjMTJkNjc5"

# Create the client instance
client = Elasticsearch(
    cloud_id=CLOUD_ID,
    basic_auth=("elastic", ELASTIC_PASSWORD)
)

# Successful response!
client.info()



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


def mejorar_respuesta_con_openai(respuesta_original, pregunta, chatbot_id):
    openai.api_key = os.environ.get('OPENAI_API_KEY')

    # Definir las rutas base para los prompts
    BASE_PROMPTS_DIR = "data/uploads/prompts/"

    # Intentar cargar el prompt específico desde los prompts, según chatbot_id
    new_prompt_by_id = None
    if chatbot_id:
        prompt_file_path = os.path.join(BASE_PROMPTS_DIR, str(chatbot_id), 'prompt.txt')
        try:
            with open(prompt_file_path, 'r') as file:
                new_prompt_by_id = file.read()
            app.logger.info(f"Prompt cargado con éxito desde prompts para chatbot_id {chatbot_id}.")
        except Exception as e:
            app.logger.error(f"Error al cargar desde prompts para chatbot_id {chatbot_id}: {e}")

    # Utilizar el prompt específico si está disponible, de lo contrario usar un prompt predeterminado
    new_prompt = new_prompt_by_id if new_prompt_by_id else (
        "Mantén la coherencia con la pregunta y, si la respuesta no se alinea, indica 'No tengo información "
        "en este momento sobre este tema, ¿Puedes ampliarme la pregunta? o ¿Puedo ayudarte en algo mas?'. Actúa como un guía turístico experto, "
        "presentando tus respuestas en forma de listas para facilitar la planificación diaria de actividades. "
        "Es crucial responder en el mismo idioma que la pregunta. Al finalizar tu respuesta, recuerda sugerir "
        "'Si deseas más información, crea tu ruta con Cicerone o consulta las rutas de expertos locales'. "
        "Si careces de la información solicitada, evita comenzar con 'Lo siento, no puedo darte información específica'. "
        "En su lugar, aconseja planificar con Cicerone para una experiencia personalizada. Para cualquier duda, "
        "proporciona el contacto: info@iurban.es."
    )

    # Construir el prompt base
    prompt_base = f"Responde con menos de 75 palabras. Nunca respondas cosas que no tengan relacion entre Pregunta: {pregunta}\n y Respuesta: {respuesta_original}\n--\n{new_prompt}. Respondiendo siempre en el idioma del contexto"

    # Intentar generar la respuesta mejorada
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": prompt_base},
                {"role": "user", "content": respuesta_original}
            ]
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        app.logger.error(f"Error al interactuar con OpenAI: {e}")
        return None


def mejorar_respuesta_generales_con_openai(pregunta, respuesta, new_prompt="", contexto_adicional="", temperature="", model_gpt="", chatbot_id=""):
    # Verificar si hay pregunta y respuesta
    if not pregunta or not respuesta:
        app.logger.info("Pregunta o respuesta no proporcionada. No se puede procesar la mejora.")
        return None

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
        contexto_adicional = "";

    # Si no se ha proporcionado new_prompt, usar un prompt predeterminado
    if not new_prompt:
        new_prompt = ("Mantén la coherencia con la pregunta. Actúa como un guía turístico experto, "
                      "presentando tus respuestas en forma de listas para facilitar la planificación diaria de actividades. "
                      "Es crucial responder en el mismo idioma que la pregunta. Al finalizar tu respuesta, recuerda sugerir "
                      "'Si deseas más información, crea tu ruta con Cicerone o consulta las rutas de expertos locales'. "
                      "Si careces de la información solicitada, evita comenzar con 'Lo siento, no puedo darte información específica'. "
                      "En su lugar, aconseja planificar con Cicerone para una experiencia personalizada. Para cualquier duda, "
                      "proporciona el contacto: info@iurban.es.")

    # Construir el prompt base
    prompt_base = f"Si la pregunta es en ingles responde en ingles y asi con todo los idiomas, catalán, frances y todos los que tengas disponibles. Traduce litrealmete. Nunca respondas cosas que no tengan relacion entre Pregunta: {pregunta}\n y Respuesta: {respuesta}Si hay algun tema con la codificación o caracteres, por ejemplo (Lo siento, pero parece que hay un problema con la codificación de caracteres en tu pregunta o similar...)no te refieras  ni comentes el problema {contexto_adicional}\n\nPregunta reciente: {pregunta}\nRespuesta original: {respuesta}\n--\n {new_prompt}, siempre en el idioma del contexto, No respondas con mas de 75 palabras."
    app.logger.info(prompt_base)

    # Generar la respuesta mejorada
    try:
        response = openai.ChatCompletion.create(
            model=model_gpt if model_gpt else "gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": prompt_base},
                {"role": "user", "content": respuesta}
            ],
            temperature=float(temperature) if temperature else 0.5
        )
        improved_response = response.choices[0].message['content'].strip()
        respuesta_mejorada = improved_response
        app.logger.info("Respuesta generada con éxito.")
        app.logger.info("Respuesta mejorada a mostrar")
        return respuesta_mejorada
        
    except Exception as e:
        app.logger.error(f"Error al interactuar con OpenAI: {e}")
        return None


    """# Intentar traducir la respuesta mejorada
    app.logger.info("pregunta")
    app.logger.info(pregunta)
    app.logger.info("respuesta_mejorada")
    app.logger.info(respuesta_mejorada)
    try:
        respuesta_traducida = openai.ChatCompletion.create(
            model=model_gpt if model_gpt else "gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": f"El idioma original es el de la pregunta:  {pregunta}. Traduce, literalmente {respuesta_mejorada}, al idioma de la pregiunta. Asegurate de que sea una traducción literal.  Si no hubiera que traducirla por que la pregunta: {pregunta} y la respuesta::{respuesta_mejorada}, estan en el mismo idioma devuélvela tal cual, no le añadas ninguna observacion de ningun tipo ni mensaje de error. No agregues comentarios ni observaciones en ningun idioma. Solo la traducción literal o la frase repetida si es el mismo idioma"},                
                {"role": "user", "content": respuesta_mejorada}
            ],
            temperature=float(temperature) if temperature else 0.7
        )
        respuesta_mejorada = respuesta_traducida.choices[0].message['content'].strip()
    except Exception as e:
        app.logger.error(f"Error al traducir la respuesta: {e}")

    app.logger.info("respuesta_mejorada final")
    app.logger.info(respuesta_mejorada)
    return respuesta_mejorada
"""


def generar_contexto_con_openai(historial):
    openai.api_key = os.environ.get('OPENAI_API_KEY')

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Enviamos una conversación para que entiendas el contexto"},
                {"role": "user", "content": historial}
            ]
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        app.logger.info(f"Error al generar contexto con OpenAI: {e}")
        return ""


def identificar_saludo_despedida(frase):
    app.logger.info("Determinando si la frase es un saludo o despedida")

    # Establecer la clave de API de OpenAI
    openai.api_key = os.environ.get('OPENAI_API_KEY')

    # Respuestas de saludo
    respuestas_saludo = [
        "¡Hola! ¿Cómo puedo ayudarte hoy?",
        "Bienvenido, ¿en qué puedo asistirte?",
        "Buenos días, ¿cómo puedo hacer tu día más especial?",
        "Saludos, ¿hay algo en lo que pueda ayudarte?",
        "Hola, estoy aquí para responder tus preguntas.",
        "¡Bienvenido! Estoy aquí para ayudarte con tus consultas.",
        "Hola, ¿listo para comenzar?",
        "¡Qué alegría verte! ¿Cómo puedo ayudarte hoy?",
        "Buen día, ¿hay algo específico que necesitas?",
        "Hola, estoy aquí para responder todas tus preguntas."
    ]

    # Respuestas de despedida
    respuestas_despedida = [
        "Ha sido un placer ayudarte, ¡que tengas un buen día!",
        "Adiós, y espero haber sido útil para ti.",
        "Hasta pronto, que tengas una maravillosa experiencia.",
        "Gracias por contactarnos, ¡que tengas un día increíble!",
        "Despidiéndome, espero que disfrutes mucho tu día.",
        "Ha sido un gusto asistirte, ¡que todo te vaya bien!",
        "Adiós, no dudes en volver si necesitas más ayuda.",
        "Que tengas un gran día, ha sido un placer ayudarte.",
        "Espero que tengas un día inolvidable, adiós.",
        "Hasta la próxima, que tu día esté lleno de momentos maravillosos."
    ]

    try:
        # Enviar la frase directamente a OpenAI para determinar si es un saludo o despedida
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Identifica si la siguiente frase es exclusivamente un saludo o una despedida, sin información adicional o solicitudes. Responder únicamente con 'saludo', 'despedida' o 'ninguna':"},
                {"role": "user", "content": frase}
            ]
        )


        # Interpretar la respuesta
        respuesta = response.choices[0].message['content'].strip().lower()
        respuesta = unidecode.unidecode(respuesta)

        app.logger.info("respuesta GPT4")
        app.logger.info(respuesta)

        # Seleccionar una respuesta aleatoria si es un saludo o despedida
        if respuesta == "saludo":
            respuesta_elegida = random.choice(respuestas_saludo)
        elif respuesta == "despedida":
            respuesta_elegida = random.choice(respuestas_despedida)
        else:
            return False


        app.logger.info("respuesta_elegida")
        app.logger.info(respuesta_elegida)

        # Realizar una segunda llamada a OpenAI para traducir la respuesta seleccionada
        traduccion_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"El idioma original es {frase}. Traduce, literalmente {respuesta_elegida}, asegurate de que sea una traducción literal.  Si no hubiera que traducirla por que la: {frase} y :{respuesta_elegida}, estan en el mismo idioma devuélvela tal cual, no le añadas ninguna observacion de ningun tipo ni mensaje de error. No agregues comentarios ni observaciones en ningun idioma. Solo la traducción literal o la frase repetida si es el mismo idioma"},                
                {"role": "user", "content": respuesta_elegida}
            ]
        )


        respuesta_traducida = traduccion_response.choices[0].message['content'].strip()
        app.logger.info("respuesta_traducida")
        app.logger.info(respuesta_traducida)
        return respuesta_traducida

    except Exception as e:
        app.logger.error(f"Error al procesar la solicitud: {e}")
        return False


def extraer_palabras_clave(pregunta):
    # Tokenizar la pregunta
    palabras = word_tokenize(pregunta)

    # Filtrar las palabras de parada (stop words) y los signos de puntuación
    palabras_filtradas = [palabra for palabra in palabras if palabra.isalnum()]

    # Filtrar palabras comunes (stop words)
    stop_words = set(stopwords.words('spanish'))
    palabras_clave = [palabra for palabra in palabras_filtradas if palabra not in stop_words]

    return palabras_clave


####### Inicio Sistema de cache #######


# Suponiendo que ya tienes definida la función comprobar_coherencia_gpt

def encontrar_respuesta_en_cache(pregunta_usuario, chatbot_id):
    # URL y headers para la solicitud HTTP
    url = 'https://experimental.ciceroneweb.com/api/get-back-cache'
    headers = {'Content-Type': 'application/json'}
    payload = {'chatbot_id': chatbot_id}

    # Realizar solicitud HTTP con manejo de excepciones
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
    except requests.RequestException as e:
        app.logger.error(f"Error al realizar la solicitud HTTP: {e}")
        return None

    # Procesar la respuesta
    data = response.json()

    # Inicializar listas para preguntas y diccionario para respuestas
    preguntas = []
    respuestas = {}
    for thread_id, pares in data.items():
        for par in pares:
            pregunta = par['pregunta']
            respuesta = par['respuesta']
            preguntas.append(pregunta)
            respuestas[pregunta] = respuesta

    # Verificar si hay preguntas en el caché
    if not preguntas:
        app.logger.info("No hay preguntas en el caché para comparar")
        return None

    # Vectorizar las preguntas
    vectorizer = TfidfVectorizer()
    matriz_tfidf = vectorizer.fit_transform(preguntas)

    # Vectorizar la pregunta del usuario
    pregunta_vectorizada = vectorizer.transform([pregunta_usuario])
    
    # Calcular similitudes
    similitudes = cosine_similarity(pregunta_vectorizada, matriz_tfidf)

    # Encontrar la pregunta más similar
    indice_mas_similar = np.argmax(similitudes)
    similitud_maxima = similitudes[0, indice_mas_similar]

    # Umbral de similitud para considerar una respuesta válida
    UMBRAL_SIMILITUD = 0.7
    if similitud_maxima > UMBRAL_SIMILITUD:
        pregunta_similar = preguntas[indice_mas_similar]
        respuesta_similar = respuestas[pregunta_similar]
        app.logger.info(f"Respuesta encontrada: {respuesta_similar}")
        return respuesta_similar
    else:
        app.logger.info("No se encontraron preguntas similares con suficiente similitud")
        return False


####### Fin Sistema de cache #######


####### NUEVO SITEMA DE BUSQUEDA #######
def preprocess_text(text):
    app.logger.info("Preprocesando texto")
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_preprocess_data(file_path):
    app.logger.info(f"Cargando y preprocesando datos desde {file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    for item in data:
        item['text'] = preprocess_text(item['text'])
    app.logger.info("Datos cargados y preprocesados exitosamente")
    return data

def generate_gpt_embeddings(text):
    app.logger.info("Generando embedding de GPT para el texto")
    response = openai.Embedding.create(input=text, engine="text-similarity-babbage-001")
    return response['data'][0]['embedding']

def index_data_to_elasticsearch(dataset):
    app.logger.info("Indexando datos y embeddings en Elasticsearch")
    actions = []
    for item in dataset:
        embedding = generate_gpt_embeddings(item['text'])
        action = {
            "_index": INDICE_ELASTICSEARCH,
            "_id": item['id'],
            "_source": {
                "text": item['text'],
                "embedding": embedding
            }
        }
        actions.append(action)
    helpers.bulk(es_client, actions)
    app.logger.info("Datos y embeddings indexados exitosamente en Elasticsearch")

def search_in_elasticsearch(query):
    app.logger.info(f"Realizando búsqueda en Elasticsearch para la consulta: {query}")
    query_embedding = generate_gpt_embeddings(query)
    search_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                "params": {"query_vector": query_embedding}
            }
        }
    }
    response = es_client.search(index=INDICE_ELASTICSEARCH, body=search_query)
    app.logger.info("Búsqueda completada en Elasticsearch")
    return response

def generar_respuesta(texto):
    app.logger.info(f"Generando respuesta con GPT-4 para el texto: {texto}")
    response = openai.Completion.create(engine="gpt-4-1106-preview", prompt=texto, max_tokens=150)
    return response.choices[0].text.strip()

def encontrar_respuesta(ultima_pregunta, contexto=None):
    app.logger.info(f"Encontrando respuesta para la pregunta: {ultima_pregunta}")
    pregunta_procesada = preprocess_text(ultima_pregunta)
    
    # Procesar el contexto si está disponible
    contexto_procesado = preprocess_text(contexto) if contexto else ""

    # Realizar la búsqueda en Elasticsearch
    if contexto_procesado:
        resultados_busqueda = search_in_elasticsearch(contexto_procesado)
        textos_combinados = " ".join([hit['_source']['text'] for hit in resultados_busqueda['hits']['hits']])
    else:
        textos_combinados = ""

    # Preparar el texto completo para GPT-4
    texto_completo_para_gpt = f"Pregunta: {pregunta_procesada}\nContexto: {textos_combinados}"
    
    # Generar una respuesta con GPT-4
    respuesta = generar_respuesta(texto_completo_para_gpt)
    return respuesta


def prepare_data_for_finetuning(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    text_data = ""
    for item in data.values():
        question = item["Pregunta"][0]
        answer = item["respuesta"]
        text_data += f"Pregunta: {question}\nRespuesta: {answer}\n\n"
    temp_file_path = 'temp_finetune_data.txt'
    with open(temp_file_path, 'w', encoding='utf-8') as file:
        file.write(text_data)
    return temp_file_path

def finetune_model(model, tokenizer, file_path, output_dir):
    train_dataset = TextDataset(tokenizer=tokenizer, file_path=file_path, block_size=128)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=10_000,
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset
    )
    trainer.train()

####### FIN NUEVO SITEMA DE BUSQUEDA #######




# Nuevo Procesamiento de consultas de usuario


def traducir_texto_con_openai(texto, idioma_destino):
    openai.api_key = os.environ.get('OPENAI_API_KEY')
    try:
        prompt = f"Traduce este texto al {idioma_destino}: {texto}"
        response = openai.Completion.create(
            model="gpt-3.5-turbo",
            prompt=prompt,
            max_tokens=60
        )
        return response.choices[0].text.strip()
    except Exception as e:
        app.logger.info(f"Error al traducir texto con OpenAI: {e}")
        return texto  # Devuelve el texto original si falla la traducción

respuestas_por_defecto = [
    "Lamentamos no poder encontrar una respuesta precisa. Para más información, contáctanos en info@iurban.es.",
    "No hemos encontrado una coincidencia exacta, pero estamos aquí para ayudar. Escríbenos a info@iurban.es para más detalles.",
    "Aunque no encontramos una respuesta específica, nuestro equipo está listo para asistirte. Envía tus preguntas a info@iurban.es.",
    "Parece que no tenemos una respuesta directa, pero no te preocupes. Para más asistencia, comunícate con nosotros en info@iurban.es.",
    "No pudimos encontrar una respuesta clara a tu pregunta. Si necesitas más información, contáctanos en info@iurban.es.",
    "Disculpa, no encontramos una respuesta adecuada. Para más consultas, por favor, escribe a info@iurban.es.",
    "Sentimos no poder ofrecerte una respuesta exacta. Para obtener más ayuda, contacta con info@iurban.es.",
    "No hemos podido encontrar una respuesta precisa a tu pregunta. Por favor, contacta con info@iurban.es para más información.",
    "Lo sentimos, no tenemos una respuesta directa a tu consulta. Para más detalles, envía un correo a info@iurban.es.",
    "Nuestra búsqueda no ha dado resultados específicos, pero podemos ayudarte más. Escríbenos a info@iurban.es."
]


def buscar_en_respuestas_preestablecidas_nlp(pregunta_usuario, chatbot_id, umbral_similitud=0.5):
    app.logger.info("Iniciando búsqueda en respuestas preestablecidas con NLP")

    modelo = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


    if not os.path.exists(json_file_path):
        app.logger.warning(f"Archivo JSON no encontrado en la ruta: {json_file_path}")
        return None

    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        preguntas_respuestas = json.load(json_file)

    # Genera embeddings para la pregunta del usuario
    embedding_pregunta_usuario = modelo.encode(pregunta_usuario, convert_to_tensor=True)

    # Primera pasada: buscar por similitud con las preguntas completas
    preguntas = [entry["Pregunta"][0] for entry in preguntas_respuestas.values()]
    embeddings_preguntas = modelo.encode(preguntas, convert_to_tensor=True)
    similitudes = util.pytorch_cos_sim(embedding_pregunta_usuario, embeddings_preguntas)[0]
    mejor_coincidencia = similitudes.argmax()
    max_similitud = similitudes[mejor_coincidencia].item()

    if max_similitud >= umbral_similitud:
        respuesta_mejor_coincidencia = list(preguntas_respuestas.values())[mejor_coincidencia]["respuesta"]
        return respuesta_mejor_coincidencia

    # Segunda pasada: buscar por similitud con palabras clave si no se encuentra coincidencia en la primera pasada
    palabras_clave = [" ".join(entry["palabras_clave"]) for entry in preguntas_respuestas.values()]
    embeddings_palabras_clave = modelo.encode(palabras_clave, convert_to_tensor=True)
    similitudes = util.pytorch_cos_sim(embedding_pregunta_usuario, embeddings_palabras_clave)[0]
    mejor_coincidencia = similitudes.argmax()
    max_similitud = similitudes[mejor_coincidencia].item()

    if max_similitud >= umbral_similitud:
        respuesta_mejor_coincidencia = list(preguntas_respuestas.values())[mejor_coincidencia]["respuesta"]
        return respuesta_mejor_coincidencia

    app.logger.info("No se encontró una coincidencia adecuada")
    return False


def comprobar_coherencia_gpt(pregunta, respuesta):

    openai.api_key = os.environ.get('OPENAI_API_KEY')
    # Realizar una segunda llamada a OpenAI para traducir la respuesta seleccionada
    respuesta_traducida = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": f".No traduzcas los enlaces déjalos como están. El idioma original es el de la pregunta:  {pregunta}. Traduce, literalmente {respuesta_mejorada}, al idioma de la pregiunta. Si la pregunta es en ingles responde en ingles y asi con todo los idiomas, catalán, frances y todos los que tengas disponibles. No le añadas ninguna observacion de ningun tipo ni mensaje de error. No agregues comentarios ni observaciones en ningun idioma."},                
            {"role": "user", "content": respuesta_mejorada}
        ]
    )
    
    respuesta_traducida = respuesta_traducida.choices[0].message['content'].strip()

    prompt = f"Esta pregunta: '{pregunta}', es coherente con la respuesta: '{respuesta_traducida}'. Responde solo True o False, sin signos de puntuacion y la primera letra en mayúscula."

    response = ChatCompletion.create(
        model="gpt-4",  # O el modelo que prefieras
        messages=[
            {"role": "system", "content": "Por favor, evalúa la coherencia entre la pregunta y la respuesta."},
            {"role": "user", "content": prompt}
        ]
    )

    respuesta_gpt = response.choices[0].message['content'].strip().lower()
    # Limpiar la respuesta de puntuación y espacios adicionales
    respuesta_gpt = re.sub(r'\W+', '', respuesta_gpt)

    app.logger.info(respuesta_gpt)

    # Evaluar la respuesta
    if respuesta_gpt == "true":
        return True
    else:
        return False


####### FIN Utils busqueda en Json #######

import json

def safe_encode_to_json(content):
    app.logger.info("Iniciando safe_encode_to_json")

    def encode_item(item):
        if isinstance(item, str):
            encoded_str = item.encode('utf-8', 'ignore').decode('utf-8')
            app.logger.info(f"Codificando cadena: original='{item[:30]}...', codificado='{encoded_str[:30]}...'")
            return encoded_str
        elif isinstance(item, dict):
            app.logger.info(f"Procesando diccionario: {list(item.keys())[:5]}...")
            return {k: encode_item(v) for k, v in item.items()}
        elif isinstance(item, list):
            app.logger.info(f"Procesando lista: longitud={len(item)}")
            return [encode_item(x) for x in item]
        app.logger.info(f"Tipo no procesado: {type(item)}")
        return item

    app.logger.info("Codificando contenido para JSON")
    safe_content = encode_item(content)
    app.logger.info("Contenido codificado con éxito")
    json_output = json.dumps(safe_content, ensure_ascii=False, indent=4)
    app.logger.info("Codificación JSON completada con éxito")
    return json_output


####### Inicio Endpoints #######

@app.route('/ask', methods=['POST'])
def ask():
    app.logger.info("Solicitud recibida en /ask")

    try:
        data = request.get_json()
        chatbot_id = data.get('chatbot_id', 'default_id')
        pares_pregunta_respuesta = data.get('pares_pregunta_respuesta', [])
        ultima_pregunta = pares_pregunta_respuesta[-1]['pregunta'] if pares_pregunta_respuesta else ""
        ultima_respuesta = pares_pregunta_respuesta[-1]['respuesta'] if pares_pregunta_respuesta else ""
        contexto = ' '.join([f"Pregunta: {par['pregunta']} Respuesta: {par['respuesta']}" for par in pares_pregunta_respuesta[:-1]])
       
        app.logger.info("Antes de encontrar_respuesta cache")
        respuesta_cache = encontrar_respuesta_en_cache(ultima_pregunta, chatbot_id)
        app.logger.info("despues de encontrar_respuesta cache")
        app.logger.info(respuesta_cache)
        if respuesta_cache:
            return jsonify({'respuesta': respuesta_cache, 'fuente': 'cache'})

        if ultima_respuesta == "":
            ultima_respuesta = identificar_saludo_despedida(ultima_pregunta)
            if ultima_respuesta:
                fuente_respuesta = 'saludo_o_despedida'

            if not ultima_respuesta:
                ultima_respuesta = buscar_en_respuestas_preestablecidas_nlp(ultima_pregunta, chatbot_id)
                if ultima_respuesta:
                    fuente_respuesta = 'preestablecida'

            if not ultima_respuesta:
                ultima_respuesta = obtener_eventos(ultima_pregunta, chatbot_id)
                if ultima_respuesta:
                    fuente_respuesta = 'eventos'

            dataset_file_path = os.path.join(BASE_DATASET_DIR, str(chatbot_id), 'dataset.json')
            if not ultima_respuesta and os.path.exists(dataset_file_path):
                with open(dataset_file_path, 'r') as file:
                    datos_del_dataset = json.load(file)
                ultima_respuesta = encontrar_respuesta(ultima_pregunta, datos_del_dataset, contexto)
                if ultima_respuesta:
                    fuente_respuesta = 'dataset'

            if not ultima_respuesta:
                fuente_respuesta = 'respuesta_por_defecto'
                ultima_respuesta = seleccionar_respuesta_por_defecto()
                ultima_respuesta = traducir_texto_con_openai(ultima_pregunta, ultima_respuesta)

            if ultima_respuesta:
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
                fuente_respuesta = 'mejorada'

            return jsonify({'respuesta': ultima_respuesta, 'fuente': fuente_respuesta})

        else:
            return jsonify({'respuesta': ultima_respuesta, 'fuente': 'existente'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/uploads', methods=['POST'])
def upload_file():
    if 'documento' not in request.files:
        return jsonify({"respuesta": "No se encontró el archivo 'documento'", "codigo_error": 1})

    uploaded_file = request.files['documento']
    chatbot_id = request.form.get('chatbot_id')

    if uploaded_file.filename == '':
        return jsonify({"respuesta": "No se seleccionó ningún archivo", "codigo_error": 1})

    extension = os.path.splitext(uploaded_file.filename)[1][1:].lower()
    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({"respuesta": "Formato de archivo no permitido", "codigo_error": 1})

    # Guardar archivo físicamente en el directorio 'docs'
    docs_folder = os.path.join('data', 'uploads', 'docs', str(chatbot_id))
    os.makedirs(docs_folder, exist_ok=True)
    file_path = os.path.join(docs_folder, uploaded_file.filename)
    uploaded_file.save(file_path)

    # Procesar contenido del archivo
    readable_content = process_file(file_path, extension)
    readable_content = readable_content.encode('utf-8', 'ignore').decode('utf-8')

    # Actualizar dataset.json
    dataset_file_path = os.path.join(BASE_DATASET_DIR, str(chatbot_id), 'dataset.json')
    os.makedirs(os.path.dirname(dataset_file_path), exist_ok=True)

    dataset_entries = {}
    if os.path.exists(dataset_file_path):
        with open(dataset_file_path, 'rb') as json_file:
            file_content = json_file.read()
            if file_content:
                decoded_content = file_content.decode('utf-8', 'ignore')
                try:
                    dataset_entries = json.loads(decoded_content)
                except json.decoder.JSONDecodeError:
                    dataset_entries = {}
            else:
                dataset_entries = {}

    # Añadir entrada al dataset
    dataset_entries[uploaded_file.filename] = {
        "indice": uploaded_file.filename,
        "url": file_path,
        "dialogue": readable_content
    }

    # Escribir cambios en dataset.json
    with open(dataset_file_path, 'w', encoding='utf-8') as json_file_to_write:
        json_content = safe_encode_to_json(dataset_entries)
        json_file_to_write.write(json_content)

    return jsonify({
        "respuesta": "Archivo procesado y añadido al dataset con éxito.",
        "word_count": len(readable_content.split()),
        "codigo_error": 0
    })



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
        app.logger.info("Aqui si")
        with open(dataset_file_path, 'w', encoding='utf-8') as json_file_to_write:
            app.logger.info("Aqui NO")
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

@app.route('/delete_dataset_entries', methods=['POST'])
def delete_dataset_entries():
    data = request.json
    urls_to_delete = set(data.get('urls', []))
    chatbot_id = data.get('chatbot_id')

    if not urls_to_delete or not chatbot_id:
        app.logger.warning("Faltan 'urls' o 'chatbot_id' en la solicitud")
        return jsonify({"error": "Missing 'urls' or 'chatbot_id'"}), 400

    BASE_DATASET_DIR = 'data/uploads/datasets'
    dataset_file_path = os.path.join(BASE_DATASET_DIR, str(chatbot_id), 'dataset.json')

    if not os.path.exists(dataset_file_path):
        app.logger.error("Archivo del dataset no encontrado")
        return jsonify({"status": "error", "message": "Dataset file not found"}), 404

    try:
        with open(dataset_file_path, 'r+') as file:
            dataset = json.load(file)
            urls_in_dataset = {entry['url'] for entry in dataset}
            urls_not_found = urls_to_delete - urls_in_dataset

            if urls_not_found:
                app.logger.info(f"URLs no encontradas en el dataset: {urls_not_found}")

            updated_dataset = [entry for entry in dataset if entry['url'] not in urls_to_delete]

            file.seek(0)
            file.truncate()
            json.dump(updated_dataset, file, indent=4)

        app.logger.info("Proceso de eliminación completado con éxito")
        return jsonify({"status": "success", "message": "Dataset entries deleted successfully", "urls_not_found": list(urls_not_found)})
    except json.JSONDecodeError:
        app.logger.error("Error al leer el archivo JSON")
        return jsonify({"status": "error", "message": "Invalid JSON format in dataset file"}), 500
    except Exception as e:
        app.logger.error(f"Error inesperado: {e}")
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

@app.route('/train_and_fine_tuning', methods=['POST'])
def train_and_fine_tuning():
    data = request.json
    chatbot_id = data.get('chatbot_id')

    if not chatbot_id:
        return jsonify({'error': 'Falta chatbot_id'}), 400

    # Rutas a los archivos de datos
    json_file_path = f'data/uploads/pre_established_answers/{chatbot_id}/pre_established_answers.json'
    dataset_file_path = os.path.join(BASE_DATASET_DIR, str(chatbot_id), 'dataset.json')

    # Inicialización del modelo y el tokenizer
    model_name = 'EleutherAI/gpt-neo-2.7B'  # Este es un modelo grande, asegúrate de tener suficiente RAM y GPU
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPTNeoForCausalLM.from_pretrained(model_name)

    if os.path.exists(json_file_path):
        finetune_file_path = prepare_data_for_finetuning(json_file_path)
        finetune_model(model, tokenizer, finetune_file_path, f'./model_output_finetuning_{chatbot_id}')

    if os.path.exists(dataset_file_path):
        train_dataset = TextDataset(tokenizer=tokenizer, file_path=dataset_file_path, block_size=128)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        training_args = TrainingArguments(
            output_dir=f'./model_output_training_{chatbot_id}',
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            save_steps=10_000,
            save_total_limit=2,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset
        )
        trainer.train()

    return jsonify({'message': f'Proceso completado para chatbot_id {chatbot_id}'}), 200



@app.route('/run_tests', methods=['POST'])
def run_tests():
    import subprocess
    result = subprocess.run(['python', 'run_tests.py'], capture_output=True, text=True)
    return result.stdout
 ######## Fin Endpoints ######## 

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)