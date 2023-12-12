import re
import openai
import os
import dateparser
import requests
import json
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask
import html  

app = Flask(__name__)

# Configuración del Logger
if not app.debug:
    if not os.path.exists('logs'):
        os.mkdir('logs')
    file_handler = RotatingFileHandler('logs/chatbot.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)

    app.logger.setLevel(logging.INFO)
    app.logger.info('Chatbot startup')

def encontrar_fechas_con_regex(texto):
    patrones_fecha = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",  # dd/mm/yyyy o mm/dd/yyyy
        r"\b\d{2,4}[/-]\d{1,2}[/-]\d{1,2}\b",  # yyyy/mm/dd
        r"\b\d{4}\b"                            # yyyy
    ]
    
    fechas_encontradas = []
    for patron in patrones_fecha:
        fechas_encontradas.extend(re.findall(patron, texto))

    app.logger.info("Fechas encontradas con regex: %s", fechas_encontradas)
    return fechas_encontradas


def interpretar_fecha_con_nlp(fecha_texto):
    fecha = dateparser.parse(fecha_texto)
    if fecha:
        fecha_format = fecha.strftime('%Y-%m-%d')
        app.logger.info("Fecha interpretada con NLP: %s", fecha_format)
        return fecha_format
    return None

def interpretar_intencion_y_fechas(texto):
    openai.api_key = os.environ.get('OPENAI_API_KEY')

    try:
        # Utiliza OpenAI GPT-4 para interpretar la intención y el contexto
        respuesta = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": texto},
            ]
        )

        texto_interpretado = respuesta.choices[0].message['content']
        app.logger.info("Texto interpretado: %s", texto_interpretado)

        # Utiliza dateparser para interpretar las fechas
        settings = {'PREFER_DATES_FROM': 'future', 'RELATIVE_BASE': datetime.datetime.now()}
        fecha_interpretada = dateparser.parse(texto_interpretado, settings=settings)
        if fecha_interpretada:
            fecha_format = fecha_interpretada.strftime('%Y-%m-%d')
            return fecha_format, fecha_format

        # Usa interpretar_fecha_con_nlp si dateparser no encuentra una fecha
        fecha_nlp = interpretar_fecha_con_nlp(texto_interpretado)
        if fecha_nlp:
            return fecha_nlp, fecha_nlp

        return None, None

    except Exception as e:
        app.logger.error("Excepción encontrada: %s", e)
        return None, None

def obtener_eventos(pregunta, chatbot_id):
    fecha_inicial, fecha_final = interpretar_intencion_y_fechas(pregunta)
    app.logger.info("Fecha inicial interpretada: %s", fecha_inicial)
    app.logger.info("Fecha final interpretada: %s", fecha_final)
    app.logger.info("ID del Chatbot utilizado: %s", chatbot_id)

    if fecha_inicial is None or fecha_final is None:
        app.logger.info("No se pudo interpretar las fechas de la pregunta.")
        return "No se pudo interpretar las fechas de la pregunta."

    url = 'https://experimental.ciceroneweb.com/api/search-event-chatbot'
    headers = {'Content-Type': 'application/json'}
    payload = {
        "start": fecha_inicial,
        "end": fecha_final,
        "chatbot_id": chatbot_id
    }

    try:
        app.logger.info("Enviando solicitud HTTP a: %s", url)
        app.logger.info("Payload de la solicitud: %s", json.dumps(payload))
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()

        eventos_data = response.json()
        app.logger.info("Datos JSON de la respuesta: %s", eventos_data)

        # Convertir los eventos a string para limpieza
        eventos_string = json.dumps(eventos_data['events'])
        
        # Limpieza del string
        eventos_string = eventos_string.replace('\xa0', ' ')
        eventos_string = eventos_string.encode('utf-8', 'ignore').decode('utf-8')
        eventos_string = eventos_string.replace('"', '').replace('\\', '')
        eventos_string = eventos_string.replace('[', '').replace(']', '')
        eventos_string = eventos_string.replace('{', '').replace('}', '')
        eventos_string = eventos_string.replace(',', '. ')  # Convertir comas en puntos para una lectura más natural

        app.logger.info("Eventos en formato string legible: %s", eventos_string)
        return eventos_string

    except requests.exceptions.RequestException as e:
        app.logger.error("Error en la solicitud HTTP: %s", e)
        return "Error al obtener eventos: " + str(e)
