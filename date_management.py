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
import datetime

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

def interpretar_intencion_y_fechas(texto, fecha_actual):
    openai.api_key = os.environ.get('OPENAI_API_KEY')

    instruccion_gpt4 = "Tu eres un asistente virtual. Tu tarea es interpretar la pregunta del usuario y devolver la fecha mencionada en un formato estándar como 'YYYY-MM-DD'. Debes entender y procesar preguntas en cualquier idioma."

    try:
        respuesta = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": instruccion_gpt4},
                {"role": "user", "content": texto},
            ]
        )

        texto_interpretado = respuesta.choices[0].message['content']
        app.logger.info("Texto interpretado: %s", texto_interpretado)

        # Utiliza dateparser para interpretar la respuesta en el formato deseado
        settings = {'PREFER_DATES_FROM': 'future', 'RELATIVE_BASE': fecha_actual}
        fecha_interpretada = dateparser.parse(texto_interpretado, settings=settings)

        if fecha_interpretada:
            fecha_format = fecha_interpretada.strftime('%Y-%m-%d')
            return fecha_format, fecha_format  # Asumimos la misma fecha inicial y final para simplificar

        return None, None

    except Exception as e:
        app.logger.error("Excepción encontrada: %s", e)
        return None, None

def obtener_eventos(pregunta, chatbot_id, fecha_actual):
    fecha_inicial, fecha_final = interpretar_intencion_y_fechas(pregunta, fecha_actual)
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

        # Convertir los eventos a string para limpieza
        eventos_string = json.dumps(eventos_data['events'])

        # Limpieza del string
        eventos_string = eventos_string.replace('\xa0', ' ')
        eventos_string = eventos_string.encode('utf-8', 'ignore').decode('utf-8')
        eventos_string = eventos_string.replace('"', '').replace('\\', '')
        eventos_string = eventos_string.replace('[', '').replace(']', '')
        eventos_string = eventos_string.replace('{', '').replace('}', '')
        eventos_string = eventos_string.replace(',', '. ')  # Convertir comas en puntos para una lectura más natural

        return eventos_string

    except requests.exceptions.RequestException as e:
        app.logger.error("Error en la solicitud HTTP: %s", e)
        return "Error al obtener eventos: " + str(e)

