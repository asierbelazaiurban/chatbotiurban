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
from datetime import datetime

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

    instruccion_gpt4 = (
        "Tu tarea es interpretar la pregunta del usuario, que puede estar en cualquier idioma, "
        "para identificar referencias a fechas. Convierte estas referencias en un formato estándar 'YYYY-MM-DD'. "
        "Considera el contexto actual y las convenciones comunes para interpretar expresiones como 'mañana', "
        "'el próximo año', etc. Devuelve únicamente las fechas identificadas, indicando cuál es la primera fecha "
        "y cuál la segunda en el rango temporal mencionado."
    )

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

        # Llama a la función para convertir las referencias temporales a fechas
        fecha_inicial, fecha_final = convertir_referencia_temporal_a_fechas(texto_interpretado, fecha_actual)

        return fecha_inicial, fecha_final

    except Exception as e:
        app.logger.error("Excepción encontrada: %s", e)
        return None, None

def convertir_referencia_temporal_a_fechas(referencia, fecha_actual):
    settings = {
        'PREFER_DATES_FROM': 'future',
        'RELATIVE_BASE': fecha_actual,
        'DATE_ORDER': 'DMY'
    }

    fecha_interpretada = dateparser.parse(referencia, settings=settings)
    if fecha_interpretada:
        fecha_inicial = fecha_interpretada.strftime('%Y-%m-%d')
        fecha_final = fecha_interpretada.strftime('%Y-%m-%d')
        return fecha_inicial, fecha_final
    else:
        return None, None

def obtener_eventos(pregunta, chatbot_id):
    fecha_actual = datetime.now()

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


