import re
import openai
import os
import dateparser
import requests
import json
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask
from datetime import datetime

app = Flask(__name__)

# Configuración del Logger
def configure_logger():
    if not app.debug and not os.path.exists('logs'):
        os.mkdir('logs')
    file_handler = RotatingFileHandler('logs/chatbot.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Chatbot startup')

configure_logger()

def get_openai_response(texto):
    openai.api_key = os.environ.get('OPENAI_API_KEY')
    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY is not set in environment variables")

    instruccion_gpt4 = ("""Tu tarea es identificar las referencias temporales en la pregunta del usuario, que puede estar en cualquier idioma. Busca expresiones como 'mañana', 'el próximo año', 'el finde', 'la semana que viene', etc., y devuelve estas referencias temporales tal como se mencionan, sin convertirlas a fechas específicas. Tu respuesta debe incluir solo las referencias temporales identificadas, sin fechas adicionales.""")

    respuesta = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": instruccion_gpt4},
            {"role": "user", "content": texto},
        ]
    )

    return respuesta.choices[0].message['content']

def interpretar_intencion_y_fechas(texto, fecha_actual):
    try:
        texto_interpretado = get_openai_response(texto)
        app.logger.info("Texto interpretado: %s", texto_interpretado)

        referencias_temporales = extraer_referencias_temporales(texto_interpretado)
        fechas = [convertir_referencia_temporal_a_fechas(referencia, fecha_actual) for referencia in referencias_temporales]

        # Asegúrate de que 'fechas' es una lista de tuplas
        fechas = [fecha for fecha in fechas if fecha is not None]
        return fechas if fechas else [(None, None)]

    except Exception as e:
        app.logger.error("Excepción encontrada: %s", e)
        # Devuelve una lista con una tupla de None si hay un error
        return [(None, None)]


def extraer_referencias_temporales(texto):
    # Implementa tu lógica para extraer referencias temporales del texto
    return re.findall('tu expresión regular aquí', texto)

def convertir_referencia_temporal_a_fechas(referencia, fecha_actual):
    try:
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
    except Exception as e:
        app.logger.error(f"Error al convertir referencia temporal '{referencia}': {e}")
        return None

        return None, None

def obtener_eventos(pregunta, chatbot_id):
    fecha_actual = datetime.now()

    resultado_fechas = interpretar_intencion_y_fechas(pregunta, fecha_actual)

    # Verifica si el resultado contiene al menos una tupla
    if resultado_fechas and len(resultado_fechas) > 0:
        fecha_inicial, fecha_final = resultado_fechas[0]
    else:
        app.logger.info("No se pudo interpretar las fechas de la pregunta.")
        return "No se pudo interpretar las fechas de la pregunta."

    # Registro de información
    app.logger.info("Fecha inicial interpretada: %s", fecha_inicial)
    app.logger.info("Fecha final interpretada: %s", fecha_final)
    app.logger.info("ID del Chatbot utilizado: %s", chatbot_id)

    # Si las fechas son None, maneja ese caso
    if fecha_inicial is None or fecha_final is None:
        app.logger.info("No se encontraron fechas en la pregunta.")
        return "No se encontraron fechas en la pregunta."

    # Preparar y enviar la solicitud
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


