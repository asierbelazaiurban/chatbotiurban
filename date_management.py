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

    instruccion_gpt4 = "Interpreta la pregunta del usuario para identificar cualquier referencia a fechas y conviértelas en un formato estándar 'YYYY-MM-DD'. Considera el contexto actual y las convenciones comunes para interpretar expresiones como 'mañana', 'el próximo año', etc."

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

        # Mejora en la interpretación de las fechas
        # Utiliza expresiones regulares para encontrar fechas en el formato 'YYYY-MM-DD' en el texto interpretado
        fechas = re.findall(r'\d{4}-\d{2}-\d{2}', texto_interpretado)
        if fechas:
            fecha_inicial = min(fechas)
            fecha_final = max(fechas)
            return fecha_inicial, fecha_final
        else:
            # Utiliza dateparser como respaldo si no se encuentran fechas con la expresión regular
            settings = {'PREFER_DATES_FROM': 'future', 'RELATIVE_BASE': fecha_actual}
            fecha_interpretada = dateparser.parse(texto_interpretado, settings=settings)
            if fecha_interpretada:
                fecha_format = fecha_interpretada.strftime('%Y-%m-%d')
                return fecha_format, fecha_format  # Asumimos la misma fecha inicial y final para simplificar

        return None, None

    except Exception as e:
        app.logger.error("Excepción encontrada: %s", e)
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


