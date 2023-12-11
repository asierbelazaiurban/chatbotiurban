from datetime import datetime, timedelta
import requests
import json
from dateutil import parser
import spacy
from flask import Flask

app = Flask(__name__)

def extraer_fechas_con_contexto(pregunta):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(pregunta)
    fechas = []
    for ent in doc.ents:
        if ent.label_ == 'DATE':
            try:
                fecha = parser.parse(ent.text)
                fechas.append(fecha)
            except ValueError:
                app.logger.error(f"Error al parsear fecha: {ent.text}")
                continue
    return fechas

def aplicar_logica_fechas(fechas):
    if len(fechas) == 0:
        # Si no hay fechas, usar la fecha actual y +4 días
        start = datetime.now()
        end = start + timedelta(days=4)
    elif len(fechas) == 1:
        # Si hay una fecha, se asume como start y end es start + 3 días
        start = fechas[0]
        end = start + timedelta(days=3)
    else:
        # Si hay dos o más fechas, se toman las primeras dos y se ordenan
        fechas.sort()
        start, end = fechas[0], fechas[1]

    return start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')

def obtener_eventos(pregunta, chatbot_id):
    fechas = extraer_fechas_con_contexto(pregunta)
    start, end = aplicar_logica_fechas(fechas)

    url = 'https://experimental.ciceroneweb.com/api/search-event-chatbot'
    headers = {'Content-Type': 'application/json'}
    payload = {
        "start": start,
        "end": end,
        "chatbot_id": chatbot_id
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        eventos_data = response.json()

        eventos = eventos_data.get('eventos', [])
        if not eventos:
            # Si no hay eventos, devuelve un mensaje específico
            return "No se han encontrado eventos en las fechas, por favor especifique a una fecha concreta"

        eventos_concatenados = ' '.join(eventos)
        return eventos_concatenados

    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error en la solicitud HTTP: {e}")
        return None


