import re
import openai
import os
from dateutil import parser
import dateparser
import requests
import json
from datetime import datetime

def encontrar_fechas_con_regex(texto):
    patrones_fecha = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",  # dd/mm/yyyy o mm/dd/yyyy
        r"\b\d{2,4}[/-]\d{1,2}[/-]\d{1,2}\b",  # yyyy/mm/dd
        r"\b\d{4}\b"                            # yyyy
    ]
    
    fechas_encontradas = []
    for patron in patrones_fecha:
        fechas_encontradas.extend(re.findall(patron, texto))

    return fechas_encontradas


def interpretar_fecha_con_nlp(fecha_texto):
    fecha = dateparser.parse(fecha_texto)
    if fecha:
        return fecha.strftime('%Y-%m-%d')
    return None

def interpretar_intencion_y_fechas(texto):
    openai.api_key = os.environ.get('OPENAI_API_KEY')

    try:
        respuesta = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant capable of understanding dates in any language."},
                {"role": "user", "content": texto},
            ]
        )

        texto_interpretado = respuesta.choices[0].message['content']

        fechas_regex = encontrar_fechas_con_regex(texto_interpretado)
        fechas_procesadas = []

        for fecha in fechas_regex:
            if re.match(r"\b\d{4}\b", fecha):  # Solo año
                fecha_inicio = f"{fecha}-01-01"
                fecha_fin = f"{fecha}-12-31"
                fechas_procesadas.extend([fecha_inicio, fecha_fin])
            else:
                fecha_procesada = interpretar_fecha_con_nlp(fecha)
                if fecha_procesada:
                    fechas_procesadas.append(fecha_procesada)

        if fechas_procesadas:
            return min(fechas_procesadas), max(fechas_procesadas)
        else:
            # Intenta interpretar con NLP si no se encontraron fechas con regex
            fecha_procesada = interpretar_fecha_con_nlp(texto_interpretado)
            if fecha_procesada:
                return fecha_procesada, fecha_procesada

        return None, None

    except Exception as e:
        return None, None

def obtener_eventos(pregunta, chatbot_id):
    fecha_inicial, fecha_final = interpretar_intencion_y_fechas(pregunta)

    if fecha_inicial is None or fecha_final is None:
        return "No se pudo interpretar las fechas de la pregunta."

    url = 'https://experimental.ciceroneweb.com/api/search-event-chatbot'
    headers = {'Content-Type': 'application/json'}
    payload = {
        "start": fecha_inicial,
        "end": fecha_final,
        "chatbot_id": chatbot_id
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        eventos_data = response.json()

        eventos = eventos_data.get('eventos', [])
        if not eventos:
            return "No se han encontrado eventos en las fechas especificadas."

        eventos_concatenados = ' '.join(eventos)
        return eventos_concatenados

    except requests.exceptions.RequestException as e:
        return f"Error en la solicitud HTTP: {e}"




