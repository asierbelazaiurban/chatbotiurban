import re
import openai
import os
from dateutil import parser
import dateparser
import requests
import json

def encontrar_fechas_con_regex(texto):
    """
    Encuentra fechas en el texto utilizando expresiones regulares.
    Devuelve una lista de todas las fechas encontradas.
    """
    patrones_fecha = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{2,4}[/-]\d{1,2}[/-]\d{1,2}\b"
    ]
    
    fechas_encontradas = []
    for patron in patrones_fecha:
        fechas_encontradas.extend(re.findall(patron, texto))

    return fechas_encontradas

def interpretar_fecha_con_nlp(fecha_texto):
    """
    Interpreta una fecha dada en texto utilizando NLP.
    """
    fecha = dateparser.parse(fecha_texto)
    if fecha:
        return fecha.strftime('%Y-%m-%d')
    return None

def interpretar_intencion_y_fechas(texto):
    """
    Interpreta la intención y las fechas del texto utilizando el modelo de OpenAI,
    y devuelve las fechas en formato MySQL.
    """
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
        fechas_procesadas = [interpretar_fecha_con_nlp(fecha) for fecha in fechas_regex]

        fechas_unicas = list(set([fecha for fecha in fechas_procesadas if fecha]))

        fecha_inicial = fechas_unicas[0] if fechas_unicas else None
        fecha_final = fechas_unicas[-1] if len(fechas_unicas) > 1 else fecha_inicial

        return fecha_inicial, fecha_final

    except Exception as e:
        return None, None

def obtener_eventos(pregunta, chatbot_id):
    """
    Obtiene eventos basados en una pregunta, interpretando las fechas y la intención.
    """
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
        # Reemplazar con el manejo de errores adecuado para tu aplicación
        return f"Error en la solicitud HTTP: {e}"


