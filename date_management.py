import re
import openai
from dateutil import parser
import dateparser

def encontrar_fechas_con_regex(texto):
    """
    Encuentra fechas en el texto utilizando expresiones regulares.
    Devuelve una lista de todas las fechas encontradas.
    """
    # Ejemplo de patrón de fecha (día/mes/año, mes/día/año, etc.)
    patrones_fecha = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{2,4}[/-]\d{1,2}[/-]\d{1,2}\b"
        # Puedes añadir más patrones según sea necesario
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
    openai.api_key = os.environ.get('OPENAI_API_KEY')

    try:
        respuesta = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": texto},
            ]
        )

        texto_interpretado = respuesta.choices[0].message['content']

        # Utiliza regex para encontrar fechas en el texto interpretado
        fechas_regex = encontrar_fechas_con_regex(texto_interpretado)

        # Convierte las fechas a formato MySQL
        fechas_procesadas = [interpretar_fecha_con_nlp(fecha) for fecha in fechas_regex]

        # Filtrar None y devolver fechas únicas
        return list(set([fecha for fecha in fechas_procesadas if fecha]))

    except Exception as e:
        return []

# Ejemplo de uso
# fechas = interpretar_intencion_y_fechas("Quiero saber los eventos en Nueva York el próximo mes.", tu_openai_api_key)
# print(fechas)


def obtener_eventos(pregunta, chatbot_id):
    fechas = interpretar_intencion_y_fechas(pregunta)

    if not fechas:
        return "No se pudo interpretar las fechas de la pregunta."

    # Suponiendo que siempre necesitamos dos fechas, inicio y fin.
    # Si solo se encuentra una fecha, se puede usar la misma para 'start' y 'end',
    # o manejarlo de otra manera según la lógica de negocio.
    start, end = fechas[0], fechas[-1] if len(fechas) > 1 else fechas[0]

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
            return "No se han encontrado eventos en las fechas especificadas."

        eventos_concatenados = ' '.join(eventos)
        return eventos_concatenados

    except requests.exceptions.RequestException as e:
        # app.logger.error(f"Error en la solicitud HTTP: {e}")
        return f"Error en la solicitud HTTP: {e}"


