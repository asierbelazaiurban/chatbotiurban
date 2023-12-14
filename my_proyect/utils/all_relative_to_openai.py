import os
from flask import Flask
import logging
from logging import FileHandler
import openai
import unidecode

from flask import Flask

app = Flask(__name__)
####### Configuración logs #######

if not os.path.exists('logs'):
    os.mkdir('logs')

file_handler = FileHandler('logs/chatbotiurban.log')
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.DEBUG)  # Usa DEBUG o INFO según necesites

app.logger.addHandler(file_handler)
app.logger.setLevel(logging.DEBUG)  # Asegúrate de que este nivel sea consistente con file_handler.setLevel

app.logger.info('Inicio de la aplicación ChatbotIUrban')


#######  #######

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
        "en este momento sobre este tema, ¿puedo ayudarte en algo más?'. Actúa como un guía turístico experto, "
        "presentando tus respuestas en forma de listas para facilitar la planificación diaria de actividades. "
        "Es crucial responder en el mismo idioma que la pregunta. Al finalizar tu respuesta, recuerda sugerir "
        "'Si deseas más información, crea tu ruta con Cicerone o consulta las rutas de expertos locales'. "
        "Si careces de la información solicitada, evita comenzar con 'Lo siento, no puedo darte información específica'. "
        "En su lugar, aconseja planificar con Cicerone para una experiencia personalizada. Para cualquier duda, "
        "proporciona el contacto: info@iurban.es."
    )

    # Construir el prompt base
    prompt_base = f"Pregunta: {pregunta}\nRespuesta: {respuesta_original}\n--\n{new_prompt}. Respondiendo siempre en el idioma del contexto"

    # Intentar generar la respuesta mejorada
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
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
    prompt_base = f"Si hay algun tema con la codificación o caracteres, por ejemplo (Lo siento, pero parece que hay un problema con la codificación de caracteres en tu pregunta o similar...)no te refieras  ni comentes el problema {contexto_adicional}\n\nPregunta reciente: {pregunta}\nRespuesta original: {respuesta}\n--\n {new_prompt}, siempre en el idioma del contexto"
    app.logger.info(prompt_base)

    # Generar la respuesta mejorada
    try:
        response = openai.ChatCompletion.create(
            model=model_gpt if model_gpt else "gpt-4",
            messages=[
                {"role": "system", "content": prompt_base},
                {"role": "user", "content": respuesta}
            ],
            temperature=float(temperature) if temperature else 0.5
        )
        improved_response = response.choices[0].message['content'].strip()
        app.logger.info("Respuesta generada con éxito.")
        return improved_response
    except Exception as e:
        app.logger.error(f"Error al interactuar con OpenAI: {e}")
        return None

import os
import unidecode
import openai
from flask import Flask

# Configurar la aplicación Flask (si estás usando Flask)
app = Flask(__name__)

####### Configuración logs #######

if not os.path.exists('logs'):
    os.mkdir('logs')

file_handler = FileHandler('logs/chatbotiurban.log')
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.DEBUG)  # Usa DEBUG o INFO según necesites

app.logger.addHandler(file_handler)
app.logger.setLevel(logging.DEBUG)  # Asegúrate de que este nivel sea consistente con file_handler.setLevel

app.logger.info('Inicio de la aplicación ChatbotIUrban')


#######  #######
def generar_contexto_con_openai(historial):
    openai.api_key = os.environ.get('OPENAI_API_KEY')

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Enviamos una conversación para que entiendas el contexto"},
                {"role": "user", "content": historial}
            ]
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        app.logger.info(f"Error al generar contexto con OpenAI: {e}")
        return ""


def buscar_en_openai_relacion_con_eventos(frase):
    app.logger.info("Hemos detectado un evento")

    # Texto fijo a concatenar
    texto_fijo = "Necesito saber si la frase que te paso está relacionada con eventos, se pregunta sobre eventos, cosas que hacer etc.... pero que solo me contestes con un si o un no. la frase es: "
    frase_combinada = texto_fijo + frase

    # Establecer la clave de API de OpenAI
    openai.api_key = os.environ.get('OPENAI_API_KEY')

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": frase_combinada}
            ]
        )

        # Interpretar la respuesta y normalizarla
        respuesta = response.choices[0].message['content'].strip().lower()
        respuesta = unidecode.unidecode(respuesta).replace(".", "")

        app.logger.info(f"Respuesta de OpenAI: {respuesta}")

        if respuesta == "si":
            app.logger.info("La respuesta es sí, relacionada con eventos")
            return True
        elif respuesta == "no":
            app.logger.info("La respuesta es no, no relacionada con eventos")
            return False
        else:
            app.logger.info("La respuesta no es ni sí ni no")
            return None
    except Exception as e:
        app.logger.error(f"Error al procesar la solicitud: {e}")
        return None
