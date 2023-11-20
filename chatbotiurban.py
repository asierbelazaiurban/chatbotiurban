#!/usr/bin/env python
# coding: utf-8


import faiss  # Ensure faiss library is installed
import numpy as np

# Global variable to store the FAISS index
faiss_index = None

def initialize_faiss_index(dimension):
    """
    Initialize the FAISS index with the specified dimension.
    """
    global faiss_index
    faiss_index = faiss.IndexFlatL2(dimension)

def get_faiss_index():
    """
    Return the current FAISS index. Ensure it has been initialized before calling this function.
    """
    global faiss_index
    if faiss_index is None:
        raise ValueError("FAISS index has not been initialized.")
    return faiss_index

# Example of how to initialize the index (adjust dimension as needed)
initialize_faiss_index(128)  # Assuming your embeddings are 128-dimensional


# In[ ]:


from flask import Flask, request, jsonify
from flask import request, jsonify
import numpy as np
import openai
import requests
from bs4 import BeautifulSoup
import faiss
import os
import shutil


# In[ ]:


app = Flask(__name__)


# In[ ]:


# Configura la clave de la API de OpenAI
api_key = os.getenv('TU_VARIABLE_DE_ENTORNO')


# In[ ]:


def generate_embedding(text, openai_api_key, chatbot_id):
    # Genera un embedding para un texto dado utilizando OpenAI.
    openai.api_key = openai.api_key # Set your OpenAI API key here
    response = openai.Embedding.create(engine="text-similarity-babbage-001", input=text)
    embedding = response['data'][0]['embedding']  # Usando el embedding de OpenAI  # Representación ficticia, reemplazar con la lógica adecuada
    return embedding


# In[ ]:


def process_results(indices, data):
    # Procesa los índices obtenidos de FAISS para recuperar información relevante.
    info = "Información relacionada con los índices en Milvus: " + ', '.join(str(idx) for idx in indices)
    return info


# In[ ]:


# Importar las bibliotecas necesarias
# Suponiendo que los datos son embeddings de dimensión 768 (cambiar según sea necesario)
dim = 768
num_data_points = 10000  # Número de puntos de datos (cambiar según sea necesario)
# Crear datos de ejemplo (reemplazar con tus propios datos)
data = np.random.rand(num_data_points, dim).astype(np.float32)
# Crear y entrenar el índice Faiss para la búsqueda de vecinos más cercanos
index = faiss.IndexFlatL2(dim)  # Usar L2 para la distancia
# Milvus adds data to the collection in a different way  # Agregar los datos al índice
# Realizar una consulta de ejemplo
query = np.random.rand(dim).astype(np.float32)
k = 5  # Número de vecinos más cercanos a buscar
distances, neighbors = index.search(query.reshape(1, -1), k)
# Mostrar los resultados
print("Índices de los vecinos más cercanos:", neighbors)
print("Distancias de los vecinos más cercanos:", distances)



# Configuraciones
UPLOAD_FOLDER = '/data/uploads/docs/'  # Ajusta esta ruta según sea necesario
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'csv', 'docx', 'xlsx', 'pptx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Esta ruta maneja la subida de archivos y almacena los embeddings en Milvus



def allowed_file(filename, chatbot_id):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Esta función verifica si el archivo tiene una extensión permitida



# Función para verificar si el archivo tiene una extensión permitida
def allowed_file(filename, chatbot_id):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Función para verificar si el archivo tiene una extensión permitida
def allowed_file(filename, chatbot_id):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



#metodo param la subida de documentos

@app.route('/uploads', methods=['POST'])
def procesar_documento():
    mensaje_palabras = ""
    try:
        # Recibiendo el archivo y el chatbot_id desde el formulario
        if 'documento' not in request.files:
            return jsonify({"respuesta": "No se encontró el archivo 'documento'", "codigo_error": 1})
        file = request.files['documento']
        chatbot_id = request.form['chatbot_id']

        if file.filename == '':
            return jsonify({"respuesta": "No se seleccionó ningún archivo", "codigo_error": 1})

        # Guardar el archivo en el sistema de archivos
        chatbot_folder = os.path.join(UPLOAD_FOLDER, str(chatbot_id))
        os.makedirs(chatbot_folder, exist_ok=True)

        destino = os.path.join(chatbot_folder, file.filename)
        file.save(destino)

        # Intentar contar las palabras en el documento
        try:
            with open(destino, 'r', encoding='utf-8') as f:
                contenido = f.read()
                numero_de_palabras = len(contenido.split())
                mensaje_palabras = f"Número de palabras en el documento: {numero_de_palabras}. "
        except Exception as e:
            mensaje_palabras = "No fue posible contar las palabras en el documento. Error: " + str(e)

        # Añadir documento a FAISS y a la base de datos
        try:
            # Aquí, 'documento' podría ser el nombre del archivo o una identificación única derivada de él
            doc_id = file.filename  # O generar un ID único basado en 'file.filename'
            add_document_to_faiss(contenido, doc_id)
        except Exception as e:
            mensaje_palabras += f"No fue posible indexar el documento en FAISS: {e}. "

        mensaje_exito = "Proceso completado con éxito. " + mensaje_palabras
        return jsonify({"respuesta": mensaje_exito, "codigo_error": 0})
    except Exception as e:
        mensaje_error = f"Error: {str(e)}. " + mensaje_palabras
        return jsonify({"respuesta": mensaje_error, "codigo_error": 1})



@app.route('/fine-tuning', methods=['POST'])
def fine_tuning():
    # Obtener los datos del cuerpo de la solicitud
    data = request.json
    training_text = data.get('training_text')
    chat_id = data.get('chat_id')

    # Validación de los datos recibidos
    if not training_text or not isinstance(chat_id, int):
        return jsonify({"status": "error", "message": "Invalid input"}), 400

    # Aquí puedes hacer algo con training_text y chat_id si es necesario

    # Datos para el proceso de fine-tuning
    training_data = {
        # Suponiendo que estos son los datos que OpenAI necesita para el fine-tuning
        "text": training_text,
        "chat_id": chat_id
    }

    # Endpoint y headers para la API de OpenAI
    openai_endpoint = "https://api.openai.com/v1/models/fine-tune"
    headers = {
        "Authorization": "Bearer " + openai.api_key
    }

    # Realizar la solicitud POST a la API de OpenAI
    response = requests.post(openai_endpoint, json=training_data, headers=headers)

    # Manejar la respuesta
    if response.status_code == 200:
        return jsonify({"status": "fine-tuning started", "response": response.json()})
    else:
        return jsonify({"status": "error", "message": response.text}), response.status_code


@app.route('/save_urls', methods=['POST'])
def save_urls(chatbot_id):
    data = request.json
    urls = data.get('urls', [])
    chatbot_id = data.get('chatbot_id')
    if not chatbot_id or len(urls) == 0:
        return jsonify({"status": "error", "message": "No chatbot_id or URLs provided"}), 400
    file_path = os.path.join('uploads/scraping/chatbotid', f'{chatbot_id}.txt')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a') as file:
        for url in urls:
            file.write(url + '\n')
    return jsonify({"status": "success", "message": "URLs saved successfully"})



@app.route('/process_urls', methods=['POST'])
def process_urls():
    data = request.json
    chatbot_id = data.get('chatbot_id')
    if not chatbot_id:
        return jsonify({"status": "error", "message": "No chatbot_id provided"}), 400

    file_path = os.path.join('uploads/scraping/chatbotid', f'{chatbot_id}.txt')
    if not os.path.exists(file_path):
        return jsonify({"status": "error", "message": "File not found"}), 404

    results = []
    with open(file_path, 'r') as file:
        urls = file.readlines()

    for url in urls:
        url = url.strip()
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text().split()
            word_count = len(text)

            # Añadir el contenido de la URL a FAISS y a la base de datos
            # Aquí, el 'url' se usa como identificador único
            try:
                add_document_to_faiss(' '.join(text), url)
                results.append({"url": url, "word_count": word_count, "indexed": True})
            except Exception as e:
                results.append({"url": url, "word_count": word_count, "indexed": False, "index_error": str(e)})

        except requests.RequestException as e:
            results.append({"url": url, "error": str(e)})

    return jsonify({"status": "success", "urls": results})


#Recibimos las urls no validas de front, de cicerone

from flask import Flask, request, jsonify
import os

app = Flask(__name__)

@app.route('/invalid_urls', methods=['POST'])
def handle_invalid_urls():
    data = request.json

    # Parámetros existentes
    urls = data.get('urls', [])
    chatbot_id = data.get('chatbot_id')

    # Nuevos parámetros
    long_text = data.get('long_text')
    chat_id = data.get('chat_id')

    # Verificar si se proporcionaron todos los datos necesarios
    if not chatbot_id or len(urls) == 0 or long_text is None or chat_id is None:
        return jsonify({"status": "error", "message": "Missing required parameters"}), 400

    # Aquí puedes hacer algo con long_text y chat_id

    file_path = os.path.join('uploads/scraping', f'{chatbot_id}.txt')

    # Verificar si el archivo existe
    if not os.path.exists(file_path):
        return jsonify({"status": "error", "message": "File not found"}), 404

    try:
        # Leer las URLs actuales del archivo
        with open(file_path, 'r') as file:
            existing_urls = file.readlines()

        # Filtrar las URLs para eliminar las proporcionadas
        updated_urls = [url for url in existing_urls if url.strip() not in urls]

        # Guardar las URLs actualizadas en el archivo
        with open(file_path, 'w') as file:
            file.writelines(updated_urls)

        return jsonify({
            "status": "success",
            "message": "URLs have been updated",
            "removed_urls": urls
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# Función para eliminar URLs
@app.route('/delete_urls', methods=['POST'])
def delete_urls():
    data = request.json
    urls_to_delete = set(data.get('urls', []))
    chatbot = data.get('chatbot')  # Cambiando de chatbot_id a chatbot
    if not chatbot or not urls_to_delete:
        return jsonify({"status": "error", "message": "No chatbot or URLs provided"}), 400

    file_path = os.path.join('data/uploads/chatbot', 'urls.txt')
    if not os.path.exists(file_path):
        return jsonify({"status": "error", "message": "File not found"}), 404

    try:
        # Leer el contenido actual del archivo
        with open(file_path, 'r') as file:
            existing_urls = set(file.read().splitlines())

        # Filtrar para eliminar las URLs especificadas
        updated_urls = existing_urls - urls_to_delete

        # Guardar el contenido actualizado en el archivo
        with open(file_path, 'w') as file:
            for url in updated_urls:
                file.write(url + '\n')

        return jsonify({"status": "success", "message": "URLs deleted successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500



# Supongamos que ya tenemos un índice FAISS y funciones para generar embeddings y procesar resultados
# index = ... (índice FAISS ya creado)
# generate_embedding = ... (función para generar embeddings a partir de texto)
# process_results = ... (función para procesar los índices de FAISS y obtener información relevante)
@app.route('/ask', methods=['POST'])
def ask():
    try:
        # Recibir la consulta de texto
        query_text = request.json.get('query')

        # Generar un embedding para la consulta usando OpenAI
        query_embedding = generate_embedding(query_text)

        # Realizar la búsqueda en FAISS
        k = 5  # Número de resultados a devolver
        distances, indices = index.search(np.array([query_embedding]).astype(np.float32), k)

        # Procesar los índices para obtener la información correspondiente
        info = process_results(indices)

        # Utilizar OpenAI para generar una respuesta comprensible en español
        response = openai.Completion.create(
            model="text-davinci-003",  # Especifica el modelo de OpenAI a utilizar
            prompt=info,
            max_tokens=150,  # Define el número máximo de tokens en la respuesta
            temperature=0.7,  # Ajusta la creatividad de la respuesta
            top_p=1,  # Controla la diversidad de la respuesta
            frequency_penalty=0,  # Penalización por frecuencia de uso de palabras
            presence_penalty=0,  # Penalización por presencia de palabras
            language="es"  # Especifica el idioma de la respuesta
        )

        # Extraer el texto de la respuesta
        response_text = response.choices[0].text.strip()

        # Devolver la respuesta
        return jsonify({"response": response_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



# Supongamos que estas son tus funciones para generar embeddings y manejar FAISS
def generate_embeddings(data, chatbot_id):
    # Esta función debe generar embeddings para tus datos usando el modelo de OpenAI
    # Aquí hay un placeholder, reemplázalo con tu lógica específica
    return [np.random.random(512) for _ in data]



def update_faiss_index(embeddings, chatbot_id):
    # Esta función debe actualizar el índice de FAISS con nuevos embeddings
    index = faiss.IndexFlatL2(512)  # Suponiendo que usas un índice FlatL2
    index.add(np.array(embeddings).astype(np.float32))
    return index


@app.route('/fine-tuning', methods=['POST'])
def fine_tune_model(chatbot_id):
    training_data = request.json
    # Envío de datos a la API de OpenAI para el proceso de fine-tuning
    openai_endpoint = "https://api.openai.com/v1/models/fine-tune"
    headers = {"Authorization": "Bearer your_api_key_here"}
    response = requests.post(openai_endpoint, json=training_data, headers=headers)
    if response.status_code == 200:
        # Aquí, inicia el proceso de reentrenamiento de FAISS
        embeddings = generate_embeddings(training_data)  # Genera nuevos embeddings
        updated_index = update_faiss_index(embeddings)  # Actualiza el índice de FAISS
        return jsonify({"status": "fine-tuning started", "FAISS index updated": True, "response": response.json()})
    else:
        return jsonify({"status": "error", "message": response.text})



# Código para diagnosticar problemas con FAISS

import faiss

# Verificar la versión de FAISS
print('Versión de FAISS:', faiss.__version__)

# Listar los atributos y métodos de FAISS
print('Atributos y métodos en FAISS:', dir(faiss))

# Si necesitas reinstalar FAISS, descomenta las siguientes líneas:
# !pip uninstall -y faiss faiss-gpu faiss-cpu
# !pip install faiss-cpu  # Para CPU
# !pip install faiss-gpu  # Para GPU


# Suponiendo que tienes un diccionario para mapear IDs de documentos a índices en FAISS
doc_id_to_faiss_index = {}

def add_document_to_faiss(document, doc_id):
    # Verifica si el documento ya está indexado
    if doc_id in doc_id_to_faiss_index:
        print(f"Documento con ID {doc_id} ya indexado.")
        return

    # Genera el embedding para el nuevo documento
    embedding = generate_embedding(document)

    # Añade el embedding al índice de FAISS y actualiza el mapeo
    faiss_index = get_faiss_index()  # Obtén tu índice de FAISS actual
    faiss_index.add(np.array([embedding]))  # Añade el nuevo embedding
    new_index = faiss_index.ntotal - 1  # El nuevo índice en FAISS
    doc_id_to_faiss_index[doc_id] = new_index  # Actualiza el mapeo

    # No olvides guardar el índice de FAISS y el mapeo actualizado de forma persistente


@app.route('/filter_urls', methods=['POST'])
def filter_urls():
    data = request.json
    provided_urls = set(data.get('urls', []))
    chatbot = data.get('chatbot')

    if not chatbot:
        return jsonify({"status": "error", "message": "No chatbot provided"}), 400

    file_path = os.path.join('data/uploads/chatbot', 'urls.txt')
    if not os.path.exists(file_path):
        return jsonify({"status": "error", "message": "File not found"}), 404

    try:
        with open(file_path, 'r') as file:
            existing_urls = set(file.read().splitlines())

        urls_to_keep = existing_urls.intersection(provided_urls)

        with open(file_path, 'w') as file:
            for url in urls_to_keep:
                file.write(url + '\n')  # Corregido aquí

        return jsonify({
            "status": "success",
            "message": "URLs have been filtered",
            "kept_urls": list(urls_to_keep)
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def add_document_to_faiss(text, url):
    # Supongamos que 'faiss_index' es tu índice FAISS y 'doc_id_to_faiss_index' es un diccionario que mapea URLs a índices FAISS
    global faiss_index, doc_id_to_faiss_index

    # Genera el embedding para el nuevo documento
    embedding = generate_embedding(text)

    # Verifica si el documento ya está indexado
    if url in doc_id_to_faiss_index:
        print(f"Documento con URL {url} ya indexado.")
        return

    # Añade el embedding al índice de FAISS y actualiza el mapeo
    faiss_index.add(np.array([embedding]))  # Añade el nuevo embedding
    new_index = faiss_index.ntotal - 1  # El nuevo índice en FAISS
    doc_id_to_faiss_index[url] = new_index  # Actualiza el mapeo

    # No se añade información a la base de datos en esta versión


if __name__ == "__main__":
    app.run(debug=False, port=5000)

