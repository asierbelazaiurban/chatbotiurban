#!/usr/bin/env python
# coding: utf-8


import faiss
import chardet  # Added for encoding detection  # Ensure faiss library is installed
import numpy as np


from flask import Flask, request, jsonify
import numpy as np
import openai
import requests
from bs4 import BeautifulSoup
import faiss
import chardet  # Added for encoding detection
import os
import shutil
import os
import json
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import time


app = Flask(__name__)


# Configura la clave de la API de OpenAI
openai_api_key = os.environ.get('OPENAI_API_KEY')

if openai_api_key is None:
    raise ValueError("La clave API de OpenAI no está definida en las variables de entorno.")


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



def generate_embedding(text, openai_api_key, chatbot_id):
    # Genera un embedding para un texto dado utilizando OpenAI.
    openai.api_key = openai.api_key # Set your OpenAI API key here
    response = openai.Embedding.create(engine="text-similarity-babbage-001", input=text)
    embedding = response['data'][0]['embedding']  # Usando el embedding de OpenAI  # Representación ficticia, reemplazar con la lógica adecuada
    return embedding


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
UPLOAD_FOLDER = 'data/uploads/'  # Ajusta esta ruta según sea necesario
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
def upload_file():
    contenido = ""  # Initialize 'contenido' to an empty string
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
            with open(destino, 'rb') as f:
                raw_data = f.read()
                encoding = chardet.detect(raw_data)['encoding']

            with open(destino, 'r', encoding=encoding) as f:
                contenido = f.read()
                numero_de_palabras = len(contenido.split())
                mensaje_palabras = f"Número de palabras en el documento: {numero_de_palabras}. "
        except Exception as e:
            mensaje_palabras = "No fue posible contar las palabras en el documento. Error: " + str(e)

        # Añadir documento a FAISS
        try:
            # Suponiendo que tienes una función 'obtener_embeddings' que toma el texto y devuelve un vector
            vector_embedding = obtener_embeddings(contenido)  # Necesitas implementar esta función

            # Añadir el embedding al índice de FAISS
            faiss_index.add(np.array([vector_embedding]))

            mensaje_palabras += f"Documento indexado en FAISS. "
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
def save_urls():
    data = request.json
    urls = data.get('urls', [])  # Asumimos que 'urls' es una lista de URLs
    chatbot_id = data.get('chatbot_id')

    if not urls or not chatbot_id:
        return jsonify({"error": "Missing 'urls' or 'chatbot_id'"}), 400

    # Crear o verificar la carpeta específica del chatbot_id
    chatbot_folder = os.path.join('data/uploads/scraping', f'{chatbot_id}')
    os.makedirs(chatbot_folder, exist_ok=True)

    # Crear o agregar a un archivo específico dentro de la carpeta del chatbot
    file_path = os.path.join(chatbot_folder, 'urls.txt')
    with open(file_path, 'a') as file:
        for url in urls:
            file.write(url + '\n')

    return jsonify({"status": "success", "message": "URLs saved successfully"})


app = Flask(__name__)

@app.route('/url_for_scraping', methods=['POST'])
def url_for_scraping():
    try:
        # Obtener URL del request
        data = request.get_json()
        base_url = data.get('url')
        chatbot_id = data.get('chatbot_id')

        if not base_url:
            return jsonify({'error': 'No URL provided'}), 400

        # Modificar la ruta del directorio para guardar los resultados
        save_dir = os.path.join('data/uploads/scraping', f'{chatbot_id}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Función para determinar si la URL pertenece al mismo dominio
        def same_domain(url):
            return urlparse(url).netloc == urlparse(base_url).netloc

        # Hacer scraping y recoger URLs
        try:
            response = requests.get(base_url)
            if response.status_code != 200:
                return jsonify({'error': f'Failed to fetch base URL with status code {response.status_code}'}), 500
            soup = BeautifulSoup(response.content, 'html.parser')
            urls = [urljoin(base_url, tag.get('href')) for tag in soup.find_all('a') if same_domain(urljoin(base_url, tag.get('href')))]
        except Exception as e:
            return jsonify({'error': f'Error during scraping base URL: {str(e)}'}), 500

        # Contar palabras en cada URL y preparar los datos
        urls_data = []
        for url in urls:
            try:
                time.sleep(1)  # Pause to avoid rate limits
                page_response = requests.get(url)
                if page_response.status_code != 200:
                    continue  # Skip URLs that fail to fetch
                page_content = page_response.text
                words = page_content.split()
                word_count = len(words)
                urls_data.append({'url': url, 'word_count': word_count})
            except Exception as e:
                print(f'Error processing URL {url}: {str(e)}')  # Logging the error

        # Guardar los datos en un archivo JSON
        with open(os.path.join(save_dir, 'urls_data.json'), 'w') as json_file:
            json.dump(urls_data, json_file)

        # Guardar solo las URLs en un archivo de texto con el nombre chatbot_id
        with open(os.path.join(save_dir, f'{chatbot_id}.txt'), 'w') as text_file:
            for url_data in urls_data:
                text_file.write(url_data['url'] + '\n')

        # Devolver al front las URLs y el conteo de palabras asociado a cada una
        return jsonify(urls_data)
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500





def process_urls():
    data = request.json
    chatbot_id = data.get('chatbot_id')
    if not chatbot_id:
        return jsonify({"status": "error", "message": "No chatbot_id provided"}), 400

    # Crear el directorio para el chatbot_id si no existe
    chatbot_folder = os.path.join('data/uploads/scraping', f'{chatbot_id}')
    if not os.path.exists(chatbot_folder):
        os.makedirs(chatbot_folder)

    results = []
    with open(os.path.join(chatbot_folder, f'{chatbot_id}.txt'), 'r') as file:
        urls = file.readlines()

    for url in urls:
        url = url.strip()
        try:
            # Extraer el nombre de dominio principal de la URL
            domain = urlparse(url).netloc.split(':')[0]
            domain_file_path = os.path.join(chatbot_folder, f'{domain}.txt')

            # Crear o abrir el archivo del dominio para escritura
            with open(domain_file_path, 'a') as domain_file:
                domain_file.write(url + '\n')

            # Resto del procesamiento
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text().split()
            word_count = len(text)

            try:
                add_document_to_faiss(' '.join(text), url)
                results.append({"url": url, "word_count": word_count, "indexed": True})
            except Exception as e:
                results.append({"url": url, "word_count": word_count, "indexed": False, "index_error": str(e)})

        except requests.RequestException as e:
            results.append({"url": url, "error": str(e)})

    return jsonify({"status": "success", "urls": results})



#Recibimos las urls no validas de front, de cicerone


@app.route('/delete_urls', methods=['POST'])
def delete_urls():
    data = request.json
    urls_to_delete = set(data.get('urls', []))  # Conjunto de URLs a eliminar
    chatbot_id = data.get('chatbot_id')  # Identificador del chatbot

    if not urls_to_delete or not chatbot_id:
        return jsonify({"error": "Missing 'urls' or 'chatbot_id'"}), 400

    # Cambiar la ruta para incluir el chatbot_id
    chatbot_folder = os.path.join('data/uploads/scraping', f'{chatbot_id}')
    if not os.path.exists(chatbot_folder):
        return jsonify({"status": "error", "message": "Chatbot folder not found"}), 404

    file_path = os.path.join(chatbot_folder, 'urls.txt')
    if not os.path.exists(file_path):
        return jsonify({"status": "error", "message": "URLs file not found in chatbot folder"}), 404

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


def obtener_embeddings(texto):
    # Llamada a la API de OpenAI para obtener embeddings
    response = openai.Embedding.create(input=texto, engine="text-similarity-babbage-001")
    # La respuesta incluye los embeddings, que puedes transformar en un array de numpy
    embedding = np.array(response['data'][0]['embedding'])
    return embedding



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
import chardet  # Added for encoding detection

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
    app.run(host='0.0.0.0', port=5000)


