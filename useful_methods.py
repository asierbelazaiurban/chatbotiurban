##!/usr/bin/env python
# coding: utf-8

def allowed_file(filename):
    # Retorna verdadero si el archivo tiene una extensión permitida
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_urls(chatbot_folder, chatbot_id):
    # Lee las URLs de un archivo y retorna una lista
    urls_file_path = os.path.join(chatbot_folder, f'{chatbot_id}.txt')
    try:
        with open(urls_file_path, 'r') as file:
            return [url.strip() for url in file.readlines()]
    except FileNotFoundError:
        app.logger.error(f"Archivo de URLs no encontrado para el chatbot_id {chatbot_id}")
        return []

def safe_request(url, max_retries=3):
    # Realiza una solicitud segura a la URL dada
    headers = {'User-Agent': 'Mozilla/5.0...'}
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                return response
        except requests.RequestException as e:
            app.logger.error(f"Attempt {attempt + 1} failed for {url}: {e}")
    return None

def procesar_pregunta(pregunta_usuario, preguntas_palabras_clave):
    # Procesa la pregunta del usuario y encuentra la mejor coincidencia
    palabras_pregunta_usuario = set(word_tokenize(pregunta_usuario.lower()))
    stopwords_ = set(stopwords.words('spanish'))
    palabras_relevantes_usuario = palabras_pregunta_usuario - stopwords_

    respuesta_mas_adeacuada, max_coincidencias = None, 0
    for pregunta, datos in preguntas_palabras_clave.items():
        palabras_clave = set(datos['palabras_clave'])
        coincidencias = palabras_relevantes_usuario.intersection(palabras_clave)
        if len(coincidencias) > max_coincidencias:
            max_coincidencias, respuesta_mas_adeacuada = len(coincidencias), datos['respuesta']

    return respuesta_mas_adeacuada

def mejorar_respuesta_con_openai(respuesta_original, pregunta, tipo='general'):
    # Mejora la respuesta utilizando OpenAI GPT-3
    openai.api_key = os.environ.get('OPENAI_API_KEY')
    prompt = construir_prompt(pregunta, respuesta_original, tipo)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=[{"role": "system", "content": "Mejora las respuestas"}, {"role": "user", "content": prompt}]
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        app.logger.error(f"Error al interactuar con OpenAI: {e}")
        return None

def construir_prompt(pregunta, respuesta_original, tipo):
    # Construye un prompt para OpenAI GPT-3 basado en el tipo de respuesta
    if tipo == 'turismo':
        return f"La pregunta es: {pregunta}\nLa respuesta original es: {respuesta_original}\nResponde como un guía turístico..."
    else:
        return f"Cuando recibas una pregunta, comienza con: '{pregunta}'..."

def extraer_palabras_clave(pregunta):
    # Extrae palabras clave de una pregunta
    palabras = word_tokenize(pregunta)
    palabras_filtradas = [palabra for palabra in palabras if palabra.isalnum()]
    stop_words = set(stopwords.words('spanish'))
    return [palabra for palabra in palabras_filtradas if palabra not in stop_words]


def convertir_a_texto(item):
    """
    Convierte un elemento de dataset en una cadena de texto.
    Esta función asume que el 'item' puede ser un diccionario, una lista, o un texto simple.
    """
    if isinstance(item, dict):
        # Concatena los valores del diccionario si 'item' es un diccionario
        return ' '.join(str(value) for value in item.values())
    elif isinstance(item, list):
        # Concatena los elementos de la lista si 'item' es una lista
        return ' '.join(str(element) for element in item)
    elif isinstance(item, str):
        # Devuelve el string si 'item' ya es una cadena de texto
        return item
    else:
        # Convierte el 'item' a cadena si es de otro tipo de dato
        return str(item)


def cargar_dataset(chatbot_id, base_dataset_dir):
    dataset_file_path = os.path.join(base_dataset_dir, str(chatbot_id), 'dataset.json')
    app.logger.info(f"Dataset con ruta {dataset_file_path}")

    try:
        with open(dataset_file_path, 'r') as file:
            data = json.load(file)
            app.logger.info(f"Dataset cargado con éxito desde {dataset_file_path}")
            return [convertir_a_texto(item) for item in data.values()]
    except Exception as e:
        app.logger.error(f"Error al cargar el dataset: {e}")
        return []

def encode_data(data):
    vectorizer = TfidfVectorizer()
    encoded_data = vectorizer.fit_transform(data)
    return encoded_data, vectorizer

def preprocess_query(query):
    tokens = word_tokenize(query.lower())
    processed_query = ' '.join(tokens)
    return processed_query
