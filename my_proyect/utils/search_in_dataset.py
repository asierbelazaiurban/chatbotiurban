import nltk
import os
import unidecode
import openai
import logging
from logging import FileHandler
from nltk.util import ngrams
from flask import Flask
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Luego puedes usar cosine_similarity como se requiera en tu código

nltk.download('popular') 

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

# Configurar la aplicación Flask (si estás usando Flask)

nltk.download('punkt')  # Para tokenización
nltk.download('averaged_perceptron_tagger')  # Para etiquetado POS

# Función auxiliar para mapear etiquetas POS a WordNet POS
def get_wordnet_pos(token):
    tag = nltk.pos_tag([token])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# Codificación TF-IDF
def encode_data(data):
    vectorizer = TfidfVectorizer()
    encoded_data = vectorizer.fit_transform(data)
    return encoded_data, vectorizer

# Procesamiento de consultas de usuario
def preprocess_query(query, n=1):
    tokens = nltk.word_tokenize(query)
    ngrams_list = list(ngrams(tokens, n))
    processed_query = ' '.join([' '.join(grams) for grams in ngrams_list])
    return processed_query.lower()

# Búsqueda de similitud
def perform_search(encoded_data, encoded_query):
    similarity_scores = cosine_similarity(encoded_data, encoded_query)
    ranked_results = np.argsort(similarity_scores, axis=0)[::-1]
    ranked_scores = np.sort(similarity_scores, axis=0)[::-1]
    return ranked_results.flatten(), ranked_scores.flatten()

# Recuperación de resultados
def retrieve_results(data, ranked_results, ranked_scores, context=1, min_words=20):
    results = []
    data_len = len(data)
    unique_results = set()
    for idx, score in zip(ranked_results, ranked_scores):
        start = max(0, idx - context)
        end = min(data_len, idx + context + 1)
        context_data = data[start:end]
        context_str = " ".join(context_data)
        if len(context_str.split()) >= min_words:
            if context_str not in unique_results:
                results.append((context_data, score))
                unique_results.add(context_str)
    return results

# Convertir elemento del dataset a texto
def convertir_a_texto(item):
    if isinstance(item, dict):
        return ' '.join(str(value) for value in item.values())
    elif isinstance(item, list):
        return ' '.join(str(element) for element in item)
    elif isinstance(item, str):
        return item
    else:
        return str(item)

# Cargar dataset
def cargar_dataset(base_dataset_dir, chatbot_id):
    dataset_file_path = os.path.join(base_dataset_dir, str(chatbot_id), 'dataset.json')
    with open(dataset_file_path, 'r') as file:
        data = json.load(file)
    return data

# Encontrar respuesta
def encontrar_respuesta(pregunta, datos_del_dataset, vectorizer, contexto, n=1):
    # Convertir los datos del dataset a texto
    datos = [convertir_a_texto(item['dialogue']) for item in datos_del_dataset.values()]

    # Preprocesar la pregunta
    pregunta_procesada = preprocess_query(pregunta + " " + contexto if contexto else pregunta, n=n)

    # Codificar la pregunta y los datos con el vectorizer
    encoded_query = vectorizer.transform([pregunta_procesada])
    encoded_data = vectorizer.transform(datos)

    # Realizar la búsqueda de similitud
    ranked_results, ranked_scores = perform_search(encoded_data, encoded_query)

    # Recuperar los resultados
    resultados = retrieve_results(datos, ranked_results, ranked_scores)
    app.logger.info("resultados")
    app.logger.info(resultados)

    # Manejar los resultados
    if not resultados:
        # Si no hay resultados, seleccionar una respuesta por defecto
        respuesta_por_defecto = seleccionar_respuesta_por_defecto()
        return traducir_texto_con_openai(respuesta_por_defecto, "Spanish")
    else:
        # Asumiendo que cada elemento en resultados es una tupla (contexto_texto, puntuacion)
        contexto_texto, puntuacion = resultados[0]

        # Verificar que el contexto_texto sea una lista de cadenas de texto
        if isinstance(contexto_texto, list) and all(isinstance(item, str) for item in contexto_texto):
            # Concatenar el texto para formar la respuesta, limitando a 100 palabras
            respuesta_concatenada = ' '.join(contexto_texto)
            palabras_respuesta = respuesta_concatenada.split()[:100]
            contexto_ampliado = ' '.join(palabras_respuesta)
            return contexto_ampliado
        else:
            app.logger.error("La estructura de los resultados no es como se esperaba.")
            return "Ocurrió un error al procesar la respuesta. La estructura de los resultados es incorrecta."

def seleccionar_respuesta_por_defecto():
    # Devuelve una respuesta por defecto de la lista
    return random.choice(respuestas_por_defecto)

def traducir_texto_con_openai(texto, idioma_destino):
    openai.api_key = os.environ.get('OPENAI_API_KEY')
    try:
        prompt = f"Traduce este texto al {idioma_destino}: {texto}"
        response = openai.Completion.create(
            model="gpt-4",
            prompt=prompt,
            max_tokens=60
        )
        return response.choices[0].text.strip()
    except Exception as e:
        app.logger.info(f"Error al traducir texto con OpenAI: {e}")
        return texto  # Devuelve el texto original si falla la traducción

respuestas_por_defecto = [
    "Lamentamos no poder encontrar una respuesta precisa. Para más información, contáctanos en info@iurban.es.",
    "No hemos encontrado una coincidencia exacta, pero estamos aquí para ayudar. Escríbenos a info@iurban.es para más detalles.",
    "Aunque no encontramos una respuesta específica, nuestro equipo está listo para asistirte. Envía tus preguntas a info@iurban.es.",
    "Parece que no tenemos una respuesta directa, pero no te preocupes. Para más asistencia, comunícate con nosotros en info@iurban.es.",
    "No pudimos encontrar una respuesta clara a tu pregunta. Si necesitas más información, contáctanos en info@iurban.es.",
    "Disculpa, no encontramos una respuesta adecuada. Para más consultas, por favor, escribe a info@iurban.es.",
    "Sentimos no poder ofrecerte una respuesta exacta. Para obtener más ayuda, contacta con info@iurban.es.",
    "No hemos podido encontrar una respuesta precisa a tu pregunta. Por favor, contacta con info@iurban.es para más información.",
    "Lo sentimos, no tenemos una respuesta directa a tu consulta. Para más detalles, envía un correo a info@iurban.es.",
    "Nuestra búsqueda no ha dado resultados específicos, pero podemos ayudarte más. Escríbenos a info@iurban.es."
]
