    # Intentar traducir la respuesta mejorada
    app.logger.info("pregunta")
    app.logger.info(pregunta)
    app.logger.info("respuesta_mejorada")
    app.logger.info(respuesta_mejorada)
    try:
        respuesta_traducida = openai.ChatCompletion.create(
            model=model_gpt if model_gpt else "gpt-4",
            messages=[
                {"role": "system", "content": f"Responde con no mas de 50 palabras. El idioma original es el de la pregunta:  {pregunta}. Traduce, literalmente {respuesta_mejorada}, al idioma de la pregiunta. Asegurate de que sea una traducción literal.  Si no hubiera que traducirla por que la pregunta: {pregunta} y la respuesta::{respuesta_mejorada}, estan en el mismo idioma devuélvela tal cual, no le añadas ninguna observacion de ningun tipo ni mensaje de error. No agregues comentarios ni observaciones en ningun idioma. Solo la traducción literal o la frase repetida si es el mismo idioma"},                
                {"role": "user", "content": respuesta_mejorada}
            ],
            temperature=float(temperature) if temperature else 0.7
        )
        respuesta_mejorada = respuesta_traducida.choices[0].message['content'].strip()
    except Exception as e:
        app.logger.error(f"Error al traducir la respuesta: {e}")

    app.logger.info("respuesta_mejorada final")
    app.logger.info(respuesta_mejorada)
    return respuesta_mejorada


    @app.route('/delete_dataset_entries', methods=['POST'])
def delete_dataset_entries():
    data = request.json
    urls_to_delete = set(data.get('urls', []))
    chatbot_id = data.get('chatbot_id')

    if not urls_to_delete or not chatbot_id:
        app.logger.warning("Faltan 'urls' o 'chatbot_id' en la solicitud")
        return jsonify({"error": "Missing 'urls' or 'chatbot_id'"}), 400

    BASE_DATASET_DIR = 'data/uploads/datasets'
    dataset_file_path = os.path.join(BASE_DATASET_DIR, str(chatbot_id), 'dataset.json')

    if not os.path.exists(dataset_file_path):
        app.logger.error("Archivo del dataset no encontrado")
        return jsonify({"status": "error", "message": "Dataset file not found"}), 404

    try:
        with open(dataset_file_path, 'r+') as file:
            dataset = json.load(file)
            urls_in_dataset = {entry['url'] for entry in dataset}
            urls_not_found = urls_to_delete - urls_in_dataset

            if urls_not_found:
                app.logger.info(f"URLs no encontradas en el dataset: {urls_not_found}")

            updated_dataset = [entry for entry in dataset if entry['url'] not in urls_to_delete]

            file.seek(0)
            file.truncate()
            json.dump(updated_dataset, file, indent=4)

        app.logger.info("Proceso de eliminación completado con éxito")
        return jsonify({"status": "success", "message": "Dataset entries deleted successfully", "urls_not_found": list(urls_not_found)})
    except json.JSONDecodeError:
        app.logger.error("Error al leer el archivo JSON")
        return jsonify({"status": "error", "message": "Invalid JSON format in dataset file"}), 500
    except Exception as e:
        app.logger.error(f"Error inesperado: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500