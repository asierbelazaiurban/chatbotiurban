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