import unittest

if __name__ == '__main__':
    tests = unittest.TestLoader().discover('/', pattern='test_*.py')
    unittest.TextTestRunner().run(tests)

#Endpoint /ask

def test_ask_endpoint(self):
    response = self.client.post('/ask', json={
        'chatbot_id': '123',
        'pares_pregunta_respuesta': [
            {'pregunta': '¿Cómo está el tiempo?', 'respuesta': 'Soleado'},
            {'pregunta': '¿Qué hora es?', 'respuesta': ''}
        ]
    })
    self.assertEqual(response.status_code, 200)
    self.assertIn('respuesta', response.json)

#Endpoint /uploads

def test_upload_file_endpoint(self):
    data = {
        'documento': (io.BytesIO(b"contenido del archivo"), 'test.txt'),
        'chatbot_id': '123'
    }
    response = self.client.post('/uploads', data=data, content_type='multipart/form-data')
    self.assertEqual(response.status_code, 200)
    self.assertIn('respuesta', response.json)


#Endpoint /save_text

def test_save_text_endpoint(self):
    response = self.client.post('/save_text', json={'texto': 'Texto de prueba', 'chatbot_id': '123'})
    self.assertEqual(response.status_code, 200)
    self.assertIn('respuesta', response.json)


#Endpoint /process_urls

def test_process_urls_endpoint(self):
    response = self.client.post('/process_urls', json={'chatbot_id': '123'})
    self.assertEqual(response.status_code, 200)
    self.assertIn('status', response.json)


#Endpoint /save_urls

def test_save_urls_endpoint(self):
    response = self.client.post('/save_urls', json={'urls': ['http://example.com'], 'chatbot_id': '123'})
    self.assertEqual(response.status_code, 200)
    self.assertIn('status', response.json)


#url_for_scraping

def test_url_for_scraping_endpoint(self):
    response = self.client.post('/url_for_scraping', json={'url': 'http://example.com', 'chatbot_id': '123'})
    self.assertEqual(response.status_code, 200)
    self.assertIn('error', response.json)


#url_for_scraping_by_sitemap

def test_url_for_scraping_by_sitemap_endpoint(self):
    response = self.client.post('/url_for_scraping_by_sitemap', json={'url': 'http://example.com/sitemap.xml', 'chatbot_id': '123'})
    self.assertEqual(response.status_code, 200)
    self.assertIn('message', response.json)


#delete_urls

def test_delete_urls_endpoint(self):
    response = self.client.post('/delete_urls', json={'urls': ['http://example.com'], 'chatbot_id': '123'})
    self.assertEqual(response.status_code, 200)
    self.assertIn('status', response.json)


#pre_established_answers

def test_pre_established_answers_endpoint(self):
    response = self.client.post('/pre_established_answers', json={'chatbot_id': '123', 'pregunta': '¿Cómo estás?', 'respuesta': 'Bien'})
    self.assertEqual(response.status_code, 200)
    self.assertIn('mensaje', response.json)



#delete_pre_established_answers

def test_delete_pre_established_answers_endpoint(self):
    response = self.client.post('/delete_pre_established_answers', json={'chatbot_id': '123', 'preguntas': ['¿Cómo estás?']})
    self.assertEqual(response.status_code, 200)
    self.assertIn('mensaje', response.json)


#change_params_prompt_temperature_and_model

def test_change_params_endpoint(self):
    response = self.client.post('/change_params_prompt_temperature_and_model', json


#list_chatbot_ids

def test_list_chatbot_ids_endpoint(self):
    response = self.client.get('/list_chatbot_ids')
    self.assertEqual(response.status_code, 200)
    # Aquí puedes añadir más aserciones según la respuesta esperada



