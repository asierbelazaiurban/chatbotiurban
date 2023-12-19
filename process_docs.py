##!/usr/bin/env python
# coding: utf-8


from flask import Flask, request, jsonify
import os
import json
import re
import pandas as pd
import docx
from pptx import Presentation
import pdfplumber

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'csv', 'docx', 'xlsx', 'pptx'}

# Funciones para leer diferentes tipos de archivos
def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def read_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ''
    return text

def read_csv_or_xlsx(file_path, extension):
    df = pd.read_csv(file_path) if extension == 'csv' else pd.read_excel(file_path)
    return df.to_string()

def read_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def read_pptx(file_path):
    ppt = Presentation(file_path)
    text = ""
    for slide in ppt.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += shape.text + "\n"
    return text

def clean_and_format_content(content):
    # Eliminar caracteres especiales y números (si es necesario)
    content = re.sub(r'[^A-Za-záéíóúÁÉÍÓÚñÑüÜ.,!? ]', '', content)

    # Espacios adicionales y líneas nuevas
    content = re.sub(r'\s+', ' ', content).strip()

    # (Opcional) aquí puedes añadir más reglas de limpieza/formato según tus necesidades
    return content

def process_file(file_path, extension):
    if extension == 'txt':
        return read_txt(file_path)
    elif extension == 'pdf':
        return read_pdf(file_path)
    elif extension in ['csv', 'xlsx']:
        return read_csv_or_xlsx(file_path, extension)
    elif extension == 'docx':
        return read_docx(file_path)
    elif extension == 'pptx':
        return read_pptx(file_path)
    else:
        return None
        
