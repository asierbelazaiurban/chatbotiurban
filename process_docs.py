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

def clean_and_format_content(content):
    # Eliminar caracteres especiales y números (mantener letras, acentos, números y signos de puntuación básicos)
    content = re.sub(r'[^A-Za-záéíóúÁÉÍÓÚñÑüÜ0-9.,!? ]', '', content)

    # Reemplazar secuencias de espacios, saltos de línea, etc., por un único espacio
    content = re.sub(r'\s+', ' ', content).strip()

    # Corregir espacios antes de signos de puntuación (opcional)
    content = re.sub(r'\s+([.,!?])', r'\1', content)

    # Manejo de codificación
    try:
        content = content.encode('utf-8', 'replace').decode('utf-8', 'replace')
    except UnicodeEncodeError:
        pass

    return content


def read_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
        except UnicodeDecodeError:
            return "Error de lectura del archivo TXT"

def read_pdf(file_path):
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    except Exception as e:
        print(f"Error al procesar PDF: {e}")
        return "Error de lectura del archivo PDF"
    return text

def read_csv_or_xlsx(file_path, extension):
    try:
        if extension == 'csv':
            df = pd.read_csv(file_path, encoding='utf-8', error_bad_lines=False)
        else:
            df = pd.read_excel(file_path)
        return df.to_string()
    except Exception as e:
        print(f"Error al procesar {extension.upper()} archivo: {e}")
        return "Error de lectura del archivo CSV/XLSX"

def read_docx(file_path):
    try:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error al procesar DOCX: {e}")
        return "Error de lectura del archivo DOCX"

def read_pptx(file_path):
    try:
        ppt = Presentation(file_path)
        text = ""
        for slide in ppt.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
        return text
    except Exception as e:
        print(f"Error al procesar PPTX: {e}")
        return "Error de lectura del archivo PPTX"

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
        return "Formato de archivo no soportado"


