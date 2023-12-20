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

import re


# Función para leer y limpiar archivos TXT
def read_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            return clean_and_format_content(content)
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                content = file.read()
                return clean_and_format_content(content)
        except UnicodeDecodeError:
            return "Error de lectura del archivo TXT"

# Función para leer y limpiar archivos PDF
def read_pdf(file_path):
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                except UnicodeDecodeError:
                    text += " [Error en la decodificación de esta página] "
        return clean_and_format_content(text)
    except Exception as e:
        return f"Error al procesar PDF: {e}"

# Función para leer y limpiar archivos CSV o XLSX
def read_csv_or_xlsx(file_path, extension):
    try:
        if extension == 'csv':
            df = pd.read_csv(file_path, encoding='utf-8', error_bad_lines=False)
        else:
            df = pd.read_excel(file_path)
        content = df.to_string()
        return clean_and_format_content(content)
    except Exception as e:
        return f"Error al procesar {extension.upper()} archivo: {e}"

# Función para leer y limpiar archivos DOCX
def read_docx(file_path):
    try:
        doc = docx.Document(file_path)
        content = "\n".join([para.text for para in doc.paragraphs])
        return clean_and_format_content(content)
    except Exception as e:
        return f"Error al procesar DOCX: {e}"

# Función para leer y limpiar archivos PPTX
def read_pptx(file_path):
    try:
        ppt = Presentation(file_path)
        text = ""
        for slide in ppt.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
        return clean_and_format_content(text)
    except Exception as e:
        return f"Error al procesar PPTX: {e}"

# Función para procesar el archivo según su extensión
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

