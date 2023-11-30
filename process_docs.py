ALLOWED_EXTENSIONS = {'txt', 'pdf', 'csv', 'docx', 'xlsx', 'pptx'}

def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def read_csv_or_xlsx(file_path, extension):
    df = pd.read_csv(file_path) if extension == 'csv' else pd.read_excel(file_path)
    return df.to_string()

def read_docx(file_path):
    doc = python_docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def read_pptx(file_path):
    ppt = Presentation(file_path)
    text = ""
    for slide in ppt.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += shape.text + "\n"
    return text

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