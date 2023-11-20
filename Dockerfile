# Usa una imagen base más general de Python
FROM python:3.8

# Establece un directorio de trabajo
WORKDIR /app

# Actualiza pip, instala dependencias esenciales y limpia para reducir el tamaño de la imagen
RUN apt-get update && apt-get install -y     build-essential     libssl-dev     libffi-dev     python3-dev     && pip install --upgrade pip     && apt-get clean     && rm -rf /var/lib/apt/lists/*

# Copia los archivos necesarios al contenedor
COPY chatbotopenai.ipynb /app
COPY requirements.txt /app

# Instala las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Instala pymilvus
RUN pip install --upgrade pymilvus

# Desinstala grpcio y luego instala una versión específica de grpcio

# Expone el puerto en el que se ejecutará la aplicación Flask (modifícalo si usas un puerto diferente)
EXPOSE 5000

# Configura un volumen para almacenamiento externo
VOLUME ["/data"]

# Comando para ejecutar la aplicación
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]