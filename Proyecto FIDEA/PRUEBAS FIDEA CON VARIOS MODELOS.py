import os
import io
import base64
import json
from google.cloud import storage
from flask import Flask, render_template, request, jsonify
from google.oauth2 import service_account
from google.cloud import vision
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from datetime import datetime
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer 
import google.generativeai as genai
import logging
import ssl
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import fitz  # PyMuPDF
from PIL import Image
import pytesseract  # Si necesitas extraer texto de imágenes

# Deshabilitar la verificación de SSL solo para desarrollo (NO recomendado para producción)
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)

# Configurar la clave API de Gemini
genai.configure(api_key="AIzaSyBm2rRpjFdvQuwKmzFSJ0CA11_0-f95IrE")
# Cargar el tokenizador y el modelo de DistilBERT
from transformers import BertTokenizer, BertForSequenceClassification

# Utiliza BERT-base
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
# Cargar un pipeline para extraer entidades nombradas o conceptos clave (NER)
modelo_ner = pipeline("ner", model="distilbert-base-uncased")

# Ruta del archivo de credenciales de Google
credentials_path = 'C:/Users/acer/Downloads/archivos ia/credentials.json1.json'
if not os.path.exists(credentials_path):
    raise Exception(f'El archivo de credenciales no se encontró en la ruta: {credentials_path}')

# Configurar las credenciales de Google Cloud
credentials = service_account.Credentials.from_service_account_file(credentials_path)
vision_client = vision.ImageAnnotatorClient(credentials=credentials)
drive_service = build('drive', 'v3', credentials=credentials)

# Cargar el modelo preentrenado RoBERTa para clasificación de cero disparos (zero-shot)
classifier = pipeline("zero-shot-classification", model="roberta-large-mnli")

# Definir las temáticas personalizadas
tematicas_personalizadas = [
    "Matemáticas", "Física", "Química", "Biología", "Geometría", "Álgebra", "Estadística",
    "Cálculo diferencial", "Cálculo integral", "Probabilidad", "Programación", "Machine Learning",
    "Inteligencia Artificial", "Análisis de datos", "Desarrollo web", "Ciberseguridad",
    "Sistemas operativos", "Redes informáticas", "Criptografía", "Gestión de bases de datos",
    "Big Data", "Robótica", "Nanotecnología", "Ingeniería genética", "Bioinformática",
    "Electrónica", "Ingeniería civil", "Arquitectura", "Urbanismo", "Energías renovables",
    "Energía solar", "Energía eólica", "Astrofísica", "Cosmología", "Astronomía", "Geología",
    "Ecología", "Ciencias ambientales", "Meteorología", "Oceanografía", "Psicología",
    "Psiquiatría", "Sociología", "Antropología", "Filosofía", "Historia", "Arqueología",
    "Lingüística", "Literatura", "Crítica literaria", "Estudios culturales", "Estudios de género",
    "Economía", "Microeconomía", "Macroeconomía", "Econometría", "Finanzas", "Contabilidad",
    "Marketing", "Gestión de recursos humanos", "Gestión de operaciones", "Liderazgo",
    "Emprendimiento", "Innovación", "Planificación estratégica", "Derecho", "Derecho internacional",
    "Derecho penal", "Derecho civil", "Derecho ambiental", "Medicina", "Cirugía", "Odontología",
    "Veterinaria", "Enfermería", "Fisioterapia", "Terapia ocupacional", "Nutrición",
    "Educación física", "Deportes", "Música", "Composición musical", "Teoría musical",
    "Interpretación musical", "Producción musical", "Artes visuales", "Pintura", "Escultura",
    "Fotografía", "Cine", "Animación", "Diseño gráfico", "Diseño industrial", "Moda",
    "Publicidad", "Comunicación", "Periodismo", "Relaciones públicas", "Ética", "Moral"
]
# Función para clasificar el texto en una temática personalizada
def clasificar_tema(texto):
    if not texto.strip():
        raise Exception("El texto para clasificar está vacío.")
    
    if not tematicas_personalizadas:
        raise Exception("No hay etiquetas disponibles para la clasificación.")

    try:
        resultado = classifier(texto, tematicas_personalizadas)
        print("Resultado de RoBERTa:", resultado)  # Debug: Mostrar resultado de RoBERTa
        return resultado['labels'][0], resultado['scores'][0]
    except Exception as e:
        raise Exception(f"Error en la clasificación del texto: {str(e)}")

# Función para clasificar usando Gemini (tema y resumen)
def clasificar_con_gemini(texto):
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    response = model.generate_content(
        f"Por favor, dame el tema especifico, resume el contenido del examen y haz una retroalimentacion pequeña en español. cada seccion debe estar separada y no quiero titulos repetidos : {texto}"
    )
    
    content = response.text.strip()
    print("Respuesta de Gemini:", content)  # Debug: Mostrar respuesta de Gemini
    return content  # Devuelve solo el contenido (resumen)

# Función para analizar el texto y proporcionar retroalimentación
def analizar_respuesta(texto):
    # Tokenizar el texto
    inputs = tokenizer(texto, return_tensors='pt', truncation=True, padding=True)
    
    # Realizar la inferencia
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Obtener las predicciones
    predicciones = torch.argmax(logits, dim=-1).item()  # Convertir a número entero
    
    # Proporcionar retroalimentación básica
    retroalimentacion = {
        0: "La respuesta es incompleta. Asegúrate de incluir todos los puntos clave.",
        1: "La respuesta es correcta, pero podría beneficiarse de ejemplos adicionales.",
        2: "La respuesta es muy completa. ¡Buen trabajo!"
    }

    # Obtener la retroalimentación basada en la predicción
    feedback = retroalimentacion.get(predicciones, "No se pudo determinar la retroalimentación.")
    
    return feedback


# Función para extraer conceptos clave usando NER
def extraer_conceptos_clave(texto_respuesta):
    """
    
    """
    entidades = modelo_ner(texto_respuesta)
    conceptos_clave = {ent['word'] for ent in entidades}  # Usamos un set para evitar duplicados
    return conceptos_clave

# Función para analizar vacíos conceptuales basándose en los conceptos
def analizar_vacios_conceptuales(texto_respuesta, conceptos_identificados):
    """
    Analiza si hay vacíos conceptuales en la respuesta del estudiante
    basándose en los conceptos clave extraídos.
    """
    vacios = []
    
    # Identificar si algunos conceptos clave no están presentes en la respuesta
    if len(conceptos_identificados) < 5:  # Número arbitrario para este ejemplo
        vacios.append("Falta profundizar en más conceptos clave.")

    feedback = "La respuesta incluye algunos conceptos clave, pero se recomienda mejorar los siguientes aspectos:\n"
    feedback += ", ".join(vacios) if vacios else "No se han identificado vacíos conceptuales significativos."
    
    return feedback

def generar_retroalimentacion(texto_respuesta):
    """f
    Genera retroalimentación automática sobre los vacíos conceptuales en la respuesta.
    """
    # Extraer los conceptos clave de la respuesta
    conceptos_identificados = extraer_conceptos_clave(texto_respuesta)
    
    # Analizar vacíos conceptuales basándose en los conceptos extraídos
    retroalimentacion = analizar_vacios_conceptuales(texto_respuesta, conceptos_identificados)
    
    return retroalimentacion
# ID del folder en Google Drive donde se subirán los archivos
DRIVE_FOLDER_ID = '1QuOJHha_SDZThBqsUAFo3ALvGhWK5loh'

# Función para convertir una imagen a texto usando Google Cloud Vision API
def convertir_foto_a_texto(image_bytes):
    image = vision.Image(content=image_bytes)
    response = vision_client.text_detection(image=image)

    if response.error.message:
        raise Exception(f'Error en la API de Vision: {response.error.message}')

    texts = response.text_annotations
    texto_extraido = texts[0].description if texts else ''
    print("Texto extraído de la imagen:", texto_extraido)  # Debug: Mostrar texto extraído
    return texto_extraido
def convertir_pdf_a_texto_con_vision(pdf_bytes):
    """
    Usa Google Vision API para extraer texto de un archivo PDF cargado al bucket.
    """
    try:
        from google.cloud import storage  # Importar librería para subir al bucket
        from google.cloud.vision_v1 import types  # Asegurar que usamos Vision v1
        BUCKET_NAME = "example-bucket-142"

        # Subir PDF al bucket temporalmente
        storage_client = storage.Client(credentials=credentials)
        bucket = storage_client.bucket(BUCKET_NAME)
        blob_name = f"temp_pdf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        blob = bucket.blob(blob_name)
        blob.upload_from_string(pdf_bytes, content_type="application/pdf")

        # Configurar fuente GCS para Vision API
        gcs_source = {"uri": f"gs://{BUCKET_NAME}/{blob_name}"}
        input_config = types.InputConfig(
            gcs_source=types.GcsSource(uri=gcs_source["uri"]),
            mime_type="application/pdf"
        )

        # Solicitar a Vision API
        feature = types.Feature(type_=types.Feature.Type.DOCUMENT_TEXT_DETECTION)
        request = types.AnnotateFileRequest(
            input_config=input_config,
            features=[feature]
        )

        # Enviar solicitud de batch
        response = vision_client.batch_annotate_files(requests=[request])
        texto_extraido = ""

        # Procesar la respuesta
        for file_response in response.responses:
            if file_response.error.message:
                raise Exception(f"Vision API Error: {file_response.error.message}")
            for page_response in file_response.responses:
                texto_extraido += page_response.full_text_annotation.text

        # Eliminar el archivo temporal del bucket
        blob.delete()

        return texto_extraido.strip()

    except Exception as e:
        raise Exception(f"Error al extraer texto del PDF con Google Vision: {str(e)}")


# Función para subir un archivo a Google Drive
def subir_archivo_a_drive(nombre_archivo, contenido, mimetype):
    archivo_bytes = io.BytesIO(contenido.encode('utf-8') if mimetype == 'text/plain' else contenido)
    file_metadata = {
        'name': nombre_archivo,
        'parents': [DRIVE_FOLDER_ID]
    }
    media = MediaIoBaseUpload(archivo_bytes, mimetype=mimetype)

    try:
        uploaded_file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        print(f'Archivo subido a Drive: {uploaded_file.get("id")}')  # Debug: Confirmar subida
        return uploaded_file.get("id")
    except Exception as e:
        raise Exception(f'Error al subir el archivo a Google Drive: {str(e)}')

# Variable global para el contador
contador_archivos = 1  # Se puede inicializar desde 1 o cargar desde un archivo externo

# Función para generar un nombre de archivo secuencial
def generar_nombre_archivo_secuencial():
    global contador_archivos
    nombre_archivo = f'Analisis Fidea #{contador_archivos} - {datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    contador_archivos += 1  # Incrementar el contador para el próximo archivo
    return nombre_archivo

# Función para guardar el texto en Google Drive con el nombre secuencial
def guardar_texto_en_drive(texto):
    file_name = generar_nombre_archivo_secuencial()  # Usar la nueva función para generar el nombre
    file_metadata = {
        'name': file_name,
        'parents': [DRIVE_FOLDER_ID],
        'mimeType': 'text/plain'
    }

    media = MediaIoBaseUpload(io.BytesIO(texto.encode('utf-8')), mimetype='text/plain')

    try:
        uploaded_file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        print(f'Archivo de texto guardado en Drive: {uploaded_file.get("id")}')  # Debug: Confirmar subida
        return uploaded_file.get('id')
    except Exception as e:
        raise Exception(f'Error al subir el archivo de texto a Drive: {str(e)}')

# Configurar logging
logging.basicConfig(level=logging.DEBUG)

# Ruta para la página principal
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint para subir imagen capturada desde la cámara
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.form:
        return jsonify({'error': 'No se recibió ninguna imagen'}), 400

    image_data = request.form['image']

    if image_data.startswith('data:image/jpeg;base64,'):
        image_data = image_data.split(',')[1]

    try:
        image_bytes = base64.b64decode(image_data)

        nombre_archivo = f'captured_image_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
        file_metadata = {
            'name': nombre_archivo,
            'parents': [DRIVE_FOLDER_ID]
        }
        media = MediaIoBaseUpload(io.BytesIO(image_bytes), mimetype='image/jpeg')

        uploaded_file1 = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        app.logger.debug(f'Archivo de imagen subido: {uploaded_file1.get("id")}')

        # Extraer texto de la imagen
        texto_extraido = convertir_foto_a_texto(image_bytes)

        # Clasificar el texto usando las temáticas personalizadas
        tema_roberta, score_roberta = clasificar_tema(texto_extraido)
        resumen_gemini = clasificar_con_gemini(texto_extraido)  # Solo se guarda el resumen
         # Analizar respuesta y proporcionar retroalimentación
        feedback = analizar_respuesta(texto_extraido)
        vacios_feedback = generar_retroalimentacion(texto_extraido)

        # Guardar el texto y el tema clasificado en Google Drive
        texto_final = (f"{texto_extraido}\n\n"
               f"Tema Detectado por RoBERTa: {tema_roberta} (Score: {score_roberta:.2f})\n"
               f"Resumen Detectado por Gemini: {resumen_gemini}\n"
               f"feedback: {feedback}\n"
               f"Vacios Conceptuales: {vacios_feedback}\n")  # Llama a la función para obtener vacíos conceptuales  
                       
                      

        archivo_texto_id = guardar_texto_en_drive(texto_final)

        return jsonify({
            'message': 'Imagen procesada exitosamente',
            'file_id': uploaded_file.get('id'),
            'archivo_texto_id': archivo_texto_id,
            'texto_extraido': texto_final,
            'tema_roberta': tema_roberta,
            'resumen_gemini': resumen_gemini,
            'feedback': feedback,
            'vacios_feedback': vacios_feedback 

            
           # Incluir la retroalimentación en la respuesta JSON
        }), 200

    except Exception as e:
        app.logger.error(f'Error al procesar la imagen: {str(e)}')
        return jsonify({'error': f'Error al procesar la imagen: {str(e)}'}), 500

def extraer_texto_de_pdf(pdf_bytes):
    texto = ""
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
            for pagina in pdf:
                texto += pagina.get_text()
    except Exception as e:
        raise Exception(f"Error al procesar el PDF: {str(e)}")
    return texto

# Endpoint para subir archivos desde el computador
@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No se recibió ningún archivo'}), 400

    file = request.files['file']

    try:
        file_bytes = io.BytesIO(file.read())
        app.logger.debug(f"Procesando archivo: {file.filename}, Tipo: {file.content_type}")

        # Subir archivo a Google Drive
        file_metadata = {
            'name': file.filename,
            'parents': [DRIVE_FOLDER_ID]
        }
        media = MediaIoBaseUpload(file_bytes, mimetype=file.content_type)

        uploaded_file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        app.logger.debug(f'Archivo subido a Google Drive: {uploaded_file.get("id")}')

        # Reiniciar el puntero del archivo
        file_bytes.seek(0)

        # Procesar el archivo
        texto_extraido = ""
        if file.content_type.startswith('image/'):
            app.logger.debug("El archivo es una imagen, extrayendo texto...")
            texto_extraido = convertir_foto_a_texto(file_bytes.getvalue())
        elif file.content_type == 'application/pdf':
          texto_extraido = convertir_pdf_a_texto_con_vision(file_bytes.getvalue())
          if not texto_extraido.strip():
            raise Exception("El PDF no contiene texto reconocible.")

       

        # Clasificar el texto usando las temáticas personalizadas
        tema_roberta, score_roberta = clasificar_tema(texto_extraido)
        resumen_gemini = clasificar_con_gemini(texto_extraido)  # Solo se guarda el resumen

        # Generar retroalimentación
        feedback = analizar_respuesta(texto_extraido)
        vacios_feedback = generar_retroalimentacion(texto_extraido)

        # Guardar el texto en Google Drive
        texto_final = (f"{texto_extraido}\n\n"
                       f"Tema Detectado por RoBERTa: {tema_roberta} (Score: {score_roberta:.2f})\n"
                       f"Resumen Detectado por Gemini: {resumen_gemini}\n"
                       f"Feedback: {feedback}\n"
                       f"Vacios Conceptuales: {vacios_feedback}\n")

        archivo_texto_id = guardar_texto_en_drive(texto_final)

        return jsonify({
            'message': 'Archivo procesado exitosamente',
            'file_id': uploaded_file.get('id'),
            'archivo_texto_id': archivo_texto_id,
            'texto_extraido': texto_final,
            'tema_roberta': tema_roberta,
            'resumen_gemini': resumen_gemini,
            'feedback': feedback,
            'vacios_feedback': vacios_feedback
        }), 200

    except Exception as e:
        app.logger.error(f'Error al procesar el archivo: {str(e)}')
        return jsonify({'error': f'Error al procesar el archivo: {str(e)}'}), 500
     

if __name__ == '__main__':
    app.run(debug=True)






