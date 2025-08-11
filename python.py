#Miguel Angel Jimenez 
#23-EISN-2-010
#Reconocimiento Facial y de Imágenes con IA
import torch 
import torchvision.transforms as transforms 
from torchvision.models import resnet18, ResNet18_Weights  # Importar el modelo ResNet18 y sus pesos preentrenados
from torch.nn.functional import cosine_similarity  # Importar la función de similitud coseno de PyTorch
from PIL import Image  
from openai import OpenAI  # Nueva importación para OpenAI v1.0+
import gradio as gr  
from dotenv import load_dotenv  
import os  # Importar os para interactuar con el sistema operativo

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Configuración del cliente OpenAI (nueva sintaxis)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Modelo preentrenado para extracción de características
pesos = ResNet18_Weights.IMAGENET1K_V1  # Cargar los pesos preentrenados de ResNet18
modelo = resnet18(weights=pesos)  # Inicializar el modelo ResNet18 con los pesos preentrenados
modelo.eval()  # Poner el modelo en modo de evaluación

# Transformaciones de las imágenes
transformacion = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionar la imagen a 224x224 píxeles
    transforms.ToTensor(),  # Convertir la imagen a un tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizar la imagen con los valores de media y desviación estándar de ImageNet
])

# Función para extraer características de una imagen
def extraer_caracteristicas(ruta_imagen):
    imagen = Image.open(ruta_imagen).convert('RGB')  # Abrir la imagen y convertirla a RGB
    imagen = transformacion(imagen).unsqueeze(0)  # Aplicar las transformaciones y añadir una dimensión batch
    with torch.no_grad():  # Desactivar el cálculo de gradientes
        caracteristicas = modelo(imagen)  # Extraer las características de la imagen usando el modelo
    return caracteristicas  # Devolver las características extraídas

# Cálculo de la similitud entre dos imágenes
def comparar_imagenes(ruta_imagen1, ruta_imagen2):
    caracteristicas1 = extraer_caracteristicas(ruta_imagen1)  # Extraer características de la primera imagen
    caracteristicas2 = extraer_caracteristicas(ruta_imagen2)  # Extraer características de la segunda imagen
    similitud = cosine_similarity(caracteristicas1, caracteristicas2)  # Calcular la similitud coseno entre las características
    return similitud.item()  # Devolver la similitud como un valor escalar

# Función para obtener la descripción de GPT-4 en español (nueva sintaxis)
def descripcion_similitud_gpt4(similitud):
    try:
        prompt = f"Describe en español el puntaje de similitud de {similitud:.2f} entre dos imágenes. Explica qué significa este valor de similitud de manera clara y concisa."
        
        # Nueva sintaxis para OpenAI v1.0+
        respuesta = client.chat.completions.create(
            model="gpt-4",  # Usar el modelo GPT-4
            messages=[{"role": "user", "content": prompt}]  # Enviar el prompt como mensaje de usuario
        )
        
        return respuesta.choices[0].message.content  # Devolver la respuesta generada por GPT-4
    
    except Exception as e:
        # En caso de error con la API, devolver una descripción manual
        if similitud >= 0.9:
            return f"Similitud muy alta ({similitud:.2f}): Las imágenes son prácticamente idénticas o muy similares."
        elif similitud >= 0.7:
            return f"Similitud alta ({similitud:.2f}): Las imágenes tienen características muy parecidas."
        elif similitud >= 0.5:
            return f"Similitud moderada ({similitud:.2f}): Las imágenes comparten algunas características similares."
        elif similitud >= 0.3:
            return f"Similitud baja ({similitud:.2f}): Las imágenes tienen pocas características en común."
        else:
            return f"Similitud muy baja ({similitud:.2f}): Las imágenes son muy diferentes entre sí."

# Función principal que procesa las imágenes
def procesar_imagenes(imagen1, imagen2):
    try:
        similitud = comparar_imagenes(imagen1, imagen2)  # Calcular la similitud entre las dos imágenes
        descripcion_gpt = descripcion_similitud_gpt4(similitud)  # Obtener la descripción de GPT-4 basada en la similitud
        return f"Similaridad: {similitud:.2f}\n\nDescripción GPT-4: {descripcion_gpt}"  # Devolver la similitud y la descripción
    except Exception as e:
        return f"Error al procesar las imágenes: {str(e)}"

# Configuración de la interfaz con Gradio
interfaz = gr.Interface(
    fn=procesar_imagenes,  # Función que se ejecutará cuando se suban imágenes
    inputs=[
        gr.Image(type="filepath", label="Primera Imagen"), 
        gr.Image(type="filepath", label="Segunda Imagen")
    ],  # Entradas de la interfaz: dos imágenes
    outputs="text",  # Salida de la interfaz: texto
    title="🔍 Reconocimiento Facial y de Imágenes con IA",  # Título de la interfaz
    description="Sube dos imágenes para comparar su similitud usando ResNet18 y obtener una descripción detallada con GPT-4.",
    examples=None,  # Puedes añadir ejemplos si tienes imágenes de muestra
    theme=gr.themes.Soft()  # Tema visual más atractivo
).launch(share=True)  # Lanzar la interfaz web