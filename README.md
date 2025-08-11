# 🤖 Comparador de Imágenes con IA

**Proyecto Final - Inteligencia Artificial**  
**Estudiante:** Miguel Angel Jimenez  
**Matrícula:** 23-EISN-2-010  
**Asignatura:** Inteligencia Artificial

---

## 📝 Descripción
Sistema que compara dos imágenes y calcula su nivel de similitud usando **ResNet18 (PyTorch)** y genera descripciones en español con **GPT-4 (OpenAI)**.  
Incluye interfaz web con **Gradio** y acceso público.

---

## ✨ Características
- 🔍 **Análisis de Similitud** (Similitud coseno)  
- 🧠 **ResNet18 preentrenado** en ImageNet  
- 🤖 **Descripciones inteligentes** con GPT-4  
- 🌐 **Interfaz web interactiva** con Gradio  

---

## 🛠️ Tecnologías
- **Python:** PyTorch, torchvision, Pillow, openai, gradio, python-dotenv  
- **Modelos:** ResNet18, GPT-4  

---

## 🚀 Instalación
```bash
# 1. Clonar repositorio
git clone https://github.com/usuario/Proyecto-de-Final-IA.git
cd Proyecto-de-Final-IA

# 2. Crear entorno virtual
python -m venv myenv
# Windows
myenv\Scripts\activate
# Linux/Mac
source myenv/bin/activate

# 3. Instalar dependencias
pip install torch torchvision pillow openai gradio python-dotenv

# 4. Configurar API key en .env
OPENAI_API_KEY=tu_clave_api
