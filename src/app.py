import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# 📄 CONFIGURACIÓN DE LA PÁGINA WEB (La "Vista")
st.set_page_config(
    page_title="Anonimizador IA - Hackaton Treelogic",
    page_icon="🛡️",
    layout="wide"
)


# 🧠 CARGAR EL MODELO (Se usa @st.cache_resource para que cargue una sola vez y la app vaya súper rápida)
@st.cache_resource
def load_model():
    return YOLO("runs/train/weights/best.pt")


try:
    model = load_model()
except Exception as e:
    st.error(
        "❌ No se encontró el modelo en 'runs/train/weights/best.pt'. Asegúrate de haber entrenado el modelo primero.")
    st.stop()


# 🛡️ LA LÓGICA DE MACHINE LEARNING + OPENCV (El "Controlador/Modelo")
def anonymize_image(opencv_img, method="pixelate"):
    # Hacemos una copia para no destruir la original
    img_processed = opencv_img.copy()

    # El modelo YOLO busca el texto médico
    results = model(img_processed, verbose=False)[0]

    # OpenCV aplica el filtro destructivo en las coordenadas encontradas
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        roi = img_processed[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        if method == "Borroso (Blur)":
            img_processed[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (51, 51), 30)
        elif method == "Banda Negra":
            img_processed[y1:y2, x1:x2] = 0
        elif method == "Pixelado":
            small = cv2.resize(roi, (10, 10), interpolation=cv2.INTER_LINEAR)
            img_processed[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

    return img_processed


# 🎨 INTERFAZ GRÁFICA (La "Vista")
st.title("🛡️ Sistema Inteligente de Desidentificación Médica")
st.subheader("Cumplimiento RGPD Automático para Radiografías")
st.markdown("---")

# 🎛️ PANEL LATERAL DE CONFIGURACIÓN
st.sidebar.header("⚙️ Configuración")
metodo_censura = st.sidebar.selectbox(
    "Selecciona el método de anonimización:",
    ["Pixelado", "Borroso (Blur)", "Banda Negra"]
)

# 📂 COMPONENTE PARA SUBIR ARCHIVOS
uploaded_file = st.sidebar.file_uploader(
    "Sube una radiografía (PNG, JPG, JPEG):",
    type=["png", "jpg", "jpeg"]
)

# 🚀 FLUJO PRINCIPAL DE LA APP
if uploaded_file is not None:
    # 1. Convertir el archivo subido a un formato que OpenCV entienda
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # 2. Crear dos columnas en la web para el "Antes" y "Después"
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 Radiografía Original")
        # Streamlit necesita RGB (OpenCV usa BGR por defecto)
        st.image(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col2:
        st.subheader("🔒 Resultado Anonimizado")

        # Estado de carga visual (Spinner)
        with st.spinner("Procesando imagen con YOLOv8..."):
            # 3. Llamamos a nuestra lógica de Machine Learning
            result_img = anonymize_image(opencv_image, method=metodo_censura)

        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), use_container_width=True)

        # 4. Botón de descarga para el usuario
        _, img_encoded = cv2.imencode('.png', result_img)
        st.download_button(
            label="📥 Descargar Radiografía Limpia",
            data=img_encoded.tobytes(),
            file_name="radiografia_anonimizada.png",
            mime="image/png"
        )
else:
    # Mensaje amigable cuando la app está vacía
    st.info("💡 Por favor, sube una radiografía en el panel izquierdo para comenzar la demostración.")