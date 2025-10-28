import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- SETTINGS ----------------
MODEL_PATH = "malaria_model.h5"
IMG_SIZE = 128

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Malaria Detection App",
    page_icon="🩺",
    layout="wide",
)

# ---------------- HEADER ----------------
st.markdown(
    """
    <div style="background: linear-gradient(90deg, #1e3a8a, #0ea5e9); 
                padding:20px; border-radius:12px; color:white;">
        <h1 style="margin-bottom:0;">🧫 Malaria Cell Classifier</h1>
        <p style="margin-top:0;">Developed by Acme Diagnostics | Powered by Deep Learning</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Optional logo (uncomment if you have logo.png)
# st.image("logo.png", width=120)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# ---------------- PREPROCESS FUNCTION ----------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension
    return img_array

# ---------------- MAIN APP ----------------
st.markdown("## 🔍 Upload an image for prediction")

uploaded_file = st.file_uploader("Choose an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼 Uploaded Image", use_column_width=True)
    st.write("")

    with st.spinner("Analyzing image... Please wait ⏳"):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        prob = float(prediction[0][0])

        # interpret result
        label = "🦠 Infected (Malaria)" if prob >= 0.5 else "✅ Uninfected"
        st.subheader(f"Prediction: {label}")
        st.metric(label="Probability (infected)", value=f"{prob*100:.2f}%")

        st.progress(min(max(prob, 0.0), 1.0))
else:
    st.info("Please upload an image to start prediction.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    """
    **About this app:**  
    This app uses a Convolutional Neural Network (CNN) trained on the *Malaria Cell Images Dataset*  
    to classify whether a blood cell is infected with malaria parasites.
    """
)
