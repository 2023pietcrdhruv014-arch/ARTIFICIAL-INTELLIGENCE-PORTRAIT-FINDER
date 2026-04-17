import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras import layers, models

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🧠",
    layout="centered"
)

# -------------------------
# Custom CSS (Design)
# -------------------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
    color: white;
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #aaaaaa;
}
.box {
    padding: 20px;
    border-radius: 15px;
    background-color: #1c1f26;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Title Section
# -------------------------
st.markdown('<p class="title">🧠 AI Image Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Detect Real vs AI Generated Images</p>', unsafe_allow_html=True)

# -------------------------
# Build Model
# -------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128,128,3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.load_weights("model.weights.h5")

# -------------------------
# Upload Section
# -------------------------
st.markdown("### 📤 Upload Image")

uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    
    col1, col2 = st.columns(2)

    # -------------------------
    # Show Image
    # -------------------------
    with col1:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

    # -------------------------
    # Prediction
    # -------------------------
    with col2:
        with st.spinner("Analyzing Image... 🔍"):
            img_resized = img.resize((128,128))
            img_array = np.array(img_resized)/255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)[0][0]

        # -------------------------
        # Result Display
        # -------------------------
        if prediction > 0.5:
            st.markdown("### ❌ FAKE IMAGE")
            st.progress(int(prediction * 100))
            st.write(f"Confidence: **{prediction*100:.2f}%**")
        else:
            st.markdown("### ✅ REAL IMAGE")
            st.progress(int((1-prediction) * 100))
            st.write(f"Confidence: **{(1-prediction)*100:.2f}%**")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")