import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

# Load trained model
model = load_model("retinopathy_model.h5")

classes = ["Mild", "Moderate", "No_DR", "Proliferate_DR", "Severe"]

st.title("Diabetic Retinopathy Detection")

uploaded_file = st.file_uploader("Upload a retinal image", type=["jpg","png"])

if uploaded_file is not None:
    
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((224,224))
    img_array = np.array(img)

    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)

    predicted_class = classes[np.argmax(prediction)]

    st.subheader("Prediction:")
    st.write(predicted_class)