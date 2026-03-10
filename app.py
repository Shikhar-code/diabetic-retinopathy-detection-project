import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

model = load_model("retinopathy_model.h5")

classes = ["Mild", "Moderate", "No_DR", "Proliferate_DR", "Severe"]

def predict(img):
    img = img.resize((224,224))
    img = np.array(img)

    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    prediction = model.predict(img)
    return classes[np.argmax(prediction)]

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Diabetic Retinopathy Detection",
    description="Upload a retinal fundus image to predict the DR stage"
)

interface.launch()
