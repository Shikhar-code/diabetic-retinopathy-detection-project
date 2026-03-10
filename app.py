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

    prediction = model.predict(img)[0]

    result = {classes[i]: float(prediction[i]) for i in range(len(classes))}

    return result


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Retinal Fundus Image"),
    outputs=gr.Label(num_top_classes=5),
    title="Diabetic Retinopathy Detection",
    description="Upload a retinal fundus image and the model will predict the stage of diabetic retinopathy.",
)

demo.launch()
