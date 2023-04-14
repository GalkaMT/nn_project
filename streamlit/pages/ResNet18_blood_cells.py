import io 
import streamlit as st
import torch
from torchvision import transforms as T
from PIL import Image

def load_model():
    model = torch.load('model_for_cells_proj.py', map_location = torch.device('cpu'))
    model.eval()
    return model

def load_image():
    uploaded_file = st.file_uploader(label = 'Загрузите изображение клеток крови')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None

def preprocess_image(img):
    preprocess = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()])
    image_for_model = preprocess(img).unsqueeze(0)
    return image_for_model

def print_predict(image_for_model):
    
    cells = {
    0:'EOSINOPHIL',
    1:'LYMPHOCYTE',
    2:'MONOCYTE',
    3:'NEUTROPHIL'    
    }
    prediction = model(image_for_model)
    return cells[prediction.argmax(axis=1)[0].tolist()]

model = load_model()

st.title('Классификация изображений клеток крови (4 класса)')
img = load_image()
result = st.button('Распознать клеточки')
if result:
    x = preprocess_image(img)
    st.write('**Ваша клеточка скорее всего относится к классу:')
    st.write(print_predict(x))
