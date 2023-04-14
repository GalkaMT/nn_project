import io 
import streamlit as st
import torch
from torchvision import transforms as T
from PIL import Image

def load_model():
    model = torch.load('model_densenet_food.pt', map_location = torch.device('cpu'))
    model.eval()
    return model

def load_image():
    uploaded_file = st.file_uploader(label = 'Загрузите фотографию еды (стритфуд)')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None

def preprocess_image(img):
    preprocess = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()])
    image_for_model = preprocess(img).unsqueeze(0)
    return image_for_model

def print_predict(image_for_model):
    
    food = {
    0:'Baked Potato',
    1:'Burger',
    2:'Crispy Chicken',
    3:'Donut',
    4:'Fries',
    5:'Hot Dog',
    6:'Pizza',
    7:'Sandwich',
    8:'Taco',
    9:'Taquito'
    }
    prediction = model(image_for_model)
    return food[prediction.argmax(axis=1)[0].tolist()], prediction.softmax(1).detach().numpy().max()

model = load_model()

st.title('Классификация изображений стритфуд еды (10 классов)')
img = load_image()
result = st.button('Распознать вкусняшку')
if result:
    x = preprocess_image(img)
    st.write(f'**Ваша вкусняшка: {print_predict(x)[0]} с вероятностью {print_predict(x)[1]:.2%}**')

