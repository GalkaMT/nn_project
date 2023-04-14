import streamlit as st
import torch
# import torchvision
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torchvision import datasets
from torchvision import transforms as T
from io import BytesIO
# from torchvision import io
# import torchutils as tu
# import json
# import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image
# from io import BytesIO
device ='cuda' if torch.cuda.is_available() else 'cpu'
model = torch.load('model_obj.pt')
model.eval()

def load_image():
    upload_file_img = st.file_uploader(label="Выберите изображение")
    if upload_file_img:
        img_data = upload_file_img.getvalue()
        st.image(img_data)
        img = Image.open(BytesIO(img_data))
        preprocess = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()])
        image_tensor = preprocess(img)
        return image_tensor.to(device)


st.title('Классификация кошечек собачек')
img = load_image()
result = st.button('Кошечка или Собачка?')
if result:
    percent = model(img.to(device).unsqueeze(0)).sigmoid().item()
    cls = 'Это песель' if round(percent) == 1 else 'Это кошечка'
    st.write(cls)
    percent = 1 - percent
    st.write(f'На {percent:.2%}')
#print(img)