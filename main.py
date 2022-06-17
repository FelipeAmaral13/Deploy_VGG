import streamlit as st
import os
import numpy as np
import cv2
import random
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

import tensorflow
import keras
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from keras.applications.vgg19 import VGG19
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

st.set_option('deprecation.showPyplotGlobalUse', False)

newpath = r'/app/deploy_vgg/Model' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

import requests
from pathlib import Path

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

#https://drive.google.com/file/d/19Rg9Ki7-AD1rCcrbGIjgZc2__5OeaTX2/view?usp=sharing

def main():
    
    file_id = '19Rg9Ki7-AD1rCcrbGIjgZc2__5OeaTX2'
    destination = r'/app/deploy_vgg/Model/myfile.h5'
    download_file_from_google_drive(file_id, destination)
    
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">
    Regressao Linear ML App </h2>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)
    TamCabeca = st.text_input("Qual o volume da cabeça (cm³)?")
    #st.text(os.listdir(r'/app/deploy_vgg/Model'))
    #os.path.abspath(r'/app/deploy_vgg/Model/myfile.h5')
    

    relative = Path('/app/deploy_vgg/Model/myfile.h5')
    absolute = relative.absolute()  # absolute is a Path object
    st.text(absolute)
    model = keras.models.load_model(absolute)


if __name__ == '__main__':
    main()
