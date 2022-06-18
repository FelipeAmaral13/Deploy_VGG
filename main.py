import streamlit as st
import os
import numpy as np
from PIL import Image
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
from tensorflow import keras

st.set_option('deprecation.showPyplotGlobalUse', False)

newpath = r'/app/deploy_vgg/Model' 
if not os.path.exists(newpath):
    os.makedirs(newpath)
try :
    os.rename("/app/deploy_vgg/modelo_VGG19_custom.h5", "/app/deploy_vgg/Model/modelo_VGG19_custom.h5")
except FileNotFoundError:
    pass



def main():
    
    
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">
    Regressao Linear ML App </h2>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg'])
    if uploaded_file is not None:
        file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
        st.write(file_details)
        st.image(Image.open(uploaded_file),width=250)
    #st.text(os.listdir(r'/app/deploy_vgg/Model'))
    #os.path.abspath(r'/app/deploy_vgg/Model/myfile.h5')
    

    #relative = Path('/app/deploy_vgg/Model/myfile.h5')
    #absolute = relative.absolute()  # absolute is a Path object
    #st.text(absolute)
    #from tensorflow import keras
        model = keras.models.load_model('/app/deploy_vgg/Model/modelo_VGG19_custom.h5')
        img_test = cv2.imread(uploaded_file, cv2.IMREAD_COLOR)
        st.write(img_test)


if __name__ == '__main__':
    main()
