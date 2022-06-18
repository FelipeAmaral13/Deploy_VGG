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
    
#os.listdir(r'/app/deploy_vgg/')


def main():
    
    
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">
    Regressao Linear ML App </h2>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)
    TamCabeca = st.text_input("Qual o volume da cabeça (cm³)?")
    st.text(os.listdir(r'/app/deploy_vgg/')[-1])
    #os.path.abspath(r'/app/deploy_vgg/Model/myfile.h5')
    

    #relative = Path('/app/deploy_vgg/Model/myfile.h5')
    #absolute = relative.absolute()  # absolute is a Path object
    #st.text(absolute)
    #from tensorflow import keras
    #model = keras.models.load_model(absolute)


if __name__ == '__main__':
    main()
