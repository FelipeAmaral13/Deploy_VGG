import streamlit as st
import os
import numpy as np
from PIL import Image
import cv2

import tensorflow
import keras
from keras.models import Model
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
    Grape Leaves Classification
    </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    model = keras.models.load_model('/app/deploy_vgg/Model/modelo_VGG19_custom.h5')

    
    uploaded_file = st.file_uploader("Upload Files",type=['png','jpeg'])
    if uploaded_file is not None:
        file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
        st.write(file_details)
        img = Image.open(uploaded_file)
        st.image(img, width=250)
        opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) 
        #img_test = cv2.imread(opencvImage, cv2.IMREAD_COLOR)
        resized_image_test= cv2.resize(opencvImage, (224, 224))
        x = np.array(resized_image_test) / 255
        x = x.reshape(-1, 224, 224, 3)
        real_predictions = model.predict(x)
        pred_grape = np.argmax(real_predictions)
        
        st.subheader("Classificação: ")
        
        if pred_grape == 0:
            st.text("AK")
        elif pred_grape == 1:
            st.text("Ala Idris")
        elif pred_grape == 2:
            st.text("Buzgulu")
        elif pred_grape == 3:
            st.text("Dimnit")
        elif pred_grape == 4:
            st.text("Nazli")
        else:
            st.text("Error")


if __name__ == '__main__':
    main()
