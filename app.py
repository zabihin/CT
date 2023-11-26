import streamlit as st
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import tensorflow as tf
import numpy as np
import hydralit_components as hc
import webbrowser
import io



 ##SETTING PAGE CONFIG TO WIDE MODE AND ADDING A TITLE AND FAVICON

st.set_page_config(layout="wide", page_title="Lung Cancer Diagnosis)",page_icon="⚕️")
st.markdown("<h1 style='text-align:center; color: red;'> Lung Cancer Diagnosis-LCD (from Chest CT SCAN )</h1>", unsafe_allow_html=True)
st.sidebar.title("What is LCD ?")


st.sidebar.write("LCD (Lung Cancer Diagnosis) is an end-to-end **CNN Image Classification Model** which can diagnose lung cancer and it's type from chest CT scan image. ")
st.sidebar.info (" **Sensitivity  :** **`99.4%`**")
st.sidebar.info (" **Specificity  :** **`96.3%`**")

st.sidebar.info (" **Accuracy :** **`80%`**")



st.sidebar.subheader("About App")

st.sidebar.info("Created by Zahra Zabihinpour as a final project in Data Fullstack (Jedha Bootcamp)")
st.sidebar.image("./jedha.png", width=100, use_column_width=100)

def diagnose(im):
    model = tf.keras.models.load_model("CT5-2.h5")

    with hc.HyLoader('Analysing the image, Wait ...', hc.Loaders.standard_loaders, index=3):
        preds=np.argmax( model.predict(im))
    return preds

label=['Adenocarcinoma','Large cell carcinoma','Normal', 'Squamous cell carcinoma']

uploaded_file = st.file_uploader(" Upload a chest CT scan image: ", type=["png","jpg","jpeg"])
if not uploaded_file:
    st.warning("Please upload an image")
    st.stop()

else:
    image = uploaded_file.read()
    st.image(image, width=400, use_column_width=400)
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        rgb_im = img.convert('RGB')
        img_arr = tf.keras.preprocessing.image.img_to_array(rgb_im) / 255
        resized_img = tf.image.resize(img_arr, (350, 350))
        img_array = np.array(resized_img).reshape(1,350,350,3)

        st.subheader("The Result is:")
        pred=diagnose(img_array)

        if pred==2:
            result="Normal (Healthy Lung)"
        elif pred==0:
            result="Cancer(Adenocarcinoma)"
        elif pred==1:
            result="Cancer (Large cell carcinoma)"
        else:
            result="Cancer (Squamous cell carcinoma)"


        st.subheader(result)
            
        





if st.button('For more information click here'):

    webbrowser.open_new_tab('https://www.cancer.org/cancer/lung-cancer/about/key-statistics.html') 
