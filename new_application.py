import streamlit as st
import gdown
import os
import tensorflow as tf
import keras
from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import pandas as pd

MODEL_PATH = 'model/terrain_recognition_model.h5'
GDRIVE_ID  = "1gHoNs4ulIA8HAd29f3uRSMTOTXLcV2MQ"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_ID}"


def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

# MODEL_PATH = r"C:\Laptop remains\STUTI\Programa\Stock Predictor\Python programs\Projects\Terrain_Recognition\Terrain_Recognition_Using_CNN\terrain_recognition\terrain_recognition_model.h5"
# @st.cache_resource
# def load_model():
#     return keras.models.load_model(MODEL_PATH)

def preprocess_image(pil_image):
    img = pil_image.resize((224, 224)).convert('RGB')
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

terrain_features = {
    'grassy': {
        'Roughness': 'Low',
        'Slipperiness': 'Moderate',
        'Treacherousness': 'Low',
        'Vegetation': 'High',
        'Hydration': 'Moderate',
        'Surface Stability': 'Stable'
    },
    'marshy': {
        'Roughness': 'Moderate',
        'Slipperiness': 'High',
        'Treacherousness': 'High',
        'Vegetation': 'Moderate',
        'Hydration': 'High',
        'Surface Stability': 'Unstable'
    },
    'rocky': {
        'Roughness': 'High',
        'Slipperiness': 'Low',
        'Treacherousness': 'Moderate',
        'Vegetation': 'Low',
        'Hydration': 'Low',
        'Surface Stability': 'Stable'
    },
    'sandy': {
        'Roughness': 'Moderate',
        'Slipperiness': 'Moderate',
        'Treacherousness': 'Low',
        'Vegetation': 'Low',
        'Hydration': 'Low',
        'Surface Stability': 'Stable'
    },
    'snowy': {
        'Roughness': 'High',
        'Slipperiness': 'High',
        'Treacherousness': 'High',
        'Vegetation': 'Low',
        'Hydration': 'Moderate',
        'Surface Stability': 'Unstable'
    }
}

st.title("Terrain Type Recognition using CNN")
uploaded_file = st.file_uploader("Upload a terrain image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    image = Image.open(uploaded_file)
    thumbnail = image.copy()
    thumbnail.thumbnail((200, 200)) 
    st.image(thumbnail, caption="Preview", width=100)
    # image = Image.open(uploaded_file)
    processed_img = preprocess_image(image)

    if st.button("Predict"):
        download_model()
        model = keras.models.load_model(MODEL_PATH)
        predictions = model.predict(processed_img)
        # if you really want to delete the .h5 file afterwards:
        os.remove(MODEL_PATH)
    
        predicted_class_index = np.argmax(predictions)
        terrain_types = ['grassy', 'marshy', 'rocky', 'sandy', 'snowy']
        predicted_terrain = terrain_types[predicted_class_index]
        confidence = np.max(predictions)
    
        st.subheader("Predicted Terrain:")
        st.success(predicted_terrain.capitalize())
        st.write(f"### Confidence Level: {confidence:.2%}")
    
        # Show terrain features as table
        terrain_feature_details = terrain_features[predicted_terrain]
        df = pd.DataFrame(list(terrain_feature_details.items()), columns=["Feature", "Detail"])
        st.subheader("Terrain Characteristics")
        st.table(df)

    
    
    


