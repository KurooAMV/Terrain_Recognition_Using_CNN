import streamlit as st
import os
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
from PIL import Image
import numpy as np
import pandas as pd

model = load_model("model/terrain_recognition_model.h5")

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
col1, col2 = st.columns(2)

if uploaded_file is not None:
    # st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    image = Image.open(uploaded_file)
    thumbnail = image.copy()
    thumbnail.thumbnail((200, 200)) 
    col1.image(thumbnail, caption="Preview",use_column_width=True)
    processed_img = preprocess_image(image)

    if st.button("Predict"):
        predictions = model.predict(processed_img)
        predicted_class_index = np.argmax(predictions)

        # Define terrain classes in correct order
        terrain_types = ['grassy', 'marshy', 'rocky', 'sandy', 'snowy']
        predicted_terrain = terrain_types[predicted_class_index]
        confidence = np.max(predictions)

        col2.subheader("Predicted Terrain:")
        col2.success(predicted_terrain.capitalize())
        col2.write(f"### Confidence Level: {confidence:.2%}")

        # Show terrain features as table
        terrain_feature_details = terrain_features.get(predicted_terrain, {})
        df = pd.DataFrame(list(terrain_feature_details.items()), columns=["Feature", "Detail"])
        col2.subheader("Terrain Characteristics")
        col2.table(df)
