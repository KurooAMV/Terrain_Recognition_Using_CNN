import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tempfile
# from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from flask import Flask, request, render_template
import gdown

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load the trained model
MODEL_PATH = 'model/terrain_recognition_model.h5'
GDRIVE_ID = '1gHoNs4ulIA8HAd29f3uRSMTOTXLcV2MQ'
GDRIVE_URL = f'https://drive.google.com/uc?id={GDRIVE_ID}'

def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        # print("Downloading model from Google Drive...")
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    else:
        return

download_model()
model = tf.keras.models.load_model(MODEL_PATH)

# Define a function to preprocess uploaded images
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))  # Adjust target_size if needed
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return img

# Define terrain features for different terrain types
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

# Define a route to render the upload form
@app.route('/')
def upload_form():
    return render_template('upload.html')

# Define a route to handle image uploads and predictions
@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file:
        uploaded_file.save('/tmp/' + uploaded_file.filename)
    
    if uploaded_file.filename != '':
        # Save the uploaded image to a temporary directory
        # image_path = os.path.join('temp', uploaded_file.filename)
        # # filename = secure_filename(uploaded_file.filename)
        # temp_dir = '/tmp'
        # image_path = os.path.join(temp_dir, filename)
        # uploaded_file.save(image_path)
        
        # Preprocess the uploaded image
        img = preprocess_image(image_path)
        
        # Make predictions on the uploaded image
        predictions = model.predict(img)
        
        # Get the predicted class (terrain type)
        predicted_class_index = np.argmax(predictions)
        
        # Map the class index to the actual terrain type (you should define this mapping)
        terrain_types = ['grassy', 'marshy', 'rocky', 'sandy', 'snowy']  # Replace with your terrain type labels
        predicted_terrain = terrain_types[predicted_class_index]
        
        # Get terrain features based on predicted terrain type
        terrain_feature_details = terrain_features.get(predicted_terrain, {})
        
        # Remove the temporary image file
        os.remove(image_path)
        
        return render_template('upload.html', 
                               prediction=f'Predicted Terrain Type: {predicted_terrain}',
                               terrain_details=terrain_feature_details)
    else:
        return render_template('upload.html', prediction='No file uploaded')

if __name__ == '__main__':
    # Create the 'temp' directory if it doesn't exist
    os.makedirs('temp', exist_ok=True)
    
    app.run(debug=True)
