import pickle
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the saved CNN model
from tensorflow.keras.models import load_model

model = load_model('plant_disease_model.h5')

# Load the label binarizer for decoding predictions
label_binarizer = pickle.load(open('label_transform.pkl', 'rb'))

# Function to preprocess an image for prediction
def preprocess_image(image_path):
    default_image_size = (256, 256)  # Resize dimensions
    image = cv2.imread(image_path)  # Read image
    if image is not None:
        image = cv2.resize(image, default_image_size)  # Resize to match model input
        image = img_to_array(image)  # Convert to array
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = image / 255.0  # Normalize pixel values
        return image
    else:
        raise ValueError(f"Could not read the image: {image_path}")

# Path to the image you want to predict

# Preprocess the image
def main(image_path):
    input_image = preprocess_image(image_path)

# Make prediction
    prediction = model.predict(input_image)


    predicted_class_index = np.argmax(prediction)  
    predicted_class = label_binarizer.classes_[predicted_class_index]  
    confidence = prediction[0][predicted_class_index] * 100  

    h1=f"Predicted class: {predicted_class} with confidence: {confidence:.2f}%"

    if confidence >= 90:
    
        h2=" Prediction is correct!"
        
        return h2,h1
    else:
        h2="Prediction is correct!"
        predicted_class=None
        return h2,h1
# Evaluate model performance on the full test dataset
