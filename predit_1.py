import pickle
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model  # If you choose to use .h5

# If you are using pickle
with open("cnn_model.pkl", 'rb') as file:
    model_json, model_weights = pickle.load(file)
model = model_from_json(model_json)
model.set_weights(model_weights)

# Or if you saved the model as .h5
# model = load_model("cnn_model.h5")

# Step 2: Load and preprocess an image
img_path = r"C:\Users\manos\OneDrive\Desktop\plantdiseade\test\test\AppleCedarRust1.JPG"
img = image.load_img(img_path, target_size=(224, 224))  # Resize image as per model input
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = img_array / 255.0  # Normalize (if required)

# Step 3: Make predictions
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)  # Get the class index

print(f"Predicted class: {predicted_class}")
