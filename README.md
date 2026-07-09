# Plant Disease Detection using CNN & ResNet50

A deep learning project for detecting plant diseases from leaf images using Convolutional Neural Networks (CNN) and GPT-2 for generating disease treatment suggestions.

## Project Structure

```
├── PlantDiseaseDetection.py      # CNN model training (root)
├── data/
│   ├── PlantDiseaseDetection.py  # CNN model training with HSV filtering + checkpoints
│   ├── cnn_model.pkl             # Saved trained model
│   └── label_transform.pkl       # Label binarizer
├── predict.py                    # Load model and predict on new images
├── predit_1.py                   # Alternative prediction script (loads .pkl model)
├── chat.py                       # GPT-2 integration for disease treatment suggestions
├── cnn_model.h5                  # Trained model (HDF5 format)
├── cnn_model.pkl                 # Trained model (pickle format)
├── label_transform.pkl           # Encoded class labels
├── plant_disease_model.h5        # Alternative trained model
└── test/                         # Test images
```

## Model Architecture

The CNN model consists of:
- **3 convolutional blocks** (32, 64, 128 filters) with ReLU activation, batch normalization, max pooling, and dropout
- **Fully connected layers** with 1024 neurons and softmax output
- **Exponential decay learning rate** scheduler
- **Image augmentation** (rotation, shift, shear, zoom, flip)

Training parameters:
- Image size: 256x256
- Epochs: 30-50
- Batch size: 8-12
- Optimizer: Adam
- Loss: Binary crossentropy

## Usage

### Training
```bash
python PlantDiseaseDetection.py
```

### Prediction
```bash
python predict.py                 # Uses plant_disease_model.h5
python predit_1.py                # Uses cnn_model.pkl
```

### Disease Treatment Suggestions (GPT-2)
```bash
python chat.py
```

## Requirements

- Python 3.x
- TensorFlow / Keras
- OpenCV
- scikit-learn
- matplotlib
- numpy
- transformers (for GPT-2 chat)

## Dataset

Uses the [New Plant Diseases Dataset (Augmented)](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) - a collection of leaf images across multiple plant species with various diseases.