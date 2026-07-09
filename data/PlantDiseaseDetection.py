import numpy as np
import pickle
import cv2

from os import listdir
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU Memory Growth Enabled")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected. Running on CPU.")



EPOCHS = 50
INIT_LR = 1e-3
BS = 8
default_image_size = tuple((256, 256))
image_size = 0

directory_root = r'C:\Users\manos\OneDrive\Desktop\plantdiseade\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)'
width=256
height=256
depth=3
def process_images_in_batches(image_list, image_labels, batch_size=1000):
    x_train, x_test, y_train, y_test = None, None, None, None
    if isinstance(image_list, np.ndarray):
        image_list = image_list.tolist()

    for i in range(0, len(image_list), batch_size):
        batch_images = image_list[i:i + batch_size]
        batch_labels = image_labels[i:i + batch_size]

        np_image_list = np.array(batch_images, dtype=np.float32) / 255.0

        assert len(batch_images) == len(batch_labels), "Batch size mismatch"

        x_train, x_test, y_train, y_test = train_test_split(np_image_list, batch_labels, test_size=0.2, random_state=42)

        print(f"Processed batch {i // batch_size + 1}")
        return x_train, x_test, y_train, y_test
def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir) 
        if image is not None:
            image = cv2.resize(image, default_image_size) 
            
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            
            lower_bound = np.array([35, 40, 40])  
            upper_bound = np.array([85, 255, 255])  
            
            
            mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
            filtered_image = cv2.bitwise_and(image, image, mask=mask)
            
            return img_to_array(filtered_image)  # Convert to array
        else:
            return np.array([])
    except Exception as e:
        print(f"Error: {e}")
        return None


image_list, label_list = [], []
try:
    print("[INFO] Loading images ...")
    root_dir = listdir(directory_root)
    for directory in root_dir :
        if directory == ".DS_Store" :
            root_dir.remove(directory)

    for plant_folder in root_dir :
        plant_disease_folder_list = listdir(f"{directory_root}/{plant_folder}")
        
        for disease_folder in plant_disease_folder_list :

            if disease_folder == ".DS_Store" :
                plant_disease_folder_list.remove(disease_folder)

        for plant_disease_folder in plant_disease_folder_list:
            print(f"[INFO] Processing {plant_disease_folder} ...")
            plant_disease_image_list = listdir(f"{directory_root}/{plant_folder}/{plant_disease_folder}/")
                
            for single_plant_disease_image in plant_disease_image_list :
                if single_plant_disease_image == ".DS_Store" :
                    plant_disease_image_list.remove(single_plant_disease_image)

            for image in plant_disease_image_list[:200]:
                image_directory = f"{directory_root}/{plant_folder}/{plant_disease_folder}/{image}"
                if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
                    image_list.append(convert_image_to_array(image_directory))
                    label_list.append(plant_disease_folder)
    print("[INFO] Image loading completed")  
except Exception as e:
    print(f"Error : {e}")

image_size = len(image_list)

label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(label_list)
pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))
n_classes = len(label_binarizer.classes_)

print(label_binarizer.classes_)

x_train, x_test, y_train, y_test =process_images_in_batches(image_list,image_labels)

print("[INFO] Spliting data to train, test")

aug = ImageDataGenerator(
    rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, 
    zoom_range=0.2,horizontal_flip=True, 
    fill_mode="nearest")

model = Sequential()
inputShape = (height, width, depth)
chanDim = -1
checkpoint_path = "checkpoints/cnn_model-{epoch:02d}.weights.h5"

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False, 
    save_freq='epoch',
    verbose=1
)

if K.image_data_format() == "channels_first":
    inputShape = (depth, x, width)
    chanDim = 1
model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(n_classes))
model.add(Activation("softmax"))
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=INIT_LR,
    decay_steps=EPOCHS,  
    decay_rate=0.96,     
    staircase=True       
)
opt = Adam(learning_rate=lr_schedule)
latest_checkpoint = tf.train.latest_checkpoint("checkpoints")
if latest_checkpoint:
    print(f"Resuming training from {latest_checkpoint}")
    model.load_weights(latest_checkpoint)
    initial_epoch = int(latest_checkpoint.split('-')[-1])  # Extract epoch number
else:
    print("No checkpoints found. Starting training from scratch.")
    initial_epoch = 0

model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
print("[INFO] training network...")
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)
BS = 10
train_generator = aug.flow(x_train, y_train, batch_size=BS)
history = model.fit(
    train_generator,
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train) // BS,
    epochs=EPOCHS, initial_epoch=initial_epoch, verbose=1,callbacks=[checkpoint_callback]
    )
print(history.history.keys())
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()

print("[INFO] Calculating model accuracy")
scores = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1]*100}")

print("[INFO] Saving model...")
pickle.dump(model,open('cnn_model.pkl', 'wb'))
