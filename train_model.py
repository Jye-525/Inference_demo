import tensorflow as tf
import numpy as np
from skimage.transform import resize
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

tf.config.set_visible_devices([], 'GPU')

train_Path = '/home/yejie/HStream/car_dataset/Train'
test_Path = '/home/yejie/HStream/car_dataset/Test'
IMAGE_SIZE = [224, 224]

# Using ImageDataGenerator to load images from the directory
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

test_datagen = ImageDataGenerator(
    rescale = 1./255
)

training_set = train_datagen.flow_from_directory(
    train_Path,
    target_size = IMAGE_SIZE,
    batch_size = 32,
    class_mode = 'categorical' # As we have more than 2 so using categorical.. for 2 we might have used binary.
)

test_set = train_datagen.flow_from_directory(
    test_Path,
    target_size = IMAGE_SIZE,
    batch_size = 32,
    class_mode = 'categorical'
)

print(test_set[0][0].shape)

num_classes = 3
# Load resnet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Set the flatten layer.
x = Flatten() (base_model.output)
prediction = Dense(num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=prediction)

# Freeze the layers in the base model to prevent them from being trained
for layer in base_model.layers:
    layer.trainable = False

# Compile the model with SGD optimizer and categorical cross-entropy loss
model.compile (
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy']
)

# Train the model
history = model.fit(
    training_set,
    epochs = 50,
    steps_per_epoch = len(training_set),
)

# model.save("./resnet50_model1")

y_pred = model.predict(test_set[0][0][:5])
print("predict is: ", y_pred)