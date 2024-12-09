import tensorflow as tf
import keras
from keras import layers
import numpy as np
import cv2
import time

import matplotlib.pyplot as plt

# tf.config.experimental.set_visible_devices([], 'GPU') # Wylaczenie GPU

#print("GPU LIST:", tf.config.list_physical_devices('GPU'))
#print("BUILD WITH CUDA:", tf.test.is_built_with_cuda())  # Installed non-gpu package

epochs = 100

def _parse_function(proto):
    feature_description = {
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64)
    }
    parsed_features = tf.io.parse_single_example(proto, feature_description)
    image = tf.io.decode_jpeg(parsed_features['image_raw'], channels=3)  # Decode image
    image = tf.image.resize(image, [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    image = tf.cast(image, tf.float32) / 255.0  # Normalize
    label = parsed_features['label']
    return image, label

def load_tfrecors(filename):
    dataset = tf.data.TFRecordDataset(filenames=[filename])
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(154)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.train_start_time = time.time()

    def on_train_end(self, logs=None):
        train_end_time = time.time()
        total_train_time = train_end_time - self.train_start_time
        print(f"\nTotal time for training: {total_train_time:.2f} seconds.")

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

train_dataset = load_tfrecors('tfrecords/train.tfrecords')
validation_dataset = load_tfrecors('tfrecords/val.tfrecords')

# Add the TimeHistory callback to track the total training time
time_callback = TimeHistory()

# Start training
training = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=[time_callback]
    )

# Current model saved in main folder (See all models in the "models" folder)
model.save("amogrer.keras")

"""Plotting the training history"""
acc = training.history['accuracy']
val_acc = training.history['val_accuracy']

loss = training.history['loss']
val_loss = training.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()