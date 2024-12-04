import tensorflow as tf
import keras
from keras import layers
from keras import ops
import numpy
import cv2


def _parse_function(proto):
    feature_description = {
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64)
    }
    parsed_features = tf.io.parse_single_example(proto, feature_description)
    image = tf.io.decode_jpeg(parsed_features['image_raw'], channels=3)  # Dekodowanie obrazu
    image = tf.image.resize(image,[224,224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    image = tf.cast(image, tf.float32) / 255.0 #normalizacja xdddd
    label = parsed_features['label']
    return image, label


def load_tfrecors(filename):

    dataset = tf.data.TFRecordDataset(filenames=[filename])

    dataset = dataset.map(_parse_function)

    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(154)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


model = tf.keras.Sequential([
    # Warstwy wejściowe i konwolucyjne
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

    # Warstwy spłaszczające i w pełni połączone
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Dropout dla regularizacji
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    
    # Wyjście
    tf.keras.layers.Dense(2, activation='softmax')  # softmax dla klasyfikacji wieloklasowej
])



model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

train_dataset = load_tfrecors('D:/Pulpit/Buraki i kapusty - github/Buraki-i-Kapusty-1/tfrecords/train.tfrecords')
validation_dataset = load_tfrecors('D:/Pulpit/Buraki i kapusty - github/Buraki-i-Kapusty-1/tfrecords/val.tfrecords')

model.fit(train_dataset, validation_data= validation_dataset,epochs=10)
model.save("amogrer.keras")