import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Define the label mapping
label_mapping = {0: 'Burak', 1: 'Kapusta'}

# Load the pre-trained model form main folder (If you have a model in the "models" folder, you can load it from there)
model = load_model("amogrer.keras")
model = keras.Sequential([model, keras.layers.Softmax()])

# Function to parse and preprocess the images from the TFRecord dataset
def parse_tfrecord(example):
    feature_description = {
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([1], tf.int64)
    }

    parsed_example = tf.io.parse_single_example(example, feature_description)

    # Decode the image from raw bytes
    image = tf.io.decode_jpeg(parsed_example['image_raw'], channels=3)  # Decode the JPEG image
    image = tf.image.resize(image, [224, 224])  # Resize to match model input
    image = image / 255.0  # Normalize to [0, 1]

    label = parsed_example['label']

    return image, label

# Load the TFRecord test dataset
tfrecord_path = "tfrecords/test.tfrecords"
raw_dataset = tf.data.TFRecordDataset(tfrecord_path)

# Parse the dataset
dataset = raw_dataset.map(parse_tfrecord)

# Shuffle the dataset for random selection
dataset = dataset.shuffle(buffer_size=1000)

fig, axes = plt.subplots(3, 3, figsize=(12, 12))

# Flatten the axes array for easier indexing
axes = axes.flatten()

# Iterate over the dataset and make predictions for each image
for i, (image, label) in enumerate(dataset.take(9)):  # Take the first 9 images for the plot
    # Convert tensor to numpy array and expand dimensions for batch processing
    image = np.expand_dims(image.numpy(), axis=0)

    # Make a prediction
    prediction_array = model.predict(image)[0]

    # Get the class with the highest probability
    predicted_class = np.argmax(prediction_array)
    prediction_confidence = prediction_array[predicted_class] * 100  # Convert to percentage

    # Get the label name from the label_mapping dictionary
    predicted_label = label_mapping[predicted_class]

    # Display the image and label with prediction confidence
    ax = axes[i]
    ax.imshow(image[0])
    ax.set_title(f"{predicted_label}, {prediction_confidence:.2f}%")
    ax.axis('off')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()