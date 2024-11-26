import tensorflow as tf
import matplotlib.pyplot as plt

# Ścieżka do pliku TFRecord
base_dir = "D:/Pulpit/PWR/neuronowe/tfrecords/burak_test.tfrecords"

@tf.autograph.experimental.do_not_convert
def parse_tfrecord_fn(example_image):
    feature_description = {
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64)
    }
    # Odczytywanie pojedynczego rekordu
    parsed_features = tf.io.parse_single_example(example_image, feature_description)

    # Dekodowanie obrazu
    image = tf.io.decode_raw(parsed_features['image_raw'], tf.uint8)
    image = tf.reshape(image, (224, 224, 3))  # Przekształcanie obrazu do odpowiednich rozmiarów

    label = tf.cast(parsed_features['label'], tf.int32)

    return image, label

# Odczyt pliku TFRecord
dataset = tf.data.TFRecordDataset(base_dir)
dataset = dataset.map(parse_tfrecord_fn)  # Zastosowanie funkcji do przetworzenia każdego rekordu



# Pobranie pierwszego obrazu i etykiety
for image, label in dataset.take(1):  # Wybieramy tylko 1 obraz
    # Konwertowanie obrazu na numpy
    image_np = image.numpy()

    # Wyświetlanie obrazu
    plt.imshow(image_np)
    plt.title(f"Etykieta: {label.numpy()}")
    plt.show()
