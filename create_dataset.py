import glob
import os
import sys
import cv2
import numpy as np
import tensorflow as tf

from PIL import Image
from matplotlib import pyplot as plt

# Funkcja do naprawy profilu obrazu
def fix_image_profile(image_path):
    """Naprawia profil kolorów obrazu, zapisując go ponownie."""
    try:
        img = Image.open(image_path)
        img.save(image_path)  # Pillow automatycznie usuwa błędny profil kolorów
        print(f"Naprawiono profil dla obrazu: {image_path}")
    except Exception as e:
        print(f"Nie udało się naprawić obrazu: {image_path}, {e}")

# Funkcja do tworzenia Feature z typu int64
def _init64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# Funkcja do tworzenia Feature z typu bytes
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Funkcja do wczytywania i przetwarzania obrazu
def load_image(addr):
    # Naprawa profilu obrazu przed wczytaniem
    fix_image_profile(addr)
    
    # Wczytanie obrazu z pliku
    img = cv2.imread(addr)
    if img is None:
        return None
    # Zmiana rozmiaru obrazu do 224x224
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    # Konwersja obrazu z BGR do RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def split_data(path, size=(224, 224), batch_size=32, train_part=0.7, val_part=0.2, test_part=0.1):
    """Podział danych na zbiory treningowe, walidacyjne i testowe.

    Args:
        path (str): Ścieżka do katalogu z danymi.
        size (tuple, optional): Rozmiar obrazów. Domyślnie (224, 224).
        batch_size (int, optional): Rozmiar partii danych. Domyślnie 32.
        train_part (float, optional): Proporcja danych treningowych. Domyślnie 0.7 (70%).
        val_part (float, optional): Proporcja danych walidacyjnych. Domyślnie 0.2 (20%).
        test_part (float, optional): Proporcja danych testowych. Domyślnie 0.1 (10%).

    Returns:
        tuple: Zbiory danych (train_dataset, val_dataset, test_dataset).
    """
    dataset = tf.keras.utils.image_dataset_from_directory(
        path,
        image_size=size,
        batch_size=batch_size
    )

    # Podział rozmiarów zbiorów
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_part)
    val_size = int(dataset_size * val_part)
    test_size = int(dataset_size * test_part)

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)
    test_dataset = dataset.skip(train_size + val_size).take(test_size)

    print(f"Rozmiar zbioru pełnego: {len(dataset)}")
    print(f"Rozmiar poszczególnych zbiorów: Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Normalizacja obrazów
    normalization_layer = tf.keras.layers.Rescaling(1./255)

    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))
    test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

    return train_dataset, val_dataset, test_dataset

def create_data_record(dataset, tfrecords_dir, record_name):
    """Tworzenie plików TFRecord z podanego zestawu danych.

    Args:
        dataset (tf.data.Dataset): Zestaw danych zawierający obrazy i etykiety.
        tfrecords_dir (str): Ścieżka do katalogu, w którym zostanie zapisany plik TFRecord.
        record_name (str): Nazwa pliku TFRecord (bez rozszerzenia).
    """
    # Nazwa pliku wyjściowego
    out_filename = os.path.join(tfrecords_dir, record_name + '.tfrecords')
    print(f"Zapisywanie pliku: {out_filename}")
    
    try:
        # Inicjalizacja writera do zapisu pliku TFRecord
        writer = tf.io.TFRecordWriter(out_filename)
        
        # Iteracja przez wszystkie obrazy w zestawie
        for batch in dataset:
            images, labels = batch
            for image, label in zip(images, labels):
                # Konwersja obrazu do numpy array
                img = image.numpy()
                label = label.numpy()

                # Przygotowanie feature'ów dla TFRecord
                depth = img.shape[2]  # Dodanie głębi obrazu
                feature = {
                    'image_raw': _bytes_feature(img.tobytes()), # Obraz zapisany jako bytes
                    'label': _init64_feature(label),            # Etykieta jako int64
                    'depth': _init64_feature(depth)             # Głębia obrazu jako int64
                }

                # Tworzenie obiektu Example z feature'ami
                example = tf.train.Example(features=tf.train.Features(feature=feature))

                # Zapisanie example do pliku TFRecord
                writer.write(example.SerializeToString())

        # Zamknięcie writera po zakończeniu zapisu
        writer.close()
        print(f"Zapisano plik: {out_filename}")
    
    except Exception as e:
        # Obsługa wyjątków i błędów
        print(f"Wystąpił błąd przy zapisywaniu pliku {out_filename}: {e}")
        
def display_random_image(tfrecords_dir, record_name):
    """Wyświetla losowy obraz z pliku TFRecord.

    Args:
        tfrecords_dir (str): Ścieżka do katalogu zawierającego pliki TFRecord.
        record_name (str): Nazwa pliku TFRecord (bez rozszerzenia).
    """
    # Wczytanie pliku TFRecord
    tfrecord_file_path = os.path.join(tfrecords_dir, record_name + '.tfrecords')
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file_path)
    
    feature_description = {
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64)
    }
    
    def _parse_function(proto):
        return tf.io.parse_single_example(proto, feature_description)
    
    parsed_dataset = raw_dataset.map(_parse_function)
    
    for record in parsed_dataset.take(1):  # Pobieramy pierwszy rekord
        img_raw = record['image_raw'].numpy()
        img = np.frombuffer(img_raw, dtype=np.float32)
        img = img.reshape((224, 224, 3))  # Odtworzenie obrazu
        label = record['label'].numpy()
        depth = record['depth'].numpy()
        print(f"Etykieta: {label}, Głębia: {depth}")
        
        # Wyświetlenie obrazu
        cv2.imshow("Obraz z TFRecord", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Zmienna bazowa, która wskazuje główny katalog
data_dir = 'data'
tfrecords_dir = 'tfrecords'

train_dataset, val_dataset, test_dataset = split_data(data_dir, batch_size=10)

create_data_record(train_dataset, tfrecords_dir, 'train')
create_data_record(val_dataset, tfrecords_dir, 'val')
create_data_record(test_dataset, tfrecords_dir, 'test')

# Test: Wyświetl losowy obraz z jednego z plików TFRecord
display_random_image(tfrecords_dir, 'train')