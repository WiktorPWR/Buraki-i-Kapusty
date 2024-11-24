from random import shuffle
import glob
import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from enum import Enum

# Funkcja do tworzenia Feature z typu int64
def _init64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# Funkcja do tworzenia Feature z typu bytes
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# Funkcja do wczytywania i przetwarzania obrazu
def load_image(addr):
    # Wczytanie obrazu z pliku
    img = cv2.imread(addr)
    if img is None:
        return None
    # Zmiana rozmiaru obrazu do 224x224
    img = cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)
    # Konwersja obrazu z BGR do RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# Funkcja do tworzenia pliku TFRecord z obrazów i etykiet
def createDataRecord(out_filename, addrs, labels):
    try:
        # Inicjalizacja writera do zapisu pliku TFRecord
        writer = tf.io.TFRecordWriter(out_filename)
        
        # Iteracja przez wszystkie obrazy w zestawie
        for i in range(len(addrs)):
            if not i % 1:  # co 1 iterację wypisujemy status
                print('Train data: {}/{}'.format(i, len(addrs)))
                sys.stdout.flush()
            
            # Wczytanie obrazu
            img = load_image(addrs[i])

            # Pobranie etykiety dla obrazu
            label = labels[i]

            # Jeśli obraz jest pusty, pomijamy
            if img is None:
                continue

            # Przygotowanie feature'ów dla TFRecord
            feature = {
                'image_raw': _bytes_feature(img.tobytes()),  # Obraz zapisany jako bytes
                'label': _init64_feature(label)  # Etykieta jako int64
            }

            # Tworzenie obiektu Example z feature'ami
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Zapisanie example do pliku TFRecord
            writer.write(example.SerializeToString())

        # Zamknięcie writera po zakończeniu zapisu
        writer.close()
        sys.stdout.flush()
        print(f"Zapisano plik: {out_filename}")
    
    except Exception as e:
        # Obsługa wyjątków i błędów
        print(f"Wystąpił błąd przy zapisywaniu pliku {out_filename}: {e}")

# Zmienna bazowa, która wskazuje główny katalog
base_dir = 'D:/Pulpit/PWR/neuronowe'

# Ścieżki do obrazów dla zestawów treningowych
kapusta_train_path = os.path.join(base_dir, 'kapusta_buraki/kapusta/train/*.*')
buraki_train_path = os.path.join(base_dir, 'kapusta_buraki/buraki/train/*.*')

# Ścieżki do obrazów dla zestawów walidacyjnych
kapusta_validation_path = os.path.join(base_dir, 'kapusta_buraki/kapusta/validation/*.*')
buraki_validation_path = os.path.join(base_dir, 'kapusta_buraki/buraki/validation/*.*')

# Ścieżki do obrazów dla zestawów testowych
kapusta_test_path = os.path.join(base_dir, 'kapusta_buraki/kapusta/test/*.*')
buraki_test_path = os.path.join(base_dir, 'kapusta_buraki/buraki/test/*.*')

# Pobieranie listy plików (ścieżek do obrazów) z katalogów
addrs_kapusta_train = glob.glob(kapusta_train_path)
addrs_buraki_train = glob.glob(buraki_train_path)

addrs_kapusta_validation = glob.glob(kapusta_validation_path)
addrs_buraki_validation = glob.glob(buraki_validation_path)

addrs_kapusta_test = glob.glob(kapusta_test_path)
addrs_buraki_test = glob.glob(buraki_test_path)

# Etykiety dla poszczególnych zestawów danych

Kapusta = 1
Burak = 0


# Tworzenie plików TFRecord dla zestawów treningowych
createDataRecord(os.path.join(base_dir, 'tfrecords/kapusta_train.tfrecords'), addrs=addrs_kapusta_train, labels=[Kapusta] * len(addrs_kapusta_train))
createDataRecord(os.path.join(base_dir, 'tfrecords/burak_train.tfrecords'), addrs=addrs_buraki_train, labels=[Burak] * len(addrs_buraki_train))

# Tworzenie plików TFRecord dla zestawów walidacyjnych
createDataRecord(os.path.join(base_dir, 'tfrecords/kapusta_validation.tfrecords'), addrs=addrs_kapusta_validation, labels=[Kapusta] * len(addrs_kapusta_validation))
createDataRecord(os.path.join(base_dir, 'tfrecords/burak_validation.tfrecords'), addrs=addrs_buraki_validation, labels=[Burak] * len(addrs_buraki_validation))

# Tworzenie plików TFRecord dla zestawów testowych
createDataRecord(os.path.join(base_dir, 'tfrecords/kapusta_test.tfrecords'), addrs=addrs_kapusta_test, labels=[Kapusta] * len(addrs_kapusta_test))
createDataRecord(os.path.join(base_dir, 'tfrecords/burak_test.tfrecords'), addrs=addrs_buraki_test, labels=[Burak] * len(addrs_buraki_test))
