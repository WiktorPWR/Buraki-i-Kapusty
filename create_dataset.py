from random import shuffle
import glob
import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image  # Dodajemy bibliotekę Pillow

# Funkcja do naprawy profilu obrazu
def fix_image_profile(image_path):
    """
    Naprawia profil kolorów obrazu, zapisując go ponownie.
    """
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

            # Jeśli obraz jest pusty lub ma niewłaściwe wymiary, pomijamy
            if img is None or img.shape != (224, 224, 3):
                print(f"Pomijam obraz o niepoprawnych wymiarach: {addrs[i]}")
                continue

            # Przygotowanie feature'ów dla TFRecord
            depth = img.shape[2]  # Dodanie głębi obrazu
            feature = {
                'image_raw': _bytes_feature(img.tobytes()),  # Obraz zapisany jako bytes
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

# Funkcja do testowania danych w pliku TFRecord
def display_random_image(tfrecord_path):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    
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
        img = np.frombuffer(img_raw, dtype=np.uint8).reshape((224, 224, 3))  # Odtworzenie obrazu
        label = record['label'].numpy()
        depth = record['depth'].numpy()
        print(f"Etykieta: {label}, Głębia: {depth}")
        
        # Wyświetlenie obrazu
        cv2.imshow("Obraz z TFRecord", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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

# Test: Wyświetl losowy obraz z jednego z plików TFRecord
display_random_image(os.path.join(base_dir, 'tfrecords/kapusta_train.tfrecords'))
