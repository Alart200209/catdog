import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam

# Путь к папке с данными
train_data_dir = 'C:/Users/alart/PycharmProjects/catdog/train'

# Размер изображений
img_width, img_height = 150, 150
batch_size = 32
epochs = 30

# Функция для загрузки и подготовки изображения
def load_and_preprocess_image(path, label):
    img = load_img(path.numpy().decode('utf-8'), target_size=(img_width, img_height))
    img_array = img_to_array(img) / 255.0  # Нормализация изображений
    return img_array, label

# Функция-обертка для использования tf.py_function
def load_and_preprocess_image_wrapper(path, label):
    img, label = tf.py_function(func=load_and_preprocess_image, inp=[path, label], Tout=[tf.float32, tf.int32])
    img.set_shape((img_width, img_height, 3))
    label.set_shape(())
    return img, label

# Функция для создания списка путей и меток
def create_dataset(directory):
    file_paths = []
    labels = []

    for img_file in os.listdir(directory):
        img_path = os.path.join(directory, img_file)
        if os.path.isfile(img_path):
            file_paths.append(img_path)
            if img_file.startswith('cat'):
                labels.append(0)
            elif img_file.startswith('dog'):
                labels.append(1)
            elif img_file.startswith('other'):
                labels.append(2)
            else:
                raise ValueError(f"Неизвестная метка для файла {img_file}")

    return file_paths, labels

# Создание списка путей и меток
file_paths, labels = create_dataset(train_data_dir)

# Разделение на обучающую и тестовую выборки
X_train_paths, X_test_paths, y_train, y_test = train_test_split(file_paths, labels, test_size=0.2, random_state=42)

# Создание tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_paths, y_train))
train_dataset = train_dataset.map(load_and_preprocess_image_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size=len(X_train_paths)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test_paths, y_test))
test_dataset = test_dataset.map(load_and_preprocess_image_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Использование предобученной модели VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Заморозка слоев предобученной модели
for layer in base_model.layers:
    layer.trainable = False

# Создание нового вывода на основе VGG16
inputs = Input(shape=(img_width, img_height, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(3, activation='softmax')(x)  # Изменение на softmax для трех классов

# Создание модели
model = Model(inputs, outputs)

# Компиляция модели
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели с аугментацией данных
history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=test_dataset
)

# Сохранение модели в формате .h5
model.save('cat_dog_classifier_vgg16_150x150.h5')

# Сохранение модели в формате SavedModel
saved_model_dir = 'C:/Users/alart/PycharmProjects/catdog/saved_model'
tf.saved_model.save(model, saved_model_dir)

# Конвертация модели в формат TFLite
def convert_to_tflite(saved_model_dir, tflite_model_path):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    # Установите оптимизацию по умолчанию
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    try:
        tflite_model = converter.convert()
        with open(tflite_model_path, "wb") as f:
            f.write(tflite_model)
        print(f"Модель успешно сохранена в формате TFLite по пути: {tflite_model_path}")
    except Exception as e:
        print("Error during conversion:", e)

# Путь к файлу TFLite
tflite_model_path = "cat_dog_classifier_vgg16_150x150.tflite"

# Конвертация и сохранение модели в формате TFLite
convert_to_tflite(saved_model_dir, tflite_model_path)
