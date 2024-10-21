import tensorflow as tf

# Путь к сохраненной модели в формате SavedModel
saved_model_dir = 'C:/Users/alart/PycharmProjects/catdog/saved_model'

# Создание конвертера для модели
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# Настройка конвертера для оптимизации совместимости
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Преобразование модели в TFLite формат
try:
    tflite_model = converter.convert()
    # Сохранение TFLite модели
    tflite_model_path = 'cat_dog_classifier_vgg16_150x150.tflite'
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite модель сохранена по пути: {tflite_model_path}")
except Exception as e:
    print(f"Ошибка при конвертации модели: {e}")
