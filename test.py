import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

# Путь к модели
model_path = 'cat_dog_classifier_vgg16_150x150.h5'

# Загрузка модели
model = load_model(model_path)

# Размер изображений
img_width, img_height = 150, 150

# Функция для предобработки изображения
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_width, img_height))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Нормализация
    return img_array

# Функция для предсказания класса изображения
def predict_image(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    class_labels = ['cat', 'dog', 'other']
    predicted_label = class_labels[predicted_class]
    predicted_probability = predictions[0][predicted_class]
    return predicted_label, predicted_probability, predictions[0]

# Классификация всех изображений в папке
test_images_dir = 'C:/Users/alart/PycharmProjects/catdog/test'
results = []
true_labels = []  # Предполагается, что вы знаете истинные метки для изображений

for img_file in os.listdir(test_images_dir):
    img_path = os.path.join(test_images_dir, img_file)
    label, probability, probabilities = predict_image(img_path)
    print(f"Image: {img_file}, Predicted class: {label}, Probability: {probability:.2f}")
    results.append({
        'image': img_file,
        'predicted_label': label,
        'predicted_probability': probability,
        'probabilities': probabilities
    })
    # Пример истинной метки, это нужно заменить на актуальные данные
    if 'cat' in img_file:
        true_labels.append(0)
    elif 'dog' in img_file:
        true_labels.append(1)
    else:
        true_labels.append(2)

predicted_probabilities = [res['probabilities'] for res in results]

# Бинаризация истинных меток
true_labels_binarized = label_binarize(true_labels, classes=[0, 1, 2])

# Расчет ROC-AUC для каждого класса
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = 3

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(true_labels_binarized[:, i], np.array(predicted_probabilities)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Построение ROC-AUC кривых
plt.figure()
colors = ['aqua', 'darkorange', 'cornflowerblue']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()

# Сводка ROC-AUC по всем классам
print("ROC-AUC scores for each class:")
for i in range(n_classes):
    print(f"Class {i}: {roc_auc[i]:.2f}")

# Общий ROC-AUC
overall_roc_auc = roc_auc_score(true_labels_binarized, predicted_probabilities, average='macro')
print(f"Overall ROC-AUC: {overall_roc_auc:.2f}")
