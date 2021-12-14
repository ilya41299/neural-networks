import numpy as np
import cv2
import os
import warnings
warnings.filterwarnings('ignore')

train_path = os.path.join('data', 'train')
test_path = os.path.join('data', 'test')
result_path = os.path.join('data', 'result')


def get_data_and_labels(need_flatten):
    data = []
    labels = []

    train_labels = os.listdir(train_path)
    for training_name in train_labels:
        current_dir = os.path.join(train_path, training_name)
        images = [f for f in os.listdir(current_dir) if
                  os.path.isfile(os.path.join(current_dir, f)) and f.endswith(".jpg")]

        # каждое изображение приводим к одному размеру и "вытягиваем в вектор" (для подачи нейронной сети)
        for file in images:
            image_path = os.path.join(current_dir, file)

            # приводим изображение, преобразуем в вектор и добавляем в массив
            image = cv2.imread(image_path)
            image = np.array(cv2.resize(image, (32, 32)))
            if need_flatten:
                image = image.flatten()
            data.append(image)

            # сохраняем соответствующую метку класса (это папка, в которой лежит файл)
            label = image_path.split(os.path.sep)[-2]
            labels.append(label)

    # нормируем значение пикселей [0:1]
    data = np.array(data, dtype=float) / 255.0
    labels = np.array(labels)

    return data, labels


def make_predictions(class_labels, model, need_flatten, model_name):
    test_images = [image for image in os.listdir(test_path) if image.endswith('.jpg')]

    for idx, image in enumerate(test_images):
        image = cv2.imread(os.path.join(test_path, image))
        image_copy = image.copy()

        image = cv2.resize(image, (32, 32))

        if need_flatten:
            image = image.flatten()

        image = image.astype('float') / 255.0

        if need_flatten:
            image = image.reshape((1, image.shape[0]))
        else:
            image = image.reshape((1, *image.shape))
        # массив вероятностей, к какому классу относится картинка
        predictions = model.predict(image)
        # выбираем индекс класса с наибольшей вероятностью и соответствующее имя
        i = predictions.argmax(axis=1)[0]
        class_label = class_labels[i]
        # в тексте выводим метку класса и вероятность принадлежности
        text = f'{class_label}: {round(predictions[0][i] * 100, 2)}%'

        cv2.putText(image_copy, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(result_path, f'{model_name}_{idx}.png'), image_copy)

