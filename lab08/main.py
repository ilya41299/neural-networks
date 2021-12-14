import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.layers import Input, Dense, Flatten, Reshape, Dropout
from keras.models import Model
from sklearn.model_selection import train_test_split

# путь до папки с датасетом
train_path = "dataset all/train"
# считываем наименования диракторий - метки классов 0, 1, ... 9
train_labels = os.listdir(train_path)
x_train = []
labels = []
# задаём количество эпох обучения
EPOCHS = 20

# функция наложения шума на изображение
def apply_noise(x):
    noise = (5 * np.random.random(x.shape)).clip(0, 1).astype(np.uint8)
    return (x + noise) % 2


# функция создания нейронной сети
def create_dense_ae():
    encoding_dim = 250 # кодовое расстояние
    # кодер
    # входной слой размерности соот-й размеру изображения: 100x100 с 1 каналом (ч/б)
    input_img = Input(
        shape=(100, 100, 1))
    flat_img = Flatten()(input_img)
    # 15% нейронов отбрасываем
    x = Dropout(0.15)(flat_img)

    # полносвязные вспомогательные слои (по уменьшению размерности)
    x = Dense(encoding_dim * 3, activation='relu')(x)
    x = Dense(encoding_dim * 2, activation='relu')(x)

    # кодирующий слой
    encoded = Dense(encoding_dim, activation='relu')(x)

    # декодер
    # выход кодера -> вход декодера
    input_encoded = Input(shape=(encoding_dim,))

    # аналогично, теперь раздуваем вектор
    x = Dense(encoding_dim * 2, activation='relu')(input_encoded)
    x = Dense(encoding_dim * 3, activation='relu')(x)

    # восстанавливаем исходные размеры изображения
    flat_decoded = Dense(100 * 100 * 1, activation='sigmoid')(x)

    # восстанавливаем массив из векторного представления
    decoded = Reshape((100, 100, 1))(flat_decoded)

    # создаем модели (взодные слои, выходные слои, имя)
    encoder = Model(input_img, encoded, name="encoder")
    decoder = Model(input_encoded, decoded, name="decoder")
    all_output = decoder(encoder(input_img))
    autoencoder = Model(input_img, all_output, name="autoencoder")

    return encoder, decoder, autoencoder


if __name__ == '__main__':

    # считывание и предобработка изображений
    for current_label in train_labels:
        current_dir = os.path.join(train_path, current_label)

        images = [f for f in os.listdir(current_dir) if
                  os.path.isfile(os.path.join(current_dir, f)) and (f.endswith(".png") or f.endswith(".jpg"))]
        for file in images:

            file_path = os.path.join(current_dir, file)
            image_file = Image.open(file_path)
            image_file = image_file.convert('1')
            image_file = tf.keras.preprocessing.image.img_to_array(image_file)
            x_train.append(image_file)
            labels.append(current_label)

    # преобразовываем list в numpy массив
    x_train = np.array(x_train)
    labels = np.array(labels)

    #Разделяем x_train на тестовую и тренировочную часть
    (trainX, testX, trainY, testY) = train_test_split(x_train, labels, test_size=0.2)

    # накладываем шум
    trainX_noise = apply_noise(trainX)
    testX_noise = apply_noise(testX)
    # создаем наш автокодировщик
    encoder, decoder, autoencoder = create_dense_ae()
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    encoder.summary()
    decoder.summary()
    autoencoder.summary()
    # обучаем сеть
    history = autoencoder.fit(
        trainX_noise,
        trainX,
        epochs=EPOCHS,
        shuffle=True,
        validation_data=(testX_noise, testX))

    # выводим график зависимости функции потерь от числа пройденных эпох
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("graph.jpg")

    # сохраним несколько изображений без шума, с шумом, очищенные от шума (после автокодировщика)
    for i in range(0, 10):

        imgs = testX[i:i+1]
        image_save = tf.keras.preprocessing.image.array_to_img(imgs[0])
        image_save = image_save.convert('RGB')
        tf.keras.preprocessing.image.save_img(f'output/x_{i}.jpg', image_save)

        imgs = testX_noise[i:i + 1]
        image_save = tf.keras.preprocessing.image.array_to_img(imgs[0])
        image_save = image_save.convert('RGB')
        tf.keras.preprocessing.image.save_img(f'output/~x_{i}.jpg', image_save)

        encoded_imgs = encoder.predict(imgs)
        decoded_imgs = decoder.predict(encoded_imgs)

        image_save = tf.keras.preprocessing.image.array_to_img(decoded_imgs[0])
        image_save = image_save.convert('RGB')
        tf.keras.preprocessing.image.save_img(f'output/^x_{i}.jpg', image_save)


