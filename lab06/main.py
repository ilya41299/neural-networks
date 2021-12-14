import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
import neural_network_models
import predict_data

warnings.filterwarnings('ignore')
plt.style.use("ggplot")
EPOCHS = 70


# построение графика функции потери и метрик в зависимости от числа эпох
def plot_graph(history, model_name):
    n_epochs = np.arange(0, EPOCHS)
    plt.figure()
    plt.plot(n_epochs, history.history['loss'], label='training_loss')
    plt.plot(n_epochs, history.history['val_loss'], label='validation_loss')
    plt.plot(n_epochs, history.history['accuracy'], label='training_accuracy')
    plt.plot(n_epochs, history.history['val_accuracy'], label='validation_accuracy')
    plt.title('Training loss and accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.savefig(f'{model_name}')


def use_model(model, model_name, need_flatten, trainX, testX, trainY, testY, classes):
    # обучаем нейросеть
    history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS)

    predictions = model.predict(testX, batch_size=32)
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classes))
    # строим график
    plot_graph(history, model_name)

    predict_data.make_predictions(classes, model, need_flatten, model_name)


def start_model(model_name):
    if model_name == 'FFNN':
        need_flatten = True
    else:
        need_flatten = False
    # получаем из датасета картинки в векторном представлении и метки классов
    data, labels = predict_data.get_data_and_labels(need_flatten)
    # разделяем данные на обучающую и тестовую выборки
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2)

    class_labels = LabelBinarizer()
    trainY = class_labels.fit_transform(trainY)
    testY = class_labels.fit_transform(testY)

    if model_name == 'FFNN':
        model = neural_network_models.get_FFNN(len(class_labels.classes_))
    else:
        model = neural_network_models.get_CNN(len(class_labels.classes_))

    use_model(model, model_name, need_flatten, trainX, testX, trainY, testY, class_labels.classes_)


if __name__ == "__main__":
    start_model('FFNN')
    start_model('CNN')
