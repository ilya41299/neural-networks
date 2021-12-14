import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM
from keras.utils import to_categorical

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from tqdm import tqdm
import gensim.downloader

EPOCHS = 12

nltk.download("stopwords")

train_file1 = 'amazon_cells_labelled.csv'
train_file2 = 'imdb_labelled.csv'
train_file3 = 'yelp_labelled.csv'
df_mas = []
df = pd.read_csv(train_file1)
df_mas.append(df)
df = pd.read_csv(train_file2)
df_mas.append(df)
df = pd.read_csv(train_file3)
df_mas.append(df)
df = pd.concat(df_mas)

docs = []
stop_words = stopwords.words('english')
stemmer = SnowballStemmer('english')
for sentence in tqdm(df['text']):
    tokens = []
    for token in sentence.split():
        if token not in stop_words:
            tokens.append(stemmer.stem(token))
        else:
            tokens.append(token)
    docs.append(' '.join(tokens))

print(docs[:5])

# скачиваем обученную на twitter`е модель
glove_vectors = gensim.downloader.load('glove-twitter-25')

tokenizer = Tokenizer()
# создаем словарь
tokenizer.fit_on_texts(docs)
# находим его размер
vocab_size = len(tokenizer.word_index) + 1

encoded_docs = tokenizer.texts_to_sequences(docs)

X = pad_sequences(encoded_docs)
Y = df['emotion'].to_numpy()
print(X)
print(Y)
print(len(X[0]))

# Создаем матрицу, где строки - все слова словаря, столбцы - представление слова в пространстве. Т.е. слова и их координаты
embedding_matrix = np.zeros((vocab_size, glove_vectors.vector_size))
print(embedding_matrix)

for word, i in tqdm(tokenizer.word_index.items()):
    if word in glove_vectors:
        embedding_matrix[i] = glove_vectors[word]
print(embedding_matrix)

model = Sequential()
model.add(Embedding(
    vocab_size,
    glove_vectors.vector_size,
    weights=[embedding_matrix],
    input_length=X.shape[1]))
model.add(Dropout(0.2))
model.add(LSTM(10, dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

(trainX, testX, trainY, testY) = train_test_split(X, Y, test_size=0.20)
history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs= EPOCHS)

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
plt.savefig(f'Graph')

score = model.evaluate(testX,  testY)


print()
print(f"Accuracy: {score[1]}")
print(f"LOSS: {score[0]}")

predicted = model.predict(testX)

predicted = [np.round(prediction) for prediction in predicted]
print(classification_report(testY, predicted, target_names=["Negative", "Positive"]))

tsne = TSNE(n_components=2, perplexity=10)
X_transformed = tsne.fit_transform(X)
fit, ax = plt.subplots(figsize=(15, 7))
ax.scatter(X_transformed[:, 0], X_transformed[:, 1], color='yellow')
for i in range(18,23):
    plt.text(X_transformed[i, 0], X_transformed[i, 1], docs[i],
             fontsize=14)
    if Y[i]:
        ax.scatter(X_transformed[i, 0], X_transformed[i, 1], color='green', label=f'positive')
    else:
        ax.scatter(X_transformed[i, 0], X_transformed[i,1], color='red', label=f'negative')

plt.savefig("TSNE.png")
