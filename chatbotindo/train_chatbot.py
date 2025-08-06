import json
import random
import pickle
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

stemmer = PorterStemmer()

with open('niat.json') as file:
    data = json.load(file)

kata = []
label = []
dokumen_x = []
dokumen_y = []

for niat in data['niat']:
    for pola in niat['pola']:
        token = nltk.word_tokenize(pola)
        kata.extend(token)
        dokumen_x.append(token)
        dokumen_y.append(niat['tag'])

    if niat['tag'] not in label:
        label.append(niat['tag'])

kata = [stemmer.stem(w.lower()) for w in kata if w.isalpha()]
kata = sorted(list(set(kata)))
label = sorted(label)

latih_x = []
latih_y = []

for idx, dokumen in enumerate(dokumen_x):
    bag = [1 if stemmer.stem(k.lower()) in [stemmer.stem(w.lower()) for w in dokumen] else 0 for k in kata]
    latih_x.append(bag)

    label_satu = [0] * len(label)
    label_satu[label.index(dokumen_y[idx])] = 1
    latih_y.append(label_satu)

latih_x = np.array(latih_x)
latih_y = np.array(latih_y)

model = Sequential()
model.add(Dense(128, input_shape=(len(latih_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(latih_y[0]), activation='softmax'))

opt = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(latih_x, latih_y, epochs=200, batch_size=5, verbose=1)
model.save('model_chatbot.h5')

with open('kata.pkl', 'wb') as f:
    pickle.dump(kata, f)

with open('label.pkl', 'wb') as f:
    pickle.dump(label, f)
