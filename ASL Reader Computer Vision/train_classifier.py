import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Loading data dari file pickle
data_dict = pickle.load(open(r'C:\Users\VICTUS\Documents\MachineLearningProject\data.pickle', 'rb'))

# Ekstrak data dan label
data = data_dict['data']
labels = data_dict['labels']

# Pastiin semua data sampel sama panjangnya
max_length = max(len(sample) for sample in data)
data_processed = [
    np.pad(sample, (0, max_length - len(sample)), mode='constant') if len(sample) < max_length else sample
    for sample in data
]

# Ubah jadi array numpy
data_array = np.asarray(data_processed)
labels_array = np.asarray(labels)

# Pisah dataset menjadi training dan testing
x_train, x_test, y_train, y_test = train_test_split(data_array, labels_array, test_size=0.2, shuffle=True, stratify=labels_array)

# Inisiasi dan train model RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Buat Prediksinya
y_predict = model.predict(x_test)

# Akurasinya dikalkulasi
score = accuracy_score(y_predict, y_test)
print('{}% Sampel berhasil di klasifikasi'.format(score * 100))

# Simpen modelnya
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
