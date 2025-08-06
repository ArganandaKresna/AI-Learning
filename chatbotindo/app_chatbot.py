import tkinter as tk
from tkinter import scrolledtext
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from tensorflow.keras.models import load_model

stemmer = PorterStemmer()
model = load_model('model_chatbot.h5')

with open('kata.pkl', 'rb') as f:
    kata = pickle.load(f)

with open('label.pkl', 'rb') as f:
    label = pickle.load(f)

with open('niat.json') as file:
    data = json.load(file)

def bersihkan_kalimat(kalimat):
    token = nltk.word_tokenize(kalimat)
    token = [stemmer.stem(w.lower()) for w in token if w.isalpha()]
    return token

def ubah_ke_bag_of_words(kalimat, kata_kunci):
    kalimat_token = bersihkan_kalimat(kalimat)
    bag = [1 if k in kalimat_token else 0 for k in kata_kunci]
    return np.array(bag)

def tanggapan(teks_pengguna):
    hasil = ubah_ke_bag_of_words(teks_pengguna, kata)
    prediksi = model.predict(np.array([hasil]))[0]
    ambang = 0.7
    if max(prediksi) > ambang:
        tag = label[np.argmax(prediksi)]
        for niat in data['niat']:
            if niat['tag'] == tag:
                return random.choice(niat['respon'])
    return "Maaf, saya tidak mengerti maksudmu."

# GUI
app = tk.Tk()
app.title("Asisten Virtual Indonesia")
app.geometry("500x550")

frame = tk.Frame(app)
frame.pack(pady=10)

chatbox = scrolledtext.ScrolledText(frame, width=60, height=25, wrap=tk.WORD, font=("Helvetica", 10))
chatbox.pack()
chatbox.config(state=tk.DISABLED)

entri = tk.Entry(app, width=50, font=("Helvetica", 12))
entri.pack(pady=5)

def kirim_pesan():
    teks = entri.get()
    if teks.strip():
        chatbox.config(state=tk.NORMAL)
        chatbox.insert(tk.END, "Kamu: " + teks + "\n")
        balasan = tanggapan(teks)
        chatbox.insert(tk.END, "Bot: " + balasan + "\n\n")
        chatbox.config(state=tk.DISABLED)
        entri.delete(0, tk.END)

tombol = tk.Button(app, text="Kirim", command=kirim_pesan)
tombol.pack()

app.mainloop()
