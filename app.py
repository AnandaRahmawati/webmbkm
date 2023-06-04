from flask import Flask, render_template, request, redirect, url_for, jsonify
import sqlite3

app = Flask(__name__)

# Membuat koneksi dengan database
def create_connection():
    conn = None
    try:
        conn = sqlite3.connect('flask.db')
        print('Database connection successful')
    except sqlite3.Error as e:
        print('Error connecting to database:', e)
    return conn

# Membuat tabel users 
def create_users_table():
    conn = create_connection()
    cursor = conn.cursor()
    
    query = '''
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        email TEXT NOT NULL,
        password TEXT NOT NULL
    )
    '''
    cursor.execute(query)
    conn.commit()
    conn.close()

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/process_login', methods=['POST'])
def process_login():
    conn = create_connection()
    cursor = conn.cursor()
    
    # Mendapatkan data dari form login
    username = request.form['username']
    password = request.form['password']
    
    # Eksekusi query untuk memeriksa data login
    query = "SELECT * FROM users WHERE username=? AND password=?"
    cursor.execute(query, (username, password))
    result = cursor.fetchone()
    
    if result:
        return redirect(url_for('menu_utama'))
    else:
        error_message = "Username atau password yang Anda masukkan salah. Silakan coba lagi."
        return render_template('login.html', error_message=error_message)

@app.route('/menu-utama')
def menu_utama():
    return render_template('menu-utama.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        conn = create_connection()
        cursor = conn.cursor()
        
        # Mendapatkan data dari form register
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Eksekusi query untuk menyimpan data register ke database
        query = "INSERT INTO users (username, email, password) VALUES (?, ?, ?)"
        cursor.execute(query, (username, email, password))
        conn.commit()
        conn.close()
        
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/process', methods=['POST'])
def process():
    message = request.json['message']

    # Panggil fungsi process_message() dari process.py
    response = process_message(message)

    return jsonify(response)


def process_message(message):
    response = "Anda mengirim pesan: " + message
    return response

import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model
model = load_model('model/model.h5')
import json
import random
intents = json.loads(open('model/kampus_merdeka1.json', encoding='utf8').read())
words = pickle.load(open('model/texts.pkl','rb'))
classes = pickle.load(open('model/labels.pkl','rb'))

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

@app.route('/logout')
def logout():
    return render_template('index.html')


if __name__ == '__main__':
    create_users_table()
    app.run(debug=True)