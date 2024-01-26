# app.py
from flask import Flask, render_template, request
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

app = Flask(__name__)

# Load the model and tokenizer
model = load_model('deploy/cnn.h5')
with open('deploy/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

sent_len = ...  # Define sent_len here

def predict_phishing(url):
    sequence = tokenizer.texts_to_sequences([url])
    padded_sequence = pad_sequences(sequence, maxlen=sent_len, padding='post', truncating='post')
    prediction = model.predict(padded_sequence)
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        url = request.form['url']
        prediction = predict_phishing(url)
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
