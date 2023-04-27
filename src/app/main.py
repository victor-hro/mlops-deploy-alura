from flask import Flask, request, jsonify, render_template
from flask_basicauth import BasicAuth
from textblob import TextBlob
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
import os
import streamlit as st

import dotenv
project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
dotenv_path = os.path.join(project_dir.split('src')[0], 'venv/.env')

model = pickle.load(open('models/model_houses.sav', 'rb'))

app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = os.getenv('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD'] = os.getenv('BASIC_AUTH_PASSWORD')


basic_auth = BasicAuth(app)
# definindo rota base. Home = '/'
@app.route('/')
def home():
    # return "API"
    return render_template('index.html')


@app.route('/sentiment/<phrase>')
@basic_auth.required
def sentment(phrase):
    tb = TextBlob(phrase)
    translate = tb.translate(from_lang='pt', to='en')
    polarity = translate.sentiment.polarity
    return "Polaridade: {}".format(polarity)

@app.route('/cotacao/', methods=['POST'])
@basic_auth.required
def cotacao():
    data = request.get_json()
    input_data = [data[col] for col in ['tamanho', 'ano', 'garagem']]
    predict = model.predict([input_data])
    return jsonify(preco=predict[0])

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')