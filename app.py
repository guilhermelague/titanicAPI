from flask import Flask, render_template, request, redirect, url_for, jsonify
from loguru import logger
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
global graph
graph = tf.get_default_graph() 
from keras.models import model_from_json

# packets proprio
from core import modelIA

app = Flask(__name__)

# LOAD THE IA MODEL
modeloIA = None
def load_model():
        global modeloIA
        # load json and create model
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        modeloIA = model_from_json(loaded_model_json)
        # load weights into new model
        modeloIA.load_weights("model.h5")
        logger.info("Loaded model from disk")
        # Compile model
        modeloIA.compile(optimizer = 'adadelta', loss = 'categorical_crossentropy', metrics = ['accuracy'])

@app.route('/')
def index():
        return render_template('index.html')


@app.route('/uploader', methods=['POST'])
def receive_file():
        if request.method == 'POST':
                global modeloIA
                try:
                        dataIn = [str(request.form['name']), 
                                float(request.form['age']),
                                int(request.form['parch']),
                                int(request.form['sibsp']),
                                int(request.form['title']),
                                int(request.form['class']),
                                int(request.form['sex']),
                                int(request.form['embarked']),
                                int(request.form['cabinLetter']), 
                                int(request.form['NumberCabin'])]
                except:
                        logger.info("Falha na montagem dos dados de entrada da IA")
                        return render_template('index.html')

                with graph.as_default():
                        answer = modelIA.get_answer(modeloIA, dataIn)
                logger.info(f"Datain: {dataIn} -- Dataout: {answer}")
                return jsonify(result=answer)
                

if __name__ == '__main__':
        load_model()
        app.run(host= '0.0.0.0', port=5000, debug=True)
