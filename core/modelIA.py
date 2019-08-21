# Import the libraries
import numpy as np
from keras.models import model_from_json

class People:
    def __init__(self, name, sex, state):
        self.name = name
        if sex == 1:
            self.sex  = "Feminino"
        elif sex == 2:
            self.sex  = "Masculino"
        self.state = state

def get_answer(modelo, dataIn):
    isAlive = modelo.predict_classes(np.array(dataIn[1:]).reshape(1,-1))
    #return [People(dataIn[0], dataIn[6], isAlive)]
    return int(isAlive)

