from keras.models import model_from_json
import pandas as pd
import numpy as np

#name,age,parch,sibsp,title,class,sex,embarked,cabinLetter,NumberCabin
dataIn = ["João", 22.0,0,1,12,3,2,3,8,200]

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(optimizer = 'adadelta', loss = 'categorical_crossentropy', metrics = ['accuracy'])
data = np.array(dataIn[1:]).reshape(1,-1)
print(data)
print(data.shape)
p_survived = loaded_model.predict_classes(data)

