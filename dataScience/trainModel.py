import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.metrics import categorical_accuracy
import pandas as pd


dataIn = pd.read_csv("dataIn.csv")

X = dataIn.iloc[:,0:9].values
y = dataIn.iloc[:,9:10].values

model = Sequential()
model.add(Dense(input_dim=X.shape[1], units=X.shape[1], use_bias=True, activation = 'relu'))
model.add(Dense(units=8, activation = 'relu', use_bias=True))
model.add(Dense(units=4, activation = 'relu', use_bias=True))
model.add(Dense(units=2, activation = 'relu', use_bias=True))
model.add(Dense(output_dim = 1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = [categorical_accuracy])

model.fit(X, y, batch_size = 64, epochs = 100, validation_split=0.05, verbose=1)

p_survived = model.predict_classes(X)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
