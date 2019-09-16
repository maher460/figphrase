# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers.core import Dense

# Helper libraries
import numpy as np
import pickle

print(tf.__version__)



with open('tester_data_ready.pkl', 'rb') as f:
    temp = pickle.load(f)

data_X = np.array(temp[0])
data_Y = np.array(temp[1])


# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28)),
#     keras.layers.Dense(128, activation=tf.nn.relu),
#     keras.layers.Dense(10, activation=tf.nn.softmax)
# ])

model = Sequential()
model.add(Dense(500, input_dim=500, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(data_X[0:600], data_Y[0:600], epochs=10)


test_loss, test_acc = model.evaluate(data_X[600:], data_Y[600:])

print('Test accuracy:', test_acc)



# predictions = model.predict(test_images)


# np.argmax(predictions[0])