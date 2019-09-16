# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# import livelossplot

from keras.models import Sequential
from keras.layers.core import Dense

# Helper libraries
import numpy as np
import pickle
import datetime

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
model.add(Dense(128, input_dim=500, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
# model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='softmax'))
# model.add(Dense(1, activation='sigmoid'))

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

class PlotLossesCallback(livelossplot.keras.PlotLossesCallback):
    def on_train_batch_begin(self, a, b): pass
    def on_train_batch_end(self, a, b): pass

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
# tensorboard_callback.set_model(model)

cw = {1:1.0, 0:2.577}

model.fit(data_X[0:551], data_Y[0:551], class_weight=cw, validation_split=0.1, epochs=25, callbacks=[tensorboard_callback])


test_loss, test_acc = model.evaluate(data_X[551:], data_Y[551:])

print('Test accuracy:', test_acc)



predictions = model.predict(data_X[551:])
print(predictions)


# np.argmax(predictions[0])