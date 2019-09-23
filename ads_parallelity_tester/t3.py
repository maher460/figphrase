# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
# import livelossplot

from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import plot_model

# Helper libraries
import numpy as np
import pickle
import datetime
import matplotlib.pyplot as plt

print(tf.__version__)



with open('data_train_test_X_Y.pkl', 'rb') as f:
    data_dicts = pickle.load(f)

# data_X = np.array(temp[0])
# data_Y = np.array(temp[1])


# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28)),
#     keras.layers.Dense(128, activation=tf.nn.relu),
#     keras.layers.Dense(10, activation=tf.nn.softmax)
# ])

model = Sequential()
model.add(Dense(4396, input_dim=4396, activation='relu'))
# model.add(Dense(500, activation='relu'))
# model.add(Dense(500, activation='relu'))
# model.add(Dense(1024, activation='relu'))
# model.add(Dense(500, activation='relu'))
# model.add(Dense(500, activation='relu'))
# model.add(Dense(256, activation='relu'))
# model.add(Dense(500, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.add(Dense(1, activation='sigmoid'))

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

plot_model(model, to_file='model.png', show_shapes=True, expand_nested=True, rankdir='LR')

# class custom_callback(keras.callbacks.TensorBoard):

#   def on_train_batch_begin(self, *args, **kwargs):
#     pass

#   def on_train_batch_end(self, *args, **kwargs):
#     pass

#   def on_test_batch_begin(self, *args, **kwargs):
#     pass

#   def on_test_batch_end(self, *args, **kwargs):
#     pass

#   def on_test_begin(self, *args, **kwargs):
#     pass

#   def on_test_end(self, *args, **kwargs):
#     pass

# log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = custom_callback(log_dir=log_dir, histogram_freq=1)
# tensorboard_callback.set_model(model)



cw = {1:1.0, 0:3.183}

accuracies = []

for i in range(len(data_dicts)):

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        
        train_text_X = data_dicts[i]["train_X"]
        train_img_X = data_dicts[i]["train_Z"]
        train_Y = np.array(data_dicts[i]["train_Y"])

        test_text_X = data_dicts[i]["test_X"]
        test_img_X = data_dicts[i]["test_Z"]
        test_Y = np.array(data_dicts[i]["test_Y"])

        train_X = np.array(list(map(lambda x: np.concatenate((x[0][200:],x[1])), zip(train_text_X, train_img_X))))
        test_X = np.array(list(map(lambda x: np.concatenate((x[0][200:],x[1])), zip(test_text_X, test_img_X))))

        print("train_X.shape: "+str(train_X.shape))
        print("train_Y.shape: "+str(train_Y.shape))
        print("test_X.shape: "+str(test_X.shape))
        print("test_Y.shape: "+str(test_Y.shape))

        history = model.fit(train_X, train_Y, class_weight=cw, validation_split=0.1, epochs=30) #, callbacks=[tensorboard_callback])

        # # Plot training & validation accuracy values
        # plt.plot(history.history['acc'])
        # plt.plot(history.history['val_acc'])
        # plt.title('Model accuracy')
        # plt.ylabel('Accuracy')
        # plt.xlabel('Epoch')
        # plt.legend(['Train', 'Test'], loc='upper left')
        # plt.savefig("acc.png")

        # # Plot training & validation loss values
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('Model loss')
        # plt.ylabel('Loss')
        # plt.xlabel('Epoch')
        # plt.legend(['Train', 'Test'], loc='upper left')
        # plt.savefig("loss.png")
        
        # print(history)


        test_loss, test_acc = model.evaluate(test_X, test_Y)

        print('Test accuracy (run '+ str(i+1) +'): ', test_acc)

        accuracies.append(test_acc) 

        # print("\nGround Truth: ")
        # print(test_Y)
        # print("\nPredictions: ")
        # predictions = model.predict(test_X)
        # print(predictions)

avg_accuracy = float(sum(accuracies)) / len(accuracies)

print(accuracies)
print(avg_accuracy)
# np.argmax(predictions[0])