
import PIL
import tensorflow as tf
import numpy as np
import os
import pickle


from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.applications import VGG16
from sklearn.preprocessing import StandardScaler

VERBOSE = True

ad_images_path = "../../ad_images/"

def getFileNames(path):
    if VERBOSE:
        print("[MaherBot] Getting filenames from path: " + path)
    return sorted(os.listdir(path))

def loadImage(path):
    temp = PIL.Image.open(path)
    img = temp.copy()
    temp.close()
    return img

filenames = getFileNames(ad_images_path)

ids = list(map(lambda x: x.split(".")[0], filenames))

model = VGG16(include_top=True, weights='imagenet')

input_shape = model.layers[0].output_shape[1:3]

print(model.layers[0].output_shape)
print(input_shape)

imgs = list(map(lambda x: loadImage(ad_images_path + x), filenames))

r_imgs = list(map(lambda x: x.resize(input_shape,PIL.Image.LANCZOS), imgs))

np_imgs = list(map(lambda x: np.array(x), r_imgs))

np_imgs = list(map(lambda x: x.astype('float'), np_imgs))

# for n in np_imgs: print(n.shape)

ids_imgs = list(filter(lambda x: x[1].shape == (224, 224, 3), zip(ids, np_imgs)))

ids = list(map(lambda x: x[0], ids_imgs))

np_imgs = list(map(lambda x: x[1], ids_imgs))

np_imgs = np.array(np_imgs)
print("blabla")
print(np_imgs.shape)

# i.resize(input_shape,PIL.Image.LANCZOS)

# img = np.array(img)
# img = img.astype('float')


layer_name = 'fc2'
fc2_layer_model = Model(inputs=model.input,
                        outputs=model.get_layer(layer_name).output)

features = fc2_layer_model.predict(np_imgs)

scaler = StandardScaler().fit(features)
sfeatures = scaler.transform(features)

results = {}

for i in range(len(ids)):
	results[ids[i]] = sfeatures[i]

with open("data_img_features.pkl", 'wb') as f:
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
