from numpy import float64
import numpy as np
from gensim import matutils
import pickle

with open('tester_w2c.pkl', 'rb') as f:
    tester_w2c = pickle.load(f)

w2c_data = {}
for k in tester_w2c.keys():
	mean = tester_w2c[k]
	mean = matutils.unitvec(np.array(mean).mean(axis=0)).astype(float64)
	w2c_data[k] = mean

with open('tester_o2c.pkl', 'rb') as f:
    tester_o2c = pickle.load(f)

with open('tester_labels.pkl', 'rb') as f:
    labels = pickle.load(f)

new = {}
keys =  list(w2c_data.keys())

keys.sort()

data_X = []
for k in keys:
	if k in tester_o2c:
		data_X.append(np.concatenate((tester_o2c[k], w2c_data[k])))

data_Y = []
for k in keys:
	data_Y.append(labels[k])


print("len(keys): " + str(len(keys)))
print("len(w2c_data.keys()): " + str(len(w2c_data.keys())))
print("len(tester_o2c.keys()): " + str(len(tester_o2c.keys())))
print("len(data_X): " + str(len(data_X)))
print("len(data_X): " + str(len(data_X)))
print("len(data_Y): " + str(len(data_Y)))