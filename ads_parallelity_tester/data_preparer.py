from numpy import float64
import numpy as np
from gensim import matutils
import pickle
import random

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

with open('data_img_features.pkl', 'rb') as f:
    img_features = pickle.load(f)

new = {}
keys =  list(w2c_data.keys())

img_keys = img_features.keys()

keys = list(map(lambda x: x in img_keys, img_features))

keys.sort()

data_X = []
data_Z = []
for k in keys:
	if k in tester_o2c:
		# print("\n")
		# print(tester_o2c[k].shape)
		# print(w2c_data[k].shape)
		data_X.append(np.concatenate((tester_o2c[k], w2c_data[k])))
		data_Z.append(img_features[k])
		# print(data_X[-1].shape)

data_Y = []
count_p = 0
count_np = 0
idx_p_list = []
idx_np_list = []
for k in keys:
	if labels[k] == "parallel":
		count_p += 1
		idx_p_list.append(len(data_Y))
		data_Y.append(1)
	else:
		count_np += 1
		idx_np_list.append(len(data_Y))
		data_Y.append(0)

print(data_Y)
print(count_p)
print(count_np)
print("len(keys): " + str(len(keys)))
print("len(w2c_data.keys()): " + str(len(w2c_data.keys())))
print("len(tester_o2c.keys()): " + str(len(tester_o2c.keys())))
print("len(data_X): " + str(len(data_X)))
print("len(data_Y): " + str(len(data_Y)))
print("len(data_Z): " + str(len(data_Z)))

# with open("tester_data_X_Y.pkl", 'wb') as f:
#     pickle.dump([data_X, data_Y], f, pickle.HIGHEST_PROTOCOL)


def cb_yes(x, p_list, np_list):
	return (x[0] in p_list) or (x[0] in np_list)

def cb_no(x, p_list, np_list):
	return (x[0] not in p_list) and (x[0] not in np_list)


data_dicts = []

for i in range(10):

	test_idx_p_list = random.sample(idx_p_list, 50)
	test_idx_np_list = random.sample(idx_np_list, 50)

	test_X = list(filter(lambda x: cb_yes(x, test_idx_p_list, test_idx_np_list), enumerate(data_X)))
	test_Y = list(filter(lambda x: cb_yes(x, test_idx_p_list, test_idx_np_list), enumerate(data_Y)))
	test_Z = list(filter(lambda x: cb_yes(x, test_idx_p_list, test_idx_np_list), enumerate(data_Z)))

	train_X = list(filter(lambda x: cb_no(x, test_idx_p_list, test_idx_np_list), enumerate(data_X)))
	train_Y = list(filter(lambda x: cb_no(x, test_idx_p_list, test_idx_np_list), enumerate(data_Y)))
	train_Z = list(filter(lambda x: cb_no(x, test_idx_p_list, test_idx_np_list), enumerate(data_Z)))

	test_X = list(map(lambda x: x[1], test_X))
	test_Y = list(map(lambda x: x[1], test_Y))
	test_Z = list(map(lambda x: x[1], test_Z))
	train_X = list(map(lambda x: x[1], train_X))
	train_Y = list(map(lambda x: x[1], train_Y))
	train_Z = list(map(lambda x: x[1], train_Z))

	data_dict = {}
	data_dict["test_X"] = test_X
	data_dict["test_Y"] = test_Y
	data_dict["test_Z"] = test_Z
	data_dict["train_X"] = train_X
	data_dict["train_Y"] = train_Y
	data_dict["train_Z"] = train_Z

	data_dicts.append(data_dict)

with open("data_train_test_X_Y.pkl", 'wb') as f:
    pickle.dump(data_dicts, f, pickle.HIGHEST_PROTOCOL)

