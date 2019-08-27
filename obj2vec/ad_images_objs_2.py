import csv
import gensim
from numpy import float64
import numpy as np
from gensim import matutils

# AD_IMGS_OBJS_PATH = "../../Keras-RetinaNet-for-Open-Images-Challenge-2018/subm/retinanet_level_1_all_levels.csv"
AD_IMGS_OBJS_PATH = "../../Keras-RetinaNet-for-Open-Images-Challenge-2018/subm/retinanet_training_level_1.csv"
AD_IMGS_ANNS_PATH = "../../ads_parallelity_dataset.csv"
LABELS_PATH = "../../class-descriptions-boxable.csv"
GOOGLENEWSMODEL_PATH = "../../GoogleNews-vectors-negative300.bin"

model = gensim.models.KeyedVectors.load_word2vec_format(GOOGLENEWSMODEL_PATH, binary=True)

labels = {}

with open(LABELS_PATH) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        labels[row[0]] = row[1]

    print("\n\n\n\t\tDONE READING LABELS FILE!\n\n\n")


res1 = {} # {img_id: [object_labels]}
# res1_b = {}

with open(AD_IMGS_OBJS_PATH) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            # print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            # print(row)
            line_count += 1
            
            temp1 = row[1].split(" ")
            composite_list = [temp1[x:x+6] for x in range(0, len(temp1),6)]

            temp1b = list(filter(lambda x: len(x)==6, composite_list))

            temp1c = list(filter(lambda x: float(x[1]) >= 0.25, temp1b))

            temp1d = list(map(lambda x: x[0], temp1c))
            temp2 = list(set(temp1d))

            res1[row[0]] = temp2

            # w2v_labels = []
            # for obj_label in temp2:
            #     if labels[obj_label] not in w2v_labels and labels[obj_label] in model:
            #         w2v_labels.append(model[labels[obj_label]])
            # res1_b[row[0]] = w2v_labels


            # if row[0] in results.keys():
            #     # if row[2] not in results[row[0]]:  #TODO: For unique/duplicates
            #     results[row[0]].append(row[2])
            # else:
            #     results[row[0]] = [row[2]]
        # if line_count % 100000 == 0:
    print(f'Processed {line_count} rows.')

# print(res1)

res2 = {}  #{img_id: ([transcriptions],[parallelities])}

with open(AD_IMGS_ANNS_PATH) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            # print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            # print(row)
            line_count += 1

            key = row[0].rsplit(".", 1)[0]
            # print(key)
            
            if key in res2.keys():
                #bla
                res2[key][0].append(row[1])
                res2[key][1].append(row[6])
            else:
                res2[key] = ([row[1]], [row[6]])

    # print(res2)

labels_new = {}
labels_model = {} #classes which are in model
labels_model_flipped = {}

count_123 = 0
for c in labels.keys():
    count_123 += 1
    print("Synonyms: " + str(count_123))
    c_label = labels[c].lower()
    c_label_joined = c_label.replace(" ", "_")
    if c_label_joined in model:
        labels_model[c_label_joined] = c
        labels_model_flipped[c] = c_label_joined
        similar_labels_tuples = model.most_similar(positive=[c_label_joined])
        similar_labels = list(map(lambda x: x[0].replace("_", " "), similar_labels_tuples))
        similar_labels.append(c_label)
    else:
        similar_labels = [c_label]
    labels_new[c] = similar_labels 

res1_b = {}
for key in res1.keys():
    for obj in res1[key]:
        if obj in labels_model_flipped:
            if key in res1_b.keys():
                res1_b[key].append(labels_model_flipped[obj])
            else:
                res1_b[key] = [labels_model_flipped[obj]]

res1_b_means = {}
for key in res1_b.keys():
    mean = list(map(lambda m: model.word_vec(m, use_norm=True), res1_b[key]))
    mean = matutils.unitvec(np.array(mean).mean(axis=0)).astype(float64)
    res1_b_means[key] = mean



res3 = {} # {img_id: [object_labels in transcriptions]}
res3_b = {}
count_321 = 0
for k in res2.keys():
    ## Method A 
    # for c in labels_new.keys():
    #     count_321 += 1
    #     print("Searching: " + str(count_321))
    #     # c_label = labels[c].lower()
    #     # c_label_joined = c_label.replace(" ", "_")
    #     # if c_label_joined in model:55
    #     #     similar_labels_tuples = model.most_similar(positive=[c_label_joined])
    #     #     similar_labels = list(map(lambda x: x[0].replace("_", " "), similar_labels_tuples))
    #     #     similar_labels.append(c_label)
    #     # else:
    #     #     similar_labels = [c_label]
        
    #     for t in res2[k][0]:
    #         for s in labels_new[c]:
    #             if s in t:
    #         # if labels[c].lower() in t:
    #                 if k in res3.keys():
    #                     res3[k].add(c)
    #                 else:
    #                     res3[k] = {c}
    
    ## Method B
    
    # count_321 += 1
    # print("Searching: " + str(count_321))
    # mean = []
    # for t in res2[k][0]:
    #     temp_bla = t.split()
    #     for b in temp_bla:
    #         if b not in mean and b in model:
    #             mean.append(b)

    # if len(mean) > 0:
    #     mean = list(map(lambda m: model.word_vec(m, use_norm=True), mean))

    #     mean = matutils.unitvec(np.array(mean).mean(axis=0)).astype(float64)

    #     dists = model.distances(mean, labels_model.keys())

    #     smallest_idx = np.argmin(dists)

    #     res3[k] = {labels_model[list(labels_model.keys())[smallest_idx]]}

    ## Method C

    count_321 += 1
    print("Searching: " + str(count_321))
    mean = []
    for t in res2[k][0]:
        temp_bla = t.split()
        for b in temp_bla:
            if b not in mean and b in model:
                mean.append(b)
    
    res3_b[k] = mean

    if len(mean) > 0:
        mean = list(map(lambda m: model.word_vec(m, use_norm=True), mean))

        mean = matutils.unitvec(np.array(mean).mean(axis=0)).astype(float64)

        res3[k] = mean

        # dists = model.distances(mean, labels_model.keys())

        # smallest_idxs = list(np.argsort(dists)[:10])

        # # res3[k] = {labels_model[list(labels_model.keys())[smallest_idx]]}

        # lm_keys = list(labels_model.keys())

        # resultz = list(map(lambda x: labels_model[lm_keys[x]], smallest_idxs))

        # res3[k] = set(resultz)




# print(res3)
# print(len(res3.keys()))


res4 = {}  # {img_id: parallel/non-parallel}

for k in res2.keys():
    p_c = 0
    np_c = 0
    for p in res2[k][1]:
        if p == 'parallel':
            p_c += 1
        else:
            np_c += 1
    if p_c > np_c:
        res4[k] = 'parallel'
    else:
        res4[k] = 'non_parallel'

with open('ids_match.pkl', 'rb') as f:
    ids_match = pickle.load(f)

blabla = []
for key in res1_b_means.keys():
    if key in ids_match and key in res3.keys():
        similarity = np.dot(matutils.unitvec(res1_b_means[key]), matutils.unitvec(res3[key]))
        blabla.append((similarity, res4[key], key))

temp_thres = 0
cur_thres = -1
cur_max = -1
min_thres = -1
max_thres = -1

temp_loss = 0

while temp_thres < 100:

    for b in blabla:
        if b[1] == 'parallel':
            temp_loss += (b[0] - temp_thres)
        if b[1] == 'non_parallel':
            temp_loss += (temp_thres - b[0])

    if temp_loss > cur_max:
        cur_thres = temp_thres
        cur_max = temp_loss

    # total_c = 0
    # total_w = 0

    # for b in blabla:
    #     if b[1] == 'parallel' and b[0] > temp_thres:
    #         total_c += 1
    #     elif b[1] == 'non_parallel' and b[0] < temp_thres:
    #         total_c += 1
    #     else:
    #         total_w += 1

    # if total_c == cur_max:
    #     if min_thres > -1:
    #         max_thres

    temp_thres += 0.001

print("cur_thres: " + str(cur_thres))

total_c = 0
total_w = 0

ids_match3 = []

import csv 
with open('ad_images_objs_2.csv', 'w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    # for line in data:
    writer.writerow(['img_id', 'correct/wrong', 'prediction', 'score(threshold:'+str(cur_thres)+')', 'image_w2v_objects', 'transcription_w2v_objects'])

    for b in blabla:
        ids_match3.append(b[2])
        img_obj_labels = res1_b[b[2]] #list(map(lambda x: labels[x], res1[b[2]]))
        t_obj_labels = res3_b[b[2]] #list(map(lambda x: labels[x], res3[b[2]]))
        if (b[1] == 'parallel' and b[0] >= cur_thres) or (b[1] == 'non_parallel' and b[0] < cur_thres):
            total_c += 1
            writer.writerow([b[2], 'correct', b[1], str(b[0]), "\t".join(img_obj_labels), "\t".join(t_obj_labels)])
        else:
            total_w += 1
            writer.writerow([b[2], 'wrong', b[1], str(b[0]), "\t".join(img_obj_labels), "\t".join(t_obj_labels)])

with open("ids_match3.pkl", 'wb') as f:
    pickle.dump(ids_match3, f, pickle.HIGHEST_PROTOCOL)

print("total_c: " + str(total_c))
print("total_w: " + str(total_w))
print("total_c_w: " + str(total_c+total_w))
print("accuracy: " + str(total_c / (total_c+total_w)))
print("negative_accuracy: " + str(total_w / (total_c+total_w)))

# print(res4)
            
# import pickle

# with open("bla3_method_c_10.pkl", 'wb') as f:
#     pickle.dump([labels, res1, res2, res3, res4], f, pickle.HIGHEST_PROTOCOL)

# def save_obj(obj, name ):
#     with open('obj/'+ name + '.pkl', 'wb') as f:
#         pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# def load_obj(name ):
#     with open('obj/' + name + '.pkl', 'rb') as f:
#         return pickle.load(f)

