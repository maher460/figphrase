import csv
import gensim
from numpy import float64
import numpy as np
from gensim import matutils
from gensim.summarization import keywords
import nltk

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 


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
model_words = [] # store all the words found in the transcripts to this list

count_123 = 0
for c in labels.keys():
    count_123 += 1
    print("Synonyms: " + str(count_123))
    c_label = labels[c].lower()
    c_label_joined = c_label.replace(" ", "_")
    if c_label_joined in model:
        labels_model[c_label_joined] = c
        similar_labels_tuples = model.most_similar(positive=[c_label_joined])
        similar_labels = list(map(lambda x: x[0].replace("_", " "), similar_labels_tuples))
        similar_labels.append(c_label)
    else:
        similar_labels = [c_label]
    labels_new[c] = similar_labels 



stop_words = set(stopwords.words('english'))  

res3 = {} # {img_id: [object_labels in transcriptions]}
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
        print(t)
        t = t.lower()
        word_tokens = word_tokenize(t)
        word_tokens = list(map(lambda x: x.lower(), word_tokens)) 
        temp_bla = [w for w in word_tokens if not w in stop_words]
        #temp_bla = keywords(t, ratio=1.0, split=True, lemmatize=True) #keeping just keywords (old 2)
        #temp_bla = t.split() #old
        print(temp_bla)
        for b in temp_bla:
            if b not in mean and b in model:
                mean.append(b)
                model_words.append(b)
                
    if len(mean) > 0:
        mean2 = list(map(lambda m: model.word_vec(m, use_norm=True), mean))

        # mean2 = matutils.unitvec(np.array(mean2).mean(axis=0)).astype(float64)

        dists = list(map(lambda m: model.distances(m, labels_model.keys()), mean2))

        smallest_idxs = list(map(lambda d: (np.argsort(d)[0], d[np.argsort(d)[0]]), dists)) 

        # res3[k] = {labels_model[list(labels_model.keys())[smallest_idx]]}

        lm_keys = list(labels_model.keys())

        # print(smallest_idxs)
        # print(lm_keys)

        # resultz = list(map(lambda s: list(map(lambda x: labels_model[lm_keys[x]], s)), smallest_idxs)) 
        resultz = list(map(lambda s: (labels_model[lm_keys[s[0]]], s[1]), smallest_idxs)) 

        res3[k] = zip(mean, resultz)





print(res3)
print(len(res3.keys()))


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

# print(res4)

model_words.extend(labels_model.keys()) # add all the words found in the object labels

model_words = list(set(model_words)) # get all unique
model_labels = {} # make Object Label -> object_label dictionary

for k in labels_model.keys():
    model_labels[labels[labels_model[k]]] = k

res_w2v_d = {} # store w2v distance of all pairs in 2D dictionary 

for k1 in model_words:
    for k2 in model_words:
        dist = model.distance(k1, k2)
        if k1 not in res_w2v_d.keys():
            res_w2v_d[k1] = {}
        res_w2v_d[k1][k2] = dist
        if k2 not in res_w2v_d.keys():
            res_w2v_d[k2] = {}
        res_w2v_d[k2][k1] = dist




            
import pickle

with open("bla3_method_matrix_1.pkl", 'wb') as f:
    pickle.dump([labels, res1, res2, res3, res4, model_labels, res_w2v_d], f, pickle.HIGHEST_PROTOCOL)

# def save_obj(obj, name ):
#     with open('obj/'+ name + '.pkl', 'wb') as f:
#         pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# def load_obj(name ):
#     with open('obj/' + name + '.pkl', 'rb') as f:
#         return pickle.load(f)

