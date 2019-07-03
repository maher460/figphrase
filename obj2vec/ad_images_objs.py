import csv

AD_IMGS_OBJS_PATH = "../../Keras-RetinaNet-for-Open-Images-Challenge-2018/subm/retinanet_level_1_all_levels.csv"
AD_IMGS_ANNS_PATH = "../../ads_parallelity_dataset.csv"
LABELS_PATH = "../../class-descriptions-boxable.csv"


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
            temp2 = list(filter(lambda x: '/m/' in x, temp1))

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


res3 = {} # {img_id: [object_labels in transcriptions]}

for k in res2.keys():
	for c in labels.keys():
		for t in res2[k][0]:
			if labels[c].lower() in t:
				if k in res3.keys():
					res3[k].add(c)
				else:
					res3[k] = {c}

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

# print(res4)
            
import pickle

with open("bla.pkl", 'wb') as f:
	pickle.dump([labels, res1, res2, res3, res4], f, pickle.HIGHEST_PROTOCOL)

# def save_obj(obj, name ):
#     with open('obj/'+ name + '.pkl', 'wb') as f:
#         pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# def load_obj(name ):
#     with open('obj/' + name + '.pkl', 'rb') as f:
#         return pickle.load(f)

