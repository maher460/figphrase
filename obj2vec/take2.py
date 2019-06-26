import csv

TEST_PATH = "../../test-annotations-bbox.csv"
LABELS_PATH = "../../class-descriptions-boxable.csv"
OUTPUT_PATH = "./open_images_test_unique.pkl"

results = {}
labels = {}

with open(LABELS_PATH) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        labels[row[0]] = row[1]

    print("\n\n\n\t\tDONE READING LABELS FILE!\n\n\n")

with open(TEST_PATH) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            # print(row)
            line_count += 1
            if row[0] in results.keys():
                if row[2] not in results[row[0]]:  #TODO: For unique/duplicates
                    results[row[0]].append(row[2])
            else:
                results[row[0]] = [row[2]]
        if line_count % 100000 == 0:
            print(f'Processed {line_count} rows.')

    print(f'Processed {line_count} rows.')
    # print(results)
    
    print("\n\n\n\t\tDONE READING CSV FILE!\n\n\n")

import pickle

with open(OUTPUT_PATH, 'wb') as f:
    pickle.dump([labels, results], f, pickle.HIGHEST_PROTOCOL)

    
# s_counts = 0
# # curr_s_counts = 0
# # curr_s_counts_list = []
# w_counts = {}
# # sent_file_num = 1
# sents = {}

# # sent_file = open(OUTPUT_PATH + "sent." + str(sent_file_num), 'w')

# for item in results.items():

#     # if curr_s_counts >= 150000:
#     #     curr_s_counts_list.append(curr_s_counts)
#     #     curr_s_counts = 0
#     #     sent_file.close()
#     #     sent_file_num += 1
#     #     sent_file = open(OUTPUT_PATH + "sent." + str(sent_file_num), 'w')

#     if (len(item[1]) >= 2 and len(item[1]) <= 39):

#         s_counts += 1
#         # curr_s_counts += 1

#         for label in item[1]:
#             if label in w_counts.keys():
#                 w_counts[label] += 1
#             else:
#                 w_counts[label] = 1
#         temp = list(map(lambda x: labels[x], item[1]))
#         if(len(item[1]) in sents.keys()):
#             sents[len(item[1])].append((item[0]," ".join(item[1]), ", ".join(temp)))
#         else:
#             sents[len(item[1])] = [(item[0]," ".join(item[1]), ", ".join(temp))]

#         if s_counts % 10000 == 0:
#             print(f'Processed {s_counts} lines.')

#     # if(len(item[1]) <= 1):
#     #     print("\n\n\t\tERROR: NUM LABELS IN IMAGE: " + str(len(item[1])) + "\n\n")

# print(f'Processed {s_counts} lines.')

# print("\n\n\n\t\tDONE PROCESSING SENTENCES!\n\n\n")

# # curr_s_counts_list.append(curr_s_counts)
# # sent_file.close()

# sent_meta_file = open(OUTPUT_PATH + "s_counts", 'w')

# for item in sents.items():

#     print(f'Writing sent.{item[0]} lines...')
    
#     sent_file = open(OUTPUT_PATH + "sent." + str(item[0]), 'w')
#     imgl_file = open(OUTPUT_PATH + "image_labels/" + "imgl." + str(item[0]), 'w')
#     for s in item[1]:
#         sent_file.write("%s\n" % s[1])
#         imgl_file.write("%s %s\n" % (s[0],s[2]))
#     sent_file.close()
#     imgl_file.close()

#     sent_meta_file.write("sent.%d\t%d\n" % (item[0], len(item[1])))

# sent_meta_file.close()

# print("\n\n\n\t\tDONE WRITING SENTENCES!\n\n\n")


# # sent_meta_file = open(OUTPUT_PATH + "s_counts", 'w')
# # # sent_meta_file.write("sent.1\t%d\n" % s_counts)
# # for idx in range(len(curr_s_counts_list)):
# #     sent_meta_file.write("sent.%d\t%d\n" % (idx+1, curr_s_counts_list[idx]))
# # sent_meta_file.close()

# words_meta_file = open(OUTPUT_PATH + "w_counts", 'w')
# for item in w_counts.items():
#     words_meta_file.write("%s\t%d\n" % item)

# total_meta_file = open(OUTPUT_PATH + "totals", 'w')
# total_meta_file.write("total sents read: %d\n" % s_counts)
# total_meta_file.write("total words read: %d\n" % len(w_counts.keys()))
# total_meta_file.close()

# print("\n\n\n\t\tDONE WRITING META FILES!\n\n\n")

