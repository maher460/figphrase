import numpy as np
import six
import sys
import traceback
import re
import argparse
import statistics

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn import preprocessing
from figpro.evaluation.model_reader2 import ModelReader
import warnings

from shutil import copyfile

warnings.filterwarnings("ignore", category=DeprecationWarning)


class ParseException(Exception):
    def __init__(self, str):
        super(ParseException, self).__init__(str)


target_exp = re.compile('\[.*\]')


def parse_input(line):
    sent = line.strip().split()
    target_pos = None
    for i, word in enumerate(sent):
        if target_exp.match(word) != None:
            target_pos = i
            if word == '[]':
                word = None
            else:
                word = word[1:-1]
            sent[i] = word
    return sent, target_pos


def parse_input_dumpstring(line):
    sent = line.strip().split()
    target_pos = None
    for i, word in enumerate(sent):
        if word.find('dumpstring') >= 0:
            target_pos = i
            word = None
    return sent, target_pos


def mult_sim(w, target_v, context_v):
    target_similarity = w.dot(target_v)
    target_similarity[target_similarity < 0] = 0.0
    context_similarity = w.dot(context_v)
    context_similarity[context_similarity < 0] = 0.0
    return (target_similarity * context_similarity)

def generate_vector_phrase(idiom):
    components = idiom.split("_")
    avg = np.zeros((0, embedding_size))
    for i, word in enumerate(components):
        v = model.get_target_vec(word)
        # if i == 0:
        #     v = v * (1/4)
        # print (v.shape)
        avg = np.vstack([avg, v])
    # print avg.shape
    avg = np.mean(avg, axis=0)
    avg = preprocessing.normalize(avg.reshape(1, -1), norm='l2')[0]
        # print avg
    return avg

def generate_vector_word(word):
    v = model.get_target_vec(word)
    avg = preprocessing.normalize(v.reshape(1, -1), norm='l2')[0]
    return avg

def usage_rec(context_v, lit_v):
    #print (context_v)
    #print (lit_v)
    lit_sim = (context_v * lit_v).sum()
    #fig_sim = (context_v * fig_v).sum()
    #print lit_sim
    #print fig_sim
    # use local attention, -0.11 ~ -0.15 all are ok
    return lit_sim

def usage_rec2(context, vec):
    context = context / np.sqrt((context * context).sum())
    vec = vec / np.sqrt((vec * vec).sum())
    lit_sim = (context * vec).sum()
    #print (context_v)
    #print (lit_v)
    # lit_sim = (context_v * lit_v).sum()
    #fig_sim = (context_v * fig_v).sum()
    #print lit_sim
    #print fig_sim
    # use local attention, -0.11 ~ -0.15 all are ok
    return lit_sim

# saved_model_path = '../model-save/semcomp.meta'
# corpus_patch = "../../obj2vec/open_images_corpus.DIR"

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpuspath', '-c',
                        default="../../obj2vec/open_images_corpus.DIR",
                        help='input corpus directory')
    parser.add_argument('--modelfile', '-m',
                        default=None,
                        help='saved model file')

    parser.add_argument('--trimfreq', '-t', default=0, type=int,
                        help='minimum frequency for word in training')

    parser.add_argument('--unit', '-u', default=200, type=int,
                        help='number of units (dimensions) of one context word')

    args = parser.parse_args()

    return args

args = parse_arguments()

embedding_size = args.unit


trim_freq = args.trimfreq

saved_model_path = args.modelfile

corpus_patch = args.corpuspath

model = ModelReader(saved_model_path, corpus_patch, trim_freq)
word2index = model.word2index
index2word = model.index2word


import pickle

with open('../obj2vec/bla3_method_matrix_1.pkl', 'rb') as f:
    res_bla = pickle.load(f)

labels = res_bla[0] # {obj_label_id: obj_label_name}
res1 = res_bla[1] # {img_id: [object_labels]}
res2 = res_bla[2] # {img_id: ([transcriptions],[parallelities])}
res3 = res_bla[3] # {img_id: [object_labels in transcriptions]}
res4 = res_bla[4] # {img_id: parallel/non-parallel}
model_labels = res_bla[5] # {Object Label: object_label}
res_w2v_d = res_bla[6] # {word: {word: distance}}

# with open('ids_match3.pkl', 'rb') as f:
#     ids_match3 = pickle.load(f)

# print(res_bla)
total = 0
sum_p = {'parallel':0, 'non_parallel':0}
count_p = {'parallel':0, 'non_parallel':0}

blabla = []

tester_o2c = {}

# hola = {}
# count_bla = 3

PE = [2842,7896,11674,12181,13058,13498,18998,19772,24824,29359,108182,100042,106665,10538,110518,126706,45971,176012,176413,143720]
PE = list(map(lambda x: str(x), PE))

PH = [833,880,1300,2580,3288,5554,5796,5853,7541,7722,10380,11727,11826,15069,19361,29143,29582,100007,107320,111534]
PH = list(map(lambda x: str(x), PH))

NPE = [407,1355,3749,5331,8422,10399,11132,12736,17834,19012,25115,105700,142678,112513,178196,95875,933301,31410,8422,175863]
NPE = list(map(lambda x: str(x), NPE))

NPH = [3146,10470,21598,25917,45364,45406,54469,70789,72112,72926,88939,96399,96923,98985,116184,119904,132190,133468,173671,177245]
NPH = list(map(lambda x: str(x), NPH))

xPE = []
xPH = []
xNPE = []
xNPH = []

for k in res3.keys():
    # count_bla = count_bla - 1
    res3[k] = list(res3[k])
    if len(res3[k]) > 0 and k in res1.keys() and len(res1[k]) > 0 and k in res2.keys() and len(res2[k]) > 0 and (k in PE or k in PH or k in NPE or k in NPH):

        # hola[k] = []
        # col_labels = list(map(lambda x: x[0]+" ("+labels[x[1][0]] +", " + str(round(x[1][1],2))+")", res3[k]))
        col_labels = []
        row_labels = [] #list(map(lambda x: labels[x], res1[k]))
        t_cells = []

        for i_obj in res1[k]:
            if labels[i_obj] in model_labels.keys():
                t_cells_row = []
                for t_obj in res3[k]:

                    if t_obj[1][1] < 0.65 or 1==1:

                        col_labels.append(t_obj[0]+" ("+labels[t_obj[1][0]] +", " + str(round(t_obj[1][1],2))+")")

                        compatibility = -1


                        total += 1

                        # temp1 = "_".join(res3[k])
                        # print("temp1: "+temp1)
                        # temp2 = " ".join(res1[k])
                        # print("temp2: "+temp2)
                        # temp3 = i_obj + " [" + t_obj[1] + "]"
                        # print("temp3: "+temp3)
                        # line = temp3

                        l1 = i_obj
                        l2 = t_obj[1][0]

                        try:

                            vec_1 = generate_vector_word(l1)
                            vec_2 = generate_vector_word(l2)

                            # compatibility1 = (usage_rec(vec_1, vec_2)+1)/2

                            compatibility2 = ((usage_rec2(vec_1, vec_2)+1)/2) * (1.0 - t_obj[1][1])

                            w3 = model_labels[labels[i_obj]]
                            w4 = t_obj[0]

                            comp_w3_w4 = (max(1.0 - res_w2v_d[w3][w4], 0.0)) * (1.0 - t_obj[1][1])

                            w5 = model_labels[labels[i_obj]]
                            w6 = model_labels[labels[t_obj[1][0]]]

                            comp_w5_w6 = (max(1.0 - res_w2v_d[w5][w6], 0.0)) * (1.0 - t_obj[1][1])


                            # compatibility3 = vec_1.dot(vec_2)


                            # # line = six.moves.input('>> ')
                            # sent, target_pos = parse_input(line)
                            # if target_pos == None:
                            #     raise ParseException("Can't find the target position.")

                            # context_v = None
                            # if len(sent) > 1:
                            #     context_v = model.context_rep(sent, target_pos)
                            #     context_v = context_v / np.sqrt((context_v * context_v).sum())

                            #     tester_o2c[k] = context_v

                            # if sent[target_pos] == None:
                            #     if context_v is not None:
                            #         m = model.most_fit_context2(context_v)
                            #     else:
                            #         raise ParseException("Can't find a context.")
                            # else:
                            #     if sent[target_pos].find("_") < 0:
                            #         if sent[target_pos] not in word2index:
                            #             raise ParseException("Target word is out of vocabulary.")
                            #         else:
                            #             target_v = generate_vector_word(sent[target_pos])
                            #     else:
                            #         target_v = generate_vector_phrase(sent[target_pos])
                            #     if target_v is not None and context_v is not None:
                            #         compatibility = (usage_rec2(target_v, context_v)+1)/2
                                    # print("Compatibility score: " + str(compatibility))
                                    # print("Truth: " + res4[k])
                                    
                                    # t_cells_row.append(compatibility) 
                                    
                                    # sum_p[res4[k]] += compatibility
                                    # count_p[res4[k]] += 1
                                    # blabla.append((compatibility, res4[k], k))

                        except EOFError:
                            break
                        except ParseException as e:
                            print("ParseException: {}".format(e))
                        except Exception:
                            exc_type, exc_value, exc_traceback = sys.exc_info()
                            print("*** print_tb:")
                            traceback.print_tb(exc_traceback, limit=1, file=sys.stdout)
                            print("*** print_exception:")
                            traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)

                        # compatibility_str = str(round(compatibility1, 3)) + " " + str(round(compatibility2, 3)) + " " + str(round(compatibility3, 3))
                        compatibility2 = round(compatibility2, 2)
                        comp_w3_w4 = round(comp_w3_w4, 2)
                        comp_w5_w6 = round(comp_w5_w6, 2)

                        compatibility_str = str(compatibility2) + ", " + str(comp_w3_w4) + ", " + str(comp_w5_w6)

                        t_cells_row.append(compatibility_str) 

                if len(t_cells_row) > 0:
                    row_labels.append(labels[i_obj])
                    t_cells.append(t_cells_row)

        # max_val = 0.0
        # min_val = 1.0
        new_t_cells = []
        new_t_cells_w3_w4 = []
        new_t_cells_w5_w6 = []
        for c_row in t_cells:
            for c_col in c_row:
                # if t_cells[c_row][c_col] > max_val:
                #     max_val = t_cells[c_row][c_col]
                # if t_cells[c_row][c_col] < min_val:
                #     min_val = t_cells[c_row][c_col]
                new_t_cells.append(float(c_col.split(", ")[0]))
                new_t_cells_w3_w4.append(float(c_col.split(", ")[1]))
                new_t_cells_w5_w6.append(float(c_col.split(", ")[2]))

        if len(new_t_cells) > 0:

            to_write_list = []
            to_write_list.append(k);

            min_val = min(new_t_cells)
            max_val = max(new_t_cells)
            mean_val = statistics.mean(new_t_cells)
            median_val = statistics.median(new_t_cells)

            min_val_w3_w4 = min(new_t_cells_w3_w4)
            max_val_w3_w4 = max(new_t_cells_w3_w4)
            mean_val_w3_w4 = statistics.mean(new_t_cells_w3_w4)
            median_val_w3_w4 = statistics.median(new_t_cells_w3_w4)

            min_val_w5_w6 = min(new_t_cells_w5_w6)
            max_val_w5_w6 = max(new_t_cells_w5_w6)
            mean_val_w5_w6 = statistics.mean(new_t_cells_w5_w6)
            median_val_w5_w6 = statistics.median(new_t_cells_w5_w6)

            to_calc_list = [k, min_val, max_val, mean_val, median_val, min_val_w3_w4, max_val_w3_w4, mean_val_w3_w4, median_val_w3_w4, min_val_w5_w6, max_val_w5_w6, mean_val_w5_w6, median_val_w5_w6] 

            print(to_calc_list)

            if k in PE:
                xPE.append(to_calc_list)
                copyfile('/afs/cs/projects/kovashka/maher/vol3/ad_images/' + k + ".jpg", '/afs/cs/projects/kovashka/maher/vol3/dataset/parallel_easy/' + k + '.png')
                copyfile('/afs/cs/projects/kovashka/maher/vol3/matrix_results/' + k + ".png", '/afs/cs/projects/kovashka/maher/vol3/dataset/parallel_easy/' + k + '_scores.png')
            elif k in PH:
                xPH.append(to_calc_list)
                copyfile('/afs/cs/projects/kovashka/maher/vol3/ad_images/' + k + ".jpg", '/afs/cs/projects/kovashka/maher/vol3/dataset/parallel_hard/' + k + '.png')
                copyfile('/afs/cs/projects/kovashka/maher/vol3/matrix_results/' + k + ".png", '/afs/cs/projects/kovashka/maher/vol3/dataset/parallel_hard/' + k + '_scores.png')
            elif k in NPE:
                xNPE.append(to_calc_list)
                copyfile('/afs/cs/projects/kovashka/maher/vol3/ad_images/' + k + ".jpg", '/afs/cs/projects/kovashka/maher/vol3/dataset/non_parallel_easy/' + k + '.png')
                copyfile('/afs/cs/projects/kovashka/maher/vol3/matrix_results/' + k + ".png", '/afs/cs/projects/kovashka/maher/vol3/dataset/non_parallel_easy/' + k + '_scores.png')
            elif k in NPH:
                xNPH.append(to_calc_list)
                copyfile('/afs/cs/projects/kovashka/maher/vol3/ad_images/' + k + ".jpg", '/afs/cs/projects/kovashka/maher/vol3/dataset/non_parallel_hard/' + k + '.png')
                copyfile('/afs/cs/projects/kovashka/maher/vol3/matrix_results/' + k + ".png", '/afs/cs/projects/kovashka/maher/vol3/dataset/non_parallel_hard/' + k + '_scores.png')

            # print(k)
            # print(t_cells)
            # print(row_labels)
            # print(col_labels)

            # font = {'family' : 'normal',
            #         'weight' : 'normal',
            #         'size'   : 8}

            # img_filename = '/afs/cs/projects/kovashka/maher/vol3/ad_images/' + k + ".jpg"
            # img = mpimg.imread(img_filename)

            # fig, axs =plt.subplots(3,1) #, gridspec_kw={'height_ratios': [1, 5, 1, 5, 5]})

            # # axs[0].text(0.0, 0.0, "ID: " + k, fontdict=font)#, 
            # #             # horizontalalignment='center', 
            # #             # verticalalignment='center', 
            # #             # transform = axs[0].transAxes)
            # # axs[0].axis('tight')
            # # axs[0].axis('off')

            # axs[0].imshow(img)
            # axs[0].axis('off')

            # # axs[2].text(0.0, 0.0, "Transcript: " + res2[k][0][0], fontdict=font)#, 
            # #             # horizontalalignment='center', 
            # #             # verticalalignment='center', 
            # #             # transform = axs[2].transAxes)
            
            # # axs[2].axis('tight')
            # # axs[2].axis('off')

            # eval_res = "\nID: " + k + "\n"
            # eval_res = eval_res + "Transcript: " + res2[k][0][0] + "\n"
            # eval_res = eval_res + "Ground Truth: " + res4[k] + "\n"
            # eval_res = eval_res + "Numbers format: o2v_similarity(o*o), w2v_similarity(o*w), w2v_similarity(o*o) " + "\n"
            # eval_res = eval_res + "min_val: " + str(min_val) + ", " + str(min_val_w3_w4) + ", " + str(min_val_w5_w6) + "\n"
            # eval_res = eval_res + "max_val: " + str(max_val) + ", " + str(max_val_w3_w4) + ", " + str(max_val_w5_w6) + "\n"
            # eval_res = eval_res + "mean_val: " + str(round(mean_val, 2)) + ", " + str(round(mean_val_w3_w4, 2)) + ", " + str(round(mean_val_w5_w6, 2)) + "\n"
            # eval_res = eval_res + "median_val: " + str(round(median_val,2)) + ", " + str(round(median_val_w3_w4,2)) + ", " + str(round(median_val_w5_w6,2)) + "\n\n"

            # axs[1].text(0.0, 0.0, eval_res, fontdict=font)
            
            # axs[1].axis('tight')
            # axs[1].axis('off')

            
            # the_table = axs[2].table(cellText=t_cells,
            #                          rowLabels=row_labels,
            #                          colLabels=col_labels,
            #                          colWidths=[0.25 for x in col_labels],
            #                          loc='center')

            # the_table.auto_set_font_size(False)
            # the_table.set_fontsize(8)
            # the_table.scale(1, 1)

            # axs[2].axis('tight')
            # axs[2].axis('off')



            # # axs[1].plot(clust_data[:,0],clust_data[:,1])
            # output_filename = '/afs/cs/projects/kovashka/maher/vol3/matrix_results/' + k + ".png"
            # plt.savefig(output_filename, dpi=200, bbox_inches='tight')
            # plt.close()

            # blabla.append(to_write_list)



# import csv 

# with open('explore2_matrix_w2v.csv', 'w') as csv_file:
#     writer = csv.writer(csv_file, delimiter=',')
#     # for line in data:
    # writer.writerow(['img_id', 'o2v_sim(o*o) min_val', 'o2v_sim(o*o) max_val', 'o2v_sim(o*o) mean', 'o2v_sim(o*o) median', 'w2v_sim(o*w) min_val', 'w2v_sim(o*w) max_val', 'w2v_sim(o*w) mean', 'w2v_sim(o*w) median', 'w2v_sim(o*o) min_val', 'w2v_sim(o*o) max_val', 'w2v_sim(o*o) mean', 'w2v_sim(o*o) median'])

#     for b in blabla:
#         writer.writerow(b)


# print("total: " + str(total))
# print("sum_p: " + str(sum_p))
# print("count_p: " + str(count_p))


import csv 

with open('explore2_matrix_all_evals.csv', 'w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    # for line in data:
    writer.writerow(['cat_1', 'cat_2', 'o2v_sim(o*o) min_val', 'o2v_sim(o*o) max_val', 'o2v_sim(o*o) mean', 'o2v_sim(o*o) median', 'w2v_sim(o*w) min_val', 'w2v_sim(o*w) max_val', 'w2v_sim(o*w) mean', 'w2v_sim(o*w) median', 'w2v_sim(o*o) min_val', 'w2v_sim(o*o) max_val', 'w2v_sim(o*o) mean', 'w2v_sim(o*o) median'])


    xPE_xPH = [xPE, xPH, xPE+xPH]
    xPE_xPH_labels = ["parallel_easy", "parallel_hard", "parallel_easy_hard"]

    xNPE_xNPH = [xNPE, xNPH, xNPE+xNPH] 
    xNPE_xNPH_labels = ["non_parallel_easy", "non_parallel_hard", "non_parallel_easy_hard"] 

    for s in range(len(xPE_xPH)):
        for t in range(len(xNPE_xNPH)):

            xP_train = xPE_xPH[s]
            xNP_train = xNPE_xNPH[t]

            xP_train = xP_train[:len(xP_train)/2]
            xNP_train = xNP_train[:len(xNP_train)/2]
            
            cur_thres = [-1] * 12
            accuracies = []
            
            for i in range(12):

                temp_thres =0
                cur_max = -1
                temp_loss =0


                while temp_thres < 100:

                    for b in xP_train:
                        temp_loss += (b[i+1] - temp_thres)
                    for b in xNP_train:
                        temp_loss += (temp_thres - b[i+1])

                    if temp_loss > cur_max:
                        cur_thres[i] = temp_thres
                        cur_max = temp_loss

                    temp_thres += 0.001

                xP_test = xPE_xPH[s]
                xNP_test = xNPE_xNPH[t]

                xP_test = xP_test[len(xP_test)/2:]
                xNP_test = xNP_test[len(xNP_test)/2:]

                total_c = 0
                total_w = 0

                for b in xP_test:
                    if b[i+1] >= cur_thres[i]:
                        total_c += 1
                    else:
                        total_w += 1
                for b in xNP_test:
                    if b[i+1] < cur_thres[i]:
                        total_c += 1
                    else:
                        total_w += 1

                if total_w > 0:
                    acc = float(total_c) / float(total_w + total_c)
                elif total_c > 0:
                    acc = 1.0
                else:
                    acc = 0.0

                accuracies.append(acc)

            to_write_bla = [xPE_xPH_labels[s], xNPE_xNPH_labels[t]]
            to_write_bla.extend(accuracies)

            writer.writerow(to_write_bla)



# # ids_match = []

# import csv 

# with open('explore2b3_method_matrix.csv', 'w') as csv_file:
#     writer = csv.writer(csv_file, delimiter=',')
#     # for line in data:
#     writer.writerow(['img_id', 'correct/wrong', 'prediction', 'score(threshold:'+str(cur_thres)+')'])

#     for b in blabla:
#         if (b[1] == 'parallel' and b[0] >= cur_thres) or (b[1] == 'non_parallel' and b[0] < cur_thres):
#             total_c += 1
#             writer.writerow([b[2], 'correct', b[1], str(b[0])])
#         else:
#             total_w += 1
#             writer.writerow([b[2], 'wrong', b[1], str(b[0])])

# # # with open("ids_match.pkl", 'wb') as f:
# # #     pickle.dump(ids_match, f, pickle.HIGHEST_PROTOCOL)

# print("total_c: " + str(total_c))
# print("total_w: " + str(total_w))



