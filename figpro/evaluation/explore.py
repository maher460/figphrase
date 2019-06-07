import numpy as np
import six
import sys
import traceback
import re
import argparse

from sklearn import preprocessing
from figpro.evaluation.model_reader import ModelReader
import warnings

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

saved_model_path = '..\\train\model-save\semcomp.meta'
corpus_patch = "..\\train\\bnc_corpus.DIR"

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpuspath', '-c',
                        default="..\\train\\bnc_corpus.DIR",
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

while True:
    try:
        line = six.moves.input('>> ')
        sent, target_pos = parse_input(line)
        if target_pos == None:
            raise ParseException("Can't find the target position.")

        context_v = None
        if len(sent) > 1:
            context_v = model.context_rep(sent, target_pos)
            context_v = context_v / np.sqrt((context_v * context_v).sum())

        if sent[target_pos] == None:
            if context_v is not None:
                m = model.most_fit_context2(context_v)
            else:
                raise ParseException("Can't find a context.")
        else:
            if sent[target_pos].find("_") < 0:
                if sent[target_pos] not in word2index:
                    raise ParseException("Target word is out of vocabulary.")
                else:
                    target_v = generate_vector_word(sent[target_pos])
            else:
                target_v = generate_vector_phrase(sent[target_pos])
            if target_v is not None and context_v is not None:
                compatibility = (usage_rec(target_v, context_v)+1)/2
                print("Compatibility score: " + str(compatibility))
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

