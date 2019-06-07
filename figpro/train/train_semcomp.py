import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import argparse
import time
import sys


from figpro.train.sentence_reader import SentenceReaderDir
from model.sem_comp import BiLSTMContext


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', '-i',
                        default=None,
                        help='input corpus directory')
    parser.add_argument('--trimfreq', '-t', default=0, type=int,
                        help='minimum frequency for word in training')
    parser.add_argument('--ns_power', '-p', default=0.75, type=float,
                        help='negative sampling power')
    parser.add_argument('--dropout', '-o', default=0.0, type=float,
                        help='NN dropout')
    parser.add_argument('--wordsfile', '-w',
                        default=None,
                        help='word embeddings output filename')
    parser.add_argument('--modelfile', '-m',
                        default=None,
                        help='model output filename')
    parser.add_argument('--cgfile', '-cg',
                        default=None,
                        help='computational graph output filename (for debug)')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', default=200, type=int,
                        help='number of units (dimensions) of one context word')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='learning minibatch size')
    parser.add_argument('--epoch', '-e', default=10, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--context', '-c', choices=['lstm'],
                        default='lstm',
                        help='context type ("lstm")')
    parser.add_argument('--deep', '-d', choices=['yes', 'no'],
                        default=None,
                        help='use deep NN architecture')
    
    args = parser.parse_args()
    
    if args.deep == 'yes':
        args.deep = True
    elif args.deep == 'no':
        args.deep = False
    else:
        raise Exception("Invalid deep choice: " + args.deep)
    
    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('Context type: {}'.format(args.context))
    print('Deep: {}'.format(args.deep))
    print('Dropout: {}'.format(args.dropout))
    print('Trimfreq: {}'.format(args.trimfreq))
    print('NS Power: {}'.format(args.ns_power))
    print('')
       
    return args 

args = parse_arguments()

context_word_units = args.unit


reader = SentenceReaderDir(args.indir, args.trimfreq, args.batchsize)
voc_size = len(reader.word2index)
print('n_vocab: %d' % (len(reader.word2index))) # excluding the three special tokens
print('corpus size: %d' % (reader.total_words))

c = np.arange(300)+3
d1 = c.reshape([30,10])

test_input = ["reduce", "the", "number","of","partner","you","have","cooperated","with","."]
test_input_vector = np.array([reader.word2index[word] if word in reader.word2index else voc_size-3 for word in test_input])
test_input_vector = test_input_vector.reshape([1,-1])

test_input2 = ["the", "book", "is","interesting","to","read","."]
test_input_vector2 = np.array([reader.word2index[word] if word in reader.word2index else voc_size-3 for word in test_input2])
test_input_vector2 = test_input_vector2.reshape([1,-1])


print(args.batchsize)
bilstm = BiLSTMContext(len(reader.word2index),args.unit,1,args.unit,args.unit,True,args.batchsize, 50)

def ClipIfNotNone(grad):
    if grad is None:
        return grad
    return tf.clip_by_value(grad, -5, 5)

loss,inputs,drop_out = bilstm(d1)
optimizer = tf.train.AdamOptimizer()
grads_and_vars = optimizer.compute_gradients(loss)
capped_gvs = [(ClipIfNotNone(grad), var) for grad, var in grads_and_vars]
train_op = optimizer.apply_gradients(capped_gvs)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    STATUS_INTERVAL = 1000000

    dic = {}
    dic2={}

    for epoch in range(args.epoch-3):
        begin_time = time.time()
        cur_at = begin_time
        word_count = 0
        next_count = STATUS_INTERVAL
        accum_loss = 0.0
        last_accum_loss = 0.0
        last_word_count = 0
        print('epoch: {0}'.format(epoch))

        reader.open()

        last_num_word = 0
#        i = 0
        for sent in reader.next_batch():

            d = sent

            lossz,tran = sess.run([loss,train_op], {inputs:d,drop_out:0.8})

            accum_loss += lossz

            word_count += len(sent)*len(sent[0]) # all sents in a batch are the same length
            accum_mean_loss = float(accum_loss)/word_count if accum_loss > 0.0 else 0.0

            if word_count >= next_count:
                now = time.time()
                duration = now - cur_at
                throuput = float((word_count-last_word_count)) / (now - cur_at)
                cur_mean_loss = (float(accum_loss)-last_accum_loss)/(word_count-last_word_count)
                print('{} words, {:.2f} sec, {:.2f} words/sec, {:.4f} accum_loss/word, {:.4f} cur_loss/word'.format(
                word_count, duration, throuput, accum_mean_loss, cur_mean_loss))
                next_count += STATUS_INTERVAL
                cur_at = now
                last_accum_loss = float(accum_loss)
                last_word_count = word_count

                predict =  bilstm.context2vec(sess, test_input_vector,4)
                logits,fit_predict = bilstm.most_fit(sess, predict, 20)
                print([reader.index2word[index] for index in fit_predict[0]])

                predict =  bilstm.context2vec(sess, test_input_vector2,3)
                logits,fit_predict = bilstm.most_fit(sess, predict, 20)
                print([reader.index2word[index] for index in fit_predict[0]])



        print ('accum words per epoch', word_count, 'accum_loss', accum_loss, 'accum_loss/word', accum_mean_loss)
        reader.close()


        if epoch >3:
            saver.save(sess, "./model-save/semcomp")




    


    
    

