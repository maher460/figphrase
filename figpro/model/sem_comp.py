__author__ = 'changsheng'
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import figpro.common.defs
from figpro.common.defs import  NEGATIVE_SAMPLING_NUM
from tensorflow.python.ops import variable_scope as vs
from figpro.common.my_loss import nce_loss
class BiLSTMContext:

    def __init__(self,n_vocab,hidden_unit,num_layers,input_dim,output_dim,deep,batch_size,max_length):

        # vocab size
        self.n_vocab = n_vocab

        self.num_words_batch = 100

        #lstm hidden size
        self.hidden_unit = hidden_unit

        #lstm layer
        self.num_layers = num_layers

        #input embeding size
        self.input_dim = input_dim

        #output embedding size
        self.output_dim = output_dim

        # word embeddings
        self.weights = tf.Variable(tf.random_normal([n_vocab, input_dim],stddev=0.35), name="input-embedding")

        # forward lstm cell
        self.cell_fw = rnn.MultiRNNCell([rnn.BasicLSTMCell(self.hidden_unit) for _ in range(self.num_layers)])

        # backward lstm cell
        self.cell_bw = rnn.MultiRNNCell([rnn.BasicLSTMCell(self.hidden_unit) for _ in range(self.num_layers)])

        # deep
        self.deep = deep

        # b size
        self.batch_size = batch_size

        # max sentence length
        self.max_length = max_length

        self.keep_prob = tf.placeholder(tf.float32,name="prob")

        self.deep_w1 = tf.Variable(tf.random_normal([ 2*self.hidden_unit, 2 * self.hidden_unit], stddev=0.35),
                                                 name="deepw1-weights")

        self.deep_w2 = tf.Variable(tf.random_normal([2 * self.hidden_unit,self.output_dim], stddev=0.35),
                                                 name="deepw2-weights")

        self.deep_b1 = tf.Variable(tf.random_normal([int(2 * self.hidden_unit)], stddev=0.35), name="deep-b1")


        self.deep_b2 = tf.Variable(tf.random_normal([self.output_dim], stddev=0.35), name="deep-b2")



        #negative sampling weight
        self.nce_weight = tf.Variable(tf.zeros([self.n_vocab,self.output_dim],tf.float32),name="output-embedding")
        #self.nce_bias = tf.Variable(tf.constant(0.0,shape=[self.n_vocab]),name="output-b")
        self.nce_bias = tf.constant(0.0,shape=[self.n_vocab],name="output-b")

        # based on context2vec, these should variable should start with zero, and bias is 0
        #self.nce_weight = tf.Variable(tf.zeros([self.n_vocab,self.output_dim],tf.float32))
        #self.nce_bias = tf.constant(0.0,shape=[self.n_vocab])

        #context holder for context rep training
        self.inputs = tf.placeholder(dtype=tf.int32, shape=[None, None],name="inp")

        #context holder for most fit function
        self.single_context = tf.placeholder(dtype=tf.float32, shape=[1,self.output_dim ],name="single-context")

        # target position holder
        self.position = tf.placeholder(dtype=tf.int32,name="position")

        # attention variable
        self.att = tf.Variable(tf.random_normal([self.output_dim], stddev=0.35), name="att")


    def __call__(self, sent):
        '''
        Train the network
        :param sent: a minibatch of sentences
        '''
        #self.reset_state()
        return self._calculate_loss(sent)

    # context representation of single sentence
    def context2vec(self,sess, sent, pos):
        cc = self._contexts_rep(sent)
        target = tf.nn.l2_normalize(cc[0,self.position,:],name = "context2vec-op")
        predict= sess.run([target], {self.inputs:sent,self.position:pos,self.keep_prob:1})
        return predict

    # context representation of batch sentences
     # bi directional lstm
    # please explore from this type of context representation
    def _contexts_rep(self, sent_arr):

        return self._contexts_rep_awe_globalatt(sent_arr) #TODO: using this to change context rep

        #add begining of sentence
        inputs1 = tf.concat([tf.fill([tf.shape(self.inputs)[0],1],self.n_vocab-2),self.inputs],1)
        #add end of sentence
        inputs2 = tf.concat([inputs1,tf.fill([tf.shape(self.inputs)[0],1],self.n_vocab-1)],1)

        inp = tf.nn.embedding_lookup(self.weights,inputs2)

        with tf.name_scope('rnn'):
            (outputs, states) = tf.nn.bidirectional_dynamic_rnn(self.cell_fw,self.cell_bw,dtype = tf.float32,inputs = inp)

            # output of bi lstm
            fw,bw = outputs

            # since sentences are padded with <eos> and <bos>, remove their outputs before concatenate
            fw = fw[:,:-2,:]
            bw = bw[:,2:,:]

            #outputs already reversed, so we do not need to do the reverse
            #bw = tf.reverse(bw,[2])

            #TODO: the self.deep is always True right now...
            if self.deep == False:

                fw_re = tf.reshape(fw,[-1,self.hidden_unit])
                bw_re = tf.reshape(bw,[-1,self.hidden_unit])

                pfw = tf.matmul(fw_re,self.fw_w) + self.fw_bias
                pbw = tf.matmul(bw_re,self.bw_w) + self.bw_bias

                pfw_or = tf.reshape(pfw,[tf.shape(self.inputs)[0], -1 ,int(self.output_dim/2)])
                pbw_or = tf.reshape(pbw,[tf.shape(self.inputs)[0], -1 ,int(self.output_dim/2)])

                cc = tf.concat([pfw_or,pbw_or],2)
                return cc

            else:
                cc = tf.concat([fw,bw],2)
                cc_re = tf.reshape(cc,[-1,2*self.hidden_unit])
                ccd1 = tf.matmul(cc_re,self.deep_w1) + self.deep_b1
                ccd1_re = tf.nn.relu(ccd1)
                ccd2 = tf.matmul(ccd1_re,self.deep_w2) + self.deep_b2
                return tf.reshape(ccd2,[tf.shape(self.inputs)[0], -1 ,self.output_dim])

    # context representation of batch sentences
    # not using average
    def _contexts_rep_awe_globalatt(self, sent_arr):


        inp = tf.nn.embedding_lookup(self.weights,self.inputs)

        i = tf.constant(0)
        total = tf.shape(inp)[1]

        r = tf.zeros([tf.shape(inp)[0],0,tf.shape(inp)[2]], dtype = tf.float32)
        condition = lambda i,r:tf.less(i,total)

        def body(i,r):
            f = inp[:,:i,:]
            b = inp[:,i+1:,:]
            z = tf.concat([f,b], axis = 1)

            # attention here(can remove or not remove)
            m = tf.tensordot(z,self.att, axes = 1)
            scl = tf.nn.softmax(m)
            z = tf.multiply(z,tf.expand_dims(scl,2))
            # attention end here

            # when without attention, should be reduce_mean, get the avearge
            r = tf.concat([r,tf.expand_dims(tf.reduce_sum(z,axis = 1),1)],axis = 1)
            return tf.add(i,1),r

        results = tf.while_loop(
                condition, body,
                loop_vars=[i,r],
        shape_invariants=[
                i.get_shape(),
                tf.TensorShape([None,None,None])])
        cc =  results[1]
        #return cc

        # add mlp layer here
        cc_re = tf.reshape(cc,[-1,self.input_dim])
        ccd1 = tf.matmul(cc_re,self.deep_w1) + self.deep_b1
        ccd1_re = tf.nn.relu(ccd1)
        ccd2 = tf.matmul(ccd1_re,self.deep_w2) + self.deep_b2
        return tf.reshape(ccd2,[tf.shape(self.inputs)[0], -1 ,self.output_dim])


    def _contexts_rep_bilstm_globalatt(self, sent_arr):

        # batch_size = len(sent_arr)

        # add begining of sentence
        inputs1 = tf.concat([tf.fill([tf.shape(self.inputs)[0], 1], self.n_vocab - 2), self.inputs], 1)
        # add end of sentence
        inputs2 = tf.concat([inputs1, tf.fill([tf.shape(self.inputs)[0], 1], self.n_vocab - 1)], 1)

        inp = tf.nn.embedding_lookup(self.weights, inputs2)

        with tf.name_scope('rnn'):
            (outputs, states) = tf.nn.bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, dtype=tf.float32,
                                                                inputs=inp)

            # output of bi lstm
            fw, bw = outputs

            # since sentences are padded with <eos> and <bos>, remove their outputs before concatenate
            fw = fw[:, :-2, :]
            bw = bw[:, 2:, :]

            i = tf.constant(0)
            total = tf.shape(fw)[1]

            r = tf.zeros([tf.shape(inp)[0], 0, tf.shape(inp)[2]], dtype=tf.float32)
            condition = lambda i, r: tf.less(i, total)

            def body(i, r):
                f = fw[:, :i, :]
                b = bw[:, i + 1:, :]
                z = tf.concat([f, b], axis=1)

                # attention here(can remove or not remove)
                m = tf.tensordot(z, self.att, axes=1)
                scl = tf.nn.softmax(m)
                z = tf.multiply(z, tf.expand_dims(scl, 2))
                # attention end here

                # when without attention, should be reduce_mean, get the avearge
                r = tf.concat([r, tf.expand_dims(tf.reduce_sum(z, axis=1), 1)], axis=1)
                return tf.add(i, 1), r

            results = tf.while_loop(
                condition, body,
                loop_vars=[i, r],
                shape_invariants=[
                    i.get_shape(),
                    tf.TensorShape([None, None, None])])
            cc = results[1]
            # return cc


            # add two mlp with relue layer here
            cc_re = tf.reshape(cc, [-1, self.hidden_unit])
            ccd1 = tf.matmul(cc_re, self.deep_w1) + self.deep_b1
            ccd1_re = tf.nn.relu(ccd1)
            ccd2 = tf.matmul(ccd1_re, self.deep_w2) + self.deep_b2
            ccd2 = tf.nn.dropout(ccd2,self.keep_prob)
            return tf.reshape(ccd2, [tf.shape(self.inputs)[0], -1, self.output_dim])

    def _contexts_rep_awe_localatt(self, sent_arr):

        # batch_size = len(sent_arr)

        # add begining of sentence
        inp = tf.nn.embedding_lookup(self.weights,self.inputs)

        i = tf.constant(0)
        total = tf.shape(inp)[1]

        r = tf.zeros([tf.shape(inp)[0],0,tf.shape(inp)[2]], dtype = tf.float32)
        condition = lambda i,r:tf.less(i,total)

        # relevance matrix
        inp_l2 = tf.nn.l2_normalize(inp, dim=2)
        inp_l2_trans = tf.transpose(inp_l2, perm=[0, 2, 1])
        rel_mat = tf.matmul(inp_l2, inp_l2_trans)

        rel_mat = tf.matrix_set_diag(rel_mat,
                                     tf.ones([tf.shape(inp_l2)[0], tf.shape(inp_l2)[1]], dtype=tf.float32) * (-1))

        def body(i,r):
            f = inp[:,:i,:]
            b = inp[:,i+1:,:]
            z = tf.concat([f,b], axis = 1)

            # local attention here
            a1 = rel_mat[:,:i,:i]
            a2 = rel_mat[:,:i,i+1:]
            a3 = rel_mat[:,i+1:,:i]
            a4 = rel_mat[:,i+1:,i+1:]
            b1 = tf.concat([a1,a2], axis = 2)
            b2 = tf.concat([a3,a4], axis = 2)
            gf = tf.concat([b1,b2], axis = 1)
            att = tf.reduce_max(gf,axis = 1)
            scl = tf.nn.softmax(att)
            z = tf.multiply(z,tf.expand_dims(scl,2))
            # attention end here

            # when without attention, should be reduce_mean, get the avearge
            r = tf.concat([r,tf.expand_dims(tf.reduce_sum(z,axis = 1),1)],axis = 1)
            return tf.add(i,1),r

        results = tf.while_loop(
                condition, body,
                loop_vars=[i,r],
        shape_invariants=[
                i.get_shape(),
                tf.TensorShape([None,None,None])])
        cc =  results[1]

            # add two mlp with relue layer here
        cc_re = tf.reshape(cc, [-1, self.hidden_unit])
        ccd1 = tf.matmul(cc_re, self.deep_w1) + self.deep_b1
        ccd1_re = tf.nn.relu(ccd1)
        ccd2 = tf.matmul(ccd1_re, self.deep_w2) + self.deep_b2
        ccd2 = tf.nn.dropout(ccd2,self.keep_prob)
        return tf.reshape(ccd2, [tf.shape(self.inputs)[0], -1, self.output_dim])


    def _contexts_rep_bilstm_localatt(self, sent_arr):

        # batch_size = len(sent_arr)
        # relevance matrix
        inp_rel = tf.nn.embedding_lookup(self.weights, self.inputs)
        inp_l2 = tf.nn.l2_normalize(inp_rel, dim=2)
        inp_l2_trans = tf.transpose(inp_l2, perm=[0, 2, 1])
        rel_mat = tf.matmul(inp_l2, inp_l2_trans)

        rel_mat = tf.matrix_set_diag(rel_mat,
                                     tf.ones([tf.shape(inp_l2)[0], tf.shape(inp_l2)[1]], dtype=tf.float32) * (-1))


        # add begining of sentence
        inputs1 = tf.concat([tf.fill([tf.shape(self.inputs)[0], 1], self.n_vocab - 2), self.inputs], 1)
        # add end of sentence
        inputs2 = tf.concat([inputs1, tf.fill([tf.shape(self.inputs)[0], 1], self.n_vocab - 1)], 1)

        inp = tf.nn.embedding_lookup(self.weights, inputs2)

        with tf.name_scope('rnn'):
            (outputs, states) = tf.nn.bidirectional_dynamic_rnn(self.cell_fw, self.cell_bw, dtype=tf.float32,
                                                                inputs=inp)

            # output of bi lstm
            fw, bw = outputs

            # since sentences are padded with <eos> and <bos>, remove their outputs before concatenate
            fw = fw[:, :-2, :]
            bw = bw[:, 2:, :]

            i = tf.constant(0)
            total = tf.shape(fw)[1]

            r = tf.zeros([tf.shape(inp)[0], 0, tf.shape(inp)[2]], dtype=tf.float32)
            condition = lambda i, r: tf.less(i, total)

            def body(i, r):
                f = fw[:, :i, :]
                b = bw[:, i + 1:, :]
                z = tf.concat([f, b], axis=1)

                # local attention here
                a1 = rel_mat[:, :i, :i]
                a2 = rel_mat[:, :i, i + 1:]
                a3 = rel_mat[:, i + 1:, :i]
                a4 = rel_mat[:, i + 1:, i + 1:]
                b1 = tf.concat([a1, a2], axis=2)
                b2 = tf.concat([a3, a4], axis=2)
                gf = tf.concat([b1, b2], axis=1)
                att = tf.reduce_max(gf, axis=1)
                scl = tf.nn.softmax(att)
                z = tf.multiply(z, tf.expand_dims(scl, 2))
                # attention end here

                # attention here(can remove or not remove)
                #m = tf.tensordot(z, self.att, axes=1)
                #scl = tf.nn.softmax(m)
                #z = tf.multiply(z, tf.expand_dims(scl, 2))
                # attention end here

                # when without attention, should be reduce_mean, get the avearge
                r = tf.concat([r, tf.expand_dims(tf.reduce_sum(z, axis=1), 1)], axis=1)
                return tf.add(i, 1), r

            results = tf.while_loop(
                condition, body,
                loop_vars=[i, r],
                shape_invariants=[
                    i.get_shape(),
                    tf.TensorShape([None, None, None])])
            cc = results[1]
            # return cc


            # add two mlp with relue layer here
            cc_re = tf.reshape(cc, [-1, self.hidden_unit])
            ccd1 = tf.matmul(cc_re, self.deep_w1) + self.deep_b1
            ccd1_re = tf.nn.relu(ccd1)
            ccd2 = tf.matmul(ccd1_re, self.deep_w2) + self.deep_b2
            ccd2 = tf.nn.dropout(ccd2,self.keep_prob)
            return tf.reshape(ccd2, [tf.shape(self.inputs)[0], -1, self.output_dim])


    # get the last output of lstm based on length; the padding data will generate 0s afther the last output

    # get the last output of lstm based on length; the padding data will generate 0s afther the last output
    def last_relevant(self, output, length):
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        out_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant,index,flat


    def _calculate_loss(self, sent):
        # sent is a batch of sentences.

        # size: batch * timestep * output dim
        #context_rep = self._contexts_rep_awe_localatt(sent)
        context_rep = self._contexts_rep(sent)

        inputs_re = tf.reshape(self.inputs,[-1,1])
        #sent_arr = self.xp.asarray(sent, dtype=np.int32)

        # size: (batch * timestep) * output dim
        cc_re = tf.reshape(context_rep,[-1,self.output_dim])

        # could change to reduce mean, but this is sensitive to learning rate
        loss = tf.reduce_sum(
            nce_loss(weights = self.nce_weight,
                   biases = self.nce_bias,
                   labels = inputs_re,
                   inputs = cc_re,
                   num_sampled = NEGATIVE_SAMPLING_NUM * int(self.batch_size)* self.max_length,
                   num_classes=self.n_vocab,partition_strategy="div"))

        return loss,self.inputs,self.keep_prob

    def most_fit(self,sess,predict, k_num):

        logits = tf.matmul(self.single_context, tf.nn.l2_normalize(tf.transpose(self.nce_weight),dim = 0))
        logits = tf.add(logits, self.nce_bias)

        val, ind = tf.nn.top_k(logits, k = k_num, sorted = True, name = "most_fit_op")
        #return context, logits,tf.argmax(logits, 1)
        logits_fit, top_k= sess.run([logits,ind], {self.single_context:predict})

        # return the logists result (i.e.,  the fit score) and the coresponding top k indices
        return logits_fit, top_k

if __name__ == '__main__':


    c = np.arange(300)+3
    d = c.reshape([30,10])

    test_input = np.array([3, 5, 7, 7,9])
    test_input = test_input.reshape([1,-1])

    bilstm = BiLSTMContext(310,100,1,100,100)

    loss,inputs = bilstm(d)

    train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss_z = sess.run([loss],{inputs:d,bilstm.keep_prob : 0.5})
        print(loss_z)
        for i in range(1000):
            lossz,tran = sess.run([loss,train_op], {inputs:d,bilstm.keep_prob : 0.8})
            if i%100 == 0:
                print (lossz)
        # calcualte context of ([6, 7, 8, 9, 10]) at position 2, so the target will be 8

        predict =  bilstm.context2vec(sess, test_input,3)
        print(predict)

        # predict whether the target is 8
        logits,fit_predict = bilstm.most_fit(sess, predict)

        print(logits)
        print(fit_predict)



















