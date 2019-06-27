__author__ = 'changsheng'
import tensorflow as tf
from figpro.train.sentence_reader import SentenceReaderDir
import numpy as np

class ModelReader:
    def __init__(self, saved_model_path, corpus_patch, trim_freq):
        print("Start to load the model.......")
        self.sess = tf.Session()
        self.new_saver = tf.train.import_meta_graph(saved_model_path)
        self.new_saver.restore(self.sess, tf.train.latest_checkpoint(saved_model_path.rsplit('/', 1)[0])) #'./model-save-awe-ga-duplicates/'))
        self.graph = tf.get_default_graph()
        print("Model loaded.")
        self.input = self.graph.get_tensor_by_name("inp:0")
        #self.drop = self.graph.get_tensor_by_name("dropout/prob:0")
        self.pos = self.graph.get_tensor_by_name("position:0")
        self.context2vec = self.graph.get_tensor_by_name("context2vec-op:0")
        self.output_embedding = self.graph.get_tensor_by_name("output-embedding:0")
        self.input_embedding = self.graph.get_tensor_by_name("input-embedding:0")
        self.output_bias = self.graph.get_tensor_by_name("output-b:0")
        self.single_context = self.graph.get_tensor_by_name("single-context:0")
        self.most_fit = self.graph.get_tensor_by_name("most_fit_op:1")
        self.drop_out = self.graph.get_tensor_by_name("prob:0")
        self.reader = SentenceReaderDir(corpus_patch,trim_freq,100)
        self.word2index = self.reader.word2index
        self.index2word = self.reader.index2word
        self.output_embedding_vec = self.sess.run(self.output_embedding)
        s = np.sqrt((self.output_embedding_vec * self.output_embedding_vec).sum(1))
        s[s==0.] = 1.
        self.output_embedding_vec /= s.reshape((s.shape[0], 1))

    def word_index(self, path):
        index2word = {}
        word2index = {}
        starting_index = 0
        with open(path) as f:
            for line in f:
                word = line.strip().lower()
                word2index[word] = starting_index
                index2word[starting_index] = word
                starting_index += 1
        return word2index,index2word

    def context_rep(self,sent,position):
        voc_size = len(self.word2index)
        test_input_vector = np.array([self.word2index[word] if word in self.word2index else voc_size-3 for word in sent])
        test_input_vector = test_input_vector.reshape([1,-1])
        o = self.sess.run(self.context2vec,{self.input:test_input_vector,self.pos:position,self.drop_out:1})
        return o

    def most_fit_context(self,context):
        o_v = np.array([context])
        o = o_v.reshape([1,-1])
        m = self.sess.run(self.most_fit,{self.single_context:o})
        return m

    def most_fit_context2(self,context):
        o_v = np.array([context])
        o = o_v.reshape([-1,1])
        w = self.output_embedding_vec
        s = np.sqrt((w * w).sum(1))
        s[s==0.] = 1.
        w /= s.reshape((s.shape[0], 1))
        similarity = (w.dot(o)+1.0)/2
        similarity = similarity.reshape(([1,-1]))
        n_result = 10
        count = 0
        res = []
        for i in (-similarity[0]).argsort():
            if not i:
                continue
            # print('{0}: {1}'.format(self.reader.index2word[i], similarity[0][i]))
            res.append(self.reader.index2word[i])
            count += 1
            # if count == n_result:
            #     break
        return res

    def fit_score(self, context, vec):
        context = context / np.sqrt((context * context).sum())
        vec = vec / np.sqrt((vec * vec).sum())
        lit_sim = (context * vec).sum()
        print (lit_sim)
        if lit_sim>-0.07:
            return 0
        else:
            return 1

    def get_target_vec(self,target):
        index = 0
        if target in self.word2index:
            index = self.word2index[target]
        else:
            # unk token
            index = len(self.word2index) - 3
        return self.output_embedding_vec[index,:]




if __name__ == '__main__':

    test_input = ["reduce", "the", "number","of","partners","you","have","cooperated","with"]

    model = ModelReader()
    o = model.context_rep(test_input,2)

    print (o)

    m = model.most_fit_context2(o)


    tv = model.get_target_vec('bread')
    m = model.most_fit_context2(tv)

    output_embedding = model.sess.run(model.output_embedding)

    input_embedding = model.sess.run(model.input_embedding)


    output_bias = model.sess.run(model.output_bias)






