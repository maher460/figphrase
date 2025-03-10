import math
import collections
import numpy as np

from figpro.common.defs import Toks, SENT_COUNTS_FILENAME, WORD_COUNTS_FILENAME


def read_batch(f, batchsize, word2index):        
    batch = []
    while len(batch) < batchsize:
        line = f.readline()
        # print(line.encode("utf-8"))
        if not line: break
        sent_words = line.strip().lower().split()
        assert(len(sent_words) > 1)
        sent_inds = []
        for word in sent_words:
            if word in word2index:
                ind = word2index[word]
            else:
                ind = word2index['<UNK>']
            sent_inds.append(ind)
        batch.append(sent_inds)
    return batch


class SentenceReaderDir(object):
    '''
    Reads a batch of sentences at a time from a corpus directory in random order.
    Assumes that the sentences are split into different files in the directory according to their word lengths.
    '''
    
    sent_counts_filename = SENT_COUNTS_FILENAME
    word_counts_filename = WORD_COUNTS_FILENAME

    def __init__(self, path, trimfreq, batchsize):
        '''
        Initialize reader.
        :param path: input directory
        :param trimfreq: treat all words with lower frequency than trimfreq as unknown words
        :param batchsize: the size of the minibatch that will be read in every iteration
        '''
        self.path = path
        self.batchsize = batchsize
        #self.trimmed_word2count, self.word2index, self.index2word = self.read_and_trim_vocab(trimfreq)
        self.total_words = 0
        self.word2index, self.index2word = self.read_and_trim_vocab_sequential(trimfreq)
        self.fds = []
        
    def open(self):
        self.fds = []
        with open(self.path+'/'+self.sent_counts_filename,encoding='utf-8') as f:
            for line in f:
                [filename, count] = line.strip().split()
                batches = int(math.ceil(float(count) / self.batchsize))
                fd = open(self.path+'/'+filename, 'r',encoding='utf-8')
                self.fds = self.fds + [fd]*batches
        np.random.seed(1034)
        np.random.shuffle(self.fds)
    

    def close(self):
        fds_set = set(self.fds)
        for f in fds_set:
            f.close()
                        
    # original context 2 vec
    def read_and_trim_vocab(self, trimfreq):
        word2count = collections.Counter()
        with open(self.path+'/'+self.word_counts_filename,encoding='utf-8') as f:
            for line in f:
                [word, count] = line.strip().lower().split()
                word2count[word] = int(count)
    
        trimmed_word2count = collections.Counter()
        index2word = {Toks.UNK:'<UNK>', Toks.BOS:'<BOS>', Toks.EOS:'<EOS>'}
        word2index = {'<UNK>':Toks.UNK, '<BOS>':Toks.BOS, '<EOS>':Toks.EOS}
        unknown_counts = 0
        for word, count in word2count.items():
            if count >= trimfreq and word.lower() != '<unk>' and word.lower() != '<rw>':    
                ind = len(word2index)
                word2index[word] = ind
                index2word[ind] = word
                trimmed_word2count[ind] = count
            else:
                unknown_counts += count
        trimmed_word2count[word2index['<BOS>']] = 0
        trimmed_word2count[word2index['<EOS>']] = 0
        trimmed_word2count[word2index['<UNK>']] = unknown_counts
        
        return trimmed_word2count, word2index, index2word

    # sequential indexing
    def read_and_trim_vocab_sequential(self, trimfreq):
        index2word = {}
        word2index = {}
        unknown_counts = 0
        starting_index = 0
        count_total = 0
        with open(self.path+'/'+self.word_counts_filename, encoding='utf-8') as f:
            for line in f:
                [word, count] = line.strip().lower().split()
                count_total =  int(count) + count_total
                if int(count) >= trimfreq and word.lower() != '<unk>' and word.lower() != '<rw>':
                    if word not in word2index:
                        index2word[starting_index] = word
                        word2index[word] = starting_index
                        starting_index = starting_index + 1

        index2word[starting_index] = '<UNK>'
        word2index['<UNK>'] = starting_index
        #print(starting_index)
        starting_index = starting_index + 1


        index2word[starting_index] = '<BOS>'
        word2index['<BOS>'] = starting_index
        #print(starting_index)
        starting_index = starting_index + 1

        index2word[starting_index] = '<EOS>'
        word2index['<EOS>'] = starting_index
        #print(starting_index)

        self.total_words = count_total

        return word2index, index2word

       
    def next_batch(self):                
        for fd in self.fds:
            batch = read_batch(fd, self.batchsize, self.word2index)
            yield batch
 
 
 
if __name__ == '__main__':
    import sys   
    reader = SentenceReaderDir(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))
    
    for i in range(2):
        print ('epoc', i)
        reader.open()
        i = 0
        j = 0
        for batch in reader.next_batch():
            if i < 3:
                print (batch)
                print("")
            i += 1
            j += len(batch)
        print ('batches', i)
        print ('sents', j)
        reader.close()
                     
