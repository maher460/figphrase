# encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
from nltk.corpus import gutenberg
sent_corpus = []
gutenberg_corpus = open('gutenberg', 'w')

for fileid in gutenberg.fileids():
    sents = gutenberg.sents(fileid)
    #print(sents[122])
    num_sents = len(gutenberg.sents(fileid))
    #print(num_sents)
    for sen in sents:
        if len(sen)>10 and len(sen)<50:
            line =  (" ").join(sen)
            gutenberg_corpus.write("%s\n" % line)
gutenberg_corpus.close()
            #sent_corpus.append(sen)
