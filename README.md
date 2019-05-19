# figphrase

python 3.6.4

tensorflow 1.5

CUDA_PATH  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0

run train command
-i C:\changsheng\idiom_dl\train\bnc_corpusv2.DIR  -w  bs -m bs  -c lstm --deep no -t 50 --dropout 0.0 -u 300 -e 30 -p 0.75 -b 400 -g 0


1 prepare your training data
one single text file, each sentence per line

run 
python context2vec/train/corpus_by_sent_length.py CORPUS_FILE [max-sentence-length]

for example
gutenberg 50


2
train the model 
-i .\bnc_corpus.DIR  -w  bs -m bs  -c lstm --deep no -t 50 --dropout 0.0 -u 200 -e 30 -p 0.75 -b 400 -g 0

embedding size 
configuration

when explore:
if input idiom or phrase, use "_"


3 explore
-c ..\\train\\bnc_corpus.DIR  -m ..\\train\model-save\semcomp.meta -t 50 -u 200

