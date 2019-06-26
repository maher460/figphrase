# Description
This project is related to the paper **A Generalized Idiom Usage Recognition Model based on Semantic Compatibility** published AAAI 2019. Please read this
paper first to have more context of this project. The preprocessed part is from context2vet (https://github.com/orenmel/context2vec)

The main idea is to train a model which can tell whether a sense (of a word) is compatibile with a given context. Negative sampling is used in the training process,
similar to word2vec. 

# Enviorment
python 3.6

tensorflow 1.8

cuda v9

sklearn

numpy


# Usage 

(1) Download the code, open it using IDE PyCharm.

(2) Prepare your training data.

Your corpus should be one single text file with one sentence per line. Run the corpus_by_sent_length.py to generate the preprocessed training data. 
```
python ./train/corpus_by_sent_length.py CORPUS_FILE [max-sentence-length]
```
In PyCharm, you can configure the parameter in "Run->Edit Configuration->Parameters". An example parameter is "bnc_corpus 50", where "bnc_corpus" is the text file
and "50" is the frequency trim parameter. 

This will create a directory CORPUS_FILE.DIR. You can explore this folder to have an idea of the processed training data. 

Note that in this repo we have already prepared a processed training data in the folder "bnc_corpus.DIR" for you so you can quickly explore the project. 
This data is a portion of the BNC corpus  and its size is fairly small. To achive a better performance, **it is highly recommend that you use large corpus as the training 
data**. 

(3） Train the semantic compatibility model 

You can run the train_semcomp.py to start the training process. An example command line is:
```
python ./train/train_semcomp.py -i .\bnc_corpus.DIR  -w  bs -m bs  -c lstm --deep no -t 50 --dropout 0.0 -u 200 -e 10 -p 0.75 -b 400 -g 0
```
All the parameters can be configuted in PyCharm similar to step 2. The progam will save the trained model each epoch (when epoch number is large than 3). 

(4) Explore the trained semantic compatibility model 

We provide an interactiv mode for you to explore the model. Just run:
```
python ./evalation/explore.py -c ..\\train\\bnc_corpus.DIR  -m ..\\train\model-save\semcomp.meta -t 50 -u 200
```
Wait the model to be loaded and then you can input some sentences to evaluate your trained model. The parameter "-t" and "-u" should be the same as in Step 3. 

When you input
```
>> c1 c2 [] c3 c4 ...
```

It will find the most compatible words with the context

When you input
```
>> c1 c2 [target] c3 c4 ...
```
It will calculate a semantic compatibility score between the target and the context. The score has been normalized to a value between 0 and 1. 

When your target is a phrase, please use "_" to concatenate the words in the phrase. For example:
```
>> I have learned how to [break_the_ice] with strangers . 
```

# Contact

Please send me an email if you have questions: liucs1986 at gmail.com  



OBJ2VEC
python evaluation/explore.py -c ../obj2vec/open_images_corpus.DIR -m ./model-save/semcomp.meta -t 50 -u 200
python ./train/train_semcomp.py -i ../obj2vec/open_images_corpus.DIR  -w  bs -m bs  -c lstm --deep yes -t 50 --dropout 0.0 -u 200 -e 10 -p 0.75 -b 400 -g 0
python ./train/train_semcomp.py -i ../obj2vec/open_images_corpus_unique.DIR  -w  bs -m bs  -c lstm --deep yes -t 50 --dropout 0.0 -u 200 -e 25 -p 0.75 -b 400 -g 0

/m/01lynh /m/032b3c /m/03bt1vf /m/07mhn /m/0d5gx
/m/01lynh /m/032b3c [] /m/07mhn /m/0d5gx


/m/03q5c7 /m/050gv4 /m/099ssp /m/0cmx8 /m/0dt3t
Saucer Plate Platter Spoon Fork

/m/03q5c7 /m/050gv4 [] /m/0cmx8 /m/0dt3t

/m/03q5c7 /m/050gv4 [/m/02rgn06] /m/0cmx8 /m/0dt3t
[Volleyball]
