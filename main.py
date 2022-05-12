#!/usr/bin/env python
# coding: utf-8

# In[4]:



import pandas as pd


import unicodedata
from unidecode import unidecode
import numpy as np
import os
from keras.models import Sequential
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from tensorflow.keras.optimizers import RMSprop
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import json




# In[5]:


data_dict = [json.loads(line) for line in open('News_Category_Dataset_v2.json', 'r')]

politics_dict = [x for x in data_dict if x['category'] == 'POLITICS']

df = pd.DataFrame(politics_dict)



# In[16]:


def sample(preds,diversity):
    preds = np.asarray(preds).astype('float64')  
    preds = np.log(preds) / diversity
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# In[17]:


class preProcessor():

    def __init__(self):
        self.NUM_OF_SEQ = None
        self.MAX_LEN = 20
        self.SEQ_JUMP = 3
        self.CORPUS_LENGHT = None
        self.corpus = self.createCorpus()
        self.chars = sorted(list(set(self.corpus)))
        self.NUM_OF_CHARS = len(self.chars)
        self.char_to_idx,self.idx_to_char = self.createIndices()
        self.sequences,self.next_chars = self.createSequences()
        self.dataX,self.dataY = self.one_hot()
       


    def createCorpus(self):
        corpus = u' '.join(df.headline)
        self.CORPUS_LENGHT= len(corpus)
        return corpus

    def createIndices(self):
        char_to_idx = {}
        idx_to_char = {}
        for i,c in enumerate(self.chars):
            char_to_idx[c]=i
            idx_to_char[i]=c
        return char_to_idx,idx_to_char

    def createSequences(self):
        sequences = []
        next_chars = []
        for i in range(0,self.CORPUS_LENGHT-self.MAX_LEN,self.SEQ_JUMP):
            sequences.append(self.corpus[i: i+self.MAX_LEN])
            next_chars.append(self.corpus[i+self.MAX_LEN])
        self.NUM_OF_SEQ = len(sequences)
        return sequences,next_chars

    def one_hot(self):
        dataX = np.zeros((self.NUM_OF_SEQ,self.MAX_LEN,self.NUM_OF_CHARS),dtype=bool)
        dataY = np.zeros((self.NUM_OF_SEQ,self.NUM_OF_CHARS),dtype=bool)
        for i,seq in enumerate(self.sequences):
            for j,c in enumerate(seq):
                dataX[i,j,self.char_to_idx[c]]=1
            dataY[i,self.char_to_idx[self.next_chars[i]]]=1
        return dataX,dataY


# In[23]:


class LSTModel():
    def __init__(self,max_len,num_of_chars,preprocessor):
        self.max_len = max_len
        self.num_of_chars = num_of_chars
        self.model = self.createModel()
        self.preprocessor = preprocessor

    def createModel(self,layer_size = 128,dropout=0.2,learning_rate=0.01,verbose=1):
        model = Sequential()
        model.add(LSTM(layer_size,return_sequences = True,input_shape=(self.max_len,self.num_of_chars)))
        model.add(Dropout(dropout))
        model.add(LSTM(layer_size, return_sequences=False))
        model.add(Dropout(dropout))
        model.add(Dense(self.num_of_chars, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=learning_rate))
        if verbose:
            print('Model Summary:')
            model.summary()
        return model

    def trainModel(self,X, y, batch_size=128, epochs=60, verbose=0):
        checkpointer = ModelCheckpoint(filepath="weights.hdf5", monitor='loss', verbose=verbose, save_best_only=True, mode='min')
        history = self.model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=[checkpointer])
        return history
    
    
    def createHeadlines(self,num_of_headlines=350,headline_length=40):
        f2=open("produced_headlines_new.txt", "w")
        self.model.load_weights('weights.hdf5')
        headlines = []
        seq_starts =[]
        diversities = [0.2, 0.5,1]
        
        
        
        for i,char in enumerate(self.preprocessor.corpus):
            if char == ' ':
                seq_starts.append(i)

        np.random.seed(294)
        print(seq_starts)
        beginnings = np.random.choice(seq_starts,size=num_of_headlines)
        
        for div in diversities:
            f2.write("---- diversity : %f\n"% div)
            print(num_of_headlines)
            for i in range(num_of_headlines):
               
                f2.write("---- Headline %d:\n" % i)
                begin = []
                begin = beginnings[i]
                headline = u''
                print('max len:', self.preprocessor.MAX_LEN)
                print("begin:", begin)
                print("end:", begin+self.preprocessor.MAX_LEN)
                print(type(self.preprocessor.corpus))
                #print("corpus:", self.preprocessor.corpus)
                sequence = self.preprocessor.corpus[begin:begin+self.preprocessor.MAX_LEN]
                print("seq:", sequence)
                headline += sequence
                f2.write("---Random Sequence beginning: %s\n" % headline)
                
                filename = "news_headline_data/6-class/filled/politics/p" + str(i) +".txt"
                f=open(filename, "w", encoding="utf-8")

            
    
                for _ in range(headline_length):
                    input_data = np.zeros((1,self.preprocessor.MAX_LEN,self.preprocessor.NUM_OF_CHARS),dtype=bool)
                    for t,char in enumerate(sequence):
                        input_data[0,t,self.preprocessor.char_to_idx[char]]=True
                    predictions = self.model.predict(input_data)[0]
                    next_idx = sample(predictions,div)
                    next_char = self.preprocessor.idx_to_char[next_idx]
                    headline += next_char
                    sequence = sequence[1:] + next_char
                f2.write("Generated using LSTM: %s\n" % headline)
                f.write(headline)
                headlines.append(headline)
        f.close()
        f2.close()
        filena="data_classes\news_headline_data\6-class\filled\politics\p66.txt"
        filena.close()
        return headlines


# In[24]:


if __name__ == "__main__":

    preprocessor = preProcessor()
    dataX = preprocessor.dataX
    dataY = preprocessor.dataY
    max_len = preprocessor.MAX_LEN

    num_of_chars = preprocessor.NUM_OF_CHARS
    lstm = LSTModel(max_len,num_of_chars,preprocessor)
    #train
    history = lstm.trainModel(dataX,dataY,epochs=60,verbose=1)
    
    f = open("loss.txt","w")
    for i,loss_data in enumerate(history.history['loss']):
        msg_annotated = "{0}\t{1}\n".format(i, loss_data)
        f.write(msg_annotated)
    f.close()
    

    headlines= lstm.createHeadlines()
    
    print(headlines)
    


