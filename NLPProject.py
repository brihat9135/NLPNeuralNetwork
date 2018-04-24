# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 20:11:01 2018

@author: cs
"""
import numpy as np
import pandas as pd
from nltk import word_tokenize
import gensim
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers.embeddings import Embedding
from keras.layers import Flatten
from keras.layers import Dense
from sklearn.metrics import accuracy_score
class ProcessData:
    
    def readintoDF(self, Filename):
        df = pd.read_csv(Filename, sep = '\t')
        return df
        
    def readyData(self, patientL, path):
        filePathList = []
        label = []
        for index, row in patientL.iterrows():
            patient = row['SubjectID']
            filepath = path + str(patient) + ".txt"
            label.append(row['label'])
            filePathList.append(filepath)
            
        listCUIs = []
        for filePath in filePathList:
            with open (filePath) as fin:
                for line in fin:
                    listCUIs.append(line)
        data_df = pd.DataFrame({'PatientCUIs': listCUIs, 'label': label})
        return data_df, listCUIs, label


    def processText(self, listCUIs, label, MAXLEN):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(listCUIs)
        sequences = tokenizer.texts_to_sequences(listCUIs)
        word_index = tokenizer.word_index
        data = pad_sequences(sequences, maxlen = MAXLEN)
        a = len(word_index)
        X_train1, X_test, y_train1, y_test = train_test_split(data, label, test_size = 0.15, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train1, y_train1, test_size = 0.15, random_state=42)
        return X_train, y_train, X_val, y_val, X_test, y_test, a     
     
    def getInitialVariables(self, listCUIs):
        CUIsLength = []
        for x in listCUIs:
            y = x.split()
            z = len(y)
            #print(z)
            CUIsLength.append(z)
        highest = max(CUIsLength)
        #print(highest)
        vocab_size = len(listCUIs)
        return highest, vocab_size
      
     
    def getEmbeddedLayer(self, vocab_size, inputsize):
        embedding = Embedding(vocab_size + 1, 300, input_length = inputsize)
        
        return embedding
     
     
    
if __name__ == "__main__":
    
    PD = ProcessData()
    patientL = PD.readintoDF("patientLabel.txt")
    patientL = patientL.head(1000)
    print(patientL.shape)
    fullData_df, listCUIs, label = PD.readyData(patientL, "Data/")
    
    Maxlen, vocabSize = PD.getInitialVariables(listCUIs)

    X_train, y_train, X_val, y_val, X_test, y_test, uniCUIS = PD.processText(listCUIs, label, Maxlen)
    print(X_train.shape)
    e = PD.getEmbeddedLayer(uniCUIS, Maxlen)
    model = Sequential()
    
    model.add(e)
    
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer ='adam', loss = 'binary_crossentropy', metrics=['acc'])
    #print("reached here")
    x = model.fit(np.array(X_train), np.array(y_train), epochs=50, verbose=0)
    
    prediction = model.predict(X_val)
    accuracy = accuracy_score(y_val, prediction)
    print(accuracy)
    
    
    
    
    