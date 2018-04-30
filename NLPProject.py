# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 20:11:01 2018

@author: cs
"""

import pandas as pd
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers.embeddings import Embedding
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import GlobalMaxPooling1D
from sklearn.metrics import accuracy_score
import numpy as np

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
        
        X_train1, X_test, y_train1, y_test = train_test_split(listCUIs, label, test_size = 0.15, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train1, y_train1, test_size = 0.15, random_state=42)
        tokenizer = Tokenizer(oov_token='UNK', lower = False)
       # print(X_train)
        #print(X_train)
        tokenizer.fit_on_texts(X_train)
        sequences = tokenizer.texts_to_sequences(X_train)
        word_index = tokenizer.word_index
        #print(word_index)
        X_train = pad_sequences(sequences, maxlen = MAXLEN)
        a = len(word_index)
        sequences_val = tokenizer.texts_to_sequences(X_val)
        X_val = pad_sequences(sequences_val, maxlen = MAXLEN)
        sequences_test = tokenizer.texts_to_sequences(X_test)
        X_test = pad_sequences(sequences_test, maxlen = MAXLEN)
        print(a)
        print(X_train.shape)
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
    patientL = patientL.head(150)
    #print(patientL)
    
    fullData_df, listCUIs, label = PD.readyData(patientL, "Data/")
    
    Maxlen, vocabSize = PD.getInitialVariables(listCUIs)
    print("vocabSize: " + str(vocabSize))
    print("Maxlen:" + str(Maxlen))

    X_train, y_train, X_val, y_val, X_test, y_test, vocab_size = PD.processText(listCUIs, label, Maxlen)
    print("a: " + str(vocab_size))
    
    e = PD.getEmbeddedLayer(vocab_size, Maxlen)
    model = Sequential()
    model.add(e)
    
    #model.add(Flatten())
    #model.add(Dense(6, activation = 'relu'))
    #model.add(Dense(3, activation = 'relu'))
    #model.add(Dense(1, activation='sigmoid'))
    model.add(Conv1D(100, 10, activation = 'relu'))
    model.add(MaxPooling1D(5))
    #model.add(Dropout(0.5))
    model.add(Conv1D(50, 10, activation = 'relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation = 'sigmoid', bias_regularizer= 'l2'))
    model.summary()
    model.compile(optimizer ='rmsprop', loss = 'binary_crossentropy', metrics=['acc'])
    #model.compile(optimizer ='sgd', loss = 'binary_crossentropy', metrics=['acc'])
    #model.compile(optimizer ='adam', loss = 'binary_crossentropy', metrics=['acc'])
    #print("reached here")
    print(X_train.shape)
    x = model.fit(np.array(X_train), np.array(y_train), epochs=3, batch_size = 1, validation_data = (np.array(X_val), np.array(y_val)))
    
    prediction = model.predict_classes(np.array(X_test), batch_size = 1)
    #print(y_test)
    #print(prediction)
    accuracy = accuracy_score(np.array(y_test), prediction)
    print(accuracy)
    
    """
    import matplotlib.pyplot as plt
    loss = x.history['loss']
    val_loss = x.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
    plt.title('Traning and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    plt.clf()
    acc = x.history['acc']
    val_acc = x.history['val_acc']
    
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
    plt.title('Traning and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.show()
    """
    
    
    
    