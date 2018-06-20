#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 10:34:23 2018

@author: cs
"""

import os
import re
import string
import pandas as pd
from keras.models import Sequential
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers.embeddings import Embedding
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import concatenate
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import Input
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras import regularizers
from keras import optimizers
from sklearn.metrics import roc_auc_score
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


class ProcessData:
    
     
    def listfiles(self, a, b):
        txtfilesTrain = []  
        count = 0
        for file in os.listdir(a):
            try:
                if file.endswith(".txt"):
                    txtfilesTrain.append([a + "/" + str(file), b])
                    count = count + 1
                else:
                    print ("There is no text file")
            except Exception as e:
                raise e
                print ("No Files found here!")
        print ("Total files found:", count )
        return txtfilesTrain
    
    
    
    def convertListtoDF(self, list1, list2):
        df1 = pd.DataFrame(list1)
        df2 = pd.DataFrame(list2)
        result = pd.concat([df1, df2])
        result.columns = ['filepath', 'label']
        
        return result
        
        
    def readintoDF(self, Filename):
        df = pd.read_csv(Filename, sep = '\t')
        return df
    
    def checkAlphaNeumeric(self, doc):
        #doc = [word for word in doc if word.isnumeric()]
        doc1 = []
        for word in doc:
            if word.isnumeric():
                #print(word)
                x = 'DigitToken'
                #print(x)
                doc1.append(x)
            else:
                if word.isalpha():
                    doc1.append(word)
                else:
                    continue
        #doc = [word for word in doc1 if word.isalpha()]
        return doc1
    
    def readyData(self, patientL):
        filePathList = []
        label = []
        for index, row in patientL.iterrows():
            patient = row['filepath']
            label.append(row['label'])
            filePathList.append(patient)
        #print(label)    
        Train_document = [open(one_document, "r").read() for one_document in filePathList]
        #print(len(Train_document))
        #print(len(label))
        #print(len(filePathList))
        data_df = pd.DataFrame({'PatientCUIs': Train_document, 'label': label})
        data_df['PatientCUIs'] = data_df.PatientCUIs.apply(lambda x : str.lower(x))
        data_df['PatientCUIs'] = data_df.PatientCUIs.apply(lambda x : x.split())
        data_df['PatientCUIs'] = data_df.PatientCUIs.apply(lambda x : PD.checkAlphaNeumeric(x))
        
        #data_df['PatientCUIs'] = data_df.PatientCUIs.apply(lambda x : re.sub(r'\b\w{1,4}\b', '', str(x)))
        #data_df['PatientCUIs'] = data_df.PatientCUIs.apply(lambda x : PD.stripPunctuation(x))
        #data_df['PatientCUIs'] = data_df.PatientCUIs.apply(lambda x : x.split(''))
        
        #data_df['PatientCUIs'] = data_df.PatientCUIs.apply(lambda x : re.sub(r'\b\w{1,4}\b', '', x))
        #data_df['PatientCUIs'] = data_df.PatientCUIs.apply(lambda x : re.sub(r'\b\w{1,4}\b', '', str(x)))
        
        
        print(data_df)
        return data_df, Train_document, label


    def processText(self, trainDF, label, MAXLEN, testDF):
        
        X_train, X_val, y_train, y_val = train_test_split(listCUIs, label, test_size = 0.15, random_state=42)
        X_train = trainDF.PatientCUIs
        y_train = trainDF.PatientCUIs
        X_test = testDF.PatientCUIs
        y_test = testDF.label
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
        #print(a)
        print(X_train.shape)
        return X_train, y_train, X_val, y_val, X_test, y_test, a, word_index
    
    
    
    def processText1(self, listCUIs, label, MAXLEN):
        
        X_train1, X_test, y_train1, y_test = train_test_split(listCUIs, label, test_size = 0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train1, y_train1, test_size = 0.2, random_state=42)
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
        print(len(X_train))
        print(len(X_val))
        print(len(X_test))
        return X_train, y_train, X_val, y_val, X_test, y_test, a 
    
    def processText2(self, listCUIs, label, MAXLEN, test_df):

        X_train, X_val, y_train, y_val = train_test_split(listCUIs, label, test_size = 0.2)
        X_test = test_df.PatientCUIs
        y_test = test_df.label
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
        print(len(X_train))
        print(len(X_val))
        print(len(X_test))
        return X_train, y_train, X_val, y_val, X_test, y_test, a, word_index
     
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
    
    def roc_auc(y_true, y_pred_prob):
        roc_auc_score(y_true, y_pred_prob)
     
    def getEmbeddedLayer(self, vocab_size, inputsize):
        embedding = Embedding(vocab_size + 1, 300, input_length = inputsize)
        return embedding
     
    def neural_model(self, filters, kernel_size):
        
        #print(filters)
        #print(kernel_size)
        e = PD.getEmbeddedLayer(vocab_size, Maxlen)
        model = Sequential()
        model.add(e)
        #model.add(Flatten())
        #model.add(Dense(6, activation = 'relu'))
        #model.add(Dense(3, activation = 'relu'))
        #model.add(Dense(1, activation='sigmoid')) 
        model.add(Conv1D(filters = filters, kernel_size = kernel_size, activation = 'relu'))
        model.add(MaxPooling1D(30, 2))
        #model.add(Conv1D(50, 1, activation = 'relu'))
        #model.add(MaxPooling1D(10, 2))
        model.add(Dropout(0.2))
        model.add(Conv1D(20, 1, activation = 'relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation = 'relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(1, activation = 'sigmoid', kernel_regularizer=regularizers.l2(0.01),
                    activity_regularizer = regularizers.l1(0.01)))
        model.summary()
        #adam = optimizers.Adam(lr=0.00005)
        rmsprop = optimizers.RMSprop(lr = 0.001)
        model.compile(optimizer = rmsprop, loss = 'binary_crossentropy', metrics=['acc'])
        #model.compile(optimizer ='sgd', loss = 'binary_crossentropy', metrics=['acc'])
        #model.compile(optimizer = adam, loss = 'binary_crossentropy', metrics=['acc'])
        #print("reached here")
        #print(X_train.shape)
        return model
     
    
if __name__ == "__main__":
    
    PD = ProcessData()
    
    loc = "/home/cs/Brihat/ARDS/ArdsSample/ncui/yes"
    loc1 = "/home/cs/Brihat/ARDS/ArdsSample/ncui/no"
    loc2 = "/home/cs/Brihat/ARDS/ArdsSample/tncui/yes"
    loc3 = "/home/cs/Brihat/ARDS/ArdsSample/tncui/no"
    
    trainPositiveList = PD.listfiles(loc, 1)
    trainNegativeList = PD.listfiles(loc1, 0)
    
    testPositiveList = PD.listfiles(loc2, 1)
    testNegativeList = PD.listfiles(loc3, 0)
    
    
    patientL = PD.convertListtoDF(trainPositiveList, trainNegativeList)
    patientTL = PD.convertListtoDF(testPositiveList, testNegativeList)
    fullData_df, listCUIs, label = PD.readyData(patientL)
    testData_df, listTCUIs, Tlabel = PD.readyData(patientTL)
    
    Maxlen, NumberofPatients = PD.getInitialVariables(listCUIs)
    print("numberofPatients: " + str(NumberofPatients))
    print("Maxlen:" + str(Maxlen))

    X_train, y_train, X_val, y_val, X_test, y_test, maxlen, word_index = PD.processText2(listCUIs, label, Maxlen, testData_df)
    #print("vocab_size: " + str(vocab_size)) 
    
    """
    embeddings_index = {}
    f = open("mimic-embeddings.txt")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype = 'float32')
        embeddings_index[word] = coefs
    f.close()
    
    print('Found %s word vectors.' % len(embeddings_index))
    
    
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        print(word)
        print(i)
        #print(embeddings_index.get(word))
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
     """     
            
    
    #print(X_train.shape)
    #print(y_train)
    
    vocab_size = len(word_index)
    #e = PD.getEmbeddedLayer(vocab_size, Maxlen)
    input_tensor = Input(shape = (Maxlen,))
    x = Embedding(vocab_size + 1, 300, input_length = Maxlen)(input_tensor)
    CNN1 = Conv1D(filters = 100, kernel_size = 3, activation = 'relu')(x)
    CNN2 = Conv1D(filters = 50, kernel_size = 3, activation = 'relu')(x)
    #print(CNN1.shape)
    #print(CNN2.shape)
    y = concatenate([CNN1, CNN2], axis= -1)
    #print(y.shape)
    y = GlobalMaxPooling1D()(y)
    y = Dropout(0.5)(y)
    z = Dense(1, activation = 'sigmoid', kernel_regularizer=regularizers.l2(0.01))(y)
    #print(z.shape)
    model = Model(inputs = input_tensor, outputs = z)
    rmsprop = optimizers.RMSprop(lr = 0.001)
    model.compile(optimizer = rmsprop, loss = 'binary_crossentropy', metrics=['acc'])
    x = model.fit(np.array(X_train), np.array(y_train), epochs=15, batch_size = 1, validation_data = (np.array(X_val), np.array(y_val)))
    predictionforROC = model.predict(np.array(X_val), batch_size = 1)
    
    #prediction = model.predict_classes(np.array(X_val), batch_size = 1)
    prediction = predictionforROC.argmax(axis = -1)
    #print(y_test)
    print(prediction)
    accuracy = accuracy_score(np.array(y_val), prediction)
    print("Accuracy:" + str(accuracy))
    print(confusion_matrix(y_val, prediction))
    print(classification_report(y_val, prediction))
    print("AUC_ROC area: " + str(roc_auc_score(y_val, predictionforROC)))
    
    
    
    
    
    