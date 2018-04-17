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


    def processText(self, listCUIs, label):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(listCUIs)
        sequences = tokenizer.texts_to_sequences(listCUIs)
        word_index = tokenizer.word_index
        data = pad_sequences(sequences)
        #print(word_index)
        X_train1, X_test, y_train1, y_test = train_test_split(data, label, test_size = 0.15, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train1, y_train1, test_size = 0.15, random_state=42)
        return X_train, y_train, X_val, y_val, X_test, y_test      
      
     
if __name__ == "__main__":
    
    PD = ProcessData()
    patientL = PD.readintoDF("patientLabel.txt")
    #print(patientL)
    fullData_df, listCUIs, label = PD.readyData(patientL, "Data/")
    X_train, y_train, X_val, y_val, X_test, y_test = PD.processText(listCUIs, label)
    
    
    
    