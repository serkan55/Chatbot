#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Juni 05 10:58:08 2025

@author: serkan
"""
"""
Required Modules to install beforehand:
-> pandas
-> sklearn

"""
'''
CSV Datei (ChatbotTraining.csv) laden und Bag of Words bilden
'''

import random
import numpy as np
import pandas as pd
import string
# train_test_split ist für die Teilung von Daten in Trainings- und Testdaten
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


######### Preparing the data #########
#---------------------------

data = pd.read_csv("ChatbotTraining.csv")
pd.set_option('display.max_columns', None)

# responses = list(zip(data.tag, data.patterns, data.responses))
# print(data.patterns)

### Auswahl der Merkmale ###
# Convert the strings objects zu fit form, because the classifier fit run only with floats
label_encoder = LabelEncoder()
data.tag = label_encoder.fit_transform(data.tag)
# data.patterns  = label_encoder.fit_transform(data.patterns)
# data.responses = label_encoder.fit_transform(data.responses)
# tags = set(data.tag)
labels = data.tag


######### CHATTING #########
#---------------------------

# Deprecated
# def tokenize(sentence: str) -> list:
    # punctuation = ['.', ',', '?', '!', ':', ';']
    # for ch in punctuation:
    #     sentence = sentence.replace(ch, '')
    # Remove punctuation
    # sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    # print(sentence)

    # return sentence.split()

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
    
    
# Download stopwords and tokenizer if you haven't already
# nltk.download("punkt") # separates the text into sentences separated by punctuation
nltk.download('punkt_tab') # separates the text into sentences separated by punctuation or tabs
nltk.download("stopwords") # the , that, a ..

### TOKENIZATION ###
# Sentence gets splitted into array of words/tokens and removed the punctuation
def tokenize(text: str) -> None:
    # text = "Natural language processing has advanced significantly. Researchers at major universities continue to push boundaries. The applications are endless."

    # Tokenize into sentences
    sentences = sent_tokenize(text)
    
    processed_sentences = []
    for sentence in sentences:
        processed_sentences.append(get_tokenized_words(sentence))
    return processed_sentences

def get_tokenized_words(sentence: str) -> list:
    # Remove punctuation from sentence
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))

    # Return tokenized sentence 
    return word_tokenize(sentence)
    

### Stemming / Lemmatization ###
# Herausfinden von Wortstämme aller einzelner Wörter
# Stemming: Grop und Schnell
# Lemmatization: Präsizer und Langsam (morphologische Analyse vom Grundwurzel)
def root_of(words: list, language: str = 'english') -> list:
    # Initialize Python porter stemmer
    port_stemmer = PorterStemmer()

    # Get the list of stop words in English
    stop_words = set(stopwords.words(language))
    
    # Remove stopwords and stem
    filtered_words = [port_stemmer.stem(word.lower()) for word in words if word.lower() not in stop_words]
    return filtered_words

#     # Perform stemming
#     print("{0:20}{1:20}".format("--Word--","--Stem--"))
#     for word in word_tokens:
#         print ("{0:20}{1:20}".format(word, port_stemmer.stem(word)))
#     return port_stemmer.stem(word.lower())
#     # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     from nltk.stem import WordNetLemmatizer
#     nltk.download("wordnet")
#     nltk.download("omw-1.4")
    
#     # Initialize wordnet lemmatizer
#     wnl = WordNetLemmatizer()
    
#     # Perform lemmatization
#     print("{0:20}{1:20}".format("--Word--","--Lemma--"))
#     for word in word_tokens:
#         print ("{0:20}{1:20}".format(word, wnl.lemmatize(word, pos="v")))
#     return wnl.lemmatize(word.lower(), pos="v")
    

### Bag of Words ###
def bag_of_sentences(text: str, vocabulary: list) -> list:
    bags: list = []
    sentences = tokenize(text)

    for sentence in sentences:
        bags.append(bag_of_words(sentence))
    return bags

def bag_of_words(sentence: str, vocabulary: list) -> list:
    bag = np.zeros(len(vocabulary), dtype=np.float32)
    words = get_tokenized_words(sentence)

    for index, word in enumerate(vocabulary):
        bag[index] = int(word in root_of(words))
    return bag.tolist()

# root_of(satz, language='german')


######### Decision Tree #########
#---------------------------

print('### Create Vocabulary ###')
### Create Vocabulary ###
def get_vocabulary(list_of_data:list) -> list:
    vocabulary: list = []
    for sentence in list_of_data:
        sentence = get_tokenized_words(sentence)
        vocabulary.extend(root_of(sentence))
    return vocabulary

patterns = data.patterns.values.tolist()
vocabulary = get_vocabulary(patterns)
bow_patterns = [bag_of_words(pattern, vocabulary) for pattern in patterns]

print('### Aufbau eines Entscheidungsbaummodells ###')
### Aufbau eines Entscheidungsbaummodells ###
# Create Decision Tree classifer object
decision_tree_classifier = DecisionTreeClassifier()

# Fit the Decision Tree Classifer with patterns and labels
decision_tree_classifier.fit(bow_patterns, labels)

responses = list(zip(labels, data.responses.values.tolist()))
print('### Starts ###')
while True:
    message = input('Message: ')
    bow_message = [bag_of_words(message, vocabulary)]

    # Predict the response for test dataset
    # find out over bow which category/tag the message belongs to
    prediction = decision_tree_classifier.predict(bow_message)

    response_options = [tupel[1] for tupel in responses if tupel[0]==prediction]
    print(random.choice(response_options))



