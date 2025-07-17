#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 12:56:21 2025

@author: Serkan Özkan
"""

import sys
import os

from sklearn.model_selection import train_test_split
# Add the parent directory to PYTHONPATH so custom_sklearn can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize


# Download stopwords and tokenizer if you haven't already
# nltk.download("punkt") # separates the text into sentences separated by punctuation
nltk.download('punkt_tab') # separates the text into sentences separated by punctuation or tabs
nltk.download("stopwords") # the , that, a ..


class Chatbot:
    """
    Class for training a chatbot using decision tree classifier.
    """

    def __init__(self, csv_file: str):
        """
        Initialize the Chatbot class with the path to the CSV file.
        """
        self._data = pd.read_csv(csv_file)
        pd.set_option('display.max_columns', None)

    @property
    def data(self) -> pd.DataFrame:
        """
        Return the data as a pandas DataFrame.
        """
        return self._data

    def label_encoder(self, label) -> LabelEncoder:
        """Convert the strings objects zu fit form, because the classifier fit run only with floats
        Return a LabelEncoder fitted to the 'label' column of the data."""
        label_encoder = LabelEncoder()
        return label_encoder.fit_transform(label)

    def tokenize(self, text: str) -> list:
        """
        Tokenize the input text into sentences and then into words.
        """
        sentences = sent_tokenize(text)
        
        processed_sentences = []
        for sentence in sentences:
            processed_sentences.append(self.get_tokenized_words(sentence))
        return processed_sentences

    def get_tokenized_words(self, sentence: str) -> list:
        """
        Remove punctuation from the sentence and return tokenized words.
        """
        # Remove punctuation from sentence
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        # Return tokenized sentence 
        return word_tokenize(sentence)

    ### Stemming / Lemmatization ###
    # Herausfinden von Wortstämme aller einzelner Wörter
    # Stemming: Grop und Schnell
    # Lemmatization: Präsizer und Langsam (morphologische Analyse vom Grundwurzel)
    def root_of(self, words: list, language: str = 'english') -> list:
        """
        Perform stemming on the list of words and remove stopwords.
        """
        port_stemmer = PorterStemmer()

        # Get the list of stop words in {language}
        stop_words = set(stopwords.words(language))
        
        # Remove stopwords and stem
        filtered_words = [port_stemmer.stem(word.lower()) for word in words if word.lower() not in stop_words]
        return filtered_words

    def bag_of_sentences(self, text: str) -> list:
        """
        Create a bag of words representation for the given text using the vocabulary.
        """
        bags: list = []
        sentences = self.tokenize(text)

        for sentence in sentences:
            bags.append(self.bag_of_words(sentence))
        return bags

    def bag_of_words(self, sentence: str, vocabulary: list) -> list:
        """
        Create a bag of words representation for the given sentence using the vocabulary.
        """
        bag = self.bag_of_words_as_vector(sentence=sentence, vocabulary=vocabulary)
        return bag.tolist()

    def bag_of_words_as_vector(self, sentence: str, vocabulary: list) -> np.ndarray:
        """
        Create a bag of words representation for the given sentence as a numpy array.
        """
        bag = np.zeros(len(vocabulary), dtype=np.float32)
        words = self.get_tokenized_words(sentence)

        for index, word in enumerate(vocabulary):
            bag[index] = int(word in self.root_of(words))
        return bag

    def get_vocabulary(self, list_of_data: list) -> list:
        """
        Create a vocabulary from the list of data.
        """
        vocabulary: list = []
        for sentence in list_of_data:
            sentence = self.get_tokenized_words(sentence)
            vocabulary.extend(self.root_of(sentence))
        return vocabulary

    def accuracy_score_with_countvector(self, patterns: list, labels: LabelEncoder) -> tuple:
        print('### CounterVector ###')
        count_vec = CountVectorizer()
        word_counts = count_vec.fit_transform(patterns)
        bag_of_words_df = pd.DataFrame(word_counts.toarray(), columns = count_vec.get_feature_names_out())

        train_data, test_data, train_y, test_y = train_test_split(bag_of_words_df, labels, test_size=0.2, random_state=42)
        
        return train_data, test_data, train_y, test_y

    def accuracy_score_with_tfidfvector(self, patterns: list, labels: LabelEncoder) -> tuple:
        print('### TfidfVectorizer ###')
        X_train, X_test, y_train, y_test = train_test_split(patterns, labels, test_size=0.2, random_state=42)
        
        # Textdaten vektorisieren
        vectorizer = TfidfVectorizer()
        X_train_vec  = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        return X_train_vec, X_test_vec, y_train, y_test
