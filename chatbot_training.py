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
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from chatbot import Chatbot


######### Preparing the data #########
#---------------------------
# Prepare the chatbot data
chatbot = Chatbot("Chatbot/ChatbotTraining.csv")
data = chatbot.data
labels = chatbot.label_encoder(data.tag)
responses = data.responses.values.tolist()
patterns = data.patterns.values.tolist()
vocabulary = chatbot.get_vocabulary(patterns)

######### Training #########
#--------------------------


def accuracy_score_with_countvector() -> None:
    
    train_data, test_data, train_y, test_y = chatbot.accuracy_score_with_countvector(patterns, labels)
    
    print('### Aufbau eines Entscheidungsbaummodells ###')
    ### Aufbau eines Entscheidungsbaummodells ###
    # Create Decision Tree classifer object
    decision_tree_classifier = DecisionTreeClassifier()

    # Fit the Decision Tree Classifer with patterns and labels
    decision_tree_classifier.fit(train_data, train_y)

    prediction = decision_tree_classifier.predict(test_data)
    accuracy = metrics.accuracy_score(test_y, prediction)
    print(f"The accuracy of the decision tree with countvector is: {accuracy:.2f}%")
    
    # Klassifizierungsbericht ausgeben
    print("\nKlassifizierungsbericht with countvector:\n", metrics.classification_report(test_y, prediction))
    print('-' * 50, '\n')

accuracy_score_with_countvector()


def accuracy_score_with_tfidfvector() -> None:
    X_train, X_test, y_train, y_test = chatbot.accuracy_score_with_tfidfvector(patterns, labels)
    
    # Modell tranieren
    print('### Aufbau eines Entscheidungsbaummodells ###')
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Modell evaluieren
    # Vorhersagen
    prediction = model.predict(X_test)

    # Genauigkeit berechnen und ausgeben
    accuracy = metrics.accuracy_score(y_test, prediction)
    print(f"The accuracy of the decision tree with tfidfvector is: {accuracy:.2f}%")
    
    # Klassifizierungsbericht ausgeben
    print("\nKlassifizierungsbericht with tfidfvector:\n", metrics.classification_report(y_test, prediction))
    print('-' * 50, '\n')

accuracy_score_with_tfidfvector()


######### Decision Tree #########
#---------------------------

bow_patterns = [chatbot.bag_of_words(pattern, vocabulary) for pattern in patterns]

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
    bow_message = [chatbot.bag_of_words(message, vocabulary)]

    # Predict the response for test dataset
    # find out over bow which category/tag the message belongs to
    prediction = decision_tree_classifier.predict(bow_message)

    response_options = [tupel[1] for tupel in responses if tupel[0]==prediction]
    print(random.choice(response_options))
