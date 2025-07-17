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
