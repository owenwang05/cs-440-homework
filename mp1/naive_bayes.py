# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm

'''
util for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def naive_bayes(train_set, train_labels, dev_set, laplace=1.0, pos_prior=0.5, silently=False):    
    
    # print("Training Set", train_set[0], len(train_set))
    # print("Training Labels", train_labels, len(train_labels))
    # print("Dev Set", dev_set[0], len(dev_set))
    
    # Training phase    
    pos = {}
    neg = {}
    words = set()
    
    total_pos = 0
    total_neg = 0

    for index in range(len(train_set)):
        document = train_set[index]
        label = train_labels[index]
        
        if label == 1:
            for word in document: 
                if word in pos: 
                    pos[word] += 1
                else:
                    pos[word] = 1
                    words.add(word)
                total_pos += 1
        elif label == 0:
            for word in document:
                if word in neg: 
                    neg[word] += 1
                else:
                    neg[word] = 1
                    words.add(word)
                total_neg += 1
    
    # Development phase 
    yhats = []
    
    for doc in tqdm(dev_set, disable=silently):
        pos_prob = math.log(pos_prior)
        neg_prob = math.log(1 - pos_prior)
        
        for word in doc:
            pos_add = laplace / (total_pos + laplace * len(words))
            neg_add = laplace / (total_pos + laplace * len(words))
            
            if word in pos: 
                pos_add = (pos[word] + laplace) / (total_pos + laplace * len(words))
                                                
            if word in neg: 
                neg_add = (neg[word] + laplace) / (total_neg + laplace * len(words))

            pos_prob += math.log(pos_add)
            neg_prob += math.log(neg_add)

        if pos_prob > neg_prob:
            yhats.append(1)
        else:
            yhats.append(0)

    return yhats
