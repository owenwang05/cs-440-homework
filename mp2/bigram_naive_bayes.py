# bigram_naive_bayes.py
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
utils for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace: {unigram_laplace}")
    print(f"Bigram Laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=True, lowercase=True, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with the bigram mixture model.
    You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def bigram_bayes(train_set, train_labels, dev_set, unigram_laplace=1.0, bigram_laplace=1.0, bigram_lambda=0.33, pos_prior=None, silently=False):
    print_values_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)
    
    # unigram counts
    pos_uni = {}
    neg_uni = {}
    unigrams = set()
    total_pos_uni = 0
    total_neg_uni = 0

    # bigram counts
    pos_bi = {}
    neg_bi = {}
    bigrams = set()
    total_pos_bi = 0 
    total_neg_bi = 0

    for index in range(len(train_set)):
        document = train_set[index]
        label = train_labels[index]
        
        # unigrams
        for word in document:
            if label == 1:
                if word in pos_uni: 
                    pos_uni[word] += 1
                else:
                    pos_uni[word] = 1
                    unigrams.add(word)
                total_pos_uni += 1
            else:
                neg_uni[word] = neg_uni.get(word, 0) + 1
                total_neg_uni += 1
            unigrams.add(word)
        
        # bigrams
        for i in range(len(document) - 1):
            bg = (document[i], document[i+1])
            if label == 1:
                pos_bi[bg] = pos_bi.get(bg, 0) + 1
                total_pos_bi += 1
            else:
                neg_bi[bg] = neg_bi.get(bg, 0) + 1
                total_neg_bi += 1
            bigrams.add(bg)

    # Development phase
    yhats = []
    for document in tqdm(dev_set, disable=silently):
        if pos_prior is None:
            pos_prior = sum(train_labels) / len(train_labels)

        # unigram
        pos_prob_uni = math.log(pos_prior)
        neg_prob_uni = math.log(1 - pos_prior)
        for word in document:
            pos_uni_prob = (pos_uni.get(word, 0) + unigram_laplace) / (total_pos_uni + unigram_laplace * len(unigrams))
            neg_uni_prob = (neg_uni.get(word, 0) + unigram_laplace) / (total_neg_uni + unigram_laplace * len(unigrams))
            pos_prob_uni += math.log(pos_uni_prob)
            neg_prob_uni += math.log(neg_uni_prob)

        # bigram
        pos_prob_bi = math.log(pos_prior)
        neg_prob_bi = math.log(1 - pos_prior)
        for i in range(len(document)-1):
            bg = (document[i], document[i+1])
            pos_bi_prob = (pos_bi.get(bg, 0) + bigram_laplace) / (total_pos_bi + bigram_laplace * len(bigrams))
            neg_bi_prob = (neg_bi.get(bg, 0) + bigram_laplace) / (total_neg_bi + bigram_laplace * len(bigrams))
            pos_prob_bi += math.log(pos_bi_prob)
            neg_prob_bi += math.log(neg_bi_prob)

        pos_prob = (1 - bigram_lambda) * pos_prob_uni + bigram_lambda * pos_prob_bi
        neg_prob = (1 - bigram_lambda) * neg_prob_uni + bigram_lambda * neg_prob_bi

        if pos_prob > neg_prob: 
            yhats.append(1)
        else:
            yhats.append(0)

    return yhats



