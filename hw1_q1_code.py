# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 11:28:23 2020

@author: admin
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def load_data():
    fake_file = open('clean_fake.txt')
    fake_list = fake_file.read().splitlines()
    real_file = open('clean_real.txt')
    real_list = real_file.read().splitlines()
    label = []
    for i in fake_list:
        label.append('fake')
    for i in real_list:
        label.append('real')
    both_list = fake_list + real_list
    vectorizer = CountVectorizer()
    both_vector = vectorizer.fit_transform(both_list)
    X_train, X_test, y_train, y_test = train_test_split(both_vector, label, test_size=0.3)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)
    return X_train, X_test, X_val, y_train, y_test, y_val

def select_knn_model():
    k_range = range(1, 21)
    val_scores = []
    train_scores = []
    X_train, X_test, X_val, y_train, y_test, y_val = load_data()
    for i in k_range:  
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        val_scores.append(knn.score(X_val, y_val))
        train_scores.append(knn.score(X_train, y_train))
    plt.plot(k_range, train_scores, label='training')    
    plt.plot(k_range, val_scores, label='validation')    
    plt.xlabel('k - Number of Nearest Neighbors')
    plt.ylabel('Accuracy')
    plt.legend()
    best_k = np.argmax(val_scores) + 1
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train, y_train)
    print('k=',best_k)
    print('accuracy on test data is', knn.score(X_test, y_test))

def select_knn_model_new():
    k_range = range(1, 21)
    val_scores = []
    train_scores = []
    X_train, X_test, X_val, y_train, y_test, y_val = load_data()
    for i in k_range:  
        knn = KNeighborsClassifier(n_neighbors=i, metric='cosine')
        knn.fit(X_train, y_train)
        val_scores.append(knn.score(X_val, y_val))
        train_scores.append(knn.score(X_train, y_train))
    plt.plot(k_range, train_scores, label='training')    
    plt.plot(k_range, val_scores, label='validation')    
    plt.xlabel('k - Number of Nearest Neighbors')
    plt.ylabel('Accuracy')
    plt.legend()
    best_k = np.argmax(val_scores) + 1
    knn = KNeighborsClassifier(n_neighbors=best_k, metric='cosine')
    knn.fit(X_train, y_train)
    print('k=',best_k)
    print('accuracy on test data is', knn.score(X_test, y_test))

#select_knn_model()
#select_knn_model_new()



