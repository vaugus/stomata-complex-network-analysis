#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NOME"""

__version__ = '1.0'
__author__ = 'Victor Augusto'
__copyright__ = "Copyright (c) 2018 - Victor Augusto"

import collections
import operator
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

class Classification(object):
	"""First line of a docstring is short and next to the quotes.

	Class and exception names are CapWords.

	Closing quotes are on their own line
	"""

	def __init__(self, truth):
		"""Constructor."""
		self.truth = truth
		self.X = [truth[0][0].tolist(), truth[1][0].tolist(), truth[2][0].tolist()]
		self.y = ['random', 'small-world', 'scale-free']

	def __del__(self):
		"""Destructor."""
		del self

	def cosine_similarity_classification(self, fv):
		print('=============== Cosine Similarity Comparison\n')
		er_fv, ws_fv, ba_fv = self.truth

		sim_er = cosine_similarity(er_fv, fv)[0][0]
		sim_ws = cosine_similarity(ws_fv, fv)[0][0]
		sim_ba = cosine_similarity(ba_fv, fv)[0][0]

		print('Cosine similarity with Erdos-Renyi         (Random): \t' + str(sim_er))
		print('Cosine similarity with Watts-Strogatz (Small-World): \t' + str(sim_ws))
		print('Cosine similarity with Barabasi-Albert (Scale-Free): \t' + str(sim_ba))
		print('\n')

	def classify(self, classifier, fv):
		clf = None
		if classifier == 'svm':
			clf = svm.SVC(gamma='scale', probability=True)
			print("=============== SVM Classifier\n")
			print(clf)
			print('\n')
			clf.fit(self.X, self.y)

		if classifier == 'random_forest':
			clf = RandomForestClassifier()
			print("=============== Random Forest Classifier\n")
			print(clf)
			print('\n')
			clf.fit(self.X, self.y)

		if classifier == 'gaussianNB':
			clf = GaussianNB()
			print("=============== Gaussian Naive Bayes Classifier\n")
			print(clf)
			print('\n')
			clf.fit(self.X, self.y)

		proba = clf.predict_proba(fv)[0]
		prediction = clf.predict(fv)[0]

		print('Probability of Random Network: ' + str(proba[0]))
		print('Probability of Small-World Network: ' + str(proba[1]))
		print('Probability of Scale-Free Network: ' + str(proba[2]))
		print('Predicted: ' + prediction)
		print('\n')