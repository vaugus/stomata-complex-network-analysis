#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NOME"""

import operator
import warnings

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from classification import Classification
from complex_network_analysis import ComplexNetworkAnalysis
from tools import Tools

warnings.filterwarnings("ignore")

__version__ = '1.0'
__author__ = 'Victor Augusto'
__copyright__ = "Copyright (c) 2018 - Victor Augusto"


def main():
	"""main method."""

	filename = str(input()).rstrip()

	thresh = float(input())

	analysis = ComplexNetworkAnalysis()
	tools = Tools()

	tools.pretty_print_parameters(filename, thresh)

	# Retrieve coordinates
	points, x, y = analysis.read_points(filename, thresh)

	# Calculate weighted edges
	edges = analysis.generate_edges(points, thresh)

	# Generate the complex network structure
	graph = analysis.generate_graph(edges)

	tools.save_graph(graph, filename)

	# Generate ground truth networks for classification
	truth = analysis.generate_ground_truth_networks(len(points))
	erdos_renyi, watts_strogatz, barabasi_albert = truth

	# Extract ground truth complex network numerical features
	er_fv = analysis.extract_features(erdos_renyi).reshape(1,-1)
	ws_fv = analysis.extract_features(watts_strogatz).reshape(1,-1)
	ba_fv = analysis.extract_features(barabasi_albert).reshape(1,-1)

	truth = [er_fv, ws_fv, ba_fv]

	# Extract complex network numerical features for test
	fv = analysis.extract_features(graph).reshape(1,-1)

	# Run classifiers over the extracted features
	classification = Classification(truth)
	classification.cosine_similarity_classification(fv)

	classifiers = ['svm', 'random_forest', 'gaussianNB']
	for c in classifiers:
		classification.classify(c, fv)
		
if __name__ == '__main__':
	main()