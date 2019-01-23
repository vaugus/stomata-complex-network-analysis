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

class ComplexNetworkAnalysis(object):

	def __init__(self):
		"""Constructor."""
		pass

	def __del__(self):
		"""Destructor."""
		del self

	def read_points(self, filename, thresh):
		# vetor para armazenar as tuplas que representarão os pontos
		points = []

		# leitura do arquivo com os pontos
		with open(filename, 'r') as filestream:
			# listas para scatterplot
			x = []
			y = []

			for line in filestream:
				# retirar caracteres especiais
				line = line.replace(' ', '')
				line = line.replace(';', '')
				line = line.rstrip()

				# dividir o texto da linha por meio da vírgula
				current_line = line.split(',')

				# unpacking das coordenadas do ponto
				a, b = current_line

				# conversão em valores inteiros
				u = int(a)
				v = int(b)

				# inserção dos pontos nas listas para scatterplot
				x.append(u)
				y.append(v)

				points.append((u,v))

		x = np.array(x)
		y = np.array(y)

		return points, x, y

	def calculate_weight(self, xa, ya, xb, yb):
		x = np.abs(xa - xb)
		y = np.abs(ya - yb)
		return np.sqrt((x*x) + (y*y))

	def generate_edges(self, points, thresh):
		edges = []

		size = len(points)

		for i in range(size):
			xa, ya = points[i]

			for j in range(i, size):
				if points[i] == points[j]:
					continue

				xb, yb = points[j]
				weight = self.calculate_weight(xa, ya, xb, yb)

				if weight <= thresh:
					edges.append((i, j, weight))

		return edges

	def generate_graph(self, edges):
		graph = nx.Graph()
		for e in edges:
			graph.add_edge(e[0], e[1], weight=e[2])

		return graph

	def degree_metrics(self, graph, plot=False):
		degree_list = graph.degree

		degrees = []
		for deg in degree_list:
			degrees.append(deg[1])

		degrees = np.array(degrees)
		deg, cnt = self._plot_degree_distribution(degrees, plot)

		# Min-max normalization to create a discrete distribution
		pk = np.interp(cnt, (min(cnt), max(cnt)), (0, 1))

		min_deg = np.min(degrees)
		max_deg = np.max(degrees)
		mean_deg = np.mean(degrees)

		# print ('Min degree: ' + str(min_deg))
		# print ('Max degree: ' + str(max_deg))
		# print ('Mean degree: ' + str(mean_deg))

		return min_deg, max_deg, mean_deg, pk

	def _plot_degree_distribution(self, degrees, plot=False):
		degree_count = collections.Counter(degrees)
		deg, cnt = zip(*degree_count.items())

		if plot:
			fig, ax = plt.subplots()
			plt.bar(deg, cnt, width=0.80, color='b')

			plt.title("Degree Histogram")
			plt.ylabel("Count")
			plt.xlabel("Degree")
			ax.set_xticks([d + 0.4 for d in deg])
			ax.set_xticklabels(deg)

			# draw graph in inset
			plt.axes([0.4, 0.4, 0.5, 0.5])
			Gcc = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)[0]
			pos = nx.spring_layout(G)
			plt.axis('off')
			nx.draw_networkx_nodes(G, pos, node_size=20)
			nx.draw_networkx_edges(G, pos, alpha=0.4)

			plt.show()

		return deg, cnt

	def adjacency_matrix(self, graph, binary=False):
		A = nx.adjacency_matrix(graph)
		A = A.todense()
		if binary:
			A[A != 0] = 1
			A = A.astype('uint8')

		return A

	def clustering_metrics(self, graph):
		tr = nx.transitivity(graph)
		avg = nx.average_clustering(graph)
		rc_raw = nx.rich_club_coefficient(graph, normalized=False)

		rc_vals = list(rc_raw.values())
		rc_edges = list(rc_raw.keys())
		num_edges = len(rc_edges)
		rc_arr = np.zeros([num_edges + 1, 2], dtype='object')

		for i in range(num_edges):
			rc_arr[i, 1] = rc_vals[i]

		# Add mean
		nonzero_arr_rich_club = np.delete(rc_arr[:, 1], [0])
		rc = np.mean(nonzero_arr_rich_club)

		del rc_raw
		del rc_edges
		del rc_vals
		del nonzero_arr_rich_club

		return tr, avg, rc

	def _entropy(self, pk):
		return entropy(pk)

	def centralities(self, graph):
		# Betweenness centrality
		bet = nx.betweenness_centrality(graph)
		# Closeness centrality
		clo = nx.closeness_centrality(graph)
		# Eigenvector centrality
		eig = nx.eigenvector_centrality(graph, max_iter=100000000)

		bet = np.array(list(bet.values()))
		clo = np.array(list(clo.values()))
		eig = np.array(list(eig.values()))

		return np.mean(bet), np.mean(clo), np.mean(eig)


	def fractal_dimension(self, Z, threshold=0.9):
		# -----------------------------------------------------------------------------
		# From https://en.wikipedia.org/wiki/Minkowski–Bouligand_dimension:
		#
		# In fractal geometry, the Minkowski–Bouligand dimension, also known as
		# Minkowski dimension or box-counting dimension, is a way of determining the
		# fractal dimension of a set S in a Euclidean space Rn, or more generally in a
		# metric space (X, d).
		# https://gist.github.com/rougier/e5eafc276a4e54f516ed5559df4242c0
		# -----------------------------------------------------------------------------

		# Only for 2d image
		assert(len(Z.shape) == 2)

		Z = Z / 255.0

		# From https://github.com/rougier/numpy-100 (#87)
		def boxcount(Z, k):
			S = np.add.reduceat(
				np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
								np.arange(0, Z.shape[1], k), axis=1)

			# We count non-empty (0) and non-full boxes (k*k)
			return len(np.where((S > 0) & (S < k*k))[0])


		# Transform Z into a binary array
		Z = (Z < threshold)

		# Minimal dimension of image
		p = min(Z.shape)

		# Greatest power of 2 less than or equal to p
		n = 2**np.floor(np.log(p)/np.log(2))

		# Extract the exponent
		n = int(np.log(n)/np.log(2))

		# Build successive box sizes (from 2**n down to 2**1)
		sizes = 2**np.arange(n, 1, -1)

		# Actual box counting with decreasing size
		counts = []
		for size in sizes:
			counts.append(boxcount(Z, size))

		# Fit the successive log(sizes) with log (counts)
		coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
		mb = -coeffs[0]

		# print("Minkowski–Bouligand dimension (computed): ", -coeffs[0])
		# print("Haussdorf dimension (theoretical):        ", (np.log(3)/np.log(2)))

		h = np.log(3)/np.log(2)

		if np.isnan(mb):
			mb = h

		return mb, h


	def extract_features(self, graph):

		graph_density = nx.density(graph)

		min_deg, max_deg, mean_deg, pk = self.degree_metrics(graph)

		tr, avg, rc = self.clustering_metrics(graph)

		entropy = self._entropy(pk)

		bet, clo, eig = self.centralities(graph)

		mb_dim, hd_dim = self.fractal_dimension(self.adjacency_matrix(graph))

		features = []

		features.append(graph_density)
		features.append(min_deg)
		features.append(max_deg)
		features.append(mean_deg)
		features.append(tr)
		features.append(avg)
		features.append(rc)
		features.append(entropy)
		features.append(bet)
		features.append(clo)
		features.append(eig)
		features.append(mb_dim)
		features.append(hd_dim)

		feature_vector = np.array(features)

		return feature_vector

	def generate_ground_truth_networks(self, n):

		p = np.random.random_sample()
		k = 2
		m = 2
		
		erdos_renyi = nx.fast_gnp_random_graph(n, p)
		watts_strogatz = nx.watts_strogatz_graph(n, k, p)
		barabasi_albert = nx.barabasi_albert_graph(n, m)

		return erdos_renyi, watts_strogatz, barabasi_albert
