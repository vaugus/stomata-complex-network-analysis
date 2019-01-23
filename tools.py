#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tools module."""

__version__ = '1.0'
__author__ = 'Victor Augusto'
__copyright__ = "Copyright (c) 2018 - Victor Augusto"

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

class Tools(object):
    """Class with tools for complex network visualization operations."""

    def __init__(self):
        """Constructor."""

    def __del__(self):
        """Destructor."""
        del self

    def save_image(self, data, dpi=100):
        """save_image method.
        Saves the ndarray object as a png file.
        :param data     Image to be saved in a png format file.
        :param dpi      Amount of dots per inch, the printing resolution.
        """
        # Get the ndarray shape and set the amount of inches in the matplotlib figure.
        shape = np.shape(data)[0:2][::-1]
        size = [ float(i) / dpi for i in shape]

        fig = plt.figure()
        fig.set_size_inches(size)
        ax = plt.Axes(fig,[0,0,1,1])

        # Do not print the default matplotlib axis.
        ax.set_axis_off()
        fig.add_axes(ax)

        plt.imshow(data, cmap="gray")

        # Save it as a png file.
        fig.savefig('out.png', dpi=dpi)

        plt.show()

    def plot_dataset(self, x, y):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        def onclick(event):
            print( 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
                event.button, event.x, event.y, event.xdata, event.ydata))

        cid = fig.canvas.mpl_connect('button_press_event', onclick)

        plt.scatter(x, y)
        plt.show()

    def plot_graph(self, graph):
        nx.draw(graph)
        plt.show()

    def save_graph(self, graph, path, dpi=100):
        """save_image method.
        Saves the ndarray object as a png file.
        :param data     Image to be saved in a png format file.
        :param dpi      Amount of dots per inch, the printing resolution.
        """
        # Get the ndarray shape and set the amount of inches in the matplotlib figure.
        # shape = np.shape(data)[0:2][::-1]
        # size = [ float(i) / dpi for i in shape]

        fig = plt.figure()
        # fig.set_size_inches(size)
        ax = plt.Axes(fig,[0,0,1,1])

        # # Do not print the default matplotlib axis.
        ax.set_axis_off()
        fig.add_axes(ax)

        nx.draw(graph)

        arr = path.split('/')
        p = '/home/victor/Documents/python-msc-workspace/laura/'

        f = arr[-1].replace('.txt', '')

        p = p + 'output/' + arr[-4] + '/' + arr[-3] + '/' + arr[-2] + '/'

        # Save it as a png file.
        fig.savefig(p + f +'.png', dpi=dpi)

    def pretty_print_parameters(self, filename, threshold):
        arr = filename.split('/')
        
        section = arr[-1].replace('.txt', '')
        leaf = arr[-2].replace('F','')
        group = arr[-3]
        day = arr[-4]

        print('=============== Info')
        print('\nDay: ' + day)
        print('\nGroup: ' + group)
        print('\nLeaf: ' + leaf)
        print('\nSection: ' + section)
        print('\nDistance Threshold: ' + str(threshold))
        print('\n')