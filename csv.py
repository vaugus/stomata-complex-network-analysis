#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tools module."""

__version__ = '1.0'
__author__ = 'Victor Augusto'
__copyright__ = "Copyright (c) 2018 - Victor Augusto"

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

class CSVTools(object):
    """Class with tools for complex network visualization operations."""

    def write_fft_csv(self, data, params):
        method, name = params
        index = ['mid', 'high', 'highest', 'kanjar']
        
        df = pd.DataFrame(data, columns=index)

        names = [name] * data.shape[0]
        colourspaces = ['grayscale', 'hsv', 'lab']

        df.insert(loc=0, column='colourspace', value=colourspaces)
        df.insert(loc=0, column='image', value=names)

        if name == 'airplane' or name == 'airplane_partial_blur':
            df.to_csv(constants.AIRPLANE_FFT_OUTPUT_PATH + name + '.csv', index=False)

        if name[0:4] == 'card':
            df.to_csv(constants.CARD_FFT_OUTPUT_PATH + name + '.csv', index=False)

        if name[0:6] == 'coin_o':
            df.to_csv(constants.COIN_O_FFT_OUTPUT_PATH + name + '.csv', index=False)

        if name[0:6] == 'coin_v':
            df.to_csv(constants.COIN_V_FFT_OUTPUT_PATH + name + '.csv', index=False)