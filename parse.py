# -*- coding: utf-8 -*-

import pandas as pd

spam = pd.read_csv('./spam.csv')
spam.to_hdf('datasets.hdf','spam',complib='blosc',complevel=9)

letter = pd.read_csv('./letter.csv')
letter.to_hdf('datasets.hdf','letter_original',complib='blosc',complevel=9)