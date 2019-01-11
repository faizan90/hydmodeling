'''
Created on Jan 11, 2019

@author: Faizan-Uni
'''
import numpy as np
import pandas as pd

LC_CLRS = ['blue', 'green', 'purple', 'orange', 'cyan', 'pink']


def get_fdc(in_ser):

    '''Get flow duration curve'''

    assert isinstance(in_ser, pd.Series), 'Expected a pd.Series object!'

    probs = (in_ser.rank(ascending=False) / (in_ser.shape[0] + 1)).values
    vals = in_ser.values.copy()

    sort_idxs = np.argsort(probs)

    probs = probs[sort_idxs]
    vals = vals[sort_idxs]
    return probs, vals
