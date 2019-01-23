'''
Created on Jan 15, 2019

@author: Faizan-Uni
'''
import os
import sys
from functools import wraps
import traceback as tb

import numpy as np
import pandas as pd

LC_CLRS = ['blue', 'green', 'purple', 'orange', 'cyan', 'deeppink']
text_sep = ';'


def traceback_wrapper(func):

    @wraps(func)
    def wrapper(*args, **kwargs):

        func_res = None

        try:
            func_res = func(*args, **kwargs)

        except:
            pre_stack = tb.format_stack()[:-1]

            err_tb = list(tb.TracebackException(*sys.exc_info()).format())

            lines = [err_tb[0]] + pre_stack + err_tb[2:]

            for line in lines:
                print(line, file=sys.stderr, end='')

        return func_res

    return wrapper


def mkdir_hm(dir_path):

    if not os.path.exists(dir_path):
        try:
            os.mkdir(dir_path)

        except FileExistsError:
            pass

    return


def get_fdc(in_ser):

    '''Get flow duration curve'''

    assert isinstance(in_ser, pd.Series), 'Expected a pd.Series object!'

    probs = (in_ser.rank(ascending=False) / (in_ser.shape[0] + 1)).values
    vals = in_ser.values.copy()

    sort_idxs = np.argsort(probs)

    probs = probs[sort_idxs]
    vals = vals[sort_idxs]
    return probs, vals


def df_to_tex(in_df, tex_loc):

    '''Need booktabs module to read it in the tex editor'''

    assert os.path.isfile(tex_loc)
    assert isinstance(in_df, (pd.Series, pd.DataFrame))

    with open(tex_loc, 'w') as hdl:
        hdl.write(in_df.to_latex())
    return


if __name__ == '__main__':

    @traceback_wrapper
    def do_stuff(a, x=2):
        print('Doing stuff...', str(a), str(x))
        raise TypeError(4)
        return

    @traceback_wrapper
    def do_stuff2():
        print('Doinf stuff2....')
        return

    do_stuff(1)

    import time
    time.sleep(0.1)
    do_stuff2()
