'''
@author: Faizan-Uni-Stuttgart

Dec 2, 2021

12:09:53 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np

from hydmodeling.models.ft import get_pocket_real_dft

DEBUG_FLAG = False


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    #==========================================================================

    # x = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)

    x = np.arange(200000, dtype=np.float64)

    beg_time = timeit.default_timer()
    y = get_pocket_real_dft(x)
    end_time = timeit.default_timer()

    print(f'Pocket FFT took: {end_time - beg_time:0.3f}')

    beg_time = timeit.default_timer()
    z = np.fft.rfft(x)
    end_time = timeit.default_timer()

    print(f'Numpy FFT took: {end_time - beg_time:0.3f}')

    # print(x)
    # print(y)
    # print(z)

    print(y[0], y[-1])
    print(z[0], z[-1])

    assert np.all(np.isclose(y, z))

    return


if __name__ == '__main__':
    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack,
    # 2. "up" move the stack up to an older frame,
    # 3. "down" move the stack down to a newer frame, and
    # 4. "interact" start an interactive interpreter.
    #==========================================================================

    if DEBUG_FLAG:
        try:
            main()

        except:
            pre_stack = tb.format_stack()[:-1]

            err_tb = list(tb.TracebackException(*sys.exc_info()).format())

            lines = [err_tb[0]] + pre_stack + err_tb[2:]

            for line in lines:
                print(line, file=sys.stderr, end='')

            import pdb
            pdb.post_mortem()
    else:
        main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))
