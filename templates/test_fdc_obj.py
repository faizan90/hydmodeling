'''
@author: Faizan-Uni-Stuttgart

Dec 29, 2020

10:24:43 AM

'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib.pyplot as plt

from hydmodeling import get_sim_probs_in_ref_cy, rank_sorted_arr_cy, get_ns_cy

plt.ioff()

DEBUG_FLAG = False


def get_sim_probs_in_ref(ref_vals_sort, ref_probs_sort, sim_vals_sort):

    '''
    All inputs must be sorted in an ascending.
    '''

    app_zero = 1e-15

    sim_probs_sort = np.full_like(sim_vals_sort, np.nan)

    n_ref_vals = ref_vals_sort.size
    n_sim_vals = sim_vals_sort.size

    min_j = 0  # To avoid testing against values that have been tested already.
    for i in range(n_sim_vals):
        x = sim_vals_sort[i]

        i1 = i2 = None

        if x < ref_vals_sort[0]:
            i1 = 0
            i2 = 1

        elif x > ref_vals_sort[-1]:
            i1 = -2
            i2 = -1

        else:
            for j in range(min_j, n_ref_vals - 1):

                # Double equals because similar values exist in reference.
                if ref_vals_sort[j] <= x <= ref_vals_sort[j + 1]:
                    i1 = j
                    i2 = j + 1

                    min_j = max(min_j, j - 1)

                    break

        if i1 is None:
            raise ValueError

        x1 = ref_vals_sort[i1]
        x2 = ref_vals_sort[i2]

        p1 = ref_probs_sort[i1]
        p2 = ref_probs_sort[i2]

        if -app_zero <= (x1 - x2) <= +app_zero:
            sim_probs_sort[i] = p1

        else:
            sim_probs_sort[i] = p1 + ((p2 - p1) * ((x - x1) / (x2 - x1)))

    assert np.all(np.isfinite(sim_probs_sort))
    return sim_probs_sort


def main():

    main_dir = Path(r'P:\Synchronize\IWS\QGIS_Neckar\fdc_obj')
    os.chdir(main_dir)

    ref_file = Path(r'neckar_daily_discharge_1961_2015.csv')
    sim_file = Path(r'calib_kfold_01__cats_outflow.csv')

    col = '420'

    beg_time = '1990-06-01'
    end_time = '1995-12-31'
#     end_time = '1962-01-31'

    off_idx = 365

    sep = ';'

#     plot_type = 'dist'
    plot_type = 'qq'

    fig_size = (10, 7)

    ref_vals = pd.read_csv(
        ref_file,
        sep=sep,
        index_col=0).loc[beg_time:end_time, col].values[off_idx:]

    sim_vals = pd.read_csv(
        sim_file,
        sep=sep,
        index_col=0).loc[beg_time:end_time, col].values[off_idx:]

    assert np.all(np.isfinite(ref_vals))
    assert np.all(np.isfinite(sim_vals))

    ref_vals_sort = np.sort(ref_vals)
    sim_vals_sort = np.sort(sim_vals)

    ref_ranks_sort = rankdata(ref_vals_sort)
    ref_probs_sort = ref_ranks_sort / (ref_vals_sort.size + 1.0)

    if False:  # Test python version
        t_beg = timeit.default_timer()

        sim_probs_sort = get_sim_probs_in_ref(
            ref_vals_sort, ref_probs_sort, sim_vals_sort)

        print('python time:', timeit.default_timer() - t_beg)

        err = ((ref_probs_sort - sim_probs_sort) ** 2).sum()
        print('Squared difference sum in probs:', err)

    if True:  # Test cython version
        t_beg = timeit.default_timer()

        sim_probs_sort = get_sim_probs_in_ref_cy(
            ref_vals_sort, ref_probs_sort, sim_vals_sort)

        print('cython time:', timeit.default_timer() - t_beg)

        err = ((ref_probs_sort - sim_probs_sort) ** 2).sum()
        print('Squared difference sum in probs:', err)

    if True:  # Compare cython and scipy versions of ranking data.
        ref_ranks_sort_cy = rank_sorted_arr_cy(ref_vals_sort, 3)
        assert np.all(np.isclose(ref_ranks_sort, ref_ranks_sort_cy))

    if True:
        ns_fdc = get_ns_cy(ref_probs_sort, sim_probs_sort, 0)
        print('NS FDC:', ns_fdc)

    if True:
        plt.figure(figsize=fig_size)

        if plot_type == 'dist':
            plt.semilogx(ref_vals_sort, ref_probs_sort, label='ref', alpha=0.75)
            plt.semilogx(sim_vals_sort, sim_probs_sort, label='sim', alpha=0.75)

            plt.xlabel('Discharge')
            plt.ylabel('Probability in ref')

        elif plot_type == 'qq':
            plt.plot(ref_probs_sort, ref_probs_sort, label='ref', alpha=0.75)
            plt.plot(ref_probs_sort, sim_probs_sort, label='sim', alpha=0.75)

            plt.xlabel('Reference probability')
            plt.ylabel('Probability in reference')

            plt.gca().set_aspect('equal')

        else:
            raise ValueError(f'Unknown plot_type: {plot_type}!')

        plt.title(
            f'Station: {col}, Begin time: {beg_time}, End time: {end_time}, '
            f'N={ref_vals_sort.size}')

        plt.legend()
        plt.grid()
        plt.gca().set_axisbelow(True)

        plt.show()

    return


if __name__ == '__main__':

    _save_log_ = False
    if _save_log_:
        from datetime import datetime
        from std_logger import StdFileLoggerCtrl

        # save all console activity to out_log_file
        out_log_file = os.path.join(
            r'P:\Synchronize\python_script_logs\\%s_log_%s.log' % (
            os.path.basename(__file__),
            datetime.now().strftime('%Y%m%d%H%M%S')))

        log_link = StdFileLoggerCtrl(out_log_file)

    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack
    # 2. "up" move the stack up to an older frame
    # 3. "down" move the stack down to a newer frame
    # 4. "interact" start an interactive interpreter
    #==========================================================================

    if DEBUG_FLAG:
        try:
            main()

        except:
            import pdb
            pdb.post_mortem()

    else:
        main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
