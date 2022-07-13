'''
@author: Faizan-Uni-Stuttgart

May 23, 2022

5:11:51 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt; plt.ioff()

DEBUG_FLAG = False


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    ref_file = r"P:\Synchronize\IWS\QGIS_Neckar\hydmod\input_hyd_data\neckar_daily_discharge_1961_2015.csv"
    sim_file = r"P:\Synchronize\IWS\QGIS_Neckar\test_stop_rope_crit_03\02_hydrographs\calib_kfold_01__cats_outflow.csv"

    col = '420'

    beg_time = '1962-01-01'
    end_time = '1970-12-31'

    sep = ';'

    mpl_prms_dict = {
        'figure.figsize': (15, 8),
        'figure.dpi': 150,
        'font.size': 16,
        'lines.linewidth': 4,
        }

    out_path = Path(
        r'P:\Synchronize\IWS\Conferences_Seminars\PhD_Seminar_IWS\PhD_Seminar_2022\qser.png')
    #==========================================================================

    set_mpl_prms(mpl_prms_dict)

    ref_ser = pd.read_csv(ref_file, sep=sep, index_col=0).loc[beg_time:end_time, col].values
    sim_ser = pd.read_csv(sim_file, sep=sep, index_col=0).loc[beg_time:end_time, col].values

    plt.figure()

    plt.plot(ref_ser, alpha=0.80, lw=2.5, c='r', label='Reference')
    plt.plot(sim_ser, alpha=0.75, lw=1.5, c='b', label='Simulation')

    plt.grid()
    plt.gca().set_axisbelow(True)

    plt.xlabel('Time step (day)')
    plt.ylabel('Discharge ($m^3.s^{-1})$')

    plt.legend()

    # plt.show()

    plt.savefig(out_path, bbox_inches='tight')

    plt.close()
    return

    def set_mpl_prms(prms_dict):

        plt.rcParams.update(prms_dict)

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
