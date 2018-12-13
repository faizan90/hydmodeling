'''
Dec 13, 2018
@author: Faizan-Uni-Stuttgart

'''
import os
import timeit
import time
from pathlib import Path

import numpy as np
import pandas as pd

import pyximport
pyximport.install()

from hydmodeling.models.fourtrans.dfti import cmpt_real_four_trans_1d_cy


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    main_dir = Path(r'P:\Synchronize\IWS\QGIS_Neckar\test_resampled_obj_ftns')
    os.chdir(main_dir)

    in_df = pd.read_csv(
        r'cat_411_qsims.csv',
        sep=';',
        usecols=['obs'])

    in_df.index = pd.date_range('2005-01-01', periods=in_df.shape[0])
    in_df = in_df.loc[:'2015-12-31']
    if (in_df.shape[0] % 2):
        in_df = in_df.iloc[:-1, :]

    n_pts = in_df.shape[0]

    orig = in_df.loc[:, 'obs'].values.astype(np.float64, order='c')
    ft = np.zeros((orig.shape[0] // 2) + 1, dtype=np.complex128)
    amps = np.zeros((orig.shape[0] // 2) - 1, dtype=np.float64)
    angs = amps.copy(order='c')

    cmpt_real_four_trans_1d_cy(orig, ft, amps, angs)

    np_ft = np.fft.rfft(orig)
    np_amps = np.abs(np_ft)[1:(n_pts // 2)]
    np_angs = np.angle(np_ft)[1:(n_pts // 2)]

    assert np.all(np.isclose(ft, np_ft))
    assert np.all(np.isclose(amps, np_amps))
    assert np.all(np.isclose(angs, np_angs))
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
    START = timeit.default_timer()  # to get the runtime of the program

    main()

    STOP = timeit.default_timer()  # Ending time
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
