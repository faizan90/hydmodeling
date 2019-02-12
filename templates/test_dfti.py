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

from hydmodeling.models.ft.dfti import (
    cmpt_real_four_trans_1d_cy,
    cmpt_cumm_freq_pcorrs_cy)


def main():

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    main_dir = Path(r'P:\Synchronize\IWS\QGIS_Neckar\rockenau_six_cats_dist_01_ns\12_discharge_sims')
    os.chdir(main_dir)

    in_df = pd.read_csv(
        r'cat_3470_calib_final_opt_qsims.csv',
        sep=';',
        usecols=['obs', 'kf_01_sim_0000'])

    in_df.index = pd.date_range('2005-01-01', periods=in_df.shape[0])
    in_df = in_df.loc['2007-01-01':'2015-12-31']
    if (in_df.shape[0] % 2):
        in_df = in_df.iloc[:-1, :]

    n_pts = in_df.shape[0]

    obs_orig = in_df.loc[:, 'obs'].values.astype(np.float64, order='c')
    obs_ft = np.zeros((obs_orig.shape[0] // 2) + 1, dtype=np.complex128)
    obs_amps = np.zeros((obs_orig.shape[0] // 2) - 1, dtype=np.float64)
    obs_angs = obs_amps.copy(order='c')

    cmpt_real_four_trans_1d_cy(obs_orig, obs_ft, obs_amps, obs_angs)

    np_obs_ft = np.fft.rfft(obs_orig)
    np_obs_amps = np.abs(np_obs_ft)[1:(n_pts // 2)]
    np_obs_angs = np.angle(np_obs_ft)[1:(n_pts // 2)]

    assert np.all(np.isclose(obs_ft, np_obs_ft))
    assert np.all(np.isclose(obs_amps, np_obs_amps))
    assert np.all(np.isclose(obs_angs, np_obs_angs))

    sim_orig = in_df.loc[:, 'kf_01_sim_0000'].values.astype(
        np.float64, order='c')
    sim_ft = np.zeros((sim_orig.shape[0] // 2) + 1, dtype=np.complex128)
    sim_amps = np.zeros((sim_orig.shape[0] // 2) - 1, dtype=np.float64)
    sim_angs = sim_amps.copy(order='c')

    cumm_corrs = sim_amps.copy(order='c')

    cmpt_cumm_freq_pcorrs_cy(
        obs_orig,
        obs_ft,
        obs_amps,
        obs_angs,
        sim_orig,
        sim_ft,
        sim_amps,
        sim_angs,
        cumm_corrs)

    pcorr = np.corrcoef(obs_orig, sim_orig)[0, 1]
    pcorr_ft = cumm_corrs[(n_pts // 2) - 2]

    np_sim_ft = np.fft.rfft(sim_orig)
    np_sim_amps = np.abs(np_sim_ft)[1:(n_pts // 2)]
    np_sim_angs = np.angle(np_sim_ft)[1:(n_pts // 2)]

    indiv_cov_arr = (np_obs_amps * np_sim_amps) * (
        np.cos(np_obs_angs - np_sim_angs))

    tot_cov = indiv_cov_arr.sum()

    cumm_rho_arr = indiv_cov_arr.cumsum() / tot_cov

    np_ft_corr = tot_cov / (
        ((np_obs_amps ** 2).sum() *
         (np_sim_amps ** 2).sum()) ** 0.5)

    cumm_rho_arr = cumm_rho_arr * np_ft_corr

    print(f'Pearson corr: {pcorr}')
    print(f'FT corr: {pcorr_ft}')
    print(f'NP FT corr: {np_ft_corr}')
    print(f'Cumm. corrs sq. diff: {((cumm_rho_arr - cumm_corrs)**2).sum()}')
    assert np.all(np.isclose(cumm_rho_arr, cumm_corrs))
    print('All tests passed!')
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
