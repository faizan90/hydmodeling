'''
Created on Aug 2, 2019

@author: Faizan-Uni
'''

import os

import h5py
import numpy as np
import matplotlib.pyplot as plt

from ..models import get_asymms_sample
from ..misc import mkdir_hm, traceback_wrapper

plt.ioff()


@traceback_wrapper
def plot_cat_stats(plot_args):

    cat_db, = plot_args

    with h5py.File(cat_db, 'r') as db:
        out_dir = db['data'].attrs['main']
        kfolds = db['data'].attrs['kfolds']
        cat = db.attrs['cat']

        off_idx = db['data'].attrs['off_idx']

        for kf in range(1, kfolds + 1):
            for run_type in ['calib', 'valid']:
                if run_type not in db:
                    continue

                plot_cat_stats_cls = PlotCatStats(
                    db[f'{run_type}/kf_{kf:02d}/qact_arr'][...],
                    db[f'{run_type}/kf_{kf:02d}/qsim_arr'][...],
                    off_idx,
                    cat,
                    kf,
                    run_type,
                    os.path.join(out_dir, '13_stats'),
                    )

                plot_cat_stats_cls.plot_emp_cops()
                plot_cat_stats_cls.plot_fts()

    return


class PlotCatStats:

    '''For internal use only'''

    def __init__(
            self,
            qobs_arr,
            qsim_arr,
            off_idx,
            cat,
            kf,
            run_type,
            out_dir,
            ):

        self._qobs_arr = qobs_arr[off_idx:]
        self._qsim_arr = qsim_arr[off_idx:]
        self._off_idx = off_idx
        self._cat = cat
        self._kf = kf
        self._run_type = run_type
        self._out_dir = out_dir
        self._n_steps = self._qobs_arr.shape[0]

        self._qobs_mean = self._qobs_arr.mean()
        self._qobs_var = self._qobs_arr.var()

        self._qsim_mean = self._qsim_arr.mean()
        self._qsim_var = self._qsim_arr.var()

        mkdir_hm(self._out_dir)

        self._qobs_probs = (
            np.argsort(np.argsort(self._qobs_arr)) + 1) / (self._n_steps + 1.0)

        self._qsim_probs = (
            np.argsort(np.argsort(self._qsim_arr)) + 1) / (self._n_steps + 1.0)

        return

    def plot_emp_cops(self):

        ecops_dir = os.path.join(self._out_dir, 'ecops')

        mkdir_hm(ecops_dir)

        spcorr = np.corrcoef(self._qobs_arr, self._qsim_arr)[0, 1]

        asymm_1, asymm_2 = get_asymms_sample(
            self._qobs_probs,
            self._qsim_probs)

        plt.figure(figsize=(12, 10))

        plt.scatter(
            self._qobs_probs,
            self._qsim_probs,
            alpha=0.1,
            color='blue')

        plt.xlabel('Observed Discharge')
        plt.ylabel('Simulated Discharge')

        plt.grid()

        plt.title(
            f'Empirical copula between observed and simulated discharges\n'
            f'Catchment: {self._cat}, Kf: {self._kf}, '
            f'Run Type: {self._run_type.upper()}, Steps: {self._n_steps}\n'
            f'Spearman Correlation: {spcorr:0.4f}, '
            f'Asymm_1: {asymm_1:0.4E}, Asymm_2: {asymm_2:0.4E}')

        out_fig_name = (
            f'ecop_obs_sim_kf_{self._kf:02d}_{self._run_type}_'
            f'cat_{self._cat}.png')

        plt.savefig(
            os.path.join(ecops_dir, out_fig_name),
            bbox_inches='tight')

        plt.close()
        return

    def plot_fts(self):

        fts_dir = os.path.join(self._out_dir, 'fts')

        mkdir_hm(fts_dir)

        n_ft_pts = self._qobs_arr.shape[0]
        if n_ft_pts % 2:
            n_ft_pts -= 1

        ft_ofst_idx = 1

        ft_obs = np.fft.rfft(self._qobs_arr[:n_ft_pts])[ft_ofst_idx:]
        ft_obs_phas = np.angle(ft_obs)
        ft_obs_amps = np.abs(ft_obs)

        freq_cov_cntrb_obs = np.cumsum(ft_obs_amps ** 2)

        max_cov_obs = freq_cov_cntrb_obs[-1]
        freq_cov_cntrb_obs /= max_cov_obs

        freq_cov_cntrb_grad_obs = (
            freq_cov_cntrb_obs[1:] - freq_cov_cntrb_obs[:-1])

        ft_sim = np.fft.rfft(self._qsim_arr[:n_ft_pts])[ft_ofst_idx:]
        ft_sim_phas = np.angle(ft_sim)
        ft_sim_amps = np.abs(ft_sim)

        max_cov_sim = (
            (ft_obs_amps ** 2).sum() * (ft_sim_amps ** 2).sum()) ** 0.5

        freq_cov_cntrb_sim = np.cumsum(
            (ft_obs_amps * ft_sim_amps) * np.cos(ft_obs_phas - ft_sim_phas))
        freq_cov_cntrb_sim /= max_cov_sim

        freq_cov_cntrb_grad_sim = (
            freq_cov_cntrb_sim[1:] - freq_cov_cntrb_sim[:-1])

        self._plot_fts_wvcbs(
            freq_cov_cntrb_obs,
            freq_cov_cntrb_sim,
            fts_dir)

        self._plot_fts_wvcbs_grad(
            freq_cov_cntrb_grad_obs,
            freq_cov_cntrb_grad_sim,
            fts_dir)

        self._plot_fts_phas_diff(ft_obs_phas, ft_sim_phas, fts_dir)

        self._plot_fts_amps_diff(ft_obs_amps, ft_sim_amps, fts_dir)

        self._plot_fts_amps(ft_obs_amps, ft_sim_amps, fts_dir)

        self._plot_fts_abs_diff(ft_obs, ft_sim, fts_dir)
        return

    def _plot_fts_abs_diff(self, ft_obs, ft_sim, fts_dir):

        line_alpha = 0.7
        line_lw = 0.9

        # abs difference
        plt.figure(figsize=(15, 7))

        plt.semilogx(
            np.abs(ft_obs - ft_sim),
            alpha=line_alpha + 0.2,
            color='red',
            lw=line_lw)

        plt.xlabel('Frequency')
        plt.ylabel('Absolute FT difference')

        plt.grid()

        plt.title(
            f'Discharge Fourier transform absolute difference\n'
            f'Catchment: {self._cat}, Kf: {self._kf}, '
            f'Run Type: {self._run_type.upper()}, Steps: {self._n_steps}\n'
            f'Obs. and Sim. mean: {self._qobs_mean:0.3f}, '
            f'{self._qsim_mean:0.3f}, '
            f'Obs. and Sim. variance: {self._qobs_var:0.3f}, '
            f'{self._qsim_var:0.3f}'
            )

        cntrb_fig_name = (
            f'ft_abs_diff_kf_{self._kf:02d}_{self._run_type}_'
            f'cat_{self._cat}.png')

        plt.savefig(os.path.join(fts_dir, cntrb_fig_name), bbox_inches='tight')
        plt.close()
        return

    def _plot_fts_amps(self, ft_obs_amps, ft_sim_amps, fts_dir):

        line_alpha = 0.7
        line_lw = 0.9

        # amplitudes
        plt.figure(figsize=(15, 7))

        plt.semilogx(
            ft_obs_amps,
            label='obs',
            alpha=line_alpha + 0.2,
            color='red',
            lw=line_lw)

        plt.semilogx(
            ft_sim_amps,
            label='sim',
            alpha=line_alpha,
            color='blue',
            lw=line_lw)

        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')

        plt.grid()
        plt.legend()

        plt.title(
            f'Discharge Fourier amplitudes\n'
            f'Catchment: {self._cat}, Kf: {self._kf}, '
            f'Run Type: {self._run_type.upper()}, Steps: {self._n_steps}\n'
            f'Obs. and Sim. mean: {self._qobs_mean:0.3f}, '
            f'{self._qsim_mean:0.3f}, '
            f'Obs. and Sim. variance: {self._qobs_var:0.3f}, '
            f'{self._qsim_var:0.3f}'
            )

        cntrb_fig_name = (
            f'ft_amps_kf_{self._kf:02d}_{self._run_type}_'
            f'cat_{self._cat}.png')

        plt.savefig(os.path.join(fts_dir, cntrb_fig_name), bbox_inches='tight')
        plt.close()
        return

    def _plot_fts_phas_diff(self, ft_obs_phas, ft_sim_phas, fts_dir):

        line_alpha = 0.7
        line_lw = 0.9

        # phase difference
        plt.figure(figsize=(15, 7))

        plt.plot(
            ft_obs_phas - ft_sim_phas,
            alpha=line_alpha + 0.2,
            color='red',
            lw=line_lw)

        plt.xlabel('Frequency')
        plt.ylabel('Phase difference')

        plt.grid()

        plt.title(
            f'Discharge Fourier phase difference\n'
            f'Catchment: {self._cat}, Kf: {self._kf}, '
            f'Run Type: {self._run_type.upper()}, Steps: {self._n_steps}'
            )

        cntrb_fig_name = (
            f'ft_phas_diff_kf_{self._kf:02d}_{self._run_type}_'
            f'cat_{self._cat}.png')

        plt.savefig(os.path.join(fts_dir, cntrb_fig_name), bbox_inches='tight')
        plt.close()
        return

    def _plot_fts_amps_diff(self, ft_obs_amps, ft_sim_amps, fts_dir):

        line_alpha = 0.7
        line_lw = 0.9

        # normalized amplitude difference
        plt.figure(figsize=(15, 7))

        plt.plot(
            (ft_obs_amps - ft_sim_amps) / ft_obs_amps,
            alpha=line_alpha + 0.2,
            color='red',
            lw=line_lw)

        plt.xlabel('Frequency')
        plt.ylabel('Amplitude difference')

        plt.grid()

        plt.title(
            f'Disharge Fourier normalized amplitude difference\n'
            f'Catchment: {self._cat}, Kf: {self._kf}, '
            f'Run Type: {self._run_type.upper()}, Steps: {self._n_steps}\n'
            f'Obs. and Sim. mean: {self._qobs_mean:0.3f}, '
            f'{self._qsim_mean:0.3f}, '
            f'Obs. and Sim. variance: {self._qobs_var:0.3f}, '
            f'{self._qsim_var:0.3f}'
            )

        cntrb_fig_name = (
            f'ft_amps_diff_kf_{self._kf:02d}_{self._run_type}_'
            f'cat_{self._cat}.png')

        plt.savefig(os.path.join(fts_dir, cntrb_fig_name), bbox_inches='tight')
        plt.close()
        return

    def _plot_fts_wvcbs(self, freq_cov_cntrb_obs, freq_cov_cntrb_sim, fts_dir):

        line_alpha = 0.7
        line_lw = 0.9

        # wvcb
        plt.figure(figsize=(15, 7))

        plt.semilogx(
            freq_cov_cntrb_obs,
            label='obs',
            alpha=line_alpha + 0.2,
            color='red',
            lw=line_lw)

        plt.semilogx(
            freq_cov_cntrb_sim,
            label='sim',
            alpha=line_alpha,
            color='blue',
            lw=line_lw)

        plt.xlabel('Frequency')
        plt.ylabel('Cumulative correlation contribution')

        plt.legend()
        plt.grid()

        plt.title(
            f'Dscharge Fourier frequency correlation contribution\n'
            f'Catchment: {self._cat}, Kf: {self._kf}, '
            f'Run Type: {self._run_type.upper()}, Steps: {self._n_steps}\n'
            f'Sim-to-Obs fourier correlation: {freq_cov_cntrb_sim[-1]:0.4f}\n'
            f'Obs. and Sim. mean: {self._qobs_mean:0.3f}, '
            f'{self._qsim_mean:0.3f}, '
            f'Obs. and Sim. variance: {self._qobs_var:0.3f}, '
            f'{self._qsim_var:0.3f}'
            )

        cntrb_fig_name = (
            f'ft_full_wvcb_kf_{self._kf:02d}_{self._run_type}_'
            f'cat_{self._cat}.png')

        plt.savefig(os.path.join(fts_dir, cntrb_fig_name), bbox_inches='tight')
        plt.close()
        return

    def _plot_fts_wvcbs_grad(
            self,
            freq_cov_cntrb_grad_obs,
            freq_cov_cntrb_grad_sim,
            fts_dir):

        line_alpha = 0.7
        line_lw = 0.9

        # wvcb gradient
        plt.figure(figsize=(15, 7))

        plt.semilogx(
            freq_cov_cntrb_grad_obs,
            label='obs',
            alpha=line_alpha + 0.2,
            color='red',
            lw=line_lw)

        plt.semilogx(
            freq_cov_cntrb_grad_sim,
            label='sim',
            alpha=line_alpha,
            color='blue',
            lw=line_lw)

        plt.xlabel('Frequency')
        plt.ylabel('Cumulative correlation contribution gradient')

        plt.legend()
        plt.grid()

        plt.title(
            f'Discharge Fourier frequency correlation contribution gradient\n'
            f'Catchment: {self._cat}, Kf: {self._kf}, '
            f'Run Type: {self._run_type.upper()}, Steps: {self._n_steps}\n'
            f'Obs. and Sim. mean: {self._qobs_mean:0.3f}, '
            f'{self._qsim_mean:0.3f}, '
            f'Obs. and Sim. variance: {self._qobs_var:0.3f}, '
            f'{self._qsim_var:0.3f}'
            )

        cntrb_fig_name = (
            f'ft_full_wvcb_grad_kf_{self._kf:02d}_{self._run_type}_'
            f'cat_{self._cat}.png')

        plt.savefig(os.path.join(fts_dir, cntrb_fig_name), bbox_inches='tight')
        plt.close()
        return
