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
