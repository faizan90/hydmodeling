'''
Created on Aug 2, 2019

@author: Faizan-Uni
'''

import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
from scipy.interpolate import interp1d

from ..models import (
    get_asymms_sample,
    get_ns_cy,
    get_ln_ns_cy,
    get_kge_cy)
from ..misc import mkdir_hm, traceback_wrapper

plt.ioff()


@traceback_wrapper
def plot_cat_diags(plot_args):

    cat_db, = plot_args

    with h5py.File(cat_db, 'r') as db:
        out_dir = db['data'].attrs['main']
        kfolds = db['data'].attrs['kfolds']
        cat = db.attrs['cat']

        off_idx = db['data'].attrs['off_idx']

#         grid_rows = db['data/rows'][...]
#         grid_rows -= grid_rows.min()
#
#         grid_cols = db['data/cols'][...]
#         grid_cols -= grid_cols.min()

        for kf in range(1, kfolds + 1):
            for run_type in ['calib', 'valid']:
                if run_type not in db:
                    continue

                if (('qact_arr' not in db[f'{run_type}/kf_{kf:02d}']) or
                    ('qsim_arr' not in db[f'{run_type}/kf_{kf:02d}'])):

                    continue

                plot_cat_diags_1d_cls = PlotCatDiagnostics1D(
                    db[f'{run_type}/kf_{kf:02d}/qact_arr'][...],
                    db[f'{run_type}/kf_{kf:02d}/qsim_arr'][...],
                    db[f'{run_type}/kf_{kf:02d}/ppt_arr'][...],
                    off_idx,
                    cat,
                    kf,
                    run_type,
#                     grid_rows,
#                     grid_cols,
                    os.path.join(out_dir, '13_diagnostics_1D'),
                    )

                plot_cat_diags_1d_cls.plot_emp_cops()
                plot_cat_diags_1d_cls.plot_fts()
                plot_cat_diags_1d_cls.plot_lorenz_curves()
                plot_cat_diags_1d_cls.plot_quantile_effs()
                plot_cat_diags_1d_cls.plot_sorted_sq_diffs()
                plot_cat_diags_1d_cls.plot_peak_qevents()
                plot_cat_diags_1d_cls.plot_mw_discharge_ratios()
                plot_cat_diags_1d_cls.plot_hi_err_qevents()
    return


class PlotCatsDiagnostics2D:

    '''For internal use only'''

    def __init__(self):
        return


class PlotCatDiagnostics1D:

    '''For internal use only'''

    def __init__(
            self,
            qobs_arr,
            qsim_arr,
            ppt_arr,
            off_idx,
            cat,
            kf,
            run_type,
#             grid_rows,
#             grid_cols,
            out_dir,
            ):

        self._qobs_arr = qobs_arr[off_idx:]
        self._qsim_arr = qsim_arr[off_idx:]
        self._ppt_arr = ppt_arr.mean(axis=0)[off_idx:]
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

        self._eff_ftns_dict = {
            'ns': get_ns_cy,
            'ln_ns': get_ln_ns_cy,
            'kge': get_kge_cy,
            }

        mkdir_hm(self._out_dir)

        self._qobs_ranks = np.argsort(np.argsort(self._qobs_arr)) + 1
        self._qobs_probs = self._qobs_ranks / (self._n_steps + 1.0)

        self._qsim_ranks = np.argsort(np.argsort(self._qsim_arr)) + 1
        self._qsim_probs = self._qsim_ranks / (self._n_steps + 1.0)

        self._ppt_ranks = np.argsort(np.argsort(self._ppt_arr)) + 1

#         if ppt_arr.shape[0] > 1:
#             self._ppt_dist_arr = ppt_arr[:, off_idx:]
#             self._grid_rows = grid_rows
#             self._grid_cols = grid_cols
#             self._grid_shape = (
#                 self._grid_rows.max() + 1, self._grid_cols.max() + 1)
#
#         else:
#             self._ppt_dist_arr = None
#             self._grid_rows = None
#             self._grid_cols = None
#             self._grid_shape = None
        return

    def plot_hi_err_qevents(self):

        n_evts = 10
        bef_steps = 10
        aft_steps = 10

        assert 0 < n_evts < self._n_steps

        sq_diffs = (self._qobs_arr - self._qsim_arr) ** 2

        qmax = max(self._qobs_arr.max(), self._qsim_arr.max())
        sum_sq_diffs = sq_diffs.sum()
        ppt_max = self._ppt_arr.max()

        qerr_idxs = np.argsort(sq_diffs)[::-1]
        hi_qerr_idxs = [qerr_idxs[0]]

        i = 0
        for qerr_idx in qerr_idxs[1:]:
            take_idx = True

            for hi_qerr_idx in hi_qerr_idxs:
                if ((hi_qerr_idx - bef_steps) <
                    qerr_idx <
                    (hi_qerr_idx + aft_steps)):

                    take_idx = False
                    break

            if take_idx:
                hi_qerr_idxs.append(qerr_idx)
                i += 1

            if i == n_evts:
                break

        self._plot_hi_err_qevents(
            hi_qerr_idxs,
            ppt_max,
            qmax,
            bef_steps,
            aft_steps,
            sq_diffs,
            sum_sq_diffs)
#
#         if self._ppt_dist_arr is not None:
#             self._plot_hi_err_qevents_2d_ppt(
#                 hi_qerr_idxs,
#                 ppt_max,
#                 bef_steps,
#                 aft_steps,
#                 sq_diffs,
#                 sum_sq_diffs)
        return

    def plot_mw_discharge_ratios(self):

        out_dir = os.path.join(self._out_dir, 'mw_ratios')
        mkdir_hm(out_dir)

        line_alpha = 0.7
        line_lw = 0.9

        ws = 365

        (ws_x_crds,
         qobs_mw_mean_arr,
         qsim_mw_mean_arr,
         qmw_diff_arr,
         qmw_ratio_arr) = (
            self._get_mw_qdiff_ratio_arrs(ws))

        fig = plt.figure(figsize=(15, 7))

        mwq_ax = plt.subplot2grid(
            (4, 1), (0, 0), rowspan=1, colspan=1, fig=fig)

        mwq_ax.plot(
            ws_x_crds,
            qobs_mw_mean_arr,
            alpha=line_alpha,
            color='red',
            label='obs',
            lw=line_lw)

        mwq_ax.plot(
            ws_x_crds,
            qsim_mw_mean_arr,
            alpha=line_alpha,
            color='blue',
            label='sim',
            lw=line_lw)

        mwq_ax.set_ylabel('Moving window\nmean discharge')
        mwq_ax.set_xticklabels([])

        mwq_ax.grid()
        mwq_ax.legend()

        mwq_ax.locator_params('y', nbins=4)

        diff_ratio_ax = plt.subplot2grid(
            (4, 1), (1, 0), rowspan=3, colspan=1, fig=fig)

        diff_ratio_ax.plot(
            ws_x_crds,
            qmw_diff_arr,
            alpha=line_alpha,
            color='blue',
            lw=line_lw,
            label='window diff.')

        diff_ratio_ax.axhline(
            qmw_diff_arr.mean(),
            ws_x_crds[0],
            ws_x_crds[-1],
            alpha=line_alpha,
            color='blue',
            lw=line_lw + 0.5,
            ls='-.',
            label='mean diff.')

        max_abs_diff = max(abs(qmw_diff_arr.min()), abs(qmw_diff_arr.max()))
        diff_ratio_ax.set_ylim(-max_abs_diff, +max_abs_diff)

        diff_ratio_ax.grid()
        diff_ratio_ax.legend(loc=1)

        diff_ratio_ax.set_xlabel('Time')
        diff_ratio_ax.set_ylabel('Moving window discharge difference')

        ratio_ax = diff_ratio_ax.twinx()

        ratio_ax.plot(
            ws_x_crds,
            qmw_ratio_arr,
            alpha=line_alpha,
            color='green',
            lw=line_lw,
            label='window ratio')

        ratio_ax.axhline(
            qmw_ratio_arr.mean(),
            ws_x_crds[0],
            ws_x_crds[1],
            alpha=line_alpha,
            color='green',
            lw=line_lw + 0.5,
            ls='-.',
            label='mean ratio')

        ratio_ax.set_ylabel('Moving window discharge ratio')

        ratio_ax.set_ylim(0, 2)

        ratio_ax.legend(loc=4)

        plt.suptitle(
            f'Moving window qsim by qobs ratio for window size: {ws} steps\n'
            f'Catchment: {self._cat}, Kf: {self._kf}, '
            f'Run Type: {self._run_type.upper()}, Steps: {self._n_steps}\n'
            f'Min. ratio: {qmw_ratio_arr.min():0.3f}, '
            f'Mean ratio: {qmw_ratio_arr.mean():0.3f}, '
            f'Max. ratio: {qmw_ratio_arr.max():0.3f}\n'
            f'Min. diff: {qmw_diff_arr.min():0.3f}, '
            f'Mean diff: {qmw_diff_arr.mean():0.3f}, '
            f'Max. diff: {qmw_diff_arr.max():0.3f}'
            )

        fig_name = (
            f'mwq_ratio_ws_{ws}_kf_{self._kf:02d}_{self._run_type}_'
            f'cat_{self._cat}.png')

        plt.savefig(
            os.path.join(out_dir, fig_name), bbox_inches='tight')

        plt.close()
        return

    def plot_peak_qevents(self):

        out_dir = os.path.join(self._out_dir, 'peaks_cmp')
        mkdir_hm(out_dir)

        line_alpha = 0.7
        line_lw = 1.3

        bef_steps = 10
        aft_steps = 10
        ws = 30
        steps_per_cycle = 365  # should be enough to have peaks_per_cycle peaks
        peaks_per_cycle = 2

        peaks_mask = self._get_peaks_mask(
            ws, steps_per_cycle, peaks_per_cycle)

        ppt_max = self._ppt_arr.max()

        qmax = max(self._qobs_arr.max(), self._qsim_arr.max())

        evt_idxs = np.where(peaks_mask)[0]

        for evt_idx in evt_idxs:
            fig = plt.figure(figsize=(15, 7))

            dis_ax = plt.subplot2grid(
                (4, 1), (1, 0), rowspan=3, colspan=1, fig=fig)

            ppt_ax = plt.subplot2grid(
                (4, 1), (0, 0), rowspan=1, colspan=1, fig=fig)

            bef_idx = max(0, evt_idx - bef_steps)
            aft_idx = min(evt_idx + aft_steps + 1, self._qobs_arr.shape[0])

            x_arr = np.arange(bef_idx, aft_idx)

            dis_ax.plot(
                x_arr,
                self._qsim_arr[bef_idx:aft_idx],
                alpha=line_alpha,
                color='blue',
                label='sim',
                lw=line_lw)

            dis_ax.plot(
                x_arr,
                self._qobs_arr[bef_idx:aft_idx],
                label='obs',
                color='red',
                alpha=line_alpha,
                lw=line_lw + 0.2)

            dis_ax.axvline(
                evt_idx,
                alpha=line_alpha,
                color='orange',
                label='event_step',
                lw=line_lw)

            for x, y in zip(x_arr, self._qsim_arr[bef_idx:aft_idx]):

                text = f'{self._qobs_ranks[x]}, {self._qsim_ranks[x]}'

                if y < (0.5 * qmax):
                    va = 'bottom'
                    text = '  ' + text

                else:
                    va = 'top'
                    text = text + '  '

                dis_ax.text(
                    x,
                    y,
                    text,
                    rotation=90,
                    alpha=0.8,
                    va=va,
                    size='x-small')

            dis_ax.set_xlabel('Time')
            dis_ax.set_ylabel('Discharge')

            dis_ax.legend()
            dis_ax.grid()

            dis_ax.set_ylim(0, qmax)

            ppt_ax.fill_between(
                x_arr,
                0,
                self._ppt_arr[bef_idx:aft_idx],
                label='ppt',
                alpha=line_alpha * 0.7,
                lw=line_lw + 0.2)

            ppt_ax.axvline(
                evt_idx,
                alpha=line_alpha,
                color='orange',
                lw=line_lw)

            for x, y in zip(x_arr, self._ppt_arr[bef_idx:aft_idx]):

                text = f'{self._ppt_ranks[x]}'

                if y < (0.5 * ppt_max):
                    va = 'bottom'
                    text = '  ' + text

                else:
                    va = 'top'
                    text = text + '  '

                ppt_ax.text(
                    x,
                    y,
                    text,
                    rotation=90,
                    alpha=0.8,
                    va=va,
                    size='x-small')

            ppt_ax.set_ylim(0, ppt_max)

            ppt_ax.set_ylabel('Precipitation')

            ppt_ax.legend()
            ppt_ax.grid()

            ppt_ax.set_xticklabels([])
            ppt_ax.locator_params('y', nbins=4)

            plt.suptitle(
                f'Peak discharge comparison at index: {evt_idx}\n'
                f'Catchment: {self._cat}, Kf: {self._kf}, '
                f'Run Type: {self._run_type.upper()}, Steps: {self._n_steps}'
                )

            fig_name = (
                f'peak_cmp_kf_{self._kf:02d}_{self._run_type}_'
                f'cat_{self._cat}_idx_{evt_idx}.png')

            plt.savefig(
                os.path.join(out_dir, fig_name), bbox_inches='tight')

            plt.close()
        return

    def plot_sorted_sq_diffs(self):

        out_dir = os.path.join(self._out_dir, 'sq_diffs')
        mkdir_hm(out_dir)

        line_alpha = 0.7

        sorted_q_idxs = np.argsort(self._qobs_arr)
        sorted_qobs = self._qobs_arr[sorted_q_idxs]
        sorted_qsim = self._qsim_arr[sorted_q_idxs]

        sorted_sq_diffs = (sorted_qobs - sorted_qsim) ** 2

        plt.figure(figsize=(20, 7))

        plt.plot(sorted_qobs, sorted_sq_diffs, alpha=line_alpha)

        plt.xlabel('Observed discharge')
        plt.ylabel('Squared difference')

        plt.grid()

        plt.title(
            f'Sorted squared difference of discharges\n'
            f'Catchment: {self._cat}, Kf: {self._kf}, '
            f'Run Type: {self._run_type.upper()}, Steps: {self._n_steps}'
            )

        fig_name = (
            f'sq_diffs_kf_{self._kf:02d}_{self._run_type}_'
            f'cat_{self._cat}.png')

        plt.savefig(
            os.path.join(out_dir, fig_name), bbox_inches='tight')

        plt.close()
        return

    def plot_quantile_effs(self):

        out_dir = os.path.join(self._out_dir, 'quant_effs')
        mkdir_hm(out_dir)

        line_alpha = 0.7

        n_quants = 10

        qobs_quants_masks_dict = self._get_quant_masks_dict(n_quants)

        q_quant_effs_dict = self._get_q_quant_effs_dict(qobs_quants_masks_dict)

        bar_x_crds = (
            np.arange(1., n_quants + 1) / n_quants) - (0.5 / n_quants)

        plt.figure(figsize=(15, 7))

        plt_texts = []
        for eff_ftn_lab in self._eff_ftns_dict:
            text_color = plt.plot(
                bar_x_crds,
                q_quant_effs_dict[eff_ftn_lab],
                marker='o',
                label=eff_ftn_lab,
                alpha=line_alpha)[0].get_color()

            for i in range(n_quants):
                txt_obj = plt.text(
                    bar_x_crds[i],
                    q_quant_effs_dict[eff_ftn_lab][i],
                    f'  {q_quant_effs_dict[eff_ftn_lab][i]:0.2f}',
                    va='top',
                    ha='left',
                    color=text_color,
                    alpha=line_alpha,
                    size='x-small')

                plt_texts.append(txt_obj)

        adjust_text(plt_texts)

        bar_x_crds_labs = [
            f'{bar_x_crds[i]:0.3f} - '
            f'({int(qobs_quants_masks_dict[i].sum())})'
            for i in range(n_quants)]

        plt.xticks(bar_x_crds, bar_x_crds_labs, rotation=90)

        plt.xlabel('Mean interval prob. - (N)')
        plt.ylabel('Efficiency')

        plt.grid()
        plt.legend()

        plt.title(
            f'Efficiences for {n_quants} quantiles of discharges\n'
            f'Catchment: {self._cat}, Kf: {self._kf}, '
            f'Run Type: {self._run_type.upper()}, Steps: {self._n_steps}'
            )

        fig_name = (
            f'quants_eff_kf_{self._kf:02d}_{self._run_type}_'
            f'cat_{self._cat}.png')

        plt.savefig(
            os.path.join(out_dir, fig_name), bbox_inches='tight')

        plt.close()
        return

    def plot_lorenz_curves(self):

        out_dir = os.path.join(self._out_dir, 'lorenz')
        mkdir_hm(out_dir)

        line_alpha = 0.7

        sorted_abs_diffs = np.sort(
            ((self._qobs_arr - self._qsim_arr) ** 2)).cumsum()

        cumm_sq_diff = sorted_abs_diffs[-1]

        sorted_abs_diffs = sorted_abs_diffs / cumm_sq_diff

        plt.figure(figsize=(15, 7))

        lorenz_x_vals = np.linspace(
            1., self._n_steps, self._n_steps) / (self._n_steps + 1.)

        trans_lorenz_x_vals = 1 - lorenz_x_vals

        plt.semilogx(
            trans_lorenz_x_vals,
            1 - trans_lorenz_x_vals,
            color='red',
            label=f'equal_contrib',
            alpha=line_alpha)

        plt.semilogx(
            trans_lorenz_x_vals,
            sorted_abs_diffs,
            label=f'sim',
            alpha=0.5)

        x_ticks = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1e0])
        plt.xticks(x_ticks, 1 - x_ticks)

        plt.gca().invert_xaxis()

        plt.xlabel('Rel. cumm. steps')
        plt.ylabel('Rel. cumm. sq. diff.')

        plt.legend()
        plt.grid()

        plt.title(
            f'Lorenz error contribution curves\n'
            f'Catchment: {self._cat}, Kf: {self._kf}, '
            f'Run Type: {self._run_type.upper()}, Steps: {self._n_steps}\n'
            f'Cummulative squared difference: {cumm_sq_diff:0.3f}'
            )

        fig_name = (
            f'lorenz_kf_{self._kf:02d}_{self._run_type}_'
            f'cat_{self._cat}.png')

        plt.savefig(
            os.path.join(out_dir, fig_name), bbox_inches='tight')

        plt.close()
        return

    def plot_fts(self):

        out_dir = os.path.join(self._out_dir, 'fts')
        mkdir_hm(out_dir)

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

#         freq_cov_cntrb_grad_obs = (
#             freq_cov_cntrb_obs[1:] - freq_cov_cntrb_obs[:-1]) / (
#                 freq_cov_cntrb_obs[1:])

        _obs_indiv_cntrb = (
            (ft_obs_amps * ft_obs_amps) * np.cos(ft_obs_phas - ft_obs_phas))
        _obs_indiv_cntrb /= max_cov_obs

        freq_cov_cntrb_grad_obs = (
            _obs_indiv_cntrb[1:] - _obs_indiv_cntrb[:-1]) / (
                _obs_indiv_cntrb[1:])

        freq_cov_cntrb_grad_obs[np.abs(freq_cov_cntrb_grad_obs) > 20] = 20

        ft_sim = np.fft.rfft(self._qsim_arr[:n_ft_pts])[ft_ofst_idx:]
        ft_sim_phas = np.angle(ft_sim)
        ft_sim_amps = np.abs(ft_sim)

        max_cov_sim = (
            (ft_obs_amps ** 2).sum() * (ft_sim_amps ** 2).sum()) ** 0.5

        freq_cov_cntrb_sim = np.cumsum(
            (ft_obs_amps * ft_sim_amps) * np.cos(ft_obs_phas - ft_sim_phas))
        freq_cov_cntrb_sim /= max_cov_sim

        _sim_indiv_cntrb = (
            (ft_obs_amps * ft_sim_amps) * np.cos(ft_obs_phas - ft_sim_phas))
        _sim_indiv_cntrb /= max_cov_sim

        freq_cov_cntrb_grad_sim = (
            _sim_indiv_cntrb[1:] - _sim_indiv_cntrb[:-1]) / (
                _sim_indiv_cntrb[1:])

        freq_cov_cntrb_grad_sim[np.abs(freq_cov_cntrb_grad_sim) > 20] = 20

        self._plot_fts_wvcbs(
            freq_cov_cntrb_obs,
            freq_cov_cntrb_sim,
            out_dir)

        self._plot_fts_wvcbs_grad(
            freq_cov_cntrb_grad_obs,
            freq_cov_cntrb_grad_sim,
            out_dir)

        self._plot_fts_phas_diff(ft_obs_phas, ft_sim_phas, out_dir)

        self._plot_fts_amps_diff(ft_obs_amps, ft_sim_amps, out_dir)

        self._plot_fts_amps(ft_obs_amps, ft_sim_amps, out_dir)

        self._plot_fts_abs_diff(ft_obs, ft_sim, out_dir)
        return

    def plot_emp_cops(self):

        out_dir = os.path.join(self._out_dir, 'ecops')
        mkdir_hm(out_dir)

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

        fig_name = (
            f'ecop_obs_sim_kf_{self._kf:02d}_{self._run_type}_'
            f'cat_{self._cat}.png')

        plt.savefig(
            os.path.join(out_dir, fig_name),
            bbox_inches='tight')

        plt.close()
        return

    def _plot_hi_err_qevents(
            self,
            hi_qerr_idxs,
            ppt_max,
            qmax,
            bef_steps,
            aft_steps,
            sq_diffs,
            sum_sq_diffs):

        out_dir = os.path.join(self._out_dir, 'hi_qerrs')
        mkdir_hm(out_dir)

        line_alpha = 0.7
        line_lw = 1.3

        for hi_qerr_idx in hi_qerr_idxs:
            fig = plt.figure(figsize=(15, 7))

            dis_ax = plt.subplot2grid(
                (4, 1), (1, 0), rowspan=3, colspan=1, fig=fig)

            ppt_ax = plt.subplot2grid(
                (4, 1), (0, 0), rowspan=1, colspan=1, fig=fig)

            bef_idx = max(0, hi_qerr_idx - bef_steps)
            aft_idx = min(hi_qerr_idx + aft_steps + 1, self._qobs_arr.shape[0])

            x_arr = np.arange(bef_idx, aft_idx)

            dis_ax.plot(
                x_arr,
                self._qsim_arr[bef_idx:aft_idx],
                alpha=line_alpha,
                color='blue',
                label='sim',
                lw=line_lw)

            dis_ax.plot(
                x_arr,
                self._qobs_arr[bef_idx:aft_idx],
                label='obs',
                color='red',
                alpha=line_alpha,
                lw=line_lw + 0.2)

            dis_ax.axvline(
                hi_qerr_idx,
                alpha=line_alpha,
                color='orange',
                label='event_step',
                lw=line_lw)

            for x, y in zip(x_arr, self._qsim_arr[bef_idx:aft_idx]):

                text = (
                    f'{100 * (sq_diffs[x] / sum_sq_diffs):0.3f}%, '
                    f'{self._qobs_ranks[x]}, {self._qsim_ranks[x]}')

                if y < (0.5 * qmax):
                    va = 'bottom'
                    text = '  ' + text

                else:
                    va = 'top'
                    text = text + '  '

                dis_ax.text(
                    x,
                    y,
                    text,
                    rotation=90,
                    alpha=0.8,
                    va=va,
                    size='x-small')

            dis_ax.set_xlabel('Time')
            dis_ax.set_ylabel('Discharge')

            dis_ax.legend()
            dis_ax.grid()

            dis_ax.set_ylim(0, qmax)

            ppt_ax.fill_between(
                x_arr,
                0,
                self._ppt_arr[bef_idx:aft_idx],
                label='ppt',
                alpha=line_alpha * 0.7,
                lw=line_lw + 0.2)

            ppt_ax.axvline(
                hi_qerr_idx,
                alpha=line_alpha,
                color='orange',
                lw=line_lw)

            for x, y in zip(x_arr, self._ppt_arr[bef_idx:aft_idx]):

                text = f'{self._ppt_ranks[x]}'

                if y < (0.5 * ppt_max):
                    va = 'bottom'
                    text = '  ' + text
                else:
                    va = 'top'
                    text = text + '  '

                ppt_ax.text(
                    x,
                    y,
                    text,
                    rotation=90,
                    alpha=0.8,
                    va=va,
                    size='x-small')

            ppt_ax.set_ylim(0, ppt_max)

            ppt_ax.set_ylabel('Precipitation')

            ppt_ax.legend()
            ppt_ax.grid()

            ppt_ax.set_xticklabels([])
            ppt_ax.locator_params('y', nbins=4)

            sq_diff = sq_diffs[hi_qerr_idx]
            tot_pcnt = 100 * (sq_diff / sum_sq_diffs)

            plt.suptitle(
                f'Hi error discharge comparison at index: {hi_qerr_idx}\n'
                f'Catchment: {self._cat}, Kf: {self._kf}, '
                f'Run Type: {self._run_type.upper()}, Steps: {self._n_steps}\n'
                f'Squared difference: {sq_diff:0.2f}, {tot_pcnt:0.3f}% '
                f'of the total ({sum_sq_diffs:0.2f})'
                )

            fig_name = (
                f'hi_qerr_kf_{self._kf:02d}_{self._run_type}_'
                f'cat_{self._cat}_idx_{hi_qerr_idx}.png')

            plt.savefig(
                os.path.join(out_dir, fig_name), bbox_inches='tight')

            plt.close()
        return

#     def _plot_hi_err_qevents_2d_ppt(
#             self,
#             hi_qerr_idxs,
#             ppt_max,
#             bef_steps,
#             aft_steps,
#             sq_diffs,
#             sum_sq_diffs):
#
#         out_dir = os.path.join(self._out_dir, 'hi_qerrs_ppt')
#         mkdir_hm(out_dir)
#
#         n_evt_steps = aft_steps + bef_steps
#
#         loc_rows = max(1, int(0.25 * n_evt_steps))
#         loc_cols = max(1, int(np.ceil(n_evt_steps / loc_rows)))
#
#         sca_fac = 3
#         loc_rows *= sca_fac
#         loc_cols *= sca_fac
#
#         legend_rows = 1
#         legend_cols = loc_cols
#
#         plot_shape = (loc_rows + legend_rows, loc_cols)
#
#         cen_crds = (0.5 * np.array(self._grid_shape)).astype(int)
#
#         for hi_qerr_idx in hi_qerr_idxs:
#             bef_idx = max(0, hi_qerr_idx - bef_steps)
#             aft_idx = min(hi_qerr_idx + aft_steps + 1, self._qobs_arr.shape[0])
#
#             curr_row = 0
#             curr_col = 0
#
#             sq_diff = sq_diffs[hi_qerr_idx]
#             tot_pcnt = 100 * (sq_diff / sum_sq_diffs)
#
#             plt.figure(figsize=(12, 14))
#
#             for step in range(bef_idx, aft_idx):
#                 ax = plt.subplot2grid(
#                     plot_shape,
#                     loc=(curr_row, curr_col),
#                     rowspan=sca_fac,
#                     colspan=sca_fac)
#
#                 plot_grid = np.full(self._grid_shape, np.nan)
#                 plot_grid[self._grid_rows, self._grid_cols] = (
#                     self._ppt_dist_arr[:, step])
#
#                 plot_grid = np.flipud(plot_grid)
#
#                 ps = ax.imshow(
#                     plot_grid,
#                     origin='lower',
#                     cmap=plt.get_cmap('gist_rainbow'),
#                     zorder=1,
#                     vmin=0,
#                     vmax=ppt_max)
#
#                 ax.text(
#                     *cen_crds,
#                     f'  {step}\n({self._ppt_ranks[step]:0.2f})',
#                     size='x-small')
#
#                 ax.set_ylim(0, self._grid_shape[0])
#                 ax.set_xlim(0, self._grid_shape[1])
#
#                 ax.set_xticks([])
#                 ax.set_yticks([])
#                 ax.set_xticklabels([])
#                 ax.set_yticklabels([])
#                 ax.set_axis_off()
#
#                 curr_col += sca_fac
#                 if curr_col >= loc_cols:
#                     curr_col = 0
#                     curr_row += sca_fac
#
#             cb_ax = plt.subplot2grid(
#                 plot_shape,
#                 loc=(plot_shape[0] - 1, 0),
#                 rowspan=1,
#                 colspan=legend_cols)
#
#             cb_ax.set_axis_off()
#             cb = plt.colorbar(
#                 ps,
#                 ax=cb_ax,
#                 fraction=0.9,
#                 aspect=20,
#                 orientation='horizontal')
#
#             cb.set_label('Precipitation')
#
#             plt.suptitle(
#                 f'Hi error discharge precipitation comparison at '
#                 f'index: {hi_qerr_idx}\n'
#                 f'Catchment: {self._cat}, Kf: {self._kf}, '
#                 f'Run Type: {self._run_type.upper()}, '
#                 f'Steps: {self._n_steps}\n'
#                 f'Squared difference: {sq_diff:0.2f}, {tot_pcnt:0.3f}% '
#                 f'of the total ({sum_sq_diffs:0.2f})'
#                 )
#
#             fig_name = (
#                 f'hi_qerr_ppt_kf_{self._kf:02d}_{self._run_type}_'
#                 f'cat_{self._cat}_idx_{hi_qerr_idx}.png')
#
#             plt.savefig(os.path.join(out_dir, fig_name), bbox_inches='tight')
#             plt.close()
#         return

    def _get_mw_qdiff_ratio_arrs(self, ws):

        ref_mv_mean_arr = np.zeros(self._n_steps - ws)
        sim_mv_mean_arr = ref_mv_mean_arr.copy()

        ws_xcrds = []
        for i in range(self._n_steps - ws):
            ref_mv_mean_arr[i] = self._qobs_arr[i:i + ws].mean()

            ws_xcrds.append(i + int(0.5 * ws))

        ws_xcrds = np.array(ws_xcrds)

        for i in range(self._n_steps - ws):
            sim_mv_mean_arr[i] = self._qsim_arr[i:i + ws].mean()

        diff_arr = (sim_mv_mean_arr - ref_mv_mean_arr)
        ratio_arr = (sim_mv_mean_arr / ref_mv_mean_arr)

        return ws_xcrds, ref_mv_mean_arr, sim_mv_mean_arr, diff_arr, ratio_arr

    def _get_peaks_mask(self, ws, steps_per_cycle, peaks_per_cycle):

        rising = self._qobs_arr[1:] - self._qobs_arr[:-1] > 0
        recing = self._qobs_arr[1:-1] - self._qobs_arr[2:] > 0

        peaks_mask = np.concatenate(([False], rising[:-1] & recing, [False]))

        n_steps = self._qobs_arr.shape[0]

        assert steps_per_cycle > peaks_per_cycle
        assert steps_per_cycle > ws
        assert steps_per_cycle < n_steps

        window_sums = np.full(steps_per_cycle, np.inf)
        for i in range(ws, steps_per_cycle + ws - 1):
            window_sums[i - ws] = self._qobs_arr[i - ws:i].sum()

        assert np.all(window_sums > 0)

        min_idx = int(0.5 * ws) + np.argmin(window_sums)

        if min_idx > (0.5 * steps_per_cycle):
            beg_idx = 0
            end_idx = min_idx

        else:
            beg_idx = min_idx
            end_idx = min_idx + steps_per_cycle

        assert n_steps >= end_idx - beg_idx, 'Too few steps!'

        out_mask = np.zeros(n_steps, dtype=bool)

        while (end_idx - n_steps) < 0:
            loop_mask = np.zeros(n_steps, dtype=bool)
            loop_mask[beg_idx:end_idx] = True
            loop_mask &= peaks_mask

            highest_idxs = np.argsort(
                self._qobs_arr[loop_mask])[-peaks_per_cycle:]

            out_mask[np.where(loop_mask)[0][highest_idxs]] = True

            beg_idx = end_idx
            end_idx += steps_per_cycle

        assert out_mask.sum(), 'No peaks selected!'
        return out_mask

    def _get_q_quant_effs_dict(self, qobs_quants_masks_dict):

        n_q_quants = len(qobs_quants_masks_dict)

        quant_effs_dict = {}
        for eff_ftn_key in self._eff_ftns_dict:
            eff_ftn = self._eff_ftns_dict[eff_ftn_key]

            quant_effs = []
            for i in range(n_q_quants):
                mask = qobs_quants_masks_dict[i]

                assert mask.sum() > 0

                quant_effs.append(
                    eff_ftn(self._qobs_arr[mask], self._qsim_arr[mask], 0))

            quant_effs_dict[eff_ftn_key] = quant_effs

        return quant_effs_dict

    def _get_quant_masks_dict(self, n_quants):

        interp_ftn = interp1d(
            np.sort(self._qobs_probs),
            np.sort(self._qobs_arr),
            bounds_error=False,
            fill_value=(self._qobs_arr.min(), self._qobs_arr.max()))

        quant_probs = np.linspace(0., 1., n_quants + 1, endpoint=True)

        quants = interp_ftn(quant_probs)
        quants[-1] = quants[-1] * 1.1

        masks_dict = {}
        for i in range(n_quants):
            masks_dict[i] = (
                (self._qobs_arr >= quants[i]) &
                (self._qobs_arr < quants[i + 1]))

        return masks_dict

    def _plot_fts_abs_diff(self, ft_obs, ft_sim, out_dir):

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

        fig_name = (
            f'ft_abs_diff_kf_{self._kf:02d}_{self._run_type}_'
            f'cat_{self._cat}.png')

        plt.savefig(os.path.join(out_dir, fig_name), bbox_inches='tight')
        plt.close()
        return

    def _plot_fts_amps(self, ft_obs_amps, ft_sim_amps, out_dir):

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

        fig_name = (
            f'ft_amps_kf_{self._kf:02d}_{self._run_type}_'
            f'cat_{self._cat}.png')

        plt.savefig(os.path.join(out_dir, fig_name), bbox_inches='tight')
        plt.close()
        return

    def _plot_fts_phas_diff(self, ft_obs_phas, ft_sim_phas, out_dir):

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

        fig_name = (
            f'ft_phas_diff_kf_{self._kf:02d}_{self._run_type}_'
            f'cat_{self._cat}.png')

        plt.savefig(os.path.join(out_dir, fig_name), bbox_inches='tight')
        plt.close()
        return

    def _plot_fts_amps_diff(self, ft_obs_amps, ft_sim_amps, out_dir):

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
            f'Discharge Fourier normalized amplitude difference\n'
            f'Catchment: {self._cat}, Kf: {self._kf}, '
            f'Run Type: {self._run_type.upper()}, Steps: {self._n_steps}\n'
            f'Obs. and Sim. mean: {self._qobs_mean:0.3f}, '
            f'{self._qsim_mean:0.3f}, '
            f'Obs. and Sim. variance: {self._qobs_var:0.3f}, '
            f'{self._qsim_var:0.3f}'
            )

        fig_name = (
            f'ft_amps_diff_kf_{self._kf:02d}_{self._run_type}_'
            f'cat_{self._cat}.png')

        plt.savefig(os.path.join(out_dir, fig_name), bbox_inches='tight')
        plt.close()
        return

    def _plot_fts_wvcbs(self, freq_cov_cntrb_obs, freq_cov_cntrb_sim, out_dir):

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
            f'Discharge Fourier frequency correlation contribution\n'
            f'Catchment: {self._cat}, Kf: {self._kf}, '
            f'Run Type: {self._run_type.upper()}, Steps: {self._n_steps}\n'
            f'Sim-to-Obs fourier correlation: {freq_cov_cntrb_sim[-1]:0.4f}\n'
            f'Obs. and Sim. mean: {self._qobs_mean:0.3f}, '
            f'{self._qsim_mean:0.3f}, '
            f'Obs. and Sim. variance: {self._qobs_var:0.3f}, '
            f'{self._qsim_var:0.3f}'
            )

        fig_name = (
            f'ft_full_wvcb_kf_{self._kf:02d}_{self._run_type}_'
            f'cat_{self._cat}.png')

        plt.savefig(os.path.join(out_dir, fig_name), bbox_inches='tight')
        plt.close()
        return

    def _plot_fts_wvcbs_grad(
            self,
            freq_cov_cntrb_grad_obs,
            freq_cov_cntrb_grad_sim,
            out_dir):

        line_alpha = 0.7
        line_lw = 0.9

        # wvcb gradient
        plt.figure(figsize=(15, 7))

        plt.plot(
            freq_cov_cntrb_grad_obs,
            label='obs',
            alpha=line_alpha + 0.2,
            color='red',
            lw=line_lw)

        plt.plot(
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
            f'Discharge Fourier frequency correlation contribution normalized'
            f' gradient\n'
            f'Catchment: {self._cat}, Kf: {self._kf}, '
            f'Run Type: {self._run_type.upper()}, Steps: {self._n_steps}\n'
            f'Obs. and Sim. mean: {self._qobs_mean:0.3f}, '
            f'{self._qsim_mean:0.3f}, '
            f'Obs. and Sim. variance: {self._qobs_var:0.3f}, '
            f'{self._qsim_var:0.3f}'
            )

        fig_name = (
            f'ft_full_wvcb_grad_kf_{self._kf:02d}_{self._run_type}_'
            f'cat_{self._cat}.png')

        plt.savefig(os.path.join(out_dir, fig_name), bbox_inches='tight')
        plt.close()
        return
