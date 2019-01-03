'''
Created on Jan 2, 2019

@author: Faizan-Uni
'''

import os
import sys
import traceback as tb
from fnmatch import filter

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text

from .perfs import get_fdc
from ..models import (
    hbv_loop_py,
    get_ns_cy,
    get_ln_ns_cy,
    get_pcorr_cy,
    get_kge_cy,
    tfm_opt_to_hbv_prms_py)

plt.ioff()


def plot_cat_qsims(cat_db):

    try:
        with h5py.File(cat_db, 'r') as db:
            opt_iters = [3, 6, 10, 15, None]
            long_short_break_freqs = ['A']

            cv_flag = db['data'].attrs['cv_flag']

            plot_sims_cls = PlotCatQSims(db)

            for long_short_break_freq in long_short_break_freqs:
                for opt_iter in opt_iters:
                    plot_sims_cls.set_run_sim_prms(
                        'calib', opt_iter, long_short_break_freq)

                    cv_args = plot_sims_cls.run_prm_sims()

                    if (not cv_flag) or (cv_args is None):
                        continue

                    plot_sims_cls.set_run_sim_prms(
                        'valid', opt_iter, long_short_break_freq)

                    plot_sims_cls.run_prm_sims(*cv_args)

    except Exception as msg:
        print('Error in plot_cat_qsims:', msg)

        * _, exc_traceback = sys.exc_info()
        tb.print_tb(exc_traceback, limit=None, file=sys.stdout)
        return
    return


class PlotCatQSims:

    '''Just to have less mess'''

    def __init__(self, db):

        self.db = db

        self.kfolds = db['data'].attrs['kfolds']
        self.cat = db.attrs['cat']

        self.conv_ratio = db['data'].attrs['conv_ratio']

        self.area_arr = db['data/area_arr'][...]
        self.n_cells = self.area_arr.shape[0]

        self.calib_db = db['calib']

        self.opt_schm = db['cdata/opt_schm_vars_dict'].attrs['opt_schm']

        self.off_idx = db['data'].attrs['off_idx']

        self.f_var_infos = db['cdata/aux_var_infos'][...]
        self.prms_idxs = db['cdata/use_prms_idxs'][...]
        self.f_vars = db['cdata/aux_vars'][...]
        self.prms_flags = db['cdata/all_prms_flags'][...]
        self.bds_arr = db['cdata/bds_arr'][...]

        self.prms_syms = db['data/all_prms_labs'][...]

        bds_db = db['data/bds_dict']
        bds_dict = {key: bds_db[key][...] for key in bds_db}

        self.hbv_min_max_bds = np.array(
            [bds_dict[prm_sym + '_bds'] for prm_sym in self.prms_syms])

        self.use_obs_flow_flag = db['data'].attrs['use_obs_flow_flag']

        out_dir = db['data'].attrs['main']

        qsims_dir = os.path.join(out_dir, '12_discharge_sims')

        if not os.path.exists(qsims_dir):
            try:
                os.mkdir(qsims_dir)

            except FileExistsError:
                pass

        self.qsims_dir = qsims_dir

        self._set_run_prms_flag = False
        self._set_lo_hi_corr_idxs_flag = False

        self.kf_corr_args_dict = {}

        self._mkdirs()
        return

    def _mkdirs(self):

        out_dirs = [
            'ft_corrs',
            'perf_cdfs',
            'ns_vs_pcorr',
            'ns_vs_clrs',
            'ann_sims',
            'fdcs',
            'prms']

        for out_dir in out_dirs:
            out_dir_path = os.path.join(self.qsims_dir, out_dir)
            if not os.path.exists(out_dir_path):
                os.mkdir(out_dir_path)

            setattr(self, out_dir + '_dir', out_dir_path)
        return

    def _get_data(self, sim_lab):

        all_kfs_dict = {}
        for i in range(1, self.kfolds + 1):
            cd_db = self.db[f'{sim_lab}/kf_{i:02d}']
            kf_dict = {key: cd_db[key][...] for key in cd_db}
            kf_dict['use_obs_flow_flag'] = self.use_obs_flow_flag

            all_kfs_dict[i] = kf_dict

        input_labs = ['tem_arr', 'ppt_arr', 'pet_arr', 'qact_arr']

        if 'extra_us_inflow' in kf_dict:
            input_labs.append('extra_us_inflow')
            self.extra_flow_flag = True

        else:
            self.extra_flow_flag = False

        input_arrs = []

        del kf_dict

        for input_lab in input_labs:
            kf_arrs = []
            for i in all_kfs_dict:
                kf_arrs.append(all_kfs_dict[i][input_lab])

                del all_kfs_dict[i][input_lab]

            axis = kf_arrs[0].ndim - 1
            input_arrs.append(np.concatenate(kf_arrs, axis=axis))

        return input_arrs, all_kfs_dict

    def _get_break_idx(self, long_short_break_freq, n_steps):

        assert isinstance(long_short_break_freq, str)

        if long_short_break_freq[-1] == 'M':

            if len(long_short_break_freq) > 1:
                inc = int(long_short_break_freq[:-1])
                assert inc > 0

            else:
                inc = 1

            break_idx = int(n_steps // (inc * 30))

        elif ((long_short_break_freq[-1] == 'A') or
              (long_short_break_freq[-1] == 'Y')):

            if len(long_short_break_freq) > 1:
                inc = int(long_short_break_freq[:-1])
                assert inc > 0

            else:
                inc = 1

            break_idx = int(n_steps // (inc * 365))

        elif long_short_break_freq[-1] == 'W':
            if len(long_short_break_freq) > 1:
                inc = int(long_short_break_freq[:-1])
                assert inc > 0

            else:
                inc = 1

            break_idx = int(n_steps // (inc * 7))

        elif long_short_break_freq[-1] == 'D':
            if len(long_short_break_freq) > 1:
                inc = int(long_short_break_freq[:-1])
                assert inc > 0

            else:
                inc = 1

            break_idx = int(n_steps // (inc))

        else:
            raise ValueError(
                f'Undefined long_short_break_freq: {long_short_break_freq}!')

        return break_idx

    def _plot_cumm_pcorrs(self, kf_str):
        # FT corrs
        fig_size = (18, 10)

        full_fig = plt.figure(figsize=fig_size)
        part_fig = plt.figure(figsize=fig_size)

        sim_plot_alpha = 0.01  # max(0.01, 5 / (n_sims / 2))

#         corr_diffs = 1 - (
#             fin_cumm_rho_arrs[:, break_idx] / fin_cumm_rho_arrs[:, -1])

        corr_diffs = (
            self.fin_cumm_rho_arrs[:, -1] -
            self.fin_cumm_rho_arrs[:, self.break_idx])

        corr_diffs /= self.fin_cumm_rho_arrs[:, -1]

        min_corr_diff = corr_diffs.min()
        min_max_corr_diff = corr_diffs.max() - min_corr_diff

        # For cases with only one vector
        if self.fin_cumm_rho_arrs.shape[0] == 1:
            min_max_corr_diff = 1

        assert min_max_corr_diff > 0, (
            'min_max_corr_diff cannot be zero or less!')

        for i in range(self.n_prm_vecs):
            fin_cumm_rho_arr = self.fin_cumm_rho_arrs[i]

            if self.sim_lab == 'calib':
                clr_val = (
                    corr_diffs[i] - min_corr_diff) / min_max_corr_diff

                self.clr_vals.append(clr_val)

                self.sim_perfs_df.loc[i, 'clrs'] = clr_val

            else:
                clr_val = self.clr_vals[i]

            assert 0 <= clr_val <= 1, (i, clr_val)

            sim_clr = self.cmap(clr_val)

            plt.figure(full_fig.number)
            plt.plot(
                fin_cumm_rho_arr,
                alpha=sim_plot_alpha,
                color=sim_clr)

            plt.figure(part_fig.number)
            plt.plot(
                fin_cumm_rho_arr[:self.part_freq_idx],
                alpha=sim_plot_alpha,
                color=sim_clr)

        indiv_cov_arr = (self.use_obs_amps * self.use_obs_amps) * (
            np.cos(self.use_obs_phis - self.use_obs_phis))

        tot_cov = indiv_cov_arr.sum()

        cumm_rho_arr = indiv_cov_arr.cumsum() / tot_cov

        ft_corr = indiv_cov_arr.sum() / (
            ((self.use_obs_amps ** 2).sum() *
             (self.use_obs_amps ** 2).sum()) ** 0.5)

        plt.figure(full_fig.number)
        plt.plot(
            (cumm_rho_arr * ft_corr),
            alpha=0.5,
            color='r',
            label='Obs.')

        plt.axvline(self.break_idx, color='orange', label='Div. Freq.')

        plt.title(
            f'Cummulative correlation contribution per fourier frequency\n'
            f'n_sims: {self.n_prm_vecs}, n_steps: {self.n_steps}\n'
            f'Long-short dividing freq.: {self.long_short_break_freq}')

        plt.xlabel('Freq. no.')
        plt.ylabel('Cumm. corr. (-)')

        plt.grid()
        plt.legend()

        plt.ylim(0, 1)

        cb = plt.colorbar(self.cmap_mappable)
        cb.set_ticks([0, 1])
        cb.set_ticklabels(['Hi. long', 'Lo. Long'])

        out_fig_name = (
            f'ft_corrs_cat_{self.cat}_{self.sim_lab}_{kf_str}_'
            f'{self.opt_iter_lab}_all_freqs.png')

        plt.savefig(
            os.path.join(self.ft_corrs_dir, out_fig_name),
            bbox_inches='tight')

        plt.close()

        plt.figure(part_fig.number)

        plt.plot(
            (cumm_rho_arr * ft_corr)[:self.part_freq_idx],
            alpha=0.7,
            color='r',
            label='Obs.')

        plt.axvline(self.break_idx, color='orange', label='Div. Freq.')

        plt.title(
            f'Correlation contribution per fourier frequency\n'
            f'n_sims: {self.n_prm_vecs}, n_steps: {self.n_steps}\n'
            f'Long-short dividing freq.: {self.long_short_break_freq}')

        plt.xlabel('Frequency no.')
        plt.ylabel('Contribution (-)')

        plt.grid()
        plt.legend()

        plt.ylim(0, 1)

        cb = plt.colorbar(self.cmap_mappable)
        cb.set_ticks([0, 1])
        cb.set_ticklabels(['Hi. long', 'Lo. Long'])

        out_fig_name = (
            f'ft_corrs_cat_{self.cat}_{self.sim_lab}_{kf_str}_'
            f'{self.opt_iter_lab}_first_{self.part_freq_idx}_freqs.png')

        plt.savefig(
            os.path.join(self.ft_corrs_dir, out_fig_name),
            bbox_inches='tight')

        plt.close()
        return

    def _get_lo_hi_corr_idxs(self, kf):

        assert not self._set_lo_hi_corr_idxs_flag

        if self.sim_lab == 'calib':
            mean_ns = self.sim_perfs_df['ns'].mean()

            eff_ftn = 'ns'
            eff_ftn_val_tol = 0.01

            n_hi_vals = 3
            n_lo_vals = n_hi_vals
            n_vals = n_hi_vals + n_lo_vals

            assert self.sim_perfs_df[eff_ftn].shape[0] >= (
                n_hi_vals + n_lo_vals)

            acc_flags_1 = (
                (self.sim_perfs_df[eff_ftn] - mean_ns).abs().rank() <= (
                    n_vals)).values

            acc_flags_2 = (
                (self.sim_perfs_df[eff_ftn].values >= (
                    mean_ns - (0.5 * eff_ftn_val_tol))) &

                (self.sim_perfs_df[eff_ftn].values <= (
                    mean_ns + (0.5 * eff_ftn_val_tol))))

            if acc_flags_1.sum() > acc_flags_2.sum():
                acc_flags_ser = acc_flags_1

            else:
                acc_flags_ser = acc_flags_2

            print(f'Selected {int(acc_flags_ser.sum())} parameters!')

            self.lc_idxs = (
                list(self.sim_perfs_df.loc[
                    acc_flags_ser, 'clrs'].nsmallest(n_hi_vals).index) +

                list(self.sim_perfs_df.loc[
                    acc_flags_ser, 'clrs'].nlargest(n_lo_vals).index))

            self.lc_clrs = [
                'blue', 'green', 'purple', 'orange', 'cyan', 'pink'][:n_vals]

            self.lc_labs = (
                [f'Hi. Long ({i})' for i in range(n_hi_vals)] +
                [f'Lo. Long ({i})' for i in range(n_lo_vals)])

            self.kf_corr_args_dict[kf] = (
                self.lc_idxs, self.lc_clrs, self.lc_labs)

        else:
            (self.lc_idxs,
             self.lc_clrs,
             self.lc_labs) = self.kf_corr_args_dict[kf]

        self.n_lcs = len(self.lc_idxs)

        assert self.n_lcs > 0

        assert self.n_lcs == len(self.lc_clrs) == len(self.lc_labs)

        self._set_lo_hi_corr_idxs_flag = True
        return

    def _plot_prm_vecs(self, kf, all_hbv_prms):

        assert self._set_lo_hi_corr_idxs_flag

        plt.figure(figsize=(18, 10))
        tick_font_size = 10

        plot_range = list(range(all_hbv_prms.shape[1]))

        norm_hbv_prms_list = []
        for i in range(self.n_lcs):
            norm_hbv_prms = (
                (all_hbv_prms[self.lc_idxs[i]] - self.hbv_min_max_bds[:, 0]) /
                (self.hbv_min_max_bds[:, 1] - self.hbv_min_max_bds[:, 0]))

            norm_hbv_prms_list.append(norm_hbv_prms)

            plt.plot(
                plot_range,
                norm_hbv_prms,
                color=self.lc_clrs[i],
                label=self.lc_labs[i],
                alpha=0.5)

        ptexts = []
        for i in range(self.n_lcs):
            hbv_prms = all_hbv_prms[self.lc_idxs[i]]
            for j in range(hbv_prms.shape[0]):
                ptext = plt.text(
                    plot_range[j],
                    norm_hbv_prms_list[i][j],
                    f'{hbv_prms[j]:0.4f}'.rstrip('0'),
                    color=self.lc_clrs[i])

                ptexts.append(ptext)

        adjust_text(ptexts, only_move={'points': 'y', 'text': 'y'})

        plt.ylim(0., 1.)
        plt.ylabel('Normalized value')

        plt.ylabel('HBV parameter')
        plt.xlim(-0.5, len(plot_range) - 0.5)
        plt.xticks(plot_range, self.prms_syms, rotation=60)

        plt.grid()
        plt.legend()

        title_str = f'Distributed HBV parameters for cat: {self.cat}'

        plt.suptitle(title_str, size=tick_font_size + 10)
        plt.subplots_adjust(hspace=0.15)

        plt.savefig(
            os.path.join(
                self.prms_dir,
                f'cat_{self.cat}_kf_{kf:02d}_{self.sim_lab}_'
                f'{self.opt_iter_lab}_prm_vecs.png'),
            bbox_inches='tight')

        plt.close()
        return

    def _plot_obj_cdfs(self, obj_ftns_dict, kf_str):

        # obj_vals
        probs = (np.arange(1.0, self.n_prm_vecs + 1) / (self.n_prm_vecs + 1))

        clr_vals_arr = np.array(self.clr_vals)
        for obj_key in obj_ftns_dict:
            obj_vals = np.array(obj_ftns_dict[obj_key][2])

            # pre_obj_vals
            __, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

            ax1.set_title(
                f'{obj_ftns_dict[obj_key][1]} distribution\n'
                f'Min. obj.: {obj_vals.min():0.4f}, '
                f'Max. obj.: {obj_vals.max():0.4f}\n'
                f'Long-short dividing freq.: {self.long_short_break_freq}')

            obj_vals_sort_idxs = np.argsort(obj_vals)
            ax1.plot(
                obj_vals[obj_vals_sort_idxs],
                probs,
                color='C0',
                alpha=0.1)

            ax1.scatter(
                obj_vals[obj_vals_sort_idxs],
                probs,
                marker='o',
                c=clr_vals_arr[obj_vals_sort_idxs],
                alpha=0.2)

            ax1.set_ylabel('Non-exceedence Probability (-)')
            ax1.grid()

            cb = plt.colorbar(
                self.cmap_mappable,
                ax=ax1,
                fraction=0.05,
                aspect=20,
                orientation='vertical')

            cb.set_ticks([0, 1])
            cb.set_ticklabels(['Hi. long', 'Lo. Long'])

            ax2.hist(obj_vals, bins=20)
            ax2.set_xlabel(f'{obj_ftns_dict[obj_key][1]} (-)')
            ax2.set_ylabel('Frequency (-)')

            out_obj_fig = (
                f'hbv_perf_cdf_{self.cat}_{self.sim_lab}_{kf_str}_'
                f'{obj_key}_{self.opt_iter_lab}_opt_qsims.png')

            plt.savefig(
                os.path.join(self.perf_cdfs_dir, out_obj_fig),
                bbox_inches='tight')
            plt.close()
        return

    def _plot_effs(self, label, col, out_dir):

        assert self._set_lo_hi_corr_idxs_flag

        plt.figure(figsize=(10, 10))

        plt.title(f'NS. vs. {label} (n={self.sim_perfs_df.shape[0]})')

        plt.scatter(
            self.sim_perfs_df['ns'].values,
            self.sim_perfs_df[col].values,
            c=self.sim_perfs_df['clrs'].values,
            alpha=0.05)

        plt.scatter(
            self.sim_perfs_df['ns'].values.mean(),
            self.sim_perfs_df[col].values.mean(),
            c='k',
            alpha=0.5,
            marker='X',
            label='Mean')

        for i in range(self.n_lcs):
            plt.scatter(
                self.sim_perfs_df.loc[self.lc_idxs[i], 'ns'],
                self.sim_perfs_df.loc[self.lc_idxs[i], col],
                c=self.lc_clrs[i],
                alpha=0.5,
                label=self.lc_labs[i])

        plt.xlabel('NS.')
        plt.ylabel(label)

        plt.grid()
        plt.legend()

        out_fig_name = (
            f'{self.cat}_eff_scatter_{self.sim_lab}_{self.opt_iter_lab}.png')

        plt.savefig(
            os.path.join(out_dir, out_fig_name),
            bbox_inches='tight')

        plt.close()
        return

    def _get_resamp_perfs(self, ref_arr, sim_arr):

        resamp_freqs = [7, 30, 91, 365]

        n_steps = ref_arr.shape[0]

        effs = []

        for resamp_freq in resamp_freqs:
            extra_steps = n_steps % resamp_freq

            ref_rshp = ref_arr[extra_steps:].reshape(
                -1, resamp_freq).mean(axis=1)

            sim_rshp = sim_arr[extra_steps:].reshape(
                -1, resamp_freq).mean(axis=1)

            ns = get_ns_cy(
                ref_rshp, sim_rshp, self.off_idx // resamp_freq)

            ln_ns = get_ln_ns_cy(
                ref_rshp, sim_rshp, self.off_idx // resamp_freq)

            kge = get_kge_cy(
                ref_rshp, sim_rshp, self.off_idx // resamp_freq)

            effs.append((resamp_freq, ns, ln_ns, kge))

        return effs

    def _plot_best_hi_lo_freq_sims(self, out_df, kf):

        assert self._set_lo_hi_corr_idxs_flag

        patt_str = f'kf_{kf:02d}*'

        sel_cols = filter(out_df.columns, patt_str)
        assert sel_cols

        corr_sims = []

        for i in range(self.n_lcs):
            corr_sims.append(out_df.loc[
                :, f'kf_{kf:02d}_sim_{self.lc_idxs[i]:04d}'].values)

        qact_arr = out_df.iloc[:, 0].values

        steps_per_plot = 365

        effs = []
        for corr_sim in corr_sims:
            effs.append(self._get_resamp_perfs(
                qact_arr.astype(np.float64), corr_sim.astype(np.float64)))

        for i in range(self.n_lcs):
            effs[i] = [(
            1,
            self.sim_perfs_df.loc[self.lc_idxs[i], 'ns'],
            self.sim_perfs_df.loc[self.lc_idxs[i], 'ln_ns'],
            self.sim_perfs_df.loc[self.lc_idxs[i], 'kge'])
            ] + effs[i]

        pre_idx = 0
        lst_idx = steps_per_plot

        corr_tit_str = ''

        eff_ftn_labs = ['NS', 'Ln_NS', 'KGE']  # corresponds to effs
        n_resamp_freqs = len(effs[0])

        for k, eff_ftn_lab in enumerate(eff_ftn_labs):
            for i in range(self.n_lcs):
                corr_tit_str += f'{self.lc_labs[i]} {eff_ftn_lab}: '.ljust(20)

                for j in range(n_resamp_freqs):
                    corr_tit_str += (
                        f'({effs[i][j][0]:3d}: {effs[i][j][k + 1]:+7.6f}), '
                        ).ljust(16)

                corr_tit_str = corr_tit_str[:-2] + '\n'

#         corr_tit_str = corr_tit_str[:-1]

        perf_file_name = (
            f'cat_{self.cat}_{self.sim_lab}_{self.opt_iter_lab}_'
            f'temporal_perfs_{pre_idx}_{lst_idx}.txt')

        with open(
            os.path.join(self.ann_sims_dir, perf_file_name), 'w') as hdl:

            hdl.write(corr_tit_str)

        while lst_idx < self.n_steps:
            x_arr = np.arange(pre_idx, lst_idx)

            plt.figure(figsize=(20, 7))

            plt.plot(
                x_arr,
                qact_arr[pre_idx:lst_idx],
                label='Obs.',
                color='red',
                alpha=0.8)

            for i in range(self.n_lcs):
                plt.plot(
                    x_arr,
                    corr_sims[i][pre_idx:lst_idx],
                    label=self.lc_labs[i],
                    color=self.lc_clrs[i],
                    alpha=0.5)

            plt.grid()
            plt.legend()

            plt.xlabel('Time')
            plt.ylabel('Discharge ($m^3/s$)')

            title_str = (
                f'Observed vs. high and low long term correlation '
                f'simulations '
                f'(pre_idx: {pre_idx}, lst_idx: {lst_idx})\n')

            # title_str += corr_tit_str

            plt.title(title_str)

            fig_name = (
                f'cat_{self.cat}_{self.sim_lab}_{self.opt_iter_lab}_'
                f'qsims_{pre_idx}_{lst_idx}.png')

            plt.savefig(
                os.path.join(self.ann_sims_dir, fig_name),
                bbox_inches='tight')

            plt.close()

            pre_idx = lst_idx
            lst_idx += steps_per_plot
        return

    def _plot_fdc(self, out_df, kf):

        assert self._set_lo_hi_corr_idxs_flag

        patt_str = f'kf_{kf:02d}*'

        sel_cols = filter(out_df.columns, patt_str)
        assert sel_cols

        qact_probs, qact_vals = get_fdc(out_df.iloc[self.off_idx:, 0])

        probs_vals = []
        for i in range(self.n_lcs):
            sim_col = f'kf_{kf:02d}_sim_{self.lc_idxs[i]:04d}'

            sim_ser = out_df.loc[:, sim_col].iloc[self.off_idx:]

            probs_vals.append(get_fdc(sim_ser))

        fig_size = (18, 10)
        plt.figure(figsize=fig_size)

        plt.semilogx(
            qact_vals,
            qact_probs,
            alpha=0.8,
            color='red',
            label='Obs.')

        for i, (qsim_probs, qsim_vals) in enumerate(probs_vals):
            plt.semilogx(
                qsim_vals,
                qsim_probs,
                alpha=0.5,
                color=self.lc_clrs[i],
                label=self.lc_labs[i])

        plt.grid()
        plt.legend()

        plt.title(
            f'Flow duration curve comparison for '
            f'actual and simulated flows for the catchemnt {self.cat}')

        plt.xlabel('Exceedence Probability (-)')
        plt.ylabel('Discharge ($m^3/s$)')

        out_path = os.path.join(
            self.fdcs_dir,
            f'cat_{self.cat}_{self.sim_lab}_{self.opt_iter_lab}_'
            f'kf_{kf:02d}_fdc_compare.png')

        plt.savefig(out_path, bbox_inches='tight')

        plt.close()
        return

    def set_run_sim_prms(self, sim_lab, opt_iter, long_short_break_freq):

        assert not self._set_run_prms_flag

        self.sim_lab = sim_lab
        self.opt_iter = opt_iter
        self.long_short_break_freq = long_short_break_freq

        self._set_run_prms_flag = True
        return

    def run_prm_sims(self, *args):

        assert self._set_run_prms_flag

        if self.opt_iter is None:
            self.opt_iter_lab = 'final'

        else:
            assert (self.opt_iter >= 0) and isinstance(self.opt_iter, int)

            n_max_opt_iters = self.calib_db['kf_01/iter_prm_vecs'].shape[0]
            if (n_max_opt_iters - 1) < self.opt_iter:
                raise ValueError(
                    f'Only {n_max_opt_iters} iterations available!')

            self.opt_iter_lab = f'{self.opt_iter}'

        plot_qsims_flag = False

        if self.sim_lab == 'calib':
            clr_vals_list = []

        else:
            clr_vals_list = args[0]

        sim_opt_data = self._get_data(self.sim_lab)

        tem_arr, ppt_arr, pet_arr, qact_arr = sim_opt_data[0][:4]

        if self.extra_flow_flag:
            us_inflow_arr = sim_opt_data[0][4]

        self.n_steps = qact_arr.shape[0]

        if (self.n_steps % 2):
            self.n_steps -= 1

        self.break_idx = self._get_break_idx(
            self.long_short_break_freq, self.n_steps)

        self.part_freq_idx = max(int(0.1 * self.n_steps // 2), 100)
        assert self.part_freq_idx > 0

        obs_arr = qact_arr

        obs_ft = np.fft.fft(obs_arr)

        self.use_obs_phis = np.angle(obs_ft[1: (self.n_steps // 2)])
        self.use_obs_amps = np.abs(obs_ft[1: (self.n_steps // 2)])

        kfs_dict = sim_opt_data[1]

        out_df = pd.DataFrame(
            data=qact_arr, columns=['obs'], dtype=np.float32)

        self.cmap = plt.get_cmap('winter')
        self.cmap_mappable = plt.cm.ScalarMappable(cmap=self.cmap)
        self.cmap_mappable.set_array([])

        for k in range(1, self.kfolds + 1):
            kf_str = f'kf_{k:02d}'

            if self.opt_schm == 'DE':
                prm_vecs = self.calib_db[kf_str + '/opt_prms'][...][None, :]

            else:
                if self.opt_iter is None:
                    prm_vecs = self.calib_db[kf_str + '/prm_vecs'][...]

                else:
                    iter_prm_vecs = self.calib_db[
                        kf_str + '/iter_prm_vecs'][...]

                    prm_vecs = iter_prm_vecs[self.opt_iter, :, :].copy('c')

            self.n_prm_vecs = prm_vecs.shape[0]
            print(f'Plotting {self.n_prm_vecs} sims only!')

            alpha = 0.01

            obj_ftns_dict = {}

            obj_ftns_dict['ns'] = [get_ns_cy, 'NS', []]

            obj_ftns_dict['ln_ns'] = [get_ln_ns_cy, 'Ln_NS', []]

            obj_ftns_dict['kge'] = [get_kge_cy, 'KGE', []]

            obj_ftns_dict['pcorr'] = [get_pcorr_cy, 'PCorr.', []]

            if plot_qsims_flag:
                plt.figure(figsize=(17, 8))

            self.fin_cumm_rho_arrs = []

            self.sim_perfs_df = pd.DataFrame(
                index=np.arange(self.n_prm_vecs),
                columns=list(obj_ftns_dict.keys()) + ['clrs'],
                dtype=np.float64)

            if self.sim_lab == 'calib':
                self.clr_vals = []

            else:
                self.clr_vals = clr_vals_list[k - 1]

                self.sim_perfs_df['clrs'][:] = self.clr_vals

            all_hbv_prms = []
            for i in range(self.n_prm_vecs):
                opt_prms = prm_vecs[i]

                hbv_prms = tfm_opt_to_hbv_prms_py(
                    self.prms_flags,
                    self.f_var_infos,
                    self.prms_idxs,
                    self.f_vars,
                    opt_prms,
                    self.bds_arr,
                    self.n_cells)

                all_hbv_prms.append(hbv_prms.mean(axis=0))

                outputs_dict = hbv_loop_py(
                    tem_arr,
                    ppt_arr,
                    pet_arr,
                    hbv_prms,
                    kfs_dict[k]['ini_arr'],
                    self.area_arr,
                    self.conv_ratio)

                assert outputs_dict['loop_ret'] == 0.0

                qsim_arr = outputs_dict['qsim_arr']

                if self.extra_flow_flag:
                    qsim_arr = qsim_arr + us_inflow_arr

                if plot_qsims_flag:
                    plt.plot(qsim_arr, color='k', alpha=alpha, lw=0.5)

                out_df[f'kf_{k:02d}_sim_{i:04d}'] = qsim_arr

                for obj_key in obj_ftns_dict:
                    obj_val = obj_ftns_dict[obj_key][0](
                        qact_arr, qsim_arr, self.off_idx)

                    obj_ftns_dict[obj_key][2].append(obj_val)

                    self.sim_perfs_df.loc[i, obj_key] = obj_val

                sim_arr = qsim_arr

                sim_ft = np.fft.fft(sim_arr)

                use_sim_phis = np.angle(sim_ft[1: (self.n_steps // 2)])

                use_sim_amps = np.abs(sim_ft[1: (self.n_steps // 2)])

                indiv_cov_arr = (self.use_obs_amps * use_sim_amps) * (
                    np.cos(self.use_obs_phis - use_sim_phis))

                tot_cov = indiv_cov_arr.sum()

                cumm_rho_arr = indiv_cov_arr.cumsum() / tot_cov

                ft_corr = indiv_cov_arr.sum() / (
                    ((self.use_obs_amps ** 2).sum() *
                     (use_sim_amps ** 2).sum()) ** 0.5)

                fin_cumm_rho_arr = cumm_rho_arr * ft_corr
                self.fin_cumm_rho_arrs.append(fin_cumm_rho_arr)

            self.fin_cumm_rho_arrs = np.array(self.fin_cumm_rho_arrs)
            all_hbv_prms = np.array(all_hbv_prms)

            if plot_qsims_flag:
                plt.plot(qact_arr, color='r', alpha=0.7, lw=0.5)

                plt.xlabel('Step no.')
                plt.ylabel('Discharge')

                plt.grid()

                plt.title(
                    f'Discharge simulation using {self.opt_schm} parameters '
                    f'(n={self.n_prm_vecs}) for the catchment: '
                    f'{self.cat}, kf: {k:02d}\n'
                    f'Optimization iteration: {self.opt_iter_lab}\n'
                    f'Long-short dividing freq.: {self.long_short_break_freq}')

                out_fig_name = (
                    f'{kf_str}_{self.cat}_{self.sim_lab}_'
                    f'{self.opt_iter_lab}_opt_qsims.png')

                plt.savefig(
                    os.path.join(self.qsims_dir, out_fig_name),
                    bbox_inches='tight',
                    dpi=150)

                plt.close()

            self._plot_cumm_pcorrs(kf_str)

            self._get_lo_hi_corr_idxs(k)

            if self.n_cells > 1:
                # this has happened above
                print(
                    f'Model is distributed. Plotting mean parameters only!')

            self._plot_prm_vecs(k, all_hbv_prms)

            out_perf_df_name = (
                f'perfs_clrs_cat_{self.cat}_{self.sim_lab}_{kf_str}_'
                f'{self.opt_iter_lab}.csv')

            self.sim_perfs_df.to_csv(
                os.path.join(self.perf_cdfs_dir, out_perf_df_name),
                float_format='%0.8f',
                sep=';')

            if self.sim_lab == 'calib':
                clr_vals_list.append(self.clr_vals)

            self._plot_obj_cdfs(obj_ftns_dict, kf_str)

            self._plot_effs('PCorr.', 'pcorr', self.ns_vs_pcorr_dir)

            self._plot_effs('Clrs.', 'clrs', self.ns_vs_clrs_dir)

            self._plot_best_hi_lo_freq_sims(out_df, k)

            self._plot_fdc(out_df, k)

            self._set_lo_hi_corr_idxs_flag = False

        out_df.to_csv(
            os.path.join(
                self.qsims_dir,
                f'cat_{self.cat}_{self.sim_lab}_{self.opt_iter_lab}'
                f'_opt_qsims.csv'),
            float_format='%0.3f',
            index=False,
            sep=';')

        self._set_run_prms_flag = False
        self._set_lo_hi_corr_idxs_flag = False
        return [clr_vals_list]
