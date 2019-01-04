# -*- coding: utf-8 -*-
'''
Created on Oct 12, 2017

@author: Faizan Anwar, IWS Uni-Stuttgart
'''

import os
import shelve
from glob import  glob
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from parse import search
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pathos.multiprocessing import ProcessPool

from ..models import (
    hbv_loop_py,
    tfm_opt_to_hbv_prms_py,
    get_ns_cy,
    get_ln_ns_cy,
    get_kge_cy,
    get_pcorr_cy,
    get_ns_var_res_cy,
    get_ln_ns_var_res_cy,
    get_ns_prt_cy,
    get_ln_ns_prt_cy,
    get_kge_prt_cy,
    get_pcorr_prt_cy)

plt.ioff()


def plot_cat_discharge_errors(plot_args):

    (cat_db,) = plot_args

    with h5py.File(cat_db, 'r') as db:
        main_out_dir = db['data'].attrs['main']
        off_idx = db['data'].attrs['off_idx']
        kfolds = db['data'].attrs['kfolds']

    qsims_dir = os.path.join(main_out_dir, '12_discharge_sims')
    errors_dir = os.path.join(qsims_dir , 'errors')

    if not os.path.exists(qsims_dir):
        try:
            os.mkdir(qsims_dir)

        except FileExistsError:
            pass

    qsim_files_patt = 'cat_[0-9]*_*_[0-9]*_freq_*_opt_qsims.csv'
    qsim_files = glob(os.path.join(qsims_dir, qsim_files_patt))

    if not qsim_files:
        print('Simulating discharges...')

        from .eval import plot_cat_qsims

        plot_cat_qsims(cat_db)

        qsim_files = glob(os.path.join(qsims_dir, qsim_files_patt))

    assert qsim_files

    if not os.path.exists(errors_dir):
        try:
            os.mkdir(errors_dir)

        except FileExistsError:
            pass

    parse_patt = 'cat_{:d}_{}_{:d}_freq_{}_opt_qsims.csv'

    n_q_quants = 10

    for qsim_file in qsim_files:
        cat, calib_valid_lab, opt_iter, freq = search(
            parse_patt, os.path.basename(qsim_file))

        qsims_df = pd.read_csv(qsim_file, sep=';').iloc[off_idx:, :]

        qobs_quants_masks_dict = get_quant_masks_dict(
            qsims_df.iloc[:, 0], n_q_quants)

        qobs_arr = qsims_df.iloc[:, 0].values

        peaks_mask = get_peaks_mask(qobs_arr)

        n_vals = qobs_arr.shape[0]
        lorenz_x_vals = np.linspace(1., n_vals + 1., n_vals) / (n_vals + 1.)

        best_prm_file_name = (
            f'best_prms_idxs_cat_{cat}_{opt_iter}_freq_{freq}.csv')

        best_prm_idxs_df = pd.read_csv(
            os.path.join(qsims_dir, best_prm_file_name), sep=';', index_col=0)

        lc_labs = best_prm_idxs_df.columns

        eff_ftns_dict = {
            'ns': get_ns_cy,
            'ln_ns': get_ln_ns_cy,
            'kge': get_kge_cy,
            'pcorr': get_pcorr_cy,
            }

        for kf in range(1, kfolds + 1):
            best_prms_lab = f'kf_{kf:02d}_idxs'

            lc_idxs = list(map(int, best_prm_idxs_df.loc[best_prms_lab, :].values))

            q_quant_effs_dicts = []
            lorenz_y_vals_list = []
            peak_effs = []

            for lc_idx in lc_idxs:
                qsim_lab = f'kf_{kf:02d}_sim_{lc_idx:04d}'
                qsim_arr = qsims_df.loc[:, qsim_lab].values

                q_quant_effs_dict = get_q_quant_effs_dict(
                    qobs_arr, qsim_arr, qobs_quants_masks_dict, eff_ftns_dict)

                q_quant_effs_dicts.append(q_quant_effs_dict)

                lorenz_y_vals = get_lorenz_arr(qobs_arr, qsim_arr)
                lorenz_y_vals_list.append(lorenz_y_vals)

                peak_effs_dict = get_peaks_effs_dict(
                    qobs_arr, qsim_arr, peaks_mask, eff_ftns_dict)

                peak_effs.append(peak_effs_dict)

            for eff_ftn_lab in eff_ftns_dict:
                out_quants_fig_name = (
                    f'quant_effs_cat_{cat}_{calib_valid_lab}_'
                    f'{opt_iter}_kf_{kf}_{eff_ftn_lab}.png')

                out_quants_fig_path = os.path.join(
                    errors_dir, out_quants_fig_name)

                plot_quant_effs(
                    q_quant_effs_dicts,
                    qobs_quants_masks_dict,
                    eff_ftn_lab,
                    lc_idxs,
                    lc_labs,
                    cat,
                    out_quants_fig_path)

            out_lorenz_name = (
                f'lorenz_curves_cat_{cat}_{calib_valid_lab}_'
                f'{opt_iter}_kf_{kf}.png')

            out_lorenz_path = os.path.join(errors_dir, out_lorenz_name)

            plot_lorenz_arr(
                lorenz_x_vals,
                lorenz_y_vals_list,
                lc_idxs,
                lc_labs,
                cat,
                out_lorenz_path)

            out_peak_effs_name = (
                f'peak_effs_cat_{cat}_{calib_valid_lab}_'
                f'{opt_iter}_kf_{kf}.png')

            out_peaks_path = os.path.join(errors_dir, out_peak_effs_name)

            plot_peak_effs(peak_effs, lc_idxs, lc_labs, cat, out_peaks_path)

    return


def plot_peak_effs(peak_effs, lc_idxs, lc_labs, cat, out_path):

    plt.figure(figsize=(15, 10))

    eff_ftn_labs = list(peak_effs[0].keys())

    x_crds = np.arange(len(peak_effs))

    for eff_ftn_lab in eff_ftn_labs:
        eff_vals = [
            peak_effs[i][eff_ftn_lab] for i in range(len(lc_idxs))]

        plt.plot(
            x_crds,
            eff_vals,
            label=f'{eff_ftn_lab}',
            marker='o',
            alpha=0.5)

    plt.xlabel('Simulation')
    plt.ylabel('Efficiency')

    x_crds_labs = [
        f'{lc_labs[i]} - ({lc_idxs[i]})' for i in range(x_crds.shape[0])]

    plt.xticks(x_crds, x_crds_labs, rotation=90)

    plt.title(
        f'Model efficiency for catchment: {cat} using various '
        f'parameter vector(s)')

    plt.grid()
    plt.legend(loc=0, framealpha=0.7)

    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    return


def plot_lorenz_arr(
        lorenz_x_vals, lorenz_y_vals_list, sim_idxs, sim_labs, cat, out_path):

    plt.figure(figsize=(10, 10))

    n_sims = len(sim_idxs)

    plt.plot(
        lorenz_x_vals,
        lorenz_x_vals,
        label=f'equal_contrib',
        alpha=0.5)

    for i in range(n_sims):
        plt.plot(
            lorenz_x_vals,
            lorenz_y_vals_list[i],
            label=f'{sim_labs[i]} ({sim_idxs[i]})',
            alpha=0.5)

    plt.xlabel('Rel. cumm. steps')
    plt.ylabel('Rel. cumm. sq. diff.')

    plt.title(
        f'Lorenz curves for catchment: {cat} using various '
        f'parameter vector(s)')

    plt.grid()
    plt.legend(loc=0, framealpha=0.7)

    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    return


def plot_quant_effs(
        quant_effs_dict,
        quants_masks_dict,
        eff_ftn_lab,
        sim_idxs,
        sim_labs,
        cat,
        out_path):

    plt.figure(figsize=(15, 7))

    n_quants = len(quant_effs_dict[0][eff_ftn_lab])

    bar_x_crds = (np.arange(1., n_quants + 1) / n_quants) - (0.5 / n_quants)

    for i in range(len(sim_idxs)):
        plt.plot(
            bar_x_crds,
            quant_effs_dict[i][eff_ftn_lab],
            label=f'{sim_labs[i]} ({sim_idxs[i]})',
            marker='o',
            alpha=0.5)

    plt.title(
        eff_ftn_lab.upper() +
        f' efficiency for {n_quants} quantiles for cat: {cat}')

    bar_x_crds_labs = [
        f'{bar_x_crds[i]:0.3f} - ({int(quants_masks_dict[i].sum())})'
        for i in range(n_quants)]

    plt.xticks(bar_x_crds, bar_x_crds_labs, rotation=90)

    plt.xlabel('Mean quantile prob. - (N)')
    plt.ylabel(eff_ftn_lab.upper())

    plt.grid()
    plt.legend(loc=0, framealpha=0.7)

    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    return


def get_lorenz_arr(ref_arr, sim_arr):

    sorted_abs_diffs = np.sort(((ref_arr - sim_arr) ** 2))
    sorted_abs_diffs = sorted_abs_diffs / sorted_abs_diffs[-1]
    return sorted_abs_diffs


def get_q_quant_effs_dict(
        qobs_arr, qsim_arr, qobs_quants_masks_dict, eff_ftns_dict):

    n_q_quants = len(qobs_quants_masks_dict)

    quant_effs_dict = {}
    for eff_ftn_key in eff_ftns_dict:
        eff_ftn = eff_ftns_dict[eff_ftn_key]

        quant_effs = []

        for i in range(n_q_quants):
            mask = qobs_quants_masks_dict[i]

            assert mask.sum() > 0

            quant_effs.append(eff_ftn(qobs_arr[mask], qsim_arr[mask], 0))

        quant_effs_dict[eff_ftn_key] = quant_effs
    return quant_effs_dict


def get_quant_masks_dict(in_ser, n_quants):

    n_steps = in_ser.shape[0]

    probs = in_ser.rank().values / (n_steps + 1)
    vals = in_ser.values

    interp_ftn = interp1d(
        np.sort(probs),
        np.sort(vals),
        bounds_error=False,
        fill_value=(vals.min(), vals.max()))

    quant_probs = np.linspace(
        0.,
        1.,
        n_quants + 1,
        endpoint=True)

    quants = interp_ftn(quant_probs)
    quants[-1] = quants[-1] * 1.1

    masks_dict = {}
    for i in range(n_quants):
        masks_dict[i] = (vals >= quants[i]) & (vals < quants[i + 1])

    return masks_dict


def get_peaks_mask(qobs_arr):

    rising = qobs_arr[1:] - qobs_arr[:-1] > 0
    recing = qobs_arr[1:-1] - qobs_arr[2:] > 0

    mask = np.concatenate(([False], rising[:-1] & recing, [False]))
    return mask


def get_peaks_effs_dict(qobs_arr, qsim_arr, mask, eff_ftns_dict):

    peak_effs_dict = {}
    for eff_ftn_key in eff_ftns_dict:
        eff_ftn = eff_ftns_dict[eff_ftn_key]

        peak_effs_dict[eff_ftn_key] = eff_ftn(
            qobs_arr[mask], qsim_arr[mask], 0)

    return peak_effs_dict


def plot_cat_vars_errors(plot_args):

    (cat_db, err_var_labs) = plot_args

    ds_labs = ['calib']

    with h5py.File(cat_db, 'r') as db:
        out_dir = db['data'].attrs['main']
        out_dir = os.path.join(out_dir, r'11_errors')

        if not os.path.exists(out_dir):
            try:
                os.mkdir(out_dir)

            except FileExistsError:
                pass

        kfolds = db['data'].attrs['kfolds']

        cat = db.attrs['cat']

        area_arr = db['data/area_arr'][...]
        rarea_arr = area_arr.reshape(-1, 1)

        for ds_lab in ds_labs:
            for kf_i in range(1, kfolds + 1):
                kf_str = f'kf_{kf_i:02d}'

                kf_grp = db[f'{ds_lab}/{kf_str}']

                qsim = kf_grp['qsim_arr'][...]

                qact = kf_grp['qact_arr'][...]

                q_errs = qsim - qact

                plot_cat_vars_errors_kf(
                    kf_grp,
                    cat,
                    kf_i,
                    ds_lab,
                    q_errs,
                    err_var_labs,
                    rarea_arr,
                    out_dir)

    return


def plot_cat_vars_errors_kf(
        kf_grp,
        cat,
        kf_i,
        ds_lab,
        q_errs,
        err_var_labs,
        rarea_arr,
        out_dir):

    for err_var_lab in err_var_labs:
        if err_var_lab == 'temperature':
            err_var = kf_grp['tem_arr'][...]

            x_lab = 'Temperature (°C)'

        else:
            raise NotImplementedError(
                f'Don\'t know what to do with {err_var_lab}!')

        err_var = (rarea_arr * err_var).sum(axis=0)

        sort_idxs = np.argsort(err_var)

        err_var_sort = err_var[sort_idxs]
        q_errs_sort = q_errs[sort_idxs]

        over_idxs = q_errs_sort >= 0

        err_var_over = err_var_sort[over_idxs]
        err_var_under = err_var_sort[~over_idxs]

        q_errs_over = q_errs_sort[over_idxs]
        q_errs_under = np.abs(q_errs_sort[~over_idxs])

        plt.figure(figsize=(20, 10))

        ax = plt.gca()

        ax.set_yscale('log')

        ax.scatter(
            err_var_over,
            q_errs_over,
            alpha=0.1,
            label='over',
            color='C0')

        ax.scatter(
            err_var_under,
            q_errs_under,
            alpha=0.1,
            label='under',
            color='C1')

        ax.axhline(
            np.exp(np.log(q_errs_over).mean()),
            color='C0',
            alpha=0.7,
            label='mean over')

        ax.axhline(
            np.exp(np.log(q_errs_under).mean()),
            color='C1',
            alpha=0.7,
            label='mean under')

        ax.set_xlabel(x_lab)
        ax.set_ylabel('Abs. discharge difference (sim. - obs.)')

        ax.grid()
        ax.legend()

        ax.set_title(
            f'Discharge differences sorted w.r.t {err_var_lab}\n'
            f'Catchment {cat}, kfold no. {kf_i:02d}')

        out_fig_name = f'{err_var_lab}_error_{cat}_{ds_lab}_kf_{kf_i:02d}.png'

        plt.savefig(str(Path(out_dir, out_fig_name)), bbox_inches='tight')

        plt.close()

    return


def plot_cat_prms_transfer_perfs(plot_args):

    '''Plot performances for a given using parameter vectors from other
    catchments.'''

    cat_db, (kf_prm_dict, cats_vars_dict) = plot_args

    with h5py.File(cat_db, 'r') as db:
        main_out_dir = db['data'].attrs['main']

        trans_out_dir = os.path.join(main_out_dir, '09_prm_trans_compare')

        try:
            os.mkdir(trans_out_dir)

        except FileExistsError:
            pass

        kfolds = db['data'].attrs['kfolds']
        cat = db.attrs['cat']

        off_idx = db['data'].attrs['off_idx']
        conv_ratio = db['data'].attrs['conv_ratio']
        use_obs_flow_flag = db['data'].attrs['use_obs_flow_flag']
        area_arr = db['data/area_arr'][...]
        n_cells = area_arr.shape[0]

        all_kfs_dict = {}
        for i in range(1, kfolds + 1):
            cd_db = db[f'calib/kf_{i:02d}']
            kf_dict = {key: cd_db[key][...] for key in cd_db}
            kf_dict['use_obs_flow_flag'] = use_obs_flow_flag

            all_kfs_dict[i] = kf_dict

    cats = list(kf_prm_dict[1].keys())
    n_cats = len(cats)

    cat_vars_dict = cats_vars_dict[cat]

    ns_arr = np.full((kfolds, n_cats), np.nan)
    ln_ns_arr = ns_arr.copy()

    if (kfolds < 6) and (ns_arr.shape[1] < 6):

        eff_strs_arr = np.full(ns_arr.shape, fill_value='', dtype='|U13')

        plot_val_strs = True

    else:
        plot_val_strs = False

    for i in range(1, kfolds + 1):
        kf_dict = all_kfs_dict[i]

        for rno, j in enumerate(kf_prm_dict):
            trans_cat_dict = kf_prm_dict[j]

            for cno, trans_cat in enumerate(trans_cat_dict):
                trans_opt_prms = trans_cat_dict[trans_cat]
                kf_dict['hbv_prms'] = tfm_opt_to_hbv_prms_py(
                    cat_vars_dict['prms_flags'],
                    cat_vars_dict['f_var_infos'],
                    cat_vars_dict['prms_idxs'],
                    cat_vars_dict['f_vars'],
                    trans_opt_prms['opt_prms'],
                    cat_vars_dict['bds_arr'],
                    n_cells)

                ns, ln_ns = get_perfs(
                    kf_dict,
                    area_arr,
                    conv_ratio,
                    off_idx)

                ns_arr[rno, cno] = ns
                ln_ns_arr[rno, cno] = ln_ns

                if plot_val_strs:
                    eff_strs_arr[rno, cno] = f'{ns:0.4f};{ln_ns:0.4f}'

        plt.figure(figsize=(n_cats * 2, kfolds * 2))
        plot_rows = 8
        plot_cols = 8

        x_ticks = np.arange(0.5, n_cats, 1)
        y_ticks = np.arange(0.5, kfolds, 1)

        x_tick_labs = cats
        y_tick_labs = np.arange(1, kfolds + 1, 1)

        perfs_list = [ns_arr, ln_ns_arr]
        n_perfs = len(perfs_list)
        titls_list = ['NS', 'Ln_NS']
        col_i = 0
        cspan = 4

        for p_i in range(n_perfs):
            ax = plt.subplot2grid(
                (plot_rows, plot_cols),
                (0, col_i),
                rowspan=7,
                colspan=cspan)

            col_i += cspan

            ps = ax.pcolormesh(
                perfs_list[p_i],
                cmap=plt.get_cmap('Blues'),
                vmin=0,
                vmax=1)

            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)

            if (p_i == 0) or (p_i == (n_perfs - 1)):
                ax.set_xticklabels(x_tick_labs, rotation=90)
                ax.set_yticklabels(y_tick_labs, rotation=90)
                ax.set_ylabel('K-fold')

            else:
                ax.set_xticklabels([])
                ax.set_yticklabels([])

            ax.set_xlabel('Catchment')

            ax.set_title(titls_list[p_i])

            ax.set_xlim(0, n_cats)
            ax.set_ylim(0, kfolds)

            if plot_val_strs:
                xx, yy = np.meshgrid(
                    np.arange(0.5, n_cats, 1),
                    np.arange(0.5, kfolds, 1))

                xx = xx.ravel()
                yy = yy.ravel()

                ravel = eff_strs_arr.ravel()
                [ax.text(
                    xx[j],
                    yy[j],
                    ravel[j].split(';')[p_i],
                    va='center',
                    ha='center',
                    rotation=45)
                 for j in range(kfolds * n_cats)]

            ax.set_aspect('equal', 'box')

        else:
            ax.yaxis.set_ticks_position('right')
            ax.yaxis.set_label_position('right')

        cb_plot = plt.subplot2grid(
            (plot_rows, plot_cols), (7, 0), rowspan=1, colspan=8)

        cb_plot.set_axis_off()
        cb = plt.colorbar(
            ps,
            ax=cb_plot,
            fraction=0.4,
            aspect=20,
            orientation='horizontal',
            extend='min')

        cb.set_ticks(np.arange(0, 1.01, 0.2))
        cb.set_label('Efficiency')

        plt.suptitle(
            f'HBV Parameter transfer performance values for the '
            f'catchment {cat} on k-fold: {i}')

        out_file_name = f'{cat}_kf_{i:02d}_prm_trans_compare.png'
        out_file_path = os.path.join(trans_out_dir, out_file_name)
        plt.savefig(out_file_path, bbox_inches='tight')
        plt.close()
    return


def get_perfs(kf_dict, area_arr, conv_ratio, off_idx):

    '''Get per formances for given parameter vectors.'''

    temp_dist_arr = kf_dict['tem_arr']
    prec_dist_arr = kf_dict['ppt_arr']
    pet_dist_arr = kf_dict['pet_arr']
    prms_dist_arr = kf_dict['hbv_prms']

    all_outputs_dict = hbv_loop_py(
        temp_dist_arr,
        prec_dist_arr,
        pet_dist_arr,
        prms_dist_arr,
        kf_dict['ini_arr'],
        area_arr,
        conv_ratio)

    assert all_outputs_dict['loop_ret'] == 0.0

    del temp_dist_arr, prec_dist_arr, pet_dist_arr, prms_dist_arr

    q_sim_arr = all_outputs_dict['qsim_arr']

    del all_outputs_dict

    extra_us_inflow_flag = 'extra_us_inflow' in kf_dict

    q_sim_arr = (q_sim_arr).copy(order='C')
    q_act_arr = kf_dict['qact_arr']
    if extra_us_inflow_flag:
        extra_us_inflow = kf_dict['extra_us_inflow']
        q_sim_arr = q_sim_arr + extra_us_inflow

    ns = get_ns_cy(q_act_arr, q_sim_arr, off_idx)

    ln_ns = get_ln_ns_cy(q_act_arr, q_sim_arr, off_idx)

    return (ns, ln_ns)


def plot_cat_kfold_effs(args):

    '''Plot catchment performances using parameters from every kfold
    on all kfolds.
    '''

    cat_db_path, (ann_cyc_flag, hgs_db_path) = args

    with h5py.File(cat_db_path, 'r') as db:
        out_dir = db['data'].attrs['main']
        out_dir = os.path.join(out_dir, r'05_kfolds_perf')

        if not os.path.exists(out_dir):
            try:
                os.mkdir(out_dir)

            except FileExistsError:
                pass

        kfolds = db['data'].attrs['kfolds']
        cat = db.attrs['cat']

        off_idx = db['data'].attrs['off_idx']
        cv_flag = db['data'].attrs['cv_flag']

        if cv_flag:
            print('plot_cat_kfold_effs not possible with cv_flag!')
            return

        else:
            use_step_flag = db['valid/kf_01/use_step_flag'][()]
            if use_step_flag:
                use_step_arr = db['valid/kf_01/use_step_arr'][...]

    kfold_q_sers_dict = {}
    with shelve.open(hgs_db_path, 'r') as db:
        for i in range(1, kfolds + 1):
            kf_str = f'kf_{i:02d}'
            kfold_q_sers_dict[i] = (
                db['valid'][kf_str]['out_cats_flow_df'][cat].values.copy('c'))

        date_idx = db['valid'][kf_str]['qact_df'][cat].index
        qact_arr = db['valid'][kf_str]['qact_df'][cat].values.copy('c')

    assert kfolds >= 1, 'kfolds can be 1 or greater only!'

    if ann_cyc_flag:
        try:
            assert np.all(np.abs((date_idx[1:] - date_idx[:-1]).days) == 1)
            q_cyc_ser = pd.Series(index=date_idx, data=qact_arr)
            q_cyc_arr = get_daily_annual_cycle(q_cyc_ser).values

        except AssertionError:
            print('Annual cycle comparision available only for daily series!')
            ann_cyc_flag = False
            q_cyc_arr = None

    else:
        q_cyc_arr = None

    sel_idxs_arr = np.linspace(
        0, qact_arr.shape[0], kfolds + 1, dtype=np.int64, endpoint=True)

    uni_sel_idxs_list = np.unique(sel_idxs_arr).tolist()
    n_sel_idxs = len(uni_sel_idxs_list)

    assert n_sel_idxs >= 2, 'kfolds too high or data points too low!'

    uni_sel_idxs_list.extend([0, uni_sel_idxs_list[-1]])

    perfo_ftns_list = [get_ns_cy, get_ln_ns_cy, get_kge_cy, get_pcorr_cy]
    perfo_ftns_names = ['NS', 'Ln_NS', 'KGE', 'P_Corr']

    key_str_left = 'eff'
    key_str_right = 'eff'

    prec_len = 6
    arr_str_len = prec_len

    if use_step_flag:
        prt_perfo_ftns_list = [
            get_ns_prt_cy, get_ln_ns_prt_cy, get_kge_prt_cy, get_pcorr_prt_cy]

        assert len(perfo_ftns_list) == len(prt_perfo_ftns_list)

        prt_calib_step_arr = use_step_arr
        prt_valid_step_arr = np.logical_not(
            use_step_arr.astype(bool)).astype(np.int32)

        arr_str_len += ((prec_len * 2) + 2)
        key_str_left += '\nprt_calib_eff\nprt_valid_eff'
        key_str_right += '\nprt_calib_eff\nprt_valid_eff'

    else:
        prt_perfo_ftns_list = []

    if q_cyc_arr is not None:
        res_perfo_ftns_list = [get_ns_var_res_cy, get_ln_ns_var_res_cy]
        n_cyc_perfs = len(res_perfo_ftns_list)

        arr_str_len += ((prec_len * 2) + 2)
        key_str_left += '\nres_eff\nann_cyc_eff'

    else:
        res_perfo_ftns_list = []
        n_cyc_perfs = 0

    n_perfs = len(perfo_ftns_names)
    over_all_perf_arr = np.full(
        shape=(n_perfs, kfolds), fill_value=np.nan)

    over_all_perf_str_arr = np.full(
        shape=(n_perfs, kfolds), fill_value='', dtype=f'|U{arr_str_len}')

    kfold_perfos_arr = np.full(
        shape=(n_perfs, kfolds, kfolds), fill_value=np.nan)

    kfold_perfos_str_arr = np.full(
        shape=(n_perfs, kfolds, kfolds),
        fill_value='',
        dtype=f'|U{arr_str_len}')

    for iter_no in range(kfolds):
        for kf_i in range(n_sel_idxs + 1):
            if kf_i == kfolds:
                continue

            cst_idx = uni_sel_idxs_list[kf_i]
            cen_idx = uni_sel_idxs_list[kf_i + 1]

            qsim_arr = kfold_q_sers_dict[iter_no + 1][cst_idx:cen_idx]

            curr_qact_arr = qact_arr[cst_idx:cen_idx]
            assert (qsim_arr.shape[0] == curr_qact_arr.shape[0]), (
                'Unequal shapes!')

            if use_step_flag:
                curr_prt_calib_arr = prt_calib_step_arr[cst_idx:cen_idx]
                curr_prt_valid_arr = prt_valid_step_arr[cst_idx:cen_idx]
                assert (
                    qsim_arr.shape[0] ==
                    curr_qact_arr.shape[0] ==
                    curr_prt_calib_arr.shape[0]), 'Unequal shapes!'

            for i, perfo_ftn in enumerate(perfo_ftns_list):
                perf_val = perfo_ftn(curr_qact_arr, qsim_arr, off_idx)

                if kf_i < kfolds:
                    kfold_perfos_arr[i, iter_no, kf_i] = perf_val
                    kfold_perfos_str_arr[i, iter_no, kf_i] = (
                        f'{perf_val:0.4f}')

                elif kf_i > kfolds:
                    over_all_perf_arr[i, iter_no] = perf_val
                    over_all_perf_str_arr[i, iter_no] = f'{perf_val:0.4f}'

                else:
                    raise RuntimeError('Fucked up!')

            for i, perfo_ftn in enumerate(prt_perfo_ftns_list):
                calib_perf_val = perfo_ftn(
                    curr_qact_arr, qsim_arr, curr_prt_calib_arr, off_idx)
                valid_perf_val = perfo_ftn(
                    curr_qact_arr, qsim_arr, curr_prt_valid_arr, off_idx)

                _cstr = f'\n{calib_perf_val:0.4f}'
                _vstr = f'\n{valid_perf_val:0.4f}'

                if kf_i < kfolds:
                    kfold_perfos_arr[i, iter_no, kf_i] = perf_val
                    kfold_perfos_str_arr[i, iter_no, kf_i] += _cstr
                    kfold_perfos_str_arr[i, iter_no, kf_i] += _vstr

                elif kf_i > kfolds:
                    over_all_perf_str_arr[i, iter_no] += _cstr
                    over_all_perf_str_arr[i, iter_no] += _vstr

                else:
                    raise RuntimeError('Fucked up!')

            if q_cyc_arr is None:
                continue

            curr_q_cyc_arr = q_cyc_arr[cst_idx:cen_idx]
            for i, res_perfo_ftn in enumerate(res_perfo_ftns_list):
                res_perf_val = res_perfo_ftn(
                    curr_qact_arr, qsim_arr, curr_q_cyc_arr, off_idx)

                cyc_perf_val = perfo_ftns_list[i](
                    curr_qact_arr, curr_q_cyc_arr, off_idx)

                rstr = f'\n{res_perf_val:0.4f}'
                ystr = f'\n{cyc_perf_val:0.4f}'

                if kf_i < kfolds:
                    kfold_perfos_str_arr[i, iter_no, kf_i] += rstr
                    kfold_perfos_str_arr[i, iter_no, kf_i] += ystr

                    if ((cyc_perf_val >
                         kfold_perfos_arr[i, iter_no, kf_i]) and
                        (res_perf_val > 0)):

                        print('1 Impossible:', cat, i, iter_no)

                elif kf_i > kfolds:
                    over_all_perf_str_arr[i, iter_no] += rstr
                    over_all_perf_str_arr[i, iter_no] += ystr

                    if ((cyc_perf_val > over_all_perf_arr[i, iter_no]) and
                        (res_perf_val > 0)):

                        print('2 Impossible:', cat, i, iter_no)

                else:
                    raise RuntimeError('Fucked up!')

    n_rows = 5
    n_cols = n_perfs
    plt.figure(figsize=(13, 5.5))

    top_xs = np.arange(0.5, kfolds, 1)
    top_ys = np.repeat(0.5, kfolds)

    bot_xx, bot_yy = np.meshgrid(
        np.arange(0.5, kfolds, 1), np.arange(0.5, kfolds, 1))

    bot_xx = bot_xx.ravel()
    bot_yy = bot_yy.ravel()

    fts = 5
    bot_xy_ticks = top_xs
    bot_xy_ticklabels = np.arange(1, kfolds + 1, 1)

    if kfolds > 6:
        over_all_perf_str_arr = np.full(
            shape=(n_perfs, kfolds), fill_value='', dtype=f'|U{arr_str_len}')

        kfold_perfos_str_arr = np.full(
            shape=(n_perfs, kfolds, kfolds),
            fill_value='',
            dtype=f'|U{arr_str_len}')

        key_str_left = ''
        key_str_right = ''

    if q_cyc_arr is not None:
        assert n_cyc_perfs == 2, (
            'Residual efficiencies\' plot only implemented in case of NS and '
            'Ln_NS!')

    for i in range(n_perfs):
        top_ax = plt.subplot2grid((n_rows, n_cols), (0, i), 1, 1)

        top_ax.pcolormesh(
            np.atleast_2d(over_all_perf_arr[i]),
            cmap=plt.get_cmap('Blues'),
            vmin=0,
            vmax=1)

        [top_ax.text(
            top_xs[j],
            top_ys[j],
            over_all_perf_str_arr[i, j],
            va='center',
            ha='center',
            size=fts)
         for j in range(kfolds)]

        bot_ax = plt.subplot2grid((n_rows, n_cols), (1, i), 3, 1)
        ps = bot_ax.pcolormesh(
            kfold_perfos_arr[i],
            cmap=plt.get_cmap('Blues'),
            vmin=0,
            vmax=1)

        ravel = kfold_perfos_str_arr[i].ravel()

        [bot_ax.text(
            bot_xx[j],
            bot_yy[j],
            ravel[j],
            va='center',
            ha='center',
            size=fts)
         for j in range(kfolds ** 2)]

        top_ax.set_xticks([])
        top_ax.set_xticklabels([])
        top_ax.set_yticks([])
        top_ax.set_yticklabels([])

        bot_ax.set_xlabel(perfo_ftns_names[i])
        bot_ax.set_xticks(bot_xy_ticks)
        bot_ax.set_xticklabels(bot_xy_ticklabels)

        if not i:
            bot_ax.set_yticks(bot_xy_ticks)
            bot_ax.set_yticklabels(bot_xy_ticklabels)
            bot_ax.set_ylabel('Split Efficiency')
            top_ax.set_ylabel('Overall\nEfficiency')

        else:
            bot_ax.set_yticks([])
            bot_ax.set_yticklabels([])

    cb_ax = plt.subplot2grid((n_rows, n_cols), (n_rows - 1, 0), 1, n_cols)
    cb_ax.set_axis_off()
    cb = plt.colorbar(
        ps,
        ax=cb_ax,
        fraction=0.4,
        aspect=20,
        orientation='horizontal',
        extend='min')

    cb.set_ticks(np.arange(0, 1.01, 0.2))
    cb.set_label('Efficiency')

    cb_ax.text(0, 0, key_str_left, va='top', size=fts)
    cb_ax.text(1, 0, key_str_right, va='top', ha='right', size=fts)

    title_str = ''
    title_str += f'{kfolds}-folds overall/split calibration and validation '
    title_str += f'results\nfor the catchment: {cat} with '
    title_str += f'{uni_sel_idxs_list[1]} steps per fold\n'
    plt.suptitle(title_str)
    plt.subplots_adjust(top=0.8)

    out_kfold_fig_loc = os.path.join(out_dir, f'kfolds_compare_{cat}.png')
    plt.savefig(out_kfold_fig_loc, bbox_inches='tight', dpi=200)
    plt.close()
    return


def get_daily_annual_cycles(in_data_df, n_cpus=1):

    '''Given full time series dataframe, get daily annual cycle dataframe
    for all columns.
    '''

    annual_cycle_df = pd.DataFrame(
        index=in_data_df.index, columns=in_data_df.columns, dtype=float)

    cat_ser_gen = (in_data_df[col].copy() for col in in_data_df.columns)

    if n_cpus > 1:
        mp_pool = ProcessPool(n_cpus)
        mp_pool.restart(True)
        try:
            ann_cycs = list(mp_pool.uimap(
                get_daily_annual_cycle, cat_ser_gen))

            for col_ser in ann_cycs:
                annual_cycle_df.update(col_ser)

            mp_pool.clear()

        except Exception as msg:
            mp_pool.close()
            mp_pool.join()
            print('Error in get_daily_annual_cycle:', msg)

    else:
        for col_ser in cat_ser_gen:
            annual_cycle_df.update(get_daily_annual_cycle(col_ser))

    return annual_cycle_df


def get_daily_annual_cycle(col_ser):

    '''Given full time series series, get daily annual cycle series.
    '''

    assert isinstance(col_ser, pd.Series), 'Expected a pd.Series object!'
    col_ser.dropna(inplace=True)

    # For each day of a year, get the days for all year and average them
    # the annual cycle is the average value and is used for every doy of
    # all years
    for month in range(1, 13):
        month_idxs = col_ser.index.month == month
        for dom in range(1, 32):
            dom_idxs = col_ser.index.day == dom
            idxs_intersect = np.logical_and(month_idxs, dom_idxs)
            curr_day_vals = col_ser.values[idxs_intersect]

            if not curr_day_vals.shape[0]:
                continue

            assert not np.any(np.isnan(curr_day_vals)), (
                'NaNs in curr_day_vals!')

            curr_day_avg_val = curr_day_vals.mean()
            col_ser.loc[idxs_intersect] = curr_day_avg_val
    return col_ser


def get_fdc(in_ser):

    '''Get flow duration curve
    '''

    assert isinstance(in_ser, pd.Series), 'Expected a pd.Series object!'

    probs = (in_ser.rank(ascending=False) / (in_ser.shape[0] + 1)).values
    vals = in_ser.values.copy()

    sort_idxs = np.argsort(probs)

    probs = probs[sort_idxs]
    vals = vals[sort_idxs]
    return probs, vals


def plot_cats_ann_cycs_fdcs_comp_kf(
        db, i, db_lab, title_lab, off_idx, out_dir):

    '''Plot observed vs. simulated annual cycles  and flow duration
    for a given kfold for all catchments.

    This function is supposed to be called by plot_cats_ann_cycs_fdcs_comp'''

    kf_str = f'kf_{i:02d}'

    kf_qact_df = db[db_lab][kf_str]['qact_df'].iloc[off_idx:]
    cats = kf_qact_df.columns

    date_idx = kf_qact_df.index

    if not np.all(np.abs((date_idx[1:] - date_idx[:-1]).days) == 1):
        print('Annual cycle comparision available only for daily series!')
        return

    qact_ann_cyc_df = get_daily_annual_cycles(
        kf_qact_df.iloc[off_idx:])

    kf_qsim_df = db[db_lab][kf_str]['out_cats_flow_df'].iloc[off_idx:]
    qsim_ann_cyc_df = get_daily_annual_cycles(kf_qsim_df)

    assert np.all(kf_qact_df.index == kf_qsim_df.index)

    plot_qact_df = qact_ann_cyc_df.iloc[:365]
    plot_qsim_df = qsim_ann_cyc_df.iloc[:365]

    doy_labs = plot_qact_df.index.dayofyear
    plt_xcrds = np.arange(1, plot_qact_df.shape[0] + 1, 1)

    for cat in cats:
        # ann cyc
        plt.plot(
            plt_xcrds,
            plot_qact_df[cat].values,
            alpha=0.7,
            label='qact')

        plt.plot(
            plt_xcrds,
            plot_qsim_df[cat].values,
            alpha=0.7,
            label='qsim')

        plt.grid()
        plt.legend()

        plt.xticks(plt_xcrds[::15], doy_labs[::15], rotation=90)

        plt.title(
            f'''
            {title_lab} time series annual cycle
            comparision for actual and simulated flows for the
            catchemnt {cat} using parameters of kfold: {i}
            ''')

        plt.xlabel('Time (day of year)')
        plt.ylabel('Discharge ($m^3/s$)')

        out_path = os.path.join(
            out_dir, f'{kf_str}_{db_lab}_ann_cyc_compare_{cat}.png')
        plt.savefig(out_path, bbox_inches='tight')

        plt.clf()

        # FDC
        qact_probs, qact_vals = get_fdc(kf_qact_df[cat])
        qsim_probs, qsim_vals = get_fdc(kf_qsim_df[cat])

        plt.semilogy(
            qact_probs,
            qact_vals,
            alpha=0.7,
            label='qact')

        plt.semilogy(
            qsim_probs,
            qsim_vals,
            alpha=0.7,
            label='qsim')

        plt.grid()
        plt.legend()

        plt.title(
            f'''
            {title_lab} flow duration curve comparison for
            actual and simulated flows for the catchemnt {cat}
            using parameters of kfold: {i}
            ''')

        plt.xlabel('Exceedence Probability (-)')
        plt.ylabel('Discharge ($m^3/s$)')

        out_path = os.path.join(
            out_dir, f'{kf_str}_{db_lab}_fdc_compare_{cat}.png')
        plt.savefig(out_path, bbox_inches='tight')

        plt.clf()
    return


def plot_cats_ann_cycs_fdcs_comp(hgs_db_path, off_idx, out_dir):

    '''Plot observed vs. simulated annual cycles  and flow duration
    for all catchments.
    '''

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    title_lab_list = ['Validation', 'Calibration']
    db_lab_list = ['valid', 'calib']

    plt.figure(figsize=(20, 10))
    with shelve.open(hgs_db_path, 'r') as db:
        kfolds = len(db['calib'].keys())

        for i in range(len(db_lab_list)):
            for j in range(1, kfolds + 1):
                plot_cats_ann_cycs_fdcs_comp_kf(
                    db,
                    j,
                    db_lab_list[i],
                    title_lab_list[i],
                    off_idx,
                    out_dir)

    plt.close()
    return
