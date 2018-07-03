# -*- coding: utf-8 -*-
'''
Created on Oct 12, 2017

@author: Faizan Anwar, IWS Uni-Stuttgart
'''

import os
import timeit
import time
import shelve
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text
from pathos.multiprocessing import ProcessPool

from ..models import (
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


def plot_cat_kfold_effs(args):
    cat_db_path, (ann_cyc_flag, hgs_db_path) = args

    with shelve.open(cat_db_path.rsplit('.', 1)[0], 'r') as db:
        out_dir = db['data']['dirs_dict']['main']
        out_dir = os.path.join(out_dir, r'05_kfolds_perf')

        if not os.path.exists(out_dir):
            try:
                os.mkdir(out_dir)
            except:
                pass

        kfolds = db['data']['kfolds']
        cat = db['cat']

        off_idx = db['data']['off_idx']

        use_step_flag = db['valid']['kf_01']['use_step_flag']
        if use_step_flag:
            use_step_arr = db['valid']['kf_01']['use_step_arr']

    kfold_q_sers_dict = {}
    with shelve.open(hgs_db_path, 'r') as db:
        for i in range(1, kfolds + 1):
            kf_str = f'kf_{i:02d}'
            kfold_q_sers_dict[i] = (
                db['valid'][kf_str]['out_cats_flow_df'][cat].values.copy(order='c'))
        else:
            date_idx = db['valid'][kf_str]['out_cats_flow_df'].index
            qact_arr = db['valid'][kf_str]['qact_df'][cat].values.copy(order='c')

    assert kfolds >= 1, 'kfolds can be 1 or greater only!'

    if ann_cyc_flag:
        try:
            assert np.all(np.abs((date_idx[1:] - date_idx[:-1]).days) == 1)
            q_cyc_ser = pd.Series(index=date_idx, data=qact_arr)
            q_cyc_arr = _get_daily_annual_cycle(q_cyc_ser).values

        except AssertionError:
            print('Annual cycle comparision available only for daily series!')
            ann_cyc_flag = False
            q_cyc_arr = None
    else:
        q_cyc_arr = None

    sel_idxs_arr = np.linspace(0,
                               qact_arr.shape[0],
                               kfolds + 1,
                               dtype=np.int64,
                               endpoint=True)

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
        prt_perfo_ftns_list = [get_ns_prt_cy,
                               get_ln_ns_prt_cy,
                               get_kge_prt_cy,
                               get_pcorr_prt_cy]
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
    over_all_perf_arr = np.full(shape=(n_perfs, kfolds),
                                fill_value=np.nan)
    over_all_perf_str_arr = np.full(shape=(n_perfs, kfolds),
                                    fill_value='',
                                    dtype=f'|U{arr_str_len}')
    kfold_perfos_arr = np.full(shape=(n_perfs, kfolds, kfolds),
                               fill_value=np.nan)
    kfold_perfos_str_arr = np.full(shape=(n_perfs, kfolds, kfolds),
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
                assert (qsim_arr.shape[0] ==
                        curr_qact_arr.shape[0] ==
                        curr_prt_calib_arr.shape[0]), 'Unequal shapes!'

            for i, perfo_ftn in enumerate(perfo_ftns_list):
                perf_val = perfo_ftn(curr_qact_arr, qsim_arr, off_idx)

                if kf_i < kfolds:
                    kfold_perfos_arr[i, iter_no, kf_i] = perf_val
                    kfold_perfos_str_arr[i, iter_no, kf_i] = f'{perf_val:0.4f}'
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
                res_perf_val = res_perfo_ftn(curr_qact_arr,
                                             qsim_arr,
                                             curr_q_cyc_arr,
                                             off_idx)

                cyc_perf_val = perfo_ftns_list[i](curr_qact_arr,
                                                  curr_q_cyc_arr,
                                                  off_idx)
                _rstr = f'\n{res_perf_val:0.4f}'
                _ystr = f'\n{cyc_perf_val:0.4f}'

                if kf_i < kfolds:
                    kfold_perfos_str_arr[i, iter_no, kf_i] += _rstr
                    kfold_perfos_str_arr[i, iter_no, kf_i] += _ystr

                    if ((cyc_perf_val > kfold_perfos_arr[i, iter_no, kf_i]) and
                        (res_perf_val > 0)):
                        print('1 Impossible:', cat, i, iter_no)

                elif kf_i > kfolds:
                    over_all_perf_str_arr[i, iter_no] += _rstr
                    over_all_perf_str_arr[i, iter_no] += _ystr

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

    bot_xx, bot_yy = np.meshgrid(np.arange(0.5, kfolds, 1),
                                 np.arange(0.5, kfolds, 1))
    bot_xx = bot_xx.ravel()
    bot_yy = bot_yy.ravel()

    fts = 5
    bot_xy_ticks = top_xs
    bot_xy_ticklabels = np.arange(1, kfolds + 1, 1)

    if q_cyc_arr is not None:
        assert n_cyc_perfs == 2, (
            'Residual efficiencies\' plot only implemented in case of NS and '
             'Ln_NS!')

    for i in range(n_perfs):
        top_ax = plt.subplot2grid((n_rows, n_cols), (0, i), 1, 1)
        top_ax.pcolormesh(np.atleast_2d(over_all_perf_arr[i]),
                          cmap=plt.get_cmap('Blues'),
                          vmin=0,
                          vmax=1)

        [top_ax.text(top_xs[j],
                     top_ys[j],
                     over_all_perf_str_arr[i, j],
                     va='center',
                     ha='center',
                     size=fts)
         for j in range(kfolds)]

        bot_ax = plt.subplot2grid((n_rows, n_cols), (1, i), 3, 1)
        _ps = bot_ax.pcolormesh(kfold_perfos_arr[i],
                                cmap=plt.get_cmap('Blues'),
                                vmin=0,
                                vmax=1)

        _ravel = kfold_perfos_str_arr[i].ravel()
        [bot_ax.text(bot_xx[j],
                     bot_yy[j],
                     _ravel[j],
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
    cb = plt.colorbar(_ps,
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


def plot_kfolds_best_hbv_prms_2d(dbs_dir):

    cats_dbs = glob(os.path.join(dbs_dir, 'cat_*.bak'))

    assert cats_dbs

    kf_prms_dict = {}
    rows_dict = {}
    cols_dict = {}

    with shelve.open(cats_dbs[0].rsplit('.', 1)[0], 'r') as db:
        kfolds = db['data']['kfolds']
        shape = db['data']['shape']

        prms_labs = db['data']['all_prms_labs']
        lumped_prms_flag = db['data']['run_as_lump_flag']

        out_dir = db['data']['dirs_dict']['main']
        out_dir = os.path.join(out_dir, r'07_2d_kfold_prms')
        if not os.path.exists(out_dir):
            try:
                os.mkdir(out_dir)
            except:
                pass

    min_row = +np.inf
    max_row = -np.inf
    min_col = +np.inf
    max_col = -np.inf

    for cat_db in cats_dbs:
        with shelve.open(cat_db.rsplit('.', 1)[0], 'r') as db:
            cat = db['cat']
            assert kfolds == db['data']['kfolds']

            if cat not in rows_dict:
                rows_dict[cat] = db['data']['rows']

                if rows_dict[cat].min() < min_row:
                    min_row = rows_dict[cat].min()

                if rows_dict[cat].max() > max_row:
                    max_row = rows_dict[cat].max()

            if cat not in cols_dict:
                cols_dict[cat] = db['data']['cols']

                if cols_dict[cat].min() < min_col:
                    min_col = cols_dict[cat].min()

                if cols_dict[cat].max() > max_col:
                    max_col = cols_dict[cat].max()

            for i in range(1, kfolds + 1):
                if i not in kf_prms_dict:
                    kf_prms_dict[i] = {}

                kf_prms_dict[i][cat] = db['calib'][f'kf_{i:02d}']['hbv_prms']

    plot_min_max_lims = (min_row - 1, max_row + 1, min_col - 1, max_col + 1)

    for i in range(1, kfolds + 1):
        _plot_kf_prms_2d(
            i,
            kf_prms_dict[i],
            out_dir,
            prms_labs,
            shape,
            rows_dict,
            cols_dict,
            plot_min_max_lims,
            lumped_prms_flag)

    return


def _plot_kf_prms_2d(
        kf_i,
        kf_prms_dict,
        out_dir,
        prms_labs,
        shape,
        rows_dict,
        cols_dict,
        plot_min_max_lims,
        lumped_prms_flag):

    for p_i, prm in enumerate(prms_labs):
        plot_grid = np.full(shape, np.nan)
        for cat in kf_prms_dict:
            cell_prms = kf_prms_dict[cat][:, p_i]
            plot_grid[rows_dict[cat], cols_dict[cat]] = cell_prms

        plt.figure()
        plt.imshow(plot_grid, origin='lower')

        if lumped_prms_flag:
            for cat in kf_prms_dict:
                plt.text(cols_dict[cat][0],
                         rows_dict[cat][0],
                         cat,
                         va='center',
                         ha='center')

        plt.colorbar(orientation='horizontal')
        plt.title(prm)
        plt.ylim(plot_min_max_lims[0], plot_min_max_lims[1])
        plt.xlim(plot_min_max_lims[2], plot_min_max_lims[3])

        plt.xticks([], [])
        plt.yticks([], [])
        plt.savefig(os.path.join(out_dir, f'{kf_i:02d}_{prm}.png'),
                    bbox_inches='tight')
        plt.close()
    return


def plot_kfolds_best_prms(dbs_dir):

    cats_dbs = glob(os.path.join(dbs_dir, 'cat_*.bak'))

    assert cats_dbs
    for cat_db in cats_dbs:
        with shelve.open(cat_db.rsplit('.', 1)[0], 'r') as db:
            cat = db['cat']
            kfolds = db['data']['kfolds']
            bds_arr = db['calib']['kf_01']['bds_arr']
            best_prms_labs = db['calib']['kf_01']['use_prms_labs']

            out_dir = db['data']['dirs_dict']['main']
            out_dir = os.path.join(out_dir, r'05_kfolds_perf')
            if not os.path.exists(out_dir):
                try:
                    os.mkdir(out_dir)
                except:
                    pass

            best_prms_list = []
            for i in range(1, kfolds + 1):
                best_prms_list.append(db['calib'][f'kf_{i:02d}']['opt_prms'])

        _kfold_best_prms(
            cat, kfolds, best_prms_list, best_prms_labs, bds_arr, out_dir)
    return


def _kfold_best_prms(
        cat, kfolds, best_prms_list, best_prms_labs, bounds_arr, out_dir):

    plt.figure(figsize=(max(20, best_prms_list[0].shape[0]), 12))
    tick_font_size = 10
    best_params_arr = np.array(best_prms_list)
    norm_pop = best_params_arr.copy()

    stats_cols = ['min', 'max', 'mean', 'stdev', 'min_bd', 'max_bd']
    n_stats_cols = len(stats_cols)

    best_params_arr = (
        (best_params_arr * (bounds_arr[:, 1] - bounds_arr[:, 0])) +
        bounds_arr[:, 0])

    n_params = bounds_arr.shape[0]
    curr_min = best_params_arr.min(axis=0)
    curr_max = best_params_arr.max(axis=0)
    curr_mean = best_params_arr.mean(axis=0)
    curr_stdev = best_params_arr.std(axis=0)
    min_opt_bounds = bounds_arr.min(axis=1)
    max_opt_bounds = bounds_arr.max(axis=1)
    xx, yy = np.meshgrid(np.arange(-0.5, n_params, 1),
                         np.arange(-0.5, n_stats_cols, 1))

    stats_arr = np.vstack([curr_min,
                           curr_max,
                           curr_mean,
                           curr_stdev,
                           min_opt_bounds,
                           max_opt_bounds])

    stats_ax = plt.subplot2grid((4, 1), (0, 0), rowspan=1, colspan=1)
    stats_ax.pcolormesh(xx,
                        yy,
                        stats_arr,
                        cmap=plt.get_cmap('Blues'),
                        vmin=0.0,
                        vmax=1e30)

    stats_xx, stats_yy = np.meshgrid(np.arange(n_stats_cols),
                                     np.arange(n_params))
    stats_xx = stats_xx.ravel()
    stats_yy = stats_yy.ravel()

    [stats_ax.text(
        stats_yy[i],
        stats_xx[i],
        (f'{stats_arr[stats_xx[i], stats_yy[i]]:0.4f}').rstrip('0'),
        va='center',
        ha='center')
     for i in range(int(n_stats_cols * n_params))]

    stats_ax.set_xticks(list(range(0, n_params)))
    stats_ax.set_xticklabels(best_prms_labs)
    stats_ax.set_xlim(-0.5, n_params - 0.5)
    stats_ax.set_yticks(list(range(0, n_stats_cols)))
    stats_ax.set_ylim(-0.5, n_stats_cols - 0.5)
    stats_ax.set_yticklabels(stats_cols)

    stats_ax.spines['left'].set_position(('outward', 10))
    stats_ax.spines['right'].set_position(('outward', 10))
    stats_ax.spines['top'].set_position(('outward', 10))
    stats_ax.spines['bottom'].set_visible(False)

    stats_ax.set_ylabel('Statistics')

    stats_ax.tick_params(labelleft=True,
                         labelbottom=False,
                         labeltop=True,
                         labelright=True)

    stats_ax.xaxis.set_ticks_position('top')
    stats_ax.yaxis.set_ticks_position('both')

    for _tick in stats_ax.get_xticklabels():
        _tick.set_rotation(60)

    params_ax = plt.subplot2grid((4, 1),
                                 (1, 0),
                                 rowspan=3,
                                 colspan=1,
                                 sharex=stats_ax)
    plot_range = list(range(0, bounds_arr.shape[0]))
    plt_texts = []
    for i in range(kfolds):
        params_ax.plot(plot_range,
                       norm_pop[i],
                       alpha=0.85,
                       label=f'Fold no: {i}')
        for j in range(bounds_arr.shape[0]):
            _ = params_ax.text(
                plot_range[j],
                norm_pop[i, j],
                (f'{best_params_arr[i, j]:0.4f}').rstrip('0'),
                va='top',
                ha='left')
            plt_texts.append(_)

    adjust_text(plt_texts, only_move={'points': 'y', 'text': 'y'})

    params_ax.set_ylim(0., 1.)
    params_ax.set_xticks(list(range(best_params_arr.shape[1])))
    params_ax.set_xticklabels(best_prms_labs)
    params_ax.set_xlim(-0.5, n_params - 0.5)
    params_ax.set_ylabel('Normalized value')
    params_ax.grid()
    params_ax.legend(framealpha=0.5)

    params_ax.tick_params(labelleft=True,
                          labelbottom=True,
                          labeltop=False,
                          labelright=True)
    params_ax.yaxis.set_ticks_position('both')

    for _tick in params_ax.get_xticklabels():
        _tick.set_rotation(60)

    title_str = 'Comparison of kfold best parameters'
    plt.suptitle(title_str, size=tick_font_size + 10)
    plt.subplots_adjust(hspace=0.15)

    out_params_fig_loc = os.path.join(
        out_dir, f'kfolds_prms_compare_{cat}.png')
    plt.savefig(out_params_fig_loc, bbox='tight_layout')
    plt.close()
    return


def _get_daily_annual_cycle(col_ser):
    assert isinstance(col_ser, pd.Series), 'Expected a pd.Series object!'
    col_ser.dropna(inplace=True)

    # For each day of a year, get the days for all year and average them
    # the annual cycle is the average value and is used for every doy of
    # all years
    for month in range(1, 13):
        for dom in range(1, 32):
            month_idxs = col_ser.index.month == month
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


def get_daily_annual_cycle(in_data_df, n_cpus=1):
    annual_cycle_df = pd.DataFrame(index=in_data_df.index,
                                   columns=in_data_df.columns,
                                   dtype=float)

    cat_ser_gen = (in_data_df[col] for col in in_data_df.columns)
    if n_cpus > 1:
        mp_pool = ProcessPool(n_cpus)
        mp_pool.restart(True)
        try:
            ann_cycs = list(mp_pool.uimap(_get_daily_annual_cycle,
                                          cat_ser_gen))

            for col_ser in ann_cycs:
                annual_cycle_df.update(col_ser)

            mp_pool.clear()
        except Exception as msg:
            mp_pool.close()
            mp_pool.join()
            print('Error in get_daily_annual_cycle:', msg)
    else:
        for col_ser in cat_ser_gen:
            annual_cycle_df.update(_get_daily_annual_cycle(col_ser))

    return annual_cycle_df


if __name__ == '__main__':
    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main_dir = Path(r'')

    os.chdir(main_dir)

    STOP = timeit.default_timer()  # Ending time
    print(('\n\a\a\a Done with everything on %s. Total run time was'
           ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
