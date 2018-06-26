# -*- coding: utf-8 -*-
'''
Created on Oct 12, 2017

@author: Faizan Anwar, IWS Uni-Stuttgart
'''

import os
import timeit
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text

from ..models import (
    hbv_c_loop_py,
    get_ns_cy,
    get_ln_ns_cy,
    get_kge_cy,
    get_pcorr_cy)  # ,
#                                       get_ns_var_res_py,
#                                       get_ln_ns_var_res_py)


def plot_cat_k_fold_effs(
        cat_data_dict,
        cat_k_fold_params_dict):

    kfolds = cat_data_dict['kfolds']
    assert kfolds >= 1, 'kfolds can be 1 or greater only!'

    conv_ratio = cat_data_dict['conv_ratio']

    ini_arr = cat_data_dict['ini_arr']
    temp_arr = cat_data_dict['temp_arr']
    prec_arr = cat_data_dict['prec_arr']
    pet_arr = cat_data_dict['pet_arr']
    q_act_arr = cat_data_dict['q_arr']

    area_arr = cat_data_dict['area_arr']

    off_idx = cat_data_dict['off_idx']
    out_dir = cat_data_dict['out_dir']
    out_suff = cat_data_dict['out_suff']

    if 'q_cyc_arr' in cat_data_dict:
        q_cyc_arr = cat_data_dict['q_cyc_arr']
    else:
        q_cyc_arr = None

    sel_idxs_arr = np.linspace(0,
                               temp_arr.shape[1],
                               kfolds + 1,
                               dtype=np.int64,
                               endpoint=True)

    uni_sel_idxs_list = np.unique(sel_idxs_arr).tolist()
    n_sel_idxs = len(uni_sel_idxs_list)
    assert n_sel_idxs >= 2, 'kfolds too high or data points too low!'

    uni_sel_idxs_list.extend([0, uni_sel_idxs_list[-1]])

    perfo_ftns_list = [get_ns_cy,
                       get_ln_ns_cy,
                       get_kge_cy,
                       get_pcorr_cy]

    perfo_ftns_names = ['NS', 'Ln_NS', 'KGE', 'P_Corr']

    n_perfs = len(perfo_ftns_names)
    over_all_perf_arr = np.full(shape=(n_perfs, kfolds), fill_value=np.nan)
    kfold_perfos_arr = np.full(shape=(n_perfs, kfolds, kfolds),
                               fill_value=np.nan)

    if q_cyc_arr is not None:
        raise NotImplementedError
        res_perfo_ftns_list = []  # [get_ns_var_res_py,
                                    # get_ln_ns_var_res_py]
        n_cyc_perfs = len(res_perfo_ftns_list)

        over_all_res_perf_arr = np.full(shape=(n_cyc_perfs, kfolds),
                                        fill_value=np.nan)
        kfold_res_perfos_arr = np.full(shape=(n_cyc_perfs, kfolds, kfolds),
                                       fill_value=np.nan)

        over_all_cyc_perf_arr = over_all_res_perf_arr.copy()
        kfold_cyc_perfos_arr = kfold_res_perfos_arr.copy()

    for iter_no in range(kfolds):
        params_arr = cat_k_fold_params_dict[iter_no]
        for kfold_iter_no in range(n_sel_idxs + 1):
            if kfold_iter_no == kfolds:
                continue

            cst_idx = uni_sel_idxs_list[kfold_iter_no]
            cen_idx = uni_sel_idxs_list[kfold_iter_no + 1]

            all_output = hbv_c_loop_py(
                np.ascontiguousarray(temp_arr[:, cst_idx:cen_idx]),
                np.ascontiguousarray(prec_arr[:, cst_idx:cen_idx]),
                np.ascontiguousarray(pet_arr[:, cst_idx:cen_idx]),
                params_arr[0],
                ini_arr,
                area_arr,
                conv_ratio)

            q_sim_arr = all_output['qsim_arr']

            extra_us_inflow_flag = 'extra_us_inflow' in cat_data_dict

            # TODO: extra_us_inflow is derived from the first kfold params
            # and therefore affect the results, even if the catchment is
            # simulated anew with parameters from another kfold.
            # The whole k_fold plotting thing might need changing.
            # we can use cats_outflow_arr directly.
            # The calib ones can be for the bottom figure while the valid ones
            # for the top.
            if extra_us_inflow_flag:
                _ = cat_data_dict['extra_us_inflow']
                extra_us_inflow = _[cst_idx:cen_idx]
                q_sim_arr = q_sim_arr + extra_us_inflow

            curr_q_act_arr = q_act_arr[cst_idx:cen_idx]

            assert q_sim_arr.shape[0] == curr_q_act_arr.shape[0], (
                'Unequal shapes!')

            for i, perfo_ftn in enumerate(perfo_ftns_list):
                perf_val = perfo_ftn(curr_q_act_arr, q_sim_arr, off_idx)

                if kfold_iter_no < kfolds:
                    kfold_perfos_arr[i, iter_no, kfold_iter_no] = perf_val
                elif kfold_iter_no > kfolds:
                    over_all_perf_arr[i, iter_no] = perf_val
                else:
                    raise RuntimeError('Fucked up!')

            if q_cyc_arr is None:
                continue

            curr_q_cyc_arr = q_cyc_arr[cst_idx:cen_idx]
            for i, res_perfo_ftn in enumerate(res_perfo_ftns_list):
                res_perf_val = res_perfo_ftn(curr_q_act_arr,
                                             q_sim_arr,
                                             curr_q_cyc_arr,
                                             off_idx)

                cyc_perf_val = perfo_ftns_list[i](curr_q_act_arr,
                                                  curr_q_cyc_arr,
                                                  off_idx)

                if kfold_iter_no < kfolds:
                    kfold_res_perfos_arr[i,
                                         iter_no,
                                         kfold_iter_no] = res_perf_val
                    kfold_cyc_perfos_arr[i,
                                         iter_no,
                                         kfold_iter_no] = cyc_perf_val

                    if ((cyc_perf_val > kfold_perfos_arr[i,
                                                         iter_no,
                                                         kfold_iter_no]) and
                            (res_perf_val > 0)):
                        print('1 Impossible:', out_suff, i, iter_no)

                elif kfold_iter_no > kfolds:
                    over_all_res_perf_arr[i, iter_no] = res_perf_val
                    over_all_cyc_perf_arr[i, iter_no] = cyc_perf_val

                    if ((cyc_perf_val > over_all_perf_arr[i, iter_no]) and
                            (res_perf_val > 0)):
                        print('2 Impossible:', out_suff, i, iter_no)

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

        if ((q_cyc_arr is not None) and (i < n_cyc_perfs)):
            [top_ax.text(top_xs[j],
                         top_ys[j],
                         ('%0.4f\n%0.4f\n%0.4f' %
                          (over_all_perf_arr[i, j],
                           over_all_res_perf_arr[i, j],
                           over_all_cyc_perf_arr[i, j])),
                         va='center',
                         ha='center',
                         size=fts)
             for j in range(kfolds)]
        else:
            [top_ax.text(top_xs[j],
                         top_ys[j],
                         ('%0.4f' % over_all_perf_arr[i, j]).rstrip('0'),
                         va='center',
                         ha='center',
                         size=fts)
             for j in range(kfolds)]

        bot_ax = plt.subplot2grid((n_rows, n_cols), (1, i), 3, 1)
        _ps = bot_ax.pcolormesh(kfold_perfos_arr[i],
                                cmap=plt.get_cmap('Blues'),
                                vmin=0,
                                vmax=1)

        _ravel = kfold_perfos_arr[i].ravel()
        if ((q_cyc_arr is not None) and (i < n_cyc_perfs)):
            _res_ravel = kfold_res_perfos_arr[i].ravel()
            _cyc_ravel = kfold_cyc_perfos_arr[i].ravel()
            [bot_ax.text(bot_xx[j],
                         bot_yy[j],
                         ('%0.4f\n%0.4f\n%0.4f' % (_ravel[j],
                                                   _res_ravel[j],
                                                   _cyc_ravel[j])),
                         va='center',
                         ha='center',
                         size=fts)
             for j in range(kfolds ** 2)]
        else:
            [bot_ax.text(bot_xx[j],
                         bot_yy[j],
                         ('%0.4f' % _ravel[j]).rstrip('0'),
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

    title_str = ''
    title_str += '%d-folds overall/split calibration and validation results\n'
    title_str += 'with residual efficiencies w.r.t the annual cycle and\n'
    title_str += 'efficiencies w.r.t. the annual cycle for the catchment: %d'
    title_str += '\nwith %d steps per fold\n'
    plt.suptitle(title_str % (kfolds, int(out_suff), uni_sel_idxs_list[1]))
    plt.subplots_adjust(top=0.8)

    out_kfold_fig_loc = os.path.join(out_dir,
                                     'kfolds_compare_%s.png' % (out_suff))
    plt.savefig(out_kfold_fig_loc, bbox_inches='tight', dpi=200)
    plt.close()

#     plt.figure(figsize=(15, 9))
#     tick_font_size = 10
#     best_params_arr = []
#     for i in range(kfolds):
#         best_params_arr.append(cat_k_fold_params_dict[i])
#     best_params_arr = np.array(best_params_arr)
#
#     param_syms = ['TT',
#                   'C_melt',
#                   'FC',
#                   'Beta',
#                   'PWP',
#                   'ur_thresh',
#                   'K_u',
#                   'K_l',
#                   'K_d',
#                   'K_ll']
#
#     stats_cols = ['min', 'max', 'mean', 'stdev', 'min_bd', 'max_bd']
#     n_stats_cols = len(stats_cols)
#
#     bounds_arr = cat_data_dict['bds_arr']
#
#     n_params = cat_k_fold_params_dict[0].shape[0]
#     curr_min = best_params_arr.min(axis=0)
#     curr_max = best_params_arr.max(axis=0)
#     curr_mean = best_params_arr.mean(axis=0)
#     curr_stdev = best_params_arr.std(axis=0)
#     min_opt_bounds = bounds_arr.min(axis=1)
#     max_opt_bounds = bounds_arr.max(axis=1)
#     xx, yy = np.meshgrid(np.arange(-0.5, n_params, 1),
#                          np.arange(-0.5, n_stats_cols, 1))
#
#     stats_arr = np.vstack([curr_min,
#                            curr_max,
#                            curr_mean,
#                            curr_stdev,
#                            min_opt_bounds,
#                            max_opt_bounds])
#
#     stats_ax = plt.subplot2grid((4, 1), (0, 0), rowspan=1, colspan=1)
#     stats_ax.pcolormesh(xx,
#                         yy,
#                         stats_arr,
#                         cmap=plt.get_cmap('Blues'),
#                         vmin=-np.inf,
#                         vmax=np.inf)
#
#     stats_xx, stats_yy = np.meshgrid(np.arange(n_stats_cols),
#                                      np.arange(n_params))
#     stats_xx = stats_xx.ravel()
#     stats_yy = stats_yy.ravel()
#
#     [stats_ax.text(stats_yy[i],
#                    stats_xx[i],
#                    ('%3.4f' % stats_arr[stats_xx[i], stats_yy[i]]).rstrip('0'),
#                    va='center',
#                    ha='center')
#      for i in range(int(n_stats_cols * n_params))]
#
#     stats_ax.set_xticks(list(range(0, n_params)))
#     stats_ax.set_xticklabels(param_syms)
#     stats_ax.set_xlim(-0.5, n_params - 0.5)
#     stats_ax.set_yticks(list(range(0, n_stats_cols)))
#     stats_ax.set_ylim(-0.5, n_stats_cols - 0.5)
#     stats_ax.set_yticklabels(stats_cols)
#
#     stats_ax.spines['left'].set_position(('outward', 10))
#     stats_ax.spines['right'].set_position(('outward', 10))
#     stats_ax.spines['top'].set_position(('outward', 10))
#     stats_ax.spines['bottom'].set_visible(False)
#
#     stats_ax.set_ylabel('Statistics')
#
#     stats_ax.tick_params(labelleft=True,
#                          labelbottom=False,
#                          labeltop=True,
#                          labelright=True)
#
#     stats_ax.xaxis.set_ticks_position('top')
#
#     norm_pop = ((best_params_arr - bounds_arr[:, 0]) /
#                 (bounds_arr[:, 1] - bounds_arr[:, 0]))
#
#     params_ax = plt.subplot2grid((4, 1),
#                                  (1, 0),
#                                  rowspan=3,
#                                  colspan=1,
#                                  sharex=stats_ax)
#     plot_range = list(range(0, bounds_arr.shape[0]))
#     plt_texts = []
#     for i in range(kfolds):
#         params_ax.plot(plot_range,
#                        norm_pop[i],
#                        alpha=0.85,
#                        label='Fold no: %d' % i)
#         for j in range(bounds_arr.shape[0]):
#             _ = params_ax.text(plot_range[j],
#                                norm_pop[i, j],
#                                ('%3.4f' %
#                                 best_params_arr[i, j]).rstrip('0'),
#                                va='top',
#                                ha='left')
#             plt_texts.append(_)
#
#     adjust_text(plt_texts, only_move={'points': 'y', 'text': 'y'})
#
#     params_ax.set_ylim(0., 1.)
#     params_ax.set_xticks(list(range(best_params_arr.shape[1])))
#     params_ax.set_xticklabels(param_syms)
#     params_ax.set_xlim(-0.5, n_params - 0.5)
#     params_ax.set_ylabel('Normalized value')
#     params_ax.grid()
#     params_ax.legend(framealpha=0.5)
#
#     title_str = 'Comparison of best parameters'
#     plt.suptitle(title_str, size=tick_font_size + 10)
#     plt.subplots_adjust(hspace=0.15)
#
#     out_params_fig_loc = os.path.join(out_dir,
#                                       'kfolds_params_compare_%s.png' % (out_suff))
#     plt.savefig(out_params_fig_loc, bbox='tight_layout')
#     plt.close()
    return


if __name__ == '__main__':
    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main_dir = Path(r'')

    os.chdir(main_dir)

    STOP = timeit.default_timer()  # Ending time
    print(('\n\a\a\a Done with everything on %s. Total run time was'
           ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
