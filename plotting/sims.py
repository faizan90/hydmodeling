# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Faizan Anwar, IWS Uni-Stuttgart)s

Read the dem_net shapefile created by TauDEM
Get only those lines that are needed to model
catchments in HBV or similar

NOT PROUD OF THIS AT ALL
"""

import os
import pickle
from pathlib import Path

import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib.font_manager import FontProperties as f_props

# from hbv_multi_cat_multi_core import (loop_HBV_py,
#                                       get_ns_py,
#                                       get_ln_ns_py,
#                                       get_kge_py,
#                                       lin_regsn_py,
#                                       get_pearson_correl_py)

plt.ioff()

# ogr.UseExceptions()


def plot_pop(opt_params_dict):
    pop = opt_params_dict['pop']
    bounds_arr = opt_params_dict['bds_arr']
    out_dir = opt_params_dict['out_dir']
    param_syms = opt_params_dict['use_prms_labs']
    out_pref = opt_params_dict['out_pref']
    out_suff = opt_params_dict['out_suff']
    plt.figure(figsize=(max(35, bounds_arr.shape[0]), 13))
    tick_font_size = 10

    stats_cols = ['min', 'max', 'mean', 'stdev', 'min_bd', 'max_bd']
    n_stats_cols = len(stats_cols)

    norm_pop = pop.copy()
    for i in range(pop.shape[0]):
        for j in range(pop.shape[1]):
            pop[i, j] = (pop[i, j] * (bounds_arr[j, 1] - bounds_arr[j, 0])) + bounds_arr[j, 0]
            if not np.isfinite(pop[i, j]):
                pop[i, j] = bounds_arr[j, 0]

    n_params = bounds_arr.shape[0]

    curr_min = pop.min(axis=0)
    curr_max = pop.max(axis=0)
    curr_mean = pop.mean(axis=0)
    curr_stdev = pop.std(axis=0)
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
                        vmin=-np.inf,
                        vmax=np.inf)

    stats_xx, stats_yy = np.meshgrid(np.arange(n_stats_cols),
                                     np.arange(n_params))
    stats_xx = stats_xx.ravel()
    stats_yy = stats_yy.ravel()

    [stats_ax.text(stats_yy[i],
                   stats_xx[i],
                   ('%3.4f' % stats_arr[stats_xx[i], stats_yy[i]]).rstrip('0'),
                   va='center',
                   ha='center')
     for i in range(int(n_stats_cols * n_params))]

    stats_ax.set_xticks(list(range(0, n_params)))
    stats_ax.set_xticklabels(param_syms)
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
    plot_range = list(range(n_params))
    for i in range(pop.shape[0]):
        params_ax.plot(plot_range,
                       norm_pop[i],
                       alpha=0.1,
                       color='k')  # ,
#                        label='Vec no: %d' % i)

    params_ax.set_ylim(0., 1.)
    params_ax.set_xticks(plot_range)
    params_ax.set_xticklabels(param_syms)
    params_ax.set_xlim(-0.5, n_params - 0.5)
    params_ax.set_ylabel('Normalized value')
    params_ax.grid()

    params_ax.tick_params(labelleft=True,
                          labelbottom=True,
                          labeltop=False,
                          labelright=True)
    params_ax.yaxis.set_ticks_position('both')

    for _tick in params_ax.get_xticklabels():
        _tick.set_rotation(60)

    title_str = 'Distributed HBV parameters - DE final population (n=%d)' % pop.shape[0]
    plt.suptitle(title_str, size=tick_font_size + 10)
    plt.subplots_adjust(hspace=0.15)

    plt.savefig(str(Path(out_dir, f'k_{out_pref}_hbv_pop_{out_suff}.png')), bbox_inches='tight')
    plt.close()

    * _, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    obj_vals = opt_params_dict['pop_curr_obj_vals']
    n_vals = float(obj_vals.shape[0])

    probs = 1 - (np.arange(1.0, n_vals + 1) / (n_vals + 1))

    ax1.set_title(('Final population objective function distribution\n'
                   'Min. obj.: %0.4f, max. obj.: %0.4f') %
                   (obj_vals.min(), obj_vals.max()))
    ax1.plot(np.sort(obj_vals), probs, marker='o', alpha=0.8)
    ax1.set_ylabel('Non-exceedence Probability (-)')
    ax1.set_xlim(ax1.set_xlim()[1], ax1.set_xlim()[0])
    ax1.grid()

    ax2.hist(obj_vals, bins=20)
    ax2.set_xlabel('Obj. ftn. values (-)')
    ax2.set_ylabel('Frequency (-)')
    plt.savefig(str(Path(out_dir, f'k_{out_pref}_hbv_obj_cdf_{out_suff}.png')), bbox_inches='tight')

    return


def plot_hbv(opt_params_dict, plot_simple_opt_flag, plot_wat_bal_flag):
#     n_recs = opt_params_dict['n_recs']
#     conv_ratio_arr = opt_params_dict['conv_ratio_arr']
    params_arr = opt_params_dict['params']
#     bounds_arr = opt_params_dict['bds_arr']

#     ini_arr = opt_params_dict['ini_arr']
#     temp_arr = opt_params_dict['temp_arr']
#     prec_arr = opt_params_dict['prec_arr']
#     pet_arr = opt_params_dict['pet_arr']
#     q_act_arr = opt_params_dict['q_arr']
#
#     off_idx = opt_params_dict['off_idx']
#
    out_dir = opt_params_dict['out_dir']
    out_suff = opt_params_dict['out_suff']
    out_pref = opt_params_dict['out_pref']

    ext_shape = opt_params_dict['shape']
    cat_row_idxs = opt_params_dict['rows']
    cat_col_idxs = opt_params_dict['cols']

    n_cells = params_arr.shape[0]
    n_prms = params_arr.shape[1]

    assert cat_row_idxs.shape[0] == n_cells
    assert cat_col_idxs.shape[0] == n_cells

#     water_bal_step_size = opt_params_dict['water_bal_step_size']
#
#     if 'use_obs_flow_flag' in opt_params_dict:
#         use_obs_flow_flag = bool(opt_params_dict['use_obs_flow_flag'])
#     else:
#         use_obs_flow_flag = False
#
#     all_output = loop_HBV_py(n_recs,
#                              conv_ratio_arr,
#                              params_arr,
#                              ini_arr,
#                              temp_arr,
#                              prec_arr,
#                              pet_arr)
#
#     snow_arr = all_output[:, 0]
#     liqu_arr = all_output[:, 1]
#     sm_arr = all_output[:, 2]
#     tot_run_arr = all_output[:, 3]
#     evap_arr = all_output[:, 4]
#     comb_run_arr = all_output[:, 5]
#     q_sim_arr = all_output[:, 6]
#     ur_sto_arr = all_output[:, 7]
#     ur_run_uu = all_output[:, 8]
#     ur_run_ul = all_output[:, 9]
#     ur_to_lr_run = all_output[:, 10]
#     lr_sto_arr = all_output[:, 11]
#     #lr_run_arr = all_output[:, 12]
#
#     extra_us_inflow_flag = 'extra_us_inflow' in opt_params_dict
    param_syms = ['tt',
                  'cm',
                  'pcm',
                  'fc',
                  'beta',
                  'pwp',
                  'ur_thr',
                  'K_uu',
                  'K_ul',
                  'K_d',
                  'K_ll']
#
#     q_sim_arr = (q_sim_arr).copy(order='C')
#     q_act_arr_diff = q_act_arr.copy()
#     if extra_us_inflow_flag:
#         extra_us_inflow = opt_params_dict['extra_us_inflow']
#         q_sim_arr = q_sim_arr + extra_us_inflow
#         # FIXME: what to do with this
#         q_act_arr_diff = q_act_arr - extra_us_inflow
#
#     _cols = ['temp', 'prec', 'pet', 'snow', 'liqu', 'sm', 'tot_run', 'evap',
#              'comb_run', 'q_sim', 'ur_sto', 'ur_run_uu', 'ur_run_ul',
#              'ur_to_lr_run', 'lr_sto', 'lr_run']
#
#     input_arr = np.stack((temp_arr, prec_arr, pet_arr), axis=1)
#     sim_df = pd.DataFrame(data=np.concatenate((input_arr, all_output), axis=1),
#                           columns=_cols, dtype=float)
#     sim_df['q_sim'][:] = q_sim_arr
#     sim_df.to_csv(os.path.join(out_dir, 'hbv_sim_in_out.csv'), sep=';')
#
#     ns = get_ns_py(q_act_arr, q_sim_arr, off_idx)
#     ln_ns = get_ln_ns_py(q_act_arr, q_sim_arr, off_idx)
#     kge = get_kge_py(q_act_arr, q_sim_arr, off_idx)
#     q_correl = get_pearson_correl_py(q_act_arr, q_sim_arr)
#
#     (tt,
#      c_melt,
#      fc,
#      beta,
#      pwp,
#      ur_thresh,
#      k_uu,
#      k_ul,
#      k_d,
#      k_ll) = params_arr[:10]

    def plot_params():
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        out_fig_loc = os.path.join(
            out_dir, f'{out_pref}_HBV_prms_plot_{out_suff}_%s.png')

        for prm_i in range(n_prms):
            plt.figure()
            plot_arr = np.full(ext_shape, np.nan)
            plot_arr[cat_row_idxs, cat_col_idxs] = params_arr[:, prm_i]
#             plot_arr = np.ma.masked_where(np.isnan(plot_arr), plot_arr)
            plt.imshow(plot_arr,
                       origin='lower',)
#                        vmin=bounds_arr[prm_i, 0],
#                        vmax=bounds_arr[prm_i, 1])
            plt.title('%s (mean, var: %0.4f, %0.4f)' %
                      (param_syms[prm_i],
                       params_arr[:, prm_i].mean(),
                       params_arr[:, prm_i].var()))

            plt.colorbar()
            plt.xlim(cat_col_idxs.min() - 1, cat_col_idxs.max() + 1)
            plt.ylim(cat_row_idxs.min() - 1, cat_row_idxs.max() + 1)

            plt.savefig(out_fig_loc % param_syms[prm_i],
                        bbox_inches='tight')

            plt.close()

        _aux_var_keys = ['aspect_scale_arr',
                         'slope_scale_arr',
                         'aspect_slope_scale_arr']

        for _key in _aux_var_keys:
            if _key not in opt_params_dict:
                continue

            plt.figure()
            plot_arr = np.full(ext_shape, np.nan)
            _aux_vals = opt_params_dict[_key]
            plot_arr[cat_row_idxs, cat_col_idxs] = _aux_vals
            plt.imshow(plot_arr,
                       origin='lower',
                       vmin=0.0,
                       vmax=1.0)
            plt.title('%s (mean, var: %0.4f, %0.4f)' % (_key,
                                                        _aux_vals.mean(),
                                                        _aux_vals.var()))
            plt.colorbar()
            plt.xlim(cat_col_idxs.min() - 1, cat_col_idxs.max() + 1)
            plt.ylim(cat_row_idxs.min() - 1, cat_row_idxs.max() + 1)

            plt.savefig(out_fig_loc % _key, bbox_inches='tight')
            plt.close()
        return

    plot_params()
#     def save_simple_opt(out_dir, out_pref, out_suff):
#         '''Save the output of the optimize function
#         '''
#         assert np.any(params_arr), 'HBV parameters are not defined!'
#         assert q_act_arr.shape[0] == q_sim_arr.shape[0], \
#             'Original and simulated discharge have unequal steps!'
#
#         if not os.path.exists(out_dir):
#             os.mkdir(out_dir)
#
#         out_fig_loc = os.path.join(out_dir,
#                                    '%s_HBV_model_plot_%s.png' % (out_pref,
#                                                                  out_suff))
#         out_params_loc = os.path.join(
#             out_dir,
#             '%s_HBV_model_params_%s.csv' % (out_pref, out_suff))
#
#         out_labs = []
#         out_labs.extend(param_syms)
#
#         if 'route_labs' in opt_params_dict:
#             out_labs.extend(opt_params_dict['route_labs'])
#
#         out_labs.extend(['ns', 'ln_ns', 'kge', 'obj_ftn', 'p_corr'])
#
#         out_params_df = pd.DataFrame(index=out_labs, columns=['value'])
#         out_params_df['value'] = \
#             np.concatenate((params_arr,
#                             [ns, ln_ns, kge, 'nothing_yet', q_correl]))
#         out_params_df.to_csv(out_params_loc, sep=str(';'), index_label='param')
#
#         cum_q = np.cumsum(q_act_arr_diff[off_idx:] / conv_ratio_arr[off_idx:])
#         cum_q_sim = np.cumsum(comb_run_arr[off_idx:])
#
#         min_vol_diff_err = 0.5
#         max_vol_diff_err = 1.5
#
#         vol_diff_arr = cum_q_sim / cum_q
#         vol_diff_arr[vol_diff_arr < min_vol_diff_err + 0.05] = \
#             min_vol_diff_err + 0.05
#         vol_diff_arr[vol_diff_arr > max_vol_diff_err - 0.05] = \
#             max_vol_diff_err - 0.05
#         vol_diff_arr = np.concatenate((np.full(shape=(off_idx - 1),
#                                                fill_value=np.nan),
#                                        vol_diff_arr))
#
#         bal_idxs = [0]
#         bal_idxs.extend(list(range(off_idx, n_recs, water_bal_step_size)))
#         act_bal_arr = []
#         sim_bal_arr = []
#
#         prec_sum_arr = []
#
#         min_vol_err_wET = 0.5
#         max_vol_err_wET = 1.5
#
#         for i in range(1, len(bal_idxs) - 1):
#             # ET accounted for
#             _curr_q_act = q_act_arr_diff[bal_idxs[i]:bal_idxs[i + 1]]
#             _curr_conv_rat = conv_ratio_arr[bal_idxs[i]:bal_idxs[i + 1]]
#             _curr_q_act_sum = np.sum(_curr_q_act / _curr_conv_rat)
#
#             _curr_prec_sum = np.sum(prec_arr[bal_idxs[i]:bal_idxs[i + 1]])
#             _curr_evap_sum = np.sum(evap_arr[bal_idxs[i]:bal_idxs[i + 1]])
#
#             _curr_comb_sum = np.sum(comb_run_arr[bal_idxs[i]:bal_idxs[i + 1]])
#
#             act_bal_arr.append(_curr_q_act_sum /
#                                (_curr_prec_sum - _curr_evap_sum))
#
#             sim_bal_arr.append(_curr_comb_sum /
#                                (_curr_prec_sum - _curr_evap_sum))
#
#             prec_sum_arr.append(_curr_prec_sum)
#
#         act_bal_arr = np.array(act_bal_arr)
#
#         act_bal_arr[act_bal_arr < min_vol_err_wET] = min_vol_err_wET
#         act_bal_arr[act_bal_arr > max_vol_err_wET] = max_vol_err_wET
#
#         act_bal_arr = np.concatenate(([np.nan, np.nan], act_bal_arr), axis=0)
#         sim_bal_arr = np.concatenate(([np.nan, np.nan], sim_bal_arr), axis=0)
#
#         prec_sum_arr = np.concatenate(([np.nan, np.nan], prec_sum_arr), axis=0)
#
#         steps_range = np.arange(0., n_recs)
#         temp_trend, temp_corr, temp_slope, temp_intercept = \
#             lin_regsn_py(steps_range, temp_arr)
#
#         temp_stats_str = ''
#         temp_stats_str += 'correlation: %0.5f, ' % temp_corr
#         temp_stats_str += 'slope: %0.5f' % temp_slope
#
#         pet_trend, pet_corr, pet_slope, pet_intercept = \
#             lin_regsn_py(steps_range, pet_arr)
#
#         pet_stats_str = ''
#         pet_stats_str += 'PET correlation: %0.5f, ' % pet_corr
#         pet_stats_str += 'PET slope: %0.5f, ' % pet_slope
#
#         et_trend, et_corr, et_slope, et_intercept = \
#             lin_regsn_py(steps_range, evap_arr.copy(order='C'))
#
#         pet_stats_str += 'ET correlation: %0.5f, ' % et_corr
#         pet_stats_str += 'ET slope: %0.5f' % et_slope
#
#         plt.figure(figsize=(11, 25), dpi=250)
#         t_rows = 16
#         t_cols = 1
#         font_size = 6
#
#         plt.suptitle('HBV Flow Simulation')
#
#         i = 0
#         params_ax = plt.subplot2grid((t_rows, t_cols),
#                                      (i, 0),
#                                      rowspan=1,
#                                      colspan=1)
#         i += 1
#         vol_err_ax = plt.subplot2grid((t_rows, t_cols),
#                                       (i, 0),
#                                       rowspan=1,
#                                       colspan=1)
#         i += 1
#         discharge_ax = plt.subplot2grid((t_rows, t_cols),
#                                         (i, 0),
#                                         rowspan=1,
#                                         colspan=1)
#         i += 1
#         balance_ax = plt.subplot2grid((t_rows, t_cols),
#                                       (i, 0),
#                                       rowspan=1,
#                                       colspan=1)
#         i += 1
#         prec_ax = plt.subplot2grid((t_rows, t_cols),
#                                    (i, 0),
#                                    rowspan=1,
#                                    colspan=1)
#         i += 1
#         liqu_ax = plt.subplot2grid((t_rows, t_cols),
#                                    (i, 0),
#                                    rowspan=1,
#                                    colspan=1)
#         i += 1
#         snow_ax = plt.subplot2grid((t_rows, t_cols),
#                                    (i, 0),
#                                    rowspan=1,
#                                    colspan=1)
#         i += 1
#         temp_ax = plt.subplot2grid((t_rows, t_cols),
#                                    (i, 0),
#                                    rowspan=1,
#                                    colspan=1)
#         i += 1
#         pet_ax = plt.subplot2grid((t_rows, t_cols),
#                                   (i, 0),
#                                   rowspan=1,
#                                   colspan=1)
#         i += 1
#         sm_ax = plt.subplot2grid((t_rows, t_cols),
#                                  (i, 0),
#                                  rowspan=1,
#                                  colspan=1)
#         i += 1
#         tot_run_ax = plt.subplot2grid((t_rows, t_cols),
#                                       (i, 0),
#                                       rowspan=1,
#                                       colspan=1)
#         i += 1
#         u_res_sto_ax = plt.subplot2grid((t_rows, t_cols),
#                                         (i, 0),
#                                         rowspan=1,
#                                         colspan=1)
#         i += 1
#         ur_run_uu_ax = plt.subplot2grid((t_rows, t_cols),
#                                         (i, 0),
#                                         rowspan=1,
#                                         colspan=1)
#         i += 1
#         ur_run_ul_ax = plt.subplot2grid((t_rows, t_cols),
#                                         (i, 0),
#                                         rowspan=1,
#                                         colspan=1)
#         i += 1
#         ur_to_lr_run_ax = plt.subplot2grid((t_rows, t_cols),
#                                            (i, 0),
#                                            rowspan=1,
#                                            colspan=1)
#         i += 1
#         l_res_sto_ax = plt.subplot2grid((t_rows, t_cols),
#                                         (i, 0),
#                                         rowspan=1,
#                                         colspan=1)
#
#         bar_x = np.arange(0, n_recs, 1)
#         discharge_ax.plot(q_act_arr, 'r-', label='Actual Flow', lw=0.8)
#         discharge_ax.plot(q_sim_arr,
#                           'b-',
#                           label='Simulated Flow',
#                           lw=0.5,
#                           alpha=0.5)
#
#         vol_err_ax.axhline(1.0, color='k', lw=1)
#         vol_err_ax.plot(vol_diff_arr,
#                         lw=0.5,
#                         label='Cumm. Runoff Error',
#                         alpha=0.95)
#         vol_err_ax.set_ylim(min_vol_diff_err, max_vol_diff_err)
#
#         scatt_size = 5
#
#         balance_ax.axhline(1.0, color='k', lw=1)
#         balance_ax.scatter(bal_idxs,
#                            act_bal_arr,
#                            c='r',
#                            marker='o',
#                            label='Actual Outflow (Q + ET)',
#                            alpha=0.6,
#                            s=scatt_size)
#         balance_ax.scatter(bal_idxs,
#                            sim_bal_arr,
#                            c='b',
#                            marker='+',
#                            label='Simulated Outflow (Q + ET)',
#                            alpha=0.6,
#                            s=scatt_size)
#         balance_ax.set_ylim(min_vol_err_wET, max_vol_err_wET)
#
#         prec_ax.bar(bar_x,
#                     prec_arr,
#                     label='Precipitation',
#                     edgecolor='none',
#                     width=1.0)
#         prec_sums_ax = prec_ax.twinx()
#         prec_sums_ax.scatter(bal_idxs,
#                              prec_sum_arr,
#                              c='b',
#                              label='Precipitation Sum',
#                              alpha=0.5,
#                              s=scatt_size)
#
#         temp_ax.plot(temp_arr, 'b-', lw=0.5, label='Temperature')
#         temp_ax.plot(temp_trend, 'b-.', lw=0.9, label='Temperature Trend')
#         temp_ax.text(temp_ax.get_xlim()[1] * 0.02,
#                      (temp_ax.get_ylim()[0] +
#                       (temp_ax.get_ylim()[1] - temp_ax.get_ylim()[0]) * 0.1),
#                      temp_stats_str,
#                      fontsize=font_size,
#                      bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 2})
#
#         pet_ax.plot(pet_arr, 'r-', lw=0.8,
#                     label='Potential Evapotranspiration')
#         pet_ax.plot(evap_arr,
#                     'b-',
#                     lw=0.5,
#                     label='Evapotranspiration',
#                     alpha=0.5)
#         pet_ax.plot(pet_trend, 'r-.', lw=0.9, label='PET Trend')
#         pet_ax.plot(et_trend, 'b-.', lw=0.9, label='ET Trend')
#         pet_ax.text(pet_ax.get_xlim()[1] * 0.02,
#                     (pet_ax.get_ylim()[1] -
#                      (pet_ax.get_ylim()[1] - pet_ax.get_ylim()[0]) * 0.1),
#                     pet_stats_str,
#                     fontsize=font_size,
#                     bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 2})
#
#         snow_ax.plot(snow_arr, lw=0.5, label='Snow')
#         liqu_ax.bar(bar_x,
#                     liqu_arr,
#                     width=1.0,
#                     edgecolor='none',
#                     label='Liquid Precipitation')
#         sm_ax.plot(sm_arr, lw=0.5, label='Soil Moisture')
#         tot_run_ax.bar(bar_x,
#                        tot_run_arr,
#                        width=1.0,
#                        label='Total Runoff',
#                        edgecolor='none')
#         u_res_sto_ax.plot(ur_sto_arr, lw=0.5,
#                           label='Upper Reservoir - Storage')
#         ur_run_uu_ax.bar(bar_x,
#                          ur_run_uu,
#                          label='Upper Reservoir - Quick Runoff',
#                          width=2.0,
#                          edgecolor='none')
#         ur_run_ul_ax.bar(bar_x,
#                          ur_run_ul,
#                          label='Upper Reservoir - Slow Runoff',
#                          width=1.0,
#                          edgecolor='none')
#         ur_to_lr_run_ax.bar(bar_x,
#                             ur_to_lr_run,
#                             label='Upper Reservoir  - Percolation',
#                             width=1.0,
#                             edgecolor='none')
#         l_res_sto_ax.plot(lr_sto_arr, lw=0.5,
#                           label='Lower Reservoir - Storage')
#
#         snow_ax.fill_between(bar_x, 0, snow_arr, alpha=0.3)
#         sm_ax.fill_between(bar_x, 0, sm_arr, alpha=0.3)
#         u_res_sto_ax.fill_between(bar_x, 0, ur_sto_arr, alpha=0.3)
#         l_res_sto_ax.fill_between(bar_x, 0, lr_sto_arr, alpha=0.3)
#
#         text = np.array([
#             ('Max. actual Q = %0.4f' % q_act_arr[off_idx:].max()).rstrip('0'),
#             ('Max. simulated Q = %0.4f' %
#              q_sim_arr[off_idx:].max()).rstrip('0'),
#             ('Min. actual Q = %0.4f' %
#              q_act_arr[off_idx:].min()).rstrip('0'),
#             ('Min. simulated Q = %0.4f' %
#              q_sim_arr[off_idx:].min()).rstrip('0'),
#             ('Mean actual Q = %0.4f' %
#              np.mean(q_act_arr[off_idx:])).rstrip('0'),
#             ('Mean simulated Q = %0.4f' %
#              np.mean(q_sim_arr[off_idx:])).rstrip('0'),
#             ('Max. input P = %0.4f' % prec_arr.max()).rstrip('0'),
#             ('Max. SD = %0.2f' % snow_arr[off_idx:].max()).rstrip('0'),
#             ('Max. liquid P = %0.4f' %
#              liqu_arr[off_idx:].max()).rstrip('0'),
#             ('Max. input T = %0.2f' %
#              temp_arr[off_idx:].max()).rstrip('0'),
#             ('Min. input T = %0.2f' %
#              temp_arr[off_idx:].min()).rstrip('0'),
#             ('Max. PET = %0.2f' % pet_arr[off_idx:].max()).rstrip('0'),
#             ('Max. simulated ET = %0.2f' %
#              evap_arr[off_idx:].max()).rstrip('0'),
#             ('Min. PET = %0.2f' % pet_arr[off_idx:].min()).rstrip('0'),
#             ('Min. simulated ET = %0.2f' %
#              evap_arr[off_idx:].min()).rstrip('0'),
#             ('Max. simulated SM = %0.2f' %
#              sm_arr[off_idx:].max()).rstrip('0'),
#             ('Min. simulated SM = %0.2f' %
#              sm_arr[off_idx:].min()).rstrip('0'),
#             ('Mean simulated SM = %0.2f' %
#              np.mean(sm_arr[off_idx:])).rstrip('0'),
#             'Warm up steps = %d' % off_idx,
#             'use_obs_flow_flag = %s' % use_obs_flow_flag,
#             ('NS = %0.4f' % ns).rstrip('0'),
#             ('Ln-NS = %0.4f' % ln_ns).rstrip('0'),
#             ('KG = %0.4f' % kge).rstrip('0'),
#             ('P_Corr = %0.4f' % q_correl).rstrip('0'),
#             ('TT = %0.4f' % tt).rstrip('0'),
#             ('C$_{melt}$ = %0.4f' % c_melt).rstrip('0'),
#             ('FC = %0.4f' % fc).rstrip('0'),
#             (r'$\beta$ = %0.4f' % beta).rstrip('0'),
#             ('PWP = %0.4f' % pwp).rstrip('0'),
#             ('UR$_{thresh}$ = %0.4f' % ur_thresh).rstrip('0'),
#             ('$K_{uu}$ = %0.4f' % k_uu).rstrip('0'),
#             ('$K_{ul}$ = %0.4f' % k_ul).rstrip('0'),
#             ('$K_d$ = %0.4f' % k_d).rstrip('0'),
#             ('$K_{ll}$ = %0.4f' % k_ll).rstrip('0'),
#             '', ''])
#
#         text = text.reshape(6, 6)
#         table = params_ax.table(cellText=text,
#                                 loc='center',
#                                 bbox=(0, 0, 1, 1),
#                                 cellLoc='left')
#         table.auto_set_font_size(False)
#         table.set_fontsize(font_size)
#         params_ax.set_axis_off()
#
#         leg_font = f_props()
#         leg_font.set_size(font_size - 1)
#
#         plot_axes = [discharge_ax,
#                      prec_ax,
#                      temp_ax,
#                      pet_ax,
#                      snow_ax,
#                      liqu_ax,
#                      sm_ax,
#                      tot_run_ax,
#                      u_res_sto_ax,
#                      ur_run_uu_ax,
#                      ur_run_ul_ax,
#                      ur_to_lr_run_ax,
#                      l_res_sto_ax,
#                      vol_err_ax,
#                      balance_ax]
#         for ax in plot_axes:
#             for tlab in ax.get_yticklabels():
#                 tlab.set_fontsize(font_size - 1)
#             for tlab in ax.get_xticklabels():
#                 tlab.set_fontsize(font_size - 1)
#             ax.grid()
#             leg = ax.legend(framealpha=0.7, prop=leg_font, ncol=4, loc=1)
#             leg.get_frame().set_edgecolor('black')
#             ax.set_xlim(0, discharge_ax.get_xlim()[1])
#
#         for ax in [prec_sums_ax]:
#             for tlab in ax.get_yticklabels():
#                 tlab.set_fontsize(font_size - 1)
#             for tlab in ax.get_xticklabels():
#                 tlab.set_fontsize(font_size - 1)
#             leg = ax.legend(framealpha=0.7, prop=leg_font, ncol=4, loc=2)
#             leg.get_frame().set_edgecolor('black')
#             ax.set_xlim(0, discharge_ax.get_xlim()[1])
#
#         plt.tight_layout(rect=[0, 0.01, 1, 0.95], h_pad=0.0)
#         plt.savefig(out_fig_loc, bbox='tight_layout')
#         plt.close()
#         return
#
#     def save_water_bal_opt(out_dir, out_pref, out_suff):
#         '''Save the output of the optimize function
#
#         Just the water balance part
#         '''
#
#         if not os.path.exists(out_dir):
#             os.mkdir(out_dir)
#
#         out_fig_loc = os.path.join(out_dir,
#                                    '%s_HBV_water_bal_%s.png' % (out_pref,
#                                                                 out_suff))
#
#         cum_q = np.cumsum(q_act_arr_diff[off_idx:] / conv_ratio_arr[off_idx:])
#
#         cum_q_sim = np.cumsum(comb_run_arr[off_idx:])
#
#         min_vol_diff_err = 0.5
#         max_vol_diff_err = 1.5
#
#         vol_diff_arr = cum_q_sim / cum_q
#         vol_diff_arr[vol_diff_arr < min_vol_diff_err +
#                      0.05] = min_vol_diff_err + 0.05
#         vol_diff_arr[vol_diff_arr > max_vol_diff_err -
#                      0.05] = max_vol_diff_err - 0.05
#         vol_diff_arr = np.concatenate((np.full(shape=(off_idx - 1),
#                                                fill_value=np.nan),
#                                        vol_diff_arr))
#
#         bal_idxs = [0]
#         bal_idxs.extend(list(range(off_idx, n_recs, water_bal_step_size)))
#
#         act_bal_w_et_arr = []
#         sim_bal_w_et_arr = []
#
#         act_bal_wo_et_arr = []
#         sim_bal_wo_et_arr = []
#
#         prec_sum_arr = []
#         evap_sum_arr = []
#         q_act_sum_arr = []
#         q_sim_sum_arr = []
#
#         min_vol_ratio_err = 0
#         max_vol_ratio_err = 1.5
#
#         for i in range(1, len(bal_idxs) - 1):
#             _curr_q_act = q_act_arr_diff[bal_idxs[i]:bal_idxs[i + 1]]
#             _curr_conv = conv_ratio_arr[bal_idxs[i]:bal_idxs[i + 1]]
#             _curr_q_act_sum = np.sum(_curr_q_act / _curr_conv)
#
#             _prec_sum = np.sum(prec_arr[bal_idxs[i]:bal_idxs[i + 1]])
#             _evap_sum = np.sum(evap_arr[bal_idxs[i]:bal_idxs[i + 1]])
#
#             _curr_comb_sum = np.sum(comb_run_arr[bal_idxs[i]:bal_idxs[i + 1]])
#
#             # ET accounted for
#             act_bal_w_et_arr.append(_curr_q_act_sum / (_prec_sum - _evap_sum))
#             sim_bal_w_et_arr.append(_curr_comb_sum / (_prec_sum - _evap_sum))
#
#             # ET not accounted for
#             act_bal_wo_et_arr.append(_curr_q_act_sum / _prec_sum)
#             sim_bal_wo_et_arr.append(_curr_comb_sum / _prec_sum)
#
#             prec_sum_arr.append(_prec_sum)
#             evap_sum_arr.append(_evap_sum)
#             q_act_sum_arr.append(_curr_q_act_sum)
#             q_sim_sum_arr.append(_curr_comb_sum)
#
#         act_bal_w_et_arr = np.array(act_bal_w_et_arr)
#         act_bal_wo_et_arr = np.array(act_bal_wo_et_arr)
#
#         act_bal_w_et_arr[act_bal_w_et_arr < min_vol_ratio_err] = \
#             min_vol_ratio_err
#         act_bal_w_et_arr[act_bal_w_et_arr > max_vol_ratio_err] = \
#             max_vol_ratio_err
#
#         act_bal_w_et_arr = \
#             np.concatenate(([np.nan, np.nan], act_bal_w_et_arr), axis=0)
#         sim_bal_w_et_arr = \
#             np.concatenate(([np.nan, np.nan], sim_bal_w_et_arr), axis=0)
#
#         act_bal_wo_et_arr[act_bal_wo_et_arr < min_vol_ratio_err] = \
#             min_vol_ratio_err
#         act_bal_wo_et_arr[act_bal_wo_et_arr > max_vol_ratio_err] = \
#             max_vol_ratio_err
#
#         act_bal_wo_et_arr = \
#             np.concatenate(([np.nan, np.nan], act_bal_wo_et_arr), axis=0)
#         sim_bal_wo_et_arr = \
#             np.concatenate(([np.nan, np.nan], sim_bal_wo_et_arr), axis=0)
#
#         prec_sum_arr = np.concatenate(([np.nan, np.nan], prec_sum_arr), axis=0)
#         evap_sum_arr = np.concatenate(([np.nan, np.nan], evap_sum_arr), axis=0)
#
#         q_act_sum_arr = \
#             np.concatenate(([np.nan, np.nan], q_act_sum_arr), axis=0)
#         q_sim_sum_arr = \
#             np.concatenate(([np.nan, np.nan], q_sim_sum_arr), axis=0)
#
#         plt.figure(figsize=(11, 6), dpi=150)
#         t_rows = 8
#         t_cols = 1
#
#         plt.suptitle('HBV Flow Simulation - Water Balance')
#
#         i = 0
#         params_ax = plt.subplot2grid((t_rows, t_cols),
#                                      (i, 0),
#                                      rowspan=1,
#                                      colspan=t_cols)
#         i += 1
#         vol_err_ax = plt.subplot2grid((t_rows, t_cols),
#                                       (i, 0),
#                                       rowspan=1,
#                                       colspan=t_cols)
#         i += 1
#         discharge_ax = plt.subplot2grid((t_rows, t_cols),
#                                         (i, 0),
#                                         rowspan=2,
#                                         colspan=t_cols)
#         i += 2
#         balance_ratio_ax = plt.subplot2grid((t_rows, t_cols),
#                                             (i, 0),
#                                             rowspan=2,
#                                             colspan=t_cols)
#         i += 2
#         balance_sum_ax = plt.subplot2grid((t_rows, t_cols),
#                                           (i, 0),
#                                           rowspan=2,
#                                           colspan=t_cols)
#         #i += 2
#
#         vol_err_ax.axhline(1, color='k', lw=1)
#         vol_err_ax.plot(vol_diff_arr,
#                         lw=0.5,
#                         label='Cumm. Runoff Error',
#                         alpha=0.95)
#         vol_err_ax.set_ylim(min_vol_diff_err, max_vol_diff_err)
#         vol_err_ax.set_xlim(0, vol_err_ax.get_xlim()[1])
#         vol_err_ax.set_xticklabels([])
#
#         discharge_ax.plot(q_act_arr,
#                           'r-',
#                           label='Actual Flow',
#                           lw=0.8,
#                           alpha=0.7)
#         discharge_ax.plot(q_sim_arr,
#                           'b-',
#                           label='Simulated Flow',
#                           lw=0.5,
#                           alpha=0.5)
#         discharge_ax.set_xlim(0, vol_err_ax.get_xlim()[1])
#         discharge_ax.set_xticklabels([])
#
#         scatt_size = 5
#
#         balance_ratio_ax.axhline(1, color='k', lw=1)
#         balance_ratio_ax.scatter(bal_idxs,
#                                  act_bal_w_et_arr,
#                                  marker='o',
#                                  label='Actual Outflow (Q + ET)',
#                                  alpha=0.6,
#                                  s=scatt_size)
#         balance_ratio_ax.scatter(bal_idxs, sim_bal_w_et_arr,
#                                  marker='+',
#                                  label='Simulated Outflow (Q + ET)',
#                                  alpha=0.6,
#                                  s=scatt_size)
#
#         balance_ratio_ax.scatter(bal_idxs,
#                                  act_bal_wo_et_arr,
#                                  marker='o',
#                                  label='Actual Outflow (Q)',
#                                  alpha=0.6,
#                                  s=scatt_size)
#         balance_ratio_ax.scatter(bal_idxs,
#                                  sim_bal_wo_et_arr,
#                                  marker='+',
#                                  label='Simulated Outflow (Q)',
#                                  alpha=0.6,
#                                  s=scatt_size)
#
#         balance_ratio_ax.set_xlim(0, vol_err_ax.get_xlim()[1])
#         balance_ratio_ax.set_ylim(min_vol_ratio_err, max_vol_ratio_err)
#         balance_ratio_ax.set_xticklabels([])
#
#         balance_sum_ax.scatter(bal_idxs,
#                                prec_sum_arr,
#                                alpha=0.6,
#                                marker='o',
#                                label='Precipitation Sum',
#                                s=scatt_size)
#         balance_sum_ax.scatter(bal_idxs,
#                                evap_sum_arr,
#                                alpha=0.6,
#                                marker='+',
#                                label='Evapotranspiration Sum',
#                                s=scatt_size)
#         balance_sum_ax.scatter(bal_idxs,
#                                q_act_sum_arr,
#                                alpha=0.6,
#                                marker='o',
#                                label='Actual Runoff Sum',
#                                s=scatt_size)
#         balance_sum_ax.scatter(bal_idxs,
#                                q_sim_sum_arr,
#                                alpha=0.6,
#                                marker='+',
#                                label='Simulated Runoff Sum',
#                                s=scatt_size)
#         balance_sum_ax.set_xlim(0, vol_err_ax.get_xlim()[1])
#         balance_sum_ax.set_ylim(0, balance_sum_ax.get_ylim()[1])
#
#         text = np.array([
#             ('Max. actual Q = %0.4f' % q_act_arr[off_idx:].max()).rstrip('0'),
#             ('Max. simulated Q = %0.4f' %
#              q_sim_arr[off_idx:].max()).rstrip('0'),
#             ('Min. actual Q = %0.4f' %
#              q_act_arr[off_idx:].min()).rstrip('0'),
#             ('Min. simulated Q = %0.4f' %
#              q_sim_arr[off_idx:].min()).rstrip('0'),
#             ('Mean actual Q = %0.4f' %
#              np.mean(q_act_arr[off_idx:])).rstrip('0'),
#             ('Mean simulated Q = %0.4f' %
#              np.mean(q_sim_arr[off_idx:])).rstrip('0'),
#             ('Max. input P = %0.4f' % prec_arr.max()).rstrip('0'),
#             ('Mean simulated ET = %0.4f' %
#              evap_arr[off_idx:].mean()).rstrip('0'),
#             'Warm up steps = %d' % off_idx,
#             'use_obs_flow_flag = %s' % use_obs_flow_flag,
#             ('NS = %0.4f' % ns).rstrip('0'),
#             ('Ln-NS = %0.4f' % ln_ns).rstrip('0'),
#             ('KG = %0.4f' % kge).rstrip('0'),
#             ('P_Corr = %0.4f' % q_correl).rstrip('0'),
#             ('TT = %0.4f' % tt).rstrip('0'),
#             ('C$_{melt}$ = %0.4f' % c_melt).rstrip('0'),
#             ('FC = %0.4f' % fc).rstrip('0'),
#             (r'$\beta$ = %0.4f' % beta).rstrip('0'),
#             ('PWP = %0.4f' % pwp).rstrip('0'),
#             ('UR$_{thresh}$ = %0.4f' % ur_thresh).rstrip('0'),
#             ('$K_{uu}$ = %0.4f' % k_uu).rstrip('0'),
#             ('$K_{ul}$ = %0.4f' % k_ul).rstrip('0'),
#             ('$K_d$ = %0.4f' % k_d).rstrip('0'),
#             ('$K_{ll}$ = %0.4f' % k_ll).rstrip('0')])
#
#         font_size = 5
#         text = text.reshape(3, 8)
#         table = params_ax.table(cellText=text,
#                                 loc='center',
#                                 bbox=(0, 0, 1, 1),
#                                 cellLoc='left')
#         table.auto_set_font_size(False)
#         table.set_fontsize(font_size)
#         params_ax.set_axis_off()
#
#         leg_font = f_props()
#         leg_font.set_size(font_size)
#
#         plot_axes = [discharge_ax, vol_err_ax,
#                      balance_ratio_ax, balance_sum_ax]
#         for ax in plot_axes:
#             for tlab in ax.get_yticklabels():
#                 tlab.set_fontsize(font_size)
#             ax.grid()
#             ax.set_aspect('auto')
#             leg = ax.legend(framealpha=0.7, prop=leg_font, ncol=4)
#             leg.get_frame().set_edgecolor('black')
#
#         for tlab in balance_sum_ax.get_xticklabels():
#             tlab.set_fontsize(font_size)
#
#         plt.tight_layout(rect=[0, 0.01, 1, 0.95], h_pad=0.0)
#         plt.savefig(out_fig_loc, bbox_inches='tight')
#         plt.close()
#         return
#
#     if plot_simple_opt_flag:
#         save_simple_opt(out_dir, out_pref, out_suff)
#
#     if plot_wat_bal_flag:
#         save_water_bal_opt(out_dir, out_pref, out_suff)
    return
