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

from ..misc import get_fdc, LC_CLRS, mkdir_hm, traceback_wrapper, text_sep
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
plt.rc('axes', axisbelow=True)


def cnvt_fig_path_to_csv_path(fig_path):

    bas_name = os.path.basename(fig_path).rsplit('.', 1)[0]
    dir_name = os.path.dirname(fig_path)

    assert bool(bas_name) & bool(dir_name)

    csv_name = f'text_{bas_name}.csv'
    csv_path = os.path.join(dir_name, csv_name)
    return csv_path


@traceback_wrapper
def plot_cat_discharge_errors(plot_args):

    (cat_db,) = plot_args

    save_text_flag = True

    with h5py.File(cat_db, 'r') as db:
        main_out_dir = db['data'].attrs['main']
        off_idx = db['data'].attrs['off_idx']
        kfolds = db['data'].attrs['kfolds']

        cat = db.attrs['cat']

        opt_schm = db['cdata/opt_schm_vars_dict'].attrs['opt_schm']

        if opt_schm != 'ROPE':
            raise NotImplementedError(
                f'Only configured for ROPE, got {opt_schm} instead!')

        acc_rate = 0.4  # db['cdata/opt_schm_vars_dict'].attrs['acc_rate']

        print(f'acc_rate: {acc_rate}')

    qsims_dir = os.path.join(main_out_dir, '12_discharge_sims')

    sim_types = ['ensemble', 'lo_hi', ]

    err_dirs_dict = {}

    if 'lo_hi' in sim_types:
        err_dirs_dict['lo_hi'] = os.path.join(qsims_dir , 'errors_lo_hi')

    if 'ensemble' in sim_types:
        err_dirs_dict['ensemble'] = os.path.join(
            qsims_dir , 'errors_ensemble')

#     plot_types = []
    plot_types = [
        'qe',
        'dt',
        'lc',
        'pe',
        'mp',
        'ps',
        'dm',
        're',
        'en',
        'sq',
        'pc',
        'pk',
        'ed',
        'vc',
        'vd',
        ]

    mkdir_hm(qsims_dir)

    qsim_files_patt = f'cat_{cat}_*_[0-9]*_freq_*_opt_qsims.csv'
    qsim_files = glob(os.path.join(qsims_dir, qsim_files_patt))

    if not qsim_files:
        print('Simulating discharges...')

        from .eval import plot_cat_qsims

        plot_cat_qsims(cat_db)

        qsim_files = glob(os.path.join(qsims_dir, qsim_files_patt))

    assert qsim_files

    for err_dir in err_dirs_dict.values():
        mkdir_hm(err_dir)

    parse_patt = 'cat_%d_{}_{:d}_freq_{}_opt_qsims.csv' % cat

    n_q_quants = 10
    mw_runoff_ws = 60

    n_ensembles = 3
    ensemble_lc_idxs = list(range(n_ensembles))
    ensemble_lc_labs = ['Hi. long.', 'Med. long.', 'Lo. Long.']
    ensemble_div_vals = np.linspace(
        0.0, 1.0, n_ensembles + 1, dtype=float, endpoint=True)

    assert len(ensemble_lc_labs) == n_ensembles

    plot_calib_valid_wvcb(
        qsim_files,
        parse_patt,
        cat,
        qsims_dir,
        kfolds)

#     for qsim_file in qsim_files:
#         calib_valid_lab, opt_iter, freq = search(
#             parse_patt, os.path.basename(qsim_file))
#
#         qsims_df_orig = pd.read_csv(qsim_file, sep=';').iloc[off_idx:, :]
#
#         qobs_quants_masks_dict = get_quant_masks_dict(
#             qsims_df_orig.iloc[:, 0], n_q_quants)
#
#         qobs_arr = qsims_df_orig.iloc[:, 0].values
#
#         peaks_mask = get_peaks_mask(qobs_arr)
#
#         n_vals = qobs_arr.shape[0]
#         lorenz_x_vals = np.linspace(1., n_vals + 1., n_vals) / (n_vals + 1.)
#
#         best_prm_file_name = (
#             f'best_prms_idxs_cat_{cat}_{opt_iter}_freq_{freq}.csv')
#
#         best_prm_idxs_df = pd.read_csv(
#             os.path.join(qsims_dir, best_prm_file_name), sep=';', index_col=0)
#
#         lo_hi_lc_labs = best_prm_idxs_df.columns.tolist()
#
#         eff_ftns_dict = {
#             'ns': get_ns_cy,
#             'ln_ns': get_ln_ns_cy,
#             'kge': get_kge_cy,
#             'pcorr': get_pcorr_cy,
#             }
#
#         qobs_driest = get_driest_period(qobs_arr)
#         qobs_dry_mask = get_driest_mask(qobs_arr)
#
#         for sim_type in sim_types:
#             out_dirs_dict = {
#                 'dm': os.path.join(err_dirs_dict[sim_type], 'driest_month'),
#                 'dt': os.path.join(err_dirs_dict[sim_type], 'driest_timing'),
#                 'lc': os.path.join(err_dirs_dict[sim_type], 'lorenz_curves'),
#                 'mp': os.path.join(err_dirs_dict[sim_type], 'mean_peaks'),
#                 'pe': os.path.join(err_dirs_dict[sim_type], 'peaks_eff'),
#                 'ps': os.path.join(err_dirs_dict[sim_type], 'peaks_sq_diff'),
#                 'qe': os.path.join(err_dirs_dict[sim_type], 'quant_effs'),
#                 're': os.path.join(
#                     err_dirs_dict[sim_type], 'rel_mw_runoff_errs'),
#                 'pk': os.path.join(err_dirs_dict[sim_type], 'peak_events'),
#                 }
#
#             if sim_type == 'ensemble':
#                 out_dirs_dict['en'] = os.path.join(
#                     err_dirs_dict[sim_type], 'mean_sims')
#
#                 out_dirs_dict['sq'] = os.path.join(
#                     err_dirs_dict[sim_type], 'sq_diffs_cmp')
#
#                 out_dirs_dict['pc'] = os.path.join(
#                     err_dirs_dict[sim_type], 'peak_cntmnt')
#
#                 out_dirs_dict['vc'] = os.path.join(
#                     err_dirs_dict[sim_type], 'value_cntmnt')
#
#                 out_dirs_dict['ed'] = os.path.join(
#                     err_dirs_dict[sim_type], 'events_prob_dist')
#
#                 out_dirs_dict['vd'] = os.path.join(
#                     err_dirs_dict[sim_type], 'values_prob_dist')
#
#             do_not_cmpt = True
#             for dir_key in out_dirs_dict:
#                 if dir_key not in plot_types:
#                     continue
#
#                 mkdir_hm(out_dirs_dict[dir_key])
#
#                 do_not_cmpt = False
#
#             if do_not_cmpt:
# #                 print('Nothing to compute!')
#                 continue
#
#             for kf in range(1, kfolds + 1):
#
#                 if sim_type == 'lo_hi':
#                     lc_idxs = list(map(
#                         int,
#                         best_prm_idxs_df.loc[f'kf_{kf:02d}_idxs', :].values))
#
#                     lc_labs = lo_hi_lc_labs
#
#                     qsims_df = qsims_df_orig
#
#                 elif sim_type == 'ensemble':
#                     lc_idxs = ensemble_lc_idxs
#                     lc_labs = ensemble_lc_labs
#
#                     perf_wvcb_df_name = (
#                         f'perfs_wvcbs_cat_{cat}_{calib_valid_lab}_kf_'
#                         f'{kf:02d}_{opt_iter}.csv')
#
#                     perf_wvcb_df_path = os.path.join(
#                         qsims_dir,
#                         'perf_cdfs',
#                         perf_wvcb_df_name)
#
#                     perfs_wvcbs_df = pd.read_csv(
#                         perf_wvcb_df_path,
#                         sep=text_sep,
#                         index_col=0)
#
#                     ensemble_avg_sims_df = pd.DataFrame(
#                         index=np.arange(0, qsims_df_orig.shape[0]),
#                         columns=ensemble_lc_labs,
#                         dtype=float)
#
#                     sims_min_df = ensemble_avg_sims_df.copy()
#                     sims_max_df = ensemble_avg_sims_df.copy()
#
#                     sims_mean_sq_diff = ensemble_avg_sims_df.copy()
#
#                     if 'ed' in plot_types:
#                         ensemble_peak_evts = {}
#
#                     if 'vd' in plot_types:
#                         ensemble_values = {}
#
#                     sorted_obj_vals = np.sort(perfs_wvcbs_df['obj'].values)
#
#                     min_obj_val = sorted_obj_vals[
#                         int((1 - acc_rate) * sorted_obj_vals.shape[0])]
#
#                     acc_prms_idxs = perfs_wvcbs_df['obj'].values > min_obj_val
#
#                     sorted_wvcb_vals = np.sort(
#                         perfs_wvcbs_df.loc[acc_prms_idxs, 'wvcb'].values)
#
#                     n_prm_vecs = sorted_wvcb_vals.shape[0] - 1
#
#                     for ens_i, ensemble_lc_lab in enumerate(ensemble_lc_labs):
#
#                         ge_bd = sorted_wvcb_vals[
#                             int(ensemble_div_vals[ens_i] * n_prm_vecs)]
#
#                         le_bd = sorted_wvcb_vals[
#                             int(ensemble_div_vals[ens_i + 1] * n_prm_vecs)]
#
#                         ge = perfs_wvcbs_df['wvcb'] >= ge_bd
#
#                         lt = perfs_wvcbs_df['wvcb'] < le_bd
#
#                         ens_wvcbs_idxs = perfs_wvcbs_df['wvcb'].loc[
#                             (ge & lt & acc_prms_idxs)].index
#
#                         assert ens_wvcbs_idxs.shape[0]
#
#                         ens_sim_labs = [
#                             f'kf_{kf:02d}_sim_{sim_idx:04d}'
#                             for sim_idx in ens_wvcbs_idxs]
#
#                         ens_sims_df = qsims_df_orig[ens_sim_labs]
#
#                         if 'ed' in plot_types:
#                             ensemble_peak_evts[ensemble_lc_lab] = (
#                                 ens_sims_df.values[peaks_mask, :].copy(
#                                     order='c'))
#
#                         if 'vd' in plot_types:
#                             ensemble_values[ensemble_lc_lab] = (
#                                 ens_sims_df.values)
#
#                         obs_ser = qsims_df_orig['obs'].copy()
#                         obs_ser.name = None
#
#                         ens_sims_sq_diffs_df = (
#                             ens_sims_df.subtract(obs_ser, axis=0)) ** 2
#
#                         ens_avg_sim_ser = ens_sims_df.mean(axis=1)
#                         ens_min_sim_ser = ens_sims_df.min(axis=1)
#                         ens_max_sim_ser = ens_sims_df.max(axis=1)
#
#                         ensemble_avg_sims_df[ensemble_lc_lab][:] = (
#                             ens_avg_sim_ser.values)
#
#                         sims_min_df[ensemble_lc_lab][:] = (
#                             ens_min_sim_ser.values)
#
#                         sims_max_df[ensemble_lc_lab][:] = (
#                             ens_max_sim_ser.values)
#
#                         sims_mean_sq_diff[ensemble_lc_lab][:] = (
#                             ens_sims_sq_diffs_df.mean(axis=1).values)
#
#                     assert not np.isnan(ensemble_avg_sims_df.values).sum()
#                     qsims_df = ensemble_avg_sims_df
#
#                 else:
#                     raise NotImplementedError
#
#                 q_quant_effs_dicts = []
#                 dry_timing_effs_dicts = []
#                 lorenz_y_vals_list = []
#                 peak_effs = []
#                 qsim_arrs = []
#                 driest_periods = []
#                 rel_cumm_errs = []
#
#                 if sim_type == 'ensemble':
#                     peak_cntmnt_cts = []
#                     event_probs = []
#                     value_cntmnt_cts = []
#                     value_probs = []
#
#                 for lc_idx in lc_idxs:
#                     if sim_type == 'lo_hi':
#                         qsim_lab = f'kf_{kf:02d}_sim_{lc_idx:04d}'
#
#                     elif sim_type == 'ensemble':
#                         qsim_lab = lc_labs[lc_idx]
#
#                     else:
#                         raise NotImplementedError
#
#                     qsim_arr = qsims_df.loc[:, qsim_lab].values
#
#                     q_quant_effs_dict = get_q_quant_effs_dict(
#                         qobs_arr,
#                         qsim_arr,
#                         qobs_quants_masks_dict,
#                         eff_ftns_dict)
#
#                     q_quant_effs_dicts.append(q_quant_effs_dict)
#
#                     qsim_dry_mask = get_driest_mask(qsim_arr)
#
#                     dry_timing_dict = get_timing_effs_dict(
#                         qobs_dry_mask, qsim_dry_mask, eff_ftns_dict)
#
#                     dry_timing_effs_dicts.append(dry_timing_dict)
#
#                     lorenz_y_vals = get_lorenz_arr(qobs_arr, qsim_arr)
#                     lorenz_y_vals_list.append(lorenz_y_vals)
#
#                     peak_effs_dict = get_peaks_effs_dict(
#                         qobs_arr, qsim_arr, peaks_mask, eff_ftns_dict)
#
#                     peak_effs.append(peak_effs_dict)
#
#                     qsim_arrs.append(qsim_arr)
#
#                     driest_periods.append(get_driest_period(qsim_arr))
#
#                     rel_cumm_errs.append(get_mv_mean_runoff_err_arr(
#                         qobs_arr, qsim_arr, mw_runoff_ws))
#
#                     if sim_type == 'ensemble':
#                         peak_cntmnt_cts.append(
#                             get_peak_cntmnt_cts(
#                                 qobs_arr,
#                                 sims_min_df[qsim_lab].values,
#                                 sims_max_df[qsim_lab].values,
#                                 peaks_mask))
#
#                         value_cntmnt_cts.append(
#                             get_peak_cntmnt_cts(
#                                 qobs_arr,
#                                 sims_min_df[qsim_lab].values,
#                                 sims_max_df[qsim_lab].values,
#                                 np.ones_like(peaks_mask)))
#
#                     if (sim_type == 'ensemble') and ('ed' in plot_types):
#                         event_probs.append(get_obs_probs_in_ensemble(
#                             qobs_arr[peaks_mask],
#                             ensemble_peak_evts[ensemble_lc_labs[lc_idx]]))
#
#                     if (sim_type == 'ensemble') and ('vd' in plot_types):
#                         value_probs.append(get_obs_probs_in_ensemble(
#                             qobs_arr,
#                             ensemble_values[ensemble_lc_labs[lc_idx]]))
#
#                 if 'qe' in plot_types:
#                     for eff_ftn_lab in eff_ftns_dict:
#                         out_quants_fig_name = (
#                             f'quant_effs_cat_{cat}_{calib_valid_lab}_'
#                             f'{opt_iter}_kf_{kf}_{eff_ftn_lab}.png')
#
#                         out_quants_fig_path = os.path.join(
#                             out_dirs_dict['qe'], out_quants_fig_name)
#
#                         plot_quant_effs(
#                             q_quant_effs_dicts,
#                             qobs_quants_masks_dict,
#                             eff_ftn_lab,
#                             lc_idxs,
#                             lc_labs,
#                             cat,
#                             out_quants_fig_path,
#                             save_text_flag)
#
#                 if 'dt' in plot_types:
#                     out_driest_fig_name = (
#                         f'driest_timing_effs_cat_{cat}_{calib_valid_lab}_'
#                         f'{opt_iter}_kf_{kf}.png')
#
#                     out_driest_fig_path = os.path.join(
#                         out_dirs_dict['dt'], out_driest_fig_name)
#
#                     plot_driest_timing_effs(
#                         dry_timing_effs_dicts,
#                         lc_idxs,
#                         lc_labs,
#                         cat,
#                         out_driest_fig_path,
#                         save_text_flag)
#
#                 if 'lc' in plot_types:
#                     out_lorenz_name = (
#                         f'lorenz_curves_cat_{cat}_{calib_valid_lab}_'
#                         f'{opt_iter}_kf_{kf}.png')
#
#                     out_lorenz_path = os.path.join(
#                         out_dirs_dict['lc'], out_lorenz_name)
#
#                     plot_lorenz_arr(
#                         lorenz_x_vals,
#                         lorenz_y_vals_list,
#                         lc_idxs,
#                         lc_labs,
#                         cat,
#                         out_lorenz_path)
#
#                 if 'pe' in plot_types:
#                     out_peak_effs_name = (
#                         f'peak_effs_cat_{cat}_{calib_valid_lab}_'
#                         f'{opt_iter}_kf_{kf}.png')
#
#                     out_peaks_path = os.path.join(
#                         out_dirs_dict['pe'], out_peak_effs_name)
#
#                     plot_peak_effs(
#                         peak_effs,
#                         lc_idxs,
#                         lc_labs,
#                         cat,
#                         out_peaks_path,
#                         save_text_flag)
#
#                 if 'mp' in plot_types:
#                     out_mean_peaks_name = (
#                         f'mean_peaks_comp_cat_{cat}_{calib_valid_lab}_'
#                         f'{opt_iter}_kf_{kf}.png')
#
#                     out_mean_peaks_path = os.path.join(
#                         out_dirs_dict['mp'], out_mean_peaks_name)
#
#                     plot_mean_peaks(
#                         peaks_mask,
#                         qobs_arr,
#                         qsim_arrs,
#                         lc_idxs,
#                         lc_labs,
#                         cat,
#                         out_mean_peaks_path,
#                         save_text_flag)
#
#                 if 'ps' in plot_types:
#                     out_peaks_sq_diff_name = (
#                         f'peaks_sq_diff_comp_cat_{cat}_{calib_valid_lab}_'
#                         f'{opt_iter}_kf_{kf}.png')
#
#                     out_peaks_sq_diff_path = os.path.join(
#                         out_dirs_dict['ps'], out_peaks_sq_diff_name)
#
#                     plot_peaks_sq_err(
#                         peaks_mask,
#                         qobs_arr,
#                         qsim_arrs,
#                         lc_idxs,
#                         lc_labs,
#                         cat,
#                         out_peaks_sq_diff_path,
#                         save_text_flag)
#
#                 if 'dm' in plot_types:
#                     out_driest_name = (
#                         f'driest_month_comp_cat_{cat}_{calib_valid_lab}_'
#                         f'{opt_iter}_kf_{kf}.png')
#
#                     out_driest_path = os.path.join(
#                         out_dirs_dict['dm'], out_driest_name)
#
#                     plot_driest_periods(
#                         qobs_driest,
#                         driest_periods,
#                         lc_labs,
#                         lc_idxs,
#                         cat,
#                         off_idx,
#                         out_driest_path,
#                         save_text_flag)
#
#                 if 're' in plot_types:
#                     out_rel_runoff_err_name = (
#                         f'rel_mv_err_cat_{cat}_{calib_valid_lab}_'
#                         f'{opt_iter}_kf_{kf}.png')
#                     out_rel_runoff_err_path = os.path.join(
#                         out_dirs_dict['re'], out_rel_runoff_err_name)
#
#                     plot_mv_runoff_errs(
#                         rel_cumm_errs,
#                         lc_labs,
#                         cat,
#                         mw_runoff_ws,
#                         out_rel_runoff_err_path)
#
#                 if (sim_type == 'lo_hi') and ('pk' in plot_types):
#                     out_peak_evts_name = (
#                         f'peak_evts_cat_{cat}_{calib_valid_lab}_'
#                         f'{opt_iter}_kf_{kf}.png')
#
#                     out_peak_evts_path = os.path.join(
#                         out_dirs_dict['pk'], out_peak_evts_name)
#
#                     plot_peak_events_seperately(
#                             cat,
#                             lc_labs,
#                             peaks_mask,
#                             qobs_arr,
#                             [qsims_df.loc[:, f'kf_{kf:02d}_sim_{lc_idx:04d}'].values
#                                 for lc_idx in lc_idxs],
#                             out_peak_evts_path)
#
#                 elif (sim_type == 'ensemble') and ('pk' in plot_types):
#                     out_peak_evts_name = (
#                         f'peak_evts_cat_{cat}_{calib_valid_lab}_'
#                         f'{opt_iter}_kf_{kf}.png')
#
#                     out_peak_evts_path = os.path.join(
#                         out_dirs_dict['pk'], out_peak_evts_name)
#
#                     plot_ensemble_peak_events_seperately(
#                         cat,
#                         lc_labs,
#                         peaks_mask,
#                         qobs_arr,
#                         qsims_df.values,
#                         sims_min_df.values,
#                         sims_max_df.values,
#                         out_peak_evts_path)
#
#                 if (sim_type == 'ensemble') and ('en' in plot_types):
#                     out_ensemble_cmp_name = (
#                         f'ensemble_means_cat_{cat}_{calib_valid_lab}_'
#                         f'{opt_iter}_kf_{kf}_idxs_%0.6d_%0.6d.png')
#
#                     out_ensemble_cmp_path = os.path.join(
#                         out_dirs_dict['en'], out_ensemble_cmp_name)
#
#                     plot_ensemble_sims(
#                         qobs_arr,
#                         qsims_df.values,
#                         sims_min_df.values,
#                         sims_max_df.values,
#                         cat,
#                         lc_labs,
#                         out_ensemble_cmp_path)
#
#                 if (sim_type == 'ensemble') and ('sq' in plot_types):
#                     out_ensemble_sq_diffs_cmp_name = (
#                         f'ensemble_mean_sq_diffs_cat_{cat}_{calib_valid_lab}_'
#                         f'{opt_iter}_kf_{kf}.png')
#
#                     out_ensemble_sq_diffs_cmp_path = os.path.join(
#                         out_dirs_dict['sq'], out_ensemble_sq_diffs_cmp_name)
#
#                     plot_ensemble_mean_sim_sq_diffs(
#                             qobs_arr,
#                             sims_mean_sq_diff.values,
#                             cat,
#                             lc_labs,
#                             out_ensemble_sq_diffs_cmp_path,
#                             save_text_flag)
#
#                 if (sim_type == 'ensemble') and ('pc' in plot_types):
#                     out_ensemble_pk_ct_name = (
#                         f'ensemble_peak_cntmnt_cts_cat_{cat}_{calib_valid_lab}_'
#                         f'{opt_iter}_kf_{kf}.png')
#
#                     out_ensemble_pk_ct_path = os.path.join(
#                         out_dirs_dict['pc'], out_ensemble_pk_ct_name)
#
#                     plot_peak_cntmnt_cts(
#                             lc_labs,
#                             cat,
#                             peak_cntmnt_cts,
#                             peaks_mask,
#                             out_ensemble_pk_ct_path,
#                             save_text_flag)
#
#                 if (sim_type == 'ensemble') and ('vc' in plot_types):
#                     out_ensemble_vl_ct_name = (
#                         f'ensemble_val_cntmnt_cts_cat_{cat}_{calib_valid_lab}_'
#                         f'{opt_iter}_kf_{kf}.png')
#
#                     out_ensemble_vl_ct_path = os.path.join(
#                         out_dirs_dict['vc'], out_ensemble_vl_ct_name)
#
#                     plot_peak_cntmnt_cts(
#                             lc_labs,
#                             cat,
#                             value_cntmnt_cts,
#                             np.ones_like(peaks_mask),
#                             out_ensemble_vl_ct_path,
#                             save_text_flag)
#
#                 if (sim_type == 'ensemble') and ('ed' in plot_types):
#                     ens_prob_ftn_cts = []
#
#                     for ens_key in ensemble_peak_evts:
#                         ens_prob_ftn_cts.append(
#                             ensemble_peak_evts[ens_key].shape[1])
#
#                     out_ensemble_ed_ct_name = (
#                         f'ensemble_peak_evts_probs_cat_{cat}_{calib_valid_lab}_'
#                         f'{opt_iter}_kf_{kf}.png')
#
#                     out_ensemble_ed_ct_path = os.path.join(
#                         out_dirs_dict['ed'], out_ensemble_ed_ct_name)
#
#                     plot_obs_probs_in_ensemble_hist(
#                         lc_labs,
#                         cat,
#                         event_probs,
#                         ens_prob_ftn_cts,
#                         out_ensemble_ed_ct_path)
#
#                 if (sim_type == 'ensemble') and ('vd' in plot_types):
#                     ens_prob_ftn_cts = []
#
#                     for ens_key in ensemble_values:
#                         ens_prob_ftn_cts.append(
#                             ensemble_values[ens_key].shape[1])
#
#                     out_ensemble_vd_ct_name = (
#                         f'ensemble_value_probs_cat_{cat}_{calib_valid_lab}_'
#                         f'{opt_iter}_kf_{kf}.png')
#
#                     out_ensemble_vd_ct_path = os.path.join(
#                         out_dirs_dict['vd'], out_ensemble_vd_ct_name)
#
#                     plot_obs_probs_in_ensemble_hist(
#                         lc_labs,
#                         cat,
#                         value_probs,
#                         ens_prob_ftn_cts,
#                         out_ensemble_vd_ct_path)

    return


def plot_calib_valid_wvcb(
        qsim_files,
        parse_patt,
        cat,
        qsims_dir,
        kfolds,
        ):

    out_dir = os.path.join(qsims_dir, 'hi_lo_calib_valid_movt')

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for qsim_file in qsim_files:
        calib_valid_lab, opt_iter, freq = search(
            parse_patt, os.path.basename(qsim_file))

        if calib_valid_lab == 'valid':
            continue

        best_prm_file_name = (
            f'best_prms_idxs_cat_{cat}_{opt_iter}_freq_{freq}.csv')

        best_prm_idxs_df = pd.read_csv(
            os.path.join(qsims_dir, best_prm_file_name), sep=';', index_col=0)

        lo_hi_lc_labs = best_prm_idxs_df.columns.tolist()

        labs = ['calib', 'valid']

        for kf in range(1, kfolds + 1):
            wvcbs_dfs = {
                lab:pd.read_csv(
                    os.path.join(qsims_dir, 'perf_cdfs',
                    f'perfs_wvcbs_cat_{cat}_{lab}_kf_{kf:02d}_{opt_iter}.csv'),
                    sep=text_sep,
                    index_col=0)
                for lab in labs}

            wvcb_vals_df = pd.DataFrame(
                index=lo_hi_lc_labs,
                columns=labs,
                dtype=float)

            for lab in labs:
                wvcb_vals = wvcbs_dfs[lab].loc[
                    best_prm_idxs_df.loc[f'kf_{kf:02d}_idxs'], 'wvcb']

                wvcb_vals_df.loc[:, lab] = wvcb_vals.values

            out_txt_name = (
                f'calib_valid_wvcbs_cat_{cat}_{lab}_kf_{kf:02d}_{opt_iter}.csv')

            wvcb_vals_df.to_csv(
                os.path.join(out_dir, out_txt_name), sep=text_sep)

            plt.figure(figsize=(13, 13))

            for i, idx in enumerate(wvcb_vals_df.index):
                plt.scatter(
                    wvcb_vals_df.loc[idx, labs[0]],
                    wvcb_vals_df.loc[idx, labs[1]],
                    label=idx,
                    c=LC_CLRS[i],
                    alpha=0.9)

            plt.grid()
            plt.legend()

            plt.xlabel(labs[0])
            plt.ylabel(labs[1])

            plt.title(f'wvcb calib/valid movement for cat {cat}')

            out_fig_name = out_txt_name.rsplit('.', 1)[0] + '.png'

            plt.savefig(
                os.path.join(out_dir, out_fig_name), bbox_inches='tight')

            plt.close()
    return


def plot_obs_probs_in_ensemble_hist(
        sim_labs,
        cat,
        qobs_probs,
        ens_prob_ftn_cts,
        out_path,
        save_text_flag=False):

    n_bins = 10

    main_bins = np.linspace(0, 1.0, n_bins + 1, dtype=float, endpoint=True)

    x_labs = [
        f'{main_bins[i]:0.2f} - {main_bins[i + 1]:0.2f}'
        for i in range(n_bins)]

    x_labs = ['Too low'] + x_labs + ['Too high']

    n_bins = len(x_labs)

    ini_width = 0.8
    width_dec = 0.2

    width = ini_width

    n_vals = qobs_probs[0].shape[0]

    x_arr = np.arange(n_bins)

    plt.figure(figsize=(15, 7))

    for i, sim_lab in enumerate(sim_labs):
        inc = (1.0 / (ens_prob_ftn_cts[i] + 1.0))

        bins = main_bins.copy()
        bins[+0] = inc
        bins[-1] = 1.0 - inc

        bins = np.concatenate(([0.0], bins, [1.0]))

        try:
            # for ens_prob_ftn_cts[i] less than
            # n_bins - 2 this raises an error
            # for now I let it go

            rel_hist = np.histogram(
                qobs_probs[i], bins=bins)[0] / float(n_vals)

        except Exception as msg:
            print(f'Error while computing histogram: {msg}')
            print(f'bins: {bins}')

            continue

        rel_hist_sum = rel_hist.sum()

        assert np.isclose(rel_hist_sum, 1.0), (
            f'Relative histogram sum not close to one ({rel_hist_sum})!')

        plt.bar(
            x_arr,
            rel_hist,
            width=width,
            label=f'{sim_lab} (N={ens_prob_ftn_cts[i]})',
            alpha=0.8)

        width -= width_dec

    plt.grid()
    plt.legend()

    plt.xlabel('Bin')
    plt.ylabel('Relative frequency')

    plt.ylim(0, 1.01)

    plt.xticks(x_arr, x_labs, rotation=45)

    plt.title(
        f'Distribution of observed probabilites in ensembles for cat: {cat}\n'
        f'(N={n_vals})')

    plt.savefig(out_path, bbox_inches='tight')

    plt.close()
    return


def plot_ensemble_peak_events_seperately(
        cat,
        sim_labs,
        peaks_mask,
        qobs_arr,
        qsims_arr,
        mins_arr,
        maxs_arr,
        out_path):

    '''Ensemble type

    out_path will be suffixed with the event index
    '''

    out_path_wo_ext = str(out_path).rsplit('.', 1)[0]

    bef_steps = 10
    aft_steps = 15

    evt_idxs = np.where(peaks_mask)[0]

    for evt_idx in evt_idxs:

        plt.figure(figsize=(20, 7))

        bef_idx = max(0, evt_idx - bef_steps)
        aft_idx = min(evt_idx + aft_steps + 1, qobs_arr.shape[0])

        x_arr = np.arange(bef_idx, aft_idx)

        for i, sim_lab in enumerate(sim_labs):
            plt.fill_between(
                x_arr,
                maxs_arr[bef_idx:aft_idx, i],
                mins_arr[bef_idx:aft_idx, i],
                label=f'{sim_labs[i]} bounds',
                edgecolor=LC_CLRS[i],
                facecolor='None',
                alpha=0.8,
                linestyle='-.')

        for i, sim_lab in enumerate(sim_labs):
            plt.plot(
                x_arr,
                qsims_arr[bef_idx:aft_idx, i],
                alpha=0.8,
                color=LC_CLRS[i],
                label=f'{sim_lab} mean')

        plt.plot(
            x_arr,
            qobs_arr[bef_idx:aft_idx],
            label='Obs.',
            color='red',
            alpha=0.9)

        title_str = (
            f'Observed vs. Ensemble comparison for catchment: {cat}\n'
            f'(Event at index: {evt_idx})\n')

        plt.title(title_str)

        plt.xlabel('Time')
        plt.ylabel('Discharge ($m^3/s$)')

        plt.legend()
        plt.grid()

        plt.savefig(out_path_wo_ext + f'_{evt_idx}.png', bbox_inches='tight')

        plt.close()
    return


def plot_peak_events_seperately(
        cat,
        sim_labs,
        peaks_mask,
        qobs_arr,
        qsims_arr,
        out_path):

    '''Non-ensemble type

    out_path will be suffixed with the event index
    '''

    out_path_wo_ext = str(out_path).rsplit('.', 1)[0]

    bef_steps = 10
    aft_steps = 15

    evt_idxs = np.where(peaks_mask)[0]

    for evt_idx in evt_idxs:

        plt.figure(figsize=(20, 7))

        bef_idx = max(0, evt_idx - bef_steps)
        aft_idx = min(evt_idx + aft_steps + 1, qobs_arr.shape[0])

        x_arr = np.arange(bef_idx, aft_idx)

        for i, sim_lab in enumerate(sim_labs):
            plt.plot(
                x_arr,
                qsims_arr[i][bef_idx:aft_idx],
                alpha=0.5,
                color=LC_CLRS[i],
                label=sim_lab)

        plt.plot(
            x_arr,
            qobs_arr[bef_idx:aft_idx],
            label='Obs.',
            color='red',
            alpha=0.8)

        title_str = (
            f'Observed vs. Simulation comparison for catchment: {cat}\n'
            f'(Event at index: {evt_idx})\n')

        plt.title(title_str)

        plt.xlabel('Time')
        plt.ylabel('Discharge ($m^3/s$)')

        plt.legend()
        plt.grid()

        plt.savefig(out_path_wo_ext + f'_{evt_idx}.png', bbox_inches='tight')

        plt.close()
    return


def plot_peak_cntmnt_cts(
        sim_labs,
        cat,
        peak_cts,
        peaks_mask,
        out_path,
        save_text_flag=False):

    '''For ensembles only'''

    plt.figure(figsize=(7, 15))

    n_peaks = peaks_mask.sum()

    for i in range(len(sim_labs)):
        plt.bar(
            i,
            peak_cts[i] / float(n_peaks),
            label=sim_labs[i],
            alpha=0.9,
            color=LC_CLRS[i],
            width=0.8)

    plt.xticks(list(range(len(sim_labs))), sim_labs, rotation=45)

    plt.title(
        f'Relative peak containment counts for cat: {cat}\n'
        f'(N={peaks_mask.shape[0]}, N_peaks={n_peaks})')

    plt.ylabel('Relative frequency')

    plt.ylim(0, 1.01)

    plt.grid()

    plt.savefig(out_path, bbox_inches='tight')

    plt.close()

    if save_text_flag:
        text_ser = pd.Series(
            index=sim_labs,
            data=peak_cts,
            dtype=float)

        text_path = cnvt_fig_path_to_csv_path(out_path)
        text_ser.to_csv(text_path, sep=text_sep)
    return


def plot_ensemble_mean_sim_sq_diffs(
        qobs_arr,
        sims_mean_sq_diff_arr,
        cat,
        sim_labs,
        out_path,
        save_text_flag=False):

    sort_idxs = np.argsort(qobs_arr)
    sort_qobs_arr = qobs_arr[sort_idxs]
    sort_qsim_hi_sq_diff_arr = sims_mean_sq_diff_arr[sort_idxs, 0]
    sort_qsim_med_sq_diff_arr = sims_mean_sq_diff_arr[sort_idxs, 1]
    sort_qsim_lo_sq_diff_arr = sims_mean_sq_diff_arr[sort_idxs, 2]

    plt.figure(figsize=(20, 7))

    plt_type = 'None (line)'

    if plt_type == 'None (line)':
        plt.plot(
            sort_qobs_arr,
            sort_qsim_hi_sq_diff_arr,
            label=f'{sim_labs[0]} mean',
            color=LC_CLRS[0],
            alpha=0.8)

        plt.plot(
            sort_qobs_arr,
            sort_qsim_med_sq_diff_arr,
            label=f'{sim_labs[1]} mean',
            color=LC_CLRS[1],
            alpha=0.8)

        plt.plot(
            sort_qobs_arr,
            sort_qsim_lo_sq_diff_arr,
            label=f'{sim_labs[2]} mean',
            color=LC_CLRS[2],
            alpha=0.8)

        plt.xlabel('Discharge ($m^3/s$)')

    elif plt_type == 'ln (line)':
        plt.semilogx(
            sort_qobs_arr,
            sort_qsim_hi_sq_diff_arr,
            label=f'{sim_labs[0]} mean',
            color=LC_CLRS[0],
            alpha=0.8)

        plt.semilogx(
            sort_qobs_arr,
            sort_qsim_med_sq_diff_arr,
            label=f'{sim_labs[1]} mean',
            color=LC_CLRS[1],
            alpha=0.8)

        plt.semilogx(
            sort_qobs_arr,
            sort_qsim_lo_sq_diff_arr,
            label=f'{sim_labs[2]} mean',
            color=LC_CLRS[2],
            alpha=0.8)

        plt.xlabel('Log discharge ($m^3/s$)')

    elif plt_type == 'None (scatter)':
        plt.scatter(
            sort_qobs_arr,
            sort_qsim_hi_sq_diff_arr,
            label=f'{sim_labs[0]} mean',
            color=LC_CLRS[0],
            alpha=0.8)

        plt.scatter(
            sort_qobs_arr,
            sort_qsim_med_sq_diff_arr,
            label=f'{sim_labs[1]} mean',
            color=LC_CLRS[1],
            alpha=0.8)

        plt.scatter(
            sort_qobs_arr,
            sort_qsim_lo_sq_diff_arr,
            label=f'{sim_labs[2]} mean',
            color=LC_CLRS[2],
            alpha=0.8)

        plt.xlabel('Discharge ($m^3/s$)')

    elif plt_type == 'ln (scatter)':
        plt.scatter(
            sort_qobs_arr,
            sort_qsim_hi_sq_diff_arr,
            label=f'{sim_labs[0]} mean',
            color=LC_CLRS[0],
            alpha=0.8)

        plt.scatter(
            sort_qobs_arr,
            sort_qsim_med_sq_diff_arr,
            label=f'{sim_labs[1]} mean',
            color=LC_CLRS[1],
            alpha=0.8)

        plt.scatter(
            sort_qobs_arr,
            sort_qsim_lo_sq_diff_arr,
            label=f'{sim_labs[2]} mean',
            color=LC_CLRS[2],
            alpha=0.8)

        plt.xlabel('Discharge ($m^3/s$)')

        plt.gca().set_xscale('log')

    else:
        raise ValueError(plt_type)

    plt.grid()
    plt.legend()

    plt.ylabel('Squared difference')

    title_str = (
        f'Ensembles\' squared difference comparison for catchment: {cat}\n'
        f'plot_type: {plt_type}')

    plt.title(title_str)

    plt.savefig(str(out_path), bbox_inches='tight')

    plt.close()

    if save_text_flag:
        text_ser = pd.Series(
            data={
                'obs': sort_qobs_arr,
                sim_labs[0]: sort_qsim_hi_sq_diff_arr,
                sim_labs[1]: sort_qsim_med_sq_diff_arr,
                sim_labs[2]: sort_qsim_lo_sq_diff_arr},
            dtype=float)

        text_path = cnvt_fig_path_to_csv_path(out_path)
        text_ser.to_csv(text_path, float_format='%0.4f', sep=text_sep)
    return


def plot_ensemble_sims(
        qobs_arr,
        qsim_mean_arrs,
        qsim_min_arrs,
        qsim_max_arrs,
        cat,
        sim_labs,
        out_path):

    n_steps = qobs_arr.shape[0]
    steps_per_plot = 365
    pre_idx = 0
    lst_idx = steps_per_plot

    n_sims = qsim_mean_arrs.shape[1]

    while True:
        x_arr = np.arange(pre_idx, lst_idx)

        plt.figure(figsize=(20, 7))

        for i in range(n_sims):
            plt.fill_between(
                x_arr,
                qsim_max_arrs[pre_idx:lst_idx, i],
                qsim_min_arrs[pre_idx:lst_idx, i],
                label=f'{sim_labs[i]} bounds',
                edgecolor=LC_CLRS[i],
                facecolor='None',
                alpha=0.8,
                linestyle='-.')

        for i in range(n_sims):
            plt.plot(
                x_arr,
                qsim_mean_arrs[pre_idx:lst_idx, i],
                label=f'{sim_labs[i]} mean',
                color=LC_CLRS[i],
                alpha=0.8)

        plt.plot(
            x_arr,
            qobs_arr[pre_idx:lst_idx],
            label='Obs.',
            color='red',
            alpha=0.9)

        plt.grid()
        plt.legend()

        plt.xlabel('Time')
        plt.ylabel('Discharge ($m^3/s$)')

        title_str = (
            f'Observed vs. Ensemble comparison for catchment: {cat}\n'
            f'(pre_idx: {pre_idx}, lst_idx: {lst_idx})\n')

        plt.title(title_str)

        plt.savefig(
            str(out_path) % (pre_idx, lst_idx),
            bbox_inches='tight')

        plt.close()

        pre_idx = lst_idx
        lst_idx = min(n_steps, lst_idx + steps_per_plot)

        if lst_idx == pre_idx:
            break
    return


def plot_mv_runoff_errs(
        sim_err_arrs,
        lc_labs,
        cat,
        mw_runoff_ws,
        out_path):

    plt.figure(figsize=(20, 10))

    n_sims = len(lc_labs)

    for i in range(n_sims):
        plt.plot(
            sim_err_arrs[i],
            label=lc_labs[i],
            color=LC_CLRS[i],
            alpha=0.4)

    plt.xlabel('Time')
    plt.ylabel('Rel. mv. runoff err.')

    plt.ylim(0.0, 2.0)

    plt.grid()
    plt.legend(framealpha=0.7)

    plt.title(
        f'Relative moving window runoff error of simulations for '
        f'cat: {cat}, Window size: {mw_runoff_ws} steps')

    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    return


def plot_driest_periods(
        qobs_driest,
        driest_periods,
        lc_labs,
        lc_idxs,
        cat,
        off_idx,
        out_path,
        save_text_flag=False):

    plt.figure(figsize=(20, 10))

    x_labs = [f'Obs. ({qobs_driest[1] + off_idx})'] + [
        f'{lc_labs[i]} ({driest_periods[i][1] + off_idx})'
        for i in range(len(lc_labs))]

    x_crds = np.arange(len(x_labs))

    plt.scatter(
        0,
        qobs_driest[0],
        color='red',
        alpha=0.8,
        marker='o')

    driest_vals = [qobs_driest[0]]
    for i in range(len(lc_idxs)):
        plt.scatter(
            i + 1,
            driest_periods[i][0],
            color=LC_CLRS[i],
            marker='o',
            alpha=0.8)

        driest_vals.append(driest_periods[i][0])

    plt.xlabel('Parameter (month step index)')
    plt.ylabel('Discharge ($m^3/s$)')

    plt.xticks(x_crds, x_labs, rotation=90)

    plt.title(
        f'Mean driest month comparison for catchment: {cat} using various '
        f'parameter vector(s)')

    plt.grid()

    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

    if save_text_flag:
        text_ser = pd.Series(index=x_labs, data=driest_vals, dtype=float)
        text_path = cnvt_fig_path_to_csv_path(out_path)
        text_ser.to_csv(text_path, float_format='%0.4f', sep=text_sep)
    return


def plot_peaks_sq_err(
        peaks_mask,
        qobs_arr,
        qsim_arrs,
        lc_idxs,
        lc_labs,
        cat,
        out_path,
        save_text_flag=False):

    plt.figure(figsize=(20, 10))

    x_labs = lc_labs
    x_crds = np.arange(len(x_labs))

    sq_diffs = []
    for i in range(len(lc_idxs)):
        sq_diff = (
            (qsim_arrs[i][peaks_mask] - qobs_arr[peaks_mask]) ** 2).sum()

        plt.scatter(
            i,
            sq_diff,
            color=LC_CLRS[i],
            marker='o',
            alpha=0.8)

        sq_diffs.append(sq_diff)

    plt.ylabel('Sq. diff. sum')

    x_crds_labs = [f'{x_labs[i]}' for i in range(len(x_crds))]

    plt.xticks(x_crds, x_crds_labs, rotation=90)

    plt.title(
        f'Squared difference sum comparison for catchment: {cat} '
        f'using various parameter vector(s)\n'
        f'(N={peaks_mask.shape[0]}, N_peaks={peaks_mask.sum()})')

    plt.grid()

    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

    if save_text_flag:
        text_ser = pd.Series(index=x_labs, data=sq_diffs, dtype=float)
        text_path = cnvt_fig_path_to_csv_path(out_path)
        text_ser.to_csv(text_path, float_format='%0.4f', sep=text_sep)
    return


def plot_mean_peaks(
        peaks_mask,
        qobs_arr,
        qsim_arrs,
        lc_idxs,
        lc_labs,
        cat,
        out_path,
        save_text_flag=False):

    plt.figure(figsize=(20, 10))

    x_labs = ['Obs.'] + lc_labs
    x_crds = np.arange(len(x_labs))

    mean_obs_peak = qobs_arr[peaks_mask].mean()

    plt.scatter(
        0,
        mean_obs_peak,
        color='red',
        alpha=0.8,
        marker='o')

    mean_peaks = [mean_obs_peak]

    for i in range(len(lc_idxs)):
        mean_sim_peak = qsim_arrs[i][peaks_mask].mean()

        plt.scatter(
            i + 1,
            mean_sim_peak,
            color=LC_CLRS[i],
            marker='o',
            alpha=0.8)

        mean_peaks.append(mean_sim_peak)

    plt.ylabel('Discharge ($m^3/s$)')

    x_crds_labs = [f'{x_labs[i]}' for i in range(len(x_crds))]

    plt.xticks(x_crds, x_crds_labs, rotation=90)

    plt.title(
        f'Mean peak value comparison for catchment: {cat} using various '
        f'parameter vector(s)\n'
        f'(N={peaks_mask.shape[0]}, N_peaks={peaks_mask.sum()})')

    plt.grid()

    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

    if save_text_flag:
        text_ser = pd.Series(index=x_labs, data=mean_peaks, dtype=float)
        text_path = cnvt_fig_path_to_csv_path(out_path)
        text_ser.to_csv(text_path, float_format='%0.4f', sep=text_sep)
    return


def plot_peak_effs(
        peak_effs, lc_idxs, lc_labs, cat, out_path, save_text_flag=False):

    plt.figure(figsize=(20, 10))

    eff_ftn_labs = list(peak_effs[0].keys())

    x_crds = np.arange(len(peak_effs))

    all_eff_vals = []
    for eff_ftn_lab in eff_ftn_labs:
        eff_vals = [
            peak_effs[i][eff_ftn_lab] for i in range(len(lc_idxs))]

        plt.plot(
            x_crds,
            eff_vals,
            label=f'{eff_ftn_lab}',
            marker='o',
            alpha=0.5)

        all_eff_vals.append(eff_vals)

    plt.xlabel('Simulation - (prm. vec. idx.)')
    plt.ylabel('Efficiency')

    x_crds_labs = [
        f'{lc_labs[i]} - ({lc_idxs[i]})' for i in range(x_crds.shape[0])]

    plt.xticks(x_crds, x_crds_labs, rotation=90)

    plt.title(
        f'Model peaks efficiency for catchment: {cat} using various '
        f'parameter vector(s)')

    plt.grid()
    plt.legend(loc=0, framealpha=0.7)

    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

    if save_text_flag:
        text_df = pd.DataFrame(
            index=eff_ftn_labs,
            data=all_eff_vals,
            columns=lc_labs,
            dtype=float)

        text_path = cnvt_fig_path_to_csv_path(out_path)
        text_df.to_csv(text_path, float_format='%0.4f', sep=text_sep)
    return


def plot_lorenz_arr(
        lorenz_x_vals,
        lorenz_y_vals_list,
        sim_idxs,
        sim_labs,
        cat,
        out_path):

    plt.figure(figsize=(20, 10))

    n_sims = len(sim_idxs)

    plt.plot(
        lorenz_x_vals,
        lorenz_x_vals,
        color='red',
        label=f'equal_contrib',
        alpha=0.5)

    for i in range(n_sims):
        plt.plot(
            lorenz_x_vals,
            lorenz_y_vals_list[i],
            color=LC_CLRS[i],
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
        out_path,
        save_text_flag=False):

    plt.figure(figsize=(20, 7))

    n_quants = len(quant_effs_dict[0][eff_ftn_lab])

    bar_x_crds = (np.arange(1., n_quants + 1) / n_quants) - (0.5 / n_quants)

    quant_eff_vals = []
    for i in range(len(sim_idxs)):
        plt.plot(
            bar_x_crds,
            quant_effs_dict[i][eff_ftn_lab],
            color=LC_CLRS[i],
            label=f'{sim_labs[i]} ({sim_idxs[i]})',
            marker='o',
            alpha=0.5)

        quant_eff_vals.append(quant_effs_dict[i][eff_ftn_lab])

    plt.title(
        eff_ftn_lab.upper() +
        f' efficiency for {n_quants} quantiles for cat: {cat}')

    bar_x_crds_labs = [
        f'{bar_x_crds[i]:0.3f} - ({int(quants_masks_dict[i].sum())})'
        for i in range(n_quants)]

    plt.xticks(bar_x_crds, bar_x_crds_labs, rotation=90)

    plt.xlabel('Mean interval prob. - (N)')
    plt.ylabel(eff_ftn_lab.upper())

    plt.grid()
    plt.legend(loc=0, framealpha=0.7)

    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

    if save_text_flag:
        text_df = pd.DataFrame(
            index=sim_labs,
            data=quant_eff_vals,
            columns=bar_x_crds_labs,
            dtype=float)

        text_path = cnvt_fig_path_to_csv_path(out_path)
        text_df.to_csv(text_path, float_format='%0.4f', sep=text_sep)
    return


def plot_driest_timing_effs(
    dry_timing_dict,
    lc_idxs,
    lc_labs,
    cat,
    out_path,
    save_text_flag=False):

    plt.figure(figsize=(20, 10))

    eff_ftn_labs = list(dry_timing_dict[0].keys())

    x_crds = np.arange(len(dry_timing_dict))

    all_eff_vals = []
    for eff_ftn_lab in eff_ftn_labs:
        eff_vals = [
            dry_timing_dict[i][eff_ftn_lab] for i in range(len(lc_idxs))]

        plt.plot(
            x_crds,
            eff_vals,
            label=f'{eff_ftn_lab}',
            marker='o',
            alpha=0.5)

        all_eff_vals.append(eff_vals)

    plt.xlabel('Simulation - (prm. vec. idx.)')
    plt.ylabel('Efficiency')

    x_crds_labs = [
        f'{lc_labs[i]} - ({lc_idxs[i]})' for i in range(x_crds.shape[0])]

    plt.xticks(x_crds, x_crds_labs, rotation=90)

    plt.title(
        f'driest month indicator efficiency for catchment: {cat} '
        f'using various parameter vector(s)')

    plt.grid()
    plt.legend(loc=0, framealpha=0.7)

    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

    if save_text_flag:
        text_df = pd.DataFrame(
            index=eff_ftn_labs,
            data=all_eff_vals,
            columns=lc_labs,
            dtype=float)

        text_path = cnvt_fig_path_to_csv_path(out_path)
        text_df.to_csv(text_path, float_format='%0.4f', sep=text_sep)
    return


def get_obs_probs_in_ensemble(
        obs_peaks_arr,
        ensemble_peaks_arr):

    n_vals = obs_peaks_arr.shape[0]
    assert n_vals == ensemble_peaks_arr.shape[0]

    probs = np.arange(1, ensemble_peaks_arr.shape[1] + 1, dtype=float) / (
        ensemble_peaks_arr.shape[1] + 1)

    interp_ftn = np.interp

    obs_probs = np.full(n_vals, np.nan)

    for i in range(n_vals):
        obs_val = obs_peaks_arr[i]

        ens_vals = np.sort(ensemble_peaks_arr[i, :])

        obs_prob = interp_ftn(obs_val, ens_vals, probs, left=0.0, right=1.0)

        obs_probs[i] = obs_prob

    assert np.all(np.isfinite(obs_probs))

    return obs_probs


def get_peak_cntmnt_cts(
        obs_arr,
        min_arr,
        max_arr,
        peaks_mask):

    ge_min = (obs_arr[peaks_mask] >= min_arr[peaks_mask])
    le_max = (obs_arr[peaks_mask] <= max_arr[peaks_mask])

    within_ct = (ge_min & le_max).sum()

    return within_ct


def get_mv_mean_runoff_err_arr(ref_arr, sim_arr, ws):

    assert ref_arr.shape[0] == sim_arr.shape[0]

    n_steps = ref_arr.shape[0]

    ref_mv_mean_arr = np.zeros(n_steps - ws)
    sim_mv_mean_arr = ref_mv_mean_arr.copy()

    for i in range(n_steps - ws):
        ref_mv_mean_arr[i] = ref_arr[i:i + ws].mean()

    for i in range(n_steps - ws):
        sim_mv_mean_arr[i] = sim_arr[i:i + ws].mean()

    err_arr = (sim_mv_mean_arr / ref_mv_mean_arr)
    err_arr[err_arr > 2.00] = 2.00
    return err_arr


def get_lorenz_arr(ref_arr, sim_arr):

    sorted_abs_diffs = np.sort(((ref_arr - sim_arr) ** 2)).cumsum()
    sorted_abs_diffs = sorted_abs_diffs / sorted_abs_diffs[-1]
    return sorted_abs_diffs


def get_timing_effs_dict(qobs_flags_arr, qsim_flags_arr, eff_ftns_dict):

    timing_effs_dict = {}
    for eff_ftn_key in eff_ftns_dict:
        eff_ftn = eff_ftns_dict[eff_ftn_key]

        timing_effs_dict[eff_ftn_key] = eff_ftn(
            qobs_flags_arr.astype(float), qsim_flags_arr.astype(float), 0)

    return timing_effs_dict


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


def get_driest_period(in_arr):

    ws = 30
    n_steps = in_arr.shape[0]

    assert n_steps >= ws

    runn_mean_arr = np.full(n_steps - ws, np.nan)

    for i in range(n_steps - ws):
        runn_mean_arr[i] = in_arr[i:i + ws].mean()

    min_mean_idx = np.argmin(runn_mean_arr)
    return (runn_mean_arr[min_mean_idx], min_mean_idx + int(0.5 * ws))


def get_peaks_mask(in_arr):

    rising = in_arr[1:] - in_arr[:-1] > 0
    recing = in_arr[1:-1] - in_arr[2:] > 0

    peaks_mask = np.concatenate(([False], rising[:-1] & recing, [False]))

    ws = 30
    steps_per_cycle = 365  # should be enough to have peaks_per_cycle peaks
    peaks_per_cycle = 5
    n_steps = in_arr.shape[0]

    assert steps_per_cycle > peaks_per_cycle
    assert steps_per_cycle > ws
    assert steps_per_cycle < n_steps

    window_sums = np.full(steps_per_cycle, np.inf)
    for i in range(ws, steps_per_cycle + ws - 1):
        window_sums[i - ws] = in_arr[i - ws:i].sum()

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

    while (beg_idx < n_steps):
        loop_mask = np.zeros(n_steps, dtype=bool)
        loop_mask[beg_idx:end_idx] = True
        loop_mask &= peaks_mask

        assert loop_mask.sum() > peaks_per_cycle, (beg_idx, end_idx)

        highest_idxs = np.argsort(in_arr[loop_mask])[-peaks_per_cycle:]

        out_mask[np.where(loop_mask)[0][highest_idxs]] = True

        beg_idx = end_idx
        end_idx += steps_per_cycle

    assert out_mask.sum(), 'No peaks selected!'
    return out_mask


def get_driest_mask(in_arr):

    ws = 30
    steps_per_cycle = 365  # should be enough to have peaks_per_cycle peaks
    n_steps = in_arr.shape[0]

    assert steps_per_cycle > ws
    assert steps_per_cycle < n_steps

    runn_mean_arr = np.full(n_steps - ws, np.nan)

    for i in range(n_steps - ws):
        runn_mean_arr[i] = in_arr[i:i + ws].mean()

    window_sums = np.full(steps_per_cycle, np.inf)
    for i in range(ws, steps_per_cycle + ws - 1):
        window_sums[i - ws] = in_arr[i - ws:i].sum()

    assert np.all(window_sums > 0)

    min_idx = int(0.5 * ws) + np.argmin(window_sums)

    if min_idx > ws:
        beg_idx = 0
        end_idx = min_idx

    else:
        beg_idx = min_idx
        end_idx = min_idx + steps_per_cycle

    assert n_steps >= end_idx - beg_idx, 'Too few steps!'

    out_mask = np.zeros(n_steps, dtype=bool)

    while (beg_idx < n_steps):
        if runn_mean_arr[beg_idx:end_idx].shape[0] < 2:
            break

        loop_mask = np.zeros(n_steps, dtype=bool)
        loop_mask[beg_idx:end_idx] = True

        driest_idx = np.argmin(runn_mean_arr[beg_idx:end_idx])

        out_mask[np.where(loop_mask)[0][driest_idx:driest_idx + ws]] = True

        beg_idx = end_idx
        end_idx += steps_per_cycle

    assert out_mask.sum(), 'No driest selected!'
    return out_mask


def get_peaks_effs_dict(qobs_arr, qsim_arr, mask, eff_ftns_dict):

    peak_effs_dict = {}
    for eff_ftn_key in eff_ftns_dict:
        eff_ftn = eff_ftns_dict[eff_ftn_key]

        peak_effs_dict[eff_ftn_key] = eff_ftn(
            qobs_arr[mask], qsim_arr[mask], 0)

    return peak_effs_dict


@traceback_wrapper
def plot_cat_vars_errors(plot_args):

    (cat_db, err_var_labs) = plot_args

    ds_labs = ['calib']

    with h5py.File(cat_db, 'r') as db:
        out_dir = db['data'].attrs['main']
        out_dir = os.path.join(out_dir, r'11_errors')

        mkdir_hm(out_dir)

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


@traceback_wrapper
def plot_cat_prms_transfer_perfs(plot_args):

    '''Plot performances for a given using parameter vectors from other
    catchments.'''

    cat_db, (kf_prm_dict, cats_vars_dict) = plot_args

    with h5py.File(cat_db, 'r') as db:
        main_out_dir = db['data'].attrs['main']

        trans_out_dir = os.path.join(main_out_dir, '09_prm_trans_compare')

        mkdir_hm(trans_out_dir)

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


@traceback_wrapper
def plot_cat_kfold_effs(args):

    '''Plot catchment performances using parameters from every kfold
    on all kfolds.
    '''

    cat_db_path, (ann_cyc_flag, hgs_db_path) = args

    with h5py.File(cat_db_path, 'r') as db:
        out_dir = db['data'].attrs['main']
        out_dir = os.path.join(out_dir, r'05_kfolds_perf')

        mkdir_hm(out_dir)

        kfolds = db['data'].attrs['kfolds']
        cat = db.attrs['cat']

        off_idx = db['data'].attrs['off_idx']
        cv_flag = db['data'].attrs['cv_flag']

        if cv_flag:
            print('plot_cat_kfold_effs not possible with cv_flag!')
            return

        else:
            use_step_flag = db['calib/kf_01/use_step_flag'][()]
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

    mkdir_hm(out_dir)

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
