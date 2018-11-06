# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Faizan Anwar, IWS Uni-Stuttgart)s

"""

import os
import h5py
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties as f_props
from matplotlib.gridspec import GridSpec
from scipy.stats import t

from ..models import (
    hbv_loop_py,
    get_ns_cy,
    get_ln_ns_cy,
    get_pcorr_cy,
    get_kge_cy,
    lin_regsn_cy,
    tfm_opt_to_hbv_prms_py)

plt.ioff()


def plot_hbv(plot_args):
    cat_db, (wat_bal_stps,
             plot_simple_flag,
             plot_wat_bal_flag,
             valid_flags) = plot_args

    with h5py.File(cat_db, 'r') as db:
        out_dir = db['data'].attrs['main']
        kfolds = db['data'].attrs['kfolds']
        cat = db.attrs['cat']

        off_idx = db['data'].attrs['off_idx']
        conv_ratio = db['data'].attrs['conv_ratio']
        prm_syms = db['data/all_prms_labs'][...]
        use_obs_flow_flag = db['data'].attrs['use_obs_flow_flag']
        area_arr = db['data/area_arr'][...]
        bds_arr = db['cdata/bds_arr'][...]

        valid_flag = valid_flags[0]

        all_kfs_dict = {}
        for i in range(1, kfolds + 1):
            cd_db = db[f'calib/kf_{i:02d}']
            kf_dict = {key: cd_db[key][...] for key in cd_db}
            kf_dict['use_obs_flow_flag'] = use_obs_flow_flag

            #TODO: add right values here!!
            kf_dict['prms_flags'] = db['cdata']['all_prms_flags'][...]
            kf_dict['f_var_infos'] = db['cdata']['aux_var_infos'][...]
            kf_dict['prms_idxs'] = db['cdata']['use_prms_idxs'][...]
            kf_dict['fvar'] = db['cdata']['aux_vars'][...]
            kf_dict['bds_dfs'] = db['cdata']['bds_arr'][...]
            kf_dict['n_cells'] = (cd_db['ini_arr'][...]).shape[0]

            if (valid_flags[1] == True):
                kf_dict['q_ext_arr'] = db[f'valid/kf_{i:02d}']['extern_data']['Q']['q'][...]
                kf_dict['snow_ext_arr'] = db[f'valid/kf_{i:02d}']['extern_data']['snow']['depth'][...]
                kf_dict['evap_ext_arr'] = db[f'valid/kf_{i:02d}']['extern_data']['ET']['total_ET'][...]
                kf_dict['ext_model_name'] = db[f'valid/kf_{i:02d}']['extern_data'].attrs.get('model')
            all_kfs_dict[i] = kf_dict

        if valid_flag == True:
            valid_tem_arr = np.concatenate([
                all_kfs_dict[i]['tem_arr'][:,db['valid']['kf_01']['valid_step_arr'][...] ==1] for i in all_kfs_dict], axis=1)
            valid_tem_arr = np.asarray(valid_tem_arr,
                                         order='C')
            valid_ppt_arr = np.concatenate([
                all_kfs_dict[i]['ppt_arr'][:,db['valid']['kf_01']['valid_step_arr'][...] ==1] for i in all_kfs_dict], axis=1)
            valid_ppt_arr = np.asarray(valid_ppt_arr,
                                         order='C')
            valid_pet_arr = np.concatenate([
                all_kfs_dict[i]['pet_arr'][:,db['valid']['kf_01']['valid_step_arr'][...] ==1] for i in all_kfs_dict], axis=1)
            valid_pet_arr = np.asarray(valid_pet_arr,
                                         order='C')
            valid_qact_arr = np.concatenate([
                all_kfs_dict[i]['qact_arr'][db['valid']['kf_01']['valid_step_arr'][...] ==1] for i in all_kfs_dict], axis=0)
            valid_qact_arr = np.asarray(valid_qact_arr,
                                         order='C')
            if valid_flags[1]== True:
                valid_q_ext_arr = np.concatenate([
                    all_kfs_dict[i]['q_ext_arr'][db['valid']['kf_01']['valid_step_arr'][...] ==1] for i in all_kfs_dict], axis = 0)
                valid_q_ext_arr = np.asarray(valid_q_ext_arr,
                                                 order='C')
                valid_snow_ext_arr = np.concatenate([
                    all_kfs_dict[i]['snow_ext_arr'][
                        db['valid']['kf_01']['valid_step_arr'][...] == 1]
                    for i in all_kfs_dict], axis=0)
                valid_snow_ext_arr = np.asarray(valid_snow_ext_arr,
                                                 order='C')
                valid_evap_ext_arr = np.concatenate([
                    all_kfs_dict[i]['evap_ext_arr'][
                        db['valid']['kf_01']['valid_step_arr'][...] == 1]
                    for i in all_kfs_dict], axis=0)
                valid_evap_ext_arr = np.asarray(valid_evap_ext_arr,
                                                 order='C')
            if 'extra_us_inflow' in kf_dict:
                valid_us_inflow_arr = np.concatenate([
                    all_kfs_dict[i]['extra_us_inflow'][np.asarray(db['valid_time']['val_data']) ==1]
                    for i in all_kfs_dict], axis=0)
                valid_us_inflow_arr = np.asarray(valid_us_inflow_arr,
                                                 order='C')

            calib_tem_arr = np.concatenate([
                all_kfs_dict[i]['tem_arr'][:,db['calib']['kf_01']['use_step_arr'][...] ==1] for i in all_kfs_dict], axis=1)
            calib_tem_arr = np.asarray(calib_tem_arr, order='C')
            calib_ppt_arr = np.concatenate([
                all_kfs_dict[i]['ppt_arr'][:,db['calib']['kf_01']['use_step_arr'][...] ==1] for i in all_kfs_dict], axis=1)
            calib_ppt_arr = np.asarray(calib_ppt_arr, order='C')
            calib_pet_arr = np.concatenate([
                all_kfs_dict[i]['pet_arr'][:,db['calib']['kf_01']['use_step_arr'][...] ==1] for i in all_kfs_dict], axis=1)
            calib_pet_arr = np.asarray(calib_pet_arr, order='C')
            calib_qact_arr = np.concatenate([
                all_kfs_dict[i]['qact_arr'][db['calib']['kf_01']['use_step_arr'][...] ==1] for i in all_kfs_dict], axis=0)
            calib_qact_arr = np.asarray(calib_qact_arr, order='C')
            if valid_flags[1]== True:
                calib_q_ext_arr = np.concatenate([
                    all_kfs_dict[i]['q_ext_arr'][db['calib']['kf_01']['use_step_arr'][...] ==1] for i in all_kfs_dict], axis = 0)
                calib_q_ext_arr = np.asarray(calib_q_ext_arr, order='C')
                calib_snow_ext_arr = np.concatenate([
                    all_kfs_dict[i]['snow_ext_arr'][db['calib']['kf_01']['use_step_arr'][...] ==1] for i in all_kfs_dict], axis = 0)
                calib_snow_ext_arr = np.asarray(calib_snow_ext_arr, order='C')
                calib_evap_ext_arr = np.concatenate([
                    all_kfs_dict[i]['evap_ext_arr'][
                        db['calib']['kf_01']['use_step_arr'][...] == 1] for
                    i in all_kfs_dict], axis=0)
                calib_evap_ext_arr = np.asarray(calib_evap_ext_arr, order='C')
            if 'extra_us_inflow' in kf_dict:
                calib_us_inflow_arr = np.concatenate([
                    all_kfs_dict[i]['extra_us_inflow'][np.asarray(db['calib']['kf_01']['use_step_arr']) ==1]
                    for i in all_kfs_dict], axis=0)
                calib_us_inflow_arr = np.asarray(calib_us_inflow_arr, order='C')

        all_tem_arr = np.concatenate([
            all_kfs_dict[i]['tem_arr'] for i in all_kfs_dict], axis=1)
        all_ppt_arr = np.concatenate([
            all_kfs_dict[i]['ppt_arr'] for i in all_kfs_dict], axis=1)
        all_pet_arr = np.concatenate([
            all_kfs_dict[i]['pet_arr'] for i in all_kfs_dict], axis=1)
        all_qact_arr = np.concatenate([
            all_kfs_dict[i]['qact_arr'] for i in all_kfs_dict], axis=0)

        if 'extra_us_inflow' in kf_dict:
            all_us_inflow_arr = np.concatenate([
                all_kfs_dict[i]['extra_us_inflow']
                for i in all_kfs_dict], axis=0)

        for i in range(1, kfolds + 1):
            kf_dict = all_kfs_dict[i]
            kf_i = f'{i:02d}'

            _plot_hbv_kf(
                kf_i,
                cat,
                kf_dict,
                area_arr,
                conv_ratio,
                prm_syms,
                off_idx,
                out_dir,
                wat_bal_stps,
                plot_simple_flag,
                plot_wat_bal_flag,
                valid_flags,
                bds_arr)

            if valid_flags[0] == True:
                kf_dict['tem_arr'] = calib_tem_arr
                kf_dict['ppt_arr'] = calib_ppt_arr
                kf_dict['pet_arr'] = calib_pet_arr
                kf_dict['qact_arr'] = calib_qact_arr
                if valid_flags[1] == True:
                    kf_dict['q_ext_arr'] = calib_q_ext_arr
                    kf_dict['snow_ext_arr'] = calib_snow_ext_arr
                    kf_dict['evap_ext_arr'] = calib_evap_ext_arr
                if 'extra_us_inflow' in kf_dict:
                    kf_dict['extra_us_inflow'] = calib_us_inflow_arr

                kf_i = f'{i:02d}' + '_calib'
                _plot_hbv_kf(
                    kf_i,
                    cat,
                    kf_dict,
                    area_arr,
                    conv_ratio,
                    prm_syms,
                    off_idx,
                    out_dir,
                    wat_bal_stps,
                    plot_simple_flag,
                    plot_wat_bal_flag,
                    valid_flags,
                    bds_arr)

                kf_dict['tem_arr'] = valid_tem_arr
                kf_dict['ppt_arr'] = valid_ppt_arr
                kf_dict['pet_arr'] = valid_pet_arr
                kf_dict['qact_arr'] = valid_qact_arr
                if valid_flags[1] == True:
                    kf_dict['q_ext_arr'] = valid_q_ext_arr
                    kf_dict['snow_ext_arr'] = valid_snow_ext_arr
                    kf_dict['evap_ext_arr'] = valid_evap_ext_arr
                if 'extra_us_inflow' in kf_dict:
                    kf_dict['extra_us_inflow'] = valid_us_inflow_arr

                kf_i = f'{i:02d}' + '_valid'
                _plot_hbv_kf(
                    kf_i,
                    cat,
                    kf_dict,
                    area_arr,
                    conv_ratio,
                    prm_syms,
                    off_idx,
                    out_dir,
                    wat_bal_stps,
                    plot_simple_flag,
                    plot_wat_bal_flag,
                    valid_flags,
                    bds_arr)

            if kfolds>1:
                kf_dict['tem_arr'] = all_tem_arr
                kf_dict['ppt_arr'] = all_ppt_arr
                kf_dict['pet_arr'] = all_pet_arr
                kf_dict['qact_arr'] = all_qact_arr

                if 'extra_us_inflow' in kf_dict:
                    kf_dict['extra_us_inflow'] = all_us_inflow_arr

                kf_i = f'all'
                _plot_hbv_kf(
                    kf_i,
                    cat,
                    kf_dict,
                    area_arr,
                    conv_ratio,
                    prm_syms,
                    off_idx,
                    out_dir,
                    wat_bal_stps,
                    plot_simple_flag,
                    plot_wat_bal_flag,
                    valid_flags,
                    bds_arr)

    return

def plot_error(plot_args):

    (cat_db, valid_flag) = plot_args

    with h5py.File(cat_db, 'r') as db:
        if valid_flag[1] == True:
            qext0 = db['valid']['kf_01']['extern_data']['Q']['q'][...]
            snow_ext0 = db['valid']['kf_01']['extern_data']['snow']['depth'][...]
        if valid_flag[0] == True:
            plot_list = 3
        else:
            plot_list = 1
        out_dir = db['data'].attrs['main']
        sim_dir = os.path.join(out_dir, r'sim_out\all_params.pkl')
        out_dir = os.path.join(out_dir, r'10_error')
        if not os.path.exists(out_dir):
            try:
                os.mkdir(out_dir)
            except:
                pass

        kfolds = db['data'].attrs['kfolds']
        cat = db.attrs['cat']

        with open(sim_dir, 'rb') as f:
            all_out = pickle.load(f)

        for i in range(1, kfolds + 1):
            kf_str = f'kf_{i:02d}'
            cd_db = db[f'calib/{kf_str}']
            qsim0 = cd_db['qsim_arr'][...]
            qact0 = cd_db['qact_arr'][...]
            tem0 = np.squeeze(cd_db['tem_arr'][...])
            ppt0 = np.squeeze(cd_db['ppt_arr'][...])
            pet0 = np.squeeze(cd_db['pet_arr'][...])
            snow0 = np.squeeze(all_out['outs_arr'])[:, 0]

            # pet0 = np.squeeze(cd_db['snow_arr'][...])

            for p in range(plot_list):
                if p==1:
                    calib_arr = db[f'valid/{kf_str}']['use_step_arr'][...]
                    qsim = qsim0[calib_arr==1]
                    qact = qact0[calib_arr==1]
                    if valid_flag[1] == True:
                        qext = qext0[calib_arr==1]
                        snow_ext = snow_ext0[calib_arr == 1]
                    tem = tem0[calib_arr==1]
                    ppt = ppt0[calib_arr==1]
                    pet = pet0[calib_arr==1]
                    snow = snow0[calib_arr == 1]
                    pl = 'calib'

                elif p==2:
                    valid_arr = db[f'valid/{kf_str}']['valid_step_arr'][...]
                    qsim = qsim0[valid_arr==1]
                    qact = qact0[valid_arr==1]
                    if valid_flag[1] == True:
                        qext = qext0[valid_arr==1]
                        snow_ext = snow_ext0[valid_arr==1]
                    tem = tem0[valid_arr==1]
                    ppt = ppt0[valid_arr==1]
                    pet = pet0[valid_arr==1]
                    snow = snow0[valid_arr == 1]
                    pl = 'valid'
                else:
                    qsim = qsim0
                    qact = qact0
                    if valid_flag[1] == True:
                        qext = qext0
                        snow_ext = snow_ext0
                    tem = tem0
                    ppt = ppt0
                    pet = pet0
                    snow = snow0
                    pl = 'all'

                if not os.path.exists(os.path.join(out_dir, pl)):
                    os.makedirs(os.path.join(out_dir, pl))

                error = qsim - qact
                variable_list = ['tem', 'ppt', 'pet', 'qact', 'snow']

                description_list = ['Temperature [°C]', 'Precipitation [mm]', 'PET [mm]', 'Qact', 'snow']
                pet_axis = [0, 7, -100, +100]
                ppt_axis = [0, 40, -100, +100]
                tem_axis = [-20, 30, -100, +100]
                q_axis = [0, 250, -100, +100]
                snow_axis = [0, 100, -100, +100]

                axis_list = [tem_axis, ppt_axis, pet_axis, q_axis, snow_axis]


                for ind, variable in enumerate(variable_list):
                    var_error = np.vstack((eval(variable),error))
                    sort_idxs = np.argsort(eval(variable))
                    #tem_error_sort = np.sort(tem_error, axis = 1)
                    var_error_sort = var_error[:,sort_idxs]

                    #ax = plt.gca()
                    plt.scatter(var_error_sort[0, :],
                                var_error_sort[1, :],
                                label='HBV', alpha=0.1,
                                s=0.5)  # tem_error_sort[1,:]
                    plt.plot(var_error_sort[0, :],
                             pd.rolling_mean(var_error_sort[1, :],
                                             window=1000, center=True))

                    #ax.set_yscale('log')
                    plt.xlabel(description_list[ind])
                    plt.ylabel('absolute Q error')
                    plt.axis(axis_list[ind])
                    plt.grid(b=True)

                    plt.savefig(str(
                        Path(out_dir, f'{pl}/{variable}_error_{cat}_kf_{i:02d}_{pl}.png')),
                                bbox_inches='tight', dpi=600)
                    if valid_flag[1] == True:
                        error_ext = qext - qact
                        if variable == 'snow':
                            variable = 'snow_ext'
                        var_error_ext = np.vstack((eval(variable), error_ext))
                        sort_idxs = np.argsort(eval(variable))
                        var_error_sort_ext = var_error_ext[:, sort_idxs]
                        plt.scatter(var_error_sort_ext[0, :],
                                    var_error_sort_ext[1, :], label='Shetran',
                                    alpha=0.1, s=0.5)
                        plt.plot(var_error_sort_ext[0, :],
                                pd.rolling_mean(var_error_sort_ext[1, :],
                                                 window=1000,
                                                 center=True))
                        plt.xlabel(description_list[ind])
                        plt.ylabel('absolute Q error')
                        plt.legend()
                        #plt.axis(axis_list[i])
                        plt.savefig(str(
                            Path(out_dir, f'{pl}/{variable}_error_ext_{cat}_kf_{i:02d}_{pl}.png')),
                            bbox_inches='tight', dpi=600)
                    plt.close()
                    if valid_flag[1] == True:
                        plt.scatter(var_error_sort_ext[0, :],
                                   var_error_sort_ext[1, :],
                                   label='Shetran',
                                   alpha=0.1, s=0.5)
                        plt.plot(var_error_sort_ext[0, :],
                                 pd.rolling_mean(var_error_sort_ext[1, :],
                                                 window=1000,
                                                 center=True))
                        plt.xlabel(description_list[ind])
                        plt.ylabel('absolute Q error')
                        plt.axis(axis_list[ind])
                        plt.legend()
                        plt.grid(b=True)
                        # plt.axis(axis_list[i])
                        plt.savefig(str(
                            Path(out_dir,
                                 f'{pl}/{variable}_error_ext_only_{cat}_kf_{i:02d}_{pl}.png')),
                            bbox_inches='tight', dpi=600)
                        plt.close()
    return

def plot_hull(plot_args):

    cat_db = plot_args

    with h5py.File(cat_db, 'r') as db:
        out_dir = db['data'].attrs['main']
        out_dir = os.path.join(out_dir, r'11_chull')
        if not os.path.exists(out_dir):
            try:
                os.mkdir(out_dir)
            except:
                pass

        kfolds = db['data'].attrs['kfolds']
        cat = db.attrs['cat']

        for k in range(1, kfolds + 1):
            kf_str = f'kf_{k:02d}'
            cd_db = db[f'calib/{kf_str}']
            chull_pts = cd_db['all_prm_vecs'][...]
            n_dims = chull_pts.shape[1]
            plt.figure(figsize=(15, 15))
            grid_axes = GridSpec(n_dims, n_dims)
            chull_min = np.min(chull_pts)
            chull_max = np.max(chull_pts)
            for i in range(n_dims):
                for j in range(n_dims):
                    if i >= j:
                        continue

                    ax = plt.subplot(grid_axes[i, j])

                    ax.set_aspect('equal', 'box')

                    ax.set_xlim(chull_min, chull_max)
                    ax.set_ylim(chull_min, chull_max)

                    ax.scatter(chull_pts[:, i], chull_pts[:, j], s=2, alpha=0.6)

                    ax.set_xticks([])
                    ax.set_xticklabels([])

                    ax.set_yticks([])
                    ax.set_yticklabels([])

                    ax.text(
                        0.95,
                        0.95,
                        f'({i}, {j})',
                        horizontalalignment='right',
                        verticalalignment='top',
                        transform=ax.transAxes)

                    plt.suptitle(
                        f'Convex hull of {n_dims}D Params in 2D\n'
                        f'Total points: {chull_pts.shape[0]}',
                        x=0.5,
                        y=0.5,
                        va='bottom',
                        ha='right')

            plt.savefig(str(
                Path(out_dir,
                     f'chull_in_2D_{cat}_kf_{k:02d}.png')),
                bbox_inches='tight')
            plt.close('all')
    return

def _plot_prm_vecs(cat_db):
    with h5py.File(cat_db, 'r') as db:
        out_dir = db['data'].attrs['main']
        out_dir = os.path.join(out_dir, r'06_prm_vecs')
        if not os.path.exists(out_dir):
            try:
                os.mkdir(out_dir)
            except:
                pass

        kfolds = db['data'].attrs['kfolds']
        cat = db.attrs['cat']
        opt_schm = db['cdata/opt_schm_vars_dict/opt_schm'].value

        calib_db = db['calib']

        for i in range(1, kfolds + 1):
            kf_str = f'kf_{i:02d}'
            prm_vecs = calib_db[kf_str + '/prm_vecs'][...]
            bounds_arr = db['cdata/bds_arr'][...]
            prm_syms = db['cdata/use_prms_labs'][...]
            cobj_vals = calib_db[kf_str + '/curr_obj_vals'][...]
            pobj_vals = calib_db[kf_str + '/pre_obj_vals'][...]
            _plot_k_prm_vecs(
                i,
                cat,
                out_dir,
                bounds_arr,
                prm_vecs,
                cobj_vals,
                pobj_vals,
                prm_syms,
                opt_schm)
    return


def _plot_k_prm_vecs(
        kf_i,
        cat,
        out_dir,
        bounds_arr,
        prm_vecs,
        cobj_vals,
        pobj_vals,
        prm_syms,
        opt_schm):

    plt.figure(figsize=(max(35, bounds_arr.shape[0]), 13))
    tick_font_size = 10

    stats_cols = ['min', 'max', 'mean', 'stdev', 'min_bd', 'max_bd']
    n_stats_cols = len(stats_cols)

    norm_prm_vecs = prm_vecs.copy()
    for i in range(prm_vecs.shape[0]):
        for j in range(prm_vecs.shape[1]):
            prm_vecs[i, j] = ((prm_vecs[i, j] * (bounds_arr[j, 1] - bounds_arr[j, 0])) +
                         bounds_arr[j, 0])
            if not np.isfinite(prm_vecs[i, j]):
                prm_vecs[i, j] = bounds_arr[j, 0]

    n_params = bounds_arr.shape[0]

    curr_min = prm_vecs.min(axis=0)
    curr_max = prm_vecs.max(axis=0)
    curr_mean = prm_vecs.mean(axis=0)
    curr_stdev = prm_vecs.std(axis=0)
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
    stats_ax.set_xticklabels(prm_syms)
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
    for i in range(prm_vecs.shape[0]):
        params_ax.plot(plot_range,
                       norm_prm_vecs[i],
                       alpha=0.1,
                       color='k')

    params_ax.set_ylim(0., 1.)
    params_ax.set_xticks(plot_range)
    params_ax.set_xticklabels(prm_syms)
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

    title_str = ('Distributed HBV parameters - '
                 f'{opt_schm} final prm_vecs (n={prm_vecs.shape[0]})')
    plt.suptitle(title_str, size=tick_font_size + 10)
    plt.subplots_adjust(hspace=0.15)

    plt.savefig(str(Path(out_dir, f'hbv_prm_vecs_{cat}_kf_{kf_i:02d}.png')),
                bbox_inches='tight')
    plt.close()

    # curr_obj_vals
    * _, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    n_vals = float(cobj_vals.shape[0])

    probs = 1 - (np.arange(1.0, n_vals + 1) / (n_vals + 1))

    ax1.set_title(('Final objective function distribution\n'
                   f'Min. obj.: {cobj_vals.min():0.4f}, '
                   f'max. obj.: {cobj_vals.max():0.4f}'))
    ax1.plot(np.sort(cobj_vals), probs, marker='o', alpha=0.8)
    ax1.set_ylabel('Non-exceedence Probability (-)')
    ax1.set_xlim(ax1.set_xlim()[1], ax1.set_xlim()[0])
    ax1.grid()

    ax2.hist(cobj_vals, bins=20)
    ax2.set_xlabel('Obj. ftn. values (-)')
    ax2.set_ylabel('Frequency (-)')
    plt.savefig(str(Path(out_dir, f'hbv_cobj_cdf_{cat}_kf_{kf_i:02d}.png')),
                bbox_inches='tight')

    # pre_obj_vals
    * _, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    n_vals = float(pobj_vals.shape[0])

    probs = 1 - (np.arange(1.0, n_vals + 1) / (n_vals + 1))

    ax1.set_title(('2nd last objective function distribution\n'
                   f'Min. obj.: {pobj_vals.min():0.4f}, '
                   f'max. obj.: {pobj_vals.max():0.4f}'))
    ax1.plot(np.sort(pobj_vals), probs, marker='o', alpha=0.8)
    ax1.set_ylabel('Non-exceedence Probability (-)')
    ax1.set_xlim(ax1.set_xlim()[1], ax1.set_xlim()[0])
    ax1.grid()

    ax2.hist(pobj_vals, bins=20)
    ax2.set_xlabel('Obj. ftn. values (-)')
    ax2.set_ylabel('Frequency (-)')
    plt.savefig(str(Path(out_dir, f'hbv_pobj_cdf_{cat}_kf_{kf_i:02d}.png')),
                bbox_inches='tight')
    plt.close('all')
    return


def _plot_hbv_kf(
        kf_str,
        cat,
        kf_dict,
        area_arr,
        conv_ratio,
        prm_syms,
        off_idx,
        out_dir,
        wat_bal_stps,
        plot_simple_flag,
        plot_wat_bal_flag,
        valid_flags,
        bds_arr):

    # if valid_flags[2] == 'DE':
    #      pass
    # elif valid_flags[2] == 'ROPE':
    #     all_kfs_dict = {}
    #     for i in range(1, kfolds + 1):
    #         cd_db = db[f'calib/kf_{i:02d}']
    #         prm_vecs = np.asarray(cd_db.attrs['prm_vecs'])
    #         for i in range(prm_vecs.shape[0]):
    #             pass
    if 'use_obs_flow_flag' in kf_dict:
        use_obs_flow_flag = bool(kf_dict['use_obs_flow_flag'])
    else:
        use_obs_flow_flag = False

    rarea_arr = area_arr.reshape(-1, 1)
    rrarea_arr = area_arr.reshape(-1, 1, 1)

    temp_dist_arr = kf_dict['tem_arr']
    prec_dist_arr = kf_dict['ppt_arr']
    pet_dist_arr = kf_dict['pet_arr']
    if valid_flags[2] == 'DE':
        prms_dist_arr = kf_dict['hbv_prms']
    elif valid_flags[2] == 'ROPE':
        hbv_params = np.zeros_like(kf_dict['prm_vecs'])
        for i in range(kf_dict['prm_vecs'].shape[1]):
            hbv_params[:,i] = bds_arr[i,0] + (kf_dict['prm_vecs'][:,i] * (bds_arr[i,1] - bds_arr[i,0] ))

        for i in range (kf_dict['prm_vecs'].shape[1]):
            opt_prms = kf_dict['prm_vecs'][i, :]
            hbv_params[i, :] = tfm_opt_to_hbv_prms_py(kf_dict['prms_flags'],
                                kf_dict['f_var_infos'],
                                kf_dict['prms_idxs'],
                                kf_dict['fvar'],
                                opt_prms,
                                kf_dict['bds_dfs'],
                                kf_dict['n_cells'])

        prms_dist_arr  = hbv_params
    prms_arr = (rarea_arr * kf_dict['hbv_prms']).sum(axis=0)
    if valid_flags[1] == True:
        q_ext_arr = kf_dict['q_ext_arr']
        snow_ext_arr = kf_dict['snow_ext_arr']
        evap_ext_arr = kf_dict['evap_ext_arr']

    n_recs = temp_dist_arr.shape[1]
    n_cells = temp_dist_arr.shape[0]

    all_sims = np.zeros((prms_dist_arr.shape[0], temp_dist_arr.shape[1]))

    if valid_flags[2] == 'ROPE':
        for i in range(prms_dist_arr.shape[0]):
            all_outputs_dict = hbv_loop_py(
                temp_dist_arr,
                prec_dist_arr,
                pet_dist_arr,
                np.expand_dims(prms_dist_arr[i,:], axis=0),
                kf_dict['ini_arr'],
                area_arr,
                conv_ratio)

            all_sims[i] = all_outputs_dict['qsim_arr']

    prms_dist_arr = kf_dict['hbv_prms']

    all_outputs_dict = hbv_loop_py(
        temp_dist_arr,
        prec_dist_arr,
        pet_dist_arr,
        prms_dist_arr,
        kf_dict['ini_arr'],
        area_arr,
        conv_ratio)

    if valid_flags[2] == 'ROPE':
        sim_mean = np.mean(all_sims, axis=0)
        sim_std = np.std(all_sims, axis=0)
        conf_level = 0.95
        t_bounds = t.interval(conf_level, all_sims.shape[0] - 1)

        ci = [sim_mean +  critval * sim_std / np.sqrt(all_sims.shape[0]) for critval in
              t_bounds]

    assert all_outputs_dict['loop_ret'] == 0.0

    temp_arr = (rarea_arr * temp_dist_arr).sum(axis=0)
    prec_arr = (rarea_arr * prec_dist_arr).sum(axis=0)
    pet_arr = (rarea_arr * pet_dist_arr).sum(axis=0)

    del temp_dist_arr, prec_dist_arr, pet_dist_arr, prms_dist_arr

    all_output = all_outputs_dict['outs_arr']
    all_output = (rrarea_arr * all_output).sum(axis=0)

    snow_arr = all_output[:, 0]
    liqu_arr = all_output[:, 1]
    sm_arr = all_output[:, 2]
    tot_run_arr = all_output[:, 3]
    evap_arr = all_output[:, 4]
    ur_sto_arr = all_output[:, 5]
    ur_run_uu = all_output[:, 6]
    ur_run_ul = all_output[:, 7]
    ur_to_lr_run = all_output[:, 8]
    lr_sto_arr = all_output[:, 9]
    lr_run_arr = lr_sto_arr * prms_arr[10]
    comb_run_arr = ur_run_uu + ur_run_ul + lr_run_arr

    q_sim_arr = all_outputs_dict['qsim_arr']

    sim_out = os.path.join(out_dir, 'sim_out')
    if not os.path.exists(sim_out):
        try:
            os.mkdir(sim_out)
        except:
            pass

    sim_out_dir = os.path.join(sim_out, 'all_params.pkl')
    with open(sim_out_dir, 'wb') as f:
        pickle.dump(all_outputs_dict, f, pickle.HIGHEST_PROTOCOL)

    extra_us_inflow_flag = 'extra_us_inflow' in kf_dict

    q_sim_arr = (q_sim_arr).copy(order='C')
    q_act_arr = kf_dict['qact_arr']
    if valid_flags[1] == True:
        q_ext_arr = kf_dict['q_ext_arr']
        snow_ext_arr = kf_dict['snow_ext_arr']
        evap_ext_arr = kf_dict['evap_ext_arr']
        q_ext_arr_diff = q_ext_arr.copy()
        ext_model_name = kf_dict['ext_model_name']
    q_act_arr_diff = q_act_arr.copy()

    if extra_us_inflow_flag:
        extra_us_inflow = kf_dict['extra_us_inflow']
        q_sim_arr = q_sim_arr + extra_us_inflow
        q_act_arr_diff = q_act_arr - extra_us_inflow
        if valid_flags[1] == True:
            q_ext_arr_diff = q_ext_arr - extra_us_inflow

    if valid_flags[0] == True:
        hbv_figs_dir = os.path.join(out_dir, '03_hbv_figs_valid')


    else:
        hbv_figs_dir = os.path.join(out_dir, '03_hbv_figs')
    if not os.path.exists(hbv_figs_dir):
        try:
            os.mkdir(hbv_figs_dir)
        except:
            pass

    sim_dict = {
        'temp': temp_arr,
        'prec': prec_arr,
        'pet': pet_arr,
        'snow': snow_arr,
        'liqu': liqu_arr,
        'sm': sm_arr,
        'tot_run': tot_run_arr,
        'evap': evap_arr,
        'ur_sto': ur_sto_arr,
        'ur_run_uu': ur_run_uu,
        'ur_run_ul': ur_run_ul,
        'ur_to_lr_run': ur_to_lr_run,
        'lr_sto': lr_sto_arr,
        'lr_run': lr_run_arr,
        'comb_run': comb_run_arr,
        'q_sim': q_sim_arr
        }

    sim_df = pd.DataFrame(sim_dict, dtype=float)
    sim_df.to_csv(os.path.join(
        hbv_figs_dir, f'kf_{kf_str}_HBV_sim_{cat}.csv'), sep=';')

    ns = get_ns_cy(q_act_arr, q_sim_arr, off_idx)
    print('Best NS:{}'.format(ns))
    if valid_flags[1] == True:
        ns_ext = get_ns_cy(q_act_arr, q_ext_arr, off_idx)
        print("NS {} {}: {:f}".format(ext_model_name, kf_str, ns_ext))
    else:
        ns_ext = 0

    for i in range(all_sims.shape[0]):
        ns = get_ns_cy(q_act_arr, all_sims[i,:], off_idx)

    ln_ns = get_ln_ns_cy(q_act_arr, q_sim_arr, off_idx)
    kge = get_kge_cy(q_act_arr, q_sim_arr, off_idx)
    q_correl = get_pcorr_cy(q_act_arr, q_sim_arr, off_idx)

    (tt,
     cm,
     p_cm,
     fc,
     beta,
     pwp,
     ur_thresh,
     k_uu,
     k_ul,
     k_d,
     k_ll) = prms_arr

    def save_simple_opt(out_dir):
        '''Save the output of the optimize function
        '''
        assert np.all(np.isfinite(prms_arr)), 'Invalid HBV parameters!'
        assert q_act_arr.shape[0] == q_sim_arr.shape[0], (
            'Original and simulated discharge have unequal steps!')

        if not os.path.exists(out_dir):
            try:
                os.mkdir(out_dir)
            except:
                pass

        out_fig_loc = os.path.join(
            out_dir, f'kf_{kf_str}_HBV_model_plot_{cat}.png')
        out_params_loc = os.path.join(
            out_dir, f'kf_{kf_str}_HBV_model_params_{cat}.csv')

        out_labs = []
        out_labs.extend(prm_syms)

        if 'route_labs' in kf_dict:
            out_labs.extend(kf_dict['route_labs'])

        out_labs.extend(['ns', 'ln_ns', 'ns_ext', 'kge', 'obj_ftn', 'p_corr'])

        out_params_df = pd.DataFrame(index=out_labs, columns=['value'])
        out_params_df['value'] = \
            np.concatenate((prms_arr,
                            [ns, ln_ns, ns_ext, kge, 'nothing_yet', q_correl]))
        out_params_df.to_csv(out_params_loc, sep=str(';'), index_label='param')

        cum_q = np.cumsum(q_act_arr_diff[off_idx:] / conv_ratio)
        cum_q_sim = np.cumsum(comb_run_arr[off_idx:])
        if valid_flags[1] == True:
            cum_q_ext = np.cumsum(q_ext_arr_diff[off_idx:] / conv_ratio)

        min_vol_diff_err = 0.5
        max_vol_diff_err = 1.5

        vol_diff_arr = cum_q_sim / cum_q
        vol_diff_arr[vol_diff_arr < min_vol_diff_err + 0.05] = (
            min_vol_diff_err + 0.05)
        vol_diff_arr[vol_diff_arr > max_vol_diff_err - 0.05] = (
            max_vol_diff_err - 0.05)
        vol_diff_arr = np.concatenate((np.full(shape=(off_idx - 1),
                                               fill_value=np.nan),
                                                vol_diff_arr))

        if valid_flags[1] == True:
            vol_diff_ext_arr = cum_q_ext / cum_q

            vol_diff_ext_arr[vol_diff_ext_arr < min_vol_diff_err + 0.05] = (
                    min_vol_diff_err + 0.05)
            vol_diff_ext_arr[vol_diff_ext_arr > max_vol_diff_err - 0.05] = (
                    max_vol_diff_err - 0.05)
            vol_diff_ext_arr = np.concatenate((np.full(shape=(off_idx - 1),
                                       fill_value=np.nan),
                                               vol_diff_ext_arr))

        bal_idxs = [0]
        bal_idxs.extend(list(range(off_idx, n_recs, wat_bal_stps)))
        act_bal_arr = []
        sim_bal_arr = []
        if valid_flags[1] == True:
            sim_bal_ext_arr = []

        prec_sum_arr = []

        min_vol_err_wET = 0.5
        max_vol_err_wET = 1.5

        for i in range(1, len(bal_idxs) - 1):
            # ET accounted for
            _curr_q_act = q_act_arr_diff[bal_idxs[i]:bal_idxs[i + 1]]
            _curr_q_act_sum = np.sum(_curr_q_act / conv_ratio)

            if valid_flags[1] ==  True:
                _curr_q_ext = q_ext_arr_diff[bal_idxs[i]:bal_idxs[i + 1]]
                _curr_q_ext_sum = np.sum(_curr_q_ext / conv_ratio)

            _curr_prec_sum = np.sum(prec_arr[bal_idxs[i]:bal_idxs[i + 1]])
            _curr_evap_sum = np.sum(evap_arr[bal_idxs[i]:bal_idxs[i + 1]])
            if valid_flags[1] == True:
                _curr_ext_evap_sum = np.sum(evap_ext_arr[bal_idxs[i]:bal_idxs[i + 1]])
            if valid_flags[1]== True:
                _curr_evap_ext_sum = np.sum(
                    evap_ext_arr[bal_idxs[i]:bal_idxs[i + 1]])

            _curr_comb_sum = np.sum(comb_run_arr[bal_idxs[i]:bal_idxs[i + 1]])

            act_bal_arr.append(_curr_q_act_sum /
                               (_curr_prec_sum - _curr_evap_sum))

            sim_bal_arr.append(_curr_comb_sum /
                               (_curr_prec_sum - _curr_evap_sum))

            if valid_flags[1] ==  True:
                sim_bal_ext_arr.append(_curr_q_ext_sum /
                                   (_curr_prec_sum - _curr_ext_evap_sum))


            prec_sum_arr.append(_curr_prec_sum)

        act_bal_arr = np.array(act_bal_arr)

        act_bal_arr[act_bal_arr < min_vol_err_wET] = min_vol_err_wET
        act_bal_arr[act_bal_arr > max_vol_err_wET] = max_vol_err_wET

        act_bal_arr = np.concatenate(([np.nan, np.nan], act_bal_arr), axis=0)
        sim_bal_arr = np.concatenate(([np.nan, np.nan], sim_bal_arr), axis=0)

        if valid_flags[1] == True:
            sim_bal_ext_arr = np.concatenate(
                ([np.nan, np.nan], sim_bal_ext_arr), axis=0)

        prec_sum_arr = np.concatenate(([np.nan, np.nan], prec_sum_arr), axis=0)

        steps_range = np.arange(0., n_recs)
        temp_trend, temp_corr, temp_slope, *_ = lin_regsn_cy(
            steps_range, temp_arr, 0)

        temp_stats_str = ''
        temp_stats_str += f'correlation: {temp_corr:0.5f}, '
        temp_stats_str += f'slope: {temp_slope:0.5f}'

        pet_trend, pet_corr, pet_slope, *_ = lin_regsn_cy(
            steps_range, pet_arr, 0)

        pet_stats_str = ''
        pet_stats_str += f'PET correlation: {pet_corr:0.5f}, '
        pet_stats_str += f'PET slope: {pet_slope:0.5f}, '

        et_trend, et_corr, et_slope, *_ = lin_regsn_cy(
            steps_range, evap_arr.copy(order='c'), 0)

        if valid_flags[1] == True:
            et_ext_trend, et_corr, et_slope, *_ = lin_regsn_cy(
                steps_range, evap_ext_arr.copy(order='c'), 0)

        pet_stats_str += f'ET correlation: {et_corr:0.5f}, '
        pet_stats_str += f'ET slope: {et_slope:0.5f}'

        plt.figure(figsize=(11, 25), dpi=250)
        t_rows = 16
        t_cols = 1
        font_size = 6

        plt.suptitle('HBV Flow Simulation')

        i = 0
        params_ax = plt.subplot2grid((t_rows, t_cols),
                                     (i, 0),
                                     rowspan=1,
                                     colspan=1)
        i += 1
        vol_err_ax = plt.subplot2grid((t_rows, t_cols),
                                      (i, 0),
                                      rowspan=1,
                                      colspan=1)
        i += 1
        discharge_ax = plt.subplot2grid((t_rows, t_cols),
                                        (i, 0),
                                        rowspan=1,
                                        colspan=1)
        i += 1
        balance_ax = plt.subplot2grid((t_rows, t_cols),
                                      (i, 0),
                                      rowspan=1,
                                      colspan=1)
        i += 1
        prec_ax = plt.subplot2grid((t_rows, t_cols),
                                   (i, 0),
                                   rowspan=1,
                                   colspan=1)
        i += 1
        liqu_ax = plt.subplot2grid((t_rows, t_cols),
                                   (i, 0),
                                   rowspan=1,
                                   colspan=1)
        i += 1
        snow_ax = plt.subplot2grid((t_rows, t_cols),
                                   (i, 0),
                                   rowspan=1,
                                   colspan=1)
        i += 1
        temp_ax = plt.subplot2grid((t_rows, t_cols),
                                   (i, 0),
                                   rowspan=1,
                                   colspan=1)
        i += 1
        pet_ax = plt.subplot2grid((t_rows, t_cols),
                                  (i, 0),
                                  rowspan=1,
                                  colspan=1)
        i += 1
        sm_ax = plt.subplot2grid((t_rows, t_cols),
                                 (i, 0),
                                 rowspan=1,
                                 colspan=1)
        i += 1
        tot_run_ax = plt.subplot2grid((t_rows, t_cols),
                                      (i, 0),
                                      rowspan=1,
                                      colspan=1)
        i += 1
        u_res_sto_ax = plt.subplot2grid((t_rows, t_cols),
                                        (i, 0),
                                        rowspan=1,
                                        colspan=1)
        i += 1
        ur_run_uu_ax = plt.subplot2grid((t_rows, t_cols),
                                        (i, 0),
                                        rowspan=1,
                                        colspan=1)
        i += 1
        ur_run_ul_ax = plt.subplot2grid((t_rows, t_cols),
                                        (i, 0),
                                        rowspan=1,
                                        colspan=1)
        i += 1
        ur_to_lr_run_ax = plt.subplot2grid((t_rows, t_cols),
                                           (i, 0),
                                           rowspan=1,
                                           colspan=1)
        i += 1
        l_res_sto_ax = plt.subplot2grid((t_rows, t_cols),
                                        (i, 0),
                                        rowspan=1,
                                        colspan=1)

        bar_x = np.arange(0, n_recs, 1)
        if valid_flags[2] == 'DE':
            discharge_ax.plot(q_act_arr, 'r-', label='Actual Flow', lw=0.8)
            discharge_ax.plot(q_sim_arr,
                              'b-',
                              label='Simulated Flow',
                              lw=0.5,
                              alpha=0.5)
        elif valid_flags[2] == 'ROPE':
            discharge_ax.fill_between(np.arange(0, n_recs, 1), ci[0], ci[1], color='#539caf',
                            alpha=0.4, label='95% CI')
            discharge_ax.plot(q_act_arr, 'r-', label='Actual Flow',
                              lw=0.8)
            # discharge_ax.plot(q_sim_arr,
            #                   'b-',
            #                   label='Simulated Flow',
            #                   lw=0.5,
            #                   alpha=0.5)
            discharge_ax.plot(sim_mean,
                              'b-',
                              label='Simulated Flow',
                              lw=0.5,
                              alpha=0.5)

        if valid_flags[1] == True:
            discharge_ax.plot(q_ext_arr,
                          'darkgrey',
                          label='Simulated {} Flow'.format(ext_model_name),
                          lw=0.5,
                          alpha=0.5)

        vol_err_ax.axhline(1.0, color='k', lw=1)
        vol_err_ax.plot(vol_diff_arr,
                        lw=0.5,
                        label='Cumm. Runoff Error',
                        alpha=0.95)
        if valid_flags[1] == True:
            vol_err_ax.plot(vol_diff_ext_arr,
                            'darkgrey',
                            lw=0.5,
                            label='Cumm. Runoff {} Error'.format(ext_model_name),
                            alpha=0.95)
        vol_err_ax.set_ylim(min_vol_diff_err, max_vol_diff_err)
        vol_err_ax.text(vol_diff_arr.shape[0] * 0.5,
                        1.25,
                        'runoff over-estimation',
                        fontsize=font_size * 0.9)
        vol_err_ax.text(vol_diff_arr.shape[0] * 0.5,
                        0.75,
                        'runoff under-estimation',
                        fontsize=font_size * 0.9)

        scatt_size = 5

        balance_ax.axhline(1.0, color='k', lw=1)
        balance_ax.scatter(bal_idxs,
                           act_bal_arr,
                           c='r',
                           marker='o',
                           label='Actual Outflow (Q + ET)',
                           alpha=0.6,
                           s=scatt_size)
        balance_ax.scatter(bal_idxs,
                           sim_bal_arr,
                           c='b',
                           marker='+',
                           label='Simulated Outflow (Q + ET)',
                           alpha=0.6,
                           s=scatt_size)
        if valid_flags[1] == True:
            balance_ax.scatter(bal_idxs,
                               sim_bal_ext_arr,
                               c='darkgrey',
                               marker='*',
                               label='Simulated {} Outflow (Q + ET)'.format(ext_model_name),
                               alpha=0.6,
                               s=scatt_size)
        balance_ax.set_ylim(min_vol_err_wET, max_vol_err_wET)

        prec_ax.bar(bar_x,
                    prec_arr,
                    label='Precipitation',
                    edgecolor='none',
                    width=1.0)
        prec_sums_ax = prec_ax.twinx()
        prec_sums_ax.scatter(bal_idxs,
                             prec_sum_arr,
                             c='b',
                             label='Precipitation Sum',
                             alpha=0.5,
                             s=scatt_size)

        temp_ax.plot(temp_arr, 'b-', lw=0.5, label='Temperature')
        temp_ax.plot(temp_trend, 'b-.', lw=0.9, label='Temperature Trend')
        temp_ax.text(temp_ax.get_xlim()[1] * 0.02,
                     (temp_ax.get_ylim()[0] +
                      (temp_ax.get_ylim()[1] - temp_ax.get_ylim()[0]) * 0.1),
                     temp_stats_str,
                     fontsize=font_size,
                     bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 2})

        pet_ax.plot(pet_arr, 'r-', lw=0.8,
                    label='Potential Evapotranspiration')
        pet_ax.plot(evap_arr,
                    'b-',
                    lw=0.5,
                    label='Evapotranspiration',
                    alpha=0.5)
        if valid_flags[1] == True:
            pet_ax.plot(evap_ext_arr,
                        'darkgrey',
                        lw=0.5,
                        label='Evapotranspiration {}'.format(ext_model_name),
                        alpha=0.5)
        pet_ax.plot(pet_trend, 'r-.', lw=0.9, label='PET Trend')
        pet_ax.plot(et_trend, 'b-.', lw=0.9, label='ET Trend')
        if valid_flags[1] == True:
            pet_ax.plot(et_ext_trend, color='darkgrey', linestyle='-.', lw=0.9, label='ET Trend')
        pet_ax.text(pet_ax.get_xlim()[1] * 0.02,
                    (pet_ax.get_ylim()[1] -
                     (pet_ax.get_ylim()[1] - pet_ax.get_ylim()[0]) * 0.1),
                    pet_stats_str,
                    fontsize=font_size,
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 2})

        snow_ax.plot(snow_arr, lw=0.5, label='Snow')
        if valid_flags[1] == True:
            snow_ax.plot(snow_ext_arr, 'darkgrey', lw=0.5, label='{} Snow'.format(ext_model_name))
        liqu_ax.bar(bar_x,
                    liqu_arr,
                    width=1.0,
                    edgecolor='none',
                    label='Liquid Precipitation')
        sm_ax.plot(sm_arr, lw=0.5, label='Soil Moisture')
        sm_ax.axhline(fc, color='r', ls='-.', lw=1, label='fc')
        sm_ax.axhline(pwp, color='b', ls='-.', lw=1, label='pwp')
        tot_run_ax.bar(bar_x,
                       tot_run_arr,
                       width=1.0,
                       label='Total Runoff',
                       edgecolor='none')
        u_res_sto_ax.plot(ur_sto_arr, lw=0.5,
                          label='Upper Reservoir - Storage')
        ur_run_uu_ax.bar(bar_x,
                         ur_run_uu,
                         label='Upper Reservoir - Quick Runoff',
                         width=2.0,
                         edgecolor='none')
        ur_run_ul_ax.bar(bar_x,
                         ur_run_ul,
                         label='Upper Reservoir - Slow Runoff',
                         width=1.0,
                         edgecolor='none')
        ur_to_lr_run_ax.bar(bar_x,
                            ur_to_lr_run,
                            label='Upper Reservoir  - Percolation',
                            width=1.0,
                            edgecolor='none')
        l_res_sto_ax.plot(lr_sto_arr, lw=0.5,
                          label='Lower Reservoir - Storage')

        snow_ax.fill_between(bar_x, 0, snow_arr, alpha=0.3)
        sm_ax.fill_between(bar_x, 0, sm_arr, alpha=0.3)
        u_res_sto_ax.fill_between(bar_x, 0, ur_sto_arr, alpha=0.3)
        l_res_sto_ax.fill_between(bar_x, 0, lr_sto_arr, alpha=0.3)

        text = np.array([
            ('Max. actual Q = %0.4f' % q_act_arr[off_idx:].max()).rstrip('0'),
            ('Max. sim. Q = %0.4f' %
             q_sim_arr[off_idx:].max()).rstrip('0'),
            ('Min. actual Q = %0.4f' %
             q_act_arr[off_idx:].min()).rstrip('0'),
            ('Min. sim. Q = %0.4f' %
             q_sim_arr[off_idx:].min()).rstrip('0'),
            ('Mean actual Q = %0.4f' %
             np.mean(q_act_arr[off_idx:])).rstrip('0'),
            ('Mean sim. Q = %0.4f' %
             np.mean(q_sim_arr[off_idx:])).rstrip('0'),
            ('Max. input P = %0.4f' % prec_arr.max()).rstrip('0'),
            ('Max. SD = %0.2f' % snow_arr[off_idx:].max()).rstrip('0'),
            ('Max. liquid P = %0.4f' %
             liqu_arr[off_idx:].max()).rstrip('0'),
            ('Max. input T = %0.2f' %
             temp_arr[off_idx:].max()).rstrip('0'),
            ('Min. input T = %0.2f' %
             temp_arr[off_idx:].min()).rstrip('0'),
            ('Max. PET = %0.2f' % pet_arr[off_idx:].max()).rstrip('0'),
            ('Max. sim. ET = %0.2f' %
             evap_arr[off_idx:].max()).rstrip('0'),
            ('Min. PET = %0.2f' % pet_arr[off_idx:].min()).rstrip('0'),
            ('Min. sim. ET = %0.2f' %
             evap_arr[off_idx:].min()).rstrip('0'),
            ('Max. sim. SM = %0.2f' %
             sm_arr[off_idx:].max()).rstrip('0'),
            ('Min. sim. SM = %0.2f' %
             sm_arr[off_idx:].min()).rstrip('0'),
            ('Mean sim. SM = %0.2f' %
             np.mean(sm_arr[off_idx:])).rstrip('0'),
            'Warm up steps = %d' % off_idx,
            'use_obs_flow_flag = %s' % use_obs_flow_flag,
            ('NS = %0.4f' % ns).rstrip('0'),
            ('Ln-NS = %0.4f' % ln_ns).rstrip('0'),
            ('KG = %0.4f' % kge).rstrip('0'),
            ('P_Corr = %0.4f' % q_correl).rstrip('0'),
            ('TT = %0.4f' % tt).rstrip('0'),
            ('CM = %0.4f' % cm).rstrip('0'),
            ('P_CM = %0.4f' % p_cm).rstrip('0'),
            ('FC = %0.4f' % fc).rstrip('0'),
            (r'$\beta$ = %0.4f' % beta).rstrip('0'),
            ('PWP = %0.4f' % pwp).rstrip('0'),
            ('UR$_{thresh}$ = %0.4f' % ur_thresh).rstrip('0'),
            ('$K_{uu}$ = %0.4f' % k_uu).rstrip('0'),
            ('$K_{ul}$ = %0.4f' % k_ul).rstrip('0'),
            ('$K_d$ = %0.4f' % k_d).rstrip('0'),
            ('$K_{ll}$ = %0.4f' % k_ll).rstrip('0'),
            ('$NS_{ext}$ = %0.4f'% ns_ext).rstrip('0')])  #

        text = text.reshape(6, 6)
        table = params_ax.table(cellText=text,
                                loc='center',
                                bbox=(0, 0, 1, 1),
                                cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        params_ax.set_axis_off()

        leg_font = f_props()
        leg_font.set_size(font_size - 1)

        plot_axes = [discharge_ax,
                     prec_ax,
                     temp_ax,
                     pet_ax,
                     snow_ax,
                     liqu_ax,
                     sm_ax,
                     tot_run_ax,
                     u_res_sto_ax,
                     ur_run_uu_ax,
                     ur_run_ul_ax,
                     ur_to_lr_run_ax,
                     l_res_sto_ax,
                     vol_err_ax,
                     balance_ax]
        for ax in plot_axes:
            for tlab in ax.get_yticklabels():
                tlab.set_fontsize(font_size - 1)
            for tlab in ax.get_xticklabels():
                tlab.set_fontsize(font_size - 1)
            ax.grid()
            leg = ax.legend(framealpha=0.7, prop=leg_font, ncol=4, loc=1)
            leg.get_frame().set_edgecolor('black')
            ax.set_xlim(0, discharge_ax.get_xlim()[1])

        for ax in [prec_sums_ax]:
            for tlab in ax.get_yticklabels():
                tlab.set_fontsize(font_size - 1)
            for tlab in ax.get_xticklabels():
                tlab.set_fontsize(font_size - 1)
            leg = ax.legend(framealpha=0.7, prop=leg_font, ncol=4, loc=2)
            leg.get_frame().set_edgecolor('black')
            ax.set_xlim(0, discharge_ax.get_xlim()[1])

        plt.tight_layout(rect=[0, 0.01, 1, 0.95], h_pad=0.0)
        plt.savefig(out_fig_loc, bbox='tight_layout')
        plt.close('all')
        return

    def save_water_bal_opt(out_dir):
        '''Save the output of the optimize function

        Just the water balance part
        '''

        assert np.all(np.isfinite(prms_arr)), 'Invalid HBV parameters!'
        assert q_act_arr.shape[0] == q_sim_arr.shape[0], (
            'Original and simulated discharge have unequal steps!')

        if not os.path.exists(out_dir):
            try:
                os.mkdir(out_dir)
            except:
                pass

        out_fig_loc = os.path.join(
            out_dir, f'kf_{kf_str}_HBV_water_bal_{cat}.png')

        cum_q = np.cumsum(q_act_arr_diff[off_idx:] / conv_ratio)

        cum_q_sim = np.cumsum(comb_run_arr[off_idx:])

        if valid_flags[1] == True:
            cum_q_ext = np.cumsum(q_ext_arr_diff[off_idx:] / conv_ratio)

        min_vol_diff_err = 0.5
        max_vol_diff_err = 1.5

        vol_diff_arr = cum_q_sim / cum_q
        vol_diff_arr[vol_diff_arr < min_vol_diff_err +
                     0.05] = min_vol_diff_err + 0.05
        vol_diff_arr[vol_diff_arr > max_vol_diff_err -
                     0.05] = max_vol_diff_err - 0.05
        vol_diff_arr = np.concatenate((np.full(shape=(off_idx - 1),
                                               fill_value=np.nan),
                                       vol_diff_arr))

        if valid_flags[1] == True:
            vol_diff_ext_arr = cum_q_ext / cum_q

            vol_diff_ext_arr[vol_diff_ext_arr < min_vol_diff_err + 0.05] = (
                    min_vol_diff_err + 0.05)
            vol_diff_ext_arr[vol_diff_ext_arr > max_vol_diff_err - 0.05] = (
                    max_vol_diff_err - 0.05)
            vol_diff_ext_arr = np.concatenate((np.full(shape=(off_idx - 1),
                                       fill_value=np.nan),
                                               vol_diff_ext_arr))

        bal_idxs = [0]
        bal_idxs.extend(list(range(off_idx, n_recs, wat_bal_stps)))

        act_bal_w_et_arr = []
        sim_bal_w_et_arr = []

        act_bal_wo_et_arr = []
        sim_bal_wo_et_arr = []

        prec_sum_arr = []
        evap_sum_arr = []
        q_act_sum_arr = []
        q_sim_sum_arr = []
        if valid_flags[1] == True:
            q_ext_sum_arr = []
            sim_bal_ext_wo_et_arr = []
            sim_bal_ext_w_et_arr = []

        min_vol_ratio_err = 0
        max_vol_ratio_err = 1.5

        for i in range(1, len(bal_idxs) - 1):
            _curr_q_act = q_act_arr_diff[bal_idxs[i]:bal_idxs[i + 1]]

            _curr_q_act_sum = np.sum(_curr_q_act / conv_ratio)
            if valid_flags[1] == True:
                _curr_q_ext = q_ext_arr_diff[
                              bal_idxs[i]:bal_idxs[i + 1]]
                _curr_q_ext_sum = np.sum(_curr_q_ext / conv_ratio)

            _prec_sum = np.sum(prec_arr[bal_idxs[i]:bal_idxs[i + 1]])
            _evap_sum = np.sum(evap_arr[bal_idxs[i]:bal_idxs[i + 1]])
            if valid_flags[1]==True:
                _evap_ext_sum = np.sum(evap_ext_arr[bal_idxs[i]:bal_idxs[i + 1]])

            _curr_comb_sum = np.sum(comb_run_arr[bal_idxs[i]:bal_idxs[i + 1]])

            # ET accounted for
            act_bal_w_et_arr.append(_curr_q_act_sum / (_prec_sum - _evap_sum))
            sim_bal_w_et_arr.append(_curr_comb_sum / (_prec_sum - _evap_sum))
            if valid_flags[1] == True:
                sim_bal_ext_w_et_arr.append(
                    _curr_q_ext_sum / (_prec_sum -_evap_ext_sum))

            # ET not accounted for
            act_bal_wo_et_arr.append(_curr_q_act_sum / _prec_sum)
            sim_bal_wo_et_arr.append(_curr_comb_sum / _prec_sum)

            if valid_flags[1] == True:
                sim_bal_ext_wo_et_arr.append(_curr_q_ext_sum / _prec_sum)

            prec_sum_arr.append(_prec_sum)
            evap_sum_arr.append(_evap_sum)
            q_act_sum_arr.append(_curr_q_act_sum)
            q_sim_sum_arr.append(_curr_comb_sum)
            if valid_flags[1] == True:
                q_ext_sum_arr.append(_curr_q_ext_sum)

        act_bal_w_et_arr = np.array(act_bal_w_et_arr)
        act_bal_wo_et_arr = np.array(act_bal_wo_et_arr)

        act_bal_w_et_arr[act_bal_w_et_arr < min_vol_ratio_err] = (
            min_vol_ratio_err)
        act_bal_w_et_arr[act_bal_w_et_arr > max_vol_ratio_err] = (
            max_vol_ratio_err)

        act_bal_w_et_arr = (
            np.concatenate(([np.nan, np.nan], act_bal_w_et_arr), axis=0))
        sim_bal_w_et_arr = (
            np.concatenate(([np.nan, np.nan], sim_bal_w_et_arr), axis=0))

        if valid_flags[1] == True:
            sim_bal_ext_w_et_arr = (
                np.concatenate(([np.nan, np.nan], sim_bal_ext_w_et_arr),
                               axis=0))

        act_bal_wo_et_arr[act_bal_wo_et_arr < min_vol_ratio_err] = (
            min_vol_ratio_err)
        act_bal_wo_et_arr[act_bal_wo_et_arr > max_vol_ratio_err] = (
            max_vol_ratio_err)

        act_bal_wo_et_arr = (
            np.concatenate(([np.nan, np.nan], act_bal_wo_et_arr), axis=0))
        sim_bal_wo_et_arr = (
            np.concatenate(([np.nan, np.nan], sim_bal_wo_et_arr), axis=0))

        if valid_flags[1] == True:
            sim_bal_ext_wo_et_arr = (
                np.concatenate(([np.nan, np.nan], sim_bal_ext_wo_et_arr),
                               axis=0))

        prec_sum_arr = np.concatenate(([np.nan, np.nan], prec_sum_arr), axis=0)
        evap_sum_arr = np.concatenate(([np.nan, np.nan], evap_sum_arr), axis=0)

        q_act_sum_arr = (
            np.concatenate(([np.nan, np.nan], q_act_sum_arr), axis=0))
        q_sim_sum_arr = (
            np.concatenate(([np.nan, np.nan], q_sim_sum_arr), axis=0))

        if valid_flags[1] == True:
            q_ext_sum_arr = (
                np.concatenate(([np.nan, np.nan], q_ext_sum_arr), axis=0))

        font_size = 5
        plt.figure(figsize=(11, 6), dpi=150)
        t_rows = 8
        t_cols = 1

        plt.suptitle('HBV Flow Simulation - Water Balance')

        i = 0
        params_ax = plt.subplot2grid((t_rows, t_cols),
                                     (i, 0),
                                     rowspan=1,
                                     colspan=t_cols)
        i += 1
        vol_err_ax = plt.subplot2grid((t_rows, t_cols),
                                      (i, 0),
                                      rowspan=1,
                                      colspan=t_cols)
        i += 1
        discharge_ax = plt.subplot2grid((t_rows, t_cols),
                                        (i, 0),
                                        rowspan=2,
                                        colspan=t_cols)
        i += 2
        balance_ratio_ax = plt.subplot2grid((t_rows, t_cols),
                                            (i, 0),
                                            rowspan=2,
                                            colspan=t_cols)
        i += 2
        balance_sum_ax = plt.subplot2grid((t_rows, t_cols),
                                          (i, 0),
                                          rowspan=2,
                                          colspan=t_cols)
        # i += 2

        vol_err_ax.axhline(1, color='k', lw=1)
        vol_err_ax.plot(vol_diff_arr,
                        lw=0.5,
                        label='Cumm. Runoff Error',
                        alpha=0.95)
        if valid_flags[1] == True:
            vol_err_ax.plot(vol_diff_ext_arr,
                            'darkgrey',
                            lw=0.5,
                            label='Cumm. Runoff Error {}'.format(ext_model_name),
                            alpha=0.95)
        vol_err_ax.set_ylim(min_vol_diff_err, max_vol_diff_err)
        vol_err_ax.set_xlim(0, vol_err_ax.get_xlim()[1])
        vol_err_ax.text(vol_diff_arr.shape[0] * 0.5,
                        1.25,
                        'runoff over-estimation',
                        fontsize=font_size * 0.9)
        vol_err_ax.text(vol_diff_arr.shape[0] * 0.5,
                        0.75,
                        'runoff under-estimation',
                        fontsize=font_size * 0.9)
        vol_err_ax.set_xticklabels([])

        discharge_ax.plot(q_act_arr,
                          'r-',
                          label='Actual Flow',
                          lw=0.8,
                          alpha=0.7)
        discharge_ax.plot(q_sim_arr,
                          'b-',
                          label='Simulated Flow',
                          lw=0.5,
                          alpha=0.5)
        if valid_flags[1] == True:
            discharge_ax.plot(q_ext_arr,
                              'darkgrey',
                              label='{} Flow'.format(ext_model_name),
                              lw=0.5,
                              alpha=0.5)
        discharge_ax.set_xlim(0, vol_err_ax.get_xlim()[1])
        discharge_ax.set_xticklabels([])

        scatt_size = 5

        balance_ratio_ax.axhline(1, color='k', lw=1)
        balance_ratio_ax.scatter(bal_idxs,
                                 act_bal_w_et_arr,
                                 marker='o',
                                 label='Actual Outflow (Q + ET)',
                                 alpha=0.6,
                                 s=scatt_size)
        balance_ratio_ax.scatter(bal_idxs, sim_bal_w_et_arr,
                                 marker='+',
                                 label='Simulated Outflow (Q + ET)',
                                 alpha=0.6,
                                 s=scatt_size)

        if valid_flags[1] == True:
            balance_ratio_ax.scatter(bal_idxs, sim_bal_ext_w_et_arr,
                                     marker='*',
                                     label='{} Outflow (Q + ET)'.format(ext_model_name),
                                     alpha=0.6,
                                     s=scatt_size)

        balance_ratio_ax.scatter(bal_idxs,
                                 act_bal_wo_et_arr,
                                 marker='o',
                                 label='Actual Outflow (Q)',
                                 alpha=0.6,
                                 s=scatt_size)
        balance_ratio_ax.scatter(bal_idxs,
                                 sim_bal_wo_et_arr,
                                 marker='+',
                                 label='Simulated Outflow (Q)',
                                 alpha=0.6,
                                 s=scatt_size)
        if valid_flags[1] == True:
            balance_ratio_ax.scatter(bal_idxs,
                                     sim_bal_ext_wo_et_arr,
                                     marker='*',
                                     label='Shertan Outflow (Q)',
                                     alpha=0.6,
                                     s=scatt_size)

        balance_ratio_ax.set_xlim(0, vol_err_ax.get_xlim()[1])
        balance_ratio_ax.set_ylim(min_vol_ratio_err, max_vol_ratio_err)
        balance_ratio_ax.set_xticklabels([])

        balance_sum_ax.scatter(bal_idxs,
                               prec_sum_arr,
                               alpha=0.6,
                               marker='o',
                               label='Precipitation Sum',
                               s=scatt_size)
        balance_sum_ax.scatter(bal_idxs,
                               evap_sum_arr,
                               alpha=0.6,
                               marker='+',
                               label='Evapotranspiration Sum',
                               s=scatt_size)
        balance_sum_ax.scatter(bal_idxs,
                               q_act_sum_arr,
                               alpha=0.6,
                               marker='o',
                               label='Actual Runoff Sum',
                               s=scatt_size)
        balance_sum_ax.scatter(bal_idxs,
                               q_sim_sum_arr,
                               alpha=0.6,
                               marker='+',
                               label='Simulated Runoff Sum',
                               s=scatt_size)
        if valid_flags[1] == True:
            balance_sum_ax.scatter(bal_idxs,
                               q_ext_sum_arr,
                               alpha=0.6,
                               marker='*',
                               label='{} Runoff Sum'.format(ext_model_name),
                               s=scatt_size)
        balance_sum_ax.set_xlim(0, vol_err_ax.get_xlim()[1])
        balance_sum_ax.set_ylim(0, balance_sum_ax.get_ylim()[1])

        text = np.array([
            ('Max. actual Q = %0.4f' % q_act_arr[off_idx:].max()).rstrip('0'),
            ('Max. sim. Q = %0.4f' %
             q_sim_arr[off_idx:].max()).rstrip('0'),
            ('Min. actual Q = %0.4f' %
             q_act_arr[off_idx:].min()).rstrip('0'),
            ('Min. sim. Q = %0.4f' %
             q_sim_arr[off_idx:].min()).rstrip('0'),
            ('Mean actual Q = %0.4f' %
             np.mean(q_act_arr[off_idx:])).rstrip('0'),
            ('Mean sim. Q = %0.4f' %
             np.mean(q_sim_arr[off_idx:])).rstrip('0'),
            ('Max. input P = %0.4f' % prec_arr.max()).rstrip('0'),
            ('Mean sim. ET = %0.4f' %
             evap_arr[off_idx:].mean()).rstrip('0'),
            'Warm up steps = %d' % off_idx,
            'use_obs_flow_flag = %s' % use_obs_flow_flag,
            ('NS = %0.4f' % ns).rstrip('0'),
            ('Ln-NS = %0.4f' % ln_ns).rstrip('0'),
            ('KG = %0.4f' % kge).rstrip('0'),
            ('P_Corr = %0.4f' % q_correl).rstrip('0'),
            ('TT = %0.4f' % tt).rstrip('0'),
            ('CM = %0.4f' % cm).rstrip('0'),
            ('P_CM = %0.4f' % p_cm).rstrip('0'),
            ('FC = %0.4f' % fc).rstrip('0'),
            (r'$\beta$ = %0.4f' % beta).rstrip('0'),
            ('PWP = %0.4f' % pwp).rstrip('0'),
            ('UR$_{thresh}$ = %0.4f' % ur_thresh).rstrip('0'),
            ('$K_{uu}$ = %0.4f' % k_uu).rstrip('0'),
            ('$K_{ul}$ = %0.4f' % k_ul).rstrip('0'),
            ('$K_d$ = %0.4f' % k_d).rstrip('0'),
            ('$K_{ll}$ = %0.4f' % k_ll).rstrip('0'),
            'n_cells = %d' % n_cells, ''])

        text = text.reshape(3, 9)
        table = params_ax.table(cellText=text,
                                loc='center',
                                bbox=(0, 0, 1, 1),
                                cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        params_ax.set_axis_off()

        leg_font = f_props()
        leg_font.set_size(font_size)

        plot_axes = [discharge_ax, vol_err_ax,
                     balance_ratio_ax, balance_sum_ax]
        for ax in plot_axes:
            for tlab in ax.get_yticklabels():
                tlab.set_fontsize(font_size)
            ax.grid()
            ax.set_aspect('auto')
            leg = ax.legend(framealpha=0.7, prop=leg_font, ncol=4)
            leg.get_frame().set_edgecolor('black')

        for tlab in balance_sum_ax.get_xticklabels():
            tlab.set_fontsize(font_size)

        plt.tight_layout(rect=[0, 0.01, 1, 0.95], h_pad=0.0)
        plt.savefig(out_fig_loc, bbox_inches='tight')
        plt.close('all')
        return

    if plot_simple_flag:
        save_simple_opt(hbv_figs_dir)

    if plot_wat_bal_flag:
        save_water_bal_opt(os.path.join(out_dir, '04_wat_bal_figs'))

    return
