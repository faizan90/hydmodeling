"""
Created on %(date)s

@author: %(Faizan Anwar, IWS Uni-Stuttgart)s

"""

import os

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties as f_props

from ..models import (
    hbv_loop_py,
    get_ns_cy,
    get_ln_ns_cy,
    get_pcorr_cy,
    get_kge_cy,
    lin_regsn_cy)

plt.ioff()


def plot_cat_hbv_sim(plot_args):

    '''Plot all HBV variables along with some model information for all
    kfolds.
    '''

    cat_db, (wat_bal_stps,
             full_sim_flag,
             wat_bal_flag) = plot_args

    with h5py.File(cat_db, 'r') as db:
        out_dir = db['data'].attrs['main']
        kfolds = db['data'].attrs['kfolds']
        cat = db.attrs['cat']

        off_idx = db['data'].attrs['off_idx']
        conv_ratio = db['data'].attrs['conv_ratio']
        prm_syms = db['data/all_prms_labs'][...]
        use_obs_flow_flag = db['data'].attrs['use_obs_flow_flag']
        area_arr = db['data/area_arr'][...]

        cv_flag = db['data'].attrs['cv_flag']

        if cv_flag:
            cv_kf_dict = {}
            for i in range(1, kfolds + 1):
                cd_db = db[f'valid/kf_{i:02d}']
                kf_dict = {key: cd_db[key][...] for key in cd_db}
                kf_dict['use_obs_flow_flag'] = use_obs_flow_flag

                cv_kf_dict[i] = kf_dict
                break

        all_kfs_dict = {}
        for i in range(1, kfolds + 1):
            cd_db = db[f'calib/kf_{i:02d}']
            kf_dict = {key: cd_db[key][...] for key in cd_db}
            kf_dict['use_obs_flow_flag'] = use_obs_flow_flag

            all_kfs_dict[i] = kf_dict

        if kfolds > 1:
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
                    all_kfs_dict[i]['extra_us_inflow'] for i in all_kfs_dict],
                    axis=0)

        for i in range(1, kfolds + 1):
            kf_dict = all_kfs_dict[i]
            kf_i = f'{i:02d}_calib'

            plot_cat_hbv_sim_kf(
                kf_i,
                cat,
                kf_dict,
                area_arr,
                conv_ratio,
                prm_syms,
                off_idx,
                out_dir,
                wat_bal_stps,
                full_sim_flag,
                wat_bal_flag)

            if (kfolds == 1) and (not cv_flag):
                continue

            kf_i = f'{i:02d}_valid'

            if cv_flag:
                kf_dict['tem_arr'] = cv_kf_dict[i]['tem_arr']
                kf_dict['ppt_arr'] = cv_kf_dict[i]['ppt_arr']
                kf_dict['pet_arr'] = cv_kf_dict[i]['pet_arr']
                kf_dict['qact_arr'] = cv_kf_dict[i]['qact_arr']

                if 'extra_us_inflow' in kf_dict:
                    kf_dict['extra_us_inflow'] = (
                        cv_kf_dict[i]['extra_us_inflow'])

            else:
                kf_dict['tem_arr'] = all_tem_arr
                kf_dict['ppt_arr'] = all_ppt_arr
                kf_dict['pet_arr'] = all_pet_arr
                kf_dict['qact_arr'] = all_qact_arr

                if 'extra_us_inflow' in kf_dict:
                    kf_dict['extra_us_inflow'] = all_us_inflow_arr

            plot_cat_hbv_sim_kf(
                kf_i,
                cat,
                kf_dict,
                area_arr,
                conv_ratio,
                prm_syms,
                off_idx,
                out_dir,
                wat_bal_stps,
                full_sim_flag,
                wat_bal_flag)

    return


def plot_cat_hbv_sim_kf(
        kf_str,
        cat,
        kf_dict,
        area_arr,
        conv_ratio,
        prm_syms,
        off_idx,
        out_dir,
        wat_bal_stps,
        full_sim_flag,
        wat_bal_flag):

    '''Plot all HBV variables for a given kfold.'''

    if 'use_obs_flow_flag' in kf_dict:
        use_obs_flow_flag = bool(kf_dict['use_obs_flow_flag'])

    else:
        use_obs_flow_flag = False

    rarea_arr = area_arr.reshape(-1, 1)
    rrarea_arr = area_arr.reshape(-1, 1, 1)

    temp_dist_arr = kf_dict['tem_arr']
    prec_dist_arr = kf_dict['ppt_arr']
    pet_dist_arr = kf_dict['pet_arr']
    prms_dist_arr = kf_dict['hbv_prms']
    prms_arr = (rarea_arr * prms_dist_arr).sum(axis=0)

    n_recs = temp_dist_arr.shape[1]
    n_cells = temp_dist_arr.shape[0]

    all_outputs_dict = hbv_loop_py(
        temp_dist_arr,
        prec_dist_arr,
        pet_dist_arr,
        prms_dist_arr,
        kf_dict['ini_arr'],
        area_arr,
        conv_ratio)

    assert all_outputs_dict['loop_ret'] == 0.0

    temp_arr = (rarea_arr * temp_dist_arr).sum(axis=0)
    prec_arr = (rarea_arr * prec_dist_arr).sum(axis=0)
    pet_arr = (rarea_arr * pet_dist_arr).sum(axis=0)

    del temp_dist_arr, prec_dist_arr, pet_dist_arr, prms_dist_arr

    all_output = all_outputs_dict['outs_arr']
    all_output = (rrarea_arr * all_output).sum(axis=0)

    q_sim_arr = all_outputs_dict['qsim_arr']

    del all_outputs_dict

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

    extra_us_inflow_flag = 'extra_us_inflow' in kf_dict

    q_sim_arr = (q_sim_arr).copy(order='C')
    q_act_arr = kf_dict['qact_arr']
    q_act_arr_diff = q_act_arr.copy()

    if extra_us_inflow_flag:
        extra_us_inflow = kf_dict['extra_us_inflow']
        q_sim_arr = q_sim_arr + extra_us_inflow
        q_act_arr_diff = q_act_arr - extra_us_inflow

    ns = get_ns_cy(q_act_arr, q_sim_arr, off_idx)
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

    cum_q = np.cumsum(q_act_arr_diff[off_idx:] / conv_ratio)
    cum_q_sim = np.cumsum(comb_run_arr[off_idx:])

    min_vol_diff_err = 0.5
    max_vol_diff_err = 1.5

    vol_diff_arr = cum_q_sim / cum_q

    vol_diff_arr[vol_diff_arr < min_vol_diff_err + 0.05] = (
        min_vol_diff_err + 0.05)

    vol_diff_arr[vol_diff_arr > max_vol_diff_err - 0.05] = (
        max_vol_diff_err - 0.05)

    vol_diff_arr = np.concatenate(
        (np.full(shape=(off_idx - 1), fill_value=np.nan), vol_diff_arr))

    bal_idxs = [0]
    bal_idxs.extend(list(range(off_idx, n_recs, wat_bal_stps)))

    if wat_bal_flag:
        wat_bals_dir = os.path.join(out_dir, '04_wat_bal_figs')

        if not os.path.exists(wat_bals_dir):
            try:
                os.mkdir(wat_bals_dir)

            except FileExistsError:
                pass

        wat_bal_text = np.array([
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

        wat_bal_sim(
            kf_str,
            cat,
            conv_ratio,
            wat_bal_text,
            bal_idxs,
            wat_bals_dir,
            prms_arr,
            q_act_arr,
            q_sim_arr,
            q_act_arr_diff,
            comb_run_arr,
            prec_arr,
            evap_arr,
            vol_diff_arr,
            min_vol_diff_err,
            max_vol_diff_err)

    if full_sim_flag:
        full_sims_dir = os.path.join(out_dir, '03_hbv_figs')
        if not os.path.exists(full_sims_dir):
            try:
                os.mkdir(full_sims_dir)

            except FileExistsError:
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
            full_sims_dir, f'kf_{kf_str}_HBV_sim_{cat}.csv'), sep=';')

        out_labs = []
        out_labs.extend(prm_syms)

        if 'route_labs' in kf_dict:
            out_labs.extend(kf_dict['route_labs'])

        out_labs.extend(['ns', 'ln_ns', 'kge', 'obj_ftn', 'p_corr'])

        out_params_df = pd.DataFrame(index=out_labs, columns=['value'])
        out_params_df['value'] = (
            np.concatenate(
                (prms_arr, [ns, ln_ns, kge, 'nothing_yet', q_correl])))

        out_params_loc = os.path.join(
            full_sims_dir, f'kf_{kf_str}_HBV_model_params_{cat}.csv')

        out_params_df.to_csv(
            out_params_loc, sep=str(';'), index_label='param')

        full_sim_text = np.array([
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
            ''])

        full_sim(
            kf_str,
            cat,
            conv_ratio,
            n_recs,
            fc,
            pwp,
            full_sims_dir,
            prms_arr,
            q_act_arr,
            q_sim_arr,
            bal_idxs,
            q_act_arr_diff,
            prec_arr,
            evap_arr,
            comb_run_arr,
            temp_arr,
            pet_arr,
            vol_diff_arr,
            min_vol_diff_err,
            max_vol_diff_err,
            snow_arr,
            liqu_arr,
            sm_arr,
            tot_run_arr,
            ur_sto_arr,
            ur_run_uu,
            ur_run_ul,
            lr_sto_arr,
            ur_to_lr_run,
            full_sim_text)

    return


def wat_bal_sim(
        kf_str,
        cat,
        conv_ratio,
        wat_bal_text,
        bal_idxs,
        out_dir,
        prms_arr,
        q_act_arr,
        q_sim_arr,
        q_act_arr_diff,
        comb_run_arr,
        prec_arr,
        evap_arr,
        vol_diff_arr,
        min_vol_diff_err,
        max_vol_diff_err):

    '''Plot some water balance variables.'''

    assert np.all(np.isfinite(prms_arr)), 'Invalid HBV parameters!'
    assert q_act_arr.shape[0] == q_sim_arr.shape[0], (
        'Original and simulated discharge have unequal steps!')

    if not os.path.exists(out_dir):
        try:
            os.mkdir(out_dir)

        except FileExistsError:
            pass

    out_fig_loc = os.path.join(
        out_dir, f'kf_{kf_str}_HBV_water_bal_{cat}.png')

    act_bal_w_et_arr = []
    sim_bal_w_et_arr = []

    act_bal_wo_et_arr = []
    sim_bal_wo_et_arr = []

    prec_sum_arr = []
    evap_sum_arr = []
    q_act_sum_arr = []
    q_sim_sum_arr = []

    min_vol_ratio_err = 0
    max_vol_ratio_err = 1.5

    for i in range(1, len(bal_idxs) - 1):
        curr_q_act = q_act_arr_diff[bal_idxs[i]:bal_idxs[i + 1]]
        curr_q_act_sum = np.sum(curr_q_act / conv_ratio)

        prec_sum = np.sum(prec_arr[bal_idxs[i]:bal_idxs[i + 1]])
        evap_sum = np.sum(evap_arr[bal_idxs[i]:bal_idxs[i + 1]])

        curr_comb_sum = np.sum(comb_run_arr[bal_idxs[i]:bal_idxs[i + 1]])

        # ET accounted for
        act_bal_w_et_arr.append(curr_q_act_sum / (prec_sum - evap_sum))
        sim_bal_w_et_arr.append(curr_comb_sum / (prec_sum - evap_sum))

        # ET not accounted for
        act_bal_wo_et_arr.append(curr_q_act_sum / prec_sum)
        sim_bal_wo_et_arr.append(curr_comb_sum / prec_sum)

        prec_sum_arr.append(prec_sum)
        evap_sum_arr.append(evap_sum)
        q_act_sum_arr.append(curr_q_act_sum)
        q_sim_sum_arr.append(curr_comb_sum)

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

    act_bal_wo_et_arr[act_bal_wo_et_arr < min_vol_ratio_err] = (
        min_vol_ratio_err)

    act_bal_wo_et_arr[act_bal_wo_et_arr > max_vol_ratio_err] = (
        max_vol_ratio_err)

    act_bal_wo_et_arr = (
        np.concatenate(([np.nan, np.nan], act_bal_wo_et_arr), axis=0))

    sim_bal_wo_et_arr = (
        np.concatenate(([np.nan, np.nan], sim_bal_wo_et_arr), axis=0))

    prec_sum_arr = np.concatenate(
        ([np.nan, np.nan], prec_sum_arr), axis=0)

    evap_sum_arr = np.concatenate(
        ([np.nan, np.nan], evap_sum_arr), axis=0)

    q_act_sum_arr = (
        np.concatenate(([np.nan, np.nan], q_act_sum_arr), axis=0))

    q_sim_sum_arr = (
        np.concatenate(([np.nan, np.nan], q_sim_sum_arr), axis=0))

    font_size = 5
    t_rows = 8
    t_cols = 1

    plt.figure(figsize=(11, 6), dpi=150)

    plt.suptitle('HBV Simulation - Water Balance')

    i = 0
    params_ax = plt.subplot2grid(
        (t_rows, t_cols), (i, 0), rowspan=1, colspan=t_cols)
    i += 1

    vol_err_ax = plt.subplot2grid(
        (t_rows, t_cols), (i, 0), rowspan=1, colspan=t_cols)
    i += 1

    discharge_ax = plt.subplot2grid(
        (t_rows, t_cols), (i, 0), rowspan=2, colspan=t_cols)
    i += 2

    balance_ratio_ax = plt.subplot2grid(
        (t_rows, t_cols), (i, 0), rowspan=2, colspan=t_cols)
    i += 2

    balance_sum_ax = plt.subplot2grid(
        (t_rows, t_cols), (i, 0), rowspan=2, colspan=t_cols)
    # i += 2

    vol_err_ax.axhline(1, color='k', lw=1)
    vol_err_ax.plot(
        vol_diff_arr, lw=0.5, label='Cumm. Runoff Error', alpha=0.95)

    vol_err_ax.set_ylim(min_vol_diff_err, max_vol_diff_err)
    vol_err_ax.set_xlim(0, vol_err_ax.get_xlim()[1])

    vol_err_ax.text(
        vol_diff_arr.shape[0] * 0.5,
        1.25,
        'runoff over-estimation',
        fontsize=font_size * 0.9)

    vol_err_ax.text(
        vol_diff_arr.shape[0] * 0.5,
        0.75,
        'runoff under-estimation',
        fontsize=font_size * 0.9)

    vol_err_ax.set_xticklabels([])

    discharge_ax.plot(
        q_act_arr,
        'r-',
        label='Actual Flow',
        lw=0.8,
        alpha=0.7)

    discharge_ax.plot(
        q_sim_arr,
        'b-',
        label='Simulated Flow',
        lw=0.5,
        alpha=0.5)

    discharge_ax.set_xlim(0, vol_err_ax.get_xlim()[1])
    discharge_ax.set_xticklabels([])

    scatt_size = 5

    balance_ratio_ax.axhline(1, color='k', lw=1)
    balance_ratio_ax.scatter(
        bal_idxs,
        act_bal_w_et_arr,
        marker='o',
        label='Actual Outflow (Q + ET)',
        alpha=0.6,
        s=scatt_size)

    balance_ratio_ax.scatter(
        bal_idxs,
        sim_bal_w_et_arr,
        marker='+',
        label='Simulated Outflow (Q + ET)',
        alpha=0.6,
        s=scatt_size)

    balance_ratio_ax.scatter(
        bal_idxs,
        act_bal_wo_et_arr,
        marker='o',
        label='Actual Outflow (Q)',
        alpha=0.6,
        s=scatt_size)

    balance_ratio_ax.scatter(
        bal_idxs,
        sim_bal_wo_et_arr,
        marker='+',
        label='Simulated Outflow (Q)',
        alpha=0.6,
        s=scatt_size)

    balance_ratio_ax.set_xlim(0, vol_err_ax.get_xlim()[1])
    balance_ratio_ax.set_ylim(min_vol_ratio_err, max_vol_ratio_err)
    balance_ratio_ax.set_xticklabels([])

    balance_sum_ax.scatter(
        bal_idxs,
        prec_sum_arr,
        alpha=0.6,
        marker='o',
        label='Precipitation Sum',
        s=scatt_size)

    balance_sum_ax.scatter(
        bal_idxs,
        evap_sum_arr,
        alpha=0.6,
        marker='+',
        label='Evapotranspiration Sum',
        s=scatt_size)

    balance_sum_ax.scatter(
        bal_idxs,
        q_act_sum_arr,
        alpha=0.6,
        marker='o',
        label='Actual Runoff Sum',
        s=scatt_size)

    balance_sum_ax.scatter(
        bal_idxs,
        q_sim_sum_arr,
        alpha=0.6,
        marker='+',
        label='Simulated Runoff Sum',
        s=scatt_size)

    balance_sum_ax.set_xlim(0, vol_err_ax.get_xlim()[1])
    balance_sum_ax.set_ylim(0, balance_sum_ax.get_ylim()[1])

    wat_bal_text = wat_bal_text.reshape(3, 9)

    table = params_ax.table(
        cellText=wat_bal_text,
        loc='center',
        bbox=(0, 0, 1, 1),
        cellLoc='left')

    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    params_ax.set_axis_off()

    leg_font = f_props()
    leg_font.set_size(font_size)

    plot_axes = [
        discharge_ax,
        vol_err_ax,
        balance_ratio_ax,
        balance_sum_ax
        ]

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


def full_sim(
        kf_str,
        cat,
        conv_ratio,
        n_recs,
        fc,
        pwp,
        out_dir,
        prms_arr,
        q_act_arr,
        q_sim_arr,
        bal_idxs,
        q_act_arr_diff,
        prec_arr,
        evap_arr,
        comb_run_arr,
        temp_arr,
        pet_arr,
        vol_diff_arr,
        min_vol_diff_err,
        max_vol_diff_err,
        snow_arr,
        liqu_arr,
        sm_arr,
        tot_run_arr,
        ur_sto_arr,
        ur_run_uu,
        ur_run_ul,
        lr_sto_arr,
        ur_to_lr_run,
        full_sim_text):

    '''Plot time series of all HBV variables'''

    assert np.all(np.isfinite(prms_arr)), 'Invalid HBV parameters!'
    assert q_act_arr.shape[0] == q_sim_arr.shape[0], (
        'Original and simulated discharge have unequal steps!')

    if not os.path.exists(out_dir):
        try:
            os.mkdir(out_dir)

        except FileExistsError:
            pass

    out_fig_loc = os.path.join(
        out_dir, f'kf_{kf_str}_HBV_model_plot_{cat}.png')

    act_bal_arr = []
    sim_bal_arr = []

    prec_sum_arr = []

    min_vol_err_w_et = 0.5
    max_vol_err_w_et = 1.5

    for i in range(1, len(bal_idxs) - 1):
        # ET accounted for
        curr_q_act = q_act_arr_diff[bal_idxs[i]:bal_idxs[i + 1]]
        curr_q_act_sum = np.sum(curr_q_act / conv_ratio)

        curr_prec_sum = np.sum(prec_arr[bal_idxs[i]:bal_idxs[i + 1]])
        curr_evap_sum = np.sum(evap_arr[bal_idxs[i]:bal_idxs[i + 1]])

        curr_comb_sum = np.sum(comb_run_arr[bal_idxs[i]:bal_idxs[i + 1]])

        act_bal_arr.append(
            curr_q_act_sum / (curr_prec_sum - curr_evap_sum))

        sim_bal_arr.append(
            curr_comb_sum / (curr_prec_sum - curr_evap_sum))

        prec_sum_arr.append(curr_prec_sum)

    act_bal_arr = np.array(act_bal_arr)

    act_bal_arr[act_bal_arr < min_vol_err_w_et] = min_vol_err_w_et
    act_bal_arr[act_bal_arr > max_vol_err_w_et] = max_vol_err_w_et

    act_bal_arr = np.concatenate(([np.nan, np.nan], act_bal_arr), axis=0)
    sim_bal_arr = np.concatenate(([np.nan, np.nan], sim_bal_arr), axis=0)

    prec_sum_arr = np.concatenate(
        ([np.nan, np.nan], prec_sum_arr), axis=0)

    steps_range = np.arange(0., n_recs)
    temp_trend, temp_corr, temp_slope, *__ = lin_regsn_cy(
        steps_range, temp_arr, 0)

    temp_stats_str = ''
    temp_stats_str += f'correlation: {temp_corr:0.5f}, '
    temp_stats_str += f'slope: {temp_slope:0.5f}'

    pet_trend, pet_corr, pet_slope, *__ = lin_regsn_cy(
        steps_range, pet_arr, 0)

    pet_stats_str = ''
    pet_stats_str += f'PET correlation: {pet_corr:0.5f}, '
    pet_stats_str += f'PET slope: {pet_slope:0.5f}, '

    et_trend, et_corr, et_slope, *__ = lin_regsn_cy(
        steps_range, evap_arr.copy(order='c'), 0)

    pet_stats_str += f'ET correlation: {et_corr:0.5f}, '
    pet_stats_str += f'ET slope: {et_slope:0.5f}'

    plt.figure(figsize=(11, 25), dpi=250)

    t_rows = 16
    t_cols = 1
    font_size = 6

    plt.suptitle('HBV Simulation')

    i = 0

    params_ax = plt.subplot2grid(
        (t_rows, t_cols), (i, 0), rowspan=1, colspan=1)
    i += 1

    vol_err_ax = plt.subplot2grid(
        (t_rows, t_cols), (i, 0), rowspan=1, colspan=1)
    i += 1

    discharge_ax = plt.subplot2grid(
        (t_rows, t_cols), (i, 0), rowspan=1, colspan=1)
    i += 1

    balance_ax = plt.subplot2grid(
        (t_rows, t_cols), (i, 0), rowspan=1, colspan=1)
    i += 1

    prec_ax = plt.subplot2grid(
        (t_rows, t_cols), (i, 0), rowspan=1, colspan=1)
    i += 1

    liqu_ax = plt.subplot2grid(
        (t_rows, t_cols), (i, 0), rowspan=1, colspan=1)
    i += 1

    snow_ax = plt.subplot2grid(
        (t_rows, t_cols), (i, 0), rowspan=1, colspan=1)
    i += 1

    temp_ax = plt.subplot2grid(
        (t_rows, t_cols), (i, 0), rowspan=1, colspan=1)
    i += 1

    pet_ax = plt.subplot2grid(
        (t_rows, t_cols), (i, 0), rowspan=1, colspan=1)
    i += 1

    sm_ax = plt.subplot2grid(
        (t_rows, t_cols), (i, 0), rowspan=1, colspan=1)
    i += 1

    tot_run_ax = plt.subplot2grid(
        (t_rows, t_cols), (i, 0), rowspan=1, colspan=1)
    i += 1

    u_res_sto_ax = plt.subplot2grid(
        (t_rows, t_cols), (i, 0), rowspan=1, colspan=1)
    i += 1

    ur_run_uu_ax = plt.subplot2grid(
        (t_rows, t_cols), (i, 0), rowspan=1, colspan=1)
    i += 1

    ur_run_ul_ax = plt.subplot2grid(
        (t_rows, t_cols), (i, 0), rowspan=1, colspan=1)
    i += 1

    ur_to_lr_run_ax = plt.subplot2grid(
        (t_rows, t_cols), (i, 0), rowspan=1, colspan=1)
    i += 1

    l_res_sto_ax = plt.subplot2grid(
        (t_rows, t_cols), (i, 0), rowspan=1, colspan=1)

    bar_x = np.arange(0, n_recs, 1)

    discharge_ax.plot(q_act_arr, 'r-', label='Actual Flow', lw=0.8)
    discharge_ax.plot(
        q_sim_arr, 'b-', label='Simulated Flow', lw=0.5, alpha=0.5)

    vol_err_ax.axhline(1.0, color='k', lw=1)
    vol_err_ax.plot(
        vol_diff_arr, lw=0.5, label='Cumm. Runoff Error', alpha=0.95)

    vol_err_ax.set_ylim(min_vol_diff_err, max_vol_diff_err)
    vol_err_ax.text(
        vol_diff_arr.shape[0] * 0.5,
        1.25,
        'runoff over-estimation',
        fontsize=font_size * 0.9)

    vol_err_ax.text(
        vol_diff_arr.shape[0] * 0.5,
        0.75,
        'runoff under-estimation',
        fontsize=font_size * 0.9)

    scatt_size = 5

    balance_ax.axhline(1.0, color='k', lw=1)
    balance_ax.scatter(
        bal_idxs,
        act_bal_arr,
        c='r',
        marker='o',
        label='Actual Outflow (Q + ET)',
        alpha=0.6,
        s=scatt_size)
    balance_ax.scatter(
        bal_idxs,
        sim_bal_arr,
        c='b',
        marker='+',
        label='Simulated Outflow (Q + ET)',
        alpha=0.6,
        s=scatt_size)
    balance_ax.set_ylim(min_vol_err_w_et, max_vol_err_w_et)

    prec_ax.bar(
        bar_x,
        prec_arr,
        label='Precipitation',
        edgecolor='none',
        width=1.0)
    prec_sums_ax = prec_ax.twinx()
    prec_sums_ax.scatter(
        bal_idxs,
        prec_sum_arr,
        c='b',
        label='Precipitation Sum',
        alpha=0.5,
        s=scatt_size)

    temp_ax.plot(temp_arr, 'b-', lw=0.5, label='Temperature')
    temp_ax.plot(temp_trend, 'b-.', lw=0.9, label='Temperature Trend')
    temp_ax.text(
        temp_ax.get_xlim()[1] * 0.02,
        (temp_ax.get_ylim()[0] +
         (temp_ax.get_ylim()[1] - temp_ax.get_ylim()[0]) * 0.1),
        temp_stats_str,
        fontsize=font_size,
        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 2})

    pet_ax.plot(
        pet_arr, 'r-', lw=0.8, label='Potential Evapotranspiration')

    pet_ax.plot(
        evap_arr, 'b-', lw=0.5, label='Evapotranspiration', alpha=0.5)
    pet_ax.plot(pet_trend, 'r-.', lw=0.9, label='PET Trend')
    pet_ax.plot(et_trend, 'b-.', lw=0.9, label='ET Trend')
    pet_ax.text(
        pet_ax.get_xlim()[1] * 0.02,
        (pet_ax.get_ylim()[1] -
         (pet_ax.get_ylim()[1] - pet_ax.get_ylim()[0]) * 0.1),
        pet_stats_str,
        fontsize=font_size,
        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 2})

    snow_ax.plot(snow_arr, lw=0.5, label='Snow')

    liqu_ax.bar(
        bar_x,
        liqu_arr,
        width=1.0,
        edgecolor='none',
        label='Liquid Precipitation')

    sm_ax.plot(sm_arr, lw=0.5, label='Soil Moisture')
    sm_ax.axhline(fc, color='r', ls='-.', lw=1, label='fc')
    sm_ax.axhline(pwp, color='b', ls='-.', lw=1, label='pwp')

    tot_run_ax.bar(
        bar_x,
        tot_run_arr,
        width=1.0,
        label='Total Runoff',
        edgecolor='none')

    u_res_sto_ax.plot(
        ur_sto_arr, lw=0.5, label='Upper Reservoir - Storage')

    ur_run_uu_ax.bar(
        bar_x,
        ur_run_uu,
        label='Upper Reservoir - Quick Runoff',
        width=2.0,
        edgecolor='none')

    ur_run_ul_ax.bar(
        bar_x,
        ur_run_ul,
        label='Upper Reservoir - Slow Runoff',
        width=1.0,
        edgecolor='none')

    ur_to_lr_run_ax.bar(
        bar_x,
        ur_to_lr_run,
        label='Upper Reservoir  - Percolation',
        width=1.0,
        edgecolor='none')

    l_res_sto_ax.plot(
        lr_sto_arr, lw=0.5, label='Lower Reservoir - Storage')

    snow_ax.fill_between(bar_x, 0, snow_arr, alpha=0.3)
    sm_ax.fill_between(bar_x, 0, sm_arr, alpha=0.3)
    u_res_sto_ax.fill_between(bar_x, 0, ur_sto_arr, alpha=0.3)
    l_res_sto_ax.fill_between(bar_x, 0, lr_sto_arr, alpha=0.3)

    full_sim_text = full_sim_text.reshape(6, 6)

    table = params_ax.table(
        cellText=full_sim_text,
        loc='center',
        bbox=(0, 0, 1, 1),
        cellLoc='left')

    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    params_ax.set_axis_off()

    leg_font = f_props()
    leg_font.set_size(font_size - 1)

    plot_axes = [
        discharge_ax,
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
        balance_ax
        ]

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
