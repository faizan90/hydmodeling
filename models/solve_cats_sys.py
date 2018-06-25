# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Faizan Anwar, IWS Uni-Stuttgart)s
"""

import os
import timeit
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathos.multiprocessing import ProcessPool

from .hbv_opt import hbv_opt_de
from .py_ftns import hbv_mult_cat_loop_py
from .misc_ftns import (
    get_aspect_scale_arr_cy,
    get_slope_scale_arr_cy,
    get_aspect_and_slope_scale_arr_cy)

plt.ioff()


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


def solve_cats_sys(
        in_cats_prcssed_df,
        in_stms_prcssed_df,
        in_dem_net_df,
        in_q_df,
        in_ppt_dfs_dict,
        in_temp_dfs_dict,
        in_pet_dfs_dict,
        aux_cell_vars_dict,
        in_date_fmt,
        start_date,
        end_date,
        time_freq,
        warm_up_steps,
        n_cpus,
        route_type,
        out_dir,
        bounds_dict,
        all_prms_flags,
        obj_ftn_wts,
        water_bal_step_size,
        min_q_thresh,
        sep,
        opt_pkl_path,
        kfolds,
        compare_ann_cyc_flag,
        use_obs_flow_flag,
        run_as_lump_flag,
        opt_schm_vars_dict):
    '''Optimize parameters for a given catchment

    Do a k folds calibration and validation as well
    '''

    assert kfolds >= 1, 'kfolds can only be 1 or greater!'

    date_range = pd.date_range(start_date, end_date, freq=time_freq)
    sel_idxs_arr = np.linspace(0,
                               date_range.shape[0],
                               kfolds + 1,
                               dtype=np.int64,
                               endpoint=True)

    n_sel_idxs = sel_idxs_arr.shape[0]
    uni_sel_idxs_arr = np.unique(sel_idxs_arr)

    assert n_sel_idxs >= 2, 'kfolds too high or data points too low!'
    assert sel_idxs_arr.shape[0] == uni_sel_idxs_arr.shape[0], (
        'kfolds too high or data points too low!')

    if compare_ann_cyc_flag:
        assert time_freq == 'D', (
            'Annual cycle only available with daily frequency!')

    print('Optimizing...')

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    sel_cats = in_cats_prcssed_df.index.values.copy(order='C')
    assert np.all(np.isfinite(sel_cats))

    start_date, end_date = pd.to_datetime([start_date, end_date],
                                          format=in_date_fmt)

    n_steps = date_range.shape[0]

    if time_freq == 'D':
        pass
    else:
        raise NotImplementedError('Invalid time-freq: %s' % str(time_freq))

    assert start_date >= in_q_df.index[0]
    assert all([start_date >= _df.index[0]
                for _df in in_ppt_dfs_dict.values()])
    assert all([start_date >= _df.index[0]
                for _df in in_temp_dfs_dict.values()])
    assert all([start_date >= _df.index[0]
                for _df in in_pet_dfs_dict.values()])

    assert end_date <= in_q_df.index[-1]
    assert all([end_date <= _df.index[-1]
                for _df in in_ppt_dfs_dict.values()])
    assert all([end_date <= _df.index[-1]
                for _df in in_temp_dfs_dict.values()])
    assert all([end_date <= _df.index[-1]
                for _df in in_pet_dfs_dict.values()])

    assert (in_ppt_dfs_dict.keys() ==
            in_temp_dfs_dict.keys() ==
            in_pet_dfs_dict.keys())

    in_q_df.columns = [int(i) for i in in_q_df.columns]
    in_q_df = in_q_df.loc[start_date:end_date, sel_cats]
    if np.any(np.isnan(in_q_df.values)):
        raise RuntimeError('NaNs in in_q_df')
    in_q_df[in_q_df.values < min_q_thresh] = min_q_thresh
    assert in_q_df.shape[0] == n_steps

    fin_ppt_dfs_dict = {}
    fin_temp_dfs_dict = {}
    fin_pet_dfs_dict = {}

    for key in sel_cats:
        fin_ppt_dfs_dict[key] = in_ppt_dfs_dict[key].loc[start_date:end_date]
        fin_temp_dfs_dict[key] = in_temp_dfs_dict[key].loc[start_date:end_date]
        fin_pet_dfs_dict[key] = in_pet_dfs_dict[key].loc[start_date:end_date]

        del in_ppt_dfs_dict[key]
        del in_temp_dfs_dict[key]
        del in_pet_dfs_dict[key]

        assert fin_ppt_dfs_dict[key].shape[0] == n_steps
        assert fin_temp_dfs_dict[key].shape[0] == n_steps
        assert fin_pet_dfs_dict[key].shape[0] == n_steps

        assert (fin_ppt_dfs_dict[key].shape[1] ==
                fin_temp_dfs_dict[key].shape[1] ==
                fin_pet_dfs_dict[key].shape[1])

        if np.any(np.isnan(fin_temp_dfs_dict[key].values)):
            raise RuntimeError('NaNs in in_temp_df')
        if np.any(np.isnan(fin_ppt_dfs_dict[key].values)):
            raise RuntimeError('NaNs in in_ppt_df')
        if np.any(np.isnan(fin_pet_dfs_dict[key].values)):
            raise RuntimeError('NaNs in in_pet_df')

    assert all([fin_ppt_dfs_dict, fin_temp_dfs_dict, fin_pet_dfs_dict])

    cat_to_idx_dict = dict(list(zip(in_cats_prcssed_df.index,
                                    list(range(in_cats_prcssed_df.shape[0])))))
    stm_to_idx_dict = dict(list(zip(in_stms_prcssed_df.index,
                                    list(range(in_stms_prcssed_df.shape[0])))))

    ini_arrs_dict = {sel_cat:
                     np.zeros((fin_temp_dfs_dict[sel_cat].shape[1], 4),
                              dtype=np.float64,
                              order='c')
                     for sel_cat in sel_cats}

    dumm_dict = None
    kfold_params_dict = {}

    if compare_ann_cyc_flag:
        full_ann_cyc_df = get_daily_annual_cycle(in_q_df, n_cpus)

    kwargs = {'use_obs_flow_flag': use_obs_flow_flag,
              'run_as_lump_flag':run_as_lump_flag}
    for kfold_iter_no in range(n_sel_idxs - 1):
        curr_strt_idx = uni_sel_idxs_arr[kfold_iter_no]
        curr_end_idx = uni_sel_idxs_arr[kfold_iter_no + 1]

        if compare_ann_cyc_flag:
            kwargs['in_q_cyc_df'] = (
                full_ann_cyc_df.iloc[curr_strt_idx:curr_end_idx])

        k_ppt_dfs_dict = {sel_cat: fin_ppt_dfs_dict[sel_cat].iloc[curr_strt_idx:curr_end_idx]
                          for sel_cat in sel_cats}

        k_temp_dfs_dict = {sel_cat: fin_temp_dfs_dict[sel_cat].iloc[curr_strt_idx:curr_end_idx]
                          for sel_cat in sel_cats}

        k_pet_dfs_dict = {sel_cat: fin_pet_dfs_dict[sel_cat].iloc[curr_strt_idx:curr_end_idx]
                          for sel_cat in sel_cats}

        k_aux_cell_vars_dict = {}
        k_aux_cell_vars_dict['area_ratios'] = {
            sel_cat: aux_cell_vars_dict['area_ratios'][sel_cat]
            for sel_cat in sel_cats}

        k_aux_cell_vars_dict['shape'] = aux_cell_vars_dict['shape']

        k_aux_cell_vars_dict['rows'] = {
            sel_cat: aux_cell_vars_dict['rows'][sel_cat]
            for sel_cat in sel_cats}

        k_aux_cell_vars_dict['cols'] = {
            sel_cat: aux_cell_vars_dict['cols'][sel_cat]
            for sel_cat in sel_cats}

        if 'aspect' in aux_cell_vars_dict:
            k_aux_cell_vars_dict['aspect'] = {
                sel_cat: aux_cell_vars_dict['aspect'][sel_cat]
                for sel_cat in sel_cats}

        if 'slope' in aux_cell_vars_dict:
            k_aux_cell_vars_dict['slope'] = {
               sel_cat: aux_cell_vars_dict['slope'][sel_cat]
               for sel_cat in sel_cats}

        if 'lulc_ratios' in aux_cell_vars_dict:
            k_aux_cell_vars_dict['lulc_ratios'] = {
                sel_cat: aux_cell_vars_dict['lulc_ratios'][sel_cat]
                for sel_cat in sel_cats}

        if 'soil_ratios' in aux_cell_vars_dict:
            k_aux_cell_vars_dict['soil_ratios'] = {
                sel_cat: aux_cell_vars_dict['soil_ratios'][sel_cat]
                for sel_cat in sel_cats}

        calib_run = True
        _ = _solve_k_cats_sys(
            in_q_df.iloc[curr_strt_idx:curr_end_idx].copy(),
            k_ppt_dfs_dict,
            k_temp_dfs_dict,
            k_pet_dfs_dict,
            k_aux_cell_vars_dict,
            in_cats_prcssed_df,
            in_stms_prcssed_df,
            in_dem_net_df,
            opt_pkl_path,
            bounds_dict,
            all_prms_flags,
            route_type,
            obj_ftn_wts,
            warm_up_steps,
            n_cpus,
            out_dir,
            water_bal_step_size,
            sep,
            kfold_iter_no + 1,
            ini_arrs_dict,
            opt_schm_vars_dict,
            cat_to_idx_dict,
            stm_to_idx_dict,
            dumm_dict,
            calib_run,
            kfolds,
            kwargs=kwargs)

        kfold_params_dict[kfold_iter_no + 1] = _

        if compare_ann_cyc_flag:
            kwargs['in_q_cyc_df'] = full_ann_cyc_df

        calib_run = False
        _ = _solve_k_cats_sys(
            in_q_df,
            fin_ppt_dfs_dict,
            fin_temp_dfs_dict,
            fin_pet_dfs_dict,
            k_aux_cell_vars_dict,
            in_cats_prcssed_df,
            in_stms_prcssed_df,
            in_dem_net_df,
            opt_pkl_path,
            bounds_dict,
            all_prms_flags,
            route_type,
            obj_ftn_wts,
            warm_up_steps,
            n_cpus,
            out_dir,
            water_bal_step_size,
            sep,
            kfold_iter_no + 1,
            ini_arrs_dict,
            opt_schm_vars_dict,
            cat_to_idx_dict,
            stm_to_idx_dict,
            kfold_params_dict[kfold_iter_no + 1],
            calib_run,
            kfolds,
            kwargs=kwargs)

    if opt_pkl_path:
        cats_kfold_prms_dict = {}
        k_folds_list = list(range(1, n_sel_idxs))

        for cat in in_cats_prcssed_df.index:
            prms_list = []

            for iter_no in k_folds_list:
                prms_list.append(kfold_params_dict[iter_no][cat])

            cats_kfold_prms_dict[cat] = prms_list

        out_params_dict_path = (opt_pkl_path.rsplit('.', -1)[0] +
                                '_k_fold_params.pkl')

        pkl_cur = open(out_params_dict_path, 'wb')
        pickle.dump(cats_kfold_prms_dict, pkl_cur)
        pkl_cur.close()

    return


def _solve_k_cats_sys(
        in_q_df,
        in_ppt_dfs_dict,
        in_temp_dfs_dict,
        in_pet_dfs_dict,
        in_aux_vars_dict,
        in_cats_prcssed_df,
        in_stms_prcssed_df,
        in_dem_net_df,
        opt_pkl_path,
        bounds_dict,
        all_prms_flags,
        route_type,
        obj_ftn_wts,
        warm_up_steps,
        n_cpus,
        out_dir,
        water_bal_step_size,
        sep,
        kfold_iter_no,
        ini_arrs_dict,
        opt_schm_vars_dict,
        cat_to_idx_dict,
        stm_to_idx_dict,
        sim_params_dict,
        calib_run,
        kfolds,
        kwargs):

    cats_outflow_arr = np.zeros((in_q_df.shape[0],
                                 in_cats_prcssed_df.shape[0]),
                                order='C',
                                dtype=np.float64)
    stms_inflow_arr = np.zeros((in_q_df.shape[0], in_dem_net_df.shape[0]),
                               order='C',
                               dtype=np.float64)
    stms_outflow_arr = stms_inflow_arr.copy(order='C')

    in_dem_net_arr = np.array(in_dem_net_df.values,
                              dtype=np.float64,
                              order='C')
    if np.any(np.isnan(in_dem_net_arr)):
        raise RuntimeError('NaNs in_dem_net_arr')

    all_us_cat_stms = []

    if opt_pkl_path:
        opt_results_dict = {}

    params_dict = {}

    if calib_run:
        params_dict = {}
        calib_valid_suff = 'calib'
    else:
        calib_valid_suff = 'valid'

    if 'in_q_cyc_df' in kwargs:
        in_q_cyc_df = kwargs['in_q_cyc_df']
    else:
        in_q_cyc_df = None

    use_obs_flow_flag = int(kwargs['use_obs_flow_flag'])
    run_as_lump_flag = int(kwargs['run_as_lump_flag'])

    for cat in in_cats_prcssed_df.index:
        curr_cat_params = []

        curr_us_stm = None
        if in_dem_net_df.shape[0] != 0:
            for _ in in_dem_net_df.index:
                us_stm_cond_1 = in_dem_net_df.loc[_, 'up_cat'] == cat
                us_stm_cond_5 = in_dem_net_df.loc[_, 'DSNODEID'] == cat
                if not (us_stm_cond_1 or us_stm_cond_5):
                    continue

                us_stm_cond_2 = in_dem_net_df.loc[_, 'out_stm'] == 1
                us_stm_cond_3 = in_dem_net_df.loc[_, 'up_strm_01'] == -2
                us_stm_cond_4 = in_dem_net_df.loc[_, 'up_strm_02'] == -2

                if us_stm_cond_5 and us_stm_cond_2:
                    curr_us_stm = _
                    break

                elif us_stm_cond_1 and us_stm_cond_3 and us_stm_cond_4:
                    curr_us_stm = -2
                    break
        else:
            curr_us_stm = -2

        assert curr_us_stm is not None, (
            'Could not find ds for cat %d!' % int(cat))

        all_us_cat_stms.append(curr_us_stm)

        curr_us_stms_idxs = []
        curr_us_stms = []
        if curr_us_stm != -2:
            for chk_opt_stm in in_stms_prcssed_df.loc[:curr_us_stm].index:
                if in_dem_net_df.loc[chk_opt_stm, 'DSNODEID'] != cat:
                    continue

                chk_opt_stm_idx = in_dem_net_df.index.get_loc(chk_opt_stm)
                curr_us_stms.append(chk_opt_stm)
                curr_us_stms_idxs.append(chk_opt_stm_idx)

        n_stms = len(curr_us_stms_idxs)
        print('n_stms:', n_stms, ', cat_no:', cat)

        q_arr = in_q_df[cat].values.copy(order='C')
        tem_arr = in_temp_dfs_dict[cat].values.T.copy(order='C')
        ppt_arr = in_ppt_dfs_dict[cat].values.T.copy(order='C')
        pet_arr = in_pet_dfs_dict[cat].values.T.copy(order='C')
        ini_arr = ini_arrs_dict[cat].copy(order='C')
        cat_area_ratios_arr = in_aux_vars_dict['area_ratios'][cat].copy(order='C')
        cat_shape = in_aux_vars_dict['shape']
        cat_rows_idxs = in_aux_vars_dict['rows'][cat]
        cat_cols_idxs = in_aux_vars_dict['cols'][cat]

        assert (tem_arr.shape[0] ==
                ppt_arr.shape[0] ==
                pet_arr.shape[0] ==
                ini_arr.shape[0] ==
                cat_area_ratios_arr.shape[0])

        assert (tem_arr.shape[1] ==
                ppt_arr.shape[1] ==
                pet_arr.shape[1] ==
                q_arr.shape[0])

        if run_as_lump_flag:
            tem_arr = (cat_area_ratios_arr.reshape(-1, 1) * tem_arr).sum(axis=0).reshape(1, -1)
            ppt_arr = (cat_area_ratios_arr.reshape(-1, 1) * ppt_arr).sum(axis=0).reshape(1, -1)
            pet_arr = (cat_area_ratios_arr.reshape(-1, 1) * pet_arr).sum(axis=0).reshape(1, -1)
            ini_arr = (cat_area_ratios_arr.reshape(-1, 1) * ini_arr).sum(axis=0).reshape(1, -1)
            cat_area_ratios_arr = np.array([in_aux_vars_dict['area_ratios'][cat].sum()])

            cat_shape = (1, 1)
            cat_rows_idxs = np.array([0])
            cat_cols_idxs = np.array([0])

        n_cells = ini_arr.shape[0]

        all_prms_labs = ['tt',
                         'cm',
                         'pcm',
                         'fc',
                         'beta',
                         'pwp',
                         'ur_thr',
                         'k_uu',
                         'k_ul',
                         'k_d',
                         'k_ll']

        _at_sp_labs = ['min', 'max', 'exp']

#         all_prms_flags = np.array([
#             [0, 0, 0, 1, 0, 0],  # tt
#             [0, 0, 0, 1, 0, 0],  # cm
#             [0, 0, 0, 1, 0, 0],  # pcm
#             [1, 0, 0, 0, 0, 0],  # fc
#             [1, 0, 0, 0, 0, 0],  # beta
#             [1, 0, 0, 0, 0, 0],  # pwp
#             [1, 0, 0, 0, 0, 0],  # ur_thr
#             [1, 0, 0, 0, 0, 0],  # k_uu
#             [1, 0, 0, 0, 0, 0],  # k_ul
#             [1, 0, 0, 0, 0, 0],  # k_d
#             [1, 0, 0, 0, 0, 0],  # k_ll
#             ], dtype=np.int32)

        assert np.all(all_prms_flags >= 0) & np.all(all_prms_flags <= 1)
        assert (np.all(all_prms_flags.sum(axis=1) > 0) &
                np.all(all_prms_flags.sum(axis=1) < 6))
        assert len(all_prms_labs) == all_prms_flags.shape[0]

        # fc and pwp calibrated using same type of criteria
        assert (np.abs(all_prms_flags[3] - all_prms_flags[5]).sum() == 0)

        # only one of the cols in aspect, slope or slope/aspect can be active
        assert (np.all(all_prms_flags[:, 3:].sum(axis=1) <= 1))

        if calib_run:
            # starting index and number of columns in aux_vars
            aux_var_infos = []

            # ravel of aux vars
            aux_vars = []

            if np.any(all_prms_flags[:, 1]):
                assert 'lulc_ratios' in in_aux_vars_dict

                lulc_arr = in_aux_vars_dict['lulc_ratios'][cat]
                if run_as_lump_flag:
                    lulc_arr = (
                        in_aux_vars_dict['area_ratios'][cat].reshape(-1, 1) * lulc_arr).sum(axis=0).reshape(1, -1)

                _ = lulc_arr.shape[1]
                lulc_drop_idxs = (lulc_arr.max(axis=0) == 0)
                if lulc_drop_idxs.sum():
                    lulc_arr = lulc_arr[:, ~lulc_drop_idxs].copy(order='c')
                    print('Land use classes reduced from %d to %d' %
                          (_, lulc_arr.shape[1]))
                n_lulc = lulc_arr.shape[1]
                assert lulc_arr.shape[0] == n_cells
                print('n_lulc: %d' % n_lulc)

                aux_var_infos.append([0, lulc_arr.shape[1]])
                aux_vars.append(lulc_arr.ravel())
            else:
                aux_var_infos.append([0, 0])

            if np.any(all_prms_flags[:, 2]):
                assert 'soil_ratios' in in_aux_vars_dict

                soil_arr = in_aux_vars_dict['soil_ratios'][cat]
                if run_as_lump_flag:
                    soil_arr = (
                        in_aux_vars_dict['area_ratios'][cat].reshape(-1, 1) * soil_arr).sum(axis=0).reshape(1, -1)

                _ = soil_arr.shape[1]
                soil_drop_idxs = (soil_arr.max(axis=0) == 0)
                if soil_drop_idxs.sum():
                    soil_arr = soil_arr[:, ~soil_drop_idxs].copy(order='c')
                    print('Soil classes reduced from %d to %d' %
                          (_, soil_arr.shape[1]))
                n_soil = soil_arr.shape[1]
                assert soil_arr.shape[0] == n_cells
                print('n_soil: %d' % n_soil)

                if aux_vars:
                    _beg_idx = sum([_.size for _ in aux_vars])
                else:
                    _beg_idx = 0

                aux_var_infos.append([_beg_idx, soil_arr.shape[1]])
                aux_vars.append(soil_arr.ravel())

            else:
                aux_var_infos.append([0, 0])

            if np.any(all_prms_flags[:, 3] | all_prms_flags[:, 5]):
                assert 'aspect' in in_aux_vars_dict

                aspect_arr = in_aux_vars_dict['aspect'][cat]

                if run_as_lump_flag:
                    aspect_arr = np.array([(in_aux_vars_dict['area_ratios'][cat] * aspect_arr).sum(axis=0)])

                assert aspect_arr.ndim == 1
                assert aspect_arr.shape[0] == n_cells

            if np.any(all_prms_flags[:, 4] | all_prms_flags[:, 5]):
                assert 'slope' in in_aux_vars_dict

                slope_arr = in_aux_vars_dict['slope'][cat]
                if run_as_lump_flag:
                    slope_arr = np.array([(in_aux_vars_dict['area_ratios'][cat] * slope_arr).sum(axis=0)])

                assert slope_arr.ndim == 1
                assert slope_arr.shape[0] == n_cells

            if np.any(all_prms_flags[:, 3]):
                aspect_scale_arr = get_aspect_scale_arr_cy(aspect_arr)

                if aux_vars:
                    _beg_idx = sum([_.size for _ in aux_vars])
                else:
                    _beg_idx = 0

                aux_var_infos.append([_beg_idx, 1])
                aux_vars.append(aspect_scale_arr)
            else:
                aux_var_infos.append([0, 0])

            if np.any(all_prms_flags[:, 4]):
                slope_scale_arr = get_slope_scale_arr_cy(slope_arr)

                if aux_vars:
                    _beg_idx = sum([_.size for _ in aux_vars])
                else:
                    _beg_idx = 0

                aux_var_infos.append([_beg_idx, 1])
                aux_vars.append(slope_scale_arr)
            else:
                aux_var_infos.append([0, 0])

            if np.any(all_prms_flags[:, 5]):
                aspect_slope_scale_arr = get_aspect_and_slope_scale_arr_cy(
                    aspect_arr,
                    slope_arr)

                if aux_vars:
                    _beg_idx = sum([_.size for _ in aux_vars])
                else:
                    _beg_idx = 0

                aux_var_infos.append([_beg_idx, 1])
                aux_vars.append(aspect_slope_scale_arr)
            else:
                aux_var_infos.append([0, 0])

            if len(aux_vars) < 1:
                aux_vars = np.array(aux_vars, dtype=np.float64)
            else:
                aux_vars = np.concatenate(aux_vars)
                aux_vars = aux_vars.astype(np.float64, order='c')

            aux_var_infos = np.atleast_2d(
                np.array(aux_var_infos, dtype=np.int32))
            assert aux_var_infos.shape[0] == 5
            assert aux_var_infos.shape[1] == 2

            use_prms_labs = []
            use_prms_idxs = np.zeros((all_prms_flags.shape[0],
                                      all_prms_flags.shape[1],
                                      2),
                                     dtype=np.int32)
            bounds_list = []
            prms_span_idxs = []
            for i in range(all_prms_flags.shape[0]):
                _beg_span_idx = len(use_prms_labs)
                if all_prms_flags[i, 0]:
                    assert np.all(all_prms_flags[i, 1:] == 0)
                    _bef_len = len(use_prms_labs)
                    use_prms_labs.append(all_prms_labs[i])
                    use_prms_idxs[i, 0, :] = _bef_len, _bef_len
                    bounds_list.append(bounds_dict[all_prms_labs[i] + '_bds'])

                if all_prms_flags[i, 1]:
                    _bef_len = len(use_prms_labs)
                    use_prms_labs.extend([
                        '%s_lc_%0.2d' % (all_prms_labs[i], j)
                        for j in range(n_lulc)])
                    use_prms_idxs[i, 1, :] = _bef_len, len(use_prms_labs)
                    bounds_list.extend(
                        [bounds_dict[all_prms_labs[i] + '_lc_bds']] * n_lulc)

                if all_prms_flags[i, 2]:
                    _bef_len = len(use_prms_labs)
                    use_prms_labs.extend([
                        '%s_sl_%0.2d' % (all_prms_labs[i], j)
                        for j in range(n_soil)])
                    use_prms_idxs[i, 2, :] = _bef_len, len(use_prms_labs)
                    bounds_list.extend(
                        [bounds_dict[all_prms_labs[i] + '_sl_bds']] * n_soil)

                if all_prms_flags[i, 3]:
                    _bef_len = len(use_prms_labs)

                    use_prms_labs.extend(
                        ['%s_at_%s' % (all_prms_labs[i], _at_sp_labs[j])
                         for j in range(3)])

                    use_prms_idxs[i, 3, :] = _bef_len, len(use_prms_labs)
                    bounds_list.extend(
                        [bounds_dict[all_prms_labs[i] + '_bds']] * 2)
                    bounds_list.append(bounds_dict['exp_bds'])

                if all_prms_flags[i, 4]:
                    _bef_len = len(use_prms_labs)

                    use_prms_labs.extend(
                        ['%s_sp_%s' % (all_prms_labs[i], _at_sp_labs[j])
                         for j in range(3)])

                    use_prms_idxs[i, 4, :] = _bef_len, len(use_prms_labs)
                    bounds_list.extend(
                        [bounds_dict[all_prms_labs[i] + '_bds']] * 2)
                    bounds_list.append(bounds_dict['exp_bds'])

                if all_prms_flags[i, 5]:
                    _bef_len = len(use_prms_labs)

                    use_prms_labs.extend(
                        ['%s_at_sp_%s' % (all_prms_labs[i], _at_sp_labs[j])
                         for j in range(3)])

                    use_prms_idxs[i, 5, :] = _bef_len, len(use_prms_labs)
                    bounds_list.extend(
                        [bounds_dict[all_prms_labs[i] + '_bds']] * 2)
                    bounds_list.append(bounds_dict['exp_bds'])

                _end_span_idx = len(use_prms_labs)
                prms_span_idxs.append([_beg_span_idx, _end_span_idx])

            # number of prms used to estimate hbv prms.
            # The rest are for routing (for now).
            n_hm_params = len(bounds_list)

            assert len(use_prms_labs) == n_hm_params

            prms_span_idxs = np.array(prms_span_idxs, dtype=np.int32)

            if n_stms > 0:
                if route_type == 0:
                    pass
                elif route_type == 1:
                    # these bounds might change for individual streams
                    lag_bds = [bounds_dict['musk_lag_bds']] * n_stms
                    wt_bds = [bounds_dict['musk_wt_bds']] * n_stms

                    for i in range(n_stms):
                        bounds_list.append(lag_bds[i])
                        bounds_list.append(wt_bds[i])

                        use_prms_labs.append(
                            'musk_%0.2d_lag' % curr_us_stms_idxs[i])
                        use_prms_labs.append(
                            'musk_%0.2d_wt' % curr_us_stms_idxs[i])

                else:
                    raise NotImplementedError('Incorrect route_type!')

            bounds_arr = np.array(bounds_list)
            assert bounds_arr.ndim == 2
            assert prms_span_idxs.ndim == 2

            curr_cat_params.append(bounds_arr)

            curr_cat_params.append(obj_ftn_wts)

            _opt_list = []
            if opt_schm_vars_dict['opt_schm'] == 'DE':
                _pop_size = int(
                    bounds_arr.shape[0] ** opt_schm_vars_dict['pop_size_exp'])
                _opt_list.extend([opt_schm_vars_dict['mu_sc_fac_bds'],
                                  opt_schm_vars_dict['cr_cnst_bds'],
                                  _pop_size])
            else:
                raise Exception

            _opt_list.extend([opt_schm_vars_dict['max_iters'],
                              opt_schm_vars_dict['max_cont_iters'],
                              opt_schm_vars_dict['obj_ftn_tol'],
                              opt_schm_vars_dict['prm_pcnt_tol']])
            curr_cat_params.append(_opt_list)

        else:
            sim_params_arr = sim_params_dict[cat]
            curr_cat_params.append(sim_params_arr)

            curr_cat_params.append([])  # to keep indicies consistent
            curr_cat_params.append([])  # to keep indicies consistent
            n_hm_params = sim_params_arr[0].shape[1]

        curr_cat_params.append([ini_arr,
                                tem_arr,
                                ppt_arr,
                                pet_arr,
                                q_arr])

        if n_stms:
            stms_idxs = np.array(curr_us_stms_idxs, dtype='int32')
        else:
            stms_idxs = np.array([0], dtype='int32')

        conv_ratio = in_cats_prcssed_df.loc[cat, 'area'] / (1000. * 86400)

        curr_cat_params.append([curr_us_stm,
                                stms_idxs,
                                cat_to_idx_dict,
                                stm_to_idx_dict])
        curr_cat_params.append([in_dem_net_arr,
                                cats_outflow_arr,
                                stms_inflow_arr,
                                stms_outflow_arr])
        curr_cat_params.append([warm_up_steps,
                                conv_ratio,
                                route_type,
                                int(cat),
                                n_cpus,
                                n_stms,
                                n_cells,
                                use_obs_flow_flag,
                                n_hm_params])

        assert cat_area_ratios_arr.shape[0] == n_cells

        if calib_run:
            curr_cat_params.append([cat_area_ratios_arr,
                                    use_prms_idxs,
                                    all_prms_flags,
                                    prms_span_idxs,
                                    aux_vars,
                                    aux_var_infos])

        else:
            curr_cat_params.append(cat_area_ratios_arr)

        print('\n')
        opt_strt_time = timeit.default_timer()

        if calib_run:
            if opt_schm_vars_dict['opt_schm'] == 'DE':
                out_params_dict = hbv_opt_de(curr_cat_params)
            else:
                raise ValueError('opt_schm (%s) can only be DE!' %
                                 opt_schm_vars_dict['opt_schm'])

            params_dict[cat] = (out_params_dict['params'],
                                out_params_dict['route_params'])
        else:
            print('valid run...')
            out_params_dict = hbv_mult_cat_loop_py(curr_cat_params)
            print('valid end...')

        opt_end_time = timeit.default_timer()
        print('Opt time was: %0.3f seconds' % (opt_end_time - opt_strt_time))
        print('\n')

        if opt_pkl_path:
            out_params_dict['n_recs'] = in_q_df.shape[0]
            out_params_dict['off_idx'] = warm_up_steps
            out_params_dict['nprocs'] = n_cpus
            out_params_dict['conv_ratio'] = conv_ratio
            out_params_dict['ini_arr'] = ini_arr
            out_params_dict['out_suff'] = str(cat)
            out_params_dict['out_dir'] = out_dir
            out_params_dict['out_pref'] = '%0.2d' % kfold_iter_no
            out_params_dict['calib_valid_suff'] = calib_valid_suff
            out_params_dict['opt_schm_vars_dict'] = opt_schm_vars_dict
            out_params_dict['run_as_lump_flag'] = run_as_lump_flag

            if calib_run:
                out_params_dict['use_prms_labs'] = use_prms_labs
                out_params_dict['bds_arr'] = bounds_arr

            out_params_dict['kfolds'] = kfolds

            out_params_dict['area_arr'] = cat_area_ratios_arr

            out_params_dict['temp_arr'] = tem_arr
            out_params_dict['prec_arr'] = ppt_arr
            out_params_dict['pet_arr'] = pet_arr
            out_params_dict['q_arr'] = q_arr

            if calib_run and np.any(all_prms_flags[:, 1]):
                out_params_dict['lulc_arr'] = lulc_arr

            if calib_run and np.any(all_prms_flags[:, 2]):
                out_params_dict['soil_arr'] = soil_arr

            if calib_run and np.any(all_prms_flags[:, 3]):
                out_params_dict['aspect_scale_arr'] = aspect_scale_arr

            if calib_run and np.any(all_prms_flags[:, 4]):
                out_params_dict['slope_scale_arr'] = slope_scale_arr

            if calib_run and np.any(all_prms_flags[:, 5]):
                out_params_dict['aspect_slope_scale_arr'] = (
                    aspect_slope_scale_arr)

            out_params_dict['all_prms_flags'] = all_prms_flags
            out_params_dict['water_bal_step_size'] = water_bal_step_size
            out_params_dict['use_obs_flow_flag'] = use_obs_flow_flag

            out_params_dict['shape'] = cat_shape
            out_params_dict['rows'] = cat_rows_idxs
            out_params_dict['cols'] = cat_cols_idxs

            if in_q_cyc_df is not None:
                out_params_dict['q_cyc_arr'] = in_q_cyc_df[cat].values.copy(order='C')

            if n_stms:
                curr_us_stm_idx = stm_to_idx_dict[curr_us_stm]
                extra_inflow = (
                    out_params_dict['stms_outflow_arr'][:, curr_us_stm_idx])
                out_params_dict['extra_us_inflow'] = (
                    extra_inflow).copy(order='C')
            if route_type == 0:
                pass
            elif route_type == 1:
                if curr_us_stm == -2:
                    pass
                else:
                    route_labs = [['lag_%d' % i, 'wt_%d' % i]
                                  for i in curr_us_stms]
                    out_params_dict['route_labs'] = route_labs
            else:
                raise NotImplementedError('Implement stuff for this routing '
                                          'type!')

            opt_results_dict[cat] = out_params_dict

        in_cats_prcssed_df.loc[cat, 'prcssed'] = True
        in_cats_prcssed_df.loc[cat, 'optd'] = True

        if curr_us_stm != -2:
            for _ in curr_us_stms:
                in_stms_prcssed_df.loc[_, 'optd'] = True
                in_stms_prcssed_df.loc[_, 'prcssed'] = True

    if opt_pkl_path:
        print('\nSaving optimization pickle...')
        _ = opt_pkl_path.rsplit('.', 1)
        _[0] = _[0] + ('__%s_kfold_%0.2d' % (calib_valid_suff, kfold_iter_no))
        _ = _[0] + '.' + _[1]

        pkl_cur = open(_, 'wb')
        pickle.dump(opt_results_dict, pkl_cur)
        pkl_cur.close()

#     print('\nPlotting stream inflows\\outflows and catchment outflows...')
#     for stm in in_dem_net_df.index:
#         if not stm in in_stms_prcssed_df.index:
#             continue
#         if not in_stms_prcssed_df.loc[stm, 'optd']:
#             continue
#
#         plt.figure(figsize=(20, 7))
#         plt.plot(stms_inflow_arr[:, stm_to_idx_dict[stm]],
#                  label='inflow',
#                  alpha=0.5)
#         plt.plot(stms_outflow_arr[:, stm_to_idx_dict[stm]],
#                  label='outflow',
#                  alpha=0.5)
#         plt.legend()
#         plt.grid()
#         out_name = '%s_kfold_%0.2d__stm_' % (calib_valid_suff, kfold_iter_no)
#         out_name += str(int(stm)) + '.png'
#         out_path = os.path.join(out_dir, out_name)
#         plt.savefig(out_path, dpi=200, bbox_inches='tight')
#         plt.close()
#
#     for i, cat in enumerate(in_cats_prcssed_df.index):
#         if not in_cats_prcssed_df.loc[cat, 'optd']:
#             continue
#
#         plt.figure(figsize=(20, 7))
#         curr_us_stm = all_us_cat_stms[i]
#         if curr_us_stm != -2:
#             us_inflow = stms_outflow_arr[:, stm_to_idx_dict[curr_us_stm]]
#         else:
#             us_inflow = np.zeros(stms_outflow_arr.shape[0])
#         plt.plot(us_inflow, label='us_inflow', alpha=0.5)
#         plt.plot(cats_outflow_arr[:, cat_to_idx_dict[cat]],
#                  label='cat_outflow',
#                  alpha=0.5)
#         plt.plot(in_q_df.loc[:, cat].values, label='q_act_arr', alpha=0.5)
#
#         plt.legend()
#         plt.grid()
#         out_name = '%s_kfold_%0.2d__cat_' % (calib_valid_suff, kfold_iter_no)
#         out_name += str(int(cat)) + '.png'
#         out_path = os.path.join(out_dir, out_name)
#         plt.savefig(out_path, dpi=200, bbox_inches='tight')
#         plt.close()

    _iter_str = '%s_kfold_%0.2d_' % (calib_valid_suff, kfold_iter_no)
    out_stm_inflow_path = os.path.join(out_dir,
                                       '%s_stms_inflow.csv' % _iter_str)
    out_stm_outflow_path = os.path.join(out_dir,
                                        '%s_stms_outflow.csv' % _iter_str)
    out_cats_outflow_path = os.path.join(out_dir,
                                         '%s_cats_outflow.csv' % _iter_str)

    out_cats_flow_df = pd.DataFrame(data=cats_outflow_arr,
                                    columns=in_cats_prcssed_df.index,
                                    index=in_q_df.index,
                                    dtype=float)

    np.savetxt(out_stm_inflow_path, stms_inflow_arr, delimiter=str(sep))
    np.savetxt(out_stm_outflow_path, stms_outflow_arr, delimiter=str(sep))
    out_cats_flow_df.to_csv(out_cats_outflow_path,
                            sep=str(sep),
                            float_format='%0.5f')

    return params_dict
