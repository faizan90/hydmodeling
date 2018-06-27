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

from .dtypes import get_fc_pwp_is
from .hbv_opt import hbv_opt_de
from .py_ftns import hbv_mult_cat_loop_py
from .misc_ftns import (
    get_aspect_scale_arr_cy,
    get_slope_scale_arr_cy,
    get_aspect_and_slope_scale_arr_cy)

plt.ioff()

fc_i, pwp_i = get_fc_pwp_is()


def solve_cats_sys(
    in_cats_prcssed_df,
    in_stms_prcssed_df,
    in_dem_net_df,
    in_q_df,
    in_ppt_dfs_dict,
    in_tem_dfs_dict,
    in_pet_dfs_dict,
    aux_cell_vars_dict,
    in_date_fmt,
    beg_date,
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
    use_obs_flow_flag,
    run_as_lump_flag,
    opt_schm_vars_dict):
    '''Optimize parameters for a given catchment

    Do a k folds calibration and validation as well
    '''

    assert isinstance(in_cats_prcssed_df, pd.DataFrame)
    assert isinstance(in_stms_prcssed_df, pd.DataFrame)
    assert isinstance(in_dem_net_df, pd.DataFrame)
    assert isinstance(in_q_df, pd.DataFrame)

    for _df in in_ppt_dfs_dict.values():
        assert isinstance(_df, pd.DataFrame)

    for _df in in_tem_dfs_dict.values():
        assert isinstance(_df, pd.DataFrame)

    for _df in in_pet_dfs_dict.values():
        assert isinstance(_df, pd.DataFrame)

    assert isinstance(aux_cell_vars_dict, dict)

    assert isinstance(in_date_fmt, str)
    assert isinstance(beg_date, str)
    assert isinstance(end_date, str)
    assert isinstance(time_freq, str)

    assert isinstance(warm_up_steps, int)
    assert warm_up_steps >= 0

    assert isinstance(n_cpus, int)
    assert n_cpus >= 1

    assert isinstance(route_type, int)
    assert route_type >= 0

    assert isinstance(out_dir, str)

    assert isinstance(bounds_dict, dict)
    assert len(bounds_dict) >= 11

    assert isinstance(all_prms_flags, np.ndarray)
    assert all_prms_flags.ndim == 2
    assert np.issubdtype(all_prms_flags.dtype, np.int32)

    assert isinstance(obj_ftn_wts, np.ndarray)
    assert obj_ftn_wts.ndim == 1
    assert np.issubdtype(obj_ftn_wts.dtype, np.float64)

    assert isinstance(water_bal_step_size, int)
    assert water_bal_step_size >= 0

    assert isinstance(min_q_thresh, (float, int))
    assert min_q_thresh >= 0

    assert isinstance(sep, str)

    assert isinstance(opt_pkl_path, str)

    assert isinstance(kfolds, int)
    assert kfolds >= 1, 'kfolds can only be 1 or greater!'

    assert isinstance(use_obs_flow_flag, bool)
    assert isinstance(run_as_lump_flag, bool)

    assert isinstance(opt_schm_vars_dict, dict)

    date_range = pd.date_range(beg_date, end_date, freq=time_freq)
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

    print('Optimizing...')

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    sel_cats = in_cats_prcssed_df.index.values.copy(order='C')
    assert np.all(np.isfinite(sel_cats))

    beg_date, end_date = pd.to_datetime([beg_date, end_date],
                                          format=in_date_fmt)

    n_steps = date_range.shape[0]

    if time_freq == 'D':
        pass
    else:
        raise NotImplementedError('Invalid time-freq: %s' % str(time_freq))

    assert beg_date >= in_q_df.index[0]
    assert all([beg_date >= _df.index[0] for _df in in_ppt_dfs_dict.values()])
    assert all([beg_date >= _df.index[0] for _df in in_tem_dfs_dict.values()])
    assert all([beg_date >= _df.index[0] for _df in in_pet_dfs_dict.values()])

    assert end_date <= in_q_df.index[-1]
    assert all([end_date <= _df.index[-1] for _df in in_ppt_dfs_dict.values()])
    assert all([end_date <= _df.index[-1] for _df in in_tem_dfs_dict.values()])
    assert all([end_date <= _df.index[-1] for _df in in_pet_dfs_dict.values()])

    assert (in_ppt_dfs_dict.keys() ==
            in_tem_dfs_dict.keys() ==
            in_pet_dfs_dict.keys())

    in_q_df.columns = pd.to_numeric(in_q_df.columns)
    in_q_df = in_q_df.loc[beg_date:end_date, sel_cats]
    if np.any(np.isnan(in_q_df.values)):
        raise RuntimeError('NaNs in in_q_df')
    in_q_df[in_q_df.values < min_q_thresh] = min_q_thresh
    assert in_q_df.shape[0] == n_steps

    fin_ppt_dfs_dict = {}
    fin_tem_dfs_dict = {}
    fin_pet_dfs_dict = {}

    for key in sel_cats:
        fin_ppt_dfs_dict[key] = in_ppt_dfs_dict[key].loc[beg_date:end_date]
        fin_tem_dfs_dict[key] = in_tem_dfs_dict[key].loc[beg_date:end_date]
        fin_pet_dfs_dict[key] = in_pet_dfs_dict[key].loc[beg_date:end_date]

        del in_ppt_dfs_dict[key]
        del in_tem_dfs_dict[key]
        del in_pet_dfs_dict[key]

        assert fin_ppt_dfs_dict[key].shape[0] == n_steps
        assert fin_tem_dfs_dict[key].shape[0] == n_steps
        assert fin_pet_dfs_dict[key].shape[0] == n_steps

        assert (fin_ppt_dfs_dict[key].shape[1] ==
                fin_tem_dfs_dict[key].shape[1] ==
                fin_pet_dfs_dict[key].shape[1])

        if np.any(np.isnan(fin_tem_dfs_dict[key].values)):
            raise RuntimeError('NaNs in in_tem_df')
        if np.any(np.isnan(fin_ppt_dfs_dict[key].values)):
            raise RuntimeError('NaNs in in_ppt_df')
        if np.any(np.isnan(fin_pet_dfs_dict[key].values)):
            raise RuntimeError('NaNs in in_pet_df')

    assert all([fin_ppt_dfs_dict, fin_tem_dfs_dict, fin_pet_dfs_dict])

    if not in_stms_prcssed_df.shape[0]:
        print('\n')
        print('INFO: A dummy stream (9999) inserted in in_stms_prcssed_df!')
        print('\n')

        assert 9999 not in in_dem_net_df.index
        in_stms_prcssed_df.loc[9999] = False, np.nan, False

    _keys = in_cats_prcssed_df.index.tolist()
    _vals = np.arange(in_cats_prcssed_df.shape[0]).tolist()
    cat_to_idx_dict = dict(zip(_keys, _vals))

    _keys = in_stms_prcssed_df.index.tolist()
    _vals = np.arange(in_stms_prcssed_df.shape[0]).tolist()
    stm_to_idx_dict = dict(zip(_keys, _vals))

    ini_arrs_dict = {sel_cat:
                     np.zeros((fin_tem_dfs_dict[sel_cat].shape[1], 4),
                              dtype=np.float64,
                              order='c')
                     for sel_cat in sel_cats}

    dumm_dict = None
    kfold_prms_dict = {}

    kwargs = {'use_obs_flow_flag': use_obs_flow_flag,
              'run_as_lump_flag':run_as_lump_flag}
    for kf_i in range(n_sel_idxs - 1):
        _beg_i = uni_sel_idxs_arr[kf_i]
        _end_i = uni_sel_idxs_arr[kf_i + 1]

        k_ppt_dfs_dict = {
            sel_cat: fin_ppt_dfs_dict[sel_cat].iloc[_beg_i:_end_i]
            for sel_cat in sel_cats}

        k_tem_dfs_dict = {
            sel_cat: fin_tem_dfs_dict[sel_cat].iloc[_beg_i:_end_i]
            for sel_cat in sel_cats}

        k_pet_dfs_dict = {
            sel_cat: fin_pet_dfs_dict[sel_cat].iloc[_beg_i:_end_i]
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

        # one time for calibration period only
        calib_run = True
        _ = _solve_k_cats_sys(
            in_q_df.iloc[_beg_i:_end_i].copy(),
            k_ppt_dfs_dict,
            k_tem_dfs_dict,
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
            kf_i + 1,
            ini_arrs_dict,
            opt_schm_vars_dict,
            cat_to_idx_dict,
            stm_to_idx_dict,
            dumm_dict,
            calib_run,
            kfolds,
            kwargs=kwargs)

        kfold_prms_dict[kf_i + 1] = _

        # for the calibrated params, run for all the time steps
        calib_run = False
        _ = _solve_k_cats_sys(
            in_q_df,
            fin_ppt_dfs_dict,
            fin_tem_dfs_dict,
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
            kf_i + 1,
            ini_arrs_dict,
            opt_schm_vars_dict,
            cat_to_idx_dict,
            stm_to_idx_dict,
            kfold_prms_dict[kf_i + 1],
            calib_run,
            kfolds,
            kwargs=kwargs)

    if opt_pkl_path:
        # for every kfold, save params of every catchment
        cats_kfold_prms_dict = {}
        k_folds_list = list(range(1, n_sel_idxs))

        for cat in in_cats_prcssed_df.index:
            prms_list = []

            for iter_no in k_folds_list:
                prms_list.append(kfold_prms_dict[iter_no][cat])

            cats_kfold_prms_dict[cat] = prms_list

        out_prms_dict_path = (opt_pkl_path.rsplit('.', -1)[0] +
                                '_k_fold_params.pkl')

        pkl_cur = open(out_prms_dict_path, 'wb')
        pickle.dump(cats_kfold_prms_dict, pkl_cur)
        pkl_cur.close()

    return


def _solve_k_cats_sys(
    in_q_df,
    in_ppt_dfs_dict,
    in_tem_dfs_dict,
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
    kf_i,
    ini_arrs_dict,
    opt_schm_vars_dict,
    cat_to_idx_dict,
    stm_to_idx_dict,
    sim_prms_dict,
    calib_run,
    kfolds,
    kwargs):

    cats_outflow_arr = np.zeros((in_q_df.shape[0],
                                 in_cats_prcssed_df.shape[0]),
                                order='C',
                                dtype=np.float64)
    stms_inflow_arr = np.zeros((in_q_df.shape[0], in_stms_prcssed_df.shape[0]),
                               order='C',
                               dtype=np.float64)
    stms_outflow_arr = stms_inflow_arr.copy(order='C')

    in_dem_net_arr = np.ascontiguousarray(in_dem_net_df.values,
                                          dtype=np.float64)
    if np.any(np.isnan(in_dem_net_arr)):
        raise RuntimeError('NaNs in_dem_net_arr')

    all_us_cat_stms = []

    if opt_pkl_path:
        opt_results_dict = {}

    prms_dict = {}

    if calib_run:
        prms_dict = {}
        calib_valid_suff = 'calib'
    else:
        calib_valid_suff = 'valid'

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

        if calib_run:
            print('n_stms:', n_stms, ', cat_no:', cat)

        q_arr = in_q_df[cat].values.copy(order='C')
        tem_arr = in_tem_dfs_dict[cat].values.T.copy(order='C')
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
            _area_rshp = cat_area_ratios_arr.reshape(-1, 1)
            tem_arr = (_area_rshp * tem_arr).sum(axis=0).reshape(1, -1)
            ppt_arr = (_area_rshp * ppt_arr).sum(axis=0).reshape(1, -1)
            pet_arr = (_area_rshp * pet_arr).sum(axis=0).reshape(1, -1)
            ini_arr = (_area_rshp * ini_arr).sum(axis=0).reshape(1, -1)
            cat_area_ratios_arr = np.array([_area_rshp.sum()])

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

            # for given flags, embed auxillary variables in a single vector
            # attach starting positions of each variable in the vector
            if np.any(all_prms_flags[:, 1]):
                assert 'lulc_ratios' in in_aux_vars_dict

                lulc_arr = in_aux_vars_dict['lulc_ratios'][cat]
                if run_as_lump_flag:
                    _area_rshp = (
                        in_aux_vars_dict['area_ratios'][cat].reshape(-1, 1))
                    lulc_arr = (
                        _area_rshp * lulc_arr).sum(axis=0).reshape(1, -1)

                _ = lulc_arr.shape[1]
                lulc_drop_idxs = (lulc_arr.max(axis=0) == 0)
                if lulc_drop_idxs.sum():
                    lulc_arr = lulc_arr[:, ~lulc_drop_idxs].copy(order='c')
                    print('Land use classes reduced from %d to %d' %
                          (_, lulc_arr.shape[1]))

                n_lulc = lulc_arr.shape[1]
                assert lulc_arr.shape[0] == n_cells
                print('n_lulc: %d' % n_lulc)
                print('lulc class ratios:\n', lulc_arr.sum(axis=0) / n_cells)
                assert np.all((lulc_arr >= 0) & (lulc_arr <= 1))

                aux_var_infos.append([0, lulc_arr.shape[1]])
                aux_vars.append(lulc_arr.ravel())
            else:
                aux_var_infos.append([0, 0])

            if np.any(all_prms_flags[:, 2]):
                assert 'soil_ratios' in in_aux_vars_dict

                soil_arr = in_aux_vars_dict['soil_ratios'][cat]
                if run_as_lump_flag:
                    _area_rshp = (
                        in_aux_vars_dict['area_ratios'][cat].reshape(-1, 1))
                    soil_arr = (
                        _area_rshp * soil_arr).sum(axis=0).reshape(1, -1)

                _ = soil_arr.shape[1]
                soil_drop_idxs = (soil_arr.max(axis=0) == 0)
                if soil_drop_idxs.sum():
                    soil_arr = soil_arr[:, ~soil_drop_idxs].copy(order='c')
                    print('Soil classes reduced from %d to %d' %
                          (_, soil_arr.shape[1]))

                n_soil = soil_arr.shape[1]
                assert soil_arr.shape[0] == n_cells
                print('n_soil: %d' % n_soil)
                assert np.all((soil_arr >= 0) & (soil_arr <= 1))

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
                    _area = in_aux_vars_dict['area_ratios'][cat]
                    aspect_arr = np.array([(_area * aspect_arr).sum(axis=0)])

                assert aspect_arr.ndim == 1
                assert aspect_arr.shape[0] == n_cells

            if np.any(all_prms_flags[:, 4] | all_prms_flags[:, 5]):
                assert 'slope' in in_aux_vars_dict

                slope_arr = in_aux_vars_dict['slope'][cat]
                if run_as_lump_flag:
                    _area = in_aux_vars_dict['area_ratios'][cat]
                    slope_arr = np.array([(_area * slope_arr).sum(axis=0)])

                assert slope_arr.ndim == 1
                assert slope_arr.shape[0] == n_cells

            if np.any(all_prms_flags[:, 3]):
                aspect_scale_arr = get_aspect_scale_arr_cy(aspect_arr)

                assert np.all((aspect_scale_arr >= 0) &
                              (aspect_scale_arr <= 1))

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

                assert np.all((slope_scale_arr >= 0) &
                              (slope_scale_arr <= 1))

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
                    aspect_arr, slope_arr)

                assert np.all((aspect_slope_scale_arr >= 0) &
                              (aspect_slope_scale_arr <= 1))

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

            # for every parameter to be calibrated, insert bounds.
            # also, indicate indices in the use_prms_idxs of parameters in the
            # optimized parameter vectors (they are one dimensional). This is
            # important is more than one variable is used to compute a given
            # hbv parameter.
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
                        [bounds_dict[all_prms_labs[i] + '_bds']] * n_lulc)

                if all_prms_flags[i, 2]:
                    _bef_len = len(use_prms_labs)
                    use_prms_labs.extend([
                        '%s_sl_%0.2d' % (all_prms_labs[i], j)
                        for j in range(n_soil)])
                    use_prms_idxs[i, 2, :] = _bef_len, len(use_prms_labs)
                    bounds_list.extend(
                        [bounds_dict[all_prms_labs[i] + '_bds']] * n_soil)

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
                pop_size = int(
                    bounds_arr.shape[0] ** opt_schm_vars_dict['pop_size_exp'])
                _opt_list.extend([opt_schm_vars_dict['mu_sc_fac_bds'],
                                  opt_schm_vars_dict['cr_cnst_bds'],
                                  pop_size])
            else:
                raise Exception

            _opt_list.extend([opt_schm_vars_dict['max_iters'],
                              opt_schm_vars_dict['max_cont_iters'],
                              opt_schm_vars_dict['obj_ftn_tol'],
                              opt_schm_vars_dict['prm_pcnt_tol']])
            curr_cat_params.append(_opt_list)

        else:
            sim_params_arr = sim_prms_dict[cat]
            curr_cat_params.append(sim_params_arr)

            curr_cat_params.append([])  # to keep indicies consistent
            curr_cat_params.append([])  # to keep indicies consistent
            n_hm_params = sim_params_arr[0].shape[1]

        curr_cat_params.append([ini_arr, tem_arr, ppt_arr, pet_arr, q_arr])

        if n_stms:
            stms_idxs = np.array(curr_us_stms_idxs, dtype=np.int32)
        else:
            stms_idxs = np.array([0], dtype=np.int32)

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

        if calib_run:
            print('\n')
            opt_strt_time = timeit.default_timer()

            if opt_schm_vars_dict['opt_schm'] == 'DE':
                out_prms_dict = hbv_opt_de(curr_cat_params)
            else:
                raise ValueError('opt_schm (%s) can only be DE!' %
                                 opt_schm_vars_dict['opt_schm'])

            prms_dict[cat] = (out_prms_dict['params'],
                              out_prms_dict['route_params'])

            opt_end_time = timeit.default_timer()
            print('Opt time was: %0.3f seconds\n' %
                  (opt_end_time - opt_strt_time))

            # a test of fc and pwp ratio
            _prms = out_prms_dict['params']
            mean_ratio = _prms[0, pwp_i] / _prms[0, fc_i]

            print(f'Mean pwp/fc ratio: {mean_ratio:0.3f}')

        else:
            out_prms_dict = hbv_mult_cat_loop_py(curr_cat_params)

        if opt_pkl_path:
            out_prms_dict['off_idx'] = warm_up_steps
            out_prms_dict['conv_ratio'] = conv_ratio
            out_prms_dict['ini_arr'] = ini_arr
            out_prms_dict['out_suff'] = str(cat)
            out_prms_dict['out_dir'] = out_dir
            out_prms_dict['out_pref'] = '%0.2d' % kf_i
            out_prms_dict['calib_valid_suff'] = calib_valid_suff
            out_prms_dict['opt_schm_vars_dict'] = opt_schm_vars_dict
            out_prms_dict['run_as_lump_flag'] = run_as_lump_flag

            if calib_run:
                out_prms_dict['use_prms_labs'] = use_prms_labs
                out_prms_dict['bds_arr'] = bounds_arr

                out_prms_dict['use_prms_idxs'] = use_prms_idxs
                out_prms_dict['all_prms_flags'] = all_prms_flags
                out_prms_dict['prms_span_idxs'] = prms_span_idxs
                out_prms_dict['aux_vars'] = aux_vars
                out_prms_dict['aux_var_infos'] = aux_var_infos

            out_prms_dict['kfolds'] = kfolds

            out_prms_dict['area_arr'] = cat_area_ratios_arr

            out_prms_dict['temp_arr'] = tem_arr
            out_prms_dict['prec_arr'] = ppt_arr
            out_prms_dict['pet_arr'] = pet_arr
            out_prms_dict['q_arr'] = q_arr

            if calib_run and np.any(all_prms_flags[:, 1]):
                out_prms_dict['lulc_arr'] = lulc_arr

            if calib_run and np.any(all_prms_flags[:, 2]):
                out_prms_dict['soil_arr'] = soil_arr

            if calib_run and np.any(all_prms_flags[:, 3]):
                out_prms_dict['aspect_scale_arr'] = aspect_scale_arr

            if calib_run and np.any(all_prms_flags[:, 4]):
                out_prms_dict['slope_scale_arr'] = slope_scale_arr

            if calib_run and np.any(all_prms_flags[:, 5]):
                out_prms_dict['aspect_slope_scale_arr'] = (
                    aspect_slope_scale_arr)

            out_prms_dict['all_prms_flags'] = all_prms_flags
            out_prms_dict['water_bal_step_size'] = water_bal_step_size
            out_prms_dict['use_obs_flow_flag'] = use_obs_flow_flag

            out_prms_dict['shape'] = cat_shape
            out_prms_dict['rows'] = cat_rows_idxs
            out_prms_dict['cols'] = cat_cols_idxs

            if n_stms:
                curr_us_stm_idx = stm_to_idx_dict[curr_us_stm]
                extra_inflow = stms_outflow_arr[:, curr_us_stm_idx]
                out_prms_dict['extra_us_inflow'] = (
                    extra_inflow).copy(order='C')

            if route_type == 0:
                pass
            elif route_type == 1:
                if curr_us_stm == -2:
                    pass
                else:
                    route_labs = [['lag_%d' % i, 'wt_%d' % i]
                                  for i in curr_us_stms]
                    out_prms_dict['route_labs'] = route_labs
            else:
                raise NotImplementedError('Implement stuff for this routing '
                                          'type!')

            opt_results_dict[cat] = out_prms_dict

        in_cats_prcssed_df.loc[cat, 'prcssed'] = True
        in_cats_prcssed_df.loc[cat, 'optd'] = True

        if curr_us_stm != -2:
            for _ in curr_us_stms:
                in_stms_prcssed_df.loc[_, 'optd'] = True
                in_stms_prcssed_df.loc[_, 'prcssed'] = True

    if False:
        # in case of debugging
        print('\nPlotting stream inflows\\outflows and catchment outflows...')
        for stm in in_dem_net_df.index:
            if not stm in in_stms_prcssed_df.index:
                continue
            if not in_stms_prcssed_df.loc[stm, 'optd']:
                continue

            plt.figure(figsize=(20, 7))
            plt.plot(stms_inflow_arr[:, stm_to_idx_dict[stm]],
                     label='inflow',
                     alpha=0.5)
            plt.plot(stms_outflow_arr[:, stm_to_idx_dict[stm]],
                     label='outflow',
                     alpha=0.5)
            plt.legend()
            plt.grid()
            out_name = '%s_kfold_%0.2d__stm_' % (calib_valid_suff, kf_i)
            out_name += str(int(stm)) + '.png'
            out_path = os.path.join(out_dir, out_name)
            plt.savefig(out_path, dpi=200, bbox_inches='tight')
            plt.close()

        for i, cat in enumerate(in_cats_prcssed_df.index):
            if not in_cats_prcssed_df.loc[cat, 'optd']:
                continue

            plt.figure(figsize=(20, 7))
            curr_us_stm = all_us_cat_stms[i]
            if curr_us_stm != -2:
                us_inflow = stms_outflow_arr[:, stm_to_idx_dict[curr_us_stm]]
            else:
                us_inflow = np.zeros(stms_outflow_arr.shape[0])
            plt.plot(us_inflow, label='us_inflow', alpha=0.5)
            plt.plot(cats_outflow_arr[:, cat_to_idx_dict[cat]],
                     label='cat_outflow',
                     alpha=0.5)
            plt.plot(in_q_df.loc[:, cat].values, label='q_act_arr', alpha=0.5)

            plt.legend()
            plt.grid()
            out_name = '%s_kfold_%0.2d__cat_' % (calib_valid_suff, kf_i)
            out_name += str(int(cat)) + '.png'
            out_path = os.path.join(out_dir, out_name)
            plt.savefig(out_path, dpi=200, bbox_inches='tight')
            plt.close()

    _iter_str = '%s_kfold_%0.2d_' % (calib_valid_suff, kf_i)

    #==========================================================================
    # out_stms_inflow_df = pd.DataFrame(data=stms_inflow_arr,
    #                                   columns=in_stms_prcssed_df.index,
    #                                   index=in_q_df.index,
    #                                   dtype=float)
    # opt_results_dict['out_stms_inflow_df'] = out_stms_inflow_df
    #
    # out_stm_inflow_path = (
    #     os.path.join(out_dir, '%s_stms_inflow.csv' % _iter_str))
    # out_stms_inflow_df.to_csv(
    #     out_stm_inflow_path, sep=str(sep), float_format='%0.5f')
    #==========================================================================

    #==========================================================================
    # out_stms_outflow_df = pd.DataFrame(data=stms_outflow_arr,
    #                                    columns=in_stms_prcssed_df.index,
    #                                    index=in_q_df.index,
    #                                    dtype=float)
    # opt_results_dict['out_stms_outflow_df'] = out_stms_outflow_df
    #
    # out_stm_outflow_path = (
    #     os.path.join(out_dir, '%s_stms_outflow.csv' % _iter_str))
    # out_stms_outflow_df.to_csv(
    #     out_stm_outflow_path, sep=str(sep), float_format='%0.5f')
    #==========================================================================

    out_cats_flow_df = pd.DataFrame(data=cats_outflow_arr,
                                    columns=in_cats_prcssed_df.index,
                                    index=in_q_df.index,
                                    dtype=float)
    opt_results_dict['out_cats_flow_df'] = out_cats_flow_df

    out_cats_outflow_path = (
        os.path.join(out_dir, '%s_cats_outflow.csv' % _iter_str))
    out_cats_flow_df.to_csv(
        out_cats_outflow_path, sep=str(sep), float_format='%0.5f')

    if opt_pkl_path:
        _ = opt_pkl_path.rsplit('.', 1)
        _[0] = _[0] + ('__%s_kfold_%0.2d' % (calib_valid_suff, kf_i))
        _ = _[0] + '.' + _[1]

        pkl_cur = open(_, 'wb')
        pickle.dump(opt_results_dict, pkl_cur)
        pkl_cur.close()

    return prms_dict
