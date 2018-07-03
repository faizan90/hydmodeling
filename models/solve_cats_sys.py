# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Faizan Anwar, IWS Uni-Stuttgart)s
"""

import os
import timeit
import shelve
import shutil
from functools import partial

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
    in_use_step_ser,
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
    min_q_thresh,
    sep,
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

    assert isinstance(in_use_step_ser, pd.Series)
    assert np.issubdtype(in_use_step_ser.values.dtype, np.int32)

    assert isinstance(in_q_df, pd.DataFrame)

    for _df in in_ppt_dfs_dict.values():
        assert isinstance(_df, pd.DataFrame)
        assert np.issubdtype(_df.values.dtype, np.float64)

    for _df in in_tem_dfs_dict.values():
        assert isinstance(_df, pd.DataFrame)
        assert np.issubdtype(_df.values.dtype, np.float64)

    for _df in in_pet_dfs_dict.values():
        assert isinstance(_df, pd.DataFrame)
        assert np.issubdtype(_df.values.dtype, np.float64)

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

    assert isinstance(min_q_thresh, (float, int))
    assert min_q_thresh >= 0

    assert isinstance(sep, str)

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
        raise NotImplementedError(f'Invalid time-freq: {time_freq}')

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

    k_aux_cell_vars_dict = _get_k_aux_vars_dict(
        aux_cell_vars_dict,
        sel_cats,
        all_prms_flags,
        run_as_lump_flag)

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

    (fin_ppt_dfs_dict,
     fin_tem_dfs_dict,
     fin_pet_dfs_dict) = _get_var_dicts(
        in_ppt_dfs_dict,
        in_tem_dfs_dict,
        in_pet_dfs_dict,
        aux_cell_vars_dict['area_ratios'],
        sel_cats,
        beg_date,
        end_date,
        run_as_lump_flag)

    del in_ppt_dfs_dict, in_tem_dfs_dict, in_pet_dfs_dict

    assert all([fin_ppt_dfs_dict, fin_tem_dfs_dict, fin_pet_dfs_dict])
    for cat in sel_cats:
        assert (n_steps ==
                fin_ppt_dfs_dict[cat].shape[0] ==
                fin_tem_dfs_dict[cat].shape[0] ==
                fin_pet_dfs_dict[cat].shape[0])

        assert (fin_ppt_dfs_dict[cat].shape[1] ==
                fin_tem_dfs_dict[cat].shape[1] ==
                fin_pet_dfs_dict[cat].shape[1])

        if np.any(np.isnan(fin_ppt_dfs_dict[cat].values)):
            raise RuntimeError('NaNs in precipiation!')
        if np.any(np.isnan(fin_tem_dfs_dict[cat].values)):
            raise RuntimeError('NaNs in temperature!')
        if np.any(np.isnan(fin_pet_dfs_dict[cat].values)):
            raise RuntimeError('NaNs in PET!')

    ini_arrs_dict = {sel_cat:
                     np.zeros((fin_tem_dfs_dict[sel_cat].shape[1], 4),
                              dtype=np.float64,
                              order='c')
                     for sel_cat in sel_cats}

    dumm_dict = None
    kfold_prms_dict = {}

    k_in_use_step_ser = in_use_step_ser.loc[beg_date:end_date]
    assert k_in_use_step_ser.shape[0] == n_steps
    assert np.all((k_in_use_step_ser.values >= 0) &
                  (k_in_use_step_ser.values <= 1))
    assert np.any(k_in_use_step_ser.values > 0)

    if in_use_step_ser.values.sum() == in_use_step_ser.shape[0]:
        use_step_flag = False
    else:
        use_step_flag = True
    print('INFO: use_step_flag:', use_step_flag)

    kwargs = {'use_obs_flow_flag': use_obs_flow_flag,
              'run_as_lump_flag': run_as_lump_flag,
              'use_step_flag': use_step_flag}

    old_wd = os.getcwd()
    os.chdir(out_dir)

    out_db_dir = os.path.join(out_dir, '01_database')
    if os.path.exists(out_db_dir):
        shutil.rmtree(out_db_dir)
    os.mkdir(out_db_dir)

    out_hgs_dir = os.path.join(out_dir, '02_hydrographs')
    if os.path.exists(out_hgs_dir):
        shutil.rmtree(out_hgs_dir)
    os.mkdir(out_hgs_dir)

    dirs_dict = {}
    dirs_dict['main'] = out_dir
    dirs_dict['db'] = out_db_dir
    dirs_dict['hgs'] = out_hgs_dir

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

        # one time for calibration period only
        calib_run = True
        _ = _solve_k_cats_sys(
            k_in_use_step_ser.iloc[_beg_i:_end_i],
            in_q_df.iloc[_beg_i:_end_i],
            k_ppt_dfs_dict,
            k_tem_dfs_dict,
            k_pet_dfs_dict,
            k_aux_cell_vars_dict,
            in_cats_prcssed_df,
            in_stms_prcssed_df,
            in_dem_net_df,
            bounds_dict,
            all_prms_flags,
            route_type,
            obj_ftn_wts,
            warm_up_steps,
            n_cpus,
            dirs_dict,
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
        _solve_k_cats_sys(
            k_in_use_step_ser,
            in_q_df,
            fin_ppt_dfs_dict,
            fin_tem_dfs_dict,
            fin_pet_dfs_dict,
            k_aux_cell_vars_dict,
            in_cats_prcssed_df,
            in_stms_prcssed_df,
            in_dem_net_df,
            bounds_dict,
            all_prms_flags,
            route_type,
            obj_ftn_wts,
            warm_up_steps,
            n_cpus,
            dirs_dict,
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

    os.chdir(old_wd)
    return


def _solve_k_cats_sys(
    in_use_step_ser,
    in_q_df,
    in_ppt_dfs_dict,
    in_tem_dfs_dict,
    in_pet_dfs_dict,
    in_aux_vars_dict,
    in_cats_prcssed_df,
    in_stms_prcssed_df,
    in_dem_net_df,
    bounds_dict,
    all_prms_flags,
    route_type,
    obj_ftn_wts,
    warm_up_steps,
    n_cpus,
    dirs_dict,
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
                                order='f',
                                dtype=np.float64)
    stms_inflow_arr = np.zeros((in_q_df.shape[0], in_stms_prcssed_df.shape[0]),
                               order='f',
                               dtype=np.float64)
    stms_outflow_arr = stms_inflow_arr.copy(order='f')

    in_dem_net_arr = np.ascontiguousarray(in_dem_net_df.values,
                                          dtype=np.float64)
    if np.any(np.isnan(in_dem_net_arr)):
        raise RuntimeError('NaNs in_dem_net_arr')

    all_us_cat_stms = []
    prms_dict = {}

    if calib_run:
        calib_valid_suff = 'calib'
    else:
        calib_valid_suff = 'valid'

    use_obs_flow_flag = int(kwargs['use_obs_flow_flag'])
    run_as_lump_flag = int(kwargs['run_as_lump_flag'])
    use_step_flag = int(kwargs['use_step_flag'])

    assert in_use_step_ser.shape[0] == in_q_df.shape[0]

    if use_step_flag:
        use_step_arr = in_use_step_ser.values.astype(np.int32, 'c')
    else:
        use_step_arr = np.array([0], dtype=np.int32)

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

        assert curr_us_stm is not None, f'Could not find ds for cat: {cat}!'

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
                q_arr.shape[0]), (
                    tem_arr.shape[1],
                    ppt_arr.shape[1],
                    pet_arr.shape[1],
                    q_arr.shape[0])

        n_cells = ini_arr.shape[0]
        print('n_cells:', n_cells)

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
                _1 = lulc_arr.shape[1]
                lulc_drop_idxs = (lulc_arr.max(axis=0) == 0)
                if lulc_drop_idxs.sum():
                    lulc_arr = lulc_arr[:, ~lulc_drop_idxs].copy(order='c')
                    _2 = lulc_arr.shape[1]
                    print(f'Land use classes reduced from {_1} to {_2}')

                n_lulc = lulc_arr.shape[1]
                assert lulc_arr.shape[0] == n_cells
                print(f'n_lulc:  {n_lulc}')
                print('lulc class ratios:\n', lulc_arr.sum(axis=0) / n_cells)
                assert np.all((lulc_arr >= 0) & (lulc_arr <= 1))

                aux_var_infos.append([0, lulc_arr.shape[1]])
                aux_vars.append(lulc_arr.ravel())
            else:
                aux_var_infos.append([0, 0])

            if np.any(all_prms_flags[:, 2]):
                assert 'soil_ratios' in in_aux_vars_dict

                soil_arr = in_aux_vars_dict['soil_ratios'][cat]
                _1 = soil_arr.shape[1]
                soil_drop_idxs = (soil_arr.max(axis=0) == 0)
                if soil_drop_idxs.sum():
                    soil_arr = soil_arr[:, ~soil_drop_idxs].copy(order='c')
                    _2 = soil_arr.shape[1]
                    print(f'Soil classes reduced from {_1} to {_2}')

                n_soil = soil_arr.shape[1]
                assert soil_arr.shape[0] == n_cells
                print(f'n_soil: {n_soil}')
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
                assert aspect_arr.ndim == 1
                assert aspect_arr.shape[0] == n_cells

            if np.any(all_prms_flags[:, 4] | all_prms_flags[:, 5]):
                assert 'slope' in in_aux_vars_dict

                slope_arr = in_aux_vars_dict['slope'][cat]
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
                    use_prms_labs.extend([f'{all_prms_labs[i]}_lc_{j:02d}'
                                          for j in range(n_lulc)])
                    use_prms_idxs[i, 1, :] = _bef_len, len(use_prms_labs)
                    bounds_list.extend(
                        [bounds_dict[all_prms_labs[i] + '_bds']] * n_lulc)

                if all_prms_flags[i, 2]:
                    _bef_len = len(use_prms_labs)
                    use_prms_labs.extend([f'{all_prms_labs[i]}_sl_{j:02d}'
                                          for j in range(n_soil)])
                    use_prms_idxs[i, 2, :] = _bef_len, len(use_prms_labs)
                    bounds_list.extend(
                        [bounds_dict[all_prms_labs[i] + '_bds']] * n_soil)

                if all_prms_flags[i, 3]:
                    _bef_len = len(use_prms_labs)

                    use_prms_labs.extend(
                        [f'{all_prms_labs[i]}_at_{_at_sp_labs[j]}'
                         for j in range(3)])

                    use_prms_idxs[i, 3, :] = _bef_len, len(use_prms_labs)
                    bounds_list.extend(
                        [bounds_dict[all_prms_labs[i] + '_bds']] * 2)
                    bounds_list.append(bounds_dict['exp_bds'])

                if all_prms_flags[i, 4]:
                    _bef_len = len(use_prms_labs)

                    use_prms_labs.extend(
                        [f'{all_prms_labs[i]}_sp_{_at_sp_labs[j]}'
                         for j in range(3)])

                    use_prms_idxs[i, 4, :] = _bef_len, len(use_prms_labs)
                    bounds_list.extend(
                        [bounds_dict[all_prms_labs[i] + '_bds']] * 2)
                    bounds_list.append(bounds_dict['exp_bds'])

                if all_prms_flags[i, 5]:
                    _bef_len = len(use_prms_labs)

                    use_prms_labs.extend(
                        [f'{all_prms_labs[i]}_at_sp_{_at_sp_labs[j]}'
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
                            f'musk_{curr_us_stms_idxs[i]}_lag')
                        use_prms_labs.append(
                            f'musk_{curr_us_stms_idxs[i]}_wt')

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
                                n_hm_params,
                                use_step_flag,
                                use_step_arr])

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
                out_db_dict = hbv_opt_de(curr_cat_params)
            else:
                raise ValueError(
                    f'opt_schm ({opt_schm_vars_dict["opt_schm"]}) '
                    'can only be DE!')

            prms_dict[cat] = (out_db_dict['hbv_prms'],
                              out_db_dict['route_prms'])

            opt_end_time = timeit.default_timer()
            print('Opt time was: %0.3f seconds\n' %
                  (opt_end_time - opt_strt_time))

            mean_time = (out_db_dict['n_calls'].mean() *
                         n_cells /
                         (opt_end_time - opt_strt_time))
            print(f'{mean_time:0.4f} hbv loops per second!')

            # a test of fc and pwp ratio
            _prms = out_db_dict['hbv_prms']
            mean_ratio = _prms[0, pwp_i] / _prms[0, fc_i]

            print(f'Mean pwp/fc ratio: {mean_ratio:0.3f}')

        else:
            out_db_dict = hbv_mult_cat_loop_py(curr_cat_params)

        out_db_dict['ini_arr'] = ini_arr
        out_db_dict['calib_valid_suff'] = calib_valid_suff
        out_db_dict['opt_schm_vars_dict'] = opt_schm_vars_dict

        out_db_dict['tem_arr'] = tem_arr
        out_db_dict['ppt_arr'] = ppt_arr
        out_db_dict['pet_arr'] = pet_arr
        out_db_dict['qact_arr'] = q_arr

        out_db_dict['use_step_flag'] = use_step_flag
        out_db_dict['use_step_arr'] = use_step_arr

        if calib_run:
            out_db_dict['use_prms_labs'] = use_prms_labs
            out_db_dict['bds_arr'] = bounds_arr

            out_db_dict['use_prms_idxs'] = use_prms_idxs
            out_db_dict['all_prms_flags'] = all_prms_flags
            out_db_dict['prms_span_idxs'] = prms_span_idxs
            out_db_dict['aux_vars'] = aux_vars
            out_db_dict['aux_var_infos'] = aux_var_infos

        if n_stms:
            curr_us_stm_idx = stm_to_idx_dict[curr_us_stm]
            extra_inflow = stms_outflow_arr[:, curr_us_stm_idx]
            out_db_dict['extra_us_inflow'] = (
                extra_inflow).copy(order='C')

        if route_type == 0:
            pass
        elif route_type == 1:
            if curr_us_stm == -2:
                pass
            else:
                route_labs = [[f'lag_{i}', f'wt_{i}']
                              for i in curr_us_stms]
                out_db_dict['route_labs'] = route_labs
        else:
            raise NotImplementedError('Implement stuff for this routing '
                                      'type!')

        _out_db_file = os.path.join(dirs_dict['db'], f'cat_{cat}')
        with shelve.open(_out_db_file, 'c', writeback=True) as db:

            if 'cat' not in db:
                db['cat'] = cat

            if calib_run:
                if 'calib' not in db:
                    db['calib'] = {}

                db['calib'][f'kf_{kf_i:02d}'] = out_db_dict

            else:
                if 'valid' not in db:
                    db['valid'] = {}

                db['valid'][f'kf_{kf_i:02d}'] = out_db_dict

            if 'data' not in db:
                db['data'] = {}
                db['data']['kfolds'] = kfolds
                db['data']['conv_ratio'] = conv_ratio
                db['data']['all_prms_flags'] = all_prms_flags
                db['data']['use_obs_flow_flag'] = use_obs_flow_flag
                db['data']['area_arr'] = cat_area_ratios_arr
                db['data']['dirs_dict'] = dirs_dict
                db['data']['off_idx'] = warm_up_steps
                db['data']['run_as_lump_flag'] = run_as_lump_flag
                db['data']['route_type'] = route_type
                db['data']['all_prms_labs'] = all_prms_labs

                db['data']['shape'] = cat_shape
                db['data']['rows'] = cat_rows_idxs
                db['data']['cols'] = cat_cols_idxs

            if not calib_run:
                if 'vdata' not in db:
                    db['vdata'] = {}

            if calib_run:
                if 'cdata' not in db:
                    db['cdata'] = {}

                if np.any(all_prms_flags[:, 1]):
                    db['cdata']['lulc_arr'] = lulc_arr

                if np.any(all_prms_flags[:, 2]):
                    db['cdata']['soil_arr'] = soil_arr

                if np.any(all_prms_flags[:, 3]):
                    db['cdata']['aspect_scale_arr'] = aspect_scale_arr

                if np.any(all_prms_flags[:, 4]):
                    db['cdata']['slope_scale_arr'] = slope_scale_arr

                if np.any(all_prms_flags[:, 5]):
                    db['cdata']['aspect_slope_scale_arr'] = (
                        aspect_slope_scale_arr)

        in_cats_prcssed_df.loc[cat, 'prcssed'] = True
        in_cats_prcssed_df.loc[cat, 'optd'] = True

        if curr_us_stm != -2:
            for _ in curr_us_stms:
                in_stms_prcssed_df.loc[_, 'optd'] = True
                in_stms_prcssed_df.loc[_, 'prcssed'] = True

    if True:
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
            out_name = f'{calib_valid_suff}_kfold_{kf_i:02d}__stm_{stm}.png'
            out_path = os.path.join(dirs_dict['hgs'], out_name)
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
            out_name = f'{calib_valid_suff}_kfold_{kf_i:02d}__cat_{cat}.png'
            out_path = os.path.join(dirs_dict['hgs'], out_name)
            plt.savefig(out_path, dpi=200, bbox_inches='tight')
            plt.close()

    _iter_str = f'{calib_valid_suff}_kfold_{kf_i:02d}_'

    if True:
        hgs_dict = {'qact_df': in_q_df}

        #======================================================================
        # out_stms_inflow_df = pd.DataFrame(data=stms_inflow_arr,
        #                                   columns=in_stms_prcssed_df.index,
        #                                   index=in_q_df.index,
        #                                   dtype=float)
        # hgs_dict['out_stms_inflow_df'] = out_stms_inflow_df
        #
        # out_stm_inflow_path = (
        #     os.path.join(dirs_dict['hgs'], f'{_iter_str}_stms_inflow.csv'))
        # out_stms_inflow_df.to_csv(
        #     out_stm_inflow_path, sep=str(sep), float_format='%0.5f')
        #======================================================================

        #======================================================================
        # out_stms_outflow_df = pd.DataFrame(data=stms_outflow_arr,
        #                                    columns=in_stms_prcssed_df.index,
        #                                    index=in_q_df.index,
        #                                    dtype=float)
        # hgs_dict['out_stms_outflow_df'] = out_stms_outflow_df
        #
        # out_stm_outflow_path = (
        #     os.path.join(dirs_dict['hgs'], f'{_iter_str}_stms_outflow.csv'))
        # out_stms_outflow_df.to_csv(
        #     out_stm_outflow_path, sep=str(sep), float_format='%0.5f')
        #======================================================================

        out_cats_flow_df = pd.DataFrame(data=cats_outflow_arr,
                                        columns=in_cats_prcssed_df.index,
                                        index=in_q_df.index,
                                        dtype=float)
        hgs_dict['out_cats_flow_df'] = out_cats_flow_df

        out_cats_outflow_path = (
            os.path.join(dirs_dict['hgs'], f'{_iter_str}_cats_outflow.csv'))
        out_cats_flow_df.to_csv(
            out_cats_outflow_path, sep=str(sep), float_format='%0.5f')

        out_hgs_dfs_path = (
            os.path.join(dirs_dict['hgs'], 'hgs_dfs'))
        with shelve.open(out_hgs_dfs_path, 'c', writeback=True) as db:
            kf_str = f'kf_{kf_i:02d}'
            if calib_run:
                if 'calib' not in db:
                    db['calib'] = {}
                db_str = 'calib'
            else:
                if 'valid' not in db:
                    db['valid'] = {}
                db_str = 'valid'

            db[db_str][kf_str] = {}
            for key in hgs_dict:
                db[db_str][kf_str][key] = hgs_dict[key]
    return prms_dict


def _get_k_aux_dict(aux_dict, area_dict, cats, lf):
    out_dict = {cat: aux_dict[cat] for cat in cats}
    if lf:
        out_dict = {cat: np.array([((area_dict[cat] * out_dict[cat].T).T).sum(axis=0)])
                    for cat in cats}
    for cat in cats:
        assert not np.any(np.isnan(out_dict[cat]))
    return out_dict


def _get_k_aux_vars_dict(
        in_dict,
        cats,
        all_prms_flags,
        run_as_lump_flag):

    out_dict = {}

    area_dict = {cat: in_dict['area_ratios'][cat] for cat in cats}
    out_dict['area_ratios'] = area_dict

    if run_as_lump_flag:
        n_cats = len(cats)
        loc_rows = max(1, int(0.25 * n_cats))
        loc_cols = max(1, int(np.ceil(n_cats / loc_rows)))

        cats_shape = (loc_rows, loc_cols)

        cats_rows_idxs = {}
        cats_cols_idxs = {}
        k = 0
        for i in range(loc_rows):
            if k >= n_cats:
                break
            for j in range(loc_cols):
                cats_rows_idxs[cats[k]] = np.array([i])
                cats_cols_idxs[cats[k]] = np.array([j])
                k += 1

                if k >= n_cats:
                    break
    else:
        cats_shape = in_dict['shape']
        cats_rows_idxs = {cat: in_dict['rows'][cat] for cat in cats}
        cats_cols_idxs = {cat: in_dict['cols'][cat] for cat in cats}

    out_dict['shape'] = cats_shape
    out_dict['rows'] = cats_rows_idxs
    out_dict['cols'] = cats_cols_idxs

    _get_aux_dict_p = partial(
        _get_k_aux_dict,
        area_dict=area_dict,
        cats=cats,
        lf=run_as_lump_flag)

    if np.any(all_prms_flags[:, 1]):
        out_dict['lulc_ratios'] = _get_aux_dict_p(in_dict['lulc_ratios'])

    if np.any(all_prms_flags[:, 2]):
        out_dict['soil_ratios'] = _get_aux_dict_p(in_dict['soil_ratios'])

    if np.any(all_prms_flags[:, 3] | all_prms_flags[:, 5]):
        out_dict['aspect'] = _get_aux_dict_p(in_dict['aspect'])

    if np.any(all_prms_flags[:, 4] | all_prms_flags[:, 5]):
        out_dict['slope'] = _get_aux_dict_p(in_dict['slope'])

    if run_as_lump_flag:
        out_dict['area_ratios'] = {
            cat: np.array([area_dict[cat].sum()]) for cat in cats}

    return out_dict


def _get_var_dict(in_df_dict, cats, area_dict, db, de, lf):
    out_dict = {}
    for cat in cats:
        _df_1 = in_df_dict[cat].loc[db:de]
        if lf:
            _df_2 = pd.DataFrame(index=_df_1.index, dtype=float, columns=[0])
            _vals = _df_1.values * area_dict[cat]
            _df_2.values[:] = _vals.sum(axis=1).reshape(-1, 1)
            _df_1 = _df_2
        out_dict[cat] = _df_1
    return out_dict


def _get_var_dicts(in_ppts,
                   in_tems,
                   in_pets,
                   area_dict,
                   cats,
                   db,
                   de,
                   lf):

    _get_var_dict_p = partial(
        _get_var_dict,
        cats=cats,
        area_dict=area_dict,
        db=db,
        de=de,
        lf=lf)

    out_ppts = _get_var_dict_p(in_ppts)
    out_tems = _get_var_dict_p(in_tems)
    out_pets = _get_var_dict_p(in_pets)

    return (out_ppts, out_tems, out_pets)
