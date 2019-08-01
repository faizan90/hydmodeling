"""
Created on %(date)s

@author: %(Faizan Anwar, IWS Uni-Stuttgart)s
"""

import os
import timeit
import shelve
import shutil
from functools import partial

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .miscs.dtypes import get_fc_pwp_is
from .opts.hbv_opt import hbv_opt
from .hyds.py_ftns import hbv_mult_cat_loop_py
from .miscs.misc_ftns import (
    get_aspect_scale_arr_cy,
    get_slope_scale_arr_cy,
    get_aspect_and_slope_scale_arr_cy)

from ..misc import mkdir_hm

plt.ioff()

fc_i, pwp_i = get_fc_pwp_is()

all_prms_labs = [
    'tt',
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


def solve_cats_sys(
        in_cats_prcssed_df,
        in_stms_prcssed_df,
        in_dem_net_df,
        in_use_step_df,
        in_q_df,
        in_ppt_dfs_dict,
        in_tem_dfs_dict,
        in_pet_dfs_dict,
        aux_cell_vars_dict,
        in_date_fmt,
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
        opt_schm_vars_dict,
        cv_list,
        use_resampled_obj_ftns_flag,
        discharge_resampling_freq,
        fourtrans_maxi_freq):

    '''Optimize parameters for a given catchment

    Do a k folds calibration and validation as well
    '''

    assert isinstance(in_cats_prcssed_df, pd.DataFrame)
    assert isinstance(in_stms_prcssed_df, pd.DataFrame)
    assert isinstance(in_dem_net_df, pd.DataFrame)

    assert isinstance(in_use_step_df, pd.DataFrame)
    assert np.issubdtype(in_use_step_df.values.dtype, np.int32)

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

    _df = None

    assert isinstance(aux_cell_vars_dict, dict)

    assert isinstance(in_date_fmt, str)
    assert isinstance(time_freq, str)

    assert isinstance(warm_up_steps, int)
    assert warm_up_steps >= 1

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

#     assert not obj_ftn_wts[3], 'Not functional anymore!'

    assert isinstance(min_q_thresh, (float, int))
    assert min_q_thresh >= 0

    assert isinstance(sep, str)

    assert isinstance(use_obs_flow_flag, bool)
    assert isinstance(run_as_lump_flag, bool)

    assert isinstance(opt_schm_vars_dict, dict)

    assert isinstance(cv_list, list)
    assert (len(cv_list) == 2) or (len(cv_list) == 4)
    for tstamp in cv_list:
        assert isinstance(tstamp, pd.Timestamp)

    assert isinstance(use_resampled_obj_ftns_flag, bool)

    assert isinstance(discharge_resampling_freq, str)
    assert discharge_resampling_freq in ['Y', 'A', 'M', 'W']

    assert isinstance(fourtrans_maxi_freq, str)
    assert fourtrans_maxi_freq[-1] in ['Y', 'A', 'M', 'W']

    sel_cats = in_cats_prcssed_df.index.values.copy(order='C')
    assert np.all(np.isfinite(sel_cats))

    in_q_df.columns = pd.to_numeric(in_q_df.columns)

    assert (
        in_ppt_dfs_dict.keys() ==
        in_tem_dfs_dict.keys() ==
        in_pet_dfs_dict.keys())

    if len(cv_list) == 2:
        assert isinstance(kfolds, int)
        assert kfolds >= 1, 'kfolds can only be 1 or greater!'
        assert cv_list[0] < cv_list[1]

        date_range = pd.date_range(cv_list[0], cv_list[1], freq=time_freq)

        sel_idxs_arr = np.linspace(
            0,
            date_range.shape[0],
            kfolds + 1,
            dtype=np.int64,
            endpoint=True)

        cv_flag = False

        n_sel_idxs = sel_idxs_arr.shape[0]

        uni_sel_idxs_arr = np.unique(sel_idxs_arr)

        assert np.all(np.ediff1d(sel_idxs_arr) > warm_up_steps), (
            'Too many k_folds!')

        assert n_sel_idxs >= 2, 'kfolds too high or data points too low!'
        assert sel_idxs_arr.shape[0] == uni_sel_idxs_arr.shape[0], (
            'kfolds too high or data points too low!')

    elif len(cv_list) == 4:
        assert cv_list[0] < cv_list[1]
        assert cv_list[2] < cv_list[3]

        cdate_range = pd.date_range(cv_list[0], cv_list[1], freq=time_freq)
        vdate_range = pd.date_range(cv_list[2], cv_list[3], freq=time_freq)

        date_range = cdate_range.append(vdate_range)

        uni_sel_idxs_arr = np.asarray([
            0,
            cdate_range.shape[0],
            cdate_range.shape[0] + vdate_range.shape[0],
            ])

        n_sel_idxs = uni_sel_idxs_arr.shape[0]

        cv_flag = True

        assert cdate_range.shape[0] > warm_up_steps, (
            'Calibration steps too few!')

        assert vdate_range.shape[0] > warm_up_steps, (
            'Validation steps too few!')

    else:
        raise ValueError(f'cv_list has incorrect number of values!')

    print('Optimizing...')

    mkdir_hm(out_dir)

    n_steps = date_range.shape[0]

    if time_freq not in ['D', '24H', '12H', '8H', '6H', '3H', '2H', 'H']:
        raise NotImplementedError(f'Invalid time-freq: {time_freq}')

    assert date_range.intersection(in_q_df.index).shape[0] == n_steps

    in_q_df = in_q_df.loc[date_range, sel_cats]

    in_q_df[in_q_df.values < min_q_thresh] = min_q_thresh

    assert all(
        [date_range.intersection(_df.index).shape[0] == n_steps
        for _df in in_ppt_dfs_dict.values()])

    assert all(
        [date_range.intersection(_df.index).shape[0] == n_steps
        for _df in in_tem_dfs_dict.values()])

    assert all(
        [date_range.intersection(_df.index).shape[0] == n_steps
        for _df in in_pet_dfs_dict.values()])

    if np.any(np.isnan(in_q_df.values)):
        raise RuntimeError('NaNs in in_q_df')

    k_aux_cell_vars_dict = get_k_aux_vars_dict(
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
     fin_pet_dfs_dict) = get_var_dicts(
        in_ppt_dfs_dict,
        in_tem_dfs_dict,
        in_pet_dfs_dict,
        aux_cell_vars_dict['area_ratios'],
        sel_cats,
        date_range,
        run_as_lump_flag)

#     from pathlib import Path
#     _outdir = Path(r'P:\Synchronize\IWS\QGIS_Neckar\hydmod\input_hyd_data')
#
#     _ppt_df = fin_ppt_dfs_dict[sel_cats[0]]
#     _tem_df = fin_tem_dfs_dict[sel_cats[0]]
#     _pet_df = fin_pet_dfs_dict[sel_cats[0]]
#
#     _ppt_df.columns = [sel_cats[0]]
#     _tem_df.columns = [sel_cats[0]]
#     _pet_df.columns = [sel_cats[0]]
#
#     for cat in sel_cats[1:]:
#         _ppt_df[cat] = fin_ppt_dfs_dict[cat].values
#         _tem_df[cat] = fin_tem_dfs_dict[cat].values
#         _pet_df[cat] = fin_pet_dfs_dict[cat].values
#
#     _ppt_df.to_csv(_outdir / 'rockenau_ppt_1961_2015_lumped.csv', float_format='%0.3f', sep=';')
#     _tem_df.to_csv(_outdir / 'rockenau_tem_1961_2015_lumped.csv', float_format='%0.3f', sep=';')
#     _pet_df.to_csv(_outdir / 'rockenau_pet_1961_2015_lumped.csv', float_format='%0.3f', sep=';')
#
#     raise Exception

    del in_ppt_dfs_dict, in_tem_dfs_dict, in_pet_dfs_dict

    assert all([fin_ppt_dfs_dict, fin_tem_dfs_dict, fin_pet_dfs_dict])

    for cat in sel_cats:
        assert (
            n_steps ==
            fin_ppt_dfs_dict[cat].shape[0] ==
            fin_tem_dfs_dict[cat].shape[0] ==
            fin_pet_dfs_dict[cat].shape[0])

        assert (
            fin_ppt_dfs_dict[cat].shape[1] ==
            fin_tem_dfs_dict[cat].shape[1] ==
            fin_pet_dfs_dict[cat].shape[1])

        assert np.all(fin_ppt_dfs_dict[cat].values >= 0)
        assert np.all(fin_pet_dfs_dict[cat].values >= 0)

    ini_arrs_dict = {
        sel_cat:
        np.zeros(
            (fin_tem_dfs_dict[sel_cat].shape[1], 4),
            dtype=np.float64,
            order='c')

        for sel_cat in sel_cats}

    dumm_dict = None
    kfold_prms_dict = {}

    k_in_use_step_df = in_use_step_df.loc[date_range]

    assert np.all(
        (k_in_use_step_df.values >= 0) & (k_in_use_step_df.values <= 1))

    assert np.any(k_in_use_step_df.values > 0)

    kwargs = {
        'use_obs_flow_flag': use_obs_flow_flag,
        'run_as_lump_flag': run_as_lump_flag,
        'min_q_thresh': min_q_thresh,
        'time_freq': time_freq,
        'cv_flag': cv_flag,
        'resamp_obj_ftns_flag': use_resampled_obj_ftns_flag,
        'discharge_resampling_freq': discharge_resampling_freq,
        'fourtrans_maxi_freq': fourtrans_maxi_freq}

    old_wd = os.getcwd()
    os.chdir(out_dir)

    out_db_dir = os.path.join(out_dir, '01_database')
    if os.path.exists(out_db_dir):
        shutil.rmtree(out_db_dir)

    mkdir_hm(out_db_dir)

    out_hgs_dir = os.path.join(out_dir, '02_hydrographs')
    if os.path.exists(out_hgs_dir):
        shutil.rmtree(out_hgs_dir)

    mkdir_hm(out_hgs_dir)

    dirs_dict = {}
    dirs_dict['main'] = out_dir
    dirs_dict['db'] = out_db_dir
    dirs_dict['hgs'] = out_hgs_dir

    for kf_i in range(n_sel_idxs - 1):
        beg_i = uni_sel_idxs_arr[kf_i]
        end_i = uni_sel_idxs_arr[kf_i + 1]

        k_ppt_dfs_dict = {
            sel_cat: fin_ppt_dfs_dict[sel_cat].iloc[beg_i:end_i]
            for sel_cat in sel_cats}

        k_tem_dfs_dict = {
            sel_cat: fin_tem_dfs_dict[sel_cat].iloc[beg_i:end_i]
            for sel_cat in sel_cats}

        k_pet_dfs_dict = {
            sel_cat: fin_pet_dfs_dict[sel_cat].iloc[beg_i:end_i]
            for sel_cat in sel_cats}

        # one time for calibration period only
        calib_run = True

        prms_dict = solve_cat(
            k_in_use_step_df.iloc[beg_i:end_i],
            in_q_df.iloc[beg_i:end_i],
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

        kfold_prms_dict[kf_i + 1] = prms_dict

        calib_run = False

        if cv_flag:
            beg_i = uni_sel_idxs_arr[kf_i + 1]
            end_i = uni_sel_idxs_arr[kf_i + 2]

            k_ppt_dfs_dict = {
                sel_cat: fin_ppt_dfs_dict[sel_cat].iloc[beg_i:end_i]
                for sel_cat in sel_cats}

            k_tem_dfs_dict = {
                sel_cat: fin_tem_dfs_dict[sel_cat].iloc[beg_i:end_i]
                for sel_cat in sel_cats}

            k_pet_dfs_dict = {
                sel_cat: fin_pet_dfs_dict[sel_cat].iloc[beg_i:end_i]
                for sel_cat in sel_cats}

            # for the calibrated params, run for all validation time steps
            prms_dict = solve_cat(
                k_in_use_step_df.iloc[beg_i:end_i],
                in_q_df.iloc[beg_i:end_i],
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
                kfold_prms_dict[kf_i + 1],
                calib_run,
                kfolds,
                kwargs=kwargs)

            break

        else:
            # for the calibrated params, run for all the time steps
            solve_cat(
                k_in_use_step_df,
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


def solve_cat(
        in_use_step_df,
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

    cats_outflow_arr = np.zeros(
        (in_q_df.shape[0], in_cats_prcssed_df.shape[0]),
        order='f',
        dtype=np.float64)

    stms_inflow_arr = np.zeros(
        (in_q_df.shape[0], in_stms_prcssed_df.shape[0]),
        order='f',
        dtype=np.float64)

    stms_outflow_arr = stms_inflow_arr.copy(order='f')

    in_dem_net_arr = np.ascontiguousarray(
        in_dem_net_df.values, dtype=np.float64)

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
    min_q_thresh = float(kwargs['min_q_thresh'])
    time_freq = str(kwargs['time_freq'])
    cv_flag = int(kwargs['cv_flag'])
    resamp_obj_ftns_flag = kwargs['resamp_obj_ftns_flag']
    discharge_resampling_freq = kwargs['discharge_resampling_freq']
    ft_maxi_freq = kwargs['fourtrans_maxi_freq']

    ft_maxi_freq_idx = get_ft_maxi_freq_idx(ft_maxi_freq, in_q_df.shape[0])

    if resamp_obj_ftns_flag:
        if discharge_resampling_freq in ['Y', 'A']:
            tag_idxs = in_q_df.index.year

        elif discharge_resampling_freq == 'M':
            tag_idxs = in_q_df.index.month

        elif discharge_resampling_freq == 'W':
            tag_idxs = in_q_df.index.weekofyear

        else:
            raise ValueError(
                'Incorrect value of discharge_resampling_freq:',
                discharge_resampling_freq)

        obj_ftn_resamp_tags_arr = get_resample_tags_arr(
            tag_idxs, warm_up_steps)

    else:
        obj_ftn_resamp_tags_arr = np.array([0, 1], dtype=np.uint64)

    assert in_use_step_df.shape[0] == in_q_df.shape[0]

    print('\n\n')
    print('#' * 10)
    print('#' * 10)
    print(f'Run type: {calib_valid_suff}')
    print(f'n_cats: {in_cats_prcssed_df.shape[0]}')
    print(f'kfold no: {kf_i}')

    print(f'n_cpus: {n_cpus}')
    print(f'n_recs: {cats_outflow_arr.shape[0]}')
    print(f'min_q_thresh: {min_q_thresh}')

    for cat in in_cats_prcssed_df.index:
        print('\n')
        print('#' * 10)
        print(f'Going through cat: {cat}')
        curr_cat_params = []

        if in_use_step_df[cat].values.sum() != in_use_step_df.shape[0]:
            use_step_flag = True
            use_step_arr = in_use_step_df[cat].values.astype(np.int32, 'c')

        else:
            use_step_flag = False
            use_step_arr = np.array([0], dtype=np.int32)

        print('INFO: use_step_flag:', use_step_flag)

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
        print(f'n_stms: {n_stms}')

        q_arr = in_q_df[cat].values.copy(order='C')
        tem_arr = in_tem_dfs_dict[cat].values.T.copy(order='C')
        ppt_arr = in_ppt_dfs_dict[cat].values.T.copy(order='C')
        pet_arr = in_pet_dfs_dict[cat].values.T.copy(order='C')
        ini_arr = ini_arrs_dict[cat].copy(order='C')
        cat_shape = in_aux_vars_dict['shape']
        cat_rows_idxs = in_aux_vars_dict['rows'][cat]
        cat_cols_idxs = in_aux_vars_dict['cols'][cat]
        cat_area_ratios_arr = in_aux_vars_dict[
            'area_ratios'][cat].copy(order='C')

        assert (
            tem_arr.shape[0] ==
            ppt_arr.shape[0] ==
            pet_arr.shape[0] ==
            ini_arr.shape[0] ==
            cat_area_ratios_arr.shape[0])

        assert (
            tem_arr.shape[1] ==
            ppt_arr.shape[1] ==
            pet_arr.shape[1] ==
            q_arr.shape[0]), (
                tem_arr.shape[1],
                ppt_arr.shape[1],
                pet_arr.shape[1],
                q_arr.shape[0])

        n_cells = ini_arr.shape[0]
        print(f'n_cells: {n_cells}')

        assert np.all(all_prms_flags >= 0) & np.all(all_prms_flags <= 1)
        assert len(all_prms_labs) == all_prms_flags.shape[0]
        assert (
            np.all(all_prms_flags.sum(axis=1) > 0) &
            np.all(all_prms_flags.sum(axis=1) < 6))

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
                print('soil class ratios:\n', soil_arr.sum(axis=0) / n_cells)

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

                assert np.all(
                    (aspect_scale_arr >= 0) & (aspect_scale_arr <= 1))

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

                assert np.all(
                    (slope_scale_arr >= 0) & (slope_scale_arr <= 1))

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

                assert np.all(
                    (aspect_slope_scale_arr >= 0) &
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
                aux_vars = np.asarray(aux_vars, dtype=np.float64)

            else:
                aux_vars = np.concatenate(aux_vars)
                aux_vars = aux_vars.astype(np.float64, order='c')

            aux_var_infos = np.atleast_2d(
                np.asarray(aux_var_infos, dtype=np.int32))

            assert aux_var_infos.shape[0] == 5
            assert aux_var_infos.shape[1] == 2

            use_prms_labs = []
            use_prms_idxs = np.zeros(
                (all_prms_flags.shape[0],
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
                    use_prms_labs.extend(
                        [f'{all_prms_labs[i]}_lc_{j:02d}'
                         for j in range(n_lulc)])

                    use_prms_idxs[i, 1, :] = _bef_len, len(use_prms_labs)

                    bounds_list.extend(
                        [bounds_dict[all_prms_labs[i] + '_bds']] * n_lulc)

                if all_prms_flags[i, 2]:
                    _bef_len = len(use_prms_labs)
                    use_prms_labs.extend(
                        [f'{all_prms_labs[i]}_sl_{j:02d}' for j in range(n_soil)])

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

            prms_span_idxs = np.asarray(prms_span_idxs, dtype=np.int32)

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

            bounds_arr = np.asarray(bounds_list)
            assert bounds_arr.ndim == 2
            assert prms_span_idxs.ndim == 2

            curr_cat_params.append(bounds_arr)

#             assert not obj_ftn_wts[3], 'Not functional anymore!'

            curr_cat_params.append(obj_ftn_wts)

            n_prm_vecs = int(
                bounds_arr.shape[0] ** opt_schm_vars_dict['n_prm_vecs_exp'])

            assert n_prm_vecs >= 3

            _opt_list = []
            # TODO: Add minimum value constraints
            if opt_schm_vars_dict['opt_schm'] == 'DE':
                _opt_list.extend(
                    [opt_schm_vars_dict['mu_sc_fac_bds'],
                     opt_schm_vars_dict['cr_cnst_bds']])

            elif opt_schm_vars_dict['opt_schm'] == 'ROPE':
                acc_rate = opt_schm_vars_dict['acc_rate']

                n_temp_rope_prm_vecs = int(
                    bounds_arr.shape[0] **
                    opt_schm_vars_dict['n_rope_prm_vecs_exp'])

                n_acc_prm_vecs = int(n_prm_vecs * acc_rate)
                assert n_acc_prm_vecs >= 3

                n_uvecs = int(
                    bounds_arr.shape[0] ** opt_schm_vars_dict['n_uvecs_exp'])
                max_chull_tries = opt_schm_vars_dict['max_chull_tries']
                depth_ftn_type = opt_schm_vars_dict['depth_ftn_type']
                min_pts_in_chull = opt_schm_vars_dict['min_pts_in_chull']

                _opt_list.extend([
                    n_temp_rope_prm_vecs,
                    n_acc_prm_vecs,
                    n_uvecs,
                    max_chull_tries,
                    depth_ftn_type,
                    min_pts_in_chull])

                print(f'n_temp_rope_prm_vecs: {n_temp_rope_prm_vecs}')
                print(f'n_acc_prm_vecs: {n_acc_prm_vecs}')
                print(f'n_uvecs: {n_uvecs}')
                print(f'max_chull_tries: {max_chull_tries}')

            elif opt_schm_vars_dict['opt_schm'] == 'BRUTE':
                n_discretize = opt_schm_vars_dict['n_discretize']
                assert n_discretize >= 2

                _opt_list.extend([n_discretize, n_prm_vecs])

            else:
                raise Exception('Incorrect opt_schm!')

            print(f'n_prm_vecs: {n_prm_vecs}')
            if opt_schm_vars_dict['opt_schm'] != 'BRUTE':
                _opt_list.extend([
                    n_prm_vecs,
                    opt_schm_vars_dict['max_iters'],
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
            stms_idxs = np.asarray(curr_us_stms_idxs, dtype=np.int32)

        else:
            stms_idxs = np.asarray([0], dtype=np.int32)

        conv_ratio = in_cats_prcssed_df.loc[cat, 'area'] / 1000

        if time_freq == 'D':
            conv_ratio /= 86400

        elif time_freq == '24H':
            conv_ratio /= 86400

        elif time_freq == '12H':
            conv_ratio /= 86400 * 0.5

        elif time_freq == '8H':
            conv_ratio /= 28800

        elif time_freq == '6H':
            conv_ratio /= 86400 * 0.25

        elif time_freq == '3H':
            conv_ratio /= 10800

        elif time_freq == '2H':
            conv_ratio /= 7200

        elif time_freq == 'H':
            conv_ratio /= 3600

        else:
            raise ValueError(f'Incorrect time_freq: {time_freq}')

        curr_cat_params.append([
            curr_us_stm,
            stms_idxs,
            cat_to_idx_dict,
            stm_to_idx_dict])

        curr_cat_params.append([
            in_dem_net_arr,
            cats_outflow_arr,
            stms_inflow_arr,
            stms_outflow_arr])

        curr_cat_params.append([
            warm_up_steps,
            conv_ratio,
            route_type,
            int(cat),
            n_cpus,
            n_stms,
            n_cells,
            use_obs_flow_flag,
            n_hm_params,
            use_step_flag,
            use_step_arr,
            min_q_thresh])

        assert cat_area_ratios_arr.shape[0] == n_cells

        if calib_run:
            curr_cat_params.append([
                cat_area_ratios_arr,
                use_prms_idxs,
                all_prms_flags,
                prms_span_idxs,
                aux_vars,
                aux_var_infos,
                resamp_obj_ftns_flag,
                obj_ftn_resamp_tags_arr])

            if opt_schm_vars_dict['opt_schm'] == 'DE':
                curr_cat_params.append(1)

            elif opt_schm_vars_dict['opt_schm'] == 'ROPE':
                curr_cat_params.append(2)

            elif opt_schm_vars_dict['opt_schm'] == 'BRUTE':
                curr_cat_params.append(3)

            else:
                raise ValueError(opt_schm_vars_dict['opt_schm'])

        else:
            curr_cat_params.append(cat_area_ratios_arr)

        curr_cat_params.append(ft_maxi_freq_idx)

        if calib_run:
            opt_strt_time = timeit.default_timer()

            if ((opt_schm_vars_dict['opt_schm'] == 'DE') or
                (opt_schm_vars_dict['opt_schm'] == 'ROPE') or
                (opt_schm_vars_dict['opt_schm'] == 'BRUTE')):

                out_db_dict = hbv_opt(curr_cat_params)

            else:
                raise ValueError(
                    f'Incorrect opt_schm ({opt_schm_vars_dict["opt_schm"]})')

            prms_dict[cat] = (
                out_db_dict['hbv_prms'], out_db_dict['route_prms'])

            opt_end_time = timeit.default_timer()
            mean_time = (
                out_db_dict['n_calls'].mean() *
                n_cells /
                (opt_end_time - opt_strt_time))

            print(f'{mean_time:0.4f} hbv loops per second per cell per '
                  'thread!')

            print('Opt time was: %0.3f seconds' %
                  (opt_end_time - opt_strt_time))

            # a test of fc and pwp ratio
            _prms = out_db_dict['hbv_prms']
            mean_ratio = _prms[0, pwp_i] / _prms[0, fc_i]

            print(f'Mean pwp/fc ratio: {mean_ratio:0.3f}')

        else:
            out_db_dict = hbv_mult_cat_loop_py(curr_cat_params)

        if n_stms:
            curr_us_stm_idx = stm_to_idx_dict[curr_us_stm]
            extra_inflow = stms_outflow_arr[:, curr_us_stm_idx]
            out_db_dict['extra_us_inflow'] = (
                extra_inflow).copy(order='C')

        out_db_dict['calib_valid_suff'] = calib_valid_suff
        out_db_dict['use_step_flag'] = use_step_flag
        out_db_dict['use_step_arr'] = use_step_arr

        _out_db_file = os.path.join(dirs_dict['db'], f'cat_{cat}.hdf5')
        with h5py.File(_out_db_file, 'a', driver='core') as db:

            if 'cat' not in db:
                db.attrs['cat'] = cat

            if calib_run or cv_flag:
                out_db_dict['tem_arr'] = tem_arr
                out_db_dict['ppt_arr'] = ppt_arr
                out_db_dict['pet_arr'] = pet_arr
                out_db_dict['qact_arr'] = q_arr
                out_db_dict['ini_arr'] = ini_arr

            if calib_run:
                for key in out_db_dict:
                    db[f'calib/kf_{kf_i:02d}/{key}'] = out_db_dict[key]

            else:
                for key in out_db_dict:
                    if (n_cells > 1) and (key == 'outs_arr'):
                        continue

                    db[f'valid/kf_{kf_i:02d}/{key}'] = out_db_dict[key]

            if 'data' not in db:
                data_sb = db.create_group('data')
                data_sb.attrs['kfolds'] = kfolds
                data_sb.attrs['conv_ratio'] = conv_ratio
                data_sb['all_prms_flags'] = all_prms_flags
                data_sb.attrs['use_obs_flow_flag'] = use_obs_flow_flag
                data_sb['area_arr'] = cat_area_ratios_arr
                data_sb.attrs['off_idx'] = warm_up_steps
                data_sb.attrs['run_as_lump_flag'] = run_as_lump_flag
                data_sb.attrs['route_type'] = route_type
                data_sb.attrs['cv_flag'] = cv_flag

                data_sb['obj_ftn_wts'] = obj_ftn_wts

                dt = h5py.special_dtype(vlen=str)
                _prms_ds = data_sb.create_dataset(
                    'all_prms_labs', (len(all_prms_labs),), dtype=dt)
                _prms_ds[:] = all_prms_labs

                data_sb['shape'] = cat_shape
                data_sb['rows'] = cat_rows_idxs
                data_sb['cols'] = cat_cols_idxs

                bds_sb = data_sb.create_group('bds_dict')
                for key in bounds_dict:
                    bds_sb[key] = bounds_dict[key]

                for key in dirs_dict:
                    data_sb.attrs[key] = dirs_dict[key]

            if not calib_run:
                pass

            if calib_run and ('cdata' not in db):
                cdata_sb = db.create_group('cdata')
                if np.any(all_prms_flags[:, 1]):
                    db['cdata/lulc_arr'] = lulc_arr

                if np.any(all_prms_flags[:, 2]):
                    db['cdata/soil_arr'] = soil_arr

                if np.any(all_prms_flags[:, 3]):
                    db['cdata/aspect_scale_arr'] = aspect_scale_arr

                if np.any(all_prms_flags[:, 4]):
                    db['cdata/slope_scale_arr'] = slope_scale_arr

                if np.any(all_prms_flags[:, 5]):
                    db['cdata/aspect_slope_scale_arr'] = (
                        aspect_slope_scale_arr)

                opt_sb = cdata_sb.create_group('opt_schm_vars_dict')
                for key in opt_schm_vars_dict:
                    opt_sb.attrs[key] = opt_schm_vars_dict[key]

                dt = h5py.special_dtype(vlen=str)
                _prms_ds = cdata_sb.create_dataset(
                    'use_prms_labs', (len(use_prms_labs),), dtype=dt)
                _prms_ds[:] = use_prms_labs

                db['cdata/bds_arr'] = bounds_arr

                db['cdata/use_prms_idxs'] = use_prms_idxs
                db['cdata/all_prms_flags'] = all_prms_flags
                db['cdata/prms_span_idxs'] = prms_span_idxs
                db['cdata/aux_vars'] = aux_vars
                db['cdata/aux_var_infos'] = aux_var_infos

        in_cats_prcssed_df.loc[cat, 'prcssed'] = True
        in_cats_prcssed_df.loc[cat, 'optd'] = True

        if curr_us_stm != -2:
            for _ in curr_us_stms:
                in_stms_prcssed_df.loc[_, 'optd'] = True
                in_stms_prcssed_df.loc[_, 'prcssed'] = True

    if False:
        # in case of debugging
        print('Plotting stream inflows\\outflows and catchment outflows...')
        for stm in in_dem_net_df.index:
            if not stm in in_stms_prcssed_df.index:
                continue
            if not in_stms_prcssed_df.loc[stm, 'optd']:
                continue

            plt.figure(figsize=(20, 7))
            plt.plot(
                stms_inflow_arr[:, stm_to_idx_dict[stm]],
                label='inflow',
                alpha=0.5)

            plt.plot(
                stms_outflow_arr[:, stm_to_idx_dict[stm]],
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

            plt.plot(
                cats_outflow_arr[:, cat_to_idx_dict[cat]],
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

        out_cats_flow_df = pd.DataFrame(
            data=cats_outflow_arr,
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

    print('#' * 10)
    print('#' * 10)
    return prms_dict


def get_k_aux_dict(aux_dict, area_dict, cats, lf):

    out_dict = {cat: aux_dict[cat] for cat in cats}

    if lf:
        out_dict = {
            cat: np.asarray([
                ((area_dict[cat] * out_dict[cat].T).T).sum(axis=0)])

            for cat in cats}

    for cat in cats:
        assert not np.any(np.isnan(out_dict[cat]))
        assert out_dict[cat].ndim == 2

    return out_dict


def get_k_aux_vars_dict(
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

    get_aux_dict_p = partial(
        get_k_aux_dict,
        area_dict=area_dict,
        cats=cats,
        lf=run_as_lump_flag)

    if np.any(all_prms_flags[:, 1]):
        out_dict['lulc_ratios'] = get_aux_dict_p(in_dict['lulc_ratios'])

    if np.any(all_prms_flags[:, 2]):
        out_dict['soil_ratios'] = get_aux_dict_p(in_dict['soil_ratios'])

    if np.any(all_prms_flags[:, 3] | all_prms_flags[:, 5]):
        out_dict['aspect'] = get_aux_dict_p(in_dict['aspect'])

    if np.any(all_prms_flags[:, 4] | all_prms_flags[:, 5]):
        out_dict['slope'] = get_aux_dict_p(in_dict['slope'])

    if run_as_lump_flag:
        out_dict['area_ratios'] = {
            cat: np.asarray([area_dict[cat].sum()]) for cat in cats}

    return out_dict


def get_var_dict(in_df_dict, cats, area_dict, date_range, lf):

    out_dict = {}

    for cat in cats:
        df_1 = in_df_dict[cat].loc[date_range]

        if lf:
            df_2 = pd.DataFrame(index=df_1.index, dtype=float, columns=[0])

            vals = df_1.values * area_dict[cat]

            df_2.values[:] = vals.sum(axis=1).reshape(-1, 1)

            df_1 = df_2

        out_dict[cat] = df_1

    for cat in cats:
        assert not np.any(np.isnan(out_dict[cat]))
        assert out_dict[cat].ndim == 2

    return out_dict


def get_var_dicts(
        in_ppts,
        in_tems,
        in_pets,
        area_dict,
        cats,
        date_range,
        lf):

    get_var_dict_p = partial(
        get_var_dict,
        cats=cats,
        area_dict=area_dict,
        date_range=date_range,
        lf=lf)

    out_ppts = get_var_dict_p(in_ppts)
    out_tems = get_var_dict_p(in_tems)
    out_pets = get_var_dict_p(in_pets)

    return (out_ppts, out_tems, out_pets)


def get_resample_tags_arr(in_resamp_idxs, warm_up_steps):

    tags = [warm_up_steps]
    n_vals = in_resamp_idxs.shape[0]

    for i in range(warm_up_steps + 1, n_vals):
        if in_resamp_idxs[i] - in_resamp_idxs[i - 1]:
            tags.append(i)

    tags.append(n_vals)
    tags = np.asarray(tags, dtype=np.uint64)

    assert np.all(tags[1:] - tags[:-1] > 0)

    return tags


def get_ft_maxi_freq_idx(ft_maxi_freq, n_recs):

    if n_recs % 2:
        n_recs -= 1  # same is used inside the opt

    ft_maxi_freq_scale = ft_maxi_freq[-1]

    ft_freq_mult = ft_maxi_freq[:-1]
    if not ft_freq_mult:
        ft_freq_mult = 1

    else:
        ft_freq_mult = float(ft_freq_mult)

    assert ft_freq_mult > 0, 'Should be greater than zero!'

    print('ft_maxi_freq_scale:', ft_maxi_freq_scale)
    print('ft_freq_mult:', ft_freq_mult)

    if (ft_maxi_freq_scale == 'A') or (ft_maxi_freq_scale == 'Y'):
        # assuming 365 days a year
        assert n_recs > (365 * ft_freq_mult)

        ft_maxi_freq_idx = n_recs // (365 * ft_freq_mult)  # leap years?

    elif ft_maxi_freq_scale == 'M':
        # assuming a 30.5 day month
        assert n_recs > (30.5 * ft_freq_mult)

        ft_maxi_freq_idx = n_recs // (30.5 * ft_freq_mult)

    elif ft_maxi_freq_scale == 'W':
        assert n_recs > (7 * ft_freq_mult)

        ft_maxi_freq_idx = n_recs // (7 * ft_freq_mult)

    else:
        raise ValueError(f'ft_maxi_freq not defined for: {ft_maxi_freq}!')

    ft_maxi_freq_idx = int(ft_maxi_freq_idx)

    print('ft_maxi_freq_idx:', ft_maxi_freq_idx)
    assert ft_maxi_freq_idx > 1, 'No. of values is too little!'

    # subtracted 1 for the zero wave
    # in opt the count starts from the second wave
    return ft_maxi_freq_idx - 1

