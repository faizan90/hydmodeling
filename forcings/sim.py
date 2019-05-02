'''
Created on May 2, 2019

@author: Faizan-Uni
'''

"""
Created on %(date)s

@author: %(Faizan Anwar, IWS Uni-Stuttgart)s
"""

import os
import shelve
import shutil
from functools import partial

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..models.miscs.dtypes import get_fc_pwp_is
from ..models.hyds.py_ftns import hbv_mult_cat_loop_py

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


def solve_cats_sys_forcings(
        in_cats_prcssed_df,
        in_stms_prcssed_df,
        in_dem_net_df,
        in_use_step_ser,
        in_q_df,
        in_ppt_dfs_dict,
        in_tem_dfs_dict,
        in_pet_dfs_dict,
        aux_cell_vars_dict,
        time_freq,
        warm_up_steps,
        n_cpus,
        route_type,
        out_dir,
        all_prms_flags,
        min_q_thresh,
        sep,
        kfolds,
        use_obs_flow_flag,
        run_as_lump_flag,
        run_times,
        sim_prms_dict):

    '''run for a given set of catchments'''

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

    _df = None

    assert isinstance(aux_cell_vars_dict, dict)

    assert isinstance(time_freq, str)

    assert isinstance(warm_up_steps, int)
    assert warm_up_steps >= 1

    assert isinstance(n_cpus, int)
    assert n_cpus >= 1

    assert isinstance(route_type, int)
    assert route_type >= 0

    assert isinstance(out_dir, str)

    assert isinstance(all_prms_flags, np.ndarray)
    assert all_prms_flags.ndim == 2
    assert np.issubdtype(all_prms_flags.dtype, np.int32)

    assert isinstance(min_q_thresh, (float, int))
    assert min_q_thresh >= 0

    assert isinstance(sep, str)

    assert isinstance(kfolds, int)
    assert kfolds > 0

    assert isinstance(use_obs_flow_flag, bool)
    assert isinstance(run_as_lump_flag, bool)

    assert isinstance(run_times, list)
    assert (len(run_times) == 2)
    for tstamp in run_times:
        assert isinstance(tstamp, pd.Timestamp)

    sel_cats = in_cats_prcssed_df.index.values.copy(order='C')
    assert np.all(np.isfinite(sel_cats))

    in_q_df.columns = pd.to_numeric(in_q_df.columns)

    assert (
        in_ppt_dfs_dict.keys() ==
        in_tem_dfs_dict.keys() ==
        in_pet_dfs_dict.keys())

    if len(run_times) == 2:
        assert run_times[0] < run_times[1]

        date_range = pd.date_range(run_times[0], run_times[1], freq=time_freq)

    else:
        raise ValueError(f'cv_list has incorrect number of values!')

    assert isinstance(sim_prms_dict, dict)

    print('Running simulation...')

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

    in_use_step_ser = in_use_step_ser.loc[date_range]

    assert np.all(
        (in_use_step_ser.values >= 0) & (in_use_step_ser.values <= 1))

    assert np.any(in_use_step_ser.values > 0)

    if in_use_step_ser.values.sum() == in_use_step_ser.shape[0]:
        use_step_flag = False

    else:
        use_step_flag = True

    print('INFO: use_step_flag:', use_step_flag)

    kwargs = {
        'use_obs_flow_flag': use_obs_flow_flag,
        'run_as_lump_flag': run_as_lump_flag,
        'use_step_flag': use_step_flag,
        'min_q_thresh': min_q_thresh,
        'time_freq': time_freq,
        'kfolds': kfolds}

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

    for kf_i in range(1, kfolds + 1):

        # for the calibrated params, run for all the time steps
        solve_cat_forcings(
            in_use_step_ser,
            in_q_df,
            fin_ppt_dfs_dict,
            fin_tem_dfs_dict,
            fin_pet_dfs_dict,
            k_aux_cell_vars_dict,
            in_cats_prcssed_df,
            in_stms_prcssed_df,
            in_dem_net_df,
            sim_prms_dict[f'kf_{kf_i:02d}'],
            all_prms_flags,
            route_type,
            warm_up_steps,
            n_cpus,
            dirs_dict,
            sep,
            ini_arrs_dict,
            cat_to_idx_dict,
            stm_to_idx_dict,
            kf_i,
            kwargs=kwargs)

    os.chdir(old_wd)
    return


def solve_cat_forcings(
        in_use_step_ser,
        in_q_df,
        in_ppt_dfs_dict,
        in_tem_dfs_dict,
        in_pet_dfs_dict,
        in_aux_vars_dict,
        in_cats_prcssed_df,
        in_stms_prcssed_df,
        in_dem_net_df,
        sim_prms_dict,
        all_prms_flags,
        route_type,
        warm_up_steps,
        n_cpus,
        dirs_dict,
        sep,
        ini_arrs_dict,
        cat_to_idx_dict,
        stm_to_idx_dict,
        kf_i,
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

    calib_valid_suff = 'calib'

    use_obs_flow_flag = int(kwargs['use_obs_flow_flag'])
    run_as_lump_flag = int(kwargs['run_as_lump_flag'])
    use_step_flag = int(kwargs['use_step_flag'])
    min_q_thresh = float(kwargs['min_q_thresh'])
    time_freq = str(kwargs['time_freq'])
    kfolds = int(kwargs['kfolds'])

    assert in_use_step_ser.shape[0] == in_q_df.shape[0]

    if use_step_flag:
        use_step_arr = in_use_step_ser.values.astype(np.int32, 'c')

    else:
        use_step_arr = np.array([0], dtype=np.int32)

    print('\n\n')
    print('#' * 10)
    print('#' * 10)
    print(f'Run type: {calib_valid_suff}')
    print(f'n_cats: {in_cats_prcssed_df.shape[0]}')

    print(f'n_cpus: {n_cpus}')
    print(f'n_recs: {cats_outflow_arr.shape[0]}')
    print(f'min_q_thresh: {min_q_thresh}')

    for cat in in_cats_prcssed_df.index:
        print('\n')
        print('#' * 10)
        print(f'Going through cat: {cat} and kfold: {kf_i}')
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

        curr_cat_params.append(cat_area_ratios_arr)

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

            out_db_dict['tem_arr'] = tem_arr
            out_db_dict['ppt_arr'] = ppt_arr
            out_db_dict['pet_arr'] = pet_arr
            out_db_dict['qact_arr'] = q_arr
            out_db_dict['ini_arr'] = ini_arr

            for key in out_db_dict:
                if (n_cells > 1) and (key == 'outs_arr'):
                    continue

                db[f'{calib_valid_suff}/kf_{kf_i:02d}/{key}'] = out_db_dict[key]

            db[f'{calib_valid_suff}/kf_{kf_i:02d}/hbv_prms'] = sim_prms_dict[cat][0]

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
                data_sb.attrs['cv_flag'] = False

                dt = h5py.special_dtype(vlen=str)
                _prms_ds = data_sb.create_dataset(
                    'all_prms_labs', (len(all_prms_labs),), dtype=dt)
                _prms_ds[:] = all_prms_labs

                data_sb['shape'] = cat_shape
                data_sb['rows'] = cat_rows_idxs
                data_sb['cols'] = cat_cols_idxs

                for key in dirs_dict:
                    data_sb.attrs[key] = dirs_dict[key]

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

            if calib_valid_suff not in db:
                db[calib_valid_suff] = {}

            db_str = calib_valid_suff  # 'valid'

            db[db_str][kf_str] = {}
            for key in hgs_dict:
                db[db_str][kf_str][key] = hgs_dict[key]

    print('#' * 10)
    print('#' * 10)
    return


def get_k_aux_dict(aux_dict, area_dict, cats, lf):

    out_dict = {cat: aux_dict[cat] for cat in cats}

    if lf:
        out_dict = {
            cat: np.array([
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
            cat: np.array([area_dict[cat].sum()]) for cat in cats}

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
    tags = np.array(tags, dtype=np.uint64)

    assert np.all(tags[1:] - tags[:-1] > 0)

    return tags


def get_ft_maxi_freq_idx(ft_maxi_freq, n_recs):

    if n_recs % 2:
        n_recs -= 1  # same is used inside the opt

    if (ft_maxi_freq == 'A') or (ft_maxi_freq == 'Y'):
        # assuming 365 days a year
        assert n_recs > 365

        ft_maxi_freq_idx = n_recs // 365  # leap years?

    elif ft_maxi_freq == 'M':
        # assuming a 30 day month
        assert n_recs > 30

        ft_maxi_freq_idx = n_recs // 30

    elif ft_maxi_freq == 'W':
        assert n_recs > 7

        ft_maxi_freq_idx = n_recs // 7

    else:
        raise ValueError(f'ft_maxi_freq not defined for: {ft_maxi_freq}!')

    assert ft_maxi_freq_idx > 1, 'No. of values is too little!'

    return ft_maxi_freq_idx - 1

