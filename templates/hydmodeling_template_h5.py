"""
Created on %(date)s

@author: %(Faizan Anwar, IWS Uni-Stuttgart)s
"""

import os
import time
import timeit
from shutil import copy2
from datetime import datetime
import configparser as cfpm
from collections import OrderedDict

import h5py
import numpy as np
import pandas as pd
from psutil import cpu_count

from hydmodeling import (
    TauDEMAnalysis,
    get_stms,
    crt_strms_rltn_tree,
    plot_strm_rltn,
    get_cumm_cats,
    solve_cats_sys,
    plot_cats_hbv_sim,
    plot_cats_prm_vecs,
    plot_cats_kfold_effs,
    plot_cats_best_prms_1d,
    plot_cats_best_prms_2d,
    plot_cats_ann_cycs_fdcs_comp,
    plot_cats_prms_transfer_perfs,
    plot_cats_prm_vecs_evo,
    plot_cats_vars_errors,
    plot_cats_qsims,
    plot_cats_discharge_errors,
    plot_cats_stats)

from hydmodeling.plotting.perfs import get_peaks_mask


def get_data_dict_from_h5(path_to_h5, ds_grp, set_na_to_zero_flag=False):

    out_data_dict = {}
    with h5py.File(path_to_h5, mode='r', driver=None) as h5_hdl:
        h5_times = pd.to_datetime(
            h5_hdl['time/time_strs'][...], format='%Y%m%dT%H%M%S')

        data_ds = h5_hdl[ds_grp]

        keys = [int(key) for key in data_ds.keys()]

        for key in keys:
            out_data_dict[key] = pd.DataFrame(
                index=h5_times, data=data_ds[str(key)][:])

            if set_na_to_zero_flag:
                nan_ct = np.isnan(out_data_dict[key].values).sum()

                if nan_ct:
                    print('\n')
                    print('#' * 30)
                    print(
                        f'WARNING: Set {nan_ct} values to zero in dataset '
                        f'{key} in file: {os.path.basename(path_to_h5)}!')
                    print('#' * 30)
                    print('\n')

                    out_data_dict[key].replace(np.nan, 0.0, inplace=True)

    return out_data_dict


def get_cell_vars_dict_from_h5(path_to_h5):

    cat_intersect_rows_dict = {}
    cat_intersect_cols_dict = {}
    cat_area_ratios_dict = {}
    extent_shape = [-np.inf, -np.inf]

    with h5py.File(path_to_h5, mode='r', driver=None) as h5_hdl:
        keys = [int(key) for key in h5_hdl['rows'].keys()]

        for key in keys:
            rows = h5_hdl[f'rows/{key}'][...]
            cols = h5_hdl[f'cols/{key}'][...]
            key_area_ratios = h5_hdl[f'rel_itsctd_area/{key}'][...]

            cat_intersect_rows_dict[key] = rows
            cat_intersect_cols_dict[key] = cols
            cat_area_ratios_dict[key] = key_area_ratios

            if rows.max() > extent_shape[0]:
                extent_shape[0] = rows.max()

            if cols.max() > extent_shape[0]:
                extent_shape[1] = cols.max()

    return {
        'rows': cat_intersect_rows_dict,
        'cols': cat_intersect_cols_dict,
        'shape': extent_shape,
        'area_ratios': cat_area_ratios_dict}


def main():
    in_ini_file = r'config_hydmodeling_template_h5.ini'

    cfp = cfpm.ConfigParser(interpolation=cfpm.ExtendedInterpolation())
    cfp.read(in_ini_file)

    in_ini_abs_path = os.path.abspath(in_ini_file)

    n_cpus = cfp['DEFAULT']['n_cpus']
    if n_cpus == 'auto':
        n_cpus = cpu_count() - 1

    else:
        n_cpus = int(n_cpus)

    old_chdir = os.getcwd()

    main_dir = cfp['DEFAULT']['main_dir']
    os.chdir(main_dir)

    hyd_analysis_flag = False
    get_stms_flag = False
    create_stms_rels_flag = False
    create_cumm_cats_flag = False
    optimize_flag = False
    plot_kfold_perfs_flag = False
    plot_best_kfold_prms_flag = False
    plot_prm_vecs_flag = False
    plot_2d_kfold_prms_flag = False
    plot_ann_cys_fdcs_flag = False
    plot_prm_trans_comp_flag = False
    plot_opt_evo_flag = False
    plot_var_errors_flag = False
    plot_hbv_vars_flag = False
    plot_stats_flag = False
    plot_qsims_flag = False
    plot_cats_discharge_errs_flag = False

#     hyd_analysis_flag = True
#     get_stms_flag = True
#     create_cumm_cats_flag = True
    create_stms_rels_flag = True
    optimize_flag = True
    plot_kfold_perfs_flag = True
    plot_best_kfold_prms_flag = True
    plot_prm_vecs_flag = True
    plot_2d_kfold_prms_flag = True
    plot_ann_cys_fdcs_flag = True
    plot_prm_trans_comp_flag = True
    plot_opt_evo_flag = True
    plot_var_errors_flag = True
    plot_hbv_vars_flag = True
    plot_stats_flag = True
#     plot_qsims_flag = True
#     plot_cats_discharge_errs_flag = True

    use_cv_time_flag = False
    use_cv_time_flag = True

    #=========================================================================
    # This performs the hydrological preprocessing
    #=========================================================================
    show_ansys_stdout = cfp['HYD_ANSYS'].getboolean('show_ansys_stdout')
    hyd_ansys_runtype = cfp['HYD_ANSYS']['hyd_ansys_runtype']
    calc_for_cats_only = cfp['HYD_ANSYS'].getboolean('calc_for_cats_only')
    max_cell_move = cfp['HYD_ANSYS'].getint('max_cell_move')
    strm_strt_thresh = cfp['HYD_ANSYS'].getint('strm_strt_thresh')

    out_pre_proc_dir = cfp['HYD_ANSYS']['out_pre_proc_dir']
    in_dem_loc = cfp['HYD_ANSYS']['in_dem_loc']
    in_gage_shp_loc = cfp['HYD_ANSYS']['in_gage_shp_loc']

    hyd_ansys = TauDEMAnalysis(
        in_dem_loc,
        in_gage_shp_loc,
        out_pre_proc_dir,
        n_cpus=n_cpus)

    hyd_ansys.run_type = hyd_ansys_runtype
    hyd_ansys.strm_orign_thresh = strm_strt_thresh
    hyd_ansys.max_cell_move = max_cell_move
    hyd_ansys.verbose = show_ansys_stdout
    hyd_ansys.area_flag = calc_for_cats_only

    if hyd_analysis_flag:
        hyd_ansys()

    #=========================================================================
    # This extracts the required streams for catchments from the shapefiles
    # that we get from TauDEM
    #=========================================================================
    in_dem_net_shp_file = hyd_ansys.dem_net
    in_wat_ids_file = hyd_ansys.watersheds_ids
    out_dem_net_shp_file = cfp['GET_STMS']['out_dem_net_shp_file']
    in_dem_file = hyd_ansys.fil
    in_cats_file = hyd_ansys.watersheds_shp
    in_gauges_coords_file = hyd_ansys.gage_shp_moved
    gauge_coords_field_name = cfp['GET_STMS']['gauge_coords_field_name']
    out_df_file = cfp['GET_STMS']['dem_net_file']
    out_wat_ids_file = cfp['GET_STMS']['out_wat_ids_file']
    sep = cfp['DEFAULT']['sep']

    if get_stms_flag:
        get_stms(
            in_dem_net_shp_file,
            in_wat_ids_file,
            in_dem_file,
            in_cats_file,
            in_gauges_coords_file,
            out_dem_net_shp_file,
            out_df_file,
            out_wat_ids_file,
            sep,
            gauge_coords_field_name)

    #=========================================================================
    # This creates a stream relationship tree based on their order of
    # occurrence in the out_df_file
    #=========================================================================
    prcss_cats_list = cfp['CREATE_STM_RELS']['prcss_cats_list'].split(sep)

    hyd_mod_dir = cfp['CREATE_STM_RELS']['hyd_mod_dir']
    out_cats_prcssed_file = cfp['CREATE_STM_RELS']['cats_prcssed_file']
    out_stms_prcssed_file = cfp['CREATE_STM_RELS']['stms_prcssed_file']
    watershed_field_name = cfp['CREATE_STM_RELS']['watershed_field_name']
    out_cats_rel_fig_path = cfp['CREATE_STM_RELS']['out_cats_rel_fig_path']

    if not os.path.exists(hyd_mod_dir):
        os.mkdir(hyd_mod_dir)

    in_ini_name, in_ini_ext = os.path.basename(in_ini_file).rsplit('.', 1)

    crnt_time = datetime.now().strftime('%Y%m%d%H%M%S')

    copy2(in_ini_abs_path, os.path.join(
        hyd_mod_dir, f'{in_ini_name}_{crnt_time}.{in_ini_ext}'))

    if create_stms_rels_flag:
        crt_strms_rltn_tree(
            prcss_cats_list,
            out_df_file,
            in_cats_file,
            out_cats_prcssed_file,
            out_stms_prcssed_file,
            sep,
            watershed_field_name)

        plot_strm_rltn(
            hyd_ansys.watersheds_shp,
            hyd_ansys.gage_shp_moved,
            out_dem_net_shp_file,
            out_df_file,
            out_cats_prcssed_file,
            out_stms_prcssed_file,
            prcss_cats_list,
            out_cats_rel_fig_path,
            sep=sep)

    out_cumm_cat_shp = cfp['CUMM_CATS']['out_cumm_cat_shp']
    out_cumm_cat_descrip_file = cfp['CUMM_CATS']['out_cumm_cat_descrip_file']

    if create_cumm_cats_flag:
        get_cumm_cats(
            in_cats_file,
            watershed_field_name,
            out_wat_ids_file,
            sep,
            out_cumm_cat_shp,
            out_cumm_cat_descrip_file,
            sep)

    #=========================================================================
    # Optimize hydrologic model
    #=========================================================================
    in_dem_net_file = cfp['GET_STMS']['dem_net_file']
    in_cats_prcssed_file = cfp['CREATE_STM_RELS']['cats_prcssed_file']
    in_stms_prcssed_file = cfp['CREATE_STM_RELS']['stms_prcssed_file']

    sep = cfp['DEFAULT']['sep']

    time_fmt = cfp['OPT_HYD_MODEL']['time_fmt']

    if use_cv_time_flag:
        (start_cdate,
         end_cdate) = cfp['OPT_HYD_MODEL']['calib_dates'].split(sep)

        (start_vdate,
         end_vdate) = cfp['OPT_HYD_MODEL']['valid_dates'].split(sep)

        start_cdate, end_cdate = pd.to_datetime(
            [start_cdate, end_cdate], format=time_fmt)

        start_vdate, end_vdate = pd.to_datetime(
            [start_vdate, end_vdate], format=time_fmt)

        kfolds = cfp['OPT_HYD_MODEL'].getint('kfolds')

        if kfolds != 1:
            print(
                f'WARNING: kfolds set to 1 from {kfolds} due '
                f'to use_cv_time_flag!')

            kfolds = 1

        cv_list = [start_cdate, end_cdate, start_vdate, end_vdate]

    else:
        kfolds = cfp['OPT_HYD_MODEL'].getint('kfolds')

        start_date, end_date = cfp['OPT_HYD_MODEL']['sim_dates'].split(sep)

        start_date, end_date = pd.to_datetime(
            [start_date, end_date], format=time_fmt)

        cv_list = [start_date, end_date]

    time_freq = cfp['OPT_HYD_MODEL']['time_freq']

    warm_up_steps = cfp['OPT_HYD_MODEL'].getint('warm_up_steps')
    water_bal_step_size = cfp['OPT_HYD_MODEL'].getint('water_bal_step_size')
    route_type = cfp['OPT_HYD_MODEL'].getint('route_type')

    compare_ann_cyc_flag = cfp['OPT_HYD_MODEL'].getboolean(
        'compare_ann_cyc_flag')
    use_obs_flow_flag = cfp['OPT_HYD_MODEL'].getboolean('use_obs_flow_flag')

    min_q_thresh = cfp['OPT_HYD_MODEL'].getfloat('min_q_thresh')
    run_as_lump_flag = cfp['OPT_HYD_MODEL'].getboolean('run_as_lump_flag')
    obj_ftn_wts = np.array(
        cfp['OPT_HYD_MODEL']['obj_ftn_wts'].split(sep), dtype=np.float64)

    use_resampled_obj_ftns_flag = cfp['OPT_HYD_MODEL'].getboolean(
        'use_resampled_obj_ftns_flag')

    discharge_resampling_freq = cfp['OPT_HYD_MODEL'][
        'discharge_resampling_freq']

    ft_beg_freq = cfp['OPT_HYD_MODEL']['ft_beg_freq']
    ft_beg_freq_inc_flag = cfp['OPT_HYD_MODEL'].getboolean(
        'ft_beg_freq_inc_flag')

    ft_end_freq = cfp['OPT_HYD_MODEL']['ft_end_freq']
    ft_end_freq_inc_flag = cfp['OPT_HYD_MODEL'].getboolean(
        'ft_end_freq_inc_flag')

    in_opt_schm_vars_dict = cfp['OPT_SCHM_VARS']

    opt_schm_vars_dict = {}
    if in_opt_schm_vars_dict['opt_schm'] == 'DE':
        opt_schm_vars_dict['opt_schm'] = 'DE'

        opt_schm_vars_dict['mu_sc_fac_bds'] = np.array(
            in_opt_schm_vars_dict['mu_sc_fac_bds'].split(sep),
            dtype=np.float64)

        opt_schm_vars_dict['cr_cnst_bds'] = np.array(
            in_opt_schm_vars_dict['cr_cnst_bds'].split(sep),
            dtype=np.float64)

    elif in_opt_schm_vars_dict['opt_schm'] == 'ROPE':
        opt_schm_vars_dict['opt_schm'] = 'ROPE'
        opt_schm_vars_dict['acc_rate'] = in_opt_schm_vars_dict.getfloat(
            'acc_rate')
        opt_schm_vars_dict['n_uvecs_exp'] = in_opt_schm_vars_dict.getfloat(
            'n_uvecs_exp')
        opt_schm_vars_dict['n_rope_prm_vecs_exp'] = (
            in_opt_schm_vars_dict.getfloat('n_rope_prm_vecs_exp'))
        opt_schm_vars_dict['max_chull_tries'] = (
            in_opt_schm_vars_dict.getint('max_chull_tries'))

        opt_schm_vars_dict['depth_ftn_type'] = (
            in_opt_schm_vars_dict.getint('depth_ftn_type'))

        opt_schm_vars_dict['min_pts_in_chull'] = (
            in_opt_schm_vars_dict.getint('min_pts_in_chull'))

    elif in_opt_schm_vars_dict['opt_schm'] == 'BRUTE':
        opt_schm_vars_dict['opt_schm'] = 'BRUTE'

        opt_schm_vars_dict['n_discretize'] = (
            in_opt_schm_vars_dict.getint('n_discretize'))

    else:
        raise NotImplementedError(
            'Incorrect opt_schm: %s' % in_opt_schm_vars_dict['opt_schm'])

    if ((in_opt_schm_vars_dict['opt_schm'] == 'DE') or (
         in_opt_schm_vars_dict['opt_schm'] == 'ROPE')):

        opt_schm_vars_dict['max_iters'] = in_opt_schm_vars_dict.getint(
            'max_iters')
        opt_schm_vars_dict['max_cont_iters'] = in_opt_schm_vars_dict.getint(
            'max_cont_iters')
        opt_schm_vars_dict['obj_ftn_tol'] = in_opt_schm_vars_dict.getfloat(
            'obj_ftn_tol')
        opt_schm_vars_dict['prm_pcnt_tol'] = in_opt_schm_vars_dict.getfloat(
            'prm_pcnt_tol')

    opt_schm_vars_dict['n_prm_vecs_exp'] = in_opt_schm_vars_dict.getfloat(
        'n_prm_vecs_exp')

    bounds_dict = OrderedDict()

    bounds_dict['tt_bds'] = [
        float(_) for _ in cfp['PARAM_BOUNDS']['tt'].split(sep)]

    bounds_dict['cm_bds'] = [
        float(_) for _ in cfp['PARAM_BOUNDS']['cm'].split(sep)]

    bounds_dict['pcm_bds'] = [
        float(_) for _ in cfp['PARAM_BOUNDS']['pcm'].split(sep)]

    bounds_dict['fc_bds'] = [
        float(_) for _ in cfp['PARAM_BOUNDS']['fc_pwp'].split(sep)]

    bounds_dict['beta_bds'] = [
        float(_) for _ in cfp['PARAM_BOUNDS']['beta'].split(sep)]

    bounds_dict['pwp_bds'] = [
        float(_) for _ in cfp['PARAM_BOUNDS']['fc_pwp'].split(sep)]

    bounds_dict['ur_thr_bds'] = [
        float(_) for _ in cfp['PARAM_BOUNDS']['ur_thr'].split(sep)]

    bounds_dict['k_uu_bds'] = [
        float(_) for _ in cfp['PARAM_BOUNDS']['k_uu'].split(sep)]

    bounds_dict['k_ul_bds'] = [
        float(_) for _ in cfp['PARAM_BOUNDS']['k_ul'].split(sep)]

    bounds_dict['k_d_bds'] = [
        float(_) for _ in cfp['PARAM_BOUNDS']['k_d'].split(sep)]

    bounds_dict['k_ll_bds'] = [
        float(_) for _ in cfp['PARAM_BOUNDS']['k_ll'].split(sep)]

    bounds_dict['exp_bds'] = [
        float(_) for _ in cfp['PARAM_BOUNDS']['exp'].split(sep)]

    bounds_dict['musk_lag_bds'] = [
        float(_) for _ in cfp['PARAM_BOUNDS']['musk_lag'].split(sep)]

    bounds_dict['musk_wt_bds'] = [
        float(_) for _ in cfp['PARAM_BOUNDS']['musk_wt'].split(sep)]

    tt_flags = [int(_) for _ in cfp['PRM_FLAGS']['tt'].split(sep)]
    cm_flags = [int(_) for _ in cfp['PRM_FLAGS']['cm'].split(sep)]
    pcm_flags = [int(_) for _ in cfp['PRM_FLAGS']['pcm'].split(sep)]
    fc_flags = [int(_) for _ in cfp['PRM_FLAGS']['fc_pwp'].split(sep)]
    beta_flags = [int(_) for _ in cfp['PRM_FLAGS']['beta'].split(sep)]
    pwp_flags = [int(_) for _ in cfp['PRM_FLAGS']['fc_pwp'].split(sep)]
    ur_thr_flags = [int(_) for _ in cfp['PRM_FLAGS']['ur_thr'].split(sep)]
    k_uu_flags = [int(_) for _ in cfp['PRM_FLAGS']['k_uu'].split(sep)]
    k_ul_flags = [int(_) for _ in cfp['PRM_FLAGS']['k_ul'].split(sep)]
    k_d_flags = [int(_) for _ in cfp['PRM_FLAGS']['k_d'].split(sep)]
    k_ll_flags = [int(_) for _ in cfp['PRM_FLAGS']['k_ll'].split(sep)]

    all_prms_flags = np.array([
        tt_flags,
        cm_flags,
        pcm_flags,
        fc_flags,
        beta_flags,
        pwp_flags,
        ur_thr_flags,
        k_uu_flags,
        k_ul_flags,
        k_d_flags,
        k_ll_flags],
        dtype=np.int32)

    if optimize_flag:
        in_cats_prcssed_df = pd.read_csv(
            in_cats_prcssed_file, sep=str(sep), index_col=0)

        in_stms_prcssed_df = pd.read_csv(
            in_stms_prcssed_file, sep=str(sep), index_col=0)

        in_dem_net_df = pd.read_csv(
            in_dem_net_file, sep=str(sep), index_col=0)

        # always in cumecs
        obs_q_file = cfp['OPT_HYD_MODEL']['obs_q_file']

        ppt_file = cfp['OPT_HYD_MODEL']['ppt_file']
        ppt_ds_grp = cfp['OPT_HYD_MODEL']['ppt_ds_grp']

        temp_file = cfp['OPT_HYD_MODEL']['temp_file']
        temp_ds_grp = cfp['OPT_HYD_MODEL']['temp_ds_grp']

        pet_file = cfp['OPT_HYD_MODEL']['pet_file']
        pet_ds_grp = cfp['OPT_HYD_MODEL']['pet_ds_grp']

        in_q_df = pd.read_csv(obs_q_file, sep=str(sep), index_col=0)
        in_q_df.index = pd.to_datetime(in_q_df.index, format=time_fmt)

        in_ppt_dfs_dict = get_data_dict_from_h5(ppt_file, ppt_ds_grp, True)
        in_temp_dfs_dict = get_data_dict_from_h5(temp_file, temp_ds_grp)
        in_pet_dfs_dict = get_data_dict_from_h5(pet_file, pet_ds_grp)
        in_cell_vars_dict = get_cell_vars_dict_from_h5(ppt_file)

        in_use_step_df = pd.DataFrame(
            columns=list(in_ppt_dfs_dict.keys()),
            index=in_q_df.index,
            data=np.ones((in_q_df.shape[0], len(in_ppt_dfs_dict.keys())),
                dtype=np.int32))

#         for col in in_use_step_df.columns:
#             if use_cv_time_flag:
#                 in_use_step_df.loc[start_cdate:end_cdate, col
#                     ] = get_peaks_mask(
#                         in_q_df.loc[start_cdate:end_cdate, str(col)].values).astype(int)
#
#             else:
#                 in_use_step_df.loc[start_cdate:end_cdate, col
#                     ] = get_peaks_mask(
#                         in_q_df.loc[start_date:end_date, str(col)].values).astype(int)

        aux_cell_vars_dict = {}
        aux_cell_vars_dict['area_ratios'] = in_cell_vars_dict['area_ratios']
        aux_cell_vars_dict['shape'] = in_cell_vars_dict['shape']
        aux_cell_vars_dict['rows'] = in_cell_vars_dict['rows']
        aux_cell_vars_dict['cols'] = in_cell_vars_dict['cols']

        if np.any(all_prms_flags[:, 1]):
            aux_cell_vars_dict['lulc_ratios'] = in_cell_vars_dict[
                'lulc_ratios']

        if np.any(all_prms_flags[:, 2]):
            aux_cell_vars_dict['soil_ratios'] = in_cell_vars_dict[
                'soil_ratios']

        if np.any(all_prms_flags[:, 3]) or np.any(all_prms_flags[:, 5]):
            aux_cell_vars_dict['aspect'] = in_cell_vars_dict['aspect']

        if np.any(all_prms_flags[:, 4]) or np.any(all_prms_flags[:, 5]):
            aux_cell_vars_dict['slope'] = in_cell_vars_dict['slope']

        _beg_t = timeit.default_timer()

        solve_cats_sys(
            in_cats_prcssed_df,
            in_stms_prcssed_df,
            in_dem_net_df,
            in_use_step_df,
            in_q_df,
            in_ppt_dfs_dict,
            in_temp_dfs_dict,
            in_pet_dfs_dict,
            aux_cell_vars_dict,
            time_fmt,
            time_freq,
            warm_up_steps,
            n_cpus,
            route_type,
            hyd_mod_dir,
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
            ft_beg_freq,
            ft_beg_freq_inc_flag,
            ft_end_freq,
            ft_end_freq_inc_flag)

        _end_t = timeit.default_timer()
        _tot_t = _end_t - _beg_t

        print('\n\n')
        print('#' * 10)
        print(f'Total calibration time was: {_tot_t:0.4f} secs!')
        print('#' * 10)

    dbs_dir = os.path.join(hyd_mod_dir, r'01_database')
    hgs_db_path = os.path.join(hyd_mod_dir, r'02_hydrographs/hgs_dfs')

    #=========================================================================
    # Plot the k-fold results
    #=========================================================================
    if plot_kfold_perfs_flag:
        _beg_t = timeit.default_timer()

        print('\n\n')
        print('#' * 10)
        print('Plotting kfold results...')

        plot_cats_kfold_effs(
            dbs_dir, hgs_db_path, compare_ann_cyc_flag, n_cpus)

        _end_t = timeit.default_timer()
        _tot_t = _end_t - _beg_t

        print(f'Took {_tot_t:0.4f} seconds!')
        print('#' * 10)

    #=========================================================================
    # Plot the best k-fold params
    #=========================================================================
    if plot_best_kfold_prms_flag:
        _beg_t = timeit.default_timer()

        print('\n\n')
        print('#' * 10)
        print('Plotting best kfold prms...')

        plot_cats_best_prms_1d(dbs_dir, n_cpus)

        _end_t = timeit.default_timer()
        _tot_t = _end_t - _beg_t

        print(f'Took {_tot_t:0.4f} seconds!')
        print('#' * 10)

    #============================ ============================================
    # Plot final parameter population
    #=========================================================================
    if plot_prm_vecs_flag:
        _beg_t = timeit.default_timer()

        print('\n\n')
        print('#' * 10)
        print('Plotting parameter vectors...')

        plot_cats_prm_vecs(dbs_dir, n_cpus)

        _end_t = timeit.default_timer()
        _tot_t = _end_t - _beg_t

        print(f'Took {_tot_t:0.4f} seconds!')
        print('#' * 10)

    #=========================================================================
    # Plot hbv prms for all catchments per kfold in 2d
    #=========================================================================
    if plot_2d_kfold_prms_flag:
        _beg_t = timeit.default_timer()

        print('\n\n')
        print('#' * 10)
        print('Plotting HBV prms in 2D...')

        plot_cats_best_prms_2d(dbs_dir)

        _end_t = timeit.default_timer()
        _tot_t = _end_t - _beg_t

        print(f'Took {_tot_t:0.4f} seconds!')
        print('#' * 10)

    #=========================================================================
    # Plot annual cycle and FDC comparison
    #=========================================================================
    if plot_ann_cys_fdcs_flag:
        _beg_t = timeit.default_timer()

        print('\n\n')
        print('#' * 10)
        print('Plotting annual cycle and FDC comparison...')

        ann_cyc_fdc_plot_dir = os.path.join(
            hyd_mod_dir, r'08_ann_cycs_fdc_comparison')

        plot_cats_ann_cycs_fdcs_comp(
            hgs_db_path, warm_up_steps, ann_cyc_fdc_plot_dir)

        _end_t = timeit.default_timer()
        _tot_t = _end_t - _beg_t

        print(f'Took {_tot_t:0.4f} seconds!')
        print('#' * 10)

    #=========================================================================
    # Plot catchment parameter transfer comparison
    #=========================================================================
    if plot_prm_trans_comp_flag:
        _beg_t = timeit.default_timer()

        print('\n\n')
        print('#' * 10)
        print('Plotting catchment parameter comparison...')

        plot_cats_prms_transfer_perfs(dbs_dir, n_cpus)

        _end_t = timeit.default_timer()
        _tot_t = _end_t - _beg_t

        print(f'Took {_tot_t:0.4f} seconds!')
        print('#' * 10)

    #=========================================================================
    # Plot catchment parameter transfer comparison
    #=========================================================================
    if plot_opt_evo_flag:

        plot_evo_obj_flag = cfp['PLOT_OPT_RES'].getboolean(
            'plot_evo_obj_flag')

        if opt_schm_vars_dict['opt_schm'] != 'DE':  # == 'ROPE':
            plot_evo_png_flag = cfp['PLOT_OPT_RES'].getboolean(
                'plot_evo_png_flag')

            plot_evo_gif_flag = cfp['PLOT_OPT_RES'].getboolean(
                'plot_evo_gif_flag')

        else:
            plot_evo_png_flag = False
            plot_evo_gif_flag = False

        evo_anim_secs = cfp['PLOT_OPT_RES'].getint('evo_anim_secs')

        _beg_t = timeit.default_timer()

        print('\n\n')
        print('#' * 10)
        print('Plotting optimization parameters\' evolution...')

        print(f'plot_evo_obj_flag: {plot_evo_obj_flag}')
        print(f'plot_evo_png_flag: {plot_evo_png_flag}')
        print(f'plot_evo_gif_flag: {plot_evo_gif_flag}')
        print(f'evo_anim_secs: {evo_anim_secs}')

        if plot_evo_obj_flag or plot_evo_png_flag or plot_evo_gif_flag:
            plot_cats_prm_vecs_evo(
                dbs_dir,
                plot_evo_obj_flag,
                plot_evo_png_flag,
                plot_evo_gif_flag,
                evo_anim_secs,
                n_cpus)

        else:
            print('All flags False, not plotting!')

        _end_t = timeit.default_timer()
        _tot_t = _end_t - _beg_t

        print(f'Took {_tot_t:0.4f} seconds!')
        print('#' * 10)

    #=========================================================================
    # Plot sorted errors w.r.t given variables
    #=========================================================================
    if plot_var_errors_flag:
        err_var_labs = cfp['PLOT_OPT_RES']['err_var_labs'].split(sep)
        assert err_var_labs, err_var_labs

        _beg_t = timeit.default_timer()

        print('\n\n')
        print('#' * 10)
        print('Plotting discharge errors against simulated variables...')

        print(f'err_var_labs: {err_var_labs}')

        plot_cats_vars_errors(dbs_dir, err_var_labs, n_cpus)

        _end_t = timeit.default_timer()
        _tot_t = _end_t - _beg_t

        print(f'Took {_tot_t:0.4f} seconds!')
        print('#' * 10)

    #==========================================================================
    # Plot stats
    #==========================================================================
    if plot_stats_flag:
        print('\n\n')
        print('#' * 10)
        print('Plotting stats...')

        _beg_t = timeit.default_timer()

        plot_cats_stats(dbs_dir, n_cpus)

        _end_t = timeit.default_timer()
        _tot_t = _end_t - _beg_t

        print(f'Took {_tot_t:0.4f} seconds!')
        print('#' * 10)

    #=========================================================================
    # plot the hbv variables
    #=========================================================================
    if plot_hbv_vars_flag:
        print('\n\n')
        print('#' * 10)
        print('Plotting hbv variables...')

        _beg_t = timeit.default_timer()

        plot_full_sim_flag = cfp['PLOT_OPT_RES'].getboolean(
            'plot_full_sim_flag')

        plot_wat_bal_flag = cfp['PLOT_OPT_RES'].getboolean(
            'plot_wat_bal_flag')

        show_warm_up_steps_flag = cfp['PLOT_OPT_RES'].getboolean(
            'show_warm_up_steps_flag')

        print(f'plot_full_sim_flag: {plot_full_sim_flag}')
        print(f'plot_wat_bal_flag: {plot_wat_bal_flag}')
        print(f'show_warm_up_steps_flag: {show_warm_up_steps_flag}')

        if plot_full_sim_flag or plot_wat_bal_flag:
            plot_cats_hbv_sim(
                dbs_dir,
                water_bal_step_size,
                plot_full_sim_flag,
                plot_wat_bal_flag,
                show_warm_up_steps_flag,
                n_cpus)

        else:
            print('Both flags False, not plotting!')

        _end_t = timeit.default_timer()
        _tot_t = _end_t - _beg_t

        print(f'Took {_tot_t:0.4f} seconds!')
        print('#' * 10)

    #=========================================================================
    # plot the discharge simulations
    #=========================================================================
    if plot_qsims_flag:
        print('\n\n')
        print('#' * 10)
        print('Plotting discharge simulations...')

        _beg_t = timeit.default_timer()

        plot_cats_qsims(dbs_dir, n_cpus)

        _end_t = timeit.default_timer()
        _tot_t = _end_t - _beg_t

        print(f'Took {_tot_t:0.4f} seconds!')
        print('#' * 10)

    #=========================================================================
    # plot the discharge simulations errors for different quantiles
    #=========================================================================
    if plot_cats_discharge_errs_flag:
        print('\n\n')
        print('#' * 10)
        print(
            'Plotting discharge simulations errors for various quantiles...')

        _beg_t = timeit.default_timer()

        plot_cats_discharge_errors(dbs_dir, n_cpus)

        _end_t = timeit.default_timer()
        _tot_t = _end_t - _beg_t

        print(f'Took {_tot_t:0.4f} seconds!')
        print('#' * 10)

    #=========================================================================

    #=========================================================================
    os.chdir(old_chdir)
    return


if __name__ == '__main__':
    _save_log_ = False
    if _save_log_:
        from std_logger import StdFileLoggerCtrl

        # save all console activity to out_log_file
        out_log_file = os.path.join(r'P:\\',
                                    r'Synchronize',
                                    r'python_script_logs',
                                    ('hydmodeling_template_log_%s.log' %
                                     datetime.now().strftime('%Y%m%d%H%M%S')))
        log_link = StdFileLoggerCtrl(out_log_file)

    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main()

    STOP = timeit.default_timer()  # Ending time
    print(('\n\a\a\a Done with everything on %s. Total run time was'
           ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
