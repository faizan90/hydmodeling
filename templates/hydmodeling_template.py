"""
Created on %(date)s

@author: %(Faizan Anwar, IWS Uni-Stuttgart)s
"""

import os
import time
import timeit
import pickle
import configparser as cfpm
from psutil import cpu_count
from collections import OrderedDict

import numpy as np
import pandas as pd

from hydmodeling import (
    TauDEMAnalysis,
    get_stms,
    crt_strms_rltn_tree,
    plot_strm_rltn,
    get_cumm_cats,
    solve_cats_sys,
    plot_vars,
    plot_prm_vecs,
    plot_kfold_effs,
    plot_kfolds_best_prms,
    plot_kfolds_best_hbv_prms_2d,
    plot_ann_cycs_fdcs_comp,
    plot_prm_trans_perfs)


def load_pickle(in_file, mode='rb'):
    with open(in_file, mode) as _pkl_hdl:
        return pickle.load(_pkl_hdl)
    return

# TODO: Plot Discharge, ppt, snow only. Add it to plot hbv ftn.
# Have an option in simple_opt ftn that take the vars and plot them in the
# same order.
# TODO: Plot subset of the data in hi-res, to see events of interest.
# TODO: Have a lower limit of the obj. val. if optmized obj. val is below this
# choose observed flow for further downstream cats.
# TODO: Have a fallback scenario for cats that cant be optimized.
# TODO: Have an array to record time spent in each ftn during optimization.
# TODO: Add a scaling factor to ppt (or pet) as an optimization parameter.
# TODO: Somehow identify flows that might be erroneous.
# passing flags from infilling?
# TODO: Have months or years as the axis labels. This can be derived from units.
# TODO: Have units, display them on figs.
# TODO: Have a check on time frequency for all input
# TODO: make snowmelt more physical.
# These parameters vary the most among kfolds
# TODO: See if chull algorithm is better than depth
# TODO: Make plots of DS cats better


def main():
    cfp = cfpm.ConfigParser(interpolation=cfpm.ExtendedInterpolation())
    cfp.read('config_hydmodeling_template.ini')

    n_cpus = cfp['DEFAULT']['n_cpus']
    if n_cpus == 'auto':
        n_cpus = cpu_count() - 1

    else:
        n_cpus = int(n_cpus)

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
    plot_hbv_vars_flag = False

#     hyd_analysis_flag = True
#     get_stms_flag = True
#     create_stms_rels_flag = True
#     create_cumm_cats_flag = True
#     optimize_flag = True
#     plot_kfold_perfs_flag = True
#     plot_best_kfold_prms_flag = True
    plot_prm_vecs_flag = True
#     plot_2d_kfold_prms_flag = True
#     plot_ann_cys_fdcs_flag = True
#     plot_prm_trans_comp_flag = True
#     plot_hbv_vars_flag = True

    # =============================================================================
    # This performs the hydrological preprocessing
    # =============================================================================
    show_ansys_stdout = cfp['HYD_ANSYS'].getboolean('show_ansys_stdout')
    hyd_ansys_runtype = cfp['HYD_ANSYS']['hyd_ansys_runtype']
    calc_for_cats_only = cfp['HYD_ANSYS'].getboolean('show_ansys_stdout')
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

    out_hyd_mod_dir = cfp['CREATE_STM_RELS']['hyd_mod_dir']
    out_cats_prcssed_file = cfp['CREATE_STM_RELS']['cats_prcssed_file']
    out_stms_prcssed_file = cfp['CREATE_STM_RELS']['stms_prcssed_file']
    watershed_field_name = cfp['CREATE_STM_RELS']['watershed_field_name']
    out_cats_rel_fig_path = cfp['CREATE_STM_RELS']['out_cats_rel_fig_path']

    if not os.path.exists(out_hyd_mod_dir):
        os.mkdir(out_hyd_mod_dir)

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

    #==========================================================================
    # Optimize hydrologic model
    #==========================================================================
    in_hyd_mod_dir = cfp['CREATE_STM_RELS']['hyd_mod_dir']

    in_dem_net_file = cfp['GET_STMS']['dem_net_file']
    in_cats_prcssed_file = cfp['CREATE_STM_RELS']['cats_prcssed_file']
    in_stms_prcssed_file = cfp['CREATE_STM_RELS']['stms_prcssed_file']

    # always in cumecs
    in_q_file = cfp['OPT_HYD_MODEL']['in_q_file']
    in_ppt_file = cfp['OPT_HYD_MODEL']['in_ppt_file']
    in_temp_file = cfp['OPT_HYD_MODEL']['in_temp_file']
    in_pet_file = cfp['OPT_HYD_MODEL']['in_pet_file']
    in_cell_vars_pkl = cfp['OPT_HYD_MODEL']['in_cell_idxs_pkl']

    sep = cfp['DEFAULT']['sep']

    in_date_fmt = cfp['OPT_HYD_MODEL']['in_date_fmt']
    start_date = cfp['OPT_HYD_MODEL']['start_date']
    end_date = cfp['OPT_HYD_MODEL']['end_date']
    time_freq = cfp['OPT_HYD_MODEL']['time_freq']

    warm_up_steps = cfp['OPT_HYD_MODEL'].getint('warm_up_steps')
    water_bal_step_size = cfp['OPT_HYD_MODEL'].getint('water_bal_step_size')
    route_type = cfp['OPT_HYD_MODEL'].getint('route_type')
    kfolds = cfp['OPT_HYD_MODEL'].getint('kfolds')
    compare_ann_cyc_flag = cfp['OPT_HYD_MODEL'].getboolean(
        'compare_ann_cyc_flag')
    use_obs_flow_flag = cfp['OPT_HYD_MODEL'].getboolean('use_obs_flow_flag')

    min_q_thresh = cfp['OPT_HYD_MODEL'].getfloat('min_q_thresh')
    run_as_lump_flag = cfp['OPT_HYD_MODEL'].getboolean('run_as_lump_flag')
    obj_ftn_wts = np.array(
        cfp['OPT_HYD_MODEL']['obj_ftn_wts'].split(sep), dtype=np.float64)

    in_opt_schm_vars_dict = cfp['OPT_SCHM_VARS']

    opt_schm_vars_dict = {}
    if in_opt_schm_vars_dict['opt_schm'] == 'DE':
        opt_schm_vars_dict['opt_schm'] = 'DE'
        opt_schm_vars_dict['mu_sc_fac_bds'] = np.array(
            in_opt_schm_vars_dict['mu_sc_fac_bds'].split(sep), dtype=np.float64)
        opt_schm_vars_dict['cr_cnst_bds'] = np.array(
            in_opt_schm_vars_dict['cr_cnst_bds'].split(sep), dtype=np.float64)

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

    else:
        raise NotImplementedError(
            'Incorrect opt_schm: %s' % in_opt_schm_vars_dict['opt_schm'])

    opt_schm_vars_dict['max_iters'] = in_opt_schm_vars_dict.getint('max_iters')
    opt_schm_vars_dict['max_cont_iters'] = in_opt_schm_vars_dict.getint(
        'max_cont_iters')
    opt_schm_vars_dict['obj_ftn_tol'] = in_opt_schm_vars_dict.getfloat(
        'obj_ftn_tol')
    opt_schm_vars_dict['prm_pcnt_tol'] = in_opt_schm_vars_dict.getfloat(
        'prm_pcnt_tol')
    opt_schm_vars_dict['n_prm_vecs_exp'] = in_opt_schm_vars_dict.getfloat(
        'n_prm_vecs_exp')

    bounds_dict = OrderedDict()
    bounds_dict['tt_bds'] = [float(_)
                             for _ in cfp['PARAM_BOUNDS']['tt'].split(sep)]
    bounds_dict['cm_bds'] = [float(_)
                             for _ in cfp['PARAM_BOUNDS']['cm'].split(sep)]
    bounds_dict['pcm_bds'] = [float(_)
                              for _ in cfp['PARAM_BOUNDS']['pcm'].split(sep)]
    bounds_dict['fc_bds'] = [float(_)
                             for _ in cfp['PARAM_BOUNDS']['fc_pwp'].split(sep)]
    bounds_dict['beta_bds'] = [float(_)
                               for _ in cfp['PARAM_BOUNDS']['beta'].split(sep)]
    bounds_dict['pwp_bds'] = [float(_)
                              for _ in cfp['PARAM_BOUNDS']['fc_pwp'].split(sep)]
    bounds_dict['ur_thr_bds'] = [
        float(_) for _ in cfp['PARAM_BOUNDS']['ur_thr'].split(sep)]
    bounds_dict['k_uu_bds'] = [float(_)
                               for _ in cfp['PARAM_BOUNDS']['k_uu'].split(sep)]
    bounds_dict['k_ul_bds'] = [float(_)
                               for _ in cfp['PARAM_BOUNDS']['k_ul'].split(sep)]
    bounds_dict['k_d_bds'] = [float(_)
                              for _ in cfp['PARAM_BOUNDS']['k_d'].split(sep)]
    bounds_dict['k_ll_bds'] = [float(_)
                               for _ in cfp['PARAM_BOUNDS']['k_ll'].split(sep)]
    bounds_dict['exp_bds'] = [float(_)
                              for _ in cfp['PARAM_BOUNDS']['exp'].split(sep)]
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
        in_q_df = pd.read_csv(in_q_file, sep=str(sep), index_col=0)
        in_q_df.index = pd.to_datetime(in_q_df.index, format=in_date_fmt)

        in_use_step_ser = pd.Series(
            index=in_q_df.index,
            data=np.ones(in_q_df.shape[0], dtype=np.int32))

#         valid_months = [1, 2, 3, 10, 11, 12]  # summer calib
# #         valid_months = [4, 5, 6, 7, 8, 9]  # winter calib
#         _bool_idxs = np.zeros(in_q_df.shape[0], dtype=bool)
#         for _month in valid_months:
#             _bool_idxs = _bool_idxs | (in_use_step_ser.index.month == _month)
#         in_use_step_ser.loc[_bool_idxs] = 0

        in_ppt_dfs_dict = load_pickle(in_ppt_file)
        in_temp_dfs_dict = load_pickle(in_temp_file)
        in_pet_dfs_dict = load_pickle(in_pet_file)
        in_cell_vars_dict = load_pickle(in_cell_vars_pkl)

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
            in_use_step_ser,
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
            in_hyd_mod_dir,
            bounds_dict,
            all_prms_flags,
            obj_ftn_wts,
            min_q_thresh,
            sep,
            kfolds,
            use_obs_flow_flag,
            run_as_lump_flag,
            opt_schm_vars_dict)

        _end_t = timeit.default_timer()
        _tot_t = _end_t - _beg_t
        print('\n\n')
        print('#' * 10)
        print(f'Total calibration time was: {_tot_t:0.4f} secs!')
        print('#' * 10)

    dbs_dir = os.path.join(in_hyd_mod_dir, r'01_database')
    hgs_db_path = os.path.join(in_hyd_mod_dir, r'02_hydrographs/hgs_dfs')

    #=========================================================================
    # Plot the k-fold results
    #=========================================================================

    if plot_kfold_perfs_flag:
        _beg_t = timeit.default_timer()

        print('\n\n')
        print('#' * 10)
        print('Plotting kfold results...')

        plot_kfold_effs(dbs_dir, hgs_db_path, compare_ann_cyc_flag, n_cpus)

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

        plot_kfolds_best_prms(dbs_dir, n_cpus)

        _end_t = timeit.default_timer()
        _tot_t = _end_t - _beg_t

        print(f'Took {_tot_t:0.4f} seconds!')
        print('#' * 10)

    #============================ ==============================================
    # Plot final parameter population
    #==========================================================================

    if plot_prm_vecs_flag:
        _beg_t = timeit.default_timer()

        print('\n\n')
        print('#' * 10)
        print('Plotting parameter vectors...')

        plot_prm_vecs(dbs_dir, n_cpus)

        _end_t = timeit.default_timer()
        _tot_t = _end_t - _beg_t

        print(f'Took {_tot_t:0.4f} seconds!')
        print('#' * 10)

    #==========================================================================
    # Plot hbv prms for all catchments per kfold in 2d
    #==========================================================================

    if plot_2d_kfold_prms_flag:
        _beg_t = timeit.default_timer()

        print('\n\n')
        print('#' * 10)
        print('Plotting HBV prms in 2D...')

        plot_kfolds_best_hbv_prms_2d(dbs_dir)

        _end_t = timeit.default_timer()
        _tot_t = _end_t - _beg_t

        print(f'Took {_tot_t:0.4f} seconds!')
        print('#' * 10)

    #==========================================================================
    # Plot annual cycle and FDC comparison
    #==========================================================================
    if plot_ann_cys_fdcs_flag:
        _beg_t = timeit.default_timer()

        print('\n\n')
        print('#' * 10)
        print('Plotting annual cycle and FDC comparison...')

        ann_cyc_fdc_plot_dir = os.path.join(
            in_hyd_mod_dir, r'08_ann_cycs_fdc_comparison')

        plot_ann_cycs_fdcs_comp(
            hgs_db_path, warm_up_steps, ann_cyc_fdc_plot_dir)

        _end_t = timeit.default_timer()
        _tot_t = _end_t - _beg_t

        print(f'Took {_tot_t:0.4f} seconds!')
        print('#' * 10)

    #==========================================================================
    # Plot catchment parameter transfer comparison
    #==========================================================================
    if plot_prm_trans_comp_flag:
        _beg_t = timeit.default_timer()

        print('\n\n')
        print('#' * 10)
        print('Plotting catchment parameter comparison...')

        plot_prm_trans_perfs(dbs_dir, n_cpus)

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

        plot_simple_opt_flag = cfp['PLOT_OPT_RES'].getboolean(
            'plot_simple_opt_flag')

        plot_dist_wat_bal_flag = cfp['PLOT_OPT_RES'].getboolean(
            'plot_wat_bal_flag')

        plot_vars(
            dbs_dir,
            water_bal_step_size,
            plot_simple_opt_flag,
            plot_dist_wat_bal_flag,
            n_cpus)

        _end_t = timeit.default_timer()
        _tot_t = _end_t - _beg_t

        print(f'Took {_tot_t:0.4f} seconds!')
        print('#' * 10)
    #==========================================================================
    return


if __name__ == '__main__':
    _save_log_ = False
    if _save_log_:
        from datetime import datetime
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
