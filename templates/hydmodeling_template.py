"""
Created on %(date)s

@author: %(Faizan Anwar, IWS Uni-Stuttgart)s
"""

import os
import sys
import time
import timeit
import pickle
import configparser as cfpm
from collections import OrderedDict

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from hydmodeling import (
    solve_cats_sys,
    plot_vars,
    plot_pops,
    plot_kfold_effs,
    plot_kfolds_best_prms,
    plot_kfolds_best_hbv_prms_2d)


def load_pickle(in_file, mode='rb'):
    with open(in_file, mode) as _pkl_hdl:
        return pickle.load(_pkl_hdl)
    return


def main():
    cfp = cfpm.ConfigParser(interpolation=cfpm.ExtendedInterpolation())
    cfp.read('config_hydmodeling_template.ini')

    n_cpus = cfp['DEFAULT'].getint('n_cpus')

    main_dir = cfp['DEFAULT']['main_dir']
    os.chdir(main_dir)

    optimize_flag = False
    plot_opt_results_flag = False
    plot_kfold_results_flag = False
    plot_kfold_prms_flag = False
    plot_pop_flag = False
    plot_2d_prms = False
    test_model_flag = False

#     optimize_flag = True
    plot_opt_results_flag = True
    plot_kfold_results_flag = True
    plot_kfold_prms_flag = True
    plot_pop_flag = True
    plot_2d_prms = True
#     test_model_flag = True

    #==========================================================================
    # Optimize distributed model
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
    compare_ann_cyc_flag = cfp['OPT_HYD_MODEL'].getboolean('compare_ann_cyc_flag')
    use_obs_flow_flag = cfp['OPT_HYD_MODEL'].getboolean('use_obs_flow_flag')

    min_q_thresh = cfp['OPT_HYD_MODEL'].getfloat('min_q_thresh')
    run_as_lump_flag = cfp['OPT_HYD_MODEL'].getboolean('run_as_lump_flag')
    obj_ftn_wts = np.array(
        cfp['OPT_HYD_MODEL']['obj_ftn_wts'].split(','), dtype=np.float64)

    in_opt_schm_vars_dict = cfp['OPT_SCHM_VARS']
    opt_schm_vars_dict = {}
    if in_opt_schm_vars_dict['opt_schm'] == 'DE':
        opt_schm_vars_dict['opt_schm'] = 'DE'
        opt_schm_vars_dict['mu_sc_fac_bds'] = np.array(
            in_opt_schm_vars_dict['mu_sc_fac_bds'].split(','), dtype=np.float64)
        opt_schm_vars_dict['cr_cnst_bds'] = np.array(
            in_opt_schm_vars_dict['cr_cnst_bds'].split(','), dtype=np.float64)
        opt_schm_vars_dict['pop_size_exp'] = in_opt_schm_vars_dict.getfloat('pop_size_exp')
    elif in_opt_schm_vars_dict['opt_schm'] == 'ROPE':
        opt_schm_vars_dict['opt_schm'] = 'ROPE'
        opt_schm_vars_dict['n_par_sets'] = cfp['OPT_HYD_MODEL'].getint('n_par_sets')
        opt_schm_vars_dict['n_final_sets'] = cfp['OPT_HYD_MODEL'].getint('n_final_sets')
        opt_schm_vars_dict['n_new_par'] = cfp['OPT_HYD_MODEL'].getint('n_new_par')
    else:
        raise NotImplementedError(
            'Incorrect opt_schm: %s' % in_opt_schm_vars_dict['opt_schm'])

    opt_schm_vars_dict['max_iters'] = in_opt_schm_vars_dict.getint('max_iters')
    opt_schm_vars_dict['max_cont_iters'] = in_opt_schm_vars_dict.getint('max_cont_iters')
    opt_schm_vars_dict['obj_ftn_tol'] = in_opt_schm_vars_dict.getfloat('obj_ftn_tol')
    opt_schm_vars_dict['prm_pcnt_tol'] = in_opt_schm_vars_dict.getfloat('prm_pcnt_tol')

    bounds_dict = OrderedDict()
    if test_model_flag:
        lump_prms_df = pd.read_csv(cfp['TEST_MODEL']['test_params_loc'], index_col=0, sep=sep)

        bounds_dict['tt_bds'] = [float(lump_prms_df.loc['TT', 'value'])] * 2
        bounds_dict['cm_bds'] = [float(lump_prms_df.loc['cm', 'value'])] * 2
        bounds_dict['pcm_bds'] = [float(lump_prms_df.loc['p_cm', 'value'])] * 2
        bounds_dict['fc_bds'] = [float(lump_prms_df.loc['FC', 'value'])] * 2
        bounds_dict['beta_bds'] = [float(lump_prms_df.loc['Beta', 'value'])] * 2
        bounds_dict['pwp_bds'] = [float(lump_prms_df.loc['PWP', 'value'])] * 2
        bounds_dict['ur_thr_bds'] = [float(lump_prms_df.loc['ur_thresh', 'value'])] * 2
        bounds_dict['k_uu_bds'] = [float(lump_prms_df.loc['K_u', 'value'])] * 2
        bounds_dict['k_ul_bds'] = [float(lump_prms_df.loc['K_l', 'value'])] * 2
        bounds_dict['k_d_bds'] = [float(lump_prms_df.loc['K_d', 'value'])] * 2
        bounds_dict['k_ll_bds'] = [float(lump_prms_df.loc['K_ll', 'value'])] * 2

    else:
        bounds_dict['tt_bds'] = [float(_) for _ in cfp['PARAM_BOUNDS']['tt'].split(',')]
        bounds_dict['cm_bds'] = [float(_) for _ in cfp['PARAM_BOUNDS']['cm'].split(',')]
        bounds_dict['pcm_bds'] = [float(_) for _ in cfp['PARAM_BOUNDS']['pcm'].split(',')]
        bounds_dict['fc_bds'] = [float(_) for _ in cfp['PARAM_BOUNDS']['fc'].split(',')]
        bounds_dict['beta_bds'] = [float(_) for _ in cfp['PARAM_BOUNDS']['beta'].split(',')]
        bounds_dict['pwp_bds'] = [float(_) for _ in cfp['PARAM_BOUNDS']['pwp'].split(',')]
        bounds_dict['ur_thr_bds'] = [float(_) for _ in cfp['PARAM_BOUNDS']['ur_thr'].split(',')]
        bounds_dict['k_uu_bds'] = [float(_) for _ in cfp['PARAM_BOUNDS']['k_uu'].split(',')]
        bounds_dict['k_ul_bds'] = [float(_) for _ in cfp['PARAM_BOUNDS']['k_ul'].split(',')]
        bounds_dict['k_d_bds'] = [float(_) for _ in cfp['PARAM_BOUNDS']['k_d'].split(',')]
        bounds_dict['k_ll_bds'] = [float(_) for _ in cfp['PARAM_BOUNDS']['k_ll'].split(',')]
        bounds_dict['exp_bds'] = [float(_) for _ in cfp['PARAM_BOUNDS']['exp'].split(',')]
        bounds_dict['musk_lag_bds'] = [float(_) for _ in cfp['PARAM_BOUNDS']['musk_lag'].split(',')]
        bounds_dict['musk_wt_bds'] = [float(_) for _ in cfp['PARAM_BOUNDS']['musk_wt'].split(',')]

    tt_flags = [int(_) for _ in cfp['PRM_FLAGS']['tt'].split(',')]
    cm_flags = [int(_) for _ in cfp['PRM_FLAGS']['cm'].split(',')]
    pcm_flags = [int(_) for _ in cfp['PRM_FLAGS']['pcm'].split(',')]
    fc_flags = [int(_) for _ in cfp['PRM_FLAGS']['fc'].split(',')]
    beta_flags = [int(_) for _ in cfp['PRM_FLAGS']['beta'].split(',')]
    pwp_flags = [int(_) for _ in cfp['PRM_FLAGS']['pwp'].split(',')]
    ur_thr_flags = [int(_) for _ in cfp['PRM_FLAGS']['ur_thr'].split(',')]
    k_uu_flags = [int(_) for _ in cfp['PRM_FLAGS']['k_uu'].split(',')]
    k_ul_flags = [int(_) for _ in cfp['PRM_FLAGS']['k_ul'].split(',')]
    k_d_flags = [int(_) for _ in cfp['PRM_FLAGS']['k_d'].split(',')]
    k_ll_flags = [int(_) for _ in cfp['PRM_FLAGS']['k_ll'].split(',')]

    all_prms_flags = np.array([tt_flags,
                               cm_flags,
                               pcm_flags,
                               fc_flags,
                               beta_flags,
                               pwp_flags,
                               ur_thr_flags,
                               k_uu_flags,
                               k_ul_flags,
                               k_d_flags,
                               k_ll_flags], dtype=np.int32)

    if optimize_flag:
        in_cats_prcssed_df = pd.read_csv(in_cats_prcssed_file,
                                         sep=str(sep),
                                         index_col=0)
        in_stms_prcssed_df = pd.read_csv(in_stms_prcssed_file,
                                         sep=str(sep),
                                         index_col=0)
        in_dem_net_df = pd.read_csv(in_dem_net_file, sep=str(sep), index_col=0)
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
            aux_cell_vars_dict['lulc_ratios'] = in_cell_vars_dict['lulc_ratios']
        if np.any(all_prms_flags[:, 2]):
            aux_cell_vars_dict['soil_ratios'] = in_cell_vars_dict['soil_ratios']
        if np.any(all_prms_flags[:, 3]) or np.any(all_prms_flags[:, 5]):
            aux_cell_vars_dict['aspect'] = in_cell_vars_dict['aspect'].reshape(-1, 1)
        if np.any(all_prms_flags[:, 4]) or np.any(all_prms_flags[:, 5]):
            aux_cell_vars_dict['slope'] = in_cell_vars_dict['slope'].reshape(-1, 1)

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

    dbs_dir = os.path.join(in_hyd_mod_dir, r'01_database')
    #=========================================================================
    # plot the optimization results
    #=========================================================================
    if plot_opt_results_flag:
        print('\n\nPlotting hbv variables...')

        plot_simple_opt_flag = cfp['PLOT_OPT_RES'].getboolean('plot_simple_opt_flag')
        plot_dist_wat_bal_flag = cfp['PLOT_OPT_RES'].getboolean('plot_wat_bal_flag')

        plot_vars(
            dbs_dir,
            water_bal_step_size,
            plot_simple_opt_flag,
            plot_dist_wat_bal_flag,
            n_cpus)

    #=========================================================================
    # Plot the k-fold results
    #=========================================================================

    if plot_kfold_results_flag:
        hgs_db_path = os.path.join(in_hyd_mod_dir, r'02_hydrographs/hgs_dfs')
        print('\n\nPlotting kfold results...')
        plot_kfold_effs(dbs_dir, hgs_db_path, compare_ann_cyc_flag, n_cpus)

    #=========================================================================
    # Plot the best k-fold params
    #=========================================================================

    if plot_kfold_prms_flag:
        print('\n\nPlotting best kfold prms...')
        plot_kfolds_best_prms(dbs_dir)

    #==========================================================================
    # Plot hbv prms for all cathcments per kfold in 2d
    #==========================================================================

    if plot_2d_prms:
        print('\n\nPlotting HBV prms in 2D...')
        plot_kfolds_best_hbv_prms_2d(dbs_dir)

    #============================ ==============================================
    # Plot parameter final population
    #==========================================================================

    if plot_pop_flag:
        print('\n\nPlotting DE population...')
        plot_pops(dbs_dir, n_cpus)
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
