# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Faizan Anwar, IWS Uni-Stuttgart)s
"""

import os
import time
import timeit
import pickle
from glob import iglob, glob
import configparser as cfpm
from collections import OrderedDict

import numpy as np
import pandas as pd

from hydmodeling import (
    solve_cats_sys,
    plot_vars,
    plot_pops,
    plot_kfold_effs)


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

    optimize = False
    plot_opt_results = False
    plot_kfold_results = False
    test_model_flag = False
    plot_pop = False

    optimize = True
#     plot_opt_results = True
    plot_kfold_results = True
#     test_model_flag = True
    plot_pop = True

    #==========================================================================
    # Optimize distributed model
    #==========================================================================
    in_hyd_mod_dir = cfp['OPT_HYD_MODEL']['in_hyd_mod_dir']

    in_dem_net_file = cfp['OPT_HYD_MODEL']['in_dem_net_file']
    in_cats_prcssed_file = cfp['OPT_HYD_MODEL']['in_cats_prcssed_file']
    in_stms_prcssed_file = cfp['OPT_HYD_MODEL']['in_stms_prcssed_file']

    # always in cumecs
    in_q_file = cfp['OPT_HYD_MODEL']['in_q_file']
    in_ppt_file = cfp['OPT_HYD_MODEL']['in_ppt_file']
    in_temp_file = cfp['OPT_HYD_MODEL']['in_temp_file']
    in_pet_file = cfp['OPT_HYD_MODEL']['in_pet_file']
    in_cell_vars_pkl = cfp['OPT_HYD_MODEL']['in_cell_idxs_pkl']

    opt_res_pkl_path = cfp['OPT_HYD_MODEL']['opt_res_pkl_path']
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

    if optimize:
        in_cats_prcssed_df = pd.read_csv(in_cats_prcssed_file,
                                         sep=str(sep),
                                         index_col=0)
        in_stms_prcssed_df = pd.read_csv(in_stms_prcssed_file,
                                         sep=str(sep),
                                         index_col=0)
        in_dem_net_df = pd.read_csv(in_dem_net_file, sep=str(sep), index_col=0)
        in_q_df = pd.read_csv(in_q_file, sep=str(sep), index_col=0)
        in_q_df.index = pd.to_datetime(in_q_df.index, format=in_date_fmt)

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
            aux_cell_vars_dict['aspect'] = in_cell_vars_dict['aspect']
        if np.any(all_prms_flags[:, 4]) or np.any(all_prms_flags[:, 5]):
            aux_cell_vars_dict['slope'] = in_cell_vars_dict['slope']

        solve_cats_sys(
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
            in_hyd_mod_dir,
            bounds_dict,
            all_prms_flags,
            obj_ftn_wts,
            water_bal_step_size,
            min_q_thresh,
            sep,
            opt_res_pkl_path,
            kfolds,
            use_obs_flow_flag,
            run_as_lump_flag,
            opt_schm_vars_dict)

    #=========================================================================
    # plot the optimization results
    #=========================================================================
    if plot_opt_results:
        plot_simple_opt_flag = cfp['PLOT_OPT_RES'].getboolean('plot_simple_opt_flag')
        plot_dist_wat_bal_flag = cfp['PLOT_OPT_RES'].getboolean('plot_dist_wat_bal_flag')
        _ext = opt_res_pkl_path.rsplit('.', 1)[-1]
        _dir = os.path.dirname(opt_res_pkl_path)

        for _pkl_path in iglob(os.path.join(_dir,
                                            '*__calib_kfold_*.%s' % _ext)):
            plot_vars(
                _pkl_path,
                n_cpus,
                plot_simple_opt_flag,
                plot_dist_wat_bal_flag)

    #=========================================================================
    # Plot the k-fold results
    #=========================================================================

    if plot_kfold_results:
        _ext = opt_res_pkl_path.rsplit('.', 1)[1]
        _dir = os.path.dirname(opt_res_pkl_path)

        _1 = 'opt_results__valid_kfold*.%s' % _ext
        kfold_opt_res_paths = glob(os.path.join(_dir, _1))

        assert kfold_opt_res_paths, 'kfold_opt_res_paths is empty!'

        plot_kfold_effs(kfold_opt_res_paths,
                        compare_ann_cyc_flag,
                        n_cpus)

    #==========================================================================
    # Plot parameter final population
    #==========================================================================

    if plot_pop:
        _ext = opt_res_pkl_path.rsplit('.', 1)[-1]
        _dir = os.path.dirname(opt_res_pkl_path)

        for _pkl_path in iglob(os.path.join(_dir,
                                            '*__calib_kfold_*.%s' % _ext)):
            plot_pops(_pkl_path, n_cpus)
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
