"""
Created on %(date)s

@author: %(Faizan Anwar, IWS Uni-Stuttgart)s
"""

import os
import time
import timeit
import pickle
from shutil import copy2
import configparser as cfpm

import h5py
import numpy as np
import pandas as pd
from psutil import cpu_count

from hydmodeling import (
    solve_cats_sys_forcings,
    plot_cats_hbv_sim,)


def load_pickle(in_file, mode='rb'):
    with open(in_file, mode) as _pkl_hdl:
        return pickle.load(_pkl_hdl)
    return


def get_ref_sim_prms_dict(ref_sim_dir, kfolds):

    cat_dbs_dir = os.path.join(ref_sim_dir, r'01_database')

    cat_dbs = os.listdir(cat_dbs_dir)

    assert cat_dbs

    kf_prms_dict = {f'kf_{kf_i:02d}': {} for kf_i in range(1, kfolds + 1)}

    for cat_db in cat_dbs:
        with h5py.File(os.path.join(cat_dbs_dir, cat_db), 'r') as db:
            cat = db.attrs['cat']

            for kf_i in range(1, kfolds + 1):
                cd_db = db[f'calib/kf_{kf_i:02d}']

                kf_prms_dict[f'kf_{kf_i:02d}'][cat] = (
                    cd_db['hbv_prms'][...], cd_db['route_prms'][...])

    return kf_prms_dict


def main():
    ref_ini_file = r'config_hydmodeling_template.ini'

    ref_cfp = cfpm.ConfigParser(interpolation=cfpm.ExtendedInterpolation())
    ref_cfp.read(ref_ini_file)

    ref_ini_abs_path = os.path.abspath(ref_ini_file)

    dst_ini_file = r'config_forcings_template.ini'

    dst_cfp = cfpm.ConfigParser(interpolation=cfpm.ExtendedInterpolation())
    dst_cfp.read(dst_ini_file)

    dst_ini_abs_path = os.path.abspath(dst_ini_file)

    n_cpus = dst_cfp['DEFAULT']['n_cpus']
    if n_cpus == 'auto':
        n_cpus = cpu_count() - 1

    else:
        n_cpus = int(n_cpus)

    old_chdir = os.getcwd()

    run_sim_flag = False
    plot_hbv_vars_flag = False

#     run_sim_flag = True
    plot_hbv_vars_flag = True

    ref_sim_dir = os.path.abspath(ref_cfp['CREATE_STM_RELS']['hyd_mod_dir'])
    dst_sim_dir = os.path.abspath(dst_cfp['DST_SIM']['dst_sim_dir'])

    if not os.path.exists(dst_sim_dir):
        os.mkdir(dst_sim_dir)

    copy2(
        ref_ini_abs_path,
        os.path.join(dst_sim_dir, os.path.basename(ref_ini_abs_path)))

    copy2(
        dst_ini_abs_path,
        os.path.join(dst_sim_dir, os.path.basename(dst_ini_abs_path)))

    in_dem_net_file = ref_cfp['GET_STMS']['dem_net_file']
    in_cats_prcssed_file = ref_cfp['CREATE_STM_RELS']['cats_prcssed_file']
    in_stms_prcssed_file = ref_cfp['CREATE_STM_RELS']['stms_prcssed_file']

    # always in cumecs
#     ref_obs_q_file = ref_cfp['OPT_HYD_MODEL']['obs_q_file']
#     ref_ppt_file = ref_cfp['OPT_HYD_MODEL']['ppt_file']
#     ref_temp_file = ref_cfp['OPT_HYD_MODEL']['temp_file']
#     ref_pet_file = ref_cfp['OPT_HYD_MODEL']['pet_file']
    ref_cell_vars_file = ref_cfp['OPT_HYD_MODEL']['cell_vars_file']

    ref_sep = ref_cfp['DEFAULT']['sep']

    dst_obs_q_file = dst_cfp['DST_SIM']['obs_q_file']
    dst_ppt_file = dst_cfp['DST_SIM']['ppt_file']
    dst_temp_file = dst_cfp['DST_SIM']['temp_file']
    dst_pet_file = dst_cfp['DST_SIM']['pet_file']

    dst_sep = dst_cfp['DEFAULT']['sep']

    time_fmt = ref_cfp['OPT_HYD_MODEL']['time_fmt']

    start_date, end_date = dst_cfp['MISC_PRMS']['sim_dates'].split(dst_sep)

    start_date, end_date = pd.to_datetime(
        [start_date, end_date], format=time_fmt)

    run_times = [start_date, end_date]

    time_freq = ref_cfp['OPT_HYD_MODEL']['time_freq']

    warm_up_steps = dst_cfp['MISC_PRMS'].getint('warm_up_steps')
    water_bal_step_size = dst_cfp['MISC_PRMS'].getint('water_bal_step_size')

    route_type = ref_cfp['OPT_HYD_MODEL'].getint('route_type')
    kfolds = ref_cfp['OPT_HYD_MODEL'].getint('kfolds')

    use_obs_flow_flag = dst_cfp['MISC_PRMS'].getboolean('use_obs_flow_flag')

    min_q_thresh = ref_cfp['OPT_HYD_MODEL'].getfloat('min_q_thresh')
    run_as_lump_flag = ref_cfp['OPT_HYD_MODEL'].getboolean('run_as_lump_flag')

    if run_sim_flag:

        sim_prms_dict = get_ref_sim_prms_dict(ref_sim_dir, kfolds)

        in_cats_prcssed_df = pd.read_csv(
            in_cats_prcssed_file, sep=str(ref_sep), index_col=0)

        in_stms_prcssed_df = pd.read_csv(
            in_stms_prcssed_file, sep=str(ref_sep), index_col=0)

        in_dem_net_df = pd.read_csv(
            in_dem_net_file, sep=str(ref_sep), index_col=0)

        in_q_df = pd.read_csv(dst_obs_q_file, sep=str(dst_sep), index_col=0)
        in_q_df.index = pd.to_datetime(in_q_df.index, format=time_fmt)

        in_use_step_ser = pd.Series(
            index=in_q_df.index,
            data=np.ones(in_q_df.shape[0], dtype=np.int32))

        in_ppt_dfs_dict = load_pickle(dst_ppt_file)
        in_temp_dfs_dict = load_pickle(dst_temp_file)
        in_pet_dfs_dict = load_pickle(dst_pet_file)

        in_cell_vars_dict = load_pickle(ref_cell_vars_file)

        aux_cell_vars_dict = {}
        aux_cell_vars_dict['area_ratios'] = in_cell_vars_dict['area_ratios']
        aux_cell_vars_dict['shape'] = in_cell_vars_dict['shape']
        aux_cell_vars_dict['rows'] = in_cell_vars_dict['rows']
        aux_cell_vars_dict['cols'] = in_cell_vars_dict['cols']

        tt_flags = [int(_) for _ in ref_cfp['PRM_FLAGS']['tt'].split(ref_sep)]
        cm_flags = [int(_) for _ in ref_cfp['PRM_FLAGS']['cm'].split(ref_sep)]
        pcm_flags = [int(_) for _ in ref_cfp['PRM_FLAGS']['pcm'].split(ref_sep)]
        fc_flags = [int(_) for _ in ref_cfp['PRM_FLAGS']['fc_pwp'].split(ref_sep)]
        beta_flags = [int(_) for _ in ref_cfp['PRM_FLAGS']['beta'].split(ref_sep)]
        pwp_flags = [int(_) for _ in ref_cfp['PRM_FLAGS']['fc_pwp'].split(ref_sep)]
        ur_thr_flags = [int(_) for _ in ref_cfp['PRM_FLAGS']['ur_thr'].split(ref_sep)]
        k_uu_flags = [int(_) for _ in ref_cfp['PRM_FLAGS']['k_uu'].split(ref_sep)]
        k_ul_flags = [int(_) for _ in ref_cfp['PRM_FLAGS']['k_ul'].split(ref_sep)]
        k_d_flags = [int(_) for _ in ref_cfp['PRM_FLAGS']['k_d'].split(ref_sep)]
        k_ll_flags = [int(_) for _ in ref_cfp['PRM_FLAGS']['k_ll'].split(ref_sep)]

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

        solve_cats_sys_forcings(
            in_cats_prcssed_df,
            in_stms_prcssed_df,
            in_dem_net_df,
            in_use_step_ser,
            in_q_df,
            in_ppt_dfs_dict,
            in_temp_dfs_dict,
            in_pet_dfs_dict,
            aux_cell_vars_dict,
            time_freq,
            warm_up_steps,
            n_cpus,
            route_type,
            dst_sim_dir,
            all_prms_flags,
            min_q_thresh,
            dst_sep,
            kfolds,
            use_obs_flow_flag,
            run_as_lump_flag,
            run_times,
            sim_prms_dict,
            )

        _end_t = timeit.default_timer()
        _tot_t = _end_t - _beg_t

        print('\n\n')
        print('#' * 10)
        print(f'Total calibration time was: {_tot_t:0.4f} secs!')
        print('#' * 10)

        del in_ppt_dfs_dict
        del in_temp_dfs_dict
        del in_pet_dfs_dict

    dbs_dir = os.path.join(dst_sim_dir, r'01_database')

    if plot_hbv_vars_flag:
        print('\n\n')
        print('#' * 10)
        print('Plotting hbv variables...')

        _beg_t = timeit.default_timer()

        plot_full_sim_flag = dst_cfp['MISC_PRMS'].getboolean(
            'plot_full_sim_flag')

        plot_wat_bal_flag = dst_cfp['MISC_PRMS'].getboolean(
            'plot_wat_bal_flag')

        show_warm_up_steps_flag = dst_cfp['MISC_PRMS'].getboolean(
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

    os.chdir(old_chdir)
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
