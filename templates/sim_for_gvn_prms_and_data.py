'''
@author: Faizan-Uni-Stuttgart

31 Jan 2020

13:37:29

'''
import os
import time
import timeit
from pathlib import Path

import h5py
import pandas as pd
import matplotlib.pyplot as plt

plt.ioff()

DEBUG_FLAG = False

from hydmodeling.models import (
    hbv_loop_py,
    tfm_opt_to_hbv_prms_py)


def get_sim_ress(kf_dict, area_arr, conv_ratio):

    temp_dist_arr = kf_dict['tem_arr']
    prec_dist_arr = kf_dict['ppt_arr']
    pet_dist_arr = kf_dict['pet_arr']
    prms_dist_arr = kf_dict['hbv_prms']

    all_outputs_dict = hbv_loop_py(
        temp_dist_arr,
        prec_dist_arr,
        pet_dist_arr,
        prms_dist_arr,
        kf_dict['ini_arr'],
        area_arr,
        conv_ratio,
        1)

    assert all_outputs_dict['loop_ret'] == 0.0

    q_sim_arr = all_outputs_dict['qsim_arr']

    extra_us_inflow_flag = 'extra_us_inflow' in kf_dict

    q_sim_arr = (q_sim_arr).copy(order='C')
    if extra_us_inflow_flag:
        extra_us_inflow = kf_dict['extra_us_inflow']
        q_sim_arr = q_sim_arr + extra_us_inflow

    return kf_dict, all_outputs_dict, q_sim_arr, kf_dict['qact_arr']


def run_sim(args):

    prm_cat_db, data_cat_db, data_ds = args

    # right now for one cat.
    # with slight changes can be use for various cats
    with h5py.File(prm_cat_db, 'r') as db:
        kfolds = db['data'].attrs['kfolds']
        assert kfolds == 1

        prm_cat = db.attrs['cat']

        f_var_infos = db['cdata/aux_var_infos'][...]
        prms_idxs = db['cdata/use_prms_idxs'][...]
        f_vars = db['cdata/aux_vars'][...]
        prms_flags = db['cdata/all_prms_flags'][...]
        bds_arr = db['cdata/bds_arr'][...]

        cat_vars_dict = {}
        cat_vars_dict['f_var_infos'] = f_var_infos
        cat_vars_dict['prms_idxs'] = prms_idxs
        cat_vars_dict['f_vars'] = f_vars
        cat_vars_dict['prms_flags'] = prms_flags
        cat_vars_dict['bds_arr'] = bds_arr

        kf_prms_dict = {}
        for i in range(1, kfolds + 1):
            kf_str = f'kf_{i:02d}'
            cd_db = db[f'calib/{kf_str}']

            opt_prms = cd_db['opt_prms'][...]

            if i not in kf_prms_dict:
                kf_prms_dict[i] = {}

            kf_prms_dict[i][prm_cat] = {}

            kf_prms_dict[i][prm_cat]['opt_prms'] = opt_prms

    with h5py.File(data_cat_db, 'r') as db:

        kfolds = db['data'].attrs['kfolds']

        assert kfolds == 1

        data_cat = db.attrs['cat']

        conv_ratio = db['data'].attrs['conv_ratio']
        use_obs_flow_flag = db['data'].attrs['use_obs_flow_flag']
        area_arr = db['data/area_arr'][...]
        n_cells = area_arr.shape[0]

        all_kfs_dict = {}
        for i in range(1, kfolds + 1):
            cd_db = db[f'{data_ds}/kf_{i:02d}']
            kf_dict = {key: cd_db[key][...] for key in cd_db}
            kf_dict['use_obs_flow_flag'] = use_obs_flow_flag

            all_kfs_dict[i] = kf_dict

    # 1 kf and 1 cat
    for i in range(1, kfolds + 1):
        kf_dict = all_kfs_dict[i]

        trans_cat_dict = kf_prms_dict[i]

        for trans_cat in trans_cat_dict:
            trans_opt_prms = trans_cat_dict[trans_cat]

            kf_dict['hbv_prms'] = tfm_opt_to_hbv_prms_py(
                cat_vars_dict['prms_flags'],
                cat_vars_dict['f_var_infos'],
                cat_vars_dict['prms_idxs'],
                cat_vars_dict['f_vars'],
                trans_opt_prms['opt_prms'],
                cat_vars_dict['bds_arr'],
                n_cells)

            ress = (
                prm_cat,
                data_cat,
                get_sim_ress(kf_dict, area_arr, conv_ratio))

            break

        break

    return ress


def main():

    main_dir = Path(r'P:\Synchronize\IWS\QGIS_Neckar\ns_peaks_calib_valid')
    os.chdir(main_dir)

    cats = [420, 3421, 3465, 3470]

    data_ds = 'valid'  # read calibration/validation data

    prms_dir = Path(r'calib_ns__5_peaks\01_database')

    data_dir = Path(r'calib_ns__5_peaks\01_database')

    out_dir = Path('valid_ns__5_peaks')

    if not out_dir.exists():
        out_dir.mkdir(exist_ok=True)

    for cat in cats:
        in_prms_h5 = prms_dir / f'cat_{cat}.hdf5'

        in_data_h5 = data_dir / f'cat_{cat}.hdf5'

        assert in_prms_h5.exists()

        assert in_data_h5.exists()

        prm_cat, data_cat, ress = run_sim((in_prms_h5, in_data_h5, data_ds))

        out_df = pd.DataFrame(data={'ref': ress[3], 'sim': ress[2]})

        out_df.to_csv(
            out_dir / f'{prm_cat}p_{data_cat}d.csv',
            sep=';',
            float_format='%0.4f')

    return


if __name__ == '__main__':

    _save_log_ = False
    if _save_log_:
        from datetime import datetime
        from std_logger import StdFileLoggerCtrl

        # save all console activity to out_log_file
        out_log_file = os.path.join(
            r'P:\Synchronize\python_script_logs\\%s_log_%s.log' % (
            os.path.basename(__file__),
            datetime.now().strftime('%Y%m%d%H%M%S')))

        log_link = StdFileLoggerCtrl(out_log_file)

    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack
    # 2. "up" move the stack up to an older frame
    # 3. "down" move the stack down to a newer frame
    # 4. "interact" start an interactive interpreter
    #==========================================================================

    if DEBUG_FLAG:
        try:
            main()

        except:
            import pdb
            pdb.post_mortem()

    else:
        main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
