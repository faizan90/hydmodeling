'''
@author: Faizan-Uni-Stuttgart

Jul 13, 2022

9:19:41 AM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  ; plt.ion()

from hydmodeling.models import (
    hbv_loop_py,
#     hbv_c_loop_py,
    get_ns_cy,
    tfm_opt_to_hbv_prms_py)

DEBUG_FLAG = False


def main():

    main_dir = Path(r'P:\Synchronize\IWS\QGIS_Neckar\ref_DE_dist')
    os.chdir(main_dir)

    # cats = [3421, 3465, 3470]
    cats = [3465]

    data_ds = 'calib'  # read calibration/validation data

    prms_dir = Path(r'01_database')

    data_dir = Path(r'01_database')

    out_dir = Path(fr'test_wat_bal')
    #==========================================================================

    out_dir.mkdir(exist_ok=True)

    for cat in cats:
        in_prms_h5 = prms_dir / f'cat_{cat}.hdf5'

        in_data_h5 = data_dir / f'cat_{cat}.hdf5'

        assert in_prms_h5.exists()

        assert in_data_h5.exists()

        prm_cat, data_cat, ress, off_idx = run_sim(
            (in_prms_h5, in_data_h5, data_ds))

        out_df = pd.DataFrame(data={'ref': ress[3], 'sim': ress[2]})

        ns = get_ns_cy(out_df['ref'].values, out_df['sim'].values, off_idx)

        print(f'Cat {cat}\'s NS: {ns:0.3f}')

        out_df.to_csv(
            out_dir / f'{prm_cat}p_{data_cat}d.csv',
            sep=';',
            float_format='%0.4f')

    return


def get_sim_ress(kf_dict, area_arr, conv_ratio):

    show_wat_bals_flag = True
    plot_flag = True

    # show_wat_bals_flag = False
    # plot_flag = False

    temp_dist_arr = kf_dict['tem_arr']
    prec_dist_arr = kf_dict['ppt_arr']
    pet_dist_arr = kf_dict['pet_arr']
    prms_dist_arr = kf_dict['hbv_prms']

    # No snow.
    # prms_dist_arr[:, 0] = -1000
    # prms_dist_arr[:, 1] = 0
    # prms_dist_arr[:, 2] = 0

    # No somo.
    # prms_dist_arr[:, 4] = 0

    # No upper reservoir upper outlet.
    # prms_dist_arr[:, 6] = 1e9
    # prms_dist_arr[:, 7] = 0

    # No upper reservoir lower outlet.
    # prms_dist_arr[:, 8] = 0

    # No groundwater.
    # prms_dist_arr[:, 9] = 0
    # prms_dist_arr[:, 10] = 0

    all_outputs_dict = hbv_loop_py(
#     all_outputs_dict = hbv_c_loop_py(
        temp_dist_arr,
        prec_dist_arr,
        pet_dist_arr,
        prms_dist_arr,
        np.array(kf_dict['ini_arr']),
        area_arr,
        conv_ratio,
        int(not show_wat_bals_flag))

    assert all_outputs_dict['loop_ret'] == 0.0

    q_sim_arr = all_outputs_dict['qsim_arr']

    #==========================================================================s
    if show_wat_bals_flag:
        # For checking water balances.
        # Change the opt_flag in hbv_loop_py to 0.

        wb_lhss = []
        wb_rhss = []

        axes = plt.subplots(1, 2, squeeze=False)[1]
        plt.show(block=False)

        # Go through each cell for all time steps.
        for i in range(all_outputs_dict['outs_arr'].shape[0]):
            outs_arr = all_outputs_dict['outs_arr'][i,:,:]

            snow_arr = outs_arr[:, 0]
            sm_arr = outs_arr[:, 2]
            evap_arr = outs_arr[:, 4]
            ur_sto_arr = outs_arr[:, 5]
            ur_run_uu = outs_arr[:, 6]
            ur_run_ul = outs_arr[:, 7]
            lr_sto_arr = outs_arr[:, 9]
            lr_run_arr = lr_sto_arr * prms_dist_arr[i, 10]

            storage = snow_arr + sm_arr + ur_sto_arr + lr_sto_arr

            runoff = (ur_run_uu + ur_run_ul)[1:]
            runoff += lr_run_arr[:-1]

            curr_sto = storage[+1:]
            prev_sto = storage[:-1]

            wb_rhs = -prev_sto + curr_sto + evap_arr[1:] + runoff

            wb_lhs = prec_dist_arr[i, 1:]

            wb_lhss.append(wb_lhs)
            wb_rhss.append(wb_rhs)

            rel_abs_diff = np.abs(wb_rhs - wb_lhs).sum() / wb_rhs.sum()

            # if rel_abs_diff >= 1e-5:
            #     print(
            #         f'Relative water balance difference not zero '
            #         f'({rel_abs_diff:0.3f}) at cell: {i}!')

            if plot_flag:

                min_xy = min([wb_rhs.min(), wb_lhs.min()])
                max_xy = max([wb_rhs.max(), wb_lhs.max()])

                axes[0, 0].scatter(wb_rhs, wb_lhs, alpha=0.7)

                axes[0, 0].plot(
                    [min_xy, max_xy],
                    [min_xy, max_xy],
                    alpha=0.25,
                    c='k',
                    ls='--')

                axes[0, 0].grid()

                axes[0, 1].plot(wb_rhs - wb_lhs, alpha=0.7)

                axes[0, 1].grid()

                plt.title(f'{i}: {rel_abs_diff:0.3%}')

                plt.draw()
                plt.pause(0.01)  # This is important.

                axes[0, 0].cla()
                axes[0, 1].cla()

        plt.close()

        # wb_lhss = np.sort(np.array(wb_lhss).ravel())
        # wb_rhss = np.sort(np.array(wb_rhss).ravel())

        wb_lhss = np.array(wb_lhss)
        wb_rhss = np.array(wb_rhss)

        wb_lhss_sum, wb_rhss_sum = wb_lhss.sum(), wb_rhss.sum()

        print('Input output sum:', wb_lhss_sum, wb_rhss_sum)

        print(
            f'Overall input by output ratio: {wb_lhss_sum / wb_rhss_sum:0.5f}')

        # assert np.isclose(abs_diff, 0.0), (
        #     f'Water balance difference too high ({abs_diff}) at cell: {i}!')
    #==========================================================================

    if 'extra_us_inflow' in kf_dict:
        q_sim_arr = q_sim_arr + kf_dict['extra_us_inflow']

    return kf_dict, all_outputs_dict, q_sim_arr, kf_dict['qact_arr']


def run_sim(args):

    prm_cat_db, data_cat_db, data_ds = args

    # right now for one cat.
    # with slight changes can be use for various cats
    with h5py.File(prm_cat_db, 'r') as db:
        kfolds = 1  # db['data'].attrs['kfolds']
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

        kfolds = 1  # db['data'].attrs['kfolds']

        assert kfolds == 1

        data_cat = db.attrs['cat']

        conv_ratio = db['data'].attrs['conv_ratio']
        use_obs_flow_flag = db['data'].attrs['use_obs_flow_flag']
        area_arr = db['data/area_arr'][...]
        n_cells = area_arr.shape[0]

        off_idx = db['data'].attrs['off_idx']

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
                get_sim_ress(kf_dict, area_arr, conv_ratio),
                off_idx)

            break

        break

    return ress


if __name__ == '__main__':
    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack,
    # 2. "up" move the stack up to an older frame,
    # 3. "down" move the stack down to a newer frame, and
    # 4. "interact" start an interactive interpreter.
    #==========================================================================

    if DEBUG_FLAG:
        try:
            main()

        except:
            pre_stack = tb.format_stack()[:-1]

            err_tb = list(tb.TracebackException(*sys.exc_info()).format())

            lines = [err_tb[0]] + pre_stack + err_tb[2:]

            for line in lines:
                print(line, file=sys.stderr, end='')

            import pdb
            pdb.post_mortem()
    else:
        main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))
