'''
@author: Faizan-Uni-Stuttgart

'''

import os
import timeit
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hydmodeling import _plot_hbv_kf


def main():
    main_dir = Path(r'P:\Synchronize\IWS\2015_Water_Balance_Peru\02_data\01_santa\quillcay_sims\ET_Santa')
    os.chdir(main_dir)

    sep = ';'
    date_fmt = '%Y-%m-%d'

    beg_time = '2016-02-20'
    end_time = '2017-11-26'

    cat = '5000'

    ppt_df = pd.read_csv('em_10_ppt.csv', sep=sep, index_col=0)
    ppt_df.index = pd.to_datetime(ppt_df.index, format=date_fmt)
    ppt_df = ppt_df.loc[beg_time:end_time, cat]

    tem_df = pd.read_csv('em_10_temp.csv', sep=sep, index_col=0)
    tem_df.index = pd.to_datetime(tem_df.index, format=date_fmt)
    tem_df = tem_df.loc[beg_time:end_time, cat]

    pet_df = pd.read_csv('em_10_pet.csv', sep=sep, index_col=0)
    pet_df.index = pd.to_datetime(pet_df.index, format=date_fmt)
    pet_df = pet_df.loc[beg_time:end_time, cat]

    qact_df = pd.read_csv('monthly_q_bulletins.csv', sep=sep, index_col=0)
    qact_df.index = pd.to_datetime(qact_df.index, format=date_fmt)

    hbv_prms_df = pd.read_csv('HBV_model_params_5000_orig.csv', sep=sep, index_col=0)

    qact_df = qact_df.resample('D')
    qact_df = qact_df.mean().interpolate(limit_direction='both')

    qact_df = qact_df.reindex(pet_df.index)
    qact_df = qact_df.loc[beg_time:end_time, cat]
    qact_df = qact_df.interpolate(limit_direction='both')

    assert not np.any(np.isnan(ppt_df.values))
    assert not np.any(np.isnan(tem_df.values))
    assert not np.any(np.isnan(pet_df.values))
    assert not np.any(np.isnan(qact_df.values))

    kf_i = 1
    kf_dict = {}
    area_arr = np.array([1.0])
    conv_ratio = 246580000.0 / (1000. * 86400)
    prm_syms = [
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
    off_idx = 170
    out_dir = os.getcwd()
    wat_bal_stps = 90
    plot_simple_flag = True
    plot_wat_bal_flag = True

    kf_dict['ppt_arr'] = ppt_df.values.reshape(1, -1).copy(order='c')
    kf_dict['tem_arr'] = tem_df.values.reshape(1, -1).copy(order='c')
    kf_dict['pet_arr'] = pet_df.values.reshape(1, -1).copy(order='c')
    kf_dict['qact_arr'] = qact_df.values.copy(order='c')
    kf_dict['hbv_prms'] = hbv_prms_df.loc[prm_syms].values.astype(float).reshape(1, -1).copy(order='c')
    kf_dict['ini_arr'] = np.zeros((1, 4))

    _plot_hbv_kf(
            kf_i,
            cat,
            kf_dict,
            area_arr,
            conv_ratio,
            prm_syms,
            off_idx,
            out_dir,
            wat_bal_stps,
            plot_simple_flag,
            plot_wat_bal_flag)

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
                                    ('xxx_log_%s.log' %
                                     datetime.now().strftime('%Y%m%d%H%M%S')))
        log_link = StdFileLoggerCtrl(out_log_file)

    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main()

    STOP = timeit.default_timer()  # Ending time
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
