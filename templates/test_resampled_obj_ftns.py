'''
Dec 10, 2018
@author: Faizan-Uni-Stuttgart

'''
import os
import timeit
import time
from pathlib import Path

import pandas as pd

from faizpy import get_ns
from hydmodeling.models.solve_cats_sys import get_resample_tags_arr


def get_resamp_ns(ref, sim, tags):

    n_tags = tags.shape[0]

    ref_resamp_vals = []

    for i in range(n_tags - 1):
        beg_tag_idx = tags[i]
        end_tag_idx = tags[i + 1]

        n_steps_per_tag = end_tag_idx - beg_tag_idx

        tag_steps_sum = 0
        for j in range(beg_tag_idx, end_tag_idx):
            tag_steps_sum += ref[j]

        tag_steps_mean = tag_steps_sum / n_steps_per_tag

        ref_resamp_vals.append(tag_steps_mean)

    ref_mean = sum(ref_resamp_vals) / len(ref_resamp_vals)

    sim_resamp_vals = []
    for i in range(n_tags - 1):
        beg_tag_idx = tags[i]
        end_tag_idx = tags[i + 1]

        n_steps_per_tag = end_tag_idx - beg_tag_idx

        tag_steps_sum = 0
        for j in range(beg_tag_idx, end_tag_idx):
            tag_steps_sum += sim[j]

        tag_steps_mean = tag_steps_sum / n_steps_per_tag

        sim_resamp_vals.append(tag_steps_mean)

    nom_sq_diff_sum = 0
    denom_sq_sum_diff = 0

    for i in range(n_tags - 1):
        nom_sq_diff_sum += (ref_resamp_vals[i] - sim_resamp_vals[i]) ** 2
        denom_sq_sum_diff += (ref_resamp_vals[i] - ref_mean) ** 2

    ns = 1 - (nom_sq_diff_sum / denom_sq_sum_diff)
    return ns


def main():

    main_dir = Path(r'P:\Synchronize\IWS\QGIS_Neckar\test_resampled_obj_ftns')
    os.chdir(main_dir)

    in_df = pd.read_csv(
        r'cat_411_qsims.csv',
        sep=';',
        usecols=['obs', 'kf_01_sim_0000'])

    in_df.index = pd.date_range('2005-01-01', periods=in_df.shape[0])
    in_df = in_df.loc['2007-06-05':'2014-07-14']

    resample_df = in_df.resample(rule='W').mean()

    orig_ns = get_ns(in_df.iloc[:, 0].values, in_df.iloc[:, 1].values)
    print(f'orig_ns: {orig_ns:0.8f}')

    resamp_ns_py = get_ns(
        resample_df.iloc[:, 0].values, resample_df.iloc[:, 1].values)
    print(f'resamp_ns_py: {resamp_ns_py:0.8f}')

    resample_tags_arr = in_df.index.weekofyear

    tags = get_resample_tags_arr(resample_tags_arr)

    resamp_ns_my = get_resamp_ns(
        in_df.iloc[:, 0].values, in_df.iloc[:, 1].values, tags)

    print(f'resamp_ns_my: {resamp_ns_my:0.8f}')
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
    START = timeit.default_timer()  # to get the runtime of the program

    main()

    STOP = timeit.default_timer()  # Ending time
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
