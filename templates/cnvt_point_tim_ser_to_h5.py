'''
@author: Faizan-Uni-Stuttgart

Jan 9, 2020

11:38:01 AM

'''
import os
import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
import h5py

DEBUG_FLAG = False


def cnvt_text_to_h5(
        in_file, out_file, beg_date, end_date, time_fmt, sep, label):

    in_file, out_file = Path(in_file), Path(out_file)
    label = str(label)

    in_df = pd.read_csv(in_file, sep=sep, index_col=0)
    in_df.index = pd.to_datetime(in_df.index, format=time_fmt)

    in_df = in_df.loc[beg_date:end_date]

    out_data_dict = {}
    out_rows_dict = {}
    out_cols_dict = {}
    out_area_ratios_dict = {}

    for i, stn in enumerate(in_df.columns):
        out_data_dict[int(stn)] = pd.DataFrame(
            data=in_df[stn].values,
            index=in_df.index,
            columns=[0],
            dtype=np.float64)

        out_rows_dict[int(stn)] = np.array([0], dtype=np.int64)
        out_cols_dict[int(stn)] = np.array([i], dtype=np.int64)
        out_area_ratios_dict[int(stn)] = np.array([1.0], dtype=np.float64)

    h5_times = in_df.index.strftime('%Y%m%dT%H%M%S')
    h5_str_dt = h5py.special_dtype(vlen=str)

    out_hdl = h5py.File(str(out_file), mode='w', driver=None)

    time_grp = out_hdl.create_group('time')
    time_strs_ds = time_grp.create_dataset(
        'time_strs',
        (h5_times.shape[0],),
        dtype=h5_str_dt)

    time_strs_ds[:] = h5_times

    stem_grp = out_hdl.create_group(in_file.stem)
    var_grp = stem_grp.create_group(label)

    out_hdl.create_group('rows')
    out_hdl.create_group('cols')
    out_hdl.create_group('rel_itsctd_area')

    for stn in in_df.columns:
        var_grp[str(stn)] = out_data_dict[int(stn)].values
        out_hdl[f'rows/{stn}'] = out_rows_dict[int(stn)]
        out_hdl[f'cols/{stn}'] = out_cols_dict[int(stn)]
        out_hdl[f'rel_itsctd_area/{stn}'] = out_area_ratios_dict[int(stn)]

    out_hdl.flush()

    return


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\QGIS_Neckar\hydmod\input_hyd_data')

    os.chdir(main_dir)

    in_files = [
        r'neckar_tot_prec_1901_2010_daysum__TOT_PREC.csv',
        r'neckar_full_neckar_ppt_interp__1961-01-01_to_2015-12-31_1km_all__EDK.csv',
        r'neckar_full_neckar_pet_kriging_1961-01-01_to_2015-12-31_1km_all__EDK.csv',
        r'neckar_full_neckar_avg_temp_kriging_1961-01-01_to_2015-12-31_1km_all__EDK.csv',
        ]

    out_files = ['neckar_ppt_cosmo.h5', 'neckar_ppt_obs.h5', 'neckar_pet_obs.h5', 'neckar_tem_obs.h5']

    labels = ['lump'] * len(in_files)

    beg_date = '1961-01-01'
    end_date = '2010-12-31'
    time_fmt = '%Y-%m-%d'

    sep = ';'

    assert in_files
    assert len(in_files) == len(out_files) == len(labels)

    for in_file, out_file, label in zip(in_files, out_files, labels):

        assert Path(in_file).exists(), f'{in_file} does not exist!'

        print(in_file, out_file, label)

        cnvt_text_to_h5(
            in_file, out_file, beg_date, end_date, time_fmt, sep, label)

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
