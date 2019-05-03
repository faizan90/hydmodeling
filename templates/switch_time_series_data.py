'''
@author: Faizan-Uni-Stuttgart

'''
import os
import time
import timeit
import pickle
from pathlib import Path
import configparser as cfpm

import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters

plt.ioff()
register_matplotlib_converters()


def load_pickle(in_file, mode='rb'):
    with open(in_file, mode) as _pkl_hdl:
        return pickle.load(_pkl_hdl)
    return


def main():

    main_dir = Path(r'P:\Synchronize\IWS\QGIS_Neckar\forcings')
    os.chdir(main_dir)

    swth_specs_file = r'time_series_switching.csv'
    sim_config_file = r'P:\Synchronize\Python3Codes\hydmodeling\templates\config_hydmodeling_template.ini'

    out_data_label = 'temp'
    swth_data_file_key = f'{out_data_label}_file'

    swtch_flag = False

    out_data_dir = main_dir / 'switched_data'

    sep = ';'

    time_fmt = '%Y-%m-%d'

    plot_flag = True

    # shoudl accomodate for event lengths as well
    warm_up_steps = 365
    warm_up_offset = pd.offsets.Day(warm_up_steps)
    lag_offset = pd.offsets.Day(10)

    cfp = cfpm.ConfigParser(interpolation=cfpm.ExtendedInterpolation())
    cfp.read(sim_config_file)

    swth_data_file = cfp['OPT_HYD_MODEL'][swth_data_file_key]

    swth_specs_df = pd.read_csv(swth_specs_file, sep=sep)

    # do before casting to datetime
    swth_specs_df['out_dir'] = ''
    for evt_idx in swth_specs_df.index:
        evt = swth_specs_df.loc[evt_idx]
        swth_specs_df.loc[evt_idx, 'out_dir'] = (
            f'scen_'
            f'{evt["ref_beg_time"]}_{evt["ref_end_time"]}'
            f'__'
            f'{evt["dst_beg_time"]}_{evt["dst_end_time"]}'
            )

    # do after out_dir assignment
    for col in swth_specs_df.columns:
        try:
            swth_specs_df[col] = pd.to_datetime(
                swth_specs_df[col], format=time_fmt)

            print(f'Converted column: {col} to datetime!')

        except Exception as msg:
            print(f'Could not convert column: {col} to datetime!')
            print('Error: ', msg)

        print('')

    out_data_dir.mkdir(exist_ok=True)

    swth_data_dfs_dict = load_pickle(swth_data_file)

    for evt_idx in swth_specs_df.index:
        print('')

        ref_beg_time = swth_specs_df.loc[evt_idx, 'ref_beg_time']
        ref_end_time = swth_specs_df.loc[evt_idx, 'ref_end_time']

        ref_peak_time = swth_specs_df.loc[evt_idx, 'ref_peak_time']

        print('Ref: ', ref_beg_time, ref_peak_time, ref_end_time)

        assert ref_beg_time <= ref_peak_time <= ref_end_time

        if swtch_flag:
            dst_lab = 'dst'

        else:
            dst_lab = 'ref'

        dst_beg_time = swth_specs_df.loc[evt_idx, f'{dst_lab}_beg_time']
        dst_end_time = swth_specs_df.loc[evt_idx, f'{dst_lab}_end_time']

        dst_peak_time = swth_specs_df.loc[evt_idx, f'{dst_lab}_peak_time']

        print('Dst: ', dst_beg_time, dst_peak_time, dst_end_time)

        assert dst_beg_time <= dst_peak_time <= dst_end_time

        out_dfs_dict = {}

        out_evt_dir = out_data_dir / swth_specs_df.loc[evt_idx, 'out_dir']

        out_evt_dir.mkdir(exist_ok=True)

        for cat in swth_data_dfs_dict:
            assert swth_data_dfs_dict[cat].index[0] <= (
                ref_beg_time - warm_up_offset)

            cat_df = swth_data_dfs_dict[cat]
            cat_df_idx = cat_df.index

            ref_peak_idx = cat_df_idx.get_loc(ref_peak_time)
            dst_peak_idx = cat_df_idx.get_loc(dst_peak_time)

            ref_steps_bef_peak = (
                ref_peak_idx -
                cat_df_idx.get_loc(ref_beg_time))

            ref_steps_aft_peak = (
                cat_df_idx.get_loc(ref_end_time) -
                ref_peak_idx)

            dst_steps_bef_peak = (
                dst_peak_idx -
                cat_df_idx.get_loc(dst_beg_time))

            dst_steps_aft_peak = (
                cat_df_idx.get_loc(dst_end_time) -
                dst_peak_idx)

            bef_steps_idx = max(ref_steps_bef_peak, dst_steps_bef_peak)
            aft_steps_idx = max(ref_steps_aft_peak, dst_steps_aft_peak)

            ref_fin_beg_time = (
                cat_df_idx[(ref_peak_idx - bef_steps_idx)] - warm_up_offset)

            ref_fin_end_time = (
                cat_df_idx[(ref_peak_idx + aft_steps_idx)] + lag_offset)

            dst_fin_beg_time = cat_df_idx[(dst_peak_idx - bef_steps_idx)]

            dst_fin_end_time = (cat_df_idx[(dst_peak_idx + aft_steps_idx)])

            ref_data_df = cat_df.loc[ref_fin_beg_time:ref_fin_end_time].copy()
            ref_data_df_orig = ref_data_df.copy()

            dst_data_df = cat_df.loc[dst_fin_beg_time:dst_fin_end_time].copy()

            ref_fin_peak_time_idx = ref_data_df.index.get_loc(ref_peak_time)

            dst_data_df.index = ref_data_df.index[
                ref_fin_peak_time_idx - bef_steps_idx:
                ref_fin_peak_time_idx + aft_steps_idx + 1]

            ref_data_df.update(dst_data_df)

            out_dfs_dict[cat] = ref_data_df

            if plot_flag:
                plt.figure(figsize=(20, 7))

                plt.plot(
                    ref_data_df_orig.mean(axis=1),
                    color='C0',
                    alpha=1.0,
                    lw=3.0,
                    label='ref_org')

                plt.plot(
                    ref_data_df.mean(axis=1),
                    color='C1',
                    alpha=1.0,
                    lw=2.0,
                    label='ref_fin')

                plt.plot(
                    dst_data_df.mean(axis=1),
                    color='C2',
                    alpha=1.0,
                    lw=1.0,
                    label='dst_org')

                plt.grid()
                plt.legend()

                plt.xlabel('Time')

                out_fig_name = f'{out_data_label}_{cat}_comparision.png'

                plt.savefig(out_evt_dir / out_fig_name, bbox_inches='tight')

                plt.close()

                print('Saved figure: ', out_fig_name)

        out_pkl_name = f'{out_data_label}_data.pkl'
        with open(out_evt_dir / out_pkl_name, 'wb') as pkl_hdl:
            pickle.dump(out_dfs_dict, pkl_hdl)

            print('Saved pickle: ', out_pkl_name)

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
