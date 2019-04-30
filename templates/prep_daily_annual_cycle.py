"""
Created on Tue Jun 13 09:57:44 2017

@author: Faizan
"""

import os
import timeit
import time
import shutil

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from matplotlib.dates import YearLocator, DateFormatter

plt.ioff()

if __name__ == '__main__':
    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main_dir = os.path.join(r'P:\Synchronize\IWS\QGIS_Neckar')

    in_data_file = os.path.join(
        r'P:\Synchronize\IWS\Discharge_data_longer_series',
        r'neckar_norm_cop_infill_discharge_infill_1961_2015',
        r'infilled_var_df_infill_stns.csv')

    out_data_file = os.path.join(r'hydmod\input_hyd_data',
                                 r'neckar_discharge_annual_cycle.csv')

    out_figs_dir = os.path.join(r'hydmod\input_hyd_data',
                                r'neckar_annual_cycle_01')

    in_sep = ';'
    in_date_fmt = '%Y-%m-%d'
    strt_date = '1961-01-01'
    end_date = '2016-12-31'
    freq = 'D'
    out_fig_size = (15, 5)

    # Two ways to calculate the annual cycle
    # If cycle_type = 1, then an average for all days of the same month and day is
    # used. If cycle_type = 2, then a triangular weightage function is used to get
    # the mean value
    cycle_type = 1

    # The number of days to use in the triangular function as buffer on one side.
    # e.g a value of 3 means that we use three days before and three days after the
    # the value and final values used is 7.
    buffer_days = 6

    # Min and max weights for the triangular function
    # Min is weight of the first and last buffer day
    # Max is the weigth of the day at which we want the annual cycle value
    # The sum of weigths should equal 1, otherwise a ValueError is raised
    # This needs to be optimized because right now it is a trial and error process
    # to get the sum of the weights equal to one
    min_max_tri_weights = [0.01, 0.155]

    years = YearLocator(5)  # put a tick on x axis every N years
    yearsFmt = DateFormatter('%Y')  # how to show the year

    os.chdir(main_dir)

    if cycle_type == 2:
        assert buffer_days, 'Buffer days not greater than 0!'
        buff_days = np.arange(1, buffer_days + 2, dtype=int)
        tri_weights_list = []
        for buff_day in buff_days:
            tri_weights_list.append((min_max_tri_weights[0] +
                                     ((min_max_tri_weights[1] -
                                       min_max_tri_weights[0]) *
                                      ((buff_day - buff_days[0]) /
                                       (buff_days[-1] - buff_days[0])))))

        tri_weights_arr = np.array(tri_weights_list +
                                   tri_weights_list[:-1][::-1])
        print('Triangular weights:', tri_weights_arr, '\n')

        wts_sum = np.sum(tri_weights_arr)
        assert np.isclose(wts_sum, 1.0), \
            'Sum of weights (%0.5f) not equal to one!' % wts_sum
    #    raise Exception

    if not os.path.exists(out_figs_dir):
        os.mkdir(out_figs_dir)
    else:
        shutil.rmtree(out_figs_dir)
        os.mkdir(out_figs_dir)

    in_data_df = pd.read_csv(in_data_file, sep=in_sep, index_col=0)
    in_data_df.index = pd.to_datetime(in_data_df.index, format=in_date_fmt)

    new_date_range = pd.date_range(strt_date, end_date, freq=freq)
    in_data_df = in_data_df.reindex(new_date_range)

    annual_cycle_df = pd.DataFrame(index=in_data_df.index,
                                   columns=in_data_df.columns,
                                   dtype=float)

    fig_title_str = ('Discharge annual cycle time series for station: \"%s\" '
                     'in Neckar catchment, Germany\n')

    plt.figure(figsize=out_fig_size)
    for col in in_data_df.columns:
        print('Going through:', col)
        # Get the time series for a given station
        col_ser = in_data_df[col].copy()
        assert isinstance(col_ser, pd.Series), 'Expected a pd.Series object!'
        col_ser.dropna(inplace=True)

        # For each day of a year, get the days for all year and average them
        # the annual cycle is the average value and is used for every doy of all
        # years
        for month in range(1, 13):
            for dom in range(1, 32):
                month_idxs = col_ser.index.month == month
                dom_idxs = col_ser.index.day == dom
                idxs_intersect = np.logical_and(month_idxs, dom_idxs)
                curr_day_vals = col_ser.values[idxs_intersect]

                if not curr_day_vals.shape[0]:
                    continue

                assert not np.any(np.isnan(curr_day_vals)), \
                    'NaNs in curr_day_vals!'

                # Change the calculation of mean value here.
                # Can use a triangular function as Prof. Bardossy said
                if cycle_type == 1:
                    curr_day_avg_val = curr_day_vals.mean()
                    col_ser.loc[idxs_intersect] = curr_day_avg_val
                elif cycle_type == 2:
                    weighted_vals_arr = np.zeros(col_ser.shape[0])
                    tri_idxs_intersect = np.full(col_ser.shape[0],
                                                 False,
                                                 dtype=bool)

                    # get steps that will be used in the triangular weighting
                    for i in range(buffer_days, col_ser.shape[0] - buffer_days):
                        if not idxs_intersect[i]:
                            continue

                        tri_idxs_intersect[i] = True
                        curr_sel_vals = \
                            col_ser.values[(i - buffer_days):
                                           (i + buffer_days + 1)]
                        curr_tri_vals = curr_sel_vals * tri_weights_arr
                        curr_wt_val = np.sum(curr_tri_vals)
                        weighted_vals_arr[i] = curr_wt_val

                    curr_day_avg_val = \
                        weighted_vals_arr[tri_idxs_intersect].mean()
                    col_ser.loc[tri_idxs_intersect] = curr_day_avg_val
                else:
                    raise ValueError('Incorrect cycle type: %s!' %
                                     str(cycle_type))

                annual_cycle_df.update(col_ser)

        # Plot the whole time series
        # It should be just repeating values every year
        gs = gridspec.GridSpec(1, 1)
        p_axis = plt.subplot(gs[0, 0])
        p_axis.plot(annual_cycle_df.index,
                    annual_cycle_df.loc[:, col].values,
                    'k-')

        p_axis.set_title(fig_title_str % col)

        p_axis.set_xlabel('Time (days)')
        p_axis.set_ylabel('Discharge ($m^3/s$)')
        p_axis.xaxis.set_major_locator(years)
        p_axis.xaxis.set_major_formatter(yearsFmt)

        p_axis.set_xlim(annual_cycle_df.index[0],
                        annual_cycle_df.index[-1])
        p_axis.grid()
        p_axis.xaxis.set_major_locator(years)
        p_axis.xaxis.set_major_formatter(yearsFmt)

        plt.savefig(os.path.join(out_figs_dir, str(col) + '.png'),
                    bbox_inches='tight',
                    dpi=400)

        plt.clf()
    #    break

    plt.close()

    annual_cycle_df.to_csv(out_data_file, sep=in_sep, float_format='%0.3f')

    STOP = timeit.default_timer()  # Ending time
    print(('\n\a\a\a Done with everything on %s. Total run time was'
           ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
