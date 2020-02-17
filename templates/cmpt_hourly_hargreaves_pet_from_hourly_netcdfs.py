'''
@author: Faizan-Uni-Stuttgart

14 Feb 2020

13:21:14

'''
import os
import time
import timeit
from pathlib import Path
from calendar import isleap

import numpy as np
import pandas as pd
import netCDF4 as nc

from pyproj import Proj, transform

from hydmodeling import get_hargreaves_pet

DEBUG_FLAG = False


def cmpt_hargreaves_pet_grids(args):

    (min_temp_grid,
     max_temp_grid,
     avg_temp_grid,
     times_arr,
     lats_grid,
     strt_index,
     end_index,
     msgs) = args

    n_cells = avg_temp_grid.shape[1] * avg_temp_grid.shape[2]

    vec_ftn = np.vectorize(get_hargreaves_pet, otypes=[np.float])

    pet_grid = np.full(avg_temp_grid.shape, np.nan)

    times_idxs = pd.DatetimeIndex(times_arr)

    degs_grid = np.deg2rad(lats_grid)

    if msgs:
        print('Output shape:', pet_grid.shape)

    for i, curr_date in enumerate(times_idxs):
        if isleap(curr_date.year):
            leap = 0
        else:
            leap = 1

#         if msgs:
#             min_ge_avg_sum = (min_temp_grid[i] > avg_temp_grid[i]).sum()
#             min_ge_max_sum = (min_temp_grid[i] > max_temp_grid[i]).sum()
#             avg_ge_max_sum = (avg_temp_grid[i] > avg_temp_grid[i]).sum()
#
#             if min_ge_avg_sum:
# #                 _idxs = min_temp_grid[i] > avg_temp_grid[i]
# #                 print(min_temp_grid[i][_idxs])
# #                 print(avg_temp_grid[i][_idxs])
#                 print((f'{min_ge_avg_sum} out of {n_cells} min. temperature '
#                        f'values greater than the mean on {curr_date}'))
#
#             if min_ge_max_sum:
#                 print(('%d out of %d min. temperature values greater than the '
#                        'max on %s') % (min_ge_max_sum,
#                                        n_cells,
#                                        str(curr_date)))
#
#             if avg_ge_max_sum:
#                 print(('%d out of %d mean temperature values greater than the '
#                        'max on %s') % (avg_ge_max_sum,
#                                        n_cells,
#                                        str(curr_date)))

        pet_vals = vec_ftn(curr_date.dayofyear,
                           degs_grid,
                           min_temp_grid[i],
                           max_temp_grid[i],
                           avg_temp_grid[i],
                           leap)

        if msgs:
#             print(f'min, max pet on {curr_date}: {pet_vals.min():0.2f}, '
#                   f'{pet_vals.max():0.2f}')
            nan_ct = np.isnan(pet_vals).sum()
            if nan_ct:
                print(f'{nan_ct} out of {n_cells} values are NaNs in pet '
                      f'grid on {curr_date}!')

        pet_grid[i] = pet_vals

    return strt_index, end_index, pet_grid


def main():

    main_dir = Path(r'E:\\')
    os.chdir(main_dir)

    in_hr_temp_file = r'E:\neckar_temp_hr_interp\neckar_temperature_spinterp_2008-01-01-00_to_2015-12-31-23_1km.nc'

    # a list of some arguments
    # [field name to use as catchment names / numbers,
    # netCDF EPSG code,
    # netCDF time name,
    # netCDF X coords name,
    # netCDF Y coords name,
    # var name in hr file
    # time var name in hr file
    args = ['DN', 31467, 'X', 'Y', 'IDW_000', 'time']

    out_units = 'mm/hour'

    msgs = True

    out_dir = main_dir / 'neckar_pet_hr_interp'

    out_pet_file = out_dir / 'neckar_pet_daily_spinterp_2008-01-01-00_to_2015-12-31-23_1km.nc'

    os.chdir(main_dir)

    if not out_dir.exists():
        out_dir.mkdir()

    in_hr_nc = nc.Dataset(in_hr_temp_file)
    lat_hr_arr = in_hr_nc.variables[args[3]][:]
    lon_hr_arr = in_hr_nc.variables[args[2]][:]

    lons_grid, lats_grid = np.meshgrid(lon_hr_arr.data, lat_hr_arr.data)

    out_crs = Proj("+init=EPSG:" + str(4326))

    in_crs = Proj("+init=EPSG:" + str(args[1]))

    lons_grid, lats_grid = transform(in_crs, out_crs, lons_grid, lats_grid)

    degs_grid = np.deg2rad(lats_grid)

    tem_hr_var = in_hr_nc.variables[args[4]]

    time_hr_var = in_hr_nc.variables[args[5]]

    time_hr_all_arr = pd.DatetimeIndex(nc.num2date(
        time_hr_var[:], time_hr_var.units, calendar=time_hr_var.calendar))

    time_hr_arr = pd.date_range(
        time_hr_all_arr[0], time_hr_all_arr[-1], freq='H')

    time_daily_arr = time_hr_arr[time_hr_arr.hour == 0]

    assert np.all(
        (time_daily_arr[1:] - time_daily_arr[:-1]).total_seconds() == 86400)

    daily_min_arr = np.zeros(
        (time_daily_arr.size,
         in_hr_nc.variables[args[4]].shape[1],
         in_hr_nc.variables[args[4]].shape[2]))

    daily_max_arr = daily_min_arr.copy()

    daily_avg_arr = daily_min_arr.copy()

#     pet_hr_vals = np.full(tem_hr_var.shape, np.nan, dtype=float)

    pet_vec_ftn = np.vectorize(get_hargreaves_pet, otypes=[np.float])

    #==========================================================================
    # out nc beg
    #==========================================================================
    out_pet_nc = nc.Dataset(str(out_pet_file), mode='w')

    out_pet_nc.set_auto_mask(False)

    out_pet_nc.createDimension(args[2], lon_hr_arr.shape[0])
    out_pet_nc.createDimension(args[3], lat_hr_arr.shape[0])
    out_pet_nc.createDimension('time', time_hr_arr.shape[0])

    x_coords_nc = out_pet_nc.createVariable(args[2], 'd', dimensions=args[2])

    x_coords_nc[:] = lon_hr_arr

    y_coords_nc = out_pet_nc.createVariable(args[3], 'd', dimensions=args[3])

    y_coords_nc[:] = lat_hr_arr

    time_nc = out_pet_nc.createVariable('time', 'i8', dimensions='time')

    time_nc[:] = in_hr_nc.variables['time'][:]
    time_nc.units = time_hr_var.units
    time_nc.calendar = time_hr_var.calendar

    interp_type = 'PET'

    print('PET for:', interp_type)

    pet_hr_interp_var = (
        out_pet_nc.createVariable(
            interp_type,
            'd',
            dimensions=('time', args[3], args[2]),
            fill_value=False))

    #==========================================================================
    # out nc end
    #==========================================================================

    for i, time_step in enumerate(time_daily_arr):

        if isleap(time_step.year):
            leap = 0

        else:
            leap = 1

        time_step_str = time_step.strftime('%Y-%m-%d')
        sel_idxs = (
            (time_hr_all_arr >= f'{time_step_str} 00:00:00') &
            (time_hr_all_arr <= f'{time_step_str} 23:59:59'))

        assert sel_idxs.sum() == 24, (
            f'{time_step} does not have 24 hours instead {sel_idxs.sum()}!')

        step_vals = tem_hr_var[sel_idxs, :, :].data

        step_mins = step_vals.min(axis=0)
        step_maxs = step_vals.max(axis=0)
        step_avgs = step_vals.mean(axis=0)

        fint_idxs = np.isfinite(step_mins)

        n_finites = fint_idxs.sum()

        if not n_finites:
            if msgs:
                print(f'No interp for {time_step}!')

            continue

        daily_min_arr[i, :, :] = step_mins
        daily_max_arr[i, :, :] = step_maxs
        daily_avg_arr[i, :, :] = step_avgs

        step_vals_ge_null = step_vals.copy()

        step_vals_ge_null[step_vals_ge_null < 0] = 0

        step_vals_ge_null_sum = step_vals_ge_null.sum(axis=0)

#         # don't want to divide by zero
#         step_vals_ge_null_sum[step_vals_ge_null_sum == 0] = 1

        step_vals_wts = step_vals_ge_null / step_vals_ge_null_sum

        step_vals_wts[:, step_vals_ge_null_sum == 0] = 1 / step_vals.shape[0]

        step_vals_wts[:, np.where(fint_idxs)[0][0], np.where(fint_idxs)[1][0]]

        assert (
            np.isfinite(step_vals_ge_null_sum).sum() ==
            n_finites)

        pet_step_vals = pet_vec_ftn(
            time_step.dayofyear,
            degs_grid,
            step_mins,
            step_maxs,
            step_avgs,
            leap)

        assert np.isfinite(pet_step_vals).sum() == n_finites

        pet_step_hr_vals = np.empty_like(step_vals)

        pet_step_hr_vals[:, :, :] = pet_step_vals

        pet_step_hr_vals *= step_vals_wts

        assert np.all(
            np.isclose(pet_step_hr_vals[:, fint_idxs].sum(axis=0),
                       pet_step_vals[fint_idxs]))

#         print(time_step, pet_step_hr_vals[:, fint_idxs].sum(axis=0).mean())

        pet_hr_interp_var[sel_idxs] = pet_step_hr_vals

#
#     curr_pet_interp_var[:] = pet_flds
#
    pet_hr_interp_var.units = out_units
    pet_hr_interp_var.standard_name = 'PET (%s)' % interp_type

    in_hr_nc.close()
    out_pet_nc.close()
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
