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

from hydmodeling import get_hargreaves_pet, change_pt_crs

DEBUG_FLAG = False


def change_crs(x, y, epsg):

    return change_pt_crs(x, y, epsg, 4326)


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

    main_dir = Path(os.getcwd())
    os.chdir(main_dir)

    in_min_temp_file = r''

    in_avg_temp_file = r''

    in_max_temp_file = r''

    # a list of some arguments
    # [field name to use as catchment names / numbers,
    # netCDF EPSG code,
    # netCDF time name,
    # netCDF X coords name,
    # netCDF Y coords name,
    # var name in min file
    # var name in max file
    # var name in avg file
    args = ['DN', 3396, 'longitude', 'latitude', 'EDK', 'EDK', 'EDK']

    units = 'mm/day'

    msgs = True

    out_dir = main_dir

    out_pet_file = out_dir / ''

    os.chdir(main_dir)

    if not out_dir.exists():
        out_dir.mkdir()

    in_avg_nc = nc.Dataset(in_avg_temp_file)
    lat_avg_arr = in_avg_nc.variables[args[3]][:]
    lon_avg_arr = in_avg_nc.variables[args[2]][:]

    time_avg_var = in_avg_nc.variables['z']

    time_avg_arr = nc.num2date(
        time_avg_var[:], time_avg_var.units, calendar=time_avg_var.calendar)

    in_min_nc = nc.Dataset(in_min_temp_file)
    lat_min_arr = in_min_nc.variables[args[3]][:]
    lon_min_arr = in_min_nc.variables[args[2]][:]

    time_min_var = in_min_nc.variables['z']

    time_min_arr = nc.num2date(
        time_min_var[:], time_min_var.units, calendar=time_min_var.calendar)

    in_max_nc = nc.Dataset(in_max_temp_file)
    lat_max_arr = in_max_nc.variables[args[3]][:]
    lon_max_arr = in_max_nc.variables[args[2]][:]

    time_max_var = in_max_nc.variables['time']

    time_max_arr = nc.num2date(
        time_max_var[:], time_max_var.units, calendar=time_max_var.calendar)

    assert (lat_min_arr.shape[0] ==
            lat_max_arr.shape[0] ==
            lat_avg_arr.shape[0])

    assert (lon_min_arr.shape[0] ==
            lon_max_arr.shape[0] ==
            lon_avg_arr.shape[0])

    assert np.all(np.isclose(lat_min_arr, lat_max_arr))

    assert np.all(np.isclose(lat_min_arr, lat_avg_arr))

    assert np.all(np.isclose(lon_min_arr, lon_max_arr))

    assert np.all(np.isclose(lon_min_arr, lon_avg_arr))

    assert (time_min_arr.shape[0] ==
            time_max_arr.shape[0] ==
            time_avg_arr.shape[0])

    assert np.all(np.isclose(time_min_arr, time_max_arr))

    assert np.all(np.isclose(time_min_arr, time_avg_arr))

    lons_grid, lats_grid = np.meshgrid(lon_min_arr, lat_min_arr)

    lons_grid, lats_grid = np.vectorize(change_crs)(
        lons_grid, lats_grid, args[1])

    out_pet_nc = nc.Dataset(str(out_pet_file), mode='w')

    out_pet_nc.set_auto_mask(False)

    out_pet_nc.createDimension(args[2], lon_min_arr.shape[0])
    out_pet_nc.createDimension(args[3], lat_min_arr.shape[0])
    out_pet_nc.createDimension('time', time_max_arr.shape[0])

    x_coords_nc = out_pet_nc.createVariable(args[2], 'd', dimensions=args[2])

    x_coords_nc[:] = lon_min_arr

    y_coords_nc = out_pet_nc.createVariable(args[3], 'd', dimensions=args[3])

    y_coords_nc[:] = lat_min_arr

    time_nc = out_pet_nc.createVariable('time', 'i8', dimensions='time')

    time_nc[:] = in_max_nc.variables['time'][:]
    time_nc.units = time_max_var.units
    time_nc.calendar = time_max_var.calendar

    idxs = np.arange(time_max_arr.shape[0])

    interp_type = 'PET'

    print('PET for:', interp_type)

    curr_pet_interp_var = (
        out_pet_nc.createVariable(
            interp_type,
            'd',
            dimensions=('time', args[3], args[2]),
            fill_value=False))

    har_vars_gen = ((in_min_nc.variables[args[4]][idxs[i]:idxs[i + 1]],
                     in_max_nc.variables[args[5]][idxs[i]:idxs[i + 1]],
                     in_avg_nc.variables[args[6]][idxs[i]:idxs[i + 1]],
                     time_max_arr[idxs[i]:idxs[i + 1]],
                     lats_grid,
                     idxs[i],
                     idxs[i + 1],
                     msgs)
                    for i in range(1))

    pet_flds = cmpt_hargreaves_pet_grids(next(har_vars_gen))[0]

    curr_pet_interp_var[:] = pet_flds

    curr_pet_interp_var.units = units
    curr_pet_interp_var.standard_name = 'PET (%s)' % interp_type

    out_pet_nc.sync()

    in_min_nc.close()
    in_max_nc.close()
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
