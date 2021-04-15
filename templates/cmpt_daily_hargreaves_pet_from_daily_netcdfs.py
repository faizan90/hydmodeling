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

import pyproj

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

        pet_vals = vec_ftn(curr_date.dayofyear,
                           degs_grid,
                           min_temp_grid[i],
                           max_temp_grid[i],
                           avg_temp_grid[i],
                           leap)

        if msgs:
            nan_ct = np.isnan(pet_vals).sum()

            if nan_ct:
                print(f'{nan_ct} out of {n_cells} values are NaNs in pet '
                      f'grid on {curr_date}!')

        pet_grid[i] = pet_vals

#         print(i, np.nanmax(pet_vals))

    return strt_index, end_index, pet_grid


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Colleagues_Students\Jochen\dS2\neckar_full\data')
    os.chdir(main_dir)

    in_min_temp_file = r'tem_min_interp\tem_min.nc'

    in_avg_temp_file = r'tem_avg_interp\tem_avg.nc'

    in_max_temp_file = r'tem_max_interp\tem_max.nc'

    # a list of some arguments
    # [field name to use as catchment names / numbers,
    # netCDF EPSG code,
    # netCDF time name,
    # netCDF X coords name,
    # netCDF Y coords name,
    # var name in min file
    # var name in max file
    # var name in avg file
    args = ['DN', 3396, 'X', 'Y', 'OK', 'OK', 'OK']

    time_var = 'time'

    units = 'mm/day'

    msgs = True

    out_dir = main_dir / 'pet_interp'

    out_pet_file = out_dir / 'pet.nc'

    os.chdir(main_dir)

    if not out_dir.exists():
        out_dir.mkdir()

    in_avg_nc = nc.Dataset(in_avg_temp_file)
    lat_avg_arr = in_avg_nc.variables[args[3]][:].data
    lon_avg_arr = in_avg_nc.variables[args[2]][:].data

    time_avg_var = in_avg_nc.variables[time_var]

    time_avg_arr = nc.num2date(
        time_avg_var[:],
        time_avg_var.units,
        calendar=time_avg_var.calendar,
        only_use_cftime_datetimes=False).data.astype(np.datetime64)

    in_min_nc = nc.Dataset(in_min_temp_file)
    lat_min_arr = in_min_nc.variables[args[3]][:].data
    lon_min_arr = in_min_nc.variables[args[2]][:].data

    time_min_var = in_min_nc.variables[time_var]

    time_min_arr = nc.num2date(
        time_min_var[:],
        time_min_var.units,
        calendar=time_min_var.calendar,
        only_use_cftime_datetimes=False).data.astype(np.datetime64)

    in_max_nc = nc.Dataset(in_max_temp_file)
    lat_max_arr = in_max_nc.variables[args[3]][:].data
    lon_max_arr = in_max_nc.variables[args[2]][:].data

    time_max_var = in_max_nc.variables[time_var]

    time_max_arr = nc.num2date(
        time_max_var[:],
        time_max_var.units,
        calendar=time_max_var.calendar,
        only_use_cftime_datetimes=False).data.astype(np.datetime64)

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

    assert np.all(time_min_arr == time_max_arr)

    assert np.all(time_min_arr == time_avg_arr)

    lons_grid, lats_grid = np.meshgrid(lon_min_arr, lat_min_arr)

    tfmr = pyproj.Transformer.from_crs(
        f'EPSG:{args[1]}', f'EPSG:{4326}', always_xy=True)

    lons_grid, lats_grid = tfmr.transform(lons_grid, lats_grid)

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

    idxs = np.array([0, time_max_arr.shape[0]])

    interp_type = 'PET'

    print('PET for:', interp_type)

    curr_pet_interp_var = (
        out_pet_nc.createVariable(
            interp_type,
            'd',
            dimensions=('time', args[3], args[2]),
            fill_value=False))

    har_vars_gen = ((in_min_nc.variables[args[4]][idxs[i]:idxs[i + 1]].data,
                     in_max_nc.variables[args[5]][idxs[i]:idxs[i + 1]].data,
                     in_avg_nc.variables[args[6]][idxs[i]:idxs[i + 1]].data,
                     time_max_arr[idxs[i]:idxs[i + 1]],
                     lats_grid,
                     idxs[i],
                     idxs[i + 1],
                     msgs)
                    for i in range(1))

    pet_flds = cmpt_hargreaves_pet_grids(next(har_vars_gen))[2]

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
