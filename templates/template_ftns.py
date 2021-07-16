'''
@author: Faizan3800X-Uni

Jul 14, 2021

9:09:48 AM
'''

import os

import h5py
import numpy as np
import pandas as pd


def get_data_dict_from_h5(path_to_h5, ds_grp, set_na_to_zero_flag=False):

    out_data_dict = {}
    with h5py.File(path_to_h5, mode='r', driver=None) as h5_hdl:
        h5_times = pd.to_datetime(
            h5_hdl['time/time_strs'][...].astype(str), format='%Y%m%dT%H%M%S')

        data_ds = h5_hdl[ds_grp]

        keys = [int(key) for key in data_ds.keys()]

        for key in keys:
            out_data_dict[key] = pd.DataFrame(
                index=h5_times, data=data_ds[str(key)][:])

            if set_na_to_zero_flag:
                nan_ct = np.isnan(out_data_dict[key].values).sum()

                if nan_ct:
                    print('\n')
                    print('#' * 30)
                    print(
                        f'WARNING: Set {nan_ct} values to zero in dataset '
                        f'{key} in file: {os.path.basename(path_to_h5)}!')
                    print('#' * 30)
                    print('\n')

                    out_data_dict[key].replace(np.nan, 0.0, inplace=True)

    return out_data_dict


def get_data_dict_from_h5_with_time(
        path_to_h5, ds_grp, beg_times, end_times, set_na_to_zero_flag=False):

    # beg_times and end_times have corresponding beg and end times
    # for the selection period

    out_data_dict = {}
    with h5py.File(path_to_h5, mode='r', driver=None) as h5_hdl:
        h5_times = pd.to_datetime(
            h5_hdl['time/time_strs'][...].astype(str), format='%Y%m%dT%H%M%S')

        select_idxs = np.zeros(h5_times.size, dtype=bool)

        for beg_time, end_time in zip(beg_times, end_times):
            sub_sel_idxs = ((h5_times >= beg_time) & (h5_times <= end_time))

            select_idxs |= sub_sel_idxs

        data_ds = h5_hdl[ds_grp]

        keys = [int(key) for key in data_ds.keys()]

        for key in keys:
            out_data_dict[key] = pd.DataFrame(
                index=h5_times[select_idxs],
                data=data_ds[str(key)][select_idxs,:])

            if set_na_to_zero_flag:
                nan_ct = np.isnan(out_data_dict[key].values).sum()

                if nan_ct:
                    print('\n')
                    print('#' * 30)
                    print(
                        f'WARNING: Set {nan_ct} values to zero in dataset '
                        f'{key} in file: {os.path.basename(path_to_h5)}!')
                    print('#' * 30)
                    print('\n')

                    out_data_dict[key].replace(np.nan, 0.0, inplace=True)

    return out_data_dict


def get_data_dict_from_h5_with_time_and_cat(
        path_to_h5,
        ds_grp,
        beg_times,
        end_times,
        cats,
        set_na_to_zero_flag=False):

    # beg_times and end_times have corresponding beg and end times
    # for the selection period

    out_data_dict = {}
    with h5py.File(path_to_h5, mode='r', driver=None) as h5_hdl:
        h5_times = pd.to_datetime(
            h5_hdl['time/time_strs'][...].astype(str), format='%Y%m%dT%H%M%S')

        select_idxs = np.zeros(h5_times.size, dtype=bool)

        for beg_time, end_time in zip(beg_times, end_times):
            sub_sel_idxs = ((h5_times >= beg_time) & (h5_times <= end_time))

            select_idxs |= sub_sel_idxs

        select_idxs = np.where(select_idxs)[0]

        data_ds = h5_hdl[ds_grp]

        keys = [int(key) for key in data_ds.keys()]

        for key in keys:

            if key not in cats:
                continue

            out_data_dict[key] = pd.DataFrame(
                index=h5_times[select_idxs],
                data=data_ds[str(key)][select_idxs,:])

            if set_na_to_zero_flag:
                nan_ct = np.isnan(out_data_dict[key].values).sum()

                if nan_ct:
                    print('\n')
                    print('#' * 30)
                    print(
                        f'WARNING: Set {nan_ct} values to zero in dataset '
                        f'{key} in file: {os.path.basename(path_to_h5)}!')
                    print('#' * 30)
                    print('\n')

                    out_data_dict[key].replace(np.nan, 0.0, inplace=True)

    assert out_data_dict, 'Nothing selected!'

    return out_data_dict


def get_cell_vars_dict_from_h5(path_to_h5, extra_ds=None):

    cat_intersect_rows_dict = {}
    cat_intersect_cols_dict = {}
    cat_area_ratios_dict = {}
    extent_shape = [-np.inf, -np.inf]

    if extra_ds is not None:
        assert isinstance(extra_ds, str), 'Only string dataset label allowed!'

        cat_extra_dss_dict = {}

    with h5py.File(path_to_h5, mode='r', driver=None) as h5_hdl:
        keys = [int(key) for key in h5_hdl['rows'].keys()]

        for key in keys:
            rows = h5_hdl[f'rows/{key}'][...]
            cols = h5_hdl[f'cols/{key}'][...]
            key_area_ratios = h5_hdl[f'rel_itsctd_area/{key}'][...]

            cat_intersect_rows_dict[key] = rows
            cat_intersect_cols_dict[key] = cols
            cat_area_ratios_dict[key] = key_area_ratios

            if rows.max() > extent_shape[0]:
                extent_shape[0] = rows.max()

            if cols.max() > extent_shape[1]:
                extent_shape[1] = cols.max()

            if extra_ds is not None:
                cat_extra_dss_dict[key] = h5_hdl[f'{extra_ds}/{key}'][...]

    out_dict = {
        'rows': cat_intersect_rows_dict,
        'cols': cat_intersect_cols_dict,
        'shape': extent_shape,
        'area_ratios': cat_area_ratios_dict}

    if extra_ds is not None:
        out_dict.update({extra_ds:cat_extra_dss_dict})

    return out_dict
