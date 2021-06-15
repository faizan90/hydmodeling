'''
@author: Faizan-Uni-Stuttgart

May 7, 2021

3:02:56 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile as shp
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

plt.ioff()

DEBUG_FLAG = False


def get_resample_years(beg_year, end_year, window_size):

    resample_years = []
    for year in range(beg_year, end_year, window_size):

        if (year + window_size - 1) > end_year:
            continue

        resample_years.append((str(year), str(year + window_size - 1)))

    return resample_years


def main():

    main_dir = Path(r'P:\Synchronize\IWS\QGIS_Neckar')
    os.chdir(main_dir)

    #==========================================================================
    # All input time series should have the same spatio-temporal resolution.
    # All should have the same columns.
    # Columns have integer labels.

    sep = ';'

    wat_bal_years = 5

    beg_time = '1961-01-01'
    end_time = '2015-12-31'

    time_fmt = '%Y-%m-%d'

    fig_size = (15, 15)
    dpi = 300

    drop_stns = [4416, 460]

    without_pet_flag = True
    min_wat_bal_ratio = 0.2
    max_wat_bal_ratio = 1.0

#     without_pet_flag = False
#     min_wat_bal_ratio = 0.8
#     max_wat_bal_ratio = 1.5

    out_dir = Path(r'water_balance_plots__neckar')

    # PPT in mm/day.
    in_ppt_file = Path(
        r'hydmod\input_hyd_data\watersheds_cumm_neckar\neckar_full_neckar_ppt_interp__1961-01-01_to_2015-12-31_1km_all__EDK.csv')

    # PET in mm/day.
    in_pet_file = Path(
        r'hydmod\input_hyd_data\pet_edk_1961_to_2015_daily_2km_all_lumped.csv')

    # Discharge in cumecs.
    in_dis_file = Path(
        r'hydmod\input_hyd_data\neckar_daily_discharge_1961_2015.csv')

    # Catchments shapefile.
    in_cat_file = Path(r'raster\taudem_out_neckar_20180624\watersheds.shp')

    cat_col = 'DN'

    # Catchment area shapefile.
    in_area_file = Path(
        r'raster\taudem_out_neckar_20180624\watersheds_cumm_cat_areas.csv')

    area_col = 'cumm_area'

    # Streams shapefile.
    in_strms_file = Path(
        r'raster\taudem_out_neckar_20180624\dem_net_cat_streams_only.shp')

    #==========================================================================

    out_dir.mkdir(exist_ok=True)

    ppt_df = pd.read_csv(in_ppt_file, sep=sep, index_col=0)
    ppt_df.index = pd.to_datetime(ppt_df.index, format=time_fmt)
    ppt_df = ppt_df.loc[beg_time:end_time]

    pet_df = pd.read_csv(in_pet_file, sep=sep, index_col=0)
    pet_df.index = pd.to_datetime(pet_df.index, format=time_fmt)
    pet_df = pet_df.loc[beg_time:end_time]

    dis_df = pd.read_csv(in_dis_file, sep=sep, index_col=0)
    dis_df.index = pd.to_datetime(dis_df.index, format=time_fmt)
    dis_df = dis_df.loc[beg_time:end_time]

    cats_hdl = shp.Reader(str(in_cat_file))

    area_ser = pd.read_csv(in_area_file, sep=sep, index_col=0)[area_col]

    strms_hdl = shp.Reader(str(in_strms_file))

    ppt_df.columns = [int(col) for col in ppt_df]
    pet_df.columns = [int(col) for col in pet_df]
    dis_df.columns = [int(col) for col in dis_df]

    ppt_df.drop(drop_stns, axis=1, inplace=True, errors='ignore')
    pet_df.drop(drop_stns, axis=1, inplace=True, errors='ignore')
    dis_df.drop(drop_stns, axis=1, inplace=True, errors='ignore')

    assert ppt_df.columns.intersection(
        pet_df.columns.intersection(
            dis_df.columns.intersection(area_ser.index))).size == (
                ppt_df.columns.size)

    assert ppt_df.index.intersection(
        pet_df.index.intersection(dis_df.index)).size == (
            ppt_df.index.size)

    # Discharge converted to runoff in mm per day.
    run_df = dis_df.copy()
    run_df.iloc[:,:] = np.nan
    for cat in ppt_df.columns:
        area = area_ser.loc[cat]

        run_df[cat] = dis_df[cat] * ((1000 * 86400) / area)

    resample_years = get_resample_years(
        ppt_df.index.year.min(),
        ppt_df.index.year.max(),
        wat_bal_years)

    assert len(resample_years)

    if len(resample_years) == 1:
        resample_idxs = pd.to_datetime([f'{resample_years[-1][1]}-12-31'])

    else:
        resample_idxs = pd.date_range(
            f'{resample_years[+0][1]}-12-31',
            f'{resample_years[-1][1]}-12-31',
            freq=f'{wat_bal_years}A')

    print(resample_years)
    print(resample_idxs)

    assert len(resample_years) == resample_idxs.size

#     ppt_df = ppt_df.resample(f'{wat_bal_years}A', closed='left').sum()
#     pet_df = pet_df.resample(f'{wat_bal_years}A', closed='left').sum()
#     run_df = run_df.resample(f'{wat_bal_years}A', closed='left').sum()

    ppt_rsm_df = pd.DataFrame(index=resample_idxs, columns=ppt_df.columns)
    pet_rsm_df = pd.DataFrame(index=resample_idxs, columns=ppt_df.columns)
    run_rsm_df = pd.DataFrame(index=resample_idxs, columns=ppt_df.columns)

    for resample_year, resample_idx in zip(resample_years, resample_idxs):
        ppt_rsm_df.loc[resample_idx] = ppt_df.loc[
            resample_year[0]:resample_year[1]].sum()

        pet_rsm_df.loc[resample_idx] = pet_df.loc[
            resample_year[0]:resample_year[1]].sum()

        run_rsm_df.loc[resample_idx] = run_df.loc[
            resample_year[0]:resample_year[1]].sum()

    wat_bal_df = ppt_rsm_df.copy()
    wat_bal_df.iloc[:,:] = np.nan

    if without_pet_flag:
        out_suff = 'without_pet'

        pet_rsm_df.iloc[:,:] = 0

    else:
        out_suff = 'with_pet'

    for cat in ppt_rsm_df.columns:
        wat_bal_df[cat] = (run_rsm_df[cat] + pet_rsm_df[cat]) / ppt_rsm_df[cat]

    wat_bal_df.to_csv(
        out_dir / f'water_balances_{out_suff}_{wat_bal_years}Y.csv',
        sep=sep,
        float_format='%0.4f')

    cat_col_idx = None
    for i, field in enumerate(cats_hdl.fields[1:]):
        if field[0] == cat_col:
            cat_col_idx = i

    assert cat_col_idx is not None

    for date in wat_bal_df.index:

        wat_bals = wat_bal_df.loc[date]

        fig, ax = plt.subplots(figsize=fig_size)

        cmap_mappable_beta = plt.cm.ScalarMappable(
            cmap='jet',
            norm=Normalize(min_wat_bal_ratio, max_wat_bal_ratio, clip=True))

        cmap_mappable_beta.set_array([])

        for record, shape in zip(cats_hdl.records(), cats_hdl.shapes()):

            if record[cat_col_idx] in drop_stns:
                continue

            pts = np.array(shape.points)

            wat_bal = wat_bals.loc[record[cat_col_idx]]

            clr = cmap_mappable_beta.to_rgba(wat_bal)

            poly = plt.Polygon(pts, color=clr, alpha=0.5, ec=None)

            ax.add_patch(poly)

            ax.plot(pts[:, 0], pts[:, 1], c='k')

            centroid = np.median(pts, axis=0)
            plt.text(
                centroid[0],
                centroid[1],
                f'{record[cat_col_idx]}\n({wat_bal:0.2f})',
                va='center',
                ha='center')

        for shape in strms_hdl.shapes():
            pts = np.array(shape.points)
            ax.plot(pts[:, 0], pts[:, 1], c='b', ls='-', alpha=0.75)

        ax.set_aspect('equal')

        plt.grid()
        plt.gca().set_axisbelow(True)

        cbaxes = fig.add_axes([0.87, 0.15, 0.05, 0.7])

        plt.colorbar(
            mappable=cmap_mappable_beta,
            cax=cbaxes,
            orientation='vertical',
            label='Relative Water Balance',
            drawedges=False)

    #     plt.tight_layout()

        plt.savefig(
            str(out_dir / f'wat_bal_{out_suff}_{wat_bal_years}Y_{date.year}.png'),
            dpi=dpi)

        print(f'wat_bal_{out_suff}_{wat_bal_years}Y_{date.year}.png')

        plt.close()
    return


if __name__ == '__main__':
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
