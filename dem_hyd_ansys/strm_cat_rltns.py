'''
Created on Oct 12, 2017

@author: Faizan Anwar, IWS Uni-Stuttgart
'''

import os
import timeit
import time
from pathlib import Path

import ogr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.ioff()


def get_shp_stuff(in_shp, field_id):

    '''
    Get every geometry and the layer extents of a given shapefile

    Only one layer allowed

    Geometries are inside a dictionary whose keys are the values of field_id
    '''

    assert os.path.exists(in_shp), 'in_shp (%s) does not exist!' % in_shp

    in_ds = ogr.Open(in_shp)

    lyr_count = in_ds.GetLayerCount()

    assert lyr_count, 'No layers in %s!' % in_shp
    assert lyr_count == 1, 'More than one layer in %s' % in_shp

    geoms_dict = {}

    shp_spt_ref = None

    in_lyr = in_ds.GetLayer(0)

    envelope = in_lyr.GetExtent()
    assert envelope, 'No envelope!'

    feat_count = in_lyr.GetFeatureCount()
    assert feat_count, 'No features in %s!' % in_shp

    if shp_spt_ref is None:
        shp_spt_ref = in_lyr.GetSpatialRef()

    assert shp_spt_ref, '%s has no spatial reference!' % in_shp

    for j in range(feat_count):
        curr_feat = in_lyr.GetFeature(j)

        if curr_feat is not None:
            cat_no = curr_feat.GetFieldAsString(str(field_id))

        else:
            continue

        geom = curr_feat.GetGeometryRef().Clone()
        assert geom is not None

        geoms_dict[cat_no] = geom

    in_ds.Destroy()

    assert geoms_dict, 'No geometries found!'

    return geoms_dict, envelope


def plot_strm_rltn(
        in_cat_shp,
        in_q_stns_shp,
        in_stms_net_shp,
        in_dem_net_file,
        in_cats_prcssed_file,
        in_stms_prcssed_file,
        final_cats_list,
        out_fig_path,
        sep=';'):

    in_cat_fld = 'DN'
    in_q_stns_fld = 'id'
    shp_stm_no_fld = 'stream_no'

    final_cats_list = [int(_c) for _c in final_cats_list]

    in_dem_net_df = pd.read_csv(
        in_dem_net_file, sep=sep, index_col=1)

    in_cats_prcssed_df = pd.read_csv(
        in_cats_prcssed_file, sep=sep, index_col=0)

    in_stms_prcssed_df = pd.read_csv(
        in_stms_prcssed_file, sep=sep, index_col=0)

    sel_cats = []
    sel_stms = []

    for fin_cat in final_cats_list:
        sel_cats.extend(in_cats_prcssed_df.loc[:fin_cat].index.tolist())

        if ((not in_stms_prcssed_df.shape[0]) or
            (fin_cat not in in_dem_net_df.index)):

            continue

        final_stms = in_dem_net_df.loc[fin_cat]

        if len(final_stms.shape) == 1:
            final_stm = final_stms.loc['stream_no']

        else:
            final_stm = final_stms.loc[
                final_stms['out_stm'].values == 1]['stream_no'].values[0]

        sel_stms.extend(in_stms_prcssed_df.loc[:final_stm].index.tolist())

    sel_cats = np.unique(sel_cats).tolist()
    sel_stms = np.unique(sel_stms).tolist()

    sel_cat_polys_list = []
    sel_q_stn_locs_list = []
    sel_stm_lines_list = []
    sel_cat_labs_list = []

    in_cat_polys_dict = get_shp_stuff(in_cat_shp, in_cat_fld)[0]
    in_q_stns_pts_dict = get_shp_stuff(in_q_stns_shp, in_q_stns_fld)[0]

    if sel_stms:
        in_stm_lines_dict = get_shp_stuff(in_stms_net_shp, shp_stm_no_fld)[0]

    else:
        in_stm_lines_dict = {}

    for _cat in in_cat_polys_dict:
        if int(_cat) not in sel_cats:
            continue

        sel_cat_polys_list.append(in_cat_polys_dict[_cat])
        sel_cat_labs_list.append(_cat)

    for _cat_pt in in_q_stns_pts_dict:
        if int(_cat_pt) not in sel_cats:
            continue

        sel_q_stn_locs_list.append(in_q_stns_pts_dict[_cat_pt])

    for _stm in in_stm_lines_dict:
        if int(_stm) not in sel_stms:
            continue

        sel_stm_lines_list.append(in_stm_lines_dict[_stm])

    cats_poly_pts = []
    q_stns_pts = []
    stm_lines_pts = []

    centroids_list = []

    for _poly in sel_cat_polys_list:
        geom_count = _poly.GetGeometryCount()
        _cent = _poly.Centroid()

        if geom_count == 1:
            _ = _poly.GetGeometryRef(0).GetPoints()

        else:
            assert _poly is not None, '_poly is None!'
            max_pts = 0
            max_pts_idx = None

            for i in range(geom_count):
                _ = _poly.GetGeometryRef(i).GetGeometryRef(0).GetPointCount()

                if _ is None:
                    break

                if _ > max_pts:
                    max_pts = _
                    max_pts_idx = i

            assert max_pts_idx is not None, 'Could not select a polygon!'

            _ = _poly.GetGeometryRef(
                max_pts_idx).GetGeometryRef(0).GetPoints()

        cats_poly_pts.append(np.array(_))
        centroids_list.append([_cent.GetX(), _cent.GetY()])

    for _cat_pt in sel_q_stn_locs_list:
        q_stns_pts.append([_cat_pt.GetX(), _cat_pt.GetY()])

    for _line in sel_stm_lines_list:
        stm_lines_pts.append(np.array(_line.GetPoints()))

    assert len(cats_poly_pts) == len(q_stns_pts), 'Missed some polygons!'

    plt.figure(figsize=(10, 10))
    plt.axes().set_aspect('equal', 'box')
    for i, _cat in enumerate(cats_poly_pts):
        plt.plot(_cat[:, 0], _cat[:, 1], 'g-', alpha=0.6)

        plt.text(
            centroids_list[i][0],
            centroids_list[i][1],
            sel_cat_labs_list[i],
            ha='center',
            va='center')

    for _stm in stm_lines_pts:
        plt.plot(_stm[:, 0], _stm[:, 1], 'b-', alpha=0.6)

    for _q_stn in q_stns_pts:
        plt.scatter(_q_stn[0], _q_stn[1], color='k', alpha=0.6)

    plt.title('Stream network and sub-catchments')
    plt.xlabel('Eastings')
    plt.ylabel('Northings')

    plt.savefig(out_fig_path, bbox_inches='tight')
    plt.close()
#     plt.show()
    return


def crt_strms_rltn_tree(
        prcss_cats_list,
        in_dem_net_file,
        in_watersheds_file,
        out_cats_prcssed_file,
        out_stms_prcssed_file,
        sep=';',
        watershed_field_name='DN'):

    in_dem_net_df = pd.read_csv(in_dem_net_file, sep=sep, index_col=0)
    dem_net_header = in_dem_net_df.columns.tolist()

    cat_vec = ogr.Open(in_watersheds_file)
    cat_lyr = cat_vec.GetLayer(0)
    cat_feat = cat_lyr.GetNextFeature()
    cats_area_dict = {}

    cats_us_list = in_dem_net_df[dem_net_header[0]].values.tolist()
    all_cats_list = []

    while cat_feat:
        f_val = cat_feat.GetFieldAsInteger(str(watershed_field_name))
        all_cats_list.append(int(f_val))

        geom = cat_feat.GetGeometryRef()
        cats_area_dict[f_val] = geom.Area()
        cat_feat = cat_lyr.GetNextFeature()
    cat_vec.Destroy()

    no_cats_us_list = []
    for _ in all_cats_list:
        if _ in cats_us_list:
            continue

        no_cats_us_list.append(_)

    def get_us_streams(curr_stream_no):

        '''Get streams that are upstream of a given stream'''

        if not ((stms_prcssed_df.shape[0] > 0) and
                (curr_stream_no == stms_prcssed_df.index[-1])):

            curr_us_stm_01 = in_dem_net_df.loc[
                curr_stream_no, dem_net_header[3]]

            curr_us_stm_02 = in_dem_net_df.loc[
                curr_stream_no, dem_net_header[4]]

            if curr_us_stm_01 != -2:
                if curr_us_stm_01 not in us_stms:
                    get_us_streams(curr_us_stm_01)

            if curr_us_stm_02 != -2:
                if curr_us_stm_02 not in us_stms:
                    get_us_streams(curr_us_stm_02)

        if curr_stream_no not in us_stms:
            us_stms.append(curr_stream_no)
        return

    def get_us_cats(cat_no):

        '''Get catchments that are upstream of a given catchment'''

        if cat_no in no_cats_us_list:
            if cat_no not in us_cats:
                us_cats.append(cat_no)
            return

        if ((cats_prcssed_df.shape[0] > 0) and
            (cat_no == cats_prcssed_df.iloc[-1, 0])):

            us_cats.append(cat_no)
            return

        for idx in in_dem_net_df.index:
            curr_cat = in_dem_net_df.loc[idx, dem_net_header[0]]
            curr_us_cat = in_dem_net_df.loc[idx, dem_net_header[-2]]

            if curr_cat == cat_no:
                if curr_us_cat == curr_cat:
                    continue

                if cat_no not in us_cats:
                    us_cats.append(cat_no)

                if curr_us_cat not in us_cats:
                    us_cats.append(curr_us_cat)
                    get_us_cats(curr_us_cat)
        return

    prcss_cats_list = [int(i) for i in prcss_cats_list]

    cats_prcssed_df = pd.DataFrame(
        columns=['prcssed', 'no_up_cats', 'area', 'cat_obj', 'optd'])
    cats_prcssed_df.index.name = 'cat_no'

    stms_prcssed_df = pd.DataFrame(columns=['prcssed', 'stm_obj', 'optd'])
    stms_prcssed_df.index.name = 'stream_no'

    for process_cat in prcss_cats_list:
        cats_prcssed_idxs = cats_prcssed_df.index

        if process_cat in cats_prcssed_idxs:
            continue

        us_cats = []
        get_us_cats(process_cat)
        us_cats = [int(i) for i in us_cats]

        for cat in reversed(us_cats):
            if cat in cats_prcssed_idxs:
                continue

            no_us_cats_flag = cat in no_cats_us_list

            cats_prcssed_df.loc[cat, :3] = [
                False, no_us_cats_flag, cats_area_dict[cat]]

            cats_prcssed_df.loc[cat, 'optd'] = False

    for process_cat in cats_prcssed_df.index:
        if ((process_cat in no_cats_us_list) and
            (process_cat not in in_dem_net_df.loc[:, dem_net_header[-2]])):

            continue

        stms_prcssed_idxs = stms_prcssed_df.index

        fin_stream_no = None
        for stream_no in in_dem_net_df.index:
            cat = in_dem_net_df.loc[stream_no, dem_net_header[0]]
            if cat != process_cat:
                continue

            out_stm = in_dem_net_df.loc[stream_no, dem_net_header[-1]]

            if out_stm == 1:
                fin_stream_no = stream_no
                break

        assert fin_stream_no is not None, (
            'Could not find the final stream for catchment %d!' % process_cat)

        if fin_stream_no == -2:
            continue

        us_stms = []
        get_us_streams(fin_stream_no)

        us_stms = [int(i) for i in us_stms]

        for stm in us_stms:
            if stm in stms_prcssed_idxs:
                continue

            stms_prcssed_df.loc[stm, ['prcssed', 'optd']] = False

    for prcss_cat in prcss_cats_list:
        assert prcss_cat in cats_prcssed_df.index, prcss_cat

    for stm_no in in_dem_net_df.index:
        if (in_dem_net_df.loc[stm_no, dem_net_header[0]]
            not in cats_prcssed_df.index):

            continue

        assert stm_no in stms_prcssed_df.index

    cats_prcssed_df.to_csv(
        out_cats_prcssed_file,
        sep=str(sep),
        index_label=cats_prcssed_df.index.name)

    stms_prcssed_df.to_csv(
        out_stms_prcssed_file,
        sep=str(sep),
        index_label=stms_prcssed_df.index.name)
    return


if __name__ == '__main__':
    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main_dir = Path(r'')

    os.chdir(main_dir)

    STOP = timeit.default_timer()  # Ending time
    print(('\n\a\a\a Done with everything on %s. Total run time was'
           ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
