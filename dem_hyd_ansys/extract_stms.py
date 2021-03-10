'''
Created on Oct 12, 2017

@author: Faizan Anwar, IWS Uni-Stuttgart
'''

import os
import tempfile
from shutil import copy2
from copy import deepcopy
from io import BytesIO as StringIO

import numpy as np
import pandas as pd
from osgeo import ogr
import shapefile as shp

from .misc import get_ras_props, get_ras_as_array_GDAL


def get_stms(in_dem_net_shp_file,
             in_wat_ids_file,
             in_dem_file,
             in_cats_file,
             in_gauges_coords_file,
             out_dem_net_shp_file,
             out_df_file,
             out_wat_ids_file,
             sep=';',
             gauge_coords_field_name='id'):

    '''Get the required stream network from the given one

    TauDEM creates many streams based on cell threshold but these
    are reduced to streams that start from a station and end at a station.

    Multi segments are merged into single ones.
    '''

    fin_field_names = ['DSNODEID', 'Length', 'Slope']

    in_dem_net_reader = shp.Reader(in_dem_net_shp_file)
    in_wat_id_arr = np.loadtxt(in_wat_ids_file,
                               delimiter=' ',
                               skiprows=1,
                               dtype=int)
    in_wat_id_arr = np.atleast_2d(in_wat_id_arr)

    shp_01 = StringIO()
    shx_01 = StringIO()
    dbf_01 = StringIO()

    shp_02 = StringIO()
    shx_02 = StringIO()
    dbf_02 = StringIO()

    shp_03 = StringIO()
    shx_03 = StringIO()
    dbf_03 = StringIO()

    out_dem_net_writer_01 = shp.Writer(
        shp=shp_01, shx=shx_01, dbf=dbf_01, shapeType=shp.POLYLINE)

    out_dem_net_writer_02 = shp.Writer(
        shp=shp_02, shx=shx_02, dbf=dbf_02, shapeType=shp.POLYLINE)

    out_dem_net_writer_03 = shp.Writer(
        shp=shp_03, shx=shx_03, dbf=dbf_03, shapeType=shp.POLYLINE)

    temp_dem_net_1_path = tempfile.NamedTemporaryFile().name + '.shp'
    temp_dem_net_2_path = tempfile.NamedTemporaryFile().name + '.shp'

    driver = ogr.GetDriverByName(str('ESRI Shapefile'))
    if os.path.exists(out_dem_net_shp_file):
        driver.DeleteDataSource(str(out_dem_net_shp_file))

    copy2(in_dem_net_shp_file.split('.')[0] + '.prj',
          out_dem_net_shp_file.split('.')[0] + '.prj')

    out_ds = driver.CreateDataSource(temp_dem_net_1_path)
    if out_ds is None:
        raise IOError('The outputs shapefile could not be deleted as it '
                      'seems to be in use!')

    dem_props = get_ras_props(in_dem_file)
    dem_x_min = dem_props[0]
    dem_y_max = dem_props[3]
    dem_pix_width = dem_props[6]
    dem_pix_heit = dem_props[7]
    dem_arr = np.flipud(get_ras_as_array_GDAL(in_dem_file))

    us_link_01_col_id = None
    us_link_02_col_id = None
    ds_node_col_id = None
    ds_link_col_id = None
    link_col_id = None
    length_col_id = None
    slope_col_id = None

    for idx, field in enumerate(in_dem_net_reader.fields):
        out_dem_net_writer_01.field(field[0], field[1], field[2], field[3])
        out_dem_net_writer_02.field(field[0], field[1], field[2], field[3])
        out_dem_net_writer_03.field(field[0], field[1], field[2], field[3])

        if field[0] == 'USLINKNO1':
            us_link_01_col_id = idx - 1

        elif field[0] == 'USLINKNO2':
            us_link_02_col_id = idx - 1

        elif field[0] == 'DSNODEID':
            ds_node_col_id = idx - 1

        elif field[0] == 'DSLINKNO':
            ds_link_col_id = idx - 1

        elif field[0] == 'LINKNO':
            link_col_id = idx - 1

        elif field[0] == 'Length':
            length_col_id = idx - 1

        elif field[0] == 'Slope':
            slope_col_id = idx - 1

    assert us_link_01_col_id is not None, 'Didn\'t get USLINKNO1 column index!'
    assert us_link_01_col_id >= 0, \
        'USLINKNO1 column index can\'t less than be zero!'

    assert us_link_02_col_id is not None, 'Didn\'t get USLINKNO2 column index!'
    assert us_link_02_col_id >= 0, \
        'USLINKNO2 column index can\'t less than be zero!'

    assert ds_node_col_id is not None, 'Didn\'t get DSNODEID column index!'
    assert ds_node_col_id >= 0, \
        'DSNODEID column index can\'t less than be zero!'

    assert ds_link_col_id is not None, 'Didn\'t get DSLINKNO column index!'
    assert ds_link_col_id >= 0, \
        'DSLINKNO column index can\'t less than be zero!'

    assert link_col_id is not None, 'Didn\'t get LINKNO column index!'
    assert link_col_id >= 0, 'LINKNO column index can\'t less than be zero!'

    assert length_col_id is not None, 'Didn\'t get Length column index!'
    assert length_col_id >= 0, 'Length column index can\'t less than be zero!'

    assert slope_col_id is not None, 'Didn\'t get Slope column index!'
    assert slope_col_id >= 0, 'Slope column index can\'t less than be zero!'

    unique_cats_list = []
    cat_count = 0
    for idx, rec in enumerate(in_dem_net_reader.records()):
        cat_no = rec[ds_node_col_id]
        if (cat_no != -1) and (cat_no not in unique_cats_list):
            unique_cats_list.append(cat_no)
            cat_count += 1

    unique_wat_ids = np.unique(in_wat_id_arr)
    for unique_cat in unique_cats_list:
        if unique_cat not in in_wat_id_arr:
            raise Exception('Catchment id not in wat_id_arr!')

    in_dem_net_recs = in_dem_net_reader.records()

    got_fin_outlet = False
    for unique_cat in unique_wat_ids:
        if unique_cat not in unique_cats_list:
#             if got_fin_outlet:
#                 raise RuntimeError('More than one outlet in watersheds_id?')
            unique_cat_idxs = np.where(in_wat_id_arr == unique_cat)
            unique_cat_idxs_rows = unique_cat_idxs[0]
            for row_no in unique_cat_idxs_rows:
                in_wat_id_arr[row_no, 1] = -1
                got_fin_outlet = True

#             if len(unique_cat_idxs) == 1:
#                 unique_cat_idxs_col = unique_cat_idxs[1][0]
#             else:
#             unique_cat_idxs_col = unique_cat_idxs[1][0]
#             in_wat_id_arr[unique_cat_idxs_row, unique_cat_idxs_col] = -1

            got_fin_outlet = True

    assert got_fin_outlet, 'No outlet?'

    np.savetxt(
        out_wat_ids_file,
        in_wat_id_arr,
        delimiter=str(sep),
        fmt='%d')

    watersheds_ds_id_dict = {}
    watersheds_us_id_dict = {}
    for idx in range(in_wat_id_arr.shape[0]):
        ds_cat = in_wat_id_arr[idx, 1]
        us_cat = in_wat_id_arr[idx, 0]
        watersheds_ds_id_dict[us_cat] = ds_cat
        if ds_cat not in list(watersheds_us_id_dict.keys()):
            watersheds_us_id_dict[ds_cat] = []
        watersheds_us_id_dict[ds_cat].append(us_cat)

    cats_not_us_cats_list = []
    cats_us_cats_list = []
    for cat_no in unique_cats_list:
        if cat_no not in in_wat_id_arr[:, 1]:
            cats_not_us_cats_list.append(cat_no)
        else:
            cats_us_cats_list.append(cat_no)

    assert (len(cats_us_cats_list) +
            len(cats_not_us_cats_list)) == len(unique_cats_list)

    del_stream_ids_list = []
    # streams directly next to catchment outlets
    direct_next_streams_node_ids_list = []

    def get_us_cats(curr_stream_idx):
        us_link_no1 = in_dem_net_recs[curr_stream_idx][us_link_01_col_id]
        us_link_no2 = in_dem_net_recs[curr_stream_idx][us_link_02_col_id]

        if us_link_no1 != -1:
            us_link_no1_idx = None
            for idx, rec in enumerate(in_dem_net_recs):
                if rec[link_col_id] == us_link_no1:
                    us_link_no1_idx = idx
                    break

            assert us_link_no1_idx is not None, \
                'Couldn\'t get index for %d' % us_link_no1

            get_us_cats(us_link_no1_idx)

        if us_link_no2 != -1:
            us_link_no2_idx = None
            for idx, rec in enumerate(in_dem_net_recs):
                if rec[link_col_id] == us_link_no2:
                    us_link_no2_idx = idx
                    break

            assert us_link_no2_idx is not None, \
                'Couldn\'t get index for %d' % us_link_no2

            get_us_cats(us_link_no2_idx)
        del_stream_ids_list.append(curr_stream_idx)
        return

    for cats_not_us_cat in cats_not_us_cats_list + cats_us_cats_list:
        cat_ds_node_no = None

        for idx, rec in enumerate(in_dem_net_recs):
            if rec[ds_node_col_id] == cats_not_us_cat:
                cat_ds_node_no = idx
                break

        assert cat_ds_node_no is not None, \
            'Couldn\'t get index for catchment %d' % cats_not_us_cat

        curr_stream_idx = cat_ds_node_no

        if cats_not_us_cat in cats_not_us_cats_list:
            del_stream_ids_list.append(curr_stream_idx)
            get_us_cats(curr_stream_idx)

        direct_next_streams_node_ids_list.append(
            in_dem_net_recs[curr_stream_idx][ds_link_col_id])

    assert len(direct_next_streams_node_ids_list) == len(unique_cats_list), (
        'unequal lengths!')

    n_recs = 0
    for idx, rec in enumerate(
        zip(in_dem_net_reader.records(), in_dem_net_reader.shapes())):

        if rec[0][us_link_01_col_id] != -1:
            del_idx_cond = (idx not in del_stream_ids_list)

            if del_idx_cond:
                out_dem_net_writer_01.line(parts=[rec[1].points])
                out_dem_net_writer_01.record(*rec[0])
                n_recs += 1

    fin_stream_idxs_list = []
    contain_cats_list = []

    if n_recs:
        recs_and_shapes = list(zip(
            out_dem_net_writer_01.records, out_dem_net_writer_01.shapes()))

    else:
        recs_and_shapes = []

#     assert recs_and_shapes

    def get_cont_streams(cat_no, ds_link_no):
        curr_idx = None
        if ds_link_no == -1:
            for idx, rec in enumerate(recs_and_shapes):
                if rec[0][ds_node_col_id] == cat_no:
                    curr_idx = idx
                    break
        else:
            for idx, rec in enumerate(recs_and_shapes):
                if rec[0][link_col_id] == ds_link_no:
                    curr_idx = idx
                    break

        if ((rec[0][ds_node_col_id] == watersheds_ds_id_dict[cat_no]) or
            (ds_link_no == -1)):

            if curr_idx is not None:
                fin_stream_idxs_list.append(curr_idx)
                contain_cats_list.append(cat_no)

            return

        else:
            assert curr_idx is not None, (cat_no, ds_link_no)

            fin_stream_idxs_list.append(curr_idx)
            contain_cats_list.append(cat_no)

            get_cont_streams(
                cat_no, recs_and_shapes[curr_idx][0][ds_link_col_id])
        return

    if cats_us_cats_list:
        for item in zip(
            (cats_not_us_cats_list + cats_us_cats_list),
            direct_next_streams_node_ids_list):

            try:
                get_cont_streams(*item)

            except Exception as msg:
                raise Exception('Error in: %s., %s' % (str(item[0]), str(msg)))

    contain_test_list = []
    for idx, fin_idx in enumerate(fin_stream_idxs_list):
        if fin_idx in contain_test_list:
            continue

        temp_cat_reqs = recs_and_shapes[fin_idx][0]
        ds_cat = watersheds_ds_id_dict[contain_cats_list[idx]]

        curr_ds_node_id = temp_cat_reqs[ds_node_col_id]
        if curr_ds_node_id != -1:
            temp_cat_reqs[ds_link_col_id] = -1

        if ds_cat == -1:
            ds_cat = contain_cats_list[idx]
        else:
            temp_cat_reqs[ds_node_col_id] = ds_cat

        out_dem_net_writer_02.line(
            parts=[recs_and_shapes[fin_idx][1].points])
        out_dem_net_writer_02.record(*temp_cat_reqs)
        contain_test_list.append(fin_idx)

    new_recs_list = []
    link_nos_list = []

    if n_recs:
        for rec in out_dem_net_writer_02.records:
            link_nos_list.append(rec[link_col_id])

        for rec in deepcopy(out_dem_net_writer_02.records):
            curr_us_link_no_1 = rec[us_link_01_col_id]
            curr_us_link_no_2 = rec[us_link_02_col_id]

            if curr_us_link_no_1 not in link_nos_list:
                curr_us_link_no_1 = -1

            if curr_us_link_no_2 not in link_nos_list:
                curr_us_link_no_2 = -1

            rec[us_link_01_col_id] = curr_us_link_no_1
            rec[us_link_02_col_id] = curr_us_link_no_2

            new_recs_list.append(rec)

        for item in zip(new_recs_list, out_dem_net_writer_02.shapes()):
            out_dem_net_writer_03.line(parts=[item[1].points])
            out_dem_net_writer_03.record(*item[0])

        out_dem_net_writer_03.save(temp_dem_net_2_path)

        temps_streams_vec = ogr.Open(temp_dem_net_2_path)
        lyr = temps_streams_vec.GetLayer(0)

        out_lyr = out_ds.CreateLayer(str('0'), geom_type=ogr.wkbMultiLineString)

        in_lyr_dfn = lyr.GetLayerDefn()
        lyr.GetLayerDefn().GetFieldDefn(ds_node_col_id).SetType(ogr.OFTInteger)
        lyr.GetLayerDefn().GetFieldDefn(length_col_id).SetWidth(16)
        lyr.GetLayerDefn().GetFieldDefn(length_col_id).SetPrecision(1)
        lyr.GetLayerDefn().GetFieldDefn(slope_col_id).SetWidth(16)
        lyr.GetLayerDefn().GetFieldDefn(slope_col_id).SetPrecision(10)
        for i in range(0, in_lyr_dfn.GetFieldCount()):
            field_dfn = in_lyr_dfn.GetFieldDefn(i)
            out_lyr.CreateField(field_dfn)

        out_lyr_dfn = out_lyr.GetLayerDefn()

        direct_us_streams_list = []

    def get_us_stream_fids(stream_fid):
        us_stream_link_no_1 = (
            feat_dict[stream_fid].GetFieldAsInteger(us_link_01_col_id))
        us_stream_link_no_2 = (
            feat_dict[stream_fid].GetFieldAsInteger(us_link_02_col_id))

        stream_cond_01 = (us_stream_link_no_1 != -1)
        stream_cond_02 = (us_stream_link_no_2 != -1)

        direct_us_streams_list.append(stream_fid)

        if stream_cond_01 and (not stream_cond_02):
            if us_stream_link_no_1 not in feat_us_links_list:
                return
            us_stream_id_01 = None
            for fid in feat_dict:
                if feat_dict[fid].GetFieldAsInteger(link_col_id) == (
                        us_stream_link_no_1):
                    us_stream_id_01 = fid
            assert us_stream_id_01 is not None, 'us_stream_id_01 is None!'
            get_us_stream_fids(us_stream_id_01)

        elif (not stream_cond_01) and stream_cond_02:
            if us_stream_link_no_2 not in feat_us_links_list:
                return
            us_stream_id_02 = None
            for fid in feat_dict:
                if feat_dict[fid].GetFieldAsInteger(link_col_id) == (
                        us_stream_link_no_2):
                    us_stream_id_02 = fid
            assert us_stream_id_02 is not None, 'us_stream_id_02 is None!'
            get_us_stream_fids(us_stream_id_02)
        return

    def get_line_elevs(in_geom):
        geom_pts = in_geom.GetPoints()
        pt_1 = geom_pts[0]
        pt_2 = geom_pts[-1]

        row_pt_1 = int((dem_y_max - pt_1[1]) / dem_pix_heit)
        col_pt_1 = int((pt_1[0] - dem_x_min) / dem_pix_width)
        z_coord_pt_1 = dem_arr[row_pt_1, col_pt_1]

        row_pt_2 = int((dem_y_max - pt_2[1]) / dem_pix_heit)
        col_pt_2 = int((pt_2[0] - dem_x_min) / dem_pix_width)
        z_coord_pt_2 = dem_arr[row_pt_2, col_pt_2]
        return (z_coord_pt_1, z_coord_pt_2)

    def get_slope(in_geom):
        length = in_geom.Length()
        (z_coord_pt_1, z_coord_pt_2) = get_line_elevs(in_geom)

        if (z_coord_pt_1 - z_coord_pt_2) == 0:
            # assuming average rounding is 0.5 meter
            slope = 1e-9
        else:
            slope = abs((z_coord_pt_2 - z_coord_pt_1) / length)
        return slope

    def merge_to_single_line(in_points_list):
        lines_count = len(in_points_list)
        us_rels_tree = {}
        for i in range(lines_count):
            pt_i_1 = in_points_list[i][0]
            pt_i_2 = in_points_list[i][-1]
            for j in range(lines_count):
                if i == j:
                    continue

                pt_j_1 = in_points_list[j][0]
                pt_j_2 = in_points_list[j][-1]

                if pt_i_1 == pt_j_2:
                    us_rels_tree[j] = i
                elif pt_i_2 == pt_j_1:
                    us_rels_tree[i] = j

        if len(us_rels_tree) == (lines_count - 1):
            stream_order_list = [None] * lines_count

            for i in range(lines_count):
                nothing_us = False
                i_order = (lines_count - 1)
                curr_ds_stream = i

                while not nothing_us:
                    try:
                        curr_us_stream = us_rels_tree[curr_ds_stream]
                        i_order -= 1
                        curr_ds_stream = curr_us_stream
                        continue

                    except KeyError:
                        stream_order_list[i] = i_order
                        nothing_us = True

            out_line = ogr.Geometry(ogr.wkbLineString)
            for i in stream_order_list:
                curr_line = in_points_list[i]
                for point in curr_line:
                    out_line.AddPoint(point[0], point[1])

            return out_line
        return

    def get_max_slope(in_multi_geom):

        geom_count = in_multi_geom.GetGeometryCount()
        points_list = []

        if not geom_count:
            geom_count = in_multi_geom.GetPointCount()

            if not geom_count:
                raise RuntimeError('No features, no points!')

            points_list.append(in_multi_geom.GetPoints())
            merged_line = merge_to_single_line(points_list)

        else:
            for geom_i in range(geom_count):
                curr_geom = in_multi_geom.GetGeometryRef(geom_i)
                points_list.append(curr_geom.GetPoints())

            merged_line = merge_to_single_line(points_list)

        return (get_slope(merged_line), merged_line)

    if n_recs:
        feat_count = lyr.GetFeatureCount()
        for cat_no in unique_cats_list:
            if cat_no in cats_not_us_cats_list:
                continue

            feat_dict = {}
            feat_us_links_list = []

            fin_cat_stream_fid = None
            for idx in range(feat_count):
                feat = lyr.GetFeature(idx)
                stream_cat_no = feat.GetFieldAsInteger(ds_node_col_id)

                if stream_cat_no != cat_no:
                    continue

                stream_fid = feat.GetFID()
                feat_us_links_list.append(feat.GetFieldAsInteger(link_col_id))
                feat_dict[stream_fid] = feat.Clone()
                stream_ds_link_no = feat.GetFieldAsInteger(ds_link_col_id)

                if stream_ds_link_no == -1:
                    fin_cat_stream_fid = stream_fid

            assert fin_cat_stream_fid is not None, (
                'Couldn\'t get fin_cat_stream_fid!')

            if len(feat_dict) == 1:
                out_feat = ogr.Feature(out_lyr_dfn)

                geom = list(feat_dict.values())[0].GetGeometryRef()

                out_feat.SetField(ds_node_col_id, int(cat_no))

                _ = feat_dict[fin_cat_stream_fid]
                _ = _.GetFieldAsInteger(us_link_01_col_id)

                out_feat.SetField(us_link_01_col_id, _)
                out_feat.SetField(length_col_id, geom.Length())
                out_feat.SetField(slope_col_id, get_slope(geom))
                out_feat.SetGeometry(geom)

                out_lyr.CreateFeature(out_feat)
                continue

            else:
                merged_stream_fids_list = []
                for curr_fid in feat_dict:
                    if curr_fid in merged_stream_fids_list:
                        continue

                    direct_us_streams_list = []
                    get_us_stream_fids(curr_fid)

                    for fid in direct_us_streams_list:
                        if fid not in merged_stream_fids_list:
                            merged_stream_fids_list.append(fid)

                    if len(direct_us_streams_list) > 1:
                        ds_cat_feat = feat_dict[direct_us_streams_list[0]].Clone()

                        for fid in direct_us_streams_list[1:]:
                            ds_cat = ds_cat_feat.GetGeometryRef()
                            curr_cat_feat = feat_dict[fid].Clone()
                            curr_cat = curr_cat_feat.GetGeometryRef()

                            ds_cat = ds_cat.Union(curr_cat)
                            merged_cat_feat = ogr.Feature(out_lyr_dfn)
                            merged_cat_feat.SetGeometry(ds_cat)
                            ds_cat_feat = merged_cat_feat

                        slope, merged_line = get_max_slope(
                            ds_cat_feat.GetGeometryRef())

                        fin_cat_feat = ogr.Feature(out_lyr_dfn)
                        fin_cat_feat.SetGeometry(merged_line)

                        fin_cat_feat.SetField(ds_node_col_id, int(cat_no))
                        fin_cat_feat.SetField(
                            us_link_01_col_id,
                            fin_cat_feat.GetFieldAsInteger(us_link_01_col_id))

                        fin_cat_feat.SetField(length_col_id, merged_line.Length())
                        fin_cat_feat.SetField(slope_col_id, slope)

                        out_lyr.CreateFeature(fin_cat_feat)

                    elif len(direct_us_streams_list) == 1:
                        geom = feat_dict[
                            direct_us_streams_list[0]].GetGeometryRef()

                        out_feat = ogr.Feature(out_lyr_dfn)
                        out_feat.SetField(ds_node_col_id, int(cat_no))

                        _ = feat_dict[direct_us_streams_list[0]]
                        _ = _.GetFieldAsInteger(us_link_01_col_id)

                        out_feat.SetField(us_link_01_col_id, _)
                        out_feat.SetField(length_col_id, geom.Length())
                        out_feat.SetField(slope_col_id, get_slope(geom))
                        out_feat.SetGeometry(geom)

                        out_lyr.CreateFeature(out_feat)

                    elif not direct_us_streams_list:
                        raise Exception('zero lines for FID: %s' % str(curr_fid))

                    else:
                        raise Exception(
                            ('Couldn\'t do anything about FID: %s, %s') % (
                                str(curr_fid), str(direct_us_streams_list)))

        feat_count = out_lyr.GetFeatureCount()
        del_feat_ids_list = []
        for i in range(feat_count):
            if i in del_feat_ids_list:
                continue

            i_feat = out_lyr.GetFeature(i)
            i_geom = i_feat.GetGeometryRef()
            for j in range(feat_count):
                if (i != j) and (i not in del_feat_ids_list):
                    j_feat = out_lyr.GetFeature(j)
                    j_geom = j_feat.GetGeometryRef()

                    if j_geom.Contains(i_geom):
                        del_feat_ids_list.append(i)

        del_feat_ids_list.sort()
        for del_feat_id in reversed(del_feat_ids_list):
            out_lyr.DeleteFeature(del_feat_id)

        field_id = 0
        things_todo = True
        while things_todo:
            curr_fin_field_ids = [
                out_lyr.GetLayerDefn().GetFieldIndex(str(i))
                for i in fin_field_names]

            field_count = out_lyr.GetLayerDefn().GetFieldCount()
            for field_id in range(field_count):
                if field_id not in curr_fin_field_ids:
                    out_lyr.DeleteField(field_id)
                    break

            else:
                things_todo = False

        out_lyr_defn = out_lyr.GetLayerDefn()
        up_stream_01_defn = ogr.FieldDefn(str('up_strm_01'))
        up_stream_01_defn.SetType(ogr.OFTInteger)
        out_lyr.CreateField(up_stream_01_defn)

        up_stream_02_defn = ogr.FieldDefn(str('up_strm_02'))
        up_stream_02_defn.SetType(ogr.OFTInteger)
        out_lyr.CreateField(up_stream_02_defn)

        stream_no_defn = ogr.FieldDefn(str('stream_no'))
        stream_no_defn.SetType(ogr.OFTInteger)
        out_lyr.CreateField(stream_no_defn)

        up_stream_cat_defn = ogr.FieldDefn(str('up_cat'))
        up_stream_cat_defn.SetType(ogr.OFTInteger)
        out_lyr.CreateField(up_stream_cat_defn)

        out_stream_cat_defn = ogr.FieldDefn(str('out_stm'))
        out_stream_cat_defn.SetType(ogr.OFTInteger)
        out_lyr.CreateField(out_stream_cat_defn)

        up_stream_01_col_id = out_lyr_defn.GetFieldIndex(str('up_strm_01'))
        up_stream_02_col_id = out_lyr_defn.GetFieldIndex(str('up_strm_02'))
        stream_no_col_id = out_lyr_defn.GetFieldIndex(str('stream_no'))
        up_stream_cat_col_id = out_lyr_defn.GetFieldIndex(str('up_cat'))
        out_stream_cat_col_id = out_lyr_defn.GetFieldIndex(str('out_stm'))

        if any([
            (up_stream_01_col_id == -1),
            (up_stream_02_col_id == -1),
            (stream_no_col_id == -1),
            (up_stream_cat_col_id == -1),
            (out_stream_cat_col_id == -1)]):

            raise Exception(
                'Fields not created properly or invalid names requested!')

        ndv = -2  # means no upstreams
        fin_stream_no = 0
        for geom_i in range(out_lyr.GetFeatureCount()):
            out_feat = out_lyr.GetFeature(geom_i)
            if out_feat is None:
                continue

            out_feat.SetField(stream_no_col_id, fin_stream_no)
            out_feat.SetField(up_stream_01_col_id, ndv)
            out_feat.SetField(up_stream_02_col_id, ndv)
            out_feat.SetField(out_stream_cat_col_id, ndv)
            fin_stream_no += 1
            out_lyr.SetFeature(out_feat)

        for geom_i in range(out_lyr.GetFeatureCount()):
            curr_stream_feat = out_lyr.GetFeature(geom_i)
            no_more_us_streams = False

            if curr_stream_feat is None:
                continue

            geoms = curr_stream_feat.GetGeometryRef()
            geoms_count = geoms.GetGeometryCount()

            if geoms_count == 0:
                stream_us_pt = geoms.GetPoints()[-1]

            else:
                raise Exception('More than one geometry in the feature!')

            for geom_j in range(out_lyr.GetFeatureCount()):
                if geom_i == geom_j:
                    continue

                curr_us_stream_feat = out_lyr.GetFeature(geom_j)
                if curr_us_stream_feat is None:
                    continue

                geoms = curr_us_stream_feat.GetGeometryRef()
                geoms_count = geoms.GetGeometryCount()

                if geoms_count == 0:
                    us_stream_us_pt = geoms.GetPoints()[0]

                else:
                    raise Exception('More than one geometry in the feature!')

                if us_stream_us_pt == stream_us_pt:
                    us_stream_no = (
                        curr_us_stream_feat.GetField(stream_no_col_id))

                    if curr_stream_feat.GetField(up_stream_01_col_id) == ndv:
                        curr_stream_feat.SetField(
                            up_stream_01_col_id, us_stream_no)

                    else:
                        curr_stream_feat.SetField(
                            up_stream_02_col_id, us_stream_no)

                        no_more_us_streams = True

                    out_lyr.SetFeature(curr_stream_feat)

                    if no_more_us_streams:
                        break

    cat_vec = ogr.Open(in_cats_file)
    cat_lyr = cat_vec.GetLayer(0)
    cat_feat = cat_lyr.GetNextFeature()
    cats_dict = {}
    while cat_feat:
        f_val = cat_feat.GetFieldAsInteger(str('DN'))
        geom = cat_feat.GetGeometryRef().Clone()
        cats_dict[f_val] = geom
        cat_feat = cat_lyr.GetNextFeature()

    coords_vec = ogr.Open(in_gauges_coords_file)
    coords_lyr = coords_vec.GetLayer(0)
    coords_feat = coords_lyr.GetNextFeature()
    coords_feat_dict = {}
    while coords_feat:
        gauge_no = coords_feat.GetFieldAsInteger(str(gauge_coords_field_name))
        gauge_geom = coords_feat.GetGeometryRef().Clone()
        coords_feat_dict[gauge_no] = gauge_geom
        coords_feat = coords_lyr.GetNextFeature()

    geoms_count = 0

    half_pix_width = dem_pix_width * 0.5
    half_pix_heit = dem_pix_heit * 0.5

    if n_recs:
        for geom_i in range(out_lyr.GetFeatureCount()):
            curr_stream_feat = out_lyr.GetFeature(geom_i)
            if curr_stream_feat is None:
                continue

            curr_stream_us_pt = ogr.Geometry(ogr.wkbPoint)
            curr_stream_ds_pt = ogr.Geometry(ogr.wkbPoint)

            stm_us_coords = curr_stream_feat.GetGeometryRef().GetPoints()[-1]
            stm_us_x_coord, stm_us_y_coord = stm_us_coords[:2]

            stm_ds_coords = curr_stream_feat.GetGeometryRef().GetPoints()[0]
            stm_ds_x_coord, stm_ds_y_coord = stm_ds_coords[:2]

            curr_stream_us_pt.AddPoint(stm_us_x_coord, stm_us_y_coord)
            curr_stream_ds_pt.AddPoint(stm_ds_x_coord, stm_ds_y_coord)
            feat_modifd = False

            for cat in cats_dict:
                if cats_dict[cat].Contains(curr_stream_us_pt):
                    curr_stream_feat.SetField(up_stream_cat_col_id, cat)
                    feat_modifd = True

                if cats_dict[cat].Contains(curr_stream_ds_pt):
                    ring = ogr.Geometry(ogr.wkbLinearRing)

                    ring.AddPoint(stm_ds_x_coord - half_pix_width,
                                  stm_ds_y_coord - half_pix_heit)

                    ring.AddPoint(stm_ds_x_coord - half_pix_width,
                                  stm_ds_y_coord + half_pix_heit)

                    ring.AddPoint(stm_ds_x_coord + half_pix_width,
                                  stm_ds_y_coord + half_pix_heit)

                    ring.AddPoint(stm_ds_x_coord + half_pix_width,
                                  stm_ds_y_coord - half_pix_heit)

                    ring.AddPoint(stm_ds_x_coord - half_pix_width,
                                  stm_ds_y_coord - half_pix_heit)

                    poly = ogr.Geometry(ogr.wkbPolygon)
                    poly.AddGeometry(ring)

                    if poly.Contains(coords_feat_dict[cat]):
                        curr_stream_feat.SetField(out_stream_cat_col_id, 1)
                        feat_modifd = True

                if feat_modifd:
                    out_lyr.SetFeature(curr_stream_feat)

            geoms_count += 1

        out_cols = [out_lyr.GetLayerDefn().GetFieldDefn(i).GetName()
                    for i in range(out_lyr.GetLayerDefn().GetFieldCount())]

        out_df = pd.DataFrame(index=list(range(geoms_count)), columns=out_cols)

        curr_field_idxs = [out_lyr.GetLayerDefn().GetFieldIndex(str(i))
                           for i in out_cols]
        geoms_count = 0
        for geom_i in range(out_lyr.GetFeatureCount()):
            curr_stream_feat = out_lyr.GetFeature(geom_i)
            if curr_stream_feat is None:
                continue
            out_df.loc[geoms_count] = [
                curr_stream_feat.GetField(i) for i in curr_field_idxs]

            geoms_count += 1

        if not out_df.shape[0]:
            assert not cats_us_cats_list
            out_df = pd.DataFrame(index=list(range(len(cats_not_us_cats_list))),
                                  columns=out_cols)

            for i, cat in enumerate(cats_not_us_cats_list):
                out_df.loc[i, out_cols] = (
                    [-1, 0, 0, -2, -2, i, cats_not_us_cats_list[i], -2])

        out_df.index = out_df['stream_no'].values
        out_df.drop(labels=['stream_no'], axis=1, inplace=True)

        out_df.to_csv(out_df_file, sep=str(sep), index_label='stream_no')

        driver.CopyDataSource(out_ds, out_dem_net_shp_file)

        cat_vec.Destroy()
        out_ds.Destroy()
        coords_vec.Destroy()
        temps_streams_vec.Destroy()

        driver.DeleteDataSource(temp_dem_net_1_path)
        driver.DeleteDataSource(temp_dem_net_2_path)

    out_df = pd.DataFrame(
        columns='stream_no;DSNODEID;Length;Slope;up_strm_01;up_strm_02;up_cat;out_stm'.split(';'))

    out_df.to_csv(out_df_file, sep=str(sep), index_label='stream_no')
    return


if __name__ == '__main__':
    pass
