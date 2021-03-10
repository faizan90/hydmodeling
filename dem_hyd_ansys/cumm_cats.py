"""
Created on %(date)s

@author: %(Faizan Anwar, IWS Uni-Stuttgart)s
"""
import timeit
import time

import os
import numpy as np
import pandas as pd

from osgeo import ogr


def get_cumm_cats(in_cat_shp,
                  id_field_name,
                  in_id_file,
                  in_sep,
                  out_cat_shp,
                  out_descrip_file,
                  out_sep):

    cat_vec = ogr.Open(in_cat_shp)
    lyr = cat_vec.GetLayer(0)
    spt_ref = lyr.GetSpatialRef()

    feat_dict = {}
    feat_area_dict = {}

    feat = lyr.GetNextFeature()
    while feat:
        f_val = feat.GetFieldAsString(id_field_name)
        feat_dict[f_val] = feat
        feat_geom = feat.GetGeometryRef()
        feat_area_dict[f_val] = feat_geom.Area()

        feat = lyr.GetNextFeature()

    wat_ids = np.loadtxt(in_id_file, delimiter=in_sep, dtype='int64')
    wat_ids = np.atleast_2d(wat_ids)

    us_cats_dict = {}
    for i in range(wat_ids.shape[0]):
        us_cats_dict[str(wat_ids[i, 0])] = []

    for i in range(wat_ids.shape[0]):
        us_cat_no = str(wat_ids[i, 0])
        ds_cat_no = str(wat_ids[i, 1])
        if ds_cat_no == '-1':
            continue
        try:
            us_cats_dict[ds_cat_no] = us_cats_dict[ds_cat_no] + [us_cat_no]
        except KeyError:
            print('Unknown catchment:', ds_cat_no)
            print('replacing this name with -1 in the wat_ids arr')
            wat_ids[i, 1] = -1
            continue

    while True:
        cond_changed = False
        for i in range(wat_ids.shape[0]):
            us_cat_no = str(wat_ids[i, 0])
            ds_cat_no = str(wat_ids[i, 1])
            if ds_cat_no == '-1':
                continue

            if us_cat_no not in us_cats_dict[ds_cat_no]:
                continue

            for cat in us_cats_dict[us_cat_no]:
                if cat in us_cats_dict[ds_cat_no]:
                    continue

                cond_changed = True
                us_cats_dict[ds_cat_no] = us_cats_dict[ds_cat_no] + [cat]

        if not cond_changed:
            break

    driver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(out_cat_shp):
        driver.DeleteDataSource(out_cat_shp)

    out_ds = driver.CreateDataSource(out_cat_shp)
    out_lyr = out_ds.CreateLayer('0', geom_type=ogr.wkbMultiPolygon)

    in_lyr_dfn = lyr.GetLayerDefn()
    for i in range(0, in_lyr_dfn.GetFieldCount()):
        field_dfn = in_lyr_dfn.GetFieldDefn(i)
        out_lyr.CreateField(field_dfn)

    out_lyr_dfn = out_lyr.GetLayerDefn()

    for key in list(us_cats_dict.keys()):
        cat_nos = us_cats_dict[key]
        if len(cat_nos) > 0:
            # print 'cat: %s, has these upstream:' % key, cat_nos
            ds_cat_feat = feat_dict[key].Clone()
            for cat_no in cat_nos:
                ds_cat = ds_cat_feat.GetGeometryRef()
                curr_cat_feat = feat_dict[cat_no].Clone()
                curr_cat = curr_cat_feat.GetGeometryRef()

                merged_cat = ds_cat.Union(curr_cat)
                merged_cat_feat = ogr.Feature(out_lyr_dfn)
                merged_cat_feat.SetField(id_field_name, key)
                merged_cat_feat.SetGeometry(merged_cat)
                ds_cat_feat = merged_cat_feat

            feat_dict[key] = merged_cat_feat

    out_cat_df = pd.DataFrame(index=list(feat_dict.keys()),
                              columns=['diff_area', 'cumm_area'])

    for key in list(feat_dict.keys()):
        cat_feat = feat_dict[key]
        out_feat = ogr.Feature(out_lyr_dfn)
        geom = cat_feat.GetGeometryRef()
        out_feat.SetGeometry(geom)
        out_cat_df.loc[key, 'cumm_area'] = geom.Area()
        out_cat_df.loc[key, 'diff_area'] = feat_area_dict[key]
        for i in range(0, out_lyr_dfn.GetFieldCount()):
            out_feat.SetField(out_lyr_dfn.GetFieldDefn(i).GetNameRef(),
                              cat_feat.GetField(i))
        out_lyr.CreateFeature(out_feat)

    out_prj = open((out_cat_shp.rsplit('.', 1)[0] + '.prj'), 'w')
    out_prj.write(spt_ref.ExportToWkt())
    out_prj.close()
    cat_vec.Destroy()
    out_ds.Destroy()

    out_cat_df.to_csv(out_descrip_file, sep=out_sep, index_label='cat_no')
    return


if __name__ == '__main__':

    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    start = timeit.default_timer()  # to get the runtime of the program

    main_dir = r'P:\Synchronize\IWS\2016_DFG_SPATE\QGIS_SPATE\raster\taudem_out'

    in_cat_shp = r'watersheds.shp'
    in_id_file = r'watersheds_id.txt'
    in_sep = ' '
    out_sep = ';'
    id_field_name = 'DN'
    out_cat_shp = r'watersheds_cumm.shp'
    out_descrip_file = r'diff_cumm_cat_areas.csv'

    os.chdir(main_dir)

    get_cumm_cats(in_cat_shp,
                  id_field_name,
                  in_id_file,
                  in_sep,
                  out_cat_shp,
                  out_descrip_file,
                  out_sep)

    stop = timeit.default_timer()  # Ending time
    print('\n\a\a\a Done with everything on %s. Total run time was about %0.4f seconds \a\a\a' % (time.asctime(), stop - start))

