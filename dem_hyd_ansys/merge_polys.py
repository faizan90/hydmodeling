'''
Created on Oct 8, 2017

@author: Faizan
'''
from os.path import exists as os_exists

from osgeo import ogr
from numpy import unique


def merge_same_id_shp_poly(in_shp, out_shp, field='DN'):

    '''Merge all polygons with the same ID in the 'field' (from TauDEM)

    Because sometimes there are some polygons from the same catchment,
    this is problem because there can only one cathcment with one ID,
    it is an artifact of gdal_polygonize.
    '''

    cat_ds = ogr.Open(in_shp)

    assert cat_ds is not None, (
        f'Unable to open file: {in_shp}!')

    lyr = cat_ds.GetLayer(0)
    spt_ref = lyr.GetSpatialRef()

    driver = ogr.GetDriverByName('ESRI Shapefile')
    if os_exists(out_shp):
        driver.DeleteDataSource(out_shp)

    feat_dict = {}
    fid_to_field_dict = {}

    feat = lyr.GetNextFeature()
    while feat:
        fid = feat.GetFID()
        f_val = feat.GetFieldAsString(field)
        feat_dict[fid] = feat.Clone()
        fid_to_field_dict[fid] = f_val
        feat = lyr.GetNextFeature()

    out_ds = driver.CreateDataSource(out_shp)
    out_lyr = out_ds.CreateLayer('0', geom_type=ogr.wkbMultiPolygon)

    in_lyr_dfn = lyr.GetLayerDefn()
    for i in range(in_lyr_dfn.GetFieldCount()):
        field_dfn = in_lyr_dfn.GetFieldDefn(i)
        out_lyr.CreateField(field_dfn)
    out_lyr_dfn = out_lyr.GetLayerDefn()

    uniq_vals = unique(list(fid_to_field_dict.values()))

    for uniq_val in uniq_vals:
        fid_list = []
        for fid in list(fid_to_field_dict.keys()):
            if fid_to_field_dict[fid] == uniq_val:
                fid_list.append(fid)

        if len(fid_list) > 1:
            cat_feat = feat_dict[fid_list[0]]
            for fid in fid_list[1:]:
                # The buffer with zero seems to fix invalid geoms somehow.
                ds_cat = cat_feat.GetGeometryRef().Buffer(0)
                curr_cat_feat = feat_dict[fid].Clone()
                curr_cat = curr_cat_feat.GetGeometryRef().Buffer(0)

                merged_cat = ds_cat.Union(curr_cat)
                merged_cat_feat = ogr.Feature(out_lyr_dfn)
                merged_cat_feat.SetField(field, uniq_val)
                merged_cat_feat.SetGeometry(merged_cat)
                cat_feat = merged_cat_feat
        else:
            cat = feat_dict[fid_list[0]]
            curr_cat_feat = ogr.Feature(out_lyr_dfn)
            curr_cat_feat.SetField(field, uniq_val)
            curr_cat_feat.SetGeometry(cat.GetGeometryRef().Buffer(0))
            cat_feat = curr_cat_feat

        for fid in fid_list:
            del feat_dict[fid]
            del fid_to_field_dict[fid]

        feat_dict[uniq_val] = cat_feat

    assert len(list(feat_dict.keys())) == len(uniq_vals), 'shit happend!'

    for key in list(feat_dict.keys()):
        cat_feat = feat_dict[key]
        out_feat = ogr.Feature(out_lyr_dfn)
        geom = cat_feat.GetGeometryRef()
        out_feat.SetGeometry(geom)
        for i in range(0, out_lyr_dfn.GetFieldCount()):
            out_feat.SetField(out_lyr_dfn.GetFieldDefn(i).GetNameRef(),
                              cat_feat.GetField(i))
        out_lyr.CreateFeature(out_feat)

    out_prj = open((out_shp.rsplit('.', 1)[0] + '.prj'), 'w')
    out_prj.write(spt_ref.ExportToWkt())
    out_prj.close()

    cat_ds.Destroy()
    out_ds.Destroy()
    return


if __name__ == '__main__':
    pass
