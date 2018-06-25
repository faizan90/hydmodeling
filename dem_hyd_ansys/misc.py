'''
Created on Oct 8, 2017

@author: Faizan, IWS Uni-Stuttgart
'''
import os

import gdal
import ogr
import numpy as np


def list_full_path(ext, file_dir):
    """
    Purpose: To return full path of files in a given folder with a \
            given extension in ascending order.
    Description of the arguments:
        ext (string) = Extension of the files to list \
            e.g. '.txt', '.tif'. \n\n
        file_dir (string) = Full path of the folder in which the files \
            reside.
    """
    new_list = []
    for elm in os.listdir(file_dir):
        if elm[-len(ext):] == ext:
            new_list.append(file_dir + '/' + elm)
    return(sorted(new_list))


def get_ras_as_array_GDAL(in_ras, in_band_no=1):
    """
    Purpose: To return the given raster's band as an array.
    Description of arguments:
        inRas (string): Full path to the input raster.
        in_band_no (int): The band number which has to be read.
    """
    in_ds = gdal.Open(in_ras, 0)
    if in_ds == None:
        raise IOError('Could not open %s for reading' % in_ras)
        return
    else:
        in_band = in_ds.GetRasterBand(in_band_no)
        rows = in_ds.RasterYSize
        cols = in_ds.RasterXSize
        ras_arr = np.array(in_band.ReadAsArray(0, 0, cols, rows))
        in_ds = None
        return ras_arr


def get_ras_props(in_ras, in_band_no=1):
    """
    Purpose: To return a given raster's extents, number of rows and columns, \
                        pixel size in x and y direction, projection, noData value, \
                        band count and GDAL data type using GDAL as a list.
    Description of the arguments:
        in_ras (string): Full path to the input raster. If the raster cannot be \
                        read by GDAL then the function returns None.
        in_band_no (int): The band which we want to use (starting from 1). \
                        Defaults to 1. Used for getting NDV.
    """
    in_ds = gdal.Open(in_ras, 0)
    if in_ds is not None:
        rows = in_ds.RasterYSize
        cols = in_ds.RasterXSize

        geotransform = in_ds.GetGeoTransform()
        x_min = geotransform[0]
        y_max = geotransform[3]

        pix_width = geotransform[1]
        pix_height = abs(geotransform[5])

        x_max = x_min + (cols * pix_width)
        y_min = y_max - (rows * pix_height)

        proj = in_ds.GetProjectionRef()

        in_band = in_ds.GetRasterBand(in_band_no)
        if in_band is not None:
            NDV = in_band.GetNoDataValue()
            gdt_type = in_band.DataType
        else:
            NDV = None
            gdt_type = None
        band_count = in_ds.RasterCount

        ras_props = [x_min, x_max, y_min, y_max, cols, rows, pix_width,
                     pix_height, proj, NDV, band_count, gdt_type]

        in_ds = None
        return ras_props
    else:
        raise RuntimeError(('Could not read the input raster (%s). Check path '
                            'and file!') % in_ras)
    return


# Need to change this. Layer and Features in a layer are different
def get_vec_props(inVecFile, layerIndex=None, layerName=None):
    """
    Purpose: To return the extents and spatial reference of a given layer \
                    (point, line, polygon) in a given ogr vector file as a list.
    Description of the arguments:
        inVecFile (string): Full path to the vector file.
        layerIndex (int): The index of the layer(starting from 0) \
                    inside the vector file. The user can specify either \
                    layerIndex or layerName.
        layerName (string): The name of the layer inside the vector file to get the properties for.
        Note: if the specified types of arguments are not used then None returned. \
                replaces the lyrNameAsStr(vecFilePath) function.
    """
    inVec = ogr.Open(inVecFile, 0)
    if inVec is not None:
        if type(layerIndex) == type(int()):
            layer = inVec.GetLayer(layerIndex)
        elif type(layerName) == type(str()):
            layer = inVec.GetLayerByName(layerName)
        else:
            raise RuntimeError(('Cannot get the specified layer in %s. The '
                                'layerIndex argument should be of integer '
                                'type or the layerName argument should have '
                                'string type.') % inVecFile)
        if layer is not None:
            geotransform = layer.GetExtent()
            xMin = geotransform[0]
            yMax = geotransform[3]
            xMax = geotransform[1]
            yMin = geotransform[2]
            spRef = layer.GetSpatialRef().ExportToWkt()

            vecProps = [xMin, xMax, yMin, yMax, spRef]
            inVec.Destroy()
            return vecProps
        else:
            raise RuntimeError(('Could not read the input vector file (%s) '
                                'for the given layer index or layer name.') %
                                inVecFile)
    return
