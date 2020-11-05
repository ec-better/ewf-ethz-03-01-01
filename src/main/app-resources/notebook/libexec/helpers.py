# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import re

from shapely import wkt
from shapely.geometry import box, Polygon
import pandas as pd
import geopandas as gpd

from osgeo import gdal, gdalnumeric, osr, ogr

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def getResolution(demFolder, return_full_paths = False):
    rasterFilePaths = [f for f in os.listdir(demFolder) if os.path.isfile(os.path.join(demFolder, f))]
    
    if return_full_paths:
        rasterFilePaths = [demFolder + '/' + f for f in rasterFilePaths if f[:4] == 'DEM_' and f[-4:] == '.tif']
        rasterFilePaths.sort(reverse=True)
    else:
        rasterFilePaths = [int(f[4:-4]) for f in rasterFilePaths if f[:4] == 'DEM_' and f[-4:] == '.tif']

    return rasterFilePaths

def readGDAL2numpy(rasterPath, return_geoInformation = False):
    try:
        ds = gdal.Open(rasterPath)
    except RuntimeError:
        print('Unable to open input file')
        sys.exit(1)
    
    data = gdalnumeric.LoadFile(rasterPath, False)
    noDataVal = ds.GetRasterBand(1).GetNoDataValue()
    try:
        if data.dtype in ['float16', 'float32', 'float64'] and noDataVal is not None:
            data[data == noDataVal] = np.NaN
    except:
        print("Issue in no data value")
        
        
    if return_geoInformation == False:
        return data
    else:
        geoTransform = ds.GetGeoTransform()
        projection = ds.GetProjection()    
        return data, geoTransform, projection

def writeNumpyArr2Geotiff(outputPath, data, geoTransform = None, projection = None, GDAL_dtype = gdal.GDT_Byte, noDataValue = None):
    nscn, npix = data.shape
    
    if np.isnan(data).any() and noDataValue is not None:
        data[np.isnan(data)] = noDataValue
        
    ds_new = gdal.GetDriverByName('GTiff').Create(outputPath, npix, nscn, 1, GDAL_dtype)
    
    if geoTransform != None:
        ds_new.SetGeoTransform(geoTransform)
        
    if projection != None:
        ds_new.SetProjection(projection)    
    
    outBand = ds_new.GetRasterBand(1)
    outBand.WriteArray(data)
    
    if noDataValue != None:
        ds_new.GetRasterBand(1).SetNoDataValue(noDataValue)
    
    # Close dataset
    ds_new.FlushCache()
    ds_new = None
    outBand = None
    
def writeNumpyArr2Saga(outputPath, data, geoTransform = None, projection = None, GDAL_dtype = gdal.GDT_Byte, noDataValue = None):
    nscn, npix = data.shape
    
    if np.isnan(data).any() and noDataValue is not None:
        data[np.isnan(data)] = noDataValue
        
    ds_new = gdal.GetDriverByName('SAGA').Create(outputPath, npix, nscn, 1, GDAL_dtype)  
    
    outBand = ds_new.GetRasterBand(1)
    outBand.WriteArray(data)
    
    if noDataValue != None:
        ds_new.GetRasterBand(1).SetNoDataValue(noDataValue)
    
    if projection != None:
        ds_new.SetProjection(projection)  
    
    # Close dataset
    ds_new.FlushCache()
    ds_new = None
    outBand = None

def wkt2bbox(wkt_input):
    wkt_geometry = wkt.loads(wkt_input)
    minx, miny, maxx, maxy = wkt_geometry.bounds
    b = box(minx, miny, maxx, maxy)
    bbox_tuple = list(b.exterior.coords)
    bbox = []
    for point in bbox_tuple:
        bbox.append([point[0],point[1]])
    return bbox

def wkt2shp(wkt_input, target_epsg, dst_file, bbox=False):
    ensure_dir(dst_file)
    if bbox:
        polygon = Polygon(wkt2bbox(wkt_input))
    else:
        polygon = wkt.loads(wkt_input)
    gpd.GeoDataFrame(pd.DataFrame(['p1'], columns = ['geom']),
                     crs = {'init':'epsg:' + str(target_epsg)},
                     geometry = [polygon]).to_file(dst_file)
    
def rescaleDEM(image, noData = None, maxVal = 255):
    if noData:
        image = np.float32(image)
        image[image == noData] = np.nan
        
    minElev = np.nanmin(image)
    maxElev = np.nanmax(image)
    
    rescaled = ( ((image - minElev)/(maxElev- minElev)) * (maxVal - 1) ) + 1
    return np.uint8(rescaled)

def joinStrArg(str1, str2, str3 = None):
    if str3 is not None:
        return str(str1) + ' ' + str(str2) + ' ' + str(str3)
    else:
        return str(str1) + ' ' + str(str2) 

def wkt2EPSG(wkt, epsg='/usr/local/share/proj/epsg', forceProj4=False):
    ''' 
    Transform a WKT string to an EPSG code
    
    Arguments
    ---------
    
    wkt: WKT definition
    epsg: the proj.4 epsg file (defaults to '/usr/local/share/proj/epsg')
    forceProj4: whether to perform brute force proj4 epsg file check (last resort)
    
    Returns: EPSG code
    
    '''
    code = None
    p_in = osr.SpatialReference()
    s = p_in.ImportFromWkt(wkt)
    if s == 5:  # invalid WKT
        return None
    if p_in.IsLocal() == 1:  # this is a local definition
        return p_in.ExportToWkt()
    if p_in.IsGeographic() == 1:  # this is a geographic srs
        cstype = 'GEOGCS'
    else:  # this is a projected srs
        cstype = 'PROJCS'
    an = p_in.GetAuthorityName(cstype)
    ac = p_in.GetAuthorityCode(cstype)
    if an is not None and ac is not None:  # return the EPSG code
        return '%s:%s' % \
            (p_in.GetAuthorityName(cstype), p_in.GetAuthorityCode(cstype))
    else:  # try brute force approach by grokking proj epsg definition file
        p_out = p_in.ExportToProj4()
        if p_out:
            if forceProj4 is True:
                return p_out
            f = open(epsg)
            for line in f:
                if line.find(p_out) != -1:
                    m = re.search('<(\\d+)>', line)
                    if m:
                        code = m.group(1)
                        break
            if code:  # match
                return 'EPSG:%s' % code
            else:  # no match
                return None
        else:
            return None
        
def getCornerCoordinates(gdal_dataSet, target_srs = False):
    """
    :param gdal_dataSet: /path/to/file   OR    gdal dataset
    :param target_srs: False for output coordinates in same coordinate system   OR   'wgs84' for lat long values   OR    custom osr.SpatialReference() object
    :return: list of corner coordinates

                --0--------3--
                  |        |
                  |        |        <--- Index of coordinates returned in list
                  |        |
                --1--------2--
    """


    if type(gdal_dataSet) is str:
        gdal_dataSet = gdal.Open(gdal_dataSet)

    gt=gdal_dataSet.GetGeoTransform()   # gt = [ulx, xres, xskew, uly, yskew, yres]
    cols = gdal_dataSet.RasterXSize
    rows = gdal_dataSet.RasterYSize

    def GetExtent(gt,cols,rows):
        ''' Return list of corner coordinates from a geotransform
            @type gt:   C{tuple/list}
            @param gt: geotransform
            @type cols:   C{int}
            @param cols: number of columns in the dataset
            @type rows:   C{int}
            @param rows: number of rows in the dataset
            @rtype:    C{[float,...,float]}
            @return:   coordinates of each corner
        '''
        ext=[]
        xarr=[0,cols]
        yarr=[0,rows]

        for px in xarr:
            for py in yarr:
                x=gt[0]+(px*gt[1])+(py*gt[2])
                y=gt[3]+(px*gt[4])+(py*gt[5])
                ext.append([x,y])
                #print(x,y)
            yarr.reverse()
        return ext

    def ReprojectCoords(coords,src_srs,tgt_srs):
        ''' Reproject a list of x,y coordinates.

            @type geom:     C{tuple/list}
            @param geom:    List of [[x,y],...[x,y]] coordinates
            @type src_srs:  C{osr.SpatialReference}
            @param src_srs: OSR SpatialReference object
            @type tgt_srs:  C{osr.SpatialReference}
            @param tgt_srs: OSR SpatialReference object
            @rtype:         C{tuple/list}
            @return:        List of transformed [[x,y],...[x,y]] coordinates
        '''
        trans_coords=[]
        transform = osr.CoordinateTransformation( src_srs, tgt_srs)
        for x,y in coords:
            x,y,z = transform.TransformPoint(x,y)
            trans_coords.append([x,y])
        return trans_coords

    ext = GetExtent(gt,cols,rows)

    src_srs=osr.SpatialReference()
    src_srs.ImportFromWkt(gdal_dataSet.GetProjection())

    if target_srs == False:
        return ext
    elif target_srs == 'wgs84':
        #target_srs = src_srs.CloneGeogCS()
        #
        target_srs=osr.SpatialReference()
        target_srs.ImportFromEPSG(4326)

    return ReprojectCoords(ext,src_srs,target_srs)

def resizeToDEM(imPath, sizeDEM = None, geoTransform = None, projection = None, noData = None):
    imDS = gdal.Open(imPath, gdal.GA_ReadOnly)
    imPix = imDS.RasterXSize
    imScn = imDS.RasterYSize
    
    nscn, npix = sizeDEM
    
    if sizeDEM is not None:
        if nscn != imScn or npix != imPix:
            print("Size Mismatch")
            image = imDS.ReadAsArray()
            if noData is not None:
                image = np.float32(image)
                image[image == noData] = np.nan
            imNew = cv2.resize(image, (npix, nscn), interpolation=cv2.INTER_CUBIC)
            
            writeNumpyArr2Geotiff(imPath, imNew, geoTransform = geoTransform, projection = projection, GDAL_dtype = gdal.GDT_UInt16, noDataValue = noData)
            
def map_uint16_to_uint8(img, lower_bound=None, upper_bound=None):
    '''
    Map a 16-bit image trough a lookup table to convert it to 8-bit.

    '''
    if not(0 <= lower_bound < 2**16) and lower_bound is not None:
        raise ValueError(
            '"lower_bound" must be in the range [0, 65535]')
    if not(0 <= upper_bound < 2**16) and upper_bound is not None:
        raise ValueError(
            '"upper_bound" must be in the range [0, 65535]')
    if lower_bound is None:
        lower_bound = np.min(img)
    if upper_bound is None:
        upper_bound = np.max(img)
    if lower_bound >= upper_bound:
        raise ValueError(
            '"lower_bound" must be smaller than "upper_bound"')
    lut = np.concatenate([
        np.zeros(lower_bound, dtype=np.uint16),
        np.linspace(0, 255, upper_bound - lower_bound).astype(np.uint16),
        np.ones(2**16 - upper_bound, dtype=np.uint16) * 255
    ])
    return lut[img].astype(np.uint8) 

def closeCV(mask, kernelSize = 11):
    kernel = np.ones((kernelSize, kernelSize),np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

def newGeoTransform(geoTransform, maskBounds):
	newGeoTransform = (geoTransform[0]+ maskBounds['xMin'] * geoTransform[1],
                   geoTransform[1],
                   geoTransform[2],
                   geoTransform[3] + maskBounds['yMin'] * geoTransform[5],
                   geoTransform[4],
                   geoTransform[5])  
	return newGeoTransform

def shrinkGeoTransform(geoTransform, factor):
	newGeoTransform = (geoTransform[0],
                   geoTransform[1] / factor,
                   geoTransform[2],
                   geoTransform[3],
                   geoTransform[4],
                   geoTransform[5] / factor)  
	return newGeoTransform
