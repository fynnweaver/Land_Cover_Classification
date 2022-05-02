#import
import numpy as np
import pandas as pd

#system packages
import os
import glob

#loading images
from osgeo.gdalconst import *
from osgeo import gdal
from scipy.ndimage import gaussian_filter, median_filter

from copy import deepcopy


#--IMPORT DATA--

#Turn raws into dataframe from directory path using gdal
def gdal_to_dataframe(dir_path, nrcan_name = 'NRCAN_transformed.tif', index = [-14, -11]):
    raw_names = list(os.listdir(dir_path))
    
    raw_df = pd.DataFrame()
    
    for i in raw_names:
        raw_img = gdal.Open(os.path.join(dir_path, i))
        
        raw_array = np.array(raw_img.ReadAsArray()).flatten()
        raw_df[i[index[0]:index[1]]] = raw_array
    
    nrcan = gdal.Open(os.path.join(dir_path, '..', nrcan_name))
    nrcan_array = np.array(nrcan.ReadAsArray())
    
    raw_df['y'] = nrcan_array.flatten()
    
    return raw_df

# --FEATURE SELECTION--

#Filter photos (gaussian or median) from directory path
def filter_raws(path_to_dir,  sigma = 5, nrcan_name = 'NRCAN_transformed.tif', index = [-14, -11], filter_type = 'gaussian'):
    raw_files = list(os.listdir(path_to_dir))

    filter_df = pd.DataFrame()
    for i in raw_files[:]:
        raw_img = gdal.Open(os.path.join(path_to_dir, i))
        rows = raw_img.RasterYSize
        cols = raw_img.RasterXSize
        
        raw_array = np.array(raw_img.ReadAsArray())
        
        if filter_type == 'gaussian':
            raw_filter = gaussian_filter(raw_array, sigma = sigma)
        elif filter_type == 'median':
            raw_filter = median_filter(raw_array, size = 10)
        
        raw_filter = raw_filter.flatten()
        filter_df[i[index[0]:index[1]]] = raw_filter
    
    nrcan = gdal.Open(os.path.join(path_to_dir, '..', nrcan_name))
    nrcan_array = np.array(nrcan.ReadAsArray())
    
    filter_df['y'] = nrcan_array.flatten()
    return filter_df


def outlier_fix(dataframe):
    
    new_frame = pd.DataFrame()
    
    for column in dataframe.columns:
        
        #make sure we don't transform y column
        if column != 'y':
            #create deepcopy to change
            temp_band = deepcopy(dataframe.loc[:,column].values)
            outlier = np.quantile(temp_band, q = 0.75) * 2
              
            #replace any above outlier with mean    
            temp_band[temp_band > outlier] = np.mean(temp_band)
            new_frame[f"{column}f"] = temp_band
    #if dataframe has a y column add back in    
    try:
        new_frame['y'] = dataframe['y']
        return new_frame
    #if not (ex: its X dataframe) just return frame
    except:
        return new_frame
        
def add_layers(df):
    """
    This function takes in a dataframe and calculates the NDVI, Moisture Index, NDWI and NDSI
    Outputs = dataframe with added layer columns 
    """
    #Create NDVI column (B08-B04)/(B08+B04)
    df['NDVI'] = (df.B08 - df.B04)/(df.B08 + df.B04)
    #Create Moisture index (B8A-B11)/(B8A+B11)
    df['Moisture'] = (df.B8A - df.B11)/(df.B8A + df.B11)
    #Create NDWI (B3-B8)/(B3+B8)
    df['NDWI'] = (df.B03 - df.B08)/(df.B03 + df.B08)
    #create NDSI (B3-B11)/(B3+B11)
    df['NDSI'] = (df.B03 - df.B11)/(df.B03 + df.B11)

   
    return df

#Calculated layers
def add_extra_layers(df):
    """
    This function takes in a dataframe and calculates an extra five layers
    Outputs = dataframe with added layer columns 
    """
     #normalized NIR/Blue normalized veg index
    df['NIRB'] = (df.B08 - df.B02)/(df.B08 + df.B02)
    #green normalized difference veg index
    df['NIRB'] = (df.B08 - df.B03)/(df.B08 + df.B03)
    #Atmospheric Resistant Green
    df['ARG'] = (df.B03 - df.B04)/(df.B03 + df.B04)
    # yellow veg index
    df['yellow'] = (0.723 * df.B03) - (0.597 * df.B04) + (0.206 * df.B06) - (0.278 * df.B09)
    #Mid-infrared veg index
    df['MIVI'] = df.B09/df.B11
    #GDVI
    df['GDVI'] = df.B08 - df.B03

    
def replace_values(df):
    """
    This function replaces the infinity values with Nan then replaces that with new infinity values
    """
    #Replace infinity values with Nan
    df.replace([np.inf, -np.inf], np.NAN, inplace=True)

    #Fill in null values
    df.fillna(999, inplace=True)

    return df

#--EVALUATION--

def print_importance(model, x):
    """
    Input: Model and X prior to split]
    Output: Series of feature importance coefficients in descending order
    """
    feature_importances = pd.DataFrame(model.feature_importances_, index = x.columns, columns = ['importance']).sort_values('importance', ascending = False)
    print(feature_importances)