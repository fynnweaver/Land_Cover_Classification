#import
import numpy as np
import pandas as pd

#system packages
import os
import glob
import pickle

#loading images
from osgeo.gdalconst import *
from osgeo import gdal
from affine import Affine

from scipy.ndimage import gaussian_filter, median_filter
from skimage import feature

from copy import deepcopy


#--IMPORT DATA--

#Turn raws into dataframe from directory path using gdal
def gdal_to_dataframe(dir_path, nrcan_name = 'NRCAN_transformed.tif', index = [-14, -11], calculate_edge = None, sigma = 3):
    """
    Use gdal to create a labelled dataframe of Sentinal-2 band values from a folder of raw band geotiffs using land cover
    classification from NRCAN. NRCAN extent must be in directory above raw directory. It should be processed using GIS to be 
    the same width, height, projection and resolution as clipped raws extents.

    INPUT
    -----
    `dir_path`: string with path to the raw files directory
    `nrcan_name`: file name of the accompanying nrcan classification extent. File must be in directory above raw directory
    `index`: Index within raw band file names where the band name is found. For files downloaded directly from EO browser
    without being renamed this will be [-14, -11], hence it is the default
    `calculate_edge`: Intakes the string of a raw band name on which to perform sklearn canny edge detection.
    Returns the feature edge. Default set to None where it will not add this feature.
    `sigma`: Modifier for the canny edge detection. If edge detection is performed sigma is for canny is inputted sigma.
    Defaults to 3
    
    OUTPUT
    ------
    Pandas DataFrame containing a column for each raw band within input directory. 
    If calculate_edge is not None dataframe includes column 'edge'
    """
    
    raw_names = list(os.listdir(dir_path))
    
    raw_df = pd.DataFrame()
    
    for i in raw_names:
        raw_img = gdal.Open(os.path.join(dir_path, i))
        raw_array = np.array(raw_img.ReadAsArray())
        band = i[index[0]:index[1]]
        raw_df[band] = raw_array.flatten()
        
        #if calculate edge is equal to band name, take that band image and get canny edge
        if calculate_edge == band:
            print('getting edge')
            edge = feature.canny(raw_array, sigma = sigma)
            edge = edge.astype(int)
    
    try:
        raw_df['edge'] = edge.flatten()
        raw_df['edge'] = raw_df['edge'].astype('int')
    except:
        pass
    
    nrcan = gdal.Open(os.path.join(dir_path, '..', nrcan_name))
    nrcan_array = np.array(nrcan.ReadAsArray())
    
    raw_df['y'] = nrcan_array.flatten()
    
    return raw_df

# --FEATURE SELECTION--

#Filter photos (gaussian or median) from directory path
def filter_raws(path_to_dir,  sigma = 5, nrcan_name = 'NRCAN_transformed.tif', index = [-14, -11], filter_type = 'gaussian'):
    """
    Use gdal to create a labelled dataframe of filtered Sentinal-2 band values from a folder of raw band geotiffs 
    Classification comes from NRCAN extent which must be processed using GIS to be the same width, height, projection and
    resolution as clipped raws extents. NRCAN extent must be in directory above raw directory.

    INPUT
    -----
    `path_to_dir`: string with path to the raw files directory
    `sigma`: Modifies the filtering strength. 
    Varies depending on transformation but generally a higher sigma means a more transformed image.
    `nrcan_name`: file name of the accompanying nrcan classification extent. File must be in directory above raw directory
    `index`: Index within raw band file names where the band name is found. For files downloaded directly from EO browser
    without being renamed this will be [-14, -11], hence it is the default
    `filter_type`: defines filter to apply. There are two current valid inputs: 'gaussian' or 'median'.
    Each apply the sklearn implementation of the named image filter

    
    OUTPUT
    ------
    Pandas DataFrame containing a column for each raw band within input directory with values having been transformed
    by defined filter (gaussian or median). 
    """
    
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
    """
    Creates a deep copy of input dataframe where outliers of each have been converted to mean value.
    Outliers here are defined as two times the 75% quartile.
    
    INPUT
    ------
    `dataframe`: pandas dataframe containing raw band values. 
    Can contain other features as well, function will only effect columns begenning with 'B'
        
    OUTPUT
    ------
    Pandas DataFrame containing a column of transformed values for each band of input dataframe 
    """
    
    new_frame = pd.DataFrame()
    
    for column in dataframe.columns:
        
        #make sure we only transform band columns
        if column[0] == 'B':
            #create deepcopy to change
            temp_band = deepcopy(dataframe.loc[:,column].values)
            outlier = int(np.quantile(temp_band, q = 0.75)) * 2
              
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
    
    
def add_fix_column(raws, target):
    """
    Save just one column of outlier fixing applied to target input band onto input dataframe
    """
    raws[target] = outlier_fix(raws)[target]
    return raws
        
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
    #Atmospheric Resistant Green
    df['ARG'] = (df.B03 - df.B04)/(df.B03 + df.B04)
    # yellow veg index
    df['yellow'] = (0.723 * df.B03) - (0.597 * df.B04) + (0.206 * df.B06) - (0.278 * df.B09)
    #Mid-infrared veg index
    df['MIVI'] = df.B09/df.B11
    #GDVI
    df['GDVI'] = df.B08 - df.B03
    
    return df

    
def replace_values(df):
    """
    This function replaces the infinity values with Nan then replaces that with new infinity values
    """
    #Replace infinity values with Nan
    df.replace([np.inf, -np.inf], np.NAN, inplace=True)

    #Fill in null values
    df.fillna(999, inplace=True)

    return df

def get_geocoord(raws_path):
    """
    Generates latitude and longitude geocoordinates from the first raw tiff within the inputted raw directory
    
    INPUT
    ------
    `raws_path`: string with path to the raw files directory
    
    OUTPUT
    ------
    lat, long: Two np arrays containing the latitude and longitude of the raws within the input directory path respectively 
    """
    #take first raw as example
    file_name = list(os.listdir(raws_path))[0]
    file_path = os.path.join(raws_path, file_name)
    
    #open raw and get affine
    ds = gdal.Open(file_path, gdal.GA_ReadOnly)
    T0 = Affine.from_gdal(*ds.GetGeoTransform())
    
    # get shape of rows and columns
    cols, rows = np.meshgrid(np.arange(ds.RasterYSize), np.arange(ds.RasterXSize))
    ds = None  # close
    
    # reference the pixel centre, so it needs to be translated by 50%:
    T1 = T0 * Affine.translation(0.5, 0.5)
    #transform from pixel coordinates to world coordinates, multiply the coordinates with the matrix
    rc2xy = lambda r, c: T1 * (c, r)
    
    # All eastings and northings -- this is much faster than np.apply_along_axis
    eastings, northings = np.vectorize(rc2xy, otypes=[float, float])(rows, cols)
    #convert to columns
    lat = eastings.flatten()
    long = northings.flatten()
    
    return lat, long

#combining all options for custom preprocess function
#combining all options for custom preprocess function
def process_data(path_csv, path_raws, nrcan_name = 'land_cover.tif', index = [0, 3], target_edge = False, edge_sigma = 3, geocoords = False,
                 target_outlier = False, gaussian = False, clustering = False, calculate_layers = False):
    """
    Master function that combines all above processing functions and applies to raws from input path to raw directory. 
    Features are generated based on inputted parameters and added as columns to outputted frame
    By default function function is equivalent to gdal_to_dataframe but with different default parameters
    
    INPUT
    ------
    `path_csv`: In the case of certain extents the NRCAN land cover tif was corrupted while a csv of correct land cover
    values remained. 
    If this is not the case, set path_csv to None and function will proceed with land cover from gdal_to_dataframe
    `path_raws`: string with path to the raw files directory
    `nrcan_name`: file name of the accompanying nrcan classification extent. File must be in directory above raw directory
    `index`: Index within raw band file names where the band name is found. 
    `target_edge`: If set equal to band name (ex: 'B01') will return canny edge (sigma 3) of that band under column labeled
    'edge'
    `geocoords`: If set to True will return lat and long geocoordinates
    `target_outlier`:  If set equal to band name (ex: 'B01') will return column of values where outliers have been
    transformed using `outlier_fix`
    Outlier fixed columns can be recognized with an appended `f` (ex: 'B01f`)
    `gaussian`: If set to true implements `raw_filters` and returns transformed columns to add on to original raw columns.
    Filtered columns can be recognized with an appended `g` (ex: 'B01g`)
    `clustering`: If set to string of path to a k-means clustering (.sav) model, performs k means clustering on raws and 
    returns column labeled 'cluster'
    `calculate_layers`: If set to True calculates 4 additional layers from raw values: NDVI, NDWI, Moisture and NDSI.
    If set to 'Extra' calculates 4 previous layers plus an additional 5 layers: NIRB, ARG, yellow, MIVI, GDVI
    
    
    OUTPUT
    ------
    X_demo, y_demo: Two pd DataFrames()
    \-> X_demo: contains all features generated by function based on input parameters
    \-> y_demo: contains associated labels of land cover classes
    """
    
    
    if path_csv is not None:
        #get y from csv and reshape
        raw = pd.read_csv(path_csv)
        raw.land_cover = raw.land_cover.astype('int')
        y_demo = raw['land_cover']
        #reshape to use y_demo with gaussian X
        y_demo = y_demo.values.reshape(2500, 2100).T
        y_demo = y_demo.flatten()
        
         #get X from gdal function
        raw = gdal_to_dataframe(path_raws, nrcan_name = nrcan_name, index = index, calculate_edge = target_edge, sigma = edge_sigma)
        X_demo = raw.drop('y', axis = 1)
    else:
        #get X from gdal function
        raw = gdal_to_dataframe(path_raws, nrcan_name = nrcan_name, index = index, calculate_edge = target_edge, sigma = edge_sigma)
        y_demo = raw.y
        X_demo = raw.drop('y', axis = 1)
    
    if geocoords is True:
        X_demo['lat'], X_demo['long'] = get_geocoord(path_raws)
    

    if target_outlier is not False:
        if target_outlier[0] == 'B':
            X_demo[target_outlier] = outlier_fix(X_demo)[target_outlier]
        else:
            X_demo = outlier_fix(X_demo)
    
    if gaussian == True:
        #filter raws from path
        gauss_demo = filter_raws(path_raws, nrcan_name = nrcan_name, index = index)

        #concat gauss and raw
        gauss_demo_reset = gauss_demo.drop('y', axis = 1)
        #rename gauss columns
        gauss_names_reset = [f'{name}g' for name in gauss_demo_reset.columns]
        gauss_demo_reset.columns = gauss_names_reset
        #reset indices
        raw.reset_index(inplace=True, drop=True)
        gauss_demo_reset.reset_index(inplace=True, drop=True)    
        merged_df = pd.concat([X_demo, gauss_demo_reset], axis = 1)

        #select X values from gaussian dataframe
        X_demo = merged_df

    if clustering is not False:
        param = pickle.load(open(clustering, 'rb'))
        demo_cluster = param.predict(X_demo.astype('double'))
        X_demo['clusters'] = demo_cluster
        
    if calculate_layers is not False:
        X_demo = add_layers(X_demo)
        X_demo = replace_values(X_demo)
        
        if calculate_layers == 'Extra':
            X_demo = add_extra_layers(X_demo)
        
    return X_demo, y_demo


#-- PREDICTION --

#creating a binary model: processing function for test and train data
def convert_binary(dataframe, target_class, col_name = 'y'):
    """
    Function to convert target land cover class from input dataframe into a One v. Rest binary dataset
    
    INPUT
    ------
    `dataframe`: pd DataFrame of raws, selected features, and labelled land cover column.
    `target_class`: int of target land cover class. It will be converted to 1 while all other classes are converted to 0
    `col_name`: Name of the labeled land cover column. Defaults to 'y'
    
    OUTPUT
    ------
    Pandas DataFrame with land cover classes converted into binary format
    """
    
    all_classes = list(range(1, 20))
    
    all_classes.remove(target_class)
    
    dataframe = dataframe.replace({col_name: all_classes}, 0)
    dataframe = dataframe.replace({col_name: target_class}, 1)
    
    return dataframe

#combining multiple models by updating a base model with specified classes of additional models
def predict_combo(models, path_csv, path_raws, process_dict, binary, index = [0, 3], nrcan_name = 'land_cover.tif'):
    
    """
    Function using master processing function to generate test predictions and then overlap the output of multiple models.
    Note that this uses predict (not predict_proba) and then overlies specified class of follow up onto a base model,
    this means the order of inputted models is very important.
    
    INPUT
    ------
    
    `models`: list of machine learning models that have already been trained. 
        \-> The first model in the list is the base model. 
        \-> The subsequent models reference the parameter `class_list` then update the base model in order
    `path_csv`: In the case of certain extents the NRCAN land cover tif was corrupted while a csv of correct land cover
    values remained. 
        \-> If this is not the case, set path_csv to None and function will proceed with land cover from gdal_to_dataframe
    `path_raws`: string with path to the raw files directory
    `process_dict`: Dictionary containing the associated processing parameter steps for each model.
        \-> See `process_data` function for more details.
        \-> One example input for a combination of 3 models is: 
                   process_dict = {
                        'target_edge': ['B8A', None, None],
                        'geocoords': [True, False, False],
                        'target_outlier' : [False, 'B01', False],
                        'gaussian' : [False, False, True],
                        'clustering' : [False, False, '../models/kcluster_13.sav'],
                        'calculate_layers' : [True, 'Extra', True]}
    `binary`: Dictionary containing information as to whether there is a binary or not. 
        \-> Should contain two keys `model` and `class`
        \-> `model`: list of the index of binary array. `class` should be a list of the target classes for each model
                (ie: the class that is 1)
        \-> Example: for an input where the second model is binary with a target class of 17:
                binary_dict = {'model': [1], 'class': [17]}
    `index`: Index within raw band file names where the band name is found. 
    `class_lists`: Nested list of the classes to take from each additional model. 
        - First list corresponds to second entered model, etc...
    `nrcan_name`: file name of the accompanying nrcan classification extent. File must be in directory above raw directory   
    
    OUTPUT
    ------
    base_pred, test_y: two np Arrays
        - base_pred: combined prediction output
        - test_y: the true values for the extent
    """
    
    pred_list = []
    
    
    
    for i in range(len(models)):
        test_X, test_y = process_data(path_csv, path_raws, index = index, 
                                      target_edge = process_dict['target_edge'][i],
                                      geocoords = process_dict['geocoords'][i],
                                      target_outlier = process_dict['target_outlier'][i],
                                      gaussian = process_dict['gaussian'][i],
                                      clustering = process_dict['clustering'][i],
                                      calculate_layers = process_dict['calculate_layers'][i], nrcan_name = nrcan_name)
        pred = pd.DataFrame(models[i].predict(test_X))
        pred_list.append(pred)
                
    return pred_list, test_y

                
def combine_pred(pred_list, class_lists, binary):
 
    bin_pred = pred_list[binary['model']]
    bin_pred = bin_pred.replace(1, binary['class'])
    
    #take first model as base model
    base_pred = pred_list[0]
    
    #select only classes in list from additional forest
    for i in range(len(class_lists)):
        subset = pred_list[i + 1][pred_list[i + 1].isin(class_lists[i]) ]
        #updat base_pred with subset
        base_pred.update(subset)
        #convert to int
        base_pred = base_pred.astype('int')
    
    return base_pred, test_y

#--EVALUATION--

def print_importance(model, x):
    """
    Return feature importance nicely formatted and sorted
    
    INPUT
    ------
    `model`: Trained classifier model
    `x`: dataframe containing columns classifier was trained on
    
    """
    feature_importances = pd.DataFrame(model.feature_importances_, index = x.columns, columns = ['importance']).sort_values('importance', ascending = False)
    print(feature_importances)