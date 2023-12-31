"""
Title: Translate Resampled Directories To Data Table
Created on Mon Jul 17 17:53:39 2023
@author: nccru
"""

# %% import packages
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.decomposition import PCA

# %% extract each image into a new file
def read_path_assignments_to_tuple_list(filepath = './chest_xray/xray_balanced_data_df.csv', image_size = 150):
    """
    Description 
    -------
    Takes the file generated from the resample_images.py file and creates
    a series of three lists of tuples. Each tuple element is a list of 
    [set, patient id, class id, vectorized image].
    
    Inputs
    -------
    `filepath` = Path to the csv file output from the resample_images.py file
    `image_size` = n x n size of the compressed image we want for the sake of
        standardization, as each image may have different baseline dimensions
    
    Returns
    -------
    `train` = list of tuples containing records associated with the training
        data set.
        
    `test` = list of tuples containing records associated with the testing
        data set.
        
    `val` = list of tuples containing records associated with the validation
        data set.
    """
    # read data
    df = pd.read_csv(filepath)
    
    # iterate through each row
    train, test, val = [], [], []
    for i in df.itertuples(index = False):
        # print('--- Patient {} ---'.format(i.patient))
        # extract file paths
        norm_paths = i.normal.strip('[]').split(',')
        bac_paths = i.bacteria.strip('[]').split(',')
        vir_paths = i.virus.strip('[]').split(',')
        
        # read in the images
        def read_image(img_path, img_size = image_size):
            p = re.sub(r"'", '', img_path).strip() # clean up path name
            img_arr = cv2.imread(p, cv2.IMREAD_GRAYSCALE) 
            img_resize = cv2.resize(img_arr, (img_size, img_size)).flatten()
            img_resize_scaled = img_resize / 255
            return img_resize_scaled
        
        def store_image(img_arr, class_id):
            if i.set == 'train':
                train.append((i.set, i.patient, class_id, img_arr))
            elif i.set == 'test':
                test.append((i.set, i.patient, class_id, img_arr))
            else:
                val.append((i.set, i.patient, class_id, img_arr))
            
        ## normal
        for path in norm_paths:
            # print(path)
            if len(path) > 0:
                img = read_image(path)
                store_image(img, 'normal')
        
        ## bacteria
        for path in bac_paths:
            if len(path) > 0:
                img = read_image(path)
                store_image(img, 'bacteria')
                
        ## virus
        for path in vir_paths:
            if len(path) > 0:
                img = read_image(path)
                store_image(img, 'virus')
    
    return train, test, val

# %% convert list of tuples into relevant shapes for plotting
def convert_tuple_list_to_sklearn_objects(train, test, val):
    """
    Description
    -------
    Given a collection of tuple lists [train, test, val], join them all
    together and reshape them meaningfully.
    
    Inputs 
    -------
    `train`, `test`, `val` = Tuple lists from the function 
        read_path_assignments_to_tuple_list
    

    Returns
    -------
    `sets` = An array of defining which set each sample belongs to.
    `y` = The labels for each vectorized image.
    `X` = The vectorized images

    """
    t = train + test + val # stack all lists together
    sets = [set_id for (set_id, patient_id, class_id, arr) in t]
    y = [class_id for (set_id, patient_id, class_id, arr) in t]
    X = np.stack([arr for (set_id, patient_id, class_id, arr) in t])
    
    return sets, y, X

# %% perform PCA on X to reduce dimensionality
def pca_and_convert_to_pandas(X, y, sets, pca_dim = 20):
    """
    Description
    -------
    Given a set of vectorized images `X`, their class labels `y`, and which 
    `sets` they belong to, perform PCA to reduce the dimensionality down to
    `pca_dim`. Then store that as a df.
    
    Note that we have to generate the PCA eigenvalues and eigenvectors on the 
    training data set, then utilize those to transform the test and val sets

    Inputs
    -------
    `X` = The vectorized image array
    `y` = The class the image is associated with
    `sets` = The [train, test, val] assignements
    `pca_dim` = The number of dimensions to reduce to

    Returns
    ------
    `df_pca` = The pandas data frame representation of reduced images.

    """
    # === slice out train test split ===
    
    ## construct pandas dataframe for easier slicing
    df = pd.DataFrame(X)
    df['y'] = y
    df['set'] = sets 

    ## slice out the sets
    X_train = df[df['set'] == 'train'].drop(['y', 'set'], axis = 1).to_numpy()
    y_train = df[df['set'] == 'train']['y'].reset_index(drop = True)
    X_test = df[df['set'] == 'test'].drop(['y', 'set'], axis = 1).to_numpy()
    y_test = df[df['set'] == 'test']['y'].reset_index(drop = True)
    X_val = df[df['set'] == 'validate'].drop(['y', 'set'], axis = 1).to_numpy()
    y_val = df[df['set'] == 'validate']['y'].reset_index(drop = True)

    # === perform PCA  on training component ===
    # === use that to transform test and val ===
    pca = PCA(n_components = pca_dim)
    X_train_pca = pca.fit_transform(X_train)  
    X_test_pca = pca.transform(X_test)
    X_val_pca = pca.transform(X_val)
    
    # === store to pandas dataframe ===
    colnames = ['pca' + str(i) for i in range(pca_dim)]

    ## train
    df_pca_train = pd.DataFrame(X_train_pca, columns = colnames)
    df_pca_train['y'] = y_train
    df_pca_train['set'] = 'train' 

    ## test
    df_pca_test = pd.DataFrame(X_test_pca, columns = colnames)
    df_pca_test['y'] = y_test
    df_pca_test['set'] = 'test' 

    ## val
    df_pca_val = pd.DataFrame(X_val_pca, columns = colnames)
    df_pca_val['y'] = y_val
    df_pca_val['set'] = 'validate' 

    ## merge all together
    df_pca = pd.concat([df_pca_train, df_pca_test, df_pca_val])\
        .reset_index(drop = True)

    return df_pca

# %%
# run through workflow and save to csv
train, test, val = read_path_assignments_to_tuple_list()
sets, y, X = convert_tuple_list_to_sklearn_objects(train, test, val)
final_df = pca_and_convert_to_pandas(X, y, sets)
final_df.to_csv('Pnuemonia Images PCA Reduced.csv')