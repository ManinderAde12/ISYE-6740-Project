# file: image_organizer
# language: Python 3.9
# purpose: create dataframe that organizes images by patient and then type (normal, virus, bacteria)
#          and then redistributes into more balanced train/test/validate subsets
#          based on patient so that all patient images are in the same subset (train/test/validate)
# NOTES: does not read in or process images, just organizes file handles
#        assumes image files are loaded in subdirectory "chest_xray" under directory with this program

# imports
import numpy as np
import pandas as pd
from random import sample
import matplotlib.image as mpimg
import os
import re

base_dir = "./chest_xray/"
sub_dirs = ["test/NORMAL",
            "test/PNEUMONIA",
            "train/NORMAL",
            "train/PNEUMONIA",
            "val/NORMAL",
            "val/PNEUMONIA"]

image_df = pd.DataFrame(columns=["source_file", "source_dir"], dtype="str")
for sub_dir in sub_dirs:
    source_dir = base_dir + sub_dir
    files = os.listdir(base_dir+sub_dir)
    temp_df = pd.DataFrame(files, columns=["source_file"])
    temp_df["source_dir"] = source_dir
    image_df = pd.concat([image_df, temp_df], ignore_index=True)
image_df = image_df.sort_values(["source_dir", "source_file"])

# set image type: normal, bacterial penumonia or viral pneumonia
types = ["normal", "bacteria", "virus"]
image_df["type"] = "normal"
image_df["type"] = np.where(image_df["source_file"].str.contains("virus"), "virus", image_df["type"])
image_df["type"] = np.where(image_df["source_file"].str.contains("bacteria"), "bacteria", image_df["type"])
for type in types:
    type_df = image_df[image_df.type == type]

# extract patient number from filename
def extract_patient(source_file):
    patient = ""
    if "virus" in source_file:
        patient = str.split(source_file, "_")[0][6:]
    if "bacteria" in source_file:
        patient = str.split(source_file, "_")[0][6:]
    if "NORMAL" in source_file:
        patient = str.split(source_file, "-")[2]
    else:
        if "IM" in source_file:
            patient = str.split(source_file, "-")[1]
    return(int(patient))

image_df["patient"] = image_df["source_file"].apply(extract_patient)
# save copy of intermediate dataframe
image_df.to_csv('./chest_xray/image_df.csv')

# create bucketized dataframe
# where each row represents a patient
# and their are separate columns for images and image counts by type
patients = list(image_df.patient.unique())
xray_data = []
for patient in patients:
    bytype = []
    for type in types:
       subset = image_df[(image_df.patient == patient) & (image_df.type == type)].copy()
       # subset["source"] = subset.source_file
       subset["source"] = subset.source_dir + "/" + subset.source_file
       bytype.append(list(subset["source"]))
    xray_data.append((patient, bytype[0], bytype[1], bytype[2]))
xray_data_df = pd.DataFrame(xray_data, columns=["patient", "normal", "bacteria", "virus"])

# track count of images
def image_count(image_list):
    return len(image_list)
for type in types:
    xray_data_df[type+"_count"] = xray_data_df[type].apply(image_count)
xray_data_df.to_csv('./chest_xray/grouped_image_df.csv')

# partition patients between train, test & validation sets
# note the balancing won't be exact as the number of images of each type vary by patient
# but trials seem to show it provides a reasonable distribution
count = xray_data_df.shape[0]
# proportion of sample for training
train_proportion = 0.6

# shuffle patients
xray_data_df = xray_data_df.sample(frac=1).reset_index(drop=True)

# label sets
xray_data_df["set"] = "train"
train_count = int(count * train_proportion)
test_count = int((count-train_count)/2)
xray_data_df.set[train_count:test_count + train_count] = "test"
# validation set will get whatever is left over ~ approximately same size as test
xray_data_df.set[test_count + train_count:] = "validate"

# display results of oversampling
for set in ["train", "test", "validate"]:
    temp_df = xray_data_df[xray_data_df.set == set]
    print("set", set,
          "normal", np.sum(temp_df.normal_count),
          "bacteria", np.sum(temp_df.bacteria_count),
          "virus", np.sum(temp_df.virus_count))

# export df to files (csv to browse, json to retrieve)
xray_data_df.to_json('./chest_xray/xray_data_df.json')
xray_data_df.to_csv('./chest_xray/xray_data_df.csv')



