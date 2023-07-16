# file resample_images
# language: Python 3.9
# purpose: resample (oversample) images to correct imbalance between normal, virus & bacteria

# imports
import numpy as np
import pandas as pd
from random import choices, randint
import matplotlib.image as mpimg
import os
import re


# load xray_data_df
xray_data_df = pd.read_json('./chest_xray/xray_data_df.json')

# oversample by set (to prevent contaminating test & validate sets)
# get list of sets & types from data)
sets = ["train", "test", "validate"]
types = ["normal", "bacteria", "virus"]
other_types = ["normal", "virus"]
print(xray_data_df.iloc[0:5])
print("columns", xray_data_df.columns)

for set in sets:
    print("set", set)
    temp_df = xray_data_df[xray_data_df["set"] == set]
    # since bacteria count is always higher than other types
    # we will use bacteria count to oversample other image types
    target_sample = np.sum(temp_df.bacteria_count)

    # oversample normal
    temp2_df = temp_df[temp_df.normal_count > 0]
    patients = list(temp2_df.patient)
    sample = np.sum(temp_df["normal_count"])
    # compute needed sample
    needed_sample = target_sample - sample

    # pull additional samples with replacement
    # samples pulled by patient not by image so as not to overweight patients with multiple images
    # for each sampled patient one normal image is added to the sample
    samples = choices(patients, k=needed_sample)

    # iterate through sample
    for sample in samples:
        images = xray_data_df.normal[xray_data_df["patient"] == sample].iloc[0]
        count = xray_data_df.normal_count[xray_data_df["patient"] == sample].iloc[0]
        if count > 1:
            added_image = images[randint(0, count - 1)]
        else:
            added_image = images[0]
        xray_data_df.normal[xray_data_df.patient == sample].iloc[0].append(added_image)
        xray_data_df.normal_count[xray_data_df.patient == sample] = count + 1

    # oversample virus
    temp2_df = temp_df[temp_df.virus_count > 0]
    patients = list(temp2_df.patient)
    sample = np.sum(temp_df["virus_count"])
    # compute needed sample
    needed_sample = target_sample - sample

    # pull additional samples with replacement
    # samples pulled by patient not by image so as not to overweight patients with multiple images
    # for each sampled patient one normal image is added to the sample
    samples = choices(patients, k=needed_sample)

    # iterate through sample
    for sample in samples:
        images = xray_data_df.virus[xray_data_df["patient"] == sample].iloc[0]
        count = xray_data_df.virus_count[xray_data_df["patient"] == sample].iloc[0]
        if count > 1:
            added_image = images[randint(0, count - 1)]
        else:
            added_image = images[0]
        xray_data_df.virus[xray_data_df.patient == sample].iloc[0].append(added_image)
        xray_data_df.virus_count[xray_data_df.patient == sample] = count + 1


for set in ["train", "test", "validate"]:
    temp_df = xray_data_df[xray_data_df.set == set]
    print("set", set,
          "normal", np.sum(temp_df.normal_count),
          "bacteria", np.sum(temp_df.bacteria_count),
          "virus", np.sum(temp_df.virus_count))

xray_data_df.to_json('./chest_xray/xray_balanced_data_df.json')
xray_data_df.to_csv('./chest_xray/xray_balanced_data_df.csv')

