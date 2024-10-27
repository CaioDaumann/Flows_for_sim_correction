# Main script to perform the Z->ee simulation corrections for the H->gg analysis

# python libraries import
import os 
import numpy as np
import glob
import torch
import pandas as pd

import yaml
from yaml import Loader
import json

# importing other scripts
import data_reading.read_data as data_reader
import plot.plot_utils        as plot_utils
import normalizing_flows.training_utils as training_utils

def test_big_boy():
    assert 1 == 1

def main():
    print("Welcome to the simulation corrections 2000!")

    # Lets call the function responsible for reading and treat the data
    # this function reads both mc and data, perform a basic selection and reweight the 4d-kinematics distirbutions
    # in the end they are saved into the a folder, so one doesnt need to go through this function all the time

    # First we read the config files with the list of variables that should be corrected and used as conditions
    # the path for the TnP MC and data files are also given here
    with open("var_training_list.json", "r") as file:
        data = json.load(file)

    var_list = data["var_list"]
    var_list_barrel_only = data["var_list_barrel_only"] # This list is need to calculate the correlation matrices, as for end-cap variables, when in barrel are always zero!
    conditions_list = data["conditions_list"]
    data_samples_path = data["data_files"]
    mc_samples_path = data["MC_files"]
    mc_samples_lumi_norm = data["MC_files_normalization"]
    Index_for_Iso_transform = data["Index_for_Iso_transform"]
    
    #loop to read over network condigurations from the yaml file: - one way to do hyperparameter optimization
    stream = open("flow_configuration.yaml", 'r')
    dictionary = yaml.load(stream,Loader)

    for key in dictionary:

        #network configurations
        n_transforms   = dictionary[key]["n_transforms"]     # number of transformation
        aux_nodes      = dictionary[key]["aux_nodes"]        # number of nodes in the auxiliary network
        aux_layers     = dictionary[key]["aux_layers"]       # number of auxiliary layers in each flow transformation
        n_splines_bins = dictionary[key]["n_splines_bins"]   # Number of rationale quadratic spline flows bins

        # Some general training parameters
        max_epoch_number = dictionary[key]["max_epochs"]
        initial_lr       = dictionary[key]["initial_lr"]
        batch_size       = dictionary[key]["batch_size"]

        DoKinematicsRW   = dictionary[key]["DoKinematicsRW"]
        
        IsAutoRegressive = dictionary[key]["IsAutoRegressive"]

        re_process_data = True
        if( re_process_data ):
            data_reader.read_zee_data(var_list, conditions_list, data_samples_path, mc_samples_path, mc_samples_lumi_norm, DoKinematicsRW)

        # Now, we call the class that handles the transformations, training and validaiton of the corrections
        corrections = training_utils.Simulation_correction( str(key),  var_list, var_list_barrel_only, conditions_list , Index_for_Iso_transform , IsAutoRegressive,n_transforms, n_splines_bins, aux_nodes, aux_layers, max_epoch_number, initial_lr, batch_size  )
        corrections.setup_flow()
        corrections.train_the_flow()

if __name__ == "__main__":
    main()