# Main script to perform the Z->ee simulation corrections for the H->gg analysis

# python libraries import
import os 
import numpy as np
import glob
import torch
import pandas as pd

import yaml
from yaml import Loader

# importing other scripts
import data_reading.read_data as data_reader
import plot.plot_utils        as plot_utils
import normalizing_flows.training_utils as training_utils

def main():
    print("Welcome to the simulation corrections 2000!")

    # Lets call the function responsible for reading and treat the data
    # this function reads both mc and data, perform a basic selection and reweight the 4d-kinematics distirbutions
    # in the end they are saved into the a folder, so one doesnt need to go through this function all the time
    re_process_data = False
    if( re_process_data ):
        data_reader.read_zee_data()

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

        # Now, we call the class that handles the transformations, training and validaiton of the corrections
        corrections = training_utils.Simulation_correction( str(key) ,n_transforms, n_splines_bins, aux_nodes, aux_layers, max_epoch_number, initial_lr, batch_size  )
        corrections.setup_flow()
        corrections.train_the_flow()

    # Now, we call the class that handles the transformations, training and validaiton of the corrections
    #corrections = training_utils.Simulation_correction()
    #corrections.setup_flow()
    #corrections.train_the_flow()


if __name__ == "__main__":
    main()