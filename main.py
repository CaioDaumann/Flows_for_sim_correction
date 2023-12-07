# Main script to perform the Z->ee simulation corrections for the H->gg analysis

# python libraries import
import os 
import numpy as np
import glob
import torch
import pandas as pd

# importing other scripts
import data_reading.read_data as data_reader
import plot.plot_utils        as plot_utils

def main():
    print("Welcome to the simulation corrections 2000!")

    # Lets call the function responsible for reading and treat the data
    # this function reads both mc and data, perform a basic selection and reweight the 4d-kinematics distirbutions
    # in the end they are saved into the a folder, so one doesnt need to go through this function all the time
    re_process_data = False
    if( re_process_data ):
        data_reader.read_zee_data()

    # Now, we call the class that handles the transformations, training and validaiton of the corrections



if __name__ == "__main__":
    main()