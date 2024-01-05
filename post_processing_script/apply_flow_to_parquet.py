"""
This script was written as a standalone form to apply the normalizing flows to the processed parquet files of HiggsDNA
One just have to specif a path to the samples and this script will take care of the rest
"""

#Importing needed libraries
import os 
import argparse
import numpy as np
import glob
import torch
import pandas as pd
import zuko

# Importing other python scripts
import utils as utils

processes = ["Zee","Zmmg","Hgg","Diphoton","GJet"]

var_list = [    "r9", 
                "sieie",
                "etaWidth",
                "phiWidth",
                "sieip",
                "s4",
                "hoe",
                "ecalPFClusterIso",
                "trkSumPtHollowConeDR03",
                "trkSumPtSolidConeDR04",
                "pfChargedIso",
                "pfChargedIsoWorstVtx",
                "esEffSigmaRR",
                "esEnergyOverRawE",
                "hcalPFClusterIso",
                "energyErr"]

conditions_list = [ "pt","ScEta","phi","fixedGridRhoAll"]

def main():

    global var_list

    # Reading the files as a dataframe
    files = glob.glob( args.mcfilespath + "*.parquet" )
    files = files[:150]

    # concatenating the files
    data  = [pd.read_parquet(f) for f in files]
    mc_df = pd.concat(data,ignore_index=True)   

    # Making sure the process is correclty selected
    if( args.process is None ):
        print( 'Specify a process! - Terminating' )
        exit()
    elif( args.process not in processes ):
        print( "Specify a existing process: ", processes )
        exit()
    
    print( '\nApplyting the normalizing flows corrections to the ', args.process , ' process samples!\n'  )

    # As a first test, lets do it only for probe photons! - after we generalize to probe and tag
    if( args.process == "Zee" ):
        input_list_probe      = [ "probe_" + s  for s in var_list]
        conditions_list_probe = [ "probe_" + s  for s in conditions_list[:-1]]
        conditions_list_probe.append( "fixedGridRhoAll" )

        input_list_tag   = [ "tag_"   + s  for s in var_list]

    # From pandas DF to pytorch tensors for the flow processing
    mc_flow_inputs       = torch.tensor(  np.array( mc_df[input_list_probe]  ) )
    mc_flow_conditions   = torch.tensor(  np.concatenate(  [np.array( mc_df[conditions_list_probe] ), 0*np.ones(  len(mc_df)  ).reshape(-1,1) ] , axis = 1  )  )

    # Now we proceed to the calculation of the corrections
    # 1. we load the PostEE flow model
    flow = zuko.flows.NSF( mc_flow_inputs.size()[1] , context = mc_flow_conditions.size()[1], bins = 10,transforms = 5, hidden_features=[256] * 2)
    path_means_std = '/net/scratch_cms3a/daumann/PhD/EarlyHgg/simulation_to_data_corrections/results/configuration_v13_big_stats_9_long_train/'
    flow.load_state_dict(torch.load( path_means_std + 'best_model_.pth', map_location=torch.device('cpu')))

    device = torch.device('cpu') 

    # 2. The samples are pre-processed [standartization and Isolation variables transformation]
    print('Pre-processing the samples...')
    input_tensor, conditions_tensor, input_mean_for_std, input_std_for_std, condition_mean_for_std,condition_std_for_std, vector_for_iso_constructors_mc = utils.perform_pre_processing(mc_flow_inputs.to(device), mc_flow_conditions.to(device), path_means_std)

    # 3. Process the samples with the flow
    print('Processing the samples...')
    samples = utils.apply_flow( input_tensor, conditions_tensor, flow )

    # 4. Inverting the transformations! - test what happens if we input the original mc here!
    corrected_inputs = utils.invert_pre_processing(samples,  input_mean_for_std, input_std_for_std, vector_for_iso_constructors_mc)

    # Now the mvaID is recalculated and added to the tree

    # Adding the corrected entries to the df
    for i in range( len(var_list) ):
        mc_df[ input_list_probe[i] + "_corr" ] =  corrected_inputs[:,i]

    print( mc_df["probe_r9_corr"] )

    # Dumping the df file with the new entries
    mc_df.to_parquet("validate_post_processing_script/test_out.parquet")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Apply a trained NSF flows to HiggsDNA output")
    parser.add_argument('-mcfilespath'   , '--mcfilespath'  , type=str, help= "Path to the MC files where the flow will be applied")
    parser.add_argument('-process', '--process', type = str, help = "Zee, Zmmg, Hgg, Diphoton, GJet")
    args = parser.parse_args()

    main() 
