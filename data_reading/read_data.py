# Loads and treat the Z->ee data and simulation
#Author: Caio Cesar Daumann : caio.cesar.cern.ch

# importing libraries
import os 
import numpy as np
import glob
import torch
import pandas as pd

# importing other scripts
import plot.plot_utils        as plot_utils
import data_reading.zmmg_utils as zmmg_utils

# Function to perform a selection in the samples
def perform_zee_selection( data_df, mc_df ):

    # first we need to calculathe the invariant mass of the electron pair
    mask_data = np.logical_and( data_df["mass"] > 80 , data_df["mass"] < 100  )
    mask_data = np.logical_and( mask_data , data_df["tag_pt"] > 40  ) #tag pt cut
    mask_data = np.logical_and( mask_data , data_df["probe_pt"] > 22  ) #probe pt cut
    mask_data = np.logical_and( mask_data , np.abs(data_df["probe_ScEta"]) < 2.5  )  # eta cut
    mask_data = np.logical_and( mask_data , np.abs(data_df["probe_phi"]) < 3.1415  ) # phi cut
    mask_data = np.logical_and( mask_data , data_df["tag_mvaID"] > 0.0  )  # tag mvaID cut

    mask_mc = np.logical_and( mc_df["mass"] > 80 , mc_df["mass"] < 100  )
    mask_mc = np.logical_and( mask_mc , mc_df["tag_pt"] > 40  )
    mask_mc = np.logical_and( mask_mc , mc_df["probe_pt"] > 22  )
    mask_mc = np.logical_and( mask_mc , np.abs(mc_df["probe_ScEta"]) < 2.5  )
    mask_mc = np.logical_and( mask_mc , np.abs(mc_df["probe_phi"]) < 3.1415  )
    mask_mc = np.logical_and( mask_mc , mc_df["tag_mvaID"] > 0.0  )
    
    # return the masks for further operations
    return mask_data, mask_mc


# this function prepare teh data to be used bu pytorch!
def separate_training_data( data_df, mc_df, mc_weights, data_weights, input_vars, condition_vars):

    mc_inputs     = torch.tensor(np.array( np.nan_to_num(mc_df[input_vars])))
    mc_conditions = torch.tensor(np.concatenate( [  np.array( np.nan_to_num(mc_df[condition_vars])), 0*np.ones( len( mc_weights )  ).reshape(-1,1)  ], axis = 1  ) )

    input_vars_data = input_vars
    for i in range( len(input_vars) ):
        input_vars_data[i] = input_vars_data[i].replace('_raw','')

    # creating the inputs and conditions tensors! - adding the bollean to the conditions tensors
    data_inputs     = torch.tensor(np.nan_to_num(np.array(data_df[input_vars_data])))
    data_conditions = torch.tensor(np.concatenate( [ np.nan_to_num(np.array( data_df[condition_vars])), np.ones( len( data_weights )  ).reshape(-1,1)  ], axis = 1  ) )

    #first we shuffle all the arrays!
    permutation = np.random.permutation(len(data_weights))
    data_inputs      = torch.tensor(data_inputs[permutation])
    data_conditions  = torch.tensor(data_conditions[permutation])
    data_weights     = torch.tensor(data_weights[permutation])

    mc_permutation = np.random.permutation(len(mc_weights))
    mc_inputs      = torch.tensor(mc_inputs[mc_permutation])
    mc_conditions  = torch.tensor(mc_conditions[mc_permutation])
    mc_weights     = torch.tensor(mc_weights[mc_permutation])


    # Now, in order not to bias the network we choose make sure the tensors of data and simulation have the same number of events
    try:
        mc_inputs = mc_inputs[ :len(data_inputs) ]
        mc_conditions = mc_conditions[ :len(data_inputs) ]
        mc_weights = mc_weights[ :len(data_inputs) ]
        
        assert len( mc_weights )    == len( data_weights )
        assert len( mc_conditions ) == len(data_conditions)
    except:
        data_inputs = data_inputs[ :len(mc_inputs) ]
        data_conditions = data_conditions[ :len(mc_inputs) ]
        data_weights = data_weights[ :len(mc_inputs) ]
        
        assert len( mc_weights )    == len( data_weights )
        assert len( mc_conditions ) == len(data_conditions)

    print( 'Number of MC events after equiparing! ', len( mc_conditions ), ' Number of data events: ', len(data_conditions))

    training_percent = 0.7
    validation_percent = 0.03 + training_percent
    testing_percet = 1 - training_percent - validation_percent

    # Now, the fun part! - Separating everyhting into trainnig, validation and testing datasets!
    data_training_inputs     = data_inputs[: int( training_percent*len(data_inputs  )) ] 
    data_training_conditions = data_conditions[: int( training_percent*len(data_inputs  )) ] 
    data_training_weights    = data_weights[: int( training_percent*len(data_inputs  )) ] 

    mc_training_inputs     = mc_inputs[: int( training_percent*len(mc_inputs  )) ] 
    mc_training_conditions = mc_conditions[: int( training_percent*len(mc_inputs  )) ] 
    mc_training_weights    = mc_weights[: int( training_percent*len(mc_inputs  )) ] 

    # now, the validation dataset
    data_validation_inputs     = data_inputs[int(training_percent*len(data_inputs  )):int( validation_percent*len(data_inputs  )) ] 
    data_validation_conditions = data_conditions[int(training_percent*len(data_inputs  )):int( validation_percent*len(data_inputs  )) ] 
    data_validation_weights    = data_weights[int(training_percent*len(data_inputs  )):int( validation_percent*len(data_inputs  )) ] 

    mc_validation_inputs     = mc_inputs[int(training_percent*len(mc_inputs  )):int( validation_percent*len(mc_inputs  )) ] 
    mc_validation_conditions = mc_conditions[int(training_percent*len(mc_inputs  )):int( validation_percent*len(mc_inputs  )) ] 
    mc_validation_weights    = mc_weights[int(training_percent*len(mc_inputs  )):int( validation_percent*len(mc_inputs  )) ] 

    # now for the grand finalle, the test tensors
    data_test_inputs     = data_inputs[int( validation_percent*len(data_inputs  )): ] 
    data_test_conditions = data_conditions[int( validation_percent*len(data_inputs  )): ] 
    data_test_weights    = data_weights[int( validation_percent*len(data_inputs  )): ] 

    mc_test_inputs     = mc_inputs[int( validation_percent*len(mc_inputs  )): ] 
    mc_test_conditions = mc_conditions[int( validation_percent*len(mc_inputs  )):] 
    mc_test_weights    = mc_weights[int( validation_percent*len(mc_inputs  )):] 

    # now, all the tensors are saved so they can be read by the training class
    path_to_save_tensors = "/net/scratch_cms3a/daumann/PhD/EarlyHgg/simulation_to_data_corrections/data_reading/saved_tensors/zee_tensors/"

    torch.save( data_training_inputs       , path_to_save_tensors + 'data_training_inputs.pt' )
    torch.save( data_training_conditions   , path_to_save_tensors + 'data_training_conditions.pt')
    torch.save( data_training_weights      , path_to_save_tensors + 'data_training_weights.pt') 

    torch.save( mc_training_inputs       , path_to_save_tensors + 'mc_training_inputs.pt' )
    torch.save( mc_training_conditions   , path_to_save_tensors + 'mc_training_conditions.pt')
    torch.save( mc_training_weights      , path_to_save_tensors + 'mc_training_weights.pt') 

    # now the validation tensors

    torch.save( data_validation_inputs       , path_to_save_tensors + 'data_validation_inputs.pt' )
    torch.save( data_validation_conditions   , path_to_save_tensors + 'data_validation_conditions.pt')
    torch.save( data_validation_weights      , path_to_save_tensors + 'data_validation_weights.pt') 

    torch.save( mc_validation_inputs       , path_to_save_tensors + 'mc_validation_inputs.pt' )
    torch.save( mc_validation_conditions   , path_to_save_tensors + 'mc_validation_conditions.pt')
    torch.save( mc_validation_weights      , path_to_save_tensors + 'mc_validation_weights.pt') 

    # now the test tensors
    torch.save( data_test_inputs       , path_to_save_tensors + 'data_test_inputs.pt' )
    torch.save( data_test_conditions   , path_to_save_tensors + 'data_test_conditions.pt')
    torch.save( data_test_weights      , path_to_save_tensors + 'data_test_weights.pt') 

    torch.save( mc_test_inputs       , path_to_save_tensors + 'mc_test_inputs.pt' )
    torch.save( mc_test_conditions   , path_to_save_tensors + 'mc_test_conditions.pt')
    torch.save( mc_test_weights      , path_to_save_tensors + 'mc_test_weights.pt')     

    return path_to_save_tensors

def read_zee_data(IsPostEE):

    # We want to correct the variables that are used as input to run3 photon MVA ID
    # Two isolation variables were added () for the calculation of pre-selection SF [pfPhoIso03, pfRelIso03_chg_quadratic]
    var_list = ["probe_energyRaw",
                "probe_raw_r9", 
                "probe_raw_sieie",
                "probe_raw_etaWidth",
                "probe_raw_phiWidth",
                "probe_raw_sieip",
                "probe_raw_s4",
                "probe_raw_hoe",
                "probe_raw_ecalPFClusterIso",
                "probe_raw_trkSumPtHollowConeDR03",
                "probe_raw_trkSumPtSolidConeDR04",
                "probe_raw_pfChargedIso",
                "probe_raw_pfChargedIsoWorstVtx",
                "probe_raw_esEffSigmaRR",
                "probe_raw_esEnergyOverRawE",
                "probe_raw_hcalPFClusterIso",
                "probe_raw_energyErr"]
      
    # These variables will be used as conditions to the normalizing flow - they will not be transformed!
    conditions_list = [ "probe_pt","probe_ScEta","probe_phi","fixedGridRhoAll"]

    #Train_PostEE = False
    if( IsPostEE ):
            
        # I think there is a problem with the samples above, lets test with the new samples!
        # Now, lets make some tests with the new samples for David BSC theses
        files_data = glob.glob("/net/scratch_cms3a/daumann/normalizing_flows_project/script_to_prepare_samples_for_paper/splited_parquet/data_train.parquet")
            
        files_mc =  glob.glob("/net/scratch_cms3a/daumann/normalizing_flows_project/script_to_prepare_samples_for_paper/splited_parquet/DY_train.parquet")
        simulation   = [pd.read_parquet(f) for f in files_mc]
        drell_yan_df = pd.concat(simulation,ignore_index=True)
        
    else:

            files_DY_mc  = glob.glob( "/net/scratch_cms3a/daumann/HiggsDNA/scripts/v13_runners/Rerun_after_probe_selection_to_train_flow/DY_preEE_v13/nominal/*.parquet" )
            files_DY_mc  = files_DY_mc[:3000]
            simulation   = [pd.read_parquet(f) for f in files_DY_mc]
            drell_yan_df = pd.concat(simulation,ignore_index=True)

            # now the data files for the epochs F and G
            #files_DY_data_E = glob.glob(  "/net/scratch_cms3a/daumann/HiggsDNA/v13_samples_2/dataE_v13/nominal/*.parquet")
            files_DY_data_C = glob.glob(    "/net/scratch_cms3a/daumann/HiggsDNA/scripts/v13_runners/Rerun_after_probe_selection_to_train_flow/dataC_v13/nominal/*.parquet")
            #files_DY_data_E = files_DY_data_E[:30000]

            #files_DY_data_F = glob.glob(  "/net/scratch_cms3a/daumann/HiggsDNA/v13_samples_2/dataF_v13/nominal/*.parquet")
            files_DY_data_D = glob.glob(    "/net/scratch_cms3a/daumann/HiggsDNA/scripts/v13_runners/Rerun_after_probe_selection_to_train_flow/dataD_v13/nominal/*.parquet")
            #files_DY_data_F = files_DY_data_F[:300000]    


    if(IsPostEE):

        # merhging both dataframes
        #files_DY_data = [files_DY_data_E,files_DY_data_G,files_DY_data_F]
        files_DY_data = [files_data]

        data   = [pd.read_parquet(f) for f in files_DY_data]
        data_df = pd.concat(data,ignore_index=True)
        # end of data reading!
    else:
        # merhging both dataframes
        files_DY_data = [files_DY_data_C,files_DY_data_D]

        data   = [pd.read_parquet(f) for f in files_DY_data]
        data_df = pd.concat(data,ignore_index=True)

    # just for debugging purposes
    #data_df      = data_df[:100000]
    #drell_yan_df = drell_yan_df[:100000]

    # Now that the data is read, we need to perform a loose selection with the objective of decrease teh background contamination
    # The cuts include:
    # Mass windown cut: We only select events in a tight mass window around the Z peak [80,100]
    # Select event with eta < 2.5
    # loose tag mvaID cut of 0.0. Since the training is performed in probe electrons we dont expect a bias
    # The selection will be done in the perform_zee_selection() function
    mask_data, mask_mc = perform_zee_selection( data_df, drell_yan_df )

    # now, due to diferences in kinematics, a rewighting in the four kinematic variables [pt,eta,phi and rho] will be perform
    mc_weights        = np.array(drell_yan_df["rw_weights"].values)[mask_mc]
    data_weights      = np.ones( len( data_df["fixedGridRhoAll"] ) )[mask_data]
    
    # Normalizing the weights to one
    mc_weights   = mc_weights/np.sum(mc_weights)
    data_weights = data_weights/np.sum(data_weights)
    
    mc_weights_before = mc_weights
    
    # now lets call a plotting function to perform the plots of the read distributions for validation porpuses
    path_to_plots = "/net/scratch_cms3a/daumann/PhD/EarlyHgg/simulation_to_data_corrections/plot/validation_plots/"

    # now as a last step, we need to split the data into training, validation and test dataset
    plot_utils.plot_distributions( path_to_plots, data_df[mask_data], drell_yan_df[mask_mc], data_weights, mc_weights , [var_list, conditions_list], weights_befores_rw = mc_weights_before)

    # now, the datsets will be separated into training, validation and test dataset, and saved for further reading by the training class!
    separate_training_data(data_df[mask_data], drell_yan_df[mask_mc], mc_weights, data_weights, var_list, conditions_list)

    print('\n End of data reading! - No errors encountered! ')
    print( 'Number of MC events: ', len(drell_yan_df[mask_mc] ), ' Number of data events: ', len(data_df[mask_data]))
    print('IsPostEE? ', IsPostEE)
    