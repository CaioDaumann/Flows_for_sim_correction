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

# Function to perform a selection in the samples
def perform_zee_selection( data_df, mc_df ):
    # first we need to calculathe the invariant mass of the electron pair

    # variables names used to calculate the mass 
    mass_vars = ["tag_pt","tag_ScEta","tag_phi","probe_pt","probe_ScEta","probe_phi","fixedGridRhoAll"]

    mass_inputs_data = np.array( data_df[mass_vars]) 
    mass_inputs_mc   = np.array( mc_df[mass_vars]  )

    # calculaitng the invariant mass with the expression of a par of massless particles 
    mass_data = np.sqrt(  2*mass_inputs_data[:,0]*mass_inputs_data[:,3]*( np.cosh(  mass_inputs_data[:,1] -  mass_inputs_data[:,4]  )  - np.cos( mass_inputs_data[:,2]  -mass_inputs_data[:,5] )  )  )
    mass_mc   = np.sqrt(  2*mass_inputs_mc[:,0]*mass_inputs_mc[:,3]*( np.cosh(  mass_inputs_mc[:,1] -  mass_inputs_mc[:,4]  )  - np.cos( mass_inputs_mc[:,2]  -mass_inputs_mc[:,5] )  )  )

    # now, in order to perform the needed cuts two masks will be created
    mask_data = np.logical_and( mass_data > 80 , mass_data < 100  )
    mask_data = np.logical_and( mask_data , mass_inputs_data[:,0] > 40  )
    mask_data = np.logical_and( mask_data , mass_inputs_data[:,3] > 20  )
    mask_data = np.logical_and( mask_data , np.abs(mass_inputs_data[:,4]) < 2.5  )
    mask_data = np.logical_and( mask_data , np.abs(mass_inputs_data[:,5]) < 3.1415  )
    mask_data = np.logical_and( mask_data , mass_inputs_data[:,6] < 100  )
    mask_data = np.logical_and( mask_data , mass_inputs_data[:,6] > 0  )
    

    mask_mc   = np.logical_and( mass_mc > 80 , mass_mc < 100  )
    mask_mc   = np.logical_and( mask_mc , mass_inputs_mc[:,0] > 40  )
    mask_mc   = np.logical_and( mask_mc , mass_inputs_mc[:,3] > 20  )
    mask_mc   = np.logical_and( mask_mc , np.abs(mass_inputs_mc[:,4]) < 2.5  )
    mask_mc   = np.logical_and( mask_mc , np.abs(mass_inputs_mc[:,5]) < 3.1415  )
    mask_mc   = np.logical_and( mask_mc , mass_inputs_mc[:,6] > 0  )
    mask_mc   = np.logical_and( mask_mc , mass_inputs_mc[:,6] < 100  )

    # return the masks for further operations
    return mask_data, mask_mc

def perform_zee_kinematics_reweighting(data_array, data_weights, mc_array, mc_weights):
    
    # As a first step, we will normalize the sum of weights to one!
    data_weights = np.array(data_weights/np.sum(data_weights))
    mc_weights   = np.array(mc_weights/np.sum( mc_weights ))

    # I cannot slice pandas df, so I am making them into numpy arrays
    data_array = np.array( data_array )
    mc_array   = np.array( mc_array )

    variable_names = [ "pt", "eta", "phi", "rho" ]

    #defining the range where the rw will be performed in each variables
    pt_min,pt_max   = 20, 120
    rho_min,rho_max = 5.0,60.0
    eta_min,eta_max = -2.501,2.501
    phi_min, phi_max = -3.15,3.15

    # now the number of bins in each distribution
    pt_bins,rho_bins,eta_bins,phi_bins = 20, 30, 10, 4

    # Now we create a 4d histogram of this kinematic variables
    mc_histo,   edges = np.histogramdd( sample =  (mc_array[:,0] ,   mc_array[:,3],      mc_array[:,1], mc_array[:,2])   , bins = (pt_bins,rho_bins,eta_bins,phi_bins), range = [   [pt_min,pt_max], [ rho_min, rho_max ],[eta_min,eta_max], [phi_min,phi_max]  ], weights = mc_weights )
    data_histo, edges = np.histogramdd( sample =  (data_array[:,0] , data_array[:,3] , data_array[:,1], data_array[:,2]) , bins = (pt_bins,rho_bins,eta_bins,phi_bins), range = [   [pt_min,pt_max], [ rho_min, rho_max ],[eta_min,eta_max], [phi_min,phi_max]  ], weights = data_weights )

    #we need to have a index [i,j] to each events, so we can rewighht based on data[i,j]/mc[i,j]
    pt_index =  np.array(pt_bins   *( mc_array[:,0] -  pt_min )/(pt_max - pt_min)   , dtype=np.int8 )
    rho_index = np.array(rho_bins  *( mc_array[:,3] - rho_min )/(rho_max - rho_min) , dtype=np.int8 )
    eta_index = np.array(eta_bins  *( mc_array[:,1] - eta_min )/(eta_max - eta_min) , dtype=np.int8 )
    phi_index = np.array(phi_bins  *( mc_array[:,2] - phi_min )/(phi_max - phi_min) , dtype=np.int8 )
    
    #if a event has a pt higher than the pt_max it will have a higher index and will overflow the histogram. So i clip it to the last value 
    pt_index[ pt_index > pt_bins - 1 ]  = pt_bins - 1
    pt_index[ pt_index <= 0 ] = 0


    rho_index[rho_index > rho_bins - 1 ] = rho_bins - 1
    rho_index[rho_index <= 0 ] = 0

    eta_index[eta_index > eta_bins - 1 ] = eta_bins - 1
    eta_index[eta_index <= 0 ] = 0

    phi_index[phi_index > phi_bins - 1 ] = phi_bins - 1
    phi_index[phi_index <= 0 ] = 0

    #calculating the SF
    sf_rw = ( data_histo[  pt_index, rho_index,eta_index, phi_index ] )/(mc_histo[pt_index, rho_index,eta_index, phi_index] + 1e-10 )
    sf_rw[ sf_rw > 10 ] = 10 # if the sf_rw is too big, we clip it to remove artifacts

    mc_weights = mc_weights* sf_rw

    # return the mc and data weights. I am return the data one here because he is now normalized
    return data_weights, mc_weights

# this function prepare teh data to be used bu pytorch!
def separate_training_data( data_df, mc_df, mc_weights, data_weights, input_vars, condition_vars):

    # creating the inputs and conditions tensors! - adding the bollean to the conditions tensors
    data_inputs     = torch.tensor(np.array(data_df[input_vars]))
    data_conditions = torch.tensor(np.concatenate( [  np.array( data_df[condition_vars]), np.ones( len( data_weights )  ).reshape(-1,1)  ], axis = 1  ) )

    mc_inputs     = torch.tensor(np.array(mc_df[input_vars]))
    mc_conditions = torch.tensor(np.concatenate( [  np.array( mc_df[condition_vars]), 0*np.ones( len( mc_weights )  ).reshape(-1,1)  ], axis = 1  ) )

    #first we shuffle all the arrays!
    permutation = np.random.permutation(len(data_weights))
    data_inputs      = torch.tensor(data_inputs[permutation])
    data_conditions  = torch.tensor(data_conditions[permutation])
    data_weights     = torch.tensor(data_weights[permutation])

    permutation = np.random.permutation(len(mc_weights))
    mc_inputs      = torch.tensor(mc_inputs[permutation])
    mc_conditions  = torch.tensor(mc_conditions[permutation])
    mc_weights     = torch.tensor(mc_weights[permutation])

    # Now, in ordert not to bias the network we choose make sure the tensors of data and simulation have the same number of events
    mc_inputs = mc_inputs[ :len(data_inputs) ]
    mc_conditions = mc_conditions[ :len(data_inputs) ]
    mc_weights = mc_weights[ :len(data_inputs) ]
    
    assert len( mc_weights )    == len( data_weights )
    assert len( mc_conditions ) == len(data_conditions)


    # Now, the fun part! - Separating everyhting into trainnig, validation and testing datasets!
    data_training_inputs     = data_inputs[: int( 0.6*len(data_inputs  )) ] 
    data_training_conditions = data_conditions[: int( 0.6*len(data_inputs  )) ] 
    data_training_weights    = data_weights[: int( 0.6*len(data_inputs  )) ] 

    mc_training_inputs     = mc_inputs[: int( 0.6*len(mc_inputs  )) ] 
    mc_training_conditions = mc_conditions[: int( 0.6*len(mc_inputs  )) ] 
    mc_training_weights    = mc_weights[: int( 0.6*len(mc_inputs  )) ] 

    # now, the validation dataset
    data_validation_inputs     = data_inputs[int( 0.6*len(data_inputs  )):int( 0.65*len(data_inputs  )) ] 
    data_validation_conditions = data_conditions[int( 0.6*len(data_inputs  )):int( 0.65*len(data_inputs  )) ] 
    data_validation_weights    = data_weights[int( 0.6*len(data_inputs  )):int( 0.65*len(data_inputs  )) ] 

    mc_validation_inputs     = mc_inputs[int( 0.6*len(mc_inputs  )):int( 0.65*len(mc_inputs  )) ] 
    mc_validation_conditions = mc_conditions[int( 0.6*len(mc_inputs  )):int( 0.65*len(mc_inputs  )) ] 
    mc_validation_weights    = mc_weights[int( 0.6*len(mc_inputs  )):int( 0.65*len(mc_inputs  )) ] 

    # now for the grand finalle, the test tensors
    data_test_inputs     = data_inputs[int( 0.65*len(data_inputs  )): ] 
    data_test_conditions = data_conditions[int( 0.65*len(data_inputs  )): ] 
    data_test_weights    = data_weights[int( 0.65*len(data_inputs  )): ] 

    mc_test_inputs     = mc_inputs[int( 0.65*len(mc_inputs  )): ] 
    mc_test_conditions = mc_conditions[int( 0.65*len(mc_inputs  )):] 
    mc_test_weights    = mc_weights[int( 0.65*len(mc_inputs  )):] 

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

def read_zee_data():

    path_to_data = "/net/scratch_cms3a/daumann/nanoAODv12_Production/"

    # We want to correct the variables that are used as input to run3 photon MVA ID
    var_list = ["probe_energyRaw",
                "probe_r9", 
                "probe_sieie",
                "probe_etaWidth",
                "probe_phiWidth",
                "probe_sieip",
                "probe_s4",
                "probe_hoe",
                "probe_ecalPFClusterIso",
                "probe_trkSumPtHollowConeDR03",
                "probe_trkSumPtSolidConeDR04",
                "probe_pfChargedIso",
                "probe_pfChargedIsoWorstVtx",
                "probe_esEffSigmaRR",
                "probe_esEnergyOverRawE",
                "probe_hcalPFClusterIso",
                "probe_energyErr"]
    
    # These variables will be used as conditions to the normalizing flow - they will not be transformed!
    conditions_list = [ "probe_pt","probe_ScEta","probe_phi","fixedGridRhoAll"]

    # Lets now read the data and simultion as pandas dataframes
    files_DY_mc  = glob.glob( path_to_data + "DY_postEE_v12/nominal/*.parquet")
    files_DY_mc = files_DY_mc[:20]
    simulation   = [pd.read_parquet(f) for f in files_DY_mc]
    drell_yan_df = pd.concat(simulation,ignore_index=True)

    # now the data files for the epochs F and G
    files_DY_data_F = glob.glob( "/net/scratch_cms3a/daumann/nanoAODv12_Production/Data_Run2022F_v12/nominal/*.parquet")
    files_DY_data_F = files_DY_data_F[:40]

    files_DY_data_G  = glob.glob( "/net/scratch_cms3a/daumann/nanoAODv12_Production/Data_Run2022G_v12/nominal/*.parquet")
    files_DY_data_G = files_DY_data_G[:40]

    # merhging both dataframes
    files_DY_data = [files_DY_data_G,files_DY_data_F]

    data   = [pd.read_parquet(f) for f in files_DY_data]
    data_df = pd.concat(data,ignore_index=True)
    # end of data reading!

    # Now that the data is read, we need to perform a loose selection with the objective of decrease teh background contamination
    # The cuts include:
    # Mass windown cut: We only select events in a tight mass window around the Z peak [80,100]
    # Select event with eta < 2.5
    # loose tag mvaID cut of 0.0. Since the training is performed in probe electrons we dont expect a bias
    # The selection will be done in the perform_zee_selection() function
    mask_data, mask_mc = perform_zee_selection( data_df, drell_yan_df )

    # now, due to diferences in kinematics, a rewighting in the four kinematic variables [pt,eta,phi and rho] will be perform
    mc_weights   = drell_yan_df["weight"]
    data_weights = np.ones( len( data_df["fixedGridRhoAll"] ) )
    data_weights, mc_weights = perform_zee_kinematics_reweighting(data_df[conditions_list][mask_data], data_weights[mask_data], drell_yan_df[conditions_list][mask_mc], mc_weights[mask_mc])

    # now lets call a plotting function to perform the plots of the read distributions for validation porpuses
    path_to_plots = "/net/scratch_cms3a/daumann/PhD/EarlyHgg/simulation_to_data_corrections/plot/validation_plots/"

    # now as a last step, we need to split the data into training, validation and test dataset
    plot_utils.plot_distributions( path_to_plots, data_df[mask_data], drell_yan_df[mask_mc], mc_weights, [var_list, conditions_list] )

    # now, the datsets will be separated into training, validation and test dataset, and saved for further reading by the training class!
    separate_training_data(data_df[mask_data], drell_yan_df[mask_mc], mc_weights, data_weights, var_list, conditions_list)

    print('\n End of data reading! - No errors encountered! ')
    print( 'Number of MC events: ', len(drell_yan_df[mask_mc] ), ' Number of data events: ', len(data_df[mask_data]))