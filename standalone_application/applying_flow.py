# Script to test the application of normalizing flows outisde of the main code enviroment

# python libraries import
import os 
import numpy as np
import glob
import torch
import pandas as pd
import zuko

import matplotlib.pyplot as plt 
import mplhep, hist
plt.style.use([mplhep.style.CMS])

# importing other scripts
import standalone_plot        as plot_utils
import zmmg_process_utils     as zmmg_utils
#from   apply_flow_zmmg        import zmmg_kinematics_reweighting

def perform_zee_selection( data_df, mc_df ):
    # first we need to calculathe the invariant mass of the electron pair

    # variables names used to calculate the mass 
    mass_vars = ["tag_pt","tag_ScEta","tag_phi","probe_pt","probe_ScEta","probe_phi","fixedGridRhoAll", "tag_mvaID"]

    mass_inputs_data = np.array( data_df[mass_vars]) 
    mass_inputs_mc   = np.array( mc_df[mass_vars]  )

    # calculaitng the invariant mass with the expression of a par of massless particles 
    mass_data = np.sqrt(  2*mass_inputs_data[:,0]*mass_inputs_data[:,3]*( np.cosh(  mass_inputs_data[:,1] -  mass_inputs_data[:,4]  )  - np.cos( mass_inputs_data[:,2]  -mass_inputs_data[:,5] )  )  )
    mass_mc   = np.sqrt(  2*mass_inputs_mc[:,0]*mass_inputs_mc[:,3]*( np.cosh(  mass_inputs_mc[:,1] -  mass_inputs_mc[:,4]  )  - np.cos( mass_inputs_mc[:,2]  -mass_inputs_mc[:,5] )  )  )

    # now, in order to perform the needed cuts two masks will be created
    mask_data = np.logical_and( mass_data > 80 , mass_data < 100  )
    mask_data = np.logical_and( mask_data , mass_inputs_data[:,0] > 40  ) #tag pt cut
    mask_data = np.logical_and( mask_data , mass_inputs_data[:,3] > 20  ) #probe pt cut
    mask_data = np.logical_and( mask_data , np.abs(mass_inputs_data[:,4]) < 2.5  )  # eta cut
    mask_data = np.logical_and( mask_data , np.abs(mass_inputs_data[:,5]) < 3.1415  ) # phi cut
    mask_data = np.logical_and( mask_data , mass_inputs_data[:,6] < 100  )
    mask_data = np.logical_and( mask_data , mass_inputs_data[:,6] > 0  )
    mask_data = np.logical_and( mask_data , mass_inputs_data[:,7] > 0.0  )  # tag mvaID cut
    

    mask_mc   = np.logical_and( mass_mc > 80 , mass_mc < 100  )
    mask_mc   = np.logical_and( mask_mc , mass_inputs_mc[:,0] > 40  )
    mask_mc   = np.logical_and( mask_mc , mass_inputs_mc[:,3] > 20  )
    mask_mc   = np.logical_and( mask_mc , np.abs(mass_inputs_mc[:,4]) < 2.5  )
    mask_mc   = np.logical_and( mask_mc , np.abs(mass_inputs_mc[:,5]) < 3.1415  )
    mask_mc   = np.logical_and( mask_mc , mass_inputs_mc[:,6] > 0  )
    mask_mc   = np.logical_and( mask_mc , mass_inputs_mc[:,6] < 100  )
    mask_mc   = np.logical_and( mask_mc , mass_inputs_mc[:,7] > 0.0  )

    # return the masks for further operations
    return mask_data, mask_mc

# This is only needed when using the DF pre-processing options!
def perform_zee_kinematics_reweighting(data_array, data_weights, mc_array, mc_weights, zmmg = False):
    
    # As a first step, we will normalize the sum of weights to one!
    data_weights = np.array(data_weights/np.sum(data_weights))
    mc_weights   = np.array(mc_weights/np.sum( mc_weights ))

    # I cannot slice pandas df, so I am making them into numpy arrays
    data_array = np.array( data_array )
    mc_array   = np.array( mc_array )

    variable_names = [ "pt", "eta", "phi", "rho" ]

    #defining the range where the rw will be performed in each variables
    if(zmmg):
        pt_min,pt_max   = 20, 50
        rho_min,rho_max = 10.0,50.0
        eta_min,eta_max = -2.501,2.501
        phi_min, phi_max = -3.15,3.15
        
        # now the number of bins in each distribution
        pt_bins,rho_bins,eta_bins,phi_bins = 10, 30, 10, 1 #8, 25, 6, 1 #was 20, 30, 10, 4        
    else:
        pt_min,pt_max   = 20, 120
        rho_min,rho_max = 5.0,60.0
        eta_min,eta_max = -2.501,2.501
        phi_min, phi_max = -3.15,3.15

        # now the number of bins in each distribution
        pt_bins,rho_bins,eta_bins,phi_bins = 10, 30, 10, 2 #8, 25, 6, 1 #was 20, 30, 10, 4

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
    sf_rw[ sf_rw > 5 ] = 5 # if the sf_rw is too big, we clip it to remove artifacts

    mc_weights = mc_weights* sf_rw

    # return the mc and data weights. I am return the data one here because he is now normalized
    return data_weights, mc_weights


# There is something off with the Isolation variables, specially with the ecalpfclusterIso! Fix it asap!
# This class is responsible to perform the transformation of the isolation variables!
# this is the class responsable for the isolation variables transformation
class Make_iso_continuous:
    def __init__(self, tensor, device, b = False):
        
        self.device = device

        #tensor = tensor.cpu()
        self.iso_bigger_zero  = tensor > 0 
        self.iso_equal_zero   = tensor == 0
        #self.lowest_iso_value = torch.min( tensor[self.iso_bigger_zero] )
        self.shift = 0.05
        if( b ):
            self.shift = b
        self.n_zero_events = torch.sum( self.iso_equal_zero )
        self.before_transform = tensor.clone().detach()
        #self.tensor_dtype = tensor.dtype()

    #Shift the continous part of the continous distribution to (self.shift), and then sample values for the discontinous part
    def shift_and_sample(self, tensor):
        
        # defining two masks to keep track of the events in the 0 peak and at the continous tails
        bigger_than_zero      = tensor  > 0
        tensor_zero           = tensor == 0
        self.lowest_iso_value = 0 

        tensor[ bigger_than_zero ] = tensor[ bigger_than_zero ] + self.shift -self.lowest_iso_value 
        tensor[ tensor_zero ]      = torch.tensor(np.random.triangular( left = 0. , mode = 0, right = self.shift*0.99, size = tensor[tensor_zero].size()[0]   ), dtype = tensor[ tensor_zero ].dtype )
        
        # now a log trasform is applied on top of the smoothing to stretch the events in the 0 traingular and "kill" the iso tails
        tensor = torch.log(  1e-3 + tensor ) 
        
        return tensor.to(self.device)
    
    #inverse operation of the above shift_and_sample transform
    def inverse_shift_and_sample( self,tensor , processed = False):
        #tensor = tensor.cpu()
        tensor = torch.exp( tensor ) - 1e-3

        bigger_than_shift = tensor > self.shift
        lower_than_shift  = tensor < self.shift

        tensor[ lower_than_shift  ] = 0
        tensor[ bigger_than_shift ] = tensor[ bigger_than_shift ] - self.shift #+ self.lowest_iso_value 
        

        #making sure the inverse operation brough exaclty the same tensor back!
        if( processed == True ):
            pass
        else:
            assert (abs(torch.sum(  self.before_transform - tensor )) < tensor.size()[0]*1e-6 )
            #added the tensor.size()[0]*1e-6 term due to possible numerical fluctioations!
              
        return tensor.to(self.device)



# perform the pre-processing of the data
def perform_pre_processing( input_tensor: torch.tensor, conditions_tensor: torch.tensor , path = False ):
    
    # Fist we make the isolation variables transformations
    indexes_for_iso_transform = [6,7,8,9,10,11,12,13,14] #[7,8,9,10,11,12,13,14,15] #[6,7,8,9,10,11,12,13,14] #this has to be changed once I take the energy raw out of the inputs
    vector_for_iso_constructors_mc   = []

    # creating the constructors
    for index in indexes_for_iso_transform:
            
        # since hoe has very low values, the shift value (value until traingular events are sampled) must be diferent here
        if( index == 6 ):
            vector_for_iso_constructors_mc.append( Make_iso_continuous(input_tensor[:,index] ,device = torch.device('cpu') , b= 0.001) )
        else:
            vector_for_iso_constructors_mc.append( Make_iso_continuous(input_tensor[:,index] , device = torch.device('cpu')) )

    # now really applying the transformations
    counter = 0
    for index in indexes_for_iso_transform:
            
        # transforming the training dataset 
        input_tensor[:,index] = vector_for_iso_constructors_mc[counter].shift_and_sample(input_tensor[:,index])
        counter = counter + 1

    # Now, the standartization -> The arrays are the same ones using during training
    # I should keep these files in a .txt or np file
    if(path):
        input_mean_for_std      = torch.tensor(np.load( path +  'input_means.npy' ))
        input_std_for_std       = torch.tensor(np.load( path +  'input_std.npy'))
        condition_mean_for_std  = torch.tensor(np.load( path +  'conditions_means.npy'))
        condition_std_for_std   = torch.tensor(np.load( path +  'conditions_std.npy'))
    else:
        
        input_mean_for_std = torch.tensor([ 9.1898e-01,  1.2240e-02,  8.5021e-03,  2.5207e-02, -4.0142e-06,
            8.4083e-01, -4.6114e+00, -1.6692e+00, -2.3899e+00, -1.4599e+00,
            -2.3577e+00,  5.8861e-01, -2.9755e+00, -3.8417e+00,  2.9375e-01,
            1.9295e+00])#torch.mean( input_tensor, 0 )
        
        input_std_for_std  = torch.tensor([1.1814e-01, 6.3080e-03, 3.6964e-03, 1.8374e-02, 8.8395e-05, 8.9386e-02,
            1.1864e+00, 2.1826e+00, 2.2732e+00, 2.3141e+00, 2.2895e+00, 8.0790e-01,
            2.6185e+00, 1.2168e+00, 1.3353e+00, 2.2628e+00]) #torch.std(  input_tensor, 0 )
            
        # the last element of the condition tensor is a boolean, so of couse we do not transform that xD
        condition_mean_for_std = torch.tensor([ 4.2801e+01, -1.5795e-02, -5.4693e-03,  2.4298e+01]) #torch.mean( conditions_tensor[:,:-1], 0 )
        condition_std_for_std  = torch.tensor([16.0865,  1.2137,  1.8124,  8.2033])

    # Standardizing!
    input_tensor = ( input_tensor - input_mean_for_std  )/input_std_for_std
    conditions_tensor[:,:-1] = ( conditions_tensor[:,:-1] - condition_mean_for_std )/condition_std_for_std

    return input_tensor, conditions_tensor, input_mean_for_std, input_std_for_std, condition_mean_for_std,condition_std_for_std, vector_for_iso_constructors_mc

# revert the corrected samples tranformation
def invert_pre_processing( input_tensor: torch.tensor, input_mean_for_std: torch.tensor, input_std_for_std: torch.tensor, vector_for_iso_constructors_mc) -> torch.tensor:
    
    indexes_for_iso_transform = [6,7,8,9,10,11,12,13,14]

    # inverting the standartization
    input_tensor = ( input_tensor*input_std_for_std + input_mean_for_std )

    # Now inverting the isolation transformation
    counter = 0
    for index in indexes_for_iso_transform:
            
        # now transforming the 
        input_tensor[:,index] = vector_for_iso_constructors_mc[counter].inverse_shift_and_sample(input_tensor[:,index], processed = True)

        counter = counter + 1
    
    return input_tensor

def apply_flow( input_tensor: torch.tensor, conditions_tensor: torch.tensor, flow )-> torch.tensor:
    
    """
    This function is responsable for applying the normalizing flow to MC samples
    it takes as input
    """

    # making sure flow and input tensors have the same type
    flow = flow.type(   input_tensor.dtype )
    conditions_tensor = conditions_tensor.type( input_tensor.dtype  )

    # Use cuda if avaliable - maybe this is causing the meory problems?
    device = torch.device('cpu')
    flow = flow.to(device)
    input_tensor = input_tensor.to(device)
    conditions_tensor = conditions_tensor.to(device)

    # Disabling pytorch gradient calculation so operation uses less memory and is faster
    with torch.no_grad():

        # Now the flow transformation is done!
        trans      = flow( conditions_tensor ).transform
        sim_latent = trans( input_tensor )

        conditions_tensor = torch.tensor(np.concatenate( [ conditions_tensor[:,:-1].cpu() , np.ones_like( conditions_tensor[:,0].cpu() ).reshape(-1,1) ], axis =1 )).to(device)

        trans2 = flow(conditions_tensor).transform
        samples = trans2.inv( sim_latent)    
    
    return samples


#Read the pytorch tensors stored by the readdata.py script
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

conditions_list = [ "probe_pt","probe_ScEta","probe_phi","fixedGridRhoAll"]

def main():

    # declaring the list with the var names from the ttres as global variables
    global var_list
    global conditions_list

    plot_postEE = True

    plots_from_pytorch = False
    zee_framework      = True

    if( zee_framework  ):

        print('\nZee application!\n')

        if( plots_from_pytorch  ):
            #Read the pytorch tensors stored by the readdata.py script
            
            print( '\nReading everything from the torch tensors!\n' )
            
            data_test_inputs     = torch.load( '/net/scratch_cms3a/daumann/PhD/EarlyHgg/simulation_to_data_corrections/data_reading/saved_tensors/zee_tensors/data_test_inputs.pt' )
            data_test_conditions = torch.load( '/net/scratch_cms3a/daumann/PhD/EarlyHgg/simulation_to_data_corrections/data_reading/saved_tensors/zee_tensors/data_test_conditions.pt')
            data_test_weights    = torch.load( '/net/scratch_cms3a/daumann/PhD/EarlyHgg/simulation_to_data_corrections/data_reading/saved_tensors/zee_tensors/data_test_weights.pt') 

            mc_test_inputs     = torch.load(   '/net/scratch_cms3a/daumann/PhD/EarlyHgg/simulation_to_data_corrections/data_reading/saved_tensors/zee_tensors/mc_test_inputs.pt' )
            mc_test_conditions = torch.load(   '/net/scratch_cms3a/daumann/PhD/EarlyHgg/simulation_to_data_corrections/data_reading/saved_tensors/zee_tensors/mc_test_conditions.pt')
            mc_test_weights    = torch.load(   '/net/scratch_cms3a/daumann/PhD/EarlyHgg/simulation_to_data_corrections/data_reading/saved_tensors/zee_tensors/mc_test_weights.pt')  
        
        else:

            # Read everything from dataframes!
            print( '\nReading everything from a DF!\n' )
            
            #files = glob.glob(    "/net/scratch_cms3a/daumann/HiggsDNA/v13_samples_flow_fix_2/dataF_v13/nominal/*.parquet")
            #files_2 = glob.glob(  "/net/scratch_cms3a/daumann/HiggsDNA/v13_samples_flow_fix_2/dataG_v13/nominal/*.parquet")
            #files_3 = glob.glob(  "/net/scratch_cms3a/daumann/HiggsDNA/v13_samples_flow_fix_2/dataE_v13/nominal/*.parquet")

            if( plot_postEE ):

                files   = glob.glob(  "/net/scratch_cms3a/daumann/HiggsDNA/v13_samples_2/dataE_v13/nominal/*.parquet")
                files_2 = glob.glob(  "/net/scratch_cms3a/daumann/HiggsDNA/v13_samples_2/dataG_v13/nominal/*.parquet")
                files_3 = glob.glob(  "/net/scratch_cms3a/daumann/HiggsDNA/v13_samples_2/dataF_v13/nominal/*.parquet")

                files = [files, files_2, files_3]

                data = [pd.read_parquet(f) for f in files]
                data_vector = pd.concat(data,ignore_index=True)

                # reafing the mc files
                #files = glob.glob(  "/net/scratch_cms3a/daumann/HiggsDNA/v13_samples_flow_fix_2/DY_postEE_v13/nominal/*.parquet")
                files = glob.glob(  "/net/scratch_cms3a/daumann/HiggsDNA/v13_samples_2/DY_postEE_v13/nominal/*.parquet")
                files = files[:300]

                data = [pd.read_parquet(f) for f in files]
                vector= pd.concat(data,ignore_index=True)        

            # transforming the df into pytorch tensors - already masked!
            data_test_inputs     = torch.tensor(  np.array( data_vector[var_list]  ) )
            data_test_conditions = torch.tensor(  np.concatenate(  [np.array( data_vector[conditions_list] ), np.ones(  len(data_vector)  ).reshape(-1,1)    ] , axis = 1  )  )
            data_test_weights    = np.ones( len(  data_vector ) )

            mc_test_inputs       = torch.tensor(  np.array( vector[var_list]  ) )
            mc_test_conditions   = torch.tensor(  np.concatenate(  [np.array( vector[conditions_list] ), 0*np.ones(  len(vector)  ).reshape(-1,1)    ] , axis = 1  )  )
            mc_test_weights      = np.array(vector["weight"])

            # performing the data selection!
            mask_data, mask_mc = perform_zee_selection( data_vector, vector )

            # we still need the selection, and the rw!
            data_test_weights, mc_test_weights = perform_zee_kinematics_reweighting(data_vector[conditions_list][mask_data], data_test_weights[mask_data], vector[conditions_list][mask_mc], mc_test_weights[mask_mc])

            # masking the arrays
            data_test_inputs     = data_test_inputs[mask_data]
            data_test_conditions = data_test_conditions[mask_data]
            #data_test_weights    = data_test_weights

            mc_test_inputs       = mc_test_inputs[mask_mc] 
            mc_test_conditions   = mc_test_conditions[mask_mc]
            #mc_test_weights      = mc_test_weights 

    else:
        #zmmy framework!
        print('\nZmumugamma application!!!\n')

        # Lets also compare it with data, of course!
        files = glob.glob(  "/net/scratch_cms3a/daumann/HiggsDNA/v13_zmmg/dataC_v13/*.parquet")
        files2 = glob.glob(  "/net/scratch_cms3a/daumann/HiggsDNA/v13_zmmg/dataD_v13/*.parquet")
        files3 = glob.glob(  "/net/scratch_cms3a/daumann/HiggsDNA/v13_zmmg/dataE_v13/*.parquet")

        files = [files,files2]#,files3]

        data = [pd.read_parquet(f) for f in files]
        data_vector = pd.concat(data,ignore_index=True)

        # Now reading MC!
        files = glob.glob(  "/net/scratch_cms3a/daumann/HiggsDNA/v13_zmmg/DY_preEE_v13/*.parquet")

        data = [pd.read_parquet(f) for f in files]
        vector= pd.concat(data,ignore_index=True)

        # replacing the probe of the zee list for the photons of the zmumugamma list!
        for index,item in enumerate(var_list):
            var_list[index] = item.replace('probe','photon')
        
        for index,item in enumerate(conditions_list):
            if('Rho' in item):
                conditions_list[index] = str("Rho_fixedGridRhoAll")
            else:
                conditions_list[index] = item.replace('probe','photon')

        # from df to np arrays
        data_vector["photon_hoe"] = data_vector["photon_hoe"]*data_vector["photon_energyRaw"]
        data_test_inputs          = np.array(data_vector[var_list])
        data_test_conditions      = np.array(data_vector[conditions_list])
        data_test_weights         = np.ones(  len(data_vector)  )
        

        vector["photon_hoe"] = vector["photon_hoe"]*vector["photon_energyRaw"]

        mc_test_inputs     = np.array(vector[var_list])
        mc_test_conditions = np.array(vector[conditions_list])
        mc_test_weights    = np.array(vector["weight_central"])

        # adding the boolean to the conditions vector
        mc_test_conditions = np.concatenate(  [mc_test_conditions , 0*np.ones_like(mc_test_conditions[:,0]).reshape(-1,1) ], axis = 1  )

        #masks and rw!

        # performing the data selection!
        mask_data, mask_mc = zmmg_utils.zmmg_selection( data_vector, vector )

        # we still need the selection, and the rw! 
        data_test_weights, mc_test_weights = perform_zee_kinematics_reweighting(data_vector[conditions_list][mask_data], data_test_weights[mask_data], vector[conditions_list][mask_mc], mc_test_weights[mask_mc], zmmg = True)        

        # masking the arrays
        data_test_inputs     = data_test_inputs[mask_data]
        data_test_conditions = data_test_conditions[mask_data]
        #data_test_weights    = data_test_weights

        mc_test_inputs       = mc_test_inputs[mask_mc] 
        mc_test_conditions   = mc_test_conditions[mask_mc]


    print( 'Data read! ' )

    # Energy raw is contained in this container, it is the first entry, but we dont correct it!
    store_mc_inputs     =  mc_test_inputs
    store_mc_conditions =  mc_test_conditions
    energy_raw_MC   = mc_test_inputs[:,0]
    mc_test_inputs  = mc_test_inputs[:,1:]

    mc_test_inputs     = torch.tensor(mc_test_inputs)
    mc_test_conditions = torch.tensor(mc_test_conditions) 

    # Defining the normalizing flow
    
    # Flow trained without rw
    #flow = zuko.flows.NSF( mc_test_inputs.size()[1] , context = mc_test_conditions.size()[1], bins = 8,transforms = 6, hidden_features=[256] * 2)
    #flow.load_state_dict(torch.load( '/net/scratch_cms3a/daumann/PhD/EarlyHgg/simulation_to_data_corrections/results/configuration_v13_no_rw_3/best_model_.pth', map_location=torch.device('cpu')))

    # Flow trained with rw
    flow = zuko.flows.NSF( mc_test_inputs.size()[1] , context = mc_test_conditions.size()[1], bins = 10,transforms = 5, hidden_features=[256] * 2)
    
    #postEE model
    if( plot_postEE ):
        path_means_std = '/net/scratch_cms3a/daumann/PhD/EarlyHgg/simulation_to_data_corrections/results/configuration_v13_big_stats_9_long_train/'
    else:
        #preEE model
        path_means_std = '/net/scratch_cms3a/daumann/PhD/EarlyHgg/simulation_to_data_corrections/results/configuration_v13_big_stats_preEE_4/'
    
    flow.load_state_dict(torch.load( path_means_std + 'best_model_.pth', map_location=torch.device('cpu')))

    device = torch.device('cpu')

    # Now we pre-process the data (standartization and Isolation variables transformations!)
    # Performing the transformations in the inputs and conditions arrays
    input_tensor, conditions_tensor, input_mean_for_std, input_std_for_std, condition_mean_for_std,condition_std_for_std, vector_for_iso_constructors_mc = perform_pre_processing(mc_test_inputs.to(device), mc_test_conditions.to(device), path_means_std)

    samples = apply_flow( input_tensor, conditions_tensor, flow )

    # Inverting the transformations! - test what happens if we input the original mc here!
    corrected_inputs = invert_pre_processing(samples,  input_mean_for_std, input_std_for_std, vector_for_iso_constructors_mc)

    # plot the corrected, uncorrected and data distributions
    mc_test_inputs = np.concatenate( [energy_raw_MC.reshape(-1,1), mc_test_inputs], axis = 1 )
    samples        = np.concatenate( [energy_raw_MC.reshape(-1,1), corrected_inputs  ], axis = 1 )

    for i in range( np.shape(samples)[1] ):

        mean = np.mean( np.array( data_test_inputs[:,i] )  )
        std  = np.std(  np.array( data_test_inputs[:,i] )  )

        # diferent binnings for zee and z->mumugamma!
        if( zee_framework ):

            data_hist     = hist.Hist(hist.axis.Regular(50, mean - 2.5*std, mean + 2.5*std))
            mc_hist       = hist.Hist(hist.axis.Regular(50, mean - 2.5*std, mean + 2.5*std))
            mc_corr_hist  = hist.Hist(hist.axis.Regular(50, mean - 2.5*std, mean + 2.5*std))

            if( 'Iso' in str(var_list[i])  or 'es' in str(var_list[i])  ):
                data_hist     = hist.Hist(hist.axis.Regular(50, 0.0, mean + 2.0*std))
                mc_hist       = hist.Hist(hist.axis.Regular(50, 0.0, mean + 2.0*std))
                mc_corr_hist  = hist.Hist(hist.axis.Regular(50, 0.0, mean + 2.0*std))
            elif( 'hoe' in str(var_list[i]) ):
                data_hist     = hist.Hist(hist.axis.Regular(20, 0.0, 0.06))
                mc_hist       = hist.Hist(hist.axis.Regular(20, 0.0, 0.06))
                mc_corr_hist  = hist.Hist(hist.axis.Regular(20, 0.0, 0.06))                
            elif( 'DR' in str(var_list[i]) and zee_framework):
                data_hist     = hist.Hist(hist.axis.Regular(50, 0.0, 5.0))
                mc_hist       = hist.Hist(hist.axis.Regular(50, 0.0, 5.0))
                mc_corr_hist  = hist.Hist(hist.axis.Regular(50, 0.0, 5.0))
            elif( 'DR' in str(var_list[i]) ):
                data_hist     = hist.Hist(hist.axis.Regular(50, 0.0, mean + 2.0*std))
                mc_hist       = hist.Hist(hist.axis.Regular(50, 0.0, mean + 2.0*std))
                mc_corr_hist  = hist.Hist(hist.axis.Regular(50, 0.0, mean + 2.0*std))        
            elif( 'energy' in str(var_list[i]) ):
                data_hist     = hist.Hist(hist.axis.Regular(30, 0.0, mean + 0.5*std))
                mc_hist       = hist.Hist(hist.axis.Regular(30, 0.0, mean + 0.5*std))
                mc_corr_hist  = hist.Hist(hist.axis.Regular(30, 0.0, mean + 0.5*std))

        else:
            data_hist     = hist.Hist(hist.axis.Regular(13, mean - 0.8*std, mean + 0.8*std))
            mc_hist       = hist.Hist(hist.axis.Regular(13, mean - 0.8*std, mean + 0.8*std))
            mc_corr_hist  = hist.Hist(hist.axis.Regular(13, mean - 0.8*std, mean + 0.8*std))

            if( 'Iso' in str(var_list[i])  or  'hoe' in str(var_list[i]) or 'es' in str(var_list[i])  ):
                data_hist     = hist.Hist(hist.axis.Regular(35, 0.0, mean + 1.0*std))
                mc_hist       = hist.Hist(hist.axis.Regular(35, 0.0, mean + 1.0*std))
                mc_corr_hist  = hist.Hist(hist.axis.Regular(35, 0.0, mean + 1.0*std))
            elif( 'DR' in str(var_list[i]) and zee_framework):
                data_hist     = hist.Hist(hist.axis.Regular(15, 0.0, 5.0))
                mc_hist       = hist.Hist(hist.axis.Regular(15, 0.0, 5.0))
                mc_corr_hist  = hist.Hist(hist.axis.Regular(15, 0.0, 5.0))
            elif( 'DR' in str(var_list[i]) ):
                data_hist     = hist.Hist(hist.axis.Regular(10, 0.0, mean + 1.6*std))
                mc_hist       = hist.Hist(hist.axis.Regular(10, 0.0, mean + 1.6*std))
                mc_corr_hist  = hist.Hist(hist.axis.Regular(10, 0.0, mean + 1.6*std))        
            elif( 'energy' in str(var_list[i]) ):
                data_hist     = hist.Hist(hist.axis.Regular(8, 0.0, mean + 0.5*std))
                mc_hist       = hist.Hist(hist.axis.Regular(8, 0.0, mean + 0.5*std))
                mc_corr_hist  = hist.Hist(hist.axis.Regular(8, 0.0, mean + 0.5*std))


        data_hist.fill(     np.array( data_test_inputs[:,i] )  )
        mc_hist.fill(       np.array( store_mc_inputs[:,i] )     , weight = mc_test_weights/1000.)
        mc_corr_hist.fill(  np.array( samples[:,i] )       , weight = mc_test_weights/1000.)

        if( zee_framework ):
            if( plots_from_pytorch ):
                plot_utils.plott( data_hist, mc_hist,mc_corr_hist, "/net/scratch_cms3a/daumann/PhD/EarlyHgg/simulation_to_data_corrections/standalone_application/plots/zee/stand_" + str(var_list[i]) + '.png',  xlabel = str(var_list[i]) )
            else:
                if( plot_postEE ):
                    plot_utils.plott( data_hist, mc_hist,mc_corr_hist, "/net/scratch_cms3a/daumann/PhD/EarlyHgg/simulation_to_data_corrections/standalone_application/plots/df_zee/postEE/stand_" + str(var_list[i]) + '.png',  xlabel = str(var_list[i]) )
                else:
                    plot_utils.plott( data_hist, mc_hist,mc_corr_hist, "/net/scratch_cms3a/daumann/PhD/EarlyHgg/simulation_to_data_corrections/standalone_application/plots/df_zee/preEE/stand_" + str(var_list[i]) + '.png',  xlabel = str(var_list[i]) )
        else:
            plot_utils.plott( data_hist, mc_hist,mc_corr_hist, "/net/scratch_cms3a/daumann/PhD/EarlyHgg/simulation_to_data_corrections/standalone_application/plots/zmmg/stand_" + var_list[i] + '.png',  xlabel = str(var_list[i]), zmmg = True )

    for i in range(  np.shape(mc_test_conditions )[1] -1 ):

        mean = np.mean( np.array( np.nan_to_num(data_test_conditions[:,i] ))  )
        std  = np.std(  np.array( np.nan_to_num(data_test_conditions[:,i] ))  )

        data_hist     = hist.Hist(hist.axis.Regular(50, mean - 2.0*std, mean + 2.0*std))
        mc_hist       = hist.Hist(hist.axis.Regular(50, mean - 2.0*std, mean + 2.0*std))
        mc_corr_hist  = hist.Hist(hist.axis.Regular(50, mean - 2.0*std, mean + 2.0*std))

        if( 'Iso' in str(conditions_list[i])  or 'DR' in str(conditions_list[i]) or 'hoe' in str(conditions_list[i]) or 'es' in str(conditions_list[i])  ):
            data_hist     = hist.Hist(hist.axis.Regular(50, 0.0, mean + 1.0*std))
            mc_hist       = hist.Hist(hist.axis.Regular(50, 0.0, mean + 1.0*std))
            mc_corr_hist  = hist.Hist(hist.axis.Regular(50, 0.0, mean + 1.0*std))
        elif( 'energy' in str(conditions_list[i]) ):
            data_hist     = hist.Hist(hist.axis.Regular(50, 0.0, mean + 1.5*std))
            mc_hist       = hist.Hist(hist.axis.Regular(50, 0.0, mean + 1.5*std))
            mc_corr_hist  = hist.Hist(hist.axis.Regular(50, 0.0, mean + 1.5*std))


        data_hist.fill(     np.array( data_test_conditions[:,i] )  )
        mc_hist.fill(       np.array( store_mc_conditions[:,i] )     , weight = mc_test_weights/1000.)
        mc_corr_hist.fill(  np.array( store_mc_conditions[:,i] )       , weight = mc_test_weights/1000.)

        if( zee_framework ):
            if( plots_from_pytorch ):
                plot_utils.plott( data_hist, mc_hist,mc_corr_hist, "/net/scratch_cms3a/daumann/PhD/EarlyHgg/simulation_to_data_corrections/standalone_application/plots/zee/stand_" + str(conditions_list[i]) + '.png',  xlabel = str(conditions_list[i]) )
            else:
                if( plot_postEE ):
                    plot_utils.plott( data_hist, mc_hist,mc_corr_hist, "/net/scratch_cms3a/daumann/PhD/EarlyHgg/simulation_to_data_corrections/standalone_application/plots/df_zee/postEE/stand_" + str(conditions_list[i]) + '.png',  xlabel = str(conditions_list[i]) )
                else:
                    plot_utils.plott( data_hist, mc_hist,mc_corr_hist, "/net/scratch_cms3a/daumann/PhD/EarlyHgg/simulation_to_data_corrections/standalone_application/plots/df_zee/preEE/stand_" + str(conditions_list[i]) + '.png',  xlabel = str(conditions_list[i]) )

        else:
            plot_utils.plott( data_hist, mc_hist,mc_corr_hist, "/net/scratch_cms3a/daumann/PhD/EarlyHgg/simulation_to_data_corrections/standalone_application/plots/zmmg/stand_" + str(conditions_list[i]) + '.png',  xlabel = str(conditions_list[i]) )

    # Now ploting the mvaID curve for mc,data and mc corrected!
    if( zee_framework ):
        if( plots_from_pytorch ):
            plot_utils.plot_mvaID_curve( store_mc_inputs, data_test_inputs , samples, store_mc_conditions, data_test_conditions, mc_test_weights, data_test_weights , "/net/scratch_cms3a/daumann/PhD/EarlyHgg/simulation_to_data_corrections/standalone_application/plots/zee/"    )
            plot_utils.plot_correlation_matrix_diference_barrel(torch.tensor(data_test_inputs), torch.tensor(data_test_conditions), torch.tensor(data_test_weights),  torch.tensor(store_mc_inputs), torch.tensor(store_mc_conditions), torch.tensor(mc_test_weights) , torch.tensor(samples),   "/net/scratch_cms3a/daumann/PhD/EarlyHgg/simulation_to_data_corrections/standalone_application/plots/zee/")
        else:
            
            if( plot_postEE ):
                plot_utils.plot_mvaID_curve( store_mc_inputs, data_test_inputs , samples, store_mc_conditions, data_test_conditions, mc_test_weights, data_test_weights , "/net/scratch_cms3a/daumann/PhD/EarlyHgg/simulation_to_data_corrections/standalone_application/plots/df_zee/postEE/"    )
                plot_utils.plot_correlation_matrix_diference_barrel(torch.tensor(data_test_inputs), torch.tensor(data_test_conditions), torch.tensor(data_test_weights),  torch.tensor(store_mc_inputs), torch.tensor(store_mc_conditions), torch.tensor(mc_test_weights) , torch.tensor(samples),   "/net/scratch_cms3a/daumann/PhD/EarlyHgg/simulation_to_data_corrections/standalone_application/plots/df_zee/postEE/")
            else:
                plot_utils.plot_mvaID_curve( store_mc_inputs, data_test_inputs , samples, store_mc_conditions, data_test_conditions, mc_test_weights, data_test_weights , "/net/scratch_cms3a/daumann/PhD/EarlyHgg/simulation_to_data_corrections/standalone_application/plots/df_zee/postEE/"    )
                plot_utils.plot_correlation_matrix_diference_barrel(torch.tensor(data_test_inputs), torch.tensor(data_test_conditions), torch.tensor(data_test_weights),  torch.tensor(store_mc_inputs), torch.tensor(store_mc_conditions), torch.tensor(mc_test_weights) , torch.tensor(samples),   "/net/scratch_cms3a/daumann/PhD/EarlyHgg/simulation_to_data_corrections/standalone_application/plots/df_zee/postEE/")    
    else:
        plot_utils.plot_mvaID_curve( store_mc_inputs, data_test_inputs, samples, store_mc_conditions, data_test_conditions, mc_test_weights, data_test_weights, "/net/scratch_cms3a/daumann/PhD/EarlyHgg/simulation_to_data_corrections/standalone_application/plots/zmmg/" , zmmg = True   )
        plot_utils.plot_correlation_matrix_diference_barrel(torch.tensor(data_test_inputs), torch.tensor(data_test_conditions), torch.tensor(data_test_weights),  torch.tensor(store_mc_inputs), torch.tensor(store_mc_conditions), torch.tensor(mc_test_weights) , torch.tensor(samples),   "/net/scratch_cms3a/daumann/PhD/EarlyHgg/simulation_to_data_corrections/standalone_application/plots/zmmg/")

if __name__ == "__main__":
    main()