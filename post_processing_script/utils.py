"""
This script is a helper to the apply_flow_to_parquet.py file
where it contain auxiliary functions thta will be called during to apply the flows
"""

# python libraries import
import os 
import numpy as np
import glob
import torch
import pandas as pd
import zuko
from typing import Any, Dict, List, Optional, Tuple
import xgboost
import awkward

import matplotlib.pyplot as plt 
import mplhep, hist
plt.style.use([mplhep.style.CMS])

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
    input_mean_for_std      = torch.tensor(np.load( path +  'input_means.npy' ))
    input_std_for_std       = torch.tensor(np.load( path +  'input_std.npy'))
    condition_mean_for_std  = torch.tensor(np.load( path +  'conditions_means.npy'))
    condition_std_for_std   = torch.tensor(np.load( path +  'conditions_std.npy'))

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

def load_photonid_mva_run3(fname: str) -> Optional[xgboost.Booster]:

    """ Reads and returns both the EB and EE Xgboost run3 mvaID models """

    photonid_mva_EB = xgboost.Booster()
    photonid_mva_EB.load_model(fname + 'model.json')

    photonid_mva_EE = xgboost.Booster()
    photonid_mva_EE.load_model(fname + 'model_endcap.json')
    
    return photonid_mva_EB, photonid_mva_EE


def calculate_photonid_mva_run3(
    mva: Tuple[Optional[xgboost.Booster], List[str]],
    photon: awkward.Array,
) -> awkward.Array:
   
    """Recompute PhotonIDMVA on-the-fly. This step is necessary considering that the inputs have to be corrected
    with the QRC process. Following is the list of features (barrel has 12, endcap two more):
    EB:
        events.Photon.energyRaw
        events.Photon.r9
        events.Photon.sieie
        events.Photon.etaWidth
        events.Photon.phiWidth
        events.Photon.sieip
        events.Photon.s4
        events.photon.hoe
        probe_ecalPFClusterIso
        probe_trkSumPtHollowConeDR03
        probe_trkSumPtSolidConeDR04
        probe_pfChargedIso
        probe_pfChargedIsoWorstVtx
        events.Photon.ScEta
        events.fixedGridRhoAll

    EE: + 
        events.Photon.energyRaw
        events.Photon.r9
        events.Photon.sieie
        events.Photon.etaWidth
        events.Photon.phiWidth
        events.Photon.sieip
        events.Photon.s4
        events.photon.hoe
        probe_ecalPFClusterIso
        probe_hcalPFClusterIso
        probe_trkSumPtHollowConeDR03
        probe_trkSumPtSolidConeDR04
        probe_pfChargedIso
        probe_pfChargedIsoWorstVtx
        events.Photon.ScEta
        events.fixedGridRhoAll    
        events.Photon.esEffSigmaRR
        events.Photon.esEnergyOverRawE
    """
    photonid_mva, var_order = mva

    if photonid_mva is None:
        return awkward.ones_like(photon.pt)


    bdt_inputs = {}
    
    bdt_inputs = np.column_stack(
        [np.array(photon[name]) for name in var_order]
    )

    tempmatrix = xgboost.DMatrix(bdt_inputs)

    mvaID = photonid_mva.predict(tempmatrix)

    # Only needed to compare to TMVA
    mvaID = 1.0 - 2.0 / (1.0 + np.exp(2.0 * mvaID))

    return mvaID


def add_corr_photonid_mva_run3( photons: awkward.Array, process) -> awkward.Array:

        preliminary_path = '/net/scratch_cms3a/daumann/HiggsDNA/higgs_dna/tools/'
        photonid_mva_EB, photonid_mva_EE = load_photonid_mva_run3(preliminary_path)

        # Now mvaID for the corrected variables

        inputs_EB = ["energyRaw",
            "r9_corr", 
            "sieie_corr",
            "etaWidth_corr",
            "phiWidth_corr",
            "sieip_corr",
            "s4_corr",
            "hoe_corr",
            "ecalPFClusterIso_corr",
            "trkSumPtHollowConeDR03_corr",
            "trkSumPtSolidConeDR04_corr",
            "pfChargedIso_corr",
            "pfChargedIsoWorstVtx_corr",
            "ScEta",
            "fixedGridRhoAll"]

        inputs_EE = ["energyRaw",
            "r9_corr", 
            "sieie_corr",
            "etaWidth_corr",
            "phiWidth_corr",
            "sieip_corr",
            "s4_corr",
            "hoe_corr",
            "ecalPFClusterIso_corr",
            "hcalPFClusterIso_corr",
            "trkSumPtHollowConeDR03_corr",
            "trkSumPtSolidConeDR04_corr",
            "pfChargedIso_corr",
            "pfChargedIsoWorstVtx_corr",
            "ScEta",
            "fixedGridRhoAll",
            "esEffSigmaRR_corr",
            "esEnergyOverRawE_corr"]

        photon_types_Zee = ["tag","probe"]
        corrected_mva_id = []

        for photon_type in photon_types_Zee:

            # Now calculating the corrected mvaID
            isEB = awkward.to_numpy(np.abs( np.array( photons[ photon_type + "_ScEta"])) < 1.5)

            inputs_EB_corr = [photon_type + "_" + s if "fixedGridRhoAll" not in s else s for s in inputs_EB]

            corr_mva_EB = calculate_photonid_mva_run3(
                [photonid_mva_EB,inputs_EB_corr], photons
            )
            
            inputs_EE_corr = [photon_type + "_" + s if "fixedGridRhoAll" not in s else s for s in inputs_EE]
            
            corr_mva_EE = calculate_photonid_mva_run3(
                [photonid_mva_EE, inputs_EE_corr], photons
            )
            corrected_mva_id.append( awkward.where(isEB, corr_mva_EB, corr_mva_EE) )

        return corrected_mva_id[0], corrected_mva_id[1]

