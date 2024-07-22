# Script to test the application of normalizing flows outisde of the main code enviroment

# python libraries import
import os 
import pandas
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
import correlation_matrix_and_profile_plots as corr_plots
#import zmmg_process_utils     as zmmg_utils
#from   apply_flow_zmmg        import zmmg_kinematics_reweighting

def perform_zee_selection( data_df, mc_df ):
    # first we need to calculathe the invariant mass of the electron pair

    # variables names used to calculate the mass 
    data_mass_vars = ["tag_pt","tag_ScEta","tag_phi","probe_pt","probe_ScEta","probe_phi","fixedGridRhoAll", "tag_mvaID"]
    mass_vars      = ["tag_pt","tag_ScEta","tag_phi","probe_pt","probe_ScEta","probe_phi","fixedGridRhoAll", "tag_mvaID"]

    mass_inputs_data = np.array( data_df[data_mass_vars]) 
    mass_inputs_mc   = np.array( mc_df[mass_vars]  )

    # calculaitng the invariant mass with the expression of a par of massless particles 
    mass_data = np.array( data_df["mass"])  #np.sqrt(  2*mass_inputs_data[:,0]*mass_inputs_data[:,3]*( np.cosh(  mass_inputs_data[:,1] -  mass_inputs_data[:,4]  )  - np.cos( mass_inputs_data[:,2]  -mass_inputs_data[:,5] )  )  )
    mass_mc   = np.array( mc_df["mass"]  )  #np.sqrt(  2*mass_inputs_mc[:,0]*mass_inputs_mc[:,3]*( np.cosh(  mass_inputs_mc[:,1] -  mass_inputs_mc[:,4]  )  - np.cos( mass_inputs_mc[:,2]  -mass_inputs_mc[:,5] )  )  )

    # now, in order to perform the needed cuts two masks will be created
    mask_data = np.logical_and( mass_data > 80 , mass_data < 100  )
    
    mask_mc   = np.logical_and( mass_mc > 80 , mass_mc < 100  )

    # return the masks for further operations
    return mask_data, mask_mc


def calculate_bins_position(array, num_bins=12):

    array_sorted = np.sort(array)  # Ensure the array is sorted
    n = len(array)
    
    # Calculate the exact number of elements per bin
    elements_per_bin = n // num_bins
    
    # Adjust bin_indices to accommodate for numpy's 0-indexing and avoid out-of-bounds access
    bin_indices = [i*elements_per_bin for i in range(1, num_bins)]
    
    # Find the array values at these adjusted indices
    bin_edges = array_sorted[bin_indices]

    bin_edges = np.insert(bin_edges, 0, np.min(array))
    bin_edges = np.append(bin_edges, np.max(array))
    
    return bin_edges

# Due to the diferences in the kinematic distirbutions of the data and MC a reweithing must be performed to account for this
def perform_reweighting(simulation_df, data_df):
    
    # Reading and normalizing the weights
    mc_weights = np.array(simulation_df["weight"])
    mc_weights = mc_weights/np.sum( mc_weights )

    data_weights = np.ones(len(data_df["probe_pt"]))
    data_weights = data_weights/np.sum( data_weights )

    # Defining the reweigthing binning! - Bins were chossen such as each bin has ~ the same number of events
    pt_bins  = calculate_bins_position(np.array(simulation_df["probe_pt"]), 40)
    eta_bins = calculate_bins_position(np.array(simulation_df["probe_ScEta"]), 40)
    rho_bins = calculate_bins_position(np.nan_to_num(np.array(simulation_df["fixedGridRhoAll"])), 40) #np.linspace( 5,65, 30) #calculate_bins_position(np.nan_to_num(np.array(simulation_df["fixedGridRhoAll"])), 70)

    bins = [ pt_bins , eta_bins, rho_bins ]

    # Calculate 3D histograms
    data1 = [ np.array(simulation_df["probe_pt"]) , np.array(simulation_df["probe_ScEta"]) , np.array(simulation_df["fixedGridRhoAll"])]
    data2 = [ np.array(data_df["probe_pt"])       , np.array(data_df["probe_ScEta"])       , np.array(data_df["fixedGridRhoAll"])]

    hist1, edges = np.histogramdd(data1, bins=bins  , weights=mc_weights   , density=True)
    hist2, _     = np.histogramdd(data2, bins=edges , weights=data_weights , density=True)

    # Compute reweighing factors
    reweight_factors = np.divide(hist2, hist1, out=np.zeros_like(hist1), where=hist1!=0)

    # Find bin indices for each point in data1
    bin_indices = np.vstack([np.digitize(data1[i], bins=edges[i]) - 1 for i in range(3)]).T

    # Ensure bin indices are within valid range
    for i in range(3):
        bin_indices[:,i] = np.clip(bin_indices[:,i], 0, len(edges[i]) - 2  )
        
    # Apply reweighing factors
    simulation_weights = mc_weights * reweight_factors[bin_indices[:,0], bin_indices[:,1], bin_indices[:,2]]

    # normalizing both to one!
    data_weights       = data_weights/np.sum( data_weights )
    simulation_weights = simulation_weights/np.sum( simulation_weights )

    return data_weights, simulation_weights


#Read the pytorch tensors stored by the readdata.py script
"""
var_list_corr = [   "probe_energyRaw",
                    "probe_corr_r9", 
                    "probe_corr_sieie",
                    "probe_corr_etaWidth",
                    "probe_corr_phiWidth",
                    "probe_corr_sieip",
                    "probe_corr_s4",
                    "probe_corr_hoe",
                    "probe_corr_ecalPFClusterIso",
                    "probe_corr_trkSumPtHollowConeDR03",
                    "probe_corr_trkSumPtSolidConeDR04",
                    "probe_corr_pfChargedIso",
                    "probe_corr_pfChargedIsoWorstVtx",
                    "probe_corr_esEffSigmaRR",
                    "probe_corr_esEnergyOverRawE",
                    "probe_corr_hcalPFClusterIso",
                    "probe_corr_energyErr",
                    "sigma_m_over_m_corr",
                    "probe_corr_mvaID_run3"]
"""

# For Pauls's validation

"""
var_list_corr = ["probe_energyRaw",
                "probe_r9_corr", 
                "probe_sieie_corr",
                "probe_etaWidth_corr",
                "probe_phiWidth_corr",
                "probe_sieip_corr",
                "probe_s4_corr",
                "probe_hoe_corr",
                "probe_ecalPFClusterIso_corr",
                "probe_trkSumPtHollowConeDR03_corr",
                "probe_trkSumPtSolidConeDR04_corr",
                "probe_pfChargedIso_corr",
                "probe_pfChargedIsoWorstVtx_corr",
                "probe_esEffSigmaRR_corr",
                "probe_esEnergyOverRawE_corr",
                "probe_hcalPFClusterIso_corr",
                "probe_energyErr_corr",
                "probe_mvaID_corr",
                "sigma_m_over_m_corr",
                "probe_pt",
                "probe_eta",
                "fixedGridRhoAll"]


data_var_list    = ["probe_energyRaw",
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
                    "probe_energyErr",
                    "probe_mvaID",
                    "sigma_m_over_m",
                    "probe_pt",
                    "probe_eta",
                    "fixedGridRhoAll"
                    ]

var_list    = [ "probe_energyRaw",
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
                "probe_energyErr",
                "probe_mvaID",
                "sigma_m_over_m",
                "probe_pt",
                "probe_eta",
                "fixedGridRhoAll"
                ]
"""


"""
var_list_corr = ["probe_energyRaw",
                "probe_pt", 
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
                "probe_pfChargedIsoWorstVtx",
                "probe_esEffSigmaRR",
                "probe_esEnergyOverRawE",
                "probe_hcalPFClusterIso",
                "probe_mvaID",
                "fixedGridRhoAll",
                "probe_ScEta",
                "probe_energyErr",
                "probe_phi"]
""" 

var_list_corr = ["probe_energyRaw",
                "probe_pt", 
                "probe_r9_corr", 
                "probe_sieie_corr",
                "probe_etaWidth_corr",
                "probe_phiWidth_corr",
                "probe_sieip_corr",
                "probe_s4_corr",
                "probe_hoe_corr",
                "probe_ecalPFClusterIso_corr",
                "probe_trkSumPtHollowConeDR03_corr",
                "probe_trkSumPtSolidConeDR04_corr",
                "probe_pfChargedIsoWorstVtx_corr",
                "probe_esEffSigmaRR_corr",
                "probe_esEnergyOverRawE_corr",
                "probe_hcalPFClusterIso_corr",
                "probe_mvaID_corr",
                "fixedGridRhoAll",
                "probe_ScEta",
                "probe_energyErr_corr",
                "probe_phi"]


data_var_list    = ["probe_energyRaw",
                    "probe_pt", 
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
                    "probe_pfChargedIsoWorstVtx",
                    "probe_esEffSigmaRR",
                    "probe_esEnergyOverRawE",
                    "probe_hcalPFClusterIso",
                    "probe_mvaID",
                    "fixedGridRhoAll",
                    "probe_ScEta",
                    "probe_energyErr",
                    "probe_phi"
                    ]


var_list    = [ "probe_energyRaw",
                "probe_pt", 
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
                "probe_raw_pfChargedIsoWorstVtx",
                "probe_raw_esEffSigmaRR",
                "probe_raw_esEnergyOverRawE",
                "probe_raw_hcalPFClusterIso",
                "probe_mvaID_nano",
                "fixedGridRhoAll",
                "probe_ScEta",
                "probe_raw_energyErr",
                "probe_phi"
                ]


data_conditions_list = [ "probe_pt","probe_ScEta","probe_phi","fixedGridRhoAll"]
conditions_list = [ "probe_pt","probe_ScEta","probe_phi","fixedGridRhoAll"]


def read_data():
    
    # Reading data!
    files = glob.glob("/net/scratch_cms3a/daumann/normalizing_flows_project/script_to_prepare_samples_for_paper/splited_parquet/data_test.parquet")

    data = [pd.read_parquet(f) for f in files]
    data_df = pd.concat(data,ignore_index=True)

    #data_df["probe_energyErr"] = data_df["probe_energyErr"]/ data_df["probe_pt"]*np.cosh( data_df["probe_eta"] ) 

    postEE_files = glob.glob("../Zee_out_Fix_AR.parquet") 
    #preEE_files  = glob.glob("/net/scratch_cms3a/daumann/HiggsDNA/scripts/v13_runners/zee_files/Rerun_after_probe_selection/DY_preEE_v13/nominal/*.parquet")

    postEE = [pd.read_parquet(f) for f in postEE_files]
    postEE = pd.concat(postEE,ignore_index=True)  

    #preEE = [pd.read_parquet(f) for f in preEE_files]
    #preEE = pd.concat(preEE,ignore_index=True)  
    
    # Now lets scale the weights by the pre and post EE luminosities
    #preEE["weight"]  = preEE["weight"]*8.1
    postEE["weight"] = postEE["weight"]*27.0
    
    mc_df = postEE
    #mc_df = pd.concat([preEE, postEE], ignore_index=True)
    
    #mc_df["probe_energyErr"]     = mc_df["probe_energyErr"]/ mc_df["probe_pt"]*np.cosh( mc_df["probe_eta"] )
    #mc_df["probe_raw_energyErr"] = mc_df["probe_raw_energyErr"]/ mc_df["probe_pt"]*np.cosh( mc_df["probe_eta"] )
    
    print(  len(data_df) , len(mc_df) )
    
    return data_df, mc_df

def main():
   
    PostEE_plots = True

    data_df, mc_df = read_data()

    # performing the data selection!
    mask_data, mask_mc = perform_zee_selection( data_df, mc_df )

    # This carries the corrected inputs
    mc_df = mc_df[mask_mc]

    mc_test_weights      = np.array( mc_df["rw_weights"] )
    
    data_df = data_df[mask_data]
    data_vector = np.array(data_df[data_var_list])
    data_test_weights    = np.ones( len(data_df["probe_r9"]))
    mc_test_weights      = len(data_test_weights)*mc_test_weights/np.sum(mc_test_weights)

    samples              = np.array( mc_df[var_list_corr]  )
    

    #mc_test_weights      = np.array( mc_df["weight"])
    mc_vector = np.array(mc_df[var_list] )
    
    # As a first trick, we plot the diferrence in correlation wrt to data!

    """ 
    if(With_coupling_blocks):
        if( PostEE_plots ):
            plot_utils.plot_correlation_matrix_diference_barrel( torch.tensor(data_vector) , torch.tensor(np.array(data_df[conditions_list])), data_test_weights, torch.tensor(np.array(mc_df[var_list])) , torch.tensor(np.array(mc_df[conditions_list])), mc_test_weights, torch.tensor(np.array(mc_df[var_list_corr])) ,  './plots/coupling/postEE/')
        else:
            plot_utils.plot_correlation_matrix_diference_barrel( torch.tensor(data_vector) , torch.tensor(np.array(data_df[conditions_list])), data_test_weights, torch.tensor(np.array(mc_df[var_list])) , torch.tensor(np.array(mc_df[conditions_list])), mc_test_weights, torch.tensor(np.array(mc_df[var_list_corr])) ,  './plots/coupling/preEE/')
    else:
        if( PostEE_plots ):
            plot_utils.plot_correlation_matrix_diference_barrel( torch.tensor(data_vector) , torch.tensor(np.array(data_df[conditions_list])), data_test_weights, torch.tensor(np.array(mc_df[var_list])) , torch.tensor(np.array(mc_df[conditions_list])), mc_test_weights, torch.tensor(np.array(mc_df[var_list_corr])) ,  './plots/postEE/')
        else:
            plot_utils.plot_correlation_matrix_diference_barrel( torch.tensor(data_vector) , torch.tensor(np.array(data_df[conditions_list])), data_test_weights, torch.tensor(np.array(mc_df[var_list])) , torch.tensor(np.array(mc_df[conditions_list])), mc_test_weights, torch.tensor(np.array(mc_df[var_list_corr])) ,  './plots/preEE/')
    """

    corr_plots.plot_correlation_matrices( torch.tensor(np.nan_to_num(np.array( data_vector )))[:5000000], torch.tensor(np.nan_to_num(np.array( mc_vector )))[:5000000], torch.tensor(np.nan_to_num( np.array(samples) ))[:5000000]  ,  torch.tensor(np.array(mc_test_weights[:5000000])), var_list, './plots/')
    corr_plots.plot_profile_barrel( nl_mva_ID = torch.tensor(mc_df["probe_mvaID_corr"].values), mc_mva_id = torch.tensor(mc_df["probe_mvaID_nano"].values) , mc_conditions = torch.tensor(mc_df[conditions_list].values) ,  data_mva_id = torch.tensor(data_df["probe_mvaID"].values), data_conditions = torch.tensor(data_df[conditions_list].values), mc_weights = mc_test_weights, data_weights = data_test_weights, path = './plots/')
    #corr_plots.pt_profile_again(predictions, test, labels, path_to_plot, mc_weights, var='pt')

    for i in range( np.shape(samples)[1] ):

        mean = np.mean( np.nan_to_num(np.array( data_vector[:,i] ))  )
        std  = np.std(  np.nan_to_num(np.array( data_vector[:,i] ))  )

        if( 'Iso' in str(var_list[i])    ):
            data_hist     = hist.new.Reg(30, 0.0, mean + 4.5*std, overflow=True).Weight() 
            mc_hist       = hist.new.Reg(30, 0.0, mean + 4.5*std, overflow=True).Weight() 
            mc_corr_hist  = hist.new.Reg(30, 0.0, mean + 4.5*std, overflow=True).Weight() 
        elif( 'es' in str(var_list[i]) ):
            data_hist     = hist.new.Reg(30, 0.0, mean + 2.8*std, overflow=True).Weight() 
            mc_hist       = hist.new.Reg(30, 0.0, mean + 2.8*std, overflow=True).Weight() 
            mc_corr_hist  = hist.new.Reg(30, 0.0, mean + 2.8*std, overflow=True).Weight()           
        elif( 'hoe' in str(var_list[i]) ):
            data_hist     = hist.new.Reg(30, 0.0, 0.08, overflow=True).Weight() 
            mc_hist       = hist.new.Reg(30, 0.0, 0.08, overflow=True).Weight() 
            mc_corr_hist  = hist.new.Reg(30, 0.0, 0.08, overflow=True).Weight() 
        elif( 'DR' in str(var_list[i]) ):
            data_hist     = hist.new.Reg(30, 0.0, 4.0, overflow=True).Weight() 
            mc_hist       = hist.new.Reg(30, 0.0, 4.0, overflow=True).Weight() 
            mc_corr_hist  = hist.new.Reg(30, 0.0, 4.0, overflow=True).Weight()        
        elif( 'energy' in str(var_list[i]) ):
            data_hist     = hist.new.Reg(36, 0.0, mean + 3.0*std, overflow=True).Weight() 
            mc_hist       = hist.new.Reg(36, 0.0, mean + 3.0*std, overflow=True).Weight() 
            mc_corr_hist  = hist.new.Reg(36, 0.0, mean + 3.0*std, overflow=True).Weight() 
        elif( 'pt' in str(var_list[i]) or 'Rho' in str(var_list[i])):
            data_hist     = hist.new.Reg(50, mean - 4*std, mean + 4*std, overflow=True).Weight() 
            mc_hist       = hist.new.Reg(50, mean - 4*std, mean + 4*std, overflow=True).Weight() 
            mc_corr_hist  = hist.new.Reg(50, mean - 4*std, mean + 4*std, overflow=True).Weight() 
        elif( 'eta' in str(var_list[i]) and 'idth' not in str(var_list[i]) ):
            data_hist     = hist.new.Reg(50, -2.5, 2.5, overflow=True).Weight() 
            mc_hist       = hist.new.Reg(50, -2.5, 2.5, overflow=True).Weight() 
            mc_corr_hist  = hist.new.Reg(50, -2.5, 2.5, overflow=True).Weight()    
        elif( 'sieie' in str(var_list[i]) ):
            data_hist     = hist.new.Reg(40, mean - 1.5*std, mean + 3.2*std, overflow=True).Weight() 
            mc_hist       = hist.new.Reg(40, mean - 1.5*std, mean + 3.2*std, overflow=True).Weight() 
            mc_corr_hist  = hist.new.Reg(40, mean - 1.5*std, mean + 3.2*std, overflow=True).Weight()             
        elif( 'r9' in str(var_list[i]) ):
            data_hist     = hist.new.Reg(30, 0.5, mean + 1.5*std, overflow=True).Weight() 
            mc_hist       = hist.new.Reg(30, 0.5, mean + 1.5*std, overflow=True).Weight() 
            mc_corr_hist  = hist.new.Reg(30, 0.5, mean + 1.5*std, overflow=True).Weight() 
        elif(  's4' in str(var_list[i]) ):    
            data_hist     = hist.new.Reg(30, mean - 4.5*std, mean + 2.5*std, overflow=True).Weight() 
            mc_hist       = hist.new.Reg(30, mean - 4.5*std, mean + 2.5*std, overflow=True).Weight() 
            mc_corr_hist  = hist.new.Reg(30, mean - 4.5*std, mean + 2.5*std, overflow=True).Weight() 
        else:
            data_hist     = hist.new.Reg(40, mean - 2.0*std, mean + 3.0*std, overflow=True).Weight() 
            mc_hist       = hist.new.Reg(40, mean - 2.0*std, mean + 3.0*std, overflow=True).Weight() 
            mc_corr_hist  = hist.new.Reg(40, mean - 2.0*std, mean + 3.0*std, overflow=True).Weight() 

        data_barrel_mask = np.abs(data_df["probe_ScEta"]) < 2.5442
        mc_barrel_mask   = np.abs(mc_df["probe_ScEta"])   < 2.5442

        data_hist.fill(     np.array( np.array(data_vector[:,i][data_barrel_mask] )) )
        mc_hist.fill(       np.array( np.array(mc_vector[:,i][mc_barrel_mask] )) , weight = len( np.array(data_vector[:,i][data_barrel_mask] ))*mc_test_weights[mc_barrel_mask])
        mc_corr_hist.fill(  np.array( np.array(samples[:,i][mc_barrel_mask]))    , weight = len( np.array(data_vector[:,i][data_barrel_mask] ))*mc_test_weights[mc_barrel_mask])

        if( PostEE_plots ):
                plot_utils.plott( data_hist, mc_hist,mc_corr_hist, "./plots_AR_model/postEE/" + str(var_list[i]) + '.pdf',  xlabel = str(var_list[i].replace("raw_","")), postEE = True, endcap=False )
        else:
                plot_utils.plott( data_hist, mc_hist,mc_corr_hist, "./plots/preEE/" + str(var_list[i]) + '.pdf',  xlabel = str(var_list[i].replace("raw_","")), postEE = False, endcap=False )


        if( 'mvaID' in str(var_list[i]) ):
            
            # plots for barrel and endcao
            if(PostEE_plots):
                data_hist     = hist.new.Reg(40, -0.9, 1.0, overflow=True).Weight() 
                mc_hist       = hist.new.Reg(40, -0.9, 1.0, overflow=True).Weight() 
                mc_corr_hist  = hist.new.Reg(40, -0.9, 1.0, overflow=True).Weight() 
            else:
                data_hist     = hist.new.Reg(40, -0.9, 1.0, overflow=True).Weight() 
                mc_hist       = hist.new.Reg(40, -0.9, 1.0, overflow=True).Weight() 
                mc_corr_hist  = hist.new.Reg(40, -0.9, 1.0, overflow=True).Weight() 

            data_barrel_mask = np.abs(data_df["probe_ScEta"]) < 1.442
            mc_barrel_mask   = np.abs(mc_df["probe_ScEta"]) < 1.442

            #print( len(mc_barrel_mask ), len(mc_vector[:,i]), len(mc_test_weights) )
            data_hist.fill(     np.array( np.array(data_vector[:,i][data_barrel_mask] )) )
            mc_hist.fill(       np.array( np.array(mc_vector[:,i][mc_barrel_mask] ))   , weight = len( np.array(data_vector[:,i][data_barrel_mask] ))*mc_test_weights[mc_barrel_mask])
            mc_corr_hist.fill(  np.array( np.array(samples[:,i][mc_barrel_mask]))  , weight = len( np.array(data_vector[:,i][data_barrel_mask] ))*mc_test_weights[mc_barrel_mask])


            if( PostEE_plots ):
                plot_utils.plott( data_hist, mc_hist,mc_corr_hist, "./plots_AR_model/postEE/barrel_" + str(var_list[i]) + '.pdf',  xlabel = str(var_list[i]), postEE = True, endcap=False )
            else:
                plot_utils.plott( data_hist, mc_hist,mc_corr_hist, "./plots/preEE/barrel_" + str(var_list[i]) + '.pdf',  xlabel = str(var_list[i]), postEE = False, endcap=False )

            #################
            # Now, endcap!
            data_hist     = hist.new.Reg(40, -0.9, 1.0, overflow=True).Weight() 
            mc_hist       = hist.new.Reg(40, -0.9, 1.0, overflow=True).Weight() 
            mc_corr_hist  = hist.new.Reg(40, -0.9, 1.0, overflow=True).Weight() 

            data_barrel_mask = np.abs(data_df["probe_ScEta"]) > 1.56
            mc_barrel_mask   = np.abs(mc_df["probe_ScEta"])   > 1.56

            data_hist.fill(     np.array( np.array(data_vector[:,i][data_barrel_mask] )) )
            mc_hist.fill(       np.array( np.array(mc_vector[:,i][mc_barrel_mask] ))   , weight = len( np.array(data_vector[:,i][data_barrel_mask] ))*mc_test_weights[mc_barrel_mask])
            mc_corr_hist.fill(  np.array( np.array(samples[:,i][mc_barrel_mask]))  , weight = len( np.array(data_vector[:,i][data_barrel_mask] ))*mc_test_weights[mc_barrel_mask])

            if( PostEE_plots ):
                plot_utils.plott( data_hist, mc_hist,mc_corr_hist, "./plots_AR_model/postEE/endcap_" + str(var_list[i]) + '.pdf',  xlabel = str(var_list[i]), postEE = True, endcap=True )                
            else:
                plot_utils.plott( data_hist, mc_hist,mc_corr_hist, "./plots/preEE/endcap_" + str(var_list[i]) + '.pdf',  xlabel = str(var_list[i]), postEE = False, endcap=True )

if __name__ == "__main__":
    main()