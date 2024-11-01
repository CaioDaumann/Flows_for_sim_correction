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

# For the paper with the IC the selection is performed in another script, so dont have to perform it again here
def perform_zee_selection( data_df, mc_df ):

    mask_data = np.logical_and( data_df["mass"].values > 80 , data_df["mass"].values < 100  )
    mask_mc   = np.logical_and( mc_df["mass"].values > 80 , mc_df["mass"].values < 100  )

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



# My post processing script has a diferrent naming w.r.t the HiggsDNA variables
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
conditions_list      = [ "probe_pt","probe_ScEta","probe_phi","fixedGridRhoAll"]

def read_data():
    
    # Reading data!
    files = glob.glob("/net/scratch_cms3a/daumann/normalizing_flows_project/script_to_prepare_samples_for_paper/splited_parquet/data_test.parquet")

    data = [pd.read_parquet(f) for f in files]
    data_df = pd.concat(data,ignore_index=True)


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
    
    return data_df, mc_df

def main():
   
    PostEE_plots = True

    data_df, mc_df = read_data()

    # performing the data selection!
    mask_data, mask_mc = perform_zee_selection( data_df, mc_df )

    mc_df, data_df  = mc_df[mask_mc], data_df[mask_data]
    
    data_vector    = np.array(data_df[data_var_list])
    samples              = np.array( mc_df[var_list_corr]  )    
    mc_vector = np.array(mc_df[var_list] )    
    
    data_test_weights    = np.ones( len(data_df["probe_r9"]))
    mc_test_weights      = np.array( mc_df["rw_weights"] )
    
    # Normalizing the mc weights to data
    mc_test_weights      = len(data_test_weights)*mc_test_weights/np.sum(mc_test_weights)

    # producing the correlation and profile plots
    corr_plots.plot_nominal_correlation_matrices( torch.tensor(np.nan_to_num(np.array( data_vector )))[:5000000], torch.tensor(np.nan_to_num(np.array( mc_vector )))[:5000000], torch.tensor(np.nan_to_num( np.array(samples) ))[:5000000]  ,  torch.tensor(np.array(mc_test_weights[:5000000])), var_list, './plots_AR_model/')
    corr_plots.plot_correlation_matrices( torch.tensor(np.nan_to_num(np.array( data_vector )))[:5000000], torch.tensor(np.nan_to_num(np.array( mc_vector )))[:5000000], torch.tensor(np.nan_to_num( np.array(samples) ))[:5000000]  ,  torch.tensor(np.array(mc_test_weights[:5000000])), var_list, './plots_AR_model/')
    corr_plots.plot_profile_barrel( nl_mva_ID = torch.tensor(mc_df["probe_mvaID_corr"].values), mc_mva_id = torch.tensor(mc_df["probe_mvaID_nano"].values) , mc_conditions = torch.tensor(mc_df[conditions_list].values) ,  data_mva_id = torch.tensor(data_df["probe_mvaID"].values), data_conditions = torch.tensor(data_df[conditions_list].values), mc_weights = mc_test_weights, data_weights = data_test_weights, path = './plots_AR_model/')
    corr_plots.scatter_plot( mc_df = mc_df,  data_df = data_df, mc_weights = mc_test_weights, data_weights = data_test_weights, path = './plots_AR_model/')

    eta_regions = ['barrel', 'endcap']
    eta_mc_masks   = [  np.abs(mc_df["probe_ScEta"]) < 1.442, np.abs(mc_df["probe_ScEta"]) > 1.566 ]
    eta_data_masks = [  np.abs(data_df["probe_ScEta"]) < 1.442, np.abs(data_df["probe_ScEta"]) > 1.566 ] 
    for entry, entry_corr, entry_data in zip(var_list, var_list_corr, data_var_list):
        for eta_region, mc_eta_mask, data_eta_mask in zip(eta_regions, eta_mc_masks, eta_data_masks):

            if eta_region == 'barrel':
                endcap = False
            else:
                endcap = True
            
            # Lets calculate the data means and stds to define the plots ranges
            mean = np.mean( np.nan_to_num( data_df[entry_data][data_eta_mask].values )  )
            std  = np.std(  np.nan_to_num( data_df[entry_data][data_eta_mask].values )  )

            if std == 0:
                std = 0.1

            # Each histogram type will have different bpunding
            if( 'Iso' in entry    ):
                data_hist     = hist.new.Reg(30, 0.0, mean + 4.5*std, overflow=True).Weight() 
                mc_hist       = hist.new.Reg(30, 0.0, mean + 4.5*std, overflow=True).Weight() 
                mc_corr_hist  = hist.new.Reg(30, 0.0, mean + 4.5*std, overflow=True).Weight() 
            elif( 'es' in entry ):
                data_hist     = hist.new.Reg(30, 0.0, mean + 2.8*std, overflow=True).Weight() 
                mc_hist       = hist.new.Reg(30, 0.0, mean + 2.8*std, overflow=True).Weight() 
                mc_corr_hist  = hist.new.Reg(30, 0.0, mean + 2.8*std, overflow=True).Weight()           
            elif( 'hoe' in entry ):
                data_hist     = hist.new.Reg(30, 0.0, 0.08, overflow=True).Weight() 
                mc_hist       = hist.new.Reg(30, 0.0, 0.08, overflow=True).Weight() 
                mc_corr_hist  = hist.new.Reg(30, 0.0, 0.08, overflow=True).Weight() 
            elif( 'DR' in entry ):
                data_hist     = hist.new.Reg(70, 0.0, 4.0, overflow=True).Weight() 
                mc_hist       = hist.new.Reg(70, 0.0, 4.0, overflow=True).Weight() 
                mc_corr_hist  = hist.new.Reg(70, 0.0, 4.0, overflow=True).Weight()        
            elif( 'energy' in entry ):
                data_hist     = hist.new.Reg(36, 0.0, mean + 3.0*std, overflow=True).Weight() 
                mc_hist       = hist.new.Reg(36, 0.0, mean + 3.0*std, overflow=True).Weight() 
                mc_corr_hist  = hist.new.Reg(36, 0.0, mean + 3.0*std, overflow=True).Weight() 
            elif( 'pt' in entry or 'Rho' in entry):
                data_hist     = hist.new.Reg(50, mean - 4*std, mean + 4*std, overflow=True).Weight() 
                mc_hist       = hist.new.Reg(50, mean - 4*std, mean + 4*std, overflow=True).Weight() 
                mc_corr_hist  = hist.new.Reg(50, mean - 4*std, mean + 4*std, overflow=True).Weight() 
            elif( 'eta' in entry and 'idth' not in entry ):
                data_hist     = hist.new.Reg(50, -2.5, 2.5, overflow=True).Weight() 
                mc_hist       = hist.new.Reg(50, -2.5, 2.5, overflow=True).Weight() 
                mc_corr_hist  = hist.new.Reg(50, -2.5, 2.5, overflow=True).Weight()    
            elif( 'sieie' in entry  ):
                data_hist     = hist.new.Reg(30, mean - 3.0*std, mean + 3.0*std, overflow=True).Weight() 
                mc_hist       = hist.new.Reg(30, mean - 3.0*std, mean + 3.0*std, overflow=True).Weight() 
                mc_corr_hist  = hist.new.Reg(30, mean - 3.0*std, mean + 3.0*std, overflow=True).Weight()             
            elif( 'sieip' in entry  ):
                data_hist     = hist.new.Reg(30, mean - 3.0*std, mean + 3.0*std, overflow=True).Weight() 
                mc_hist       = hist.new.Reg(30, mean - 3.0*std, mean + 3.0*std, overflow=True).Weight() 
                mc_corr_hist  = hist.new.Reg(30, mean - 3.0*std, mean + 3.0*std, overflow=True).Weight() 
            elif( 'r9' in entry ):
                data_hist     = hist.new.Reg(30, 0.5, mean + 1.5*std, overflow=True).Weight() 
                mc_hist       = hist.new.Reg(30, 0.5, mean + 1.5*std, overflow=True).Weight() 
                mc_corr_hist  = hist.new.Reg(30, 0.5, mean + 1.5*std, overflow=True).Weight() 
            elif(  's4' in entry ):    
                data_hist     = hist.new.Reg(30, mean - 4.5*std, mean + 2.5*std, overflow=True).Weight() 
                mc_hist       = hist.new.Reg(30, mean - 4.5*std, mean + 2.5*std, overflow=True).Weight() 
                mc_corr_hist  = hist.new.Reg(30, mean - 4.5*std, mean + 2.5*std, overflow=True).Weight() 
            else:
                data_hist     = hist.new.Reg(30, mean - 2.0*std, mean + 3.0*std, overflow=True).Weight() 
                mc_hist       = hist.new.Reg(30, mean - 2.0*std, mean + 3.0*std, overflow=True).Weight() 
                mc_corr_hist  = hist.new.Reg(30, mean - 2.0*std, mean + 3.0*std, overflow=True).Weight() 

            data_hist.fill(     np.array( np.array(data_df[entry_data][data_eta_mask] )) )
            mc_hist.fill(       np.array( np.array(mc_df[entry][mc_eta_mask] ))        , weight = len( np.array(data_df[entry_data][data_eta_mask] ))*mc_test_weights[mc_eta_mask])
            mc_corr_hist.fill(  np.array( np.array(mc_df[entry_corr][mc_eta_mask]))    , weight = len( np.array(data_df[entry_data][data_eta_mask] ))*mc_test_weights[mc_eta_mask])
            
            plot_utils.plott( data_hist, mc_hist,mc_corr_hist, f'./plots_AR_model/forPaper/{entry_data}_{eta_region}.pdf',  xlabel = str(entry.replace("_raw","").replace("_nano","")), postEE = True, endcap = endcap)


            # Now only the mvaID
            if( 'mvaID' in entry_data ):
                
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
                data_hist.fill(     np.array( np.array(data_df[entry_data][data_barrel_mask] )) )
                mc_hist.fill(       np.array( np.array(mc_df[entry][mc_barrel_mask] ))   , weight = len( np.array(data_df[entry_data][data_barrel_mask] ))*mc_test_weights[mc_barrel_mask])
                mc_corr_hist.fill(  np.array( np.array(mc_df[entry_corr][mc_barrel_mask]))  , weight = len( np.array(data_df[entry_data][data_barrel_mask] ))*mc_test_weights[mc_barrel_mask])

                plot_utils.plott( data_hist, mc_hist,mc_corr_hist, "./plots_AR_model/forPaper/barrel_" + entry_data + '.pdf',  xlabel = entry_data, postEE = True, endcap=False )

                #################
                # Now, endcap!
                data_hist     = hist.new.Reg(40, -0.9, 1.0, overflow=True).Weight() 
                mc_hist       = hist.new.Reg(40, -0.9, 1.0, overflow=True).Weight() 
                mc_corr_hist  = hist.new.Reg(40, -0.9, 1.0, overflow=True).Weight() 

                data_barrel_mask = np.abs(data_df["probe_ScEta"]) > 1.56
                mc_barrel_mask   = np.abs(mc_df["probe_ScEta"])   > 1.56

                data_hist.fill(     np.array( np.array(data_df[entry_data][data_barrel_mask] )) )
                mc_hist.fill(       np.array( np.array(mc_df[entry][mc_barrel_mask] ))   , weight = len( np.array(data_df[entry_data][data_barrel_mask] ))*mc_test_weights[mc_barrel_mask])
                mc_corr_hist.fill(  np.array( np.array(mc_df[entry_corr][mc_barrel_mask]))  , weight = len( np.array(data_df[entry_data][data_barrel_mask] ))*mc_test_weights[mc_barrel_mask])


                plot_utils.plott( data_hist, mc_hist,mc_corr_hist, "./plots_AR_model/forPaper/endcap_" + entry_data + '.pdf',  xlabel = entry_data, postEE = True, endcap=True )                



    exit()
    # old script from here below!

    # Loop over the variables to produce the marginal plots
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
            data_hist     = hist.new.Reg(130, 0.0, 4.0, overflow=True).Weight() 
            mc_hist       = hist.new.Reg(130, 0.0, 4.0, overflow=True).Weight() 
            mc_corr_hist  = hist.new.Reg(130, 0.0, 4.0, overflow=True).Weight()        
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