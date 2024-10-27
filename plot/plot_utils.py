# script made to plot the main validation and resulting distributions

# importing modules
import torch
import numpy as np
import matplotlib.pyplot as plt
import mplhep, hist
plt.style.use([mplhep.style.CMS])
import mplhep as hep
import xgboost

# Names of the used variables, I copied it here only so it is easier to use it acess the labels and names of teh distirbutions
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

# some variables have value of zero in barrel, so we must exclude them. I created this matrix so it is easier to do that!
var_list_matrix_barrel = ["probe_energyRaw",
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
            "probe_energyErr"]

# The next three functions are related to the plotting of the profiles of the MVA as a function of the kinematical variables
# This function calculate the means of the input quantiles! - Weighted mean, of couse.
def weighted_quantiles_interpolate(values, weights, quantiles=0.5):
    i = np.argsort(values)
    c = np.cumsum(weights[i])
    return values[i[np.searchsorted(c, np.array(quantiles) * c[-1])]]

# this is the main plotting function, all the other will basically set up something to call this one in the end!
def plott(data_hist,mc_hist,mc_rw_hist ,output_filename,xlabel,region=None  ):

    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
    
    hep.histplot(
        mc_hist,
        label = r'$Z\rightarrow ee$',
        yerr=True,
        density = True,
        color = "blue",
        linewidth=3,
        ax=ax[0]
    )

    hep.histplot(
        mc_rw_hist,
        label = r'$Z\rightarrow ee$ Corr',
        density = True,
        color = "red",
        linewidth=3,
        ax=ax[0]
    )

    hep.histplot(
        data_hist,
        label = "Data",
        density = True,
        color="black",
        linewidth=3,
        histtype='errorbar',
        markersize=12,
        elinewidth=3,
        ax=ax[0]
    )


    ax[0].set_xlabel('')
    ax[0].margins(y=0.15)
    ax[0].set_ylim(0, 1.05*ax[0].get_ylim()[1])
    ax[0].tick_params(labelsize=22)

    #log scale for Iso variables
    if( "Iso" in str(xlabel) or "DR" in str(xlabel) or "esE" in str(xlabel)   ): # or 'r9' in str(xlabel) or 's4' in str(xlabel)
        ax[0].set_yscale('log')
        #ax[0].set_ylim(0.001,( np.max(data_hist)/1.5e6 ))
        ax[0].set_ylim(0.001, 10.05*ax[0].get_ylim()[1])
        

    # line at 1
    ax[1].axhline(1, 0, 1, label=None, linestyle='--', color="black", linewidth=1)#, alpha=0.5)

    #ratio
    data_hist_numpy = data_hist.to_numpy()
    mc_hist_numpy   = mc_hist.to_numpy()
    mc_hist_rw_numpy   = mc_rw_hist.to_numpy()

    integral_data = data_hist.sum() * (data_hist_numpy[1][1] - data_hist_numpy[1][0])
    integral_mc = mc_hist.sum() * (mc_hist_numpy[1][1] - mc_hist_numpy[1][0])

    #ratio betwenn normalizng flows prediction and data
    ratio = (data_hist_numpy[0] / integral_data) / ( (mc_hist_numpy[0] + 1e-15 ) / integral_mc)
    ratio = np.nan_to_num(ratio)

    integral_mc_rw = mc_rw_hist.sum() * (mc_hist_rw_numpy[1][1] - mc_hist_rw_numpy[1][0])
    ratio_rw = (data_hist_numpy[0] / integral_data) / ( (mc_hist_rw_numpy[0] +1e-15 ) / integral_mc_rw)
    ratio_rw = np.nan_to_num(ratio_rw)

    errors_nom = (np.sqrt(data_hist_numpy[0])/integral_data) / ( (mc_hist_numpy[0] + 1e-15 ) / integral_mc)
    errors_nom = np.abs(np.nan_to_num(errors_nom))

    hep.histplot(
        ratio,
        bins=data_hist_numpy[1],
        label=None,
        color="blue",
        histtype='errorbar',
        yerr=errors_nom,
        markersize=12,
        elinewidth=3,
        alpha=1,
        ax=ax[1]
    )


    hep.histplot(
        ratio_rw,
        bins=data_hist_numpy[1],
        label=None,
        color="red",
        histtype='errorbar',
        yerr=errors_nom,
        markersize=12,
        elinewidth=3,
        alpha=1,
        ax=ax[1]
    )

    ax[0].set_ylabel("Fraction of events / GeV", fontsize=26)
    #ax[1].set_ylabel("Data / MC", fontsize=26)
    #ax.set_xlabel( str(xlabel), fontsize=26)
    ax[1].set_ylabel("Data / MC", fontsize=26)
    ax[1].set_xlabel( str(xlabel) , fontsize=26)
    if region:
        if not "ZpT" in region:
            ax[0].text(0.05, 0.75, "Region: " + region.replace("_", "-"), fontsize=22, transform=ax[0].transAxes)
        else:
            ax[0].text(0.05, 0.75, "Region: " + region.split("_ZpT_")[0].replace("_", "-"), fontsize=22, transform=ax[0].transAxes)
            ax[0].text(0.05, 0.68, r"$p_\mathrm{T}(Z)$: " + region.split("_ZpT_")[1].replace("_", "-") + "$\,$GeV", fontsize=22, transform=ax[0].transAxes)
    ax[0].tick_params(labelsize=24)
    #ax.set_ylim(0., 1.1*ax.get_ylim()[1])
    ax[1].set_ylim(0.71, 1.29)
    if( 'mva' in xlabel ):
        ax[1].set_ylim(0.79, 1.21)

    ax[0].legend(
        loc="upper right", fontsize=24
    )

    hep.cms.label(data=True, ax=ax[0], loc=0, label="Private Work", com=13.6, lumi=21.7)

    #plt.subplots_adjust(hspace=0.03)

    plt.tight_layout()

    fig.savefig(output_filename)

    return 0

# This is a plot distribution suitable for dataframes, if one wants to plot tensors see "plot_distributions_for_tensors()"
def plot_distributions( path, data_df, mc_df, data_weights, mc_weights, variables_to_plot, weights_befores_rw = False ):

    for set in variables_to_plot:

        for key in set:

            mean = np.mean( np.nan_to_num(np.array(data_df[key.replace('_raw','')])) )
            std  = np.std(  np.nan_to_num(np.array(data_df[key.replace('_raw','')])) )

            if( 'Iso' in key or 'DR' in key   ):
                data_hist            = hist.Hist(hist.axis.Regular(100, 0.0, 3.5))
                mc_hist              = hist.Hist(hist.axis.Regular(100, 0.0, 3.5))
                mc_rw_hist           = hist.Hist(hist.axis.Regular(100, 0.0, 3.5))
            elif( 'hoe' in  key ):
                data_hist            = hist.Hist(hist.axis.Regular(100, 0.0, 0.08))
                mc_hist              = hist.Hist(hist.axis.Regular(100, 0.0, 0.08))
                mc_rw_hist           = hist.Hist(hist.axis.Regular(100, 0.0, 0.08))
            else:
                data_hist            = hist.Hist(hist.axis.Regular(100, mean - 3.0*std, mean + 4.0*std))
                mc_hist              = hist.Hist(hist.axis.Regular(100, mean - 3.0*std, mean + 4.0*std))
                mc_rw_hist           = hist.Hist(hist.axis.Regular(100, mean - 3.0*std, mean + 4.0*std))

            data_hist.fill( np.array(data_df[key.replace('_raw','')]   ) , weight = data_weights )
            
            if( len(weights_befores_rw)  ):
                mc_hist.fill( np.array(  mc_df[key]), weight = weights_befores_rw )
            else:
                mc_hist.fill( np.array(  mc_df[key]) )

            #print( np.shape( np.array(drell_yan_df[key]) ) , np.shape( mc_weights  ) )

            mc_rw_hist.fill( np.array(mc_df[key]) , weight = mc_weights )

            plott( data_hist , mc_hist, mc_rw_hist , path +  str(key) +".png", xlabel = str(key)  )

def plot_distributions_for_tensors( data_tensor, mc_tensor, flow_samples, mc_weights, plot_path, variables_list ):

    for i in range( np.shape( data_tensor )[1] ):

            mean = np.mean( np.array(data_tensor[:,i]) )
            std  = np.std(  np.array(data_tensor[:,i]) )

            if( 'Iso' in str(variables_list[i]) or 'DR' in str(variables_list[i]) or 'esE' in str(variables_list[i]) or 'hoe' in str(variables_list[i]) or 'energy' in str(variables_list[i])  ):
                data_hist            = hist.Hist(hist.axis.Regular(50, 0.0 , mean + 2.0*std))
                mc_hist              = hist.Hist(hist.axis.Regular(50, 0.0 , mean + 2.0*std))
                mc_rw_hist           = hist.Hist(hist.axis.Regular(50, 0.0 , mean + 2.0*std))
            else:
                data_hist            = hist.Hist(hist.axis.Regular(50, mean - 2.5*std, mean + 2.5*std))
                mc_hist              = hist.Hist(hist.axis.Regular(50, mean - 2.5*std, mean + 2.5*std))
                mc_rw_hist           = hist.Hist(hist.axis.Regular(50, mean - 2.5*std, mean + 2.5*std))

            if( 'DR04' in str(variables_list[i])  ):
                data_hist            = hist.Hist(hist.axis.Regular(50, 0.0 , 5.0))
                mc_hist              = hist.Hist(hist.axis.Regular(50, 0.0 , 5.0))
                mc_rw_hist           = hist.Hist(hist.axis.Regular(50, 0.0 , 5.0))

            data_hist.fill( np.array(data_tensor[:,i]   )  )
            mc_hist.fill(  np.array( mc_tensor[:,i]),  weight = 1e6*mc_weights )
            mc_rw_hist.fill( np.array( flow_samples[:,i]) , weight = 1e6*mc_weights )

            plott( data_hist , mc_hist, mc_rw_hist , plot_path +  str(variables_list[i]) +".png", xlabel = str(variables_list[i])  )

def plot_loss_cruve(training,validation, plot_path):

        fig, ax1 = plt.subplots()

        # Plot training loss on the first axis
        color = 'tab:blue'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Training Loss', color=color)
        ax1.plot(training, color=color, marker='o', label='Training Loss')
        ax1.plot(validation, color='tab:orange', marker='x', label='Validation Loss')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend()

        # Title and show the plot
        plt.title('Training Loss and MVA_chsqrd')

        plt.savefig( plot_path + 'loss_plot.png') 
        plt.close()

def plot_mvaID_curve(mc_inputs,data_inputs,nl_inputs, mc_conditions, data_conditions,mc_weights, data_weights, plot_path):
    
    model_path = './run3_mvaID_models/'

    photonid_mva = xgboost.Booster()
    photonid_mva.load_model( model_path + "model.json" )

    # The mvaID model are separted into barrel and endcap, we first evaluate the barrel here. So, we make a cut in eta
    mask_mc,mask_data = np.abs(mc_conditions[:,1]) < 1.4222, np.abs(data_conditions[:,1]) < 1.4222

    # apply the barrel only condition
    data_inputs, mc_inputs, nl_inputs  =  data_inputs[mask_data]    , mc_inputs[mask_mc]     ,nl_inputs[mask_mc]
    data_conditions,mc_conditions      = data_conditions[mask_data] , mc_conditions[mask_mc]
    data_weights,mc_weights            = data_weights[mask_data]    , mc_weights[mask_mc]

    #mkaing sure there are no inf or nan's on the tensors
    np.nan_to_num(nl_inputs, nan=0.0, posinf = 0.0, neginf = 0.0)
    np.nan_to_num(mc_conditions, nan=0.0, posinf = 0.0, neginf = 0.0)

    data_tempmatrix = xgboost.DMatrix( np.concatenate( [data_inputs[:,:-4] , data_conditions[:,1].reshape(-1,1) , data_conditions[:,3].reshape(-1,1) ], axis =1 ) )
    nl_tempmatrix   = xgboost.DMatrix( np.concatenate( [nl_inputs[:,:-4]   , mc_conditions[:,1].reshape(-1,1)   , mc_conditions[:,3].reshape(-1,1) ], axis =1 ) )
    mc_tempmatrix   = xgboost.DMatrix( np.concatenate( [mc_inputs[:,:-4]   , mc_conditions[:,1] .reshape(-1,1)  , mc_conditions[:,3].reshape(-1,1) ], axis =1 ) )

    #evaluating the network!
    data_mvaID = photonid_mva.predict(data_tempmatrix)
    nl_mvaID = photonid_mva.predict(nl_tempmatrix)
    mc_mvaID = photonid_mva.predict(mc_tempmatrix)

    # needed transformation. See more details oin HiggsDNA -> https://gitlab.cern.ch/HiggsDNA-project/HiggsDNA/-/blob/master/higgs_dna/tools/photonid_mva.py?ref_type=heads#L56
    data_mvaID = 1 - (2/(1+np.exp( 2*data_mvaID )))
    nl_mvaID   = 1 - (2/(1+np.exp( 2*nl_mvaID )))
    mc_mvaID   = 1 - (2/(1+np.exp( 2*mc_mvaID )))

    plot_profile_barrel( nl_mvaID, mc_mvaID ,mc_conditions,  data_mvaID, data_conditions, mc_weights, data_weights, plot_path)

    # now, we create and fill the histograms with the mvaID distributions
    mc_mva      = hist.Hist(hist.axis.Regular(42, -0.9, 1.0))
    nl_mva      = hist.Hist(hist.axis.Regular(42, -0.9, 1.0))
    data_mva    = hist.Hist(hist.axis.Regular(42, -0.9, 1.0))

    mc_mva.fill( mc_mvaID, weight = (1e6)*mc_weights )
    nl_mva.fill( nl_mvaID, weight = (1e6)*mc_weights )
    data_mva.fill( data_mvaID, weight = (1e6)*data_weights  )

    plott( data_mva , mc_mva, nl_mva , plot_path + '/mvaID_barrel.png', xlabel = "Barrel mvaID"  )


def plot_mvaID_curve_endcap(mc_inputs,data_inputs,nl_inputs, mc_conditions, data_conditions,mc_weights, data_weights, plot_path):
    
    model_path = './run3_mvaID_models/'

    photonid_mva = xgboost.Booster()
    photonid_mva.load_model( model_path + "model_endcap.json" )

    # The mvaID model are separted into barrel and endcap, we first evaluate the barrel here. So, we make a cut in eta
    mask_mc,mask_data = np.abs(mc_conditions[:,1]) > 1.56, np.abs(data_conditions[:,1]) > 1.56

    # apply the barrel only condition
    data_inputs, mc_inputs, nl_inputs  =  data_inputs[mask_data]    , mc_inputs[mask_mc]     ,nl_inputs[mask_mc]
    data_conditions,mc_conditions      = data_conditions[mask_data] , mc_conditions[mask_mc]
    data_weights,mc_weights            = data_weights[mask_data]    , mc_weights[mask_mc]

    #mkaing sure there are no inf or nan's on the tensors
    np.nan_to_num(nl_inputs, nan=0.0, posinf = 0.0, neginf = 0.0)
    np.nan_to_num(mc_conditions, nan=0.0, posinf = 0.0, neginf = 0.0)

    # since we now also have the energyErr in the input vector, we need to take one out
    data_inputs = data_inputs[:,:-1]
    mc_inputs   = mc_inputs[:,:-1]
    nl_inputs   = nl_inputs[:,:-1]

    data_hcalIso = data_inputs[:, np.shape(data_inputs)[1] -1 ]
    mc_hcalIso   = mc_inputs[:, np.shape(data_inputs)[1] -1 ]
    nl_hcalIso   = nl_inputs[:, np.shape(data_inputs)[1] -1 ]

    data_tempmatrix = xgboost.DMatrix( np.concatenate( [data_inputs[:,:9], data_hcalIso.reshape(-1,1)  , data_inputs[:,9:np.shape(data_inputs)[1] -3] , data_conditions[:,1].reshape(-1,1) , data_conditions[:,3].reshape(-1,1)  , data_inputs[:, np.shape(data_inputs)[1] -3:np.shape(data_inputs)[1] -1] ], axis =1 ) )
    nl_tempmatrix   = xgboost.DMatrix( np.concatenate( [nl_inputs[:,:9]  , nl_hcalIso.reshape(-1,1)    , nl_inputs[:,9:np.shape(data_inputs)[1] -3]   , mc_conditions[:,1].reshape(-1,1)   , mc_conditions[:,3].reshape(-1,1)    , nl_inputs[:,   np.shape(data_inputs)[1] -3:np.shape(data_inputs)[1] -1] ], axis =1 ) )
    mc_tempmatrix   = xgboost.DMatrix( np.concatenate( [mc_inputs[:,:9]  , mc_hcalIso.reshape(-1,1)    , mc_inputs[:,9:np.shape(data_inputs)[1] -3]   ,mc_conditions[:,1] .reshape(-1,1)   , mc_conditions[:,3].reshape(-1,1)    , mc_inputs[:,   np.shape(data_inputs)[1] -3:np.shape(data_inputs)[1] -1] ], axis =1 ) )

    #evaluating the network!
    data_mvaID = photonid_mva.predict(data_tempmatrix)
    nl_mvaID = photonid_mva.predict(nl_tempmatrix)
    mc_mvaID = photonid_mva.predict(mc_tempmatrix)

    # needed transformation. See more details oin HiggsDNA -> https://gitlab.cern.ch/HiggsDNA-project/HiggsDNA/-/blob/master/higgs_dna/tools/photonid_mva.py?ref_type=heads#L56
    data_mvaID = 1 - (2/(1+np.exp( 2*data_mvaID )))
    nl_mvaID   = 1 - (2/(1+np.exp( 2*nl_mvaID )))
    mc_mvaID   = 1 - (2/(1+np.exp( 2*mc_mvaID )))

    plot_profile_endcap( nl_mvaID, mc_mvaID ,mc_conditions,  data_mvaID, data_conditions, mc_weights, data_weights, plot_path)

    # now, we create and fill the histograms with the mvaID distributions
    mc_mva      = hist.Hist(hist.axis.Regular(30, -0.9, 1.0))
    nl_mva      = hist.Hist(hist.axis.Regular(30, -0.9, 1.0))
    data_mva    = hist.Hist(hist.axis.Regular(30, -0.9, 1.0))

    mc_mva.fill( mc_mvaID, weight     = (1e6)*mc_weights )
    nl_mva.fill( nl_mvaID, weight     = (1e6)*mc_weights )
    data_mva.fill( data_mvaID, weight = (1e6)*data_weights  )

    plott( data_mva , mc_mva, nl_mva , plot_path + '/mvaID_endcap.png', xlabel = "End cap mvaID"  )

def plot_distributions_after_transformations(training_inputs, training_conditions, training_weights):

    # two masks to separate events betwenn mc and data
    data_mask = training_conditions[:,-1] == 1
    mc_mask   = training_conditions[:,-1] == 0

    #print( '\n\nHere we go!\n\n' )
    #print( np.shape( training_inputs ), len( var_list ) )
    #print( '\n\nHere we go!\n\n' )

    # now we plot the distributions
    for i in range( np.shape( training_inputs[data_mask] )[1] ):

                mean = np.mean( np.array(training_inputs[data_mask][:,i]) )
                std  = np.std(  np.array(training_inputs[data_mask][:,i]) )


                data_hist            = hist.Hist(hist.axis.Regular(70, mean - 3.0*std, mean + 3.0*std))
                mc_hist              = hist.Hist(hist.axis.Regular(70, mean - 3.0*std, mean + 3.0*std))
                mc_rw_hist           = hist.Hist(hist.axis.Regular(70, mean - 3.0*std, mean + 3.0*std))

                data_hist.fill( np.array(training_inputs[data_mask][:,i]   )  )
                mc_hist.fill(  np.array( training_inputs[mc_mask][:,i]),  weight = 1e6*training_weights[mc_mask] )
                #mc_rw_hist.fill( np.array( flow_samples[:,i]) , weight = 1e6*mc_weights )

                plott( data_hist , mc_hist, mc_hist , 'plots/validation_plots/transformation/after_transform_' +  str(var_list[i]) +".png", xlabel = str(var_list[i])  )


def plot_correlation_matrix_diference_barrel(
    var_list_barrel_only, 
    match_indices_barrel,
    data, 
    data_conditions,
    data_weights,
    mc, 
    mc_conditions,
    mc_weights,
    mc_corrected,
    path
):
    """
    Plots the differences between the correlation matrices of data and Monte Carlo (MC) samples 
    in the barrel region.

    Parameters:
    - var_list_barrel_only: List[str]
        Variable names for the barrel region.
    - match_indices_barrel: List[int]
        Indices of variables corresponding to the barrel region.
    - data: np.ndarray
        Data samples (events x variables).
    - data_conditions: np.ndarray
        Conditions or event-level parameters for the data.
    - data_weights: np.ndarray
        Weights for data events.
    - mc: np.ndarray
        MC samples.
    - mc_conditions: np.ndarray
        Conditions or event-level parameters for the MC.
    - mc_weights: np.ndarray
        Weights for MC events.
    - mc_corrected: np.ndarray
        Corrected MC samples.
    - path: str
        Directory where the plots will be saved.
    """

    # Remove 'probe_' prefix from variable names
    var_list_barrel_only = [var.replace('probe_', '').replace('Cone', '').replace('trk', '').replace('Pt', '').replace('Charged', '').replace('Cluster', '') for var in var_list_barrel_only]

    # Apply barrel-only condition (|eta| < 1.4222)
    barrel_eta_cut = 1.4222
    mask_data = np.abs(data_conditions[:, 1]) < barrel_eta_cut
    mask_mc = np.abs(mc_conditions[:, 1]) < barrel_eta_cut

    # Apply the barrel condition to the data
    data = data[mask_data]
    data_weights = data_weights[mask_data]

    # Apply the barrel condition to the MC samples
    mc = mc[mask_mc]
    mc_weights = mc_weights[mask_mc]
    mc_corrected = mc_corrected[mask_mc]

    # Select only the variables related to barrel variables
    data = data[:, match_indices_barrel]
    mc = mc[:, match_indices_barrel]
    mc_corrected = mc_corrected[:, match_indices_barrel]

    # Handle negative weights by taking absolute values
    data_weights_abs = np.abs(data_weights)
    mc_weights_abs = np.abs(mc_weights)

    # Function to compute weighted covariance matrix
    def weighted_covariance(X, weights):
        average = np.average(X, axis=0, weights=weights)
        X_centered = X - average
        cov = np.cov(X_centered, rowvar=False, aweights=weights)
        return cov

    # Compute weighted covariance matrices
    data_cov = weighted_covariance(data, data_weights_abs)
    mc_cov = weighted_covariance(mc, mc_weights_abs)
    mc_corrected_cov = weighted_covariance(mc_corrected, mc_weights_abs)

    # Convert covariance matrices to correlation matrices
    def covariance_to_correlation(cov):
        diag = np.sqrt(np.diag(cov))
        outer_diag = np.outer(diag, diag)
        corr = cov / outer_diag
        corr[cov == 0] = 0
        return corr

    data_corr = covariance_to_correlation(data_cov)
    mc_corr = covariance_to_correlation(mc_cov)
    mc_corrected_corr = covariance_to_correlation(mc_corrected_cov)

    # Helper function to plot difference matrices
    def plot_difference_matrix(diff_matrix, variable_names, xlabel, output_filename):
        fig, ax = plt.subplots(figsize=(20, 20))
        cax = ax.matshow(100 * diff_matrix, cmap='bwr', vmin=-5, vmax=5)
        fig.colorbar(cax)

        mean_diff = np.mean(np.abs(diff_matrix)) * 100

        for (i, j), z in np.ndenumerate(100 * diff_matrix):
            if abs(z) >= 1:
                ax.text(j, i, f'{z:.1f}', ha='center', va='center', fontsize=20)

        ax.set_xticks(np.arange(len(variable_names)))
        ax.set_yticks(np.arange(len(variable_names)))
        ax.set_xticklabels(variable_names, fontsize=25, rotation=90)
        ax.set_yticklabels(variable_names, fontsize=25)
        ax.set_xlabel(f'{xlabel} - Metric: {mean_diff:.2f}', loc = 'center', fontsize=30)

        plt.tight_layout()
        plt.savefig(output_filename)
        plt.close()

    # Plotting difference between data and corrected MC correlation matrices
    diff_corr_corrected = data_corr - mc_corrected_corr
    plot_difference_matrix(
        diff_matrix=diff_corr_corrected,
        variable_names=var_list_barrel_only,
        xlabel='(Corr_MC_Corrected - Corr_Data)',
        output_filename=f'{path}/correlation_matrix_corrected_barrel.png'
    )

    # Plotting difference between data and uncorrected MC correlation matrices
    diff_corr = data_corr - mc_corr
    plot_difference_matrix(
        diff_matrix=diff_corr,
        variable_names=var_list_barrel_only,
        xlabel='(Corr_MC - Corr_Data)',
        output_filename=f'{path}/correlation_matrix_barrel.png'
    )

# Old one, exclude if the above one works!!
def plot_correlation_matrix_diference_barrel__(var_list_barrel_only ,data, data_conditions,data_weights,mc , mc_conditions,mc_weights,mc_corrected,path):

    # lets do this for barrel only for now ...
    mask_mc,mask_data = np.abs(mc_conditions[:,1]) < 1.4222, np.abs(data_conditions[:,1]) < 1.4222

    # apply the barrel only condition
    data, mc, mc_corrected              = data[mask_data]            , mc[mask_mc]             ,mc_corrected[mask_mc]
    data_conditions,mc_conditions      = data_conditions[mask_data] , mc_conditions[mask_mc]
    data_weights,mc_weights            = data_weights[mask_data]    , mc_weights[mask_mc]

    energy_err_data = data[:,-1:]
    energy_err_mc = mc[:,-1:]
    energy_err_mc_corrected = mc_corrected[:,-1:]

    # no energy raw in correlation matrices!
    mc           = mc[:,1: int( data.size()[1]  -6 ) ]
    mc_corrected = mc_corrected[:,1: int( data.size()[1]  -6 ) ]
    data         = data[:,1: int( data.size()[1]  -6 ) ]

    data = torch.cat( [  data, energy_err_data .view(-1,1)  ], axis = 1 )
    mc = torch.cat( [  mc, energy_err_mc .view(-1,1)  ], axis = 1 )
    mc_corrected = torch.cat( [  mc_corrected, energy_err_mc_corrected .view(-1,1)  ], axis = 1 )

    # Some weights can of course be negative, so I had to use the abs here, since it does not accept negative weights ...
    data_corr         = torch.cov( data.T         , aweights = torch.Tensor( abs(data_weights) ))
    mc_corr           = torch.cov( mc.T           , aweights = torch.Tensor( abs(mc_weights)   ))
    mc_corrected_corr = torch.cov( mc_corrected.T , aweights = torch.Tensor( abs(mc_weights)   ))

    #from covariance to correlation matrices
    data_corr         = torch.inverse( torch.sqrt( torch.diag_embed( torch.diag(data_corr))) ) @ data_corr @  torch.inverse( torch.sqrt(torch.diag_embed(torch.diag(data_corr))) ) 
    mc_corr           = torch.inverse( torch.sqrt( torch.diag_embed(torch.diag(mc_corr)) )) @ mc_corr @  torch.inverse( torch.sqrt(torch.diag_embed(torch.diag(mc_corr))) ) 
    mc_corrected_corr = torch.inverse( torch.diag_embed(torch.sqrt(torch.diag(mc_corrected_corr)) )) @ mc_corrected_corr @  torch.inverse( torch.sqrt(torch.diag_embed(torch.diag(mc_corrected_corr))) ) 
    # end of matrix evaluations, now the plotting part!

    #plloting part
    fig, ax = plt.subplots(figsize=(33,33))
    ax.matshow( 100*( data_corr - mc_corrected_corr ), cmap = 'bwr', vmin=-5, vmax=5)

    #ploting the cov matrix values
    mean,count = 0,0
    for (i, j), z in np.ndenumerate( 100*( data_corr - mc_corrected_corr )):
        mean = mean + abs(z)
        count = count + 1
        if( abs(z) < 1  ):
            pass
        else:
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', fontsize = 60)    
    
    mean = mean/count
    ax.set_xlabel(r'(Corr_MC$^{Corrected}$-Corr_data)  - Metric: ' + str(mean), loc = 'center' ,fontsize = 70)
    #plt.tight_layout()
    plt.title( mean )
    
    #var_list_matrix_barrel=var_list_matrix_barrel.replace('probe_', '')

    ax.set_xticks(np.arange(len(var_list_barrel_only)-1))
    ax.set_yticks(np.arange(len(var_list_barrel_only)-1))
    
    ax.set_xticklabels(var_list_barrel_only[1:],fontsize = 45 ,rotation=90)
    ax.set_yticklabels(var_list_barrel_only[1:],fontsize = 45 ,rotation=0)


    plt.savefig(path + '/correlation_matrix_corrected_barrel.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(33,33))
    ax.matshow( 100*( data_corr - mc_corr ), cmap = 'bwr', vmin=-5, vmax=5)   
    
    #ploting the cov matrix values
    mean,count = 0,0
    for (i, j), z in np.ndenumerate(100*( data_corr - mc_corr )):
        mean = mean + abs(z)
        count = count + 1
        if( abs(z) < 1  ):
            pass
        else:
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', fontsize = 60)  
    mean = mean/count   

    ax.set_xticks(np.arange(len(var_list_barrel_only)-1))
    ax.set_yticks(np.arange(len(var_list_barrel_only)-1))
    
    ax.set_xticklabels(var_list_barrel_only[1:],fontsize = 45,rotation=90)
    ax.set_yticklabels(var_list_barrel_only[1:],fontsize = 45,rotation=0)

    ax.set_xlabel(r'(Corr_MC-Corr_data  - Metric: ' + str(mean), loc = 'center' ,fontsize = 70)

    plt.savefig(path + '/correlation_matrix_barrel.png')

def plot_correlation_matrix_diference_endcap(data, data_conditions,data_weights,mc , mc_conditions,mc_weights,mc_corrected,path):

    # Selecting only end-cap events
    mask_mc,mask_data = np.abs(mc_conditions[:,1]) > 1.56, np.abs(data_conditions[:,1]) > 1.56

    # apply the barrel only condition
    data, mc, mc_corrected              = data[mask_data]            , mc[mask_mc]             ,mc_corrected[mask_mc]
    data_conditions,mc_conditions      = data_conditions[mask_data] , mc_conditions[mask_mc]
    data_weights,mc_weights            = data_weights[mask_data]    , mc_weights[mask_mc]
    
    # removing energy Raw from here!
    data          = data[:,1:]
    mc            = mc[:,1:]
    mc_corrected  = mc_corrected[:,1:]   

    # Some weights can of course be negative, so I had to use the abs here, since it does not accept negative weights ...
    data_corr         = torch.cov( data.T         , aweights = torch.Tensor( abs(data_weights) ))
    mc_corr           = torch.cov( mc.T           , aweights = torch.Tensor( abs(mc_weights)   ))
    mc_corrected_corr = torch.cov( mc_corrected.T , aweights = torch.Tensor( abs(mc_weights)   ))

    #from covariance to correlation matrices
    data_corr         = torch.inverse( torch.sqrt( torch.diag_embed( torch.diag(data_corr))) ) @ data_corr @  torch.inverse( torch.sqrt(torch.diag_embed(torch.diag(data_corr))) ) 
    mc_corr           = torch.inverse( torch.sqrt( torch.diag_embed(torch.diag(mc_corr)) )) @ mc_corr @  torch.inverse( torch.sqrt(torch.diag_embed(torch.diag(mc_corr))) ) 
    mc_corrected_corr = torch.inverse( torch.diag_embed(torch.sqrt(torch.diag(mc_corrected_corr)) )) @ mc_corrected_corr @  torch.inverse( torch.sqrt(torch.diag_embed(torch.diag(mc_corrected_corr))) ) 
    # end of matrix evaluations, now the plotting part!

    #plloting part
    fig, ax = plt.subplots(figsize=(33,33))
    ax.matshow( 100*( data_corr - mc_corrected_corr ), cmap = 'bwr', vmin=-5, vmax=5)

    #ploting the cov matrix values
    mean,count = 0,0
    for (i, j), z in np.ndenumerate( 100*( data_corr - mc_corrected_corr )):
        mean = mean + abs(z)
        count = count + 1
        if( abs(z) < 1  ):
            pass
        else:
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', fontsize = 60)    
    
    mean = mean/count
    ax.set_xlabel(r'(Corr_MC$^{Corrected}$-Corr_data)  - Metric: ' + str(mean), loc = 'center' ,fontsize = 70)
    #plt.tight_layout()
    plt.title( mean )

    #adding axis labels
    #ax.set_xticklabels(['']+var_names)
    #ax.set_yticklabels(['']+var_names)
    
    ax.set_xticks(np.arange(len(var_list)-1))
    ax.set_yticks(np.arange(len(var_list)-1))
    
    ax.set_xticklabels(var_list[1:],fontsize = 45 ,rotation=90)
    ax.set_yticklabels(var_list[1:],fontsize = 45 ,rotation=0)


    plt.savefig(path + '/correlation_matrix_corrected_endcap.png')

    

    plt.close()
    fig, ax = plt.subplots(figsize=(33,33))
    ax.matshow( 100*( data_corr - mc_corr ), cmap = 'bwr', vmin=-5, vmax=5)
    #sns.heatmap(data_corr, annot=True, cmap='coolwarm', fmt='.2f', square=True)
    #plt.title(f'Correlation Matrix for {key}')    
    
    #ploting the cov matrix values
    mean,count = 0,0
    for (i, j), z in np.ndenumerate(100*( data_corr - mc_corr )):
        mean = mean + abs(z)
        count = count + 1
        if( abs(z) < 1  ):
            pass
        else:
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', fontsize = 60)  
    mean = mean/count   

    ax.set_xticks(np.arange(len(var_list)-1))
    ax.set_yticks(np.arange(len(var_list)-1))
    
    ax.set_xticklabels(var_list[1:],fontsize = 45,rotation=90)
    ax.set_yticklabels(var_list[1:],fontsize = 45,rotation=0)

    ax.set_xlabel(r'(Corr_MC-Corr_data  - Metric: ' + str(mean), loc = 'center' ,fontsize = 70)

    plt.savefig(path + '/correlation_matrix_endcap.png')


#The events are binned in bins of equal number of events of each profilling variable, than the median is calculated!
def plot_profile_barrel( nl_mva_ID, mc_mva_id ,mc_conditions,  data_mva_id, data_conditions, mc_weights, data_weights,path):

    # Barrel only mask!
    mask_mc   = np.abs( mc_conditions[:,1])   < 1.442
    mask_data = np.abs( data_conditions[:,1]) < 1.442 

    nl_mva_ID = nl_mva_ID[mask_mc]
    mc_mva_id = mc_mva_id[mask_mc]
    mc_weights = mc_weights[mask_mc]
    mc_conditions = mc_conditions[mask_mc]

    data_mva_id     = data_mva_id[mask_data]
    data_conditions = data_conditions[mask_data]
    data_weights    = data_weights[mask_data]

    #lets call the function ...
    plot_mvaID_profile_barrel( nl_mva_ID,mc_mva_id,mc_conditions[:,0],data_mva_id,data_conditions[:,0],mc_weights,data_weights , path, var = 'pt' )
    plot_mvaID_profile_barrel( nl_mva_ID,mc_mva_id,mc_conditions[:,1],data_mva_id,data_conditions[:,1],mc_weights,data_weights , path, var = 'eta' )
    plot_mvaID_profile_barrel( nl_mva_ID,mc_mva_id,mc_conditions[:,2],data_mva_id,data_conditions[:,2],mc_weights,data_weights , path ,var = 'phi' )
    plot_mvaID_profile_barrel( nl_mva_ID,mc_mva_id,mc_conditions[:,3],data_mva_id,data_conditions[:,3],mc_weights,data_weights , path ,var = 'rho' )



#The events are binned in bins of equal number of events of each profilling variable, than the median is calculated!
def plot_profile_endcap( nl_mva_ID, mc_mva_id ,mc_conditions,  data_mva_id, data_conditions, mc_weights, data_weights,path):

    # Barrel only mask!
    mask_mc   = np.abs( mc_conditions[:,1])   > 1.56
    mask_data = np.abs( data_conditions[:,1]) > 1.56 

    nl_mva_ID = nl_mva_ID[mask_mc]
    mc_mva_id = mc_mva_id[mask_mc]
    mc_weights = mc_weights[mask_mc]
    mc_conditions = mc_conditions[mask_mc]

    data_mva_id     = data_mva_id[mask_data]
    data_conditions = data_conditions[mask_data]
    data_weights    = data_weights[mask_data]

    #lets call the function ...
    plot_mvaID_profile_endcap( nl_mva_ID,mc_mva_id,mc_conditions[:,0],data_mva_id,data_conditions[:,0],mc_weights,data_weights , path, var = 'pt' )
    #plot_mvaID_profile_endcap( nl_mva_ID,mc_mva_id,mc_conditions[:,1],data_mva_id,data_conditions[:,1],mc_weights,data_weights , path, var = 'eta' )
    plot_mvaID_profile_endcap( nl_mva_ID,mc_mva_id,mc_conditions[:,2],data_mva_id,data_conditions[:,2],mc_weights,data_weights , path ,var = 'phi' )
    plot_mvaID_profile_endcap( nl_mva_ID,mc_mva_id,mc_conditions[:,3],data_mva_id,data_conditions[:,3],mc_weights,data_weights , path ,var = 'rho' )



# Lets do this separatly, first, we do the plots at the barrel only!
def plot_mvaID_profile_barrel( nl_mva_ID,mc_mva_id,var_mc,data_mva_id,var_data,mc_weights,data_weights,path,var = 'pt' ):
    
    if 'pt' in var:
        bins = np.linspace( 25.0, 65.0, 14 )
    elif 'phi' in var:
        bins = np.linspace( -3.1415, 3.1415, 14)
    elif 'eta' in var:
        bins = np.linspace( -1.442, 1.442, 14 )
    elif 'rho' in var:
        bins = np.linspace( 10.0, 45.0, 14 )

    #arrays to store the 
    position, nl_mean, data_mean, mc_mean = [],[],[],[]
    nl_mean_q25, data_mean_q25, mc_mean_q25 = [],[],[]
    nl_mean_q75, data_mean_q75, mc_mean_q75 = [],[],[]

    for i in range( 0, int(len( bins ) -1) ):

        mva_nl_window     = nl_mva_ID[   ( var_mc  > bins[i]) &   ( var_mc < bins[i+1])   ]
        mva_mc_window     = mc_mva_id[   ( var_mc  > bins[i]) &   ( var_mc < bins[i+1])   ]
        mc_weights_window = mc_weights[  ( var_mc  > bins[i]) &   ( var_mc < bins[i+1])  ]

        mva_data_window     = data_mva_id[ ( var_data  > bins[i]) & ( var_data < bins[i+1])  ] 
        data_weights_window = data_weights[ ( var_data  > bins[i]) & ( var_data < bins[i+1])  ] 

        position.append(  bins[i] + (bins[i+1] -bins[i] )/2.   )
        nl_mean.append(   weighted_quantiles_interpolate( mva_nl_window      , mc_weights_window )   )   #np.median(  mva_nl_window )   )
        mc_mean.append(   weighted_quantiles_interpolate( mva_mc_window      , mc_weights_window )   )
        data_mean.append( weighted_quantiles_interpolate( mva_data_window    , data_weights_window ) )

        nl_mean_q25.append(   weighted_quantiles_interpolate( mva_nl_window  , mc_weights_window     , quantiles= 0.25  ) )   #np.median(  mva_nl_window )   )
        mc_mean_q25.append(   weighted_quantiles_interpolate( mva_mc_window  , mc_weights_window     , quantiles= 0.25  ) )
        data_mean_q25.append( weighted_quantiles_interpolate( mva_data_window, data_weights_window   , quantiles= 0.25  ) )

        nl_mean_q75.append(   weighted_quantiles_interpolate( mva_nl_window  , mc_weights_window     , quantiles = 0.75 ) )   #np.median(  mva_nl_window )   )
        mc_mean_q75.append(   weighted_quantiles_interpolate( mva_mc_window  , mc_weights_window     , quantiles = 0.75 ) )
        data_mean_q75.append( weighted_quantiles_interpolate( mva_data_window, data_weights_window   , quantiles = 0.75 ) )

    # Plotting the 3 quantiles
    plt.figure(figsize=(10, 6))
    plt.plot( position , nl_mean ,  linewidth  = 2 , color = 'red' , label = 'MC corrected' )
    plt.plot( position , mc_mean ,  linewidth  = 2 , color = 'blue'  , label = 'MC nominal'   )
    plt.plot( position , data_mean , linewidth = 2 , color = 'green', label = 'Data'  )


    plt.plot( position , nl_mean_q25 ,  linewidth  = 2 , linestyle='dashed', color = 'red'  )
    plt.plot( position , mc_mean_q25 ,  linewidth  = 2 , linestyle='dashed',color = 'blue'    )
    plt.plot( position , data_mean_q25 , linewidth = 2 , linestyle='dashed', color = 'green' )

    plt.plot( position , nl_mean_q75 ,  linewidth  = 2 , linestyle='dashed', color = 'red'  )
    plt.plot( position , mc_mean_q75 ,  linewidth  = 2 , linestyle='dashed',color = 'blue'    )
    plt.plot( position , data_mean_q75 , linewidth = 2 , linestyle='dashed', color = 'green' )


    if 'eta' in var:
        plt.xlabel( r'$\eta$' , fontsize = 25 )
    if 'phi' in var:
        plt.xlabel( r'$\phi$' ,  fontsize = 25)
    if 'rho' in var:
        plt.xlabel( r'$\rho$' ,  fontsize = 25 )
    if 'pt' in var:
        plt.xlabel( r'$p_{T}$ [GeV]', fontsize = 25 )

    plt.ylabel( 'Photon MVA ID' )
    plt.legend(fontsize=15)

    plt.ylim( 0.0 , 1.0 )

    if 'eta' in var:
        plt.ylim( 0.0 , 1.0 )
    if 'phi' in var:
        plt.ylim( 0.0 , 1.0 )
    if 'rho' in var:
        plt.ylim( 0.0 , 1.0 )

    plt.tight_layout()

    plt.savefig( path + '/profile_' + str(var) +'.png' )

    plt.close()


def plot_mvaID_profile_endcap( nl_mva_ID,mc_mva_id,var_mc,data_mva_id,var_data,mc_weights,data_weights,path,var = 'pt' ):
    
    if 'pt' in var:
        bins = np.linspace( 25.0, 80.0, 20 )
    elif 'phi' in var:
        bins = np.linspace( -3.1415, 3.1415, 20)
    elif 'eta' in var:
        bins = np.linspace( [[-2.5,-1.442],  [1.442, 2.5] ], 20 )
    elif 'rho' in var:
        bins = np.linspace( 5.0, 50.0, 20 )

    #arrays to store the 
    position, nl_mean, data_mean, mc_mean   = [],[],[],[]
    nl_mean_q25, data_mean_q25, mc_mean_q25 = [],[],[]
    nl_mean_q75, data_mean_q75, mc_mean_q75 = [],[],[]

    for i in range( 0, int(len( bins ) -1) ):

        mva_nl_window     = nl_mva_ID[   ( var_mc  > bins[i]) &   ( var_mc < bins[i+1])   ]
        mva_mc_window     = mc_mva_id[   ( var_mc  > bins[i]) &   ( var_mc < bins[i+1])   ]
        mc_weights_window = mc_weights[  ( var_mc  > bins[i]) &   ( var_mc < bins[i+1])  ]

        mva_data_window     = data_mva_id[ ( var_data  > bins[i]) & ( var_data < bins[i+1])  ] 
        data_weights_window = data_weights[ ( var_data  > bins[i]) & ( var_data < bins[i+1])  ] 

        position.append(  bins[i] + (bins[i+1] -bins[i] )/2.   )
        nl_mean.append(   weighted_quantiles_interpolate( mva_nl_window      , mc_weights_window )   )   #np.median(  mva_nl_window )   )
        mc_mean.append(   weighted_quantiles_interpolate( mva_mc_window      , mc_weights_window )   )
        data_mean.append( weighted_quantiles_interpolate( mva_data_window    , data_weights_window ) )

        nl_mean_q25.append(   weighted_quantiles_interpolate( mva_nl_window  , mc_weights_window     , quantiles= 0.25  ) )   #np.median(  mva_nl_window )   )
        mc_mean_q25.append(   weighted_quantiles_interpolate( mva_mc_window  , mc_weights_window     , quantiles= 0.25  ) )
        data_mean_q25.append( weighted_quantiles_interpolate( mva_data_window, data_weights_window   , quantiles= 0.25  ) )

        nl_mean_q75.append(   weighted_quantiles_interpolate( mva_nl_window  , mc_weights_window     , quantiles = 0.75 ) )   #np.median(  mva_nl_window )   )
        mc_mean_q75.append(   weighted_quantiles_interpolate( mva_mc_window  , mc_weights_window     , quantiles = 0.75 ) )
        data_mean_q75.append( weighted_quantiles_interpolate( mva_data_window, data_weights_window   , quantiles = 0.75 ) )

    # Plotting the 3 quantiles
    plt.figure(figsize=(10, 6))
    plt.plot( position , nl_mean ,  linewidth  = 2 , color = 'red' , label = 'MC corrected' )
    plt.plot( position , mc_mean ,  linewidth  = 2 , color = 'blue'  , label = 'MC nominal'   )
    plt.plot( position , data_mean , linewidth = 2 , color = 'green', label = 'Data'  )


    plt.plot( position , nl_mean_q25 ,  linewidth  = 2 , linestyle='dashed', color = 'red'  )
    plt.plot( position , mc_mean_q25 ,  linewidth  = 2 , linestyle='dashed',color = 'blue'    )
    plt.plot( position , data_mean_q25 , linewidth = 2 , linestyle='dashed', color = 'green' )

    plt.plot( position , nl_mean_q75 ,  linewidth  = 2 , linestyle='dashed', color = 'red'  )
    plt.plot( position , mc_mean_q75 ,  linewidth  = 2 , linestyle='dashed',color = 'blue'    )
    plt.plot( position , data_mean_q75 , linewidth = 2 , linestyle='dashed', color = 'green' )


    if 'eta' in var:
        plt.xlabel( r'$\eta$' , fontsize = 25 )
    if 'phi' in var:
        plt.xlabel( r'$\phi$' ,  fontsize = 25)
    if 'rho' in var:
        plt.xlabel( r'$\rho$' ,  fontsize = 25 )
    if 'pt' in var:
        plt.xlabel( r'$p_{T}$ [GeV]', fontsize = 25 )

    plt.ylabel( 'Photon MVA ID' )
    plt.legend(fontsize=15)

    plt.ylim( 0.0 , 1.0 )

    if 'eta' in var:
        plt.ylim( 0.0 , 1.0 )
    if 'phi' in var:
        plt.ylim( 0.0 , 1.0 )
    if 'rho' in var:
        plt.ylim( 0.0 , 1.0 )

    plt.tight_layout()

    plt.savefig( path + '/profile_endcap_' + str(var) +'.png' )

    plt.close()
