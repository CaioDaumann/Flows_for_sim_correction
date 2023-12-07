# script made to plot the main validation and resulting distributions

# importing modules
import torch
import numpy as np
import matplotlib.pyplot as plt
import mplhep, hist
plt.style.use([mplhep.style.CMS])
import mplhep as hep
import xgboost

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
        yerr=True,
        density = True,
        color="black",
        #linewidth=3,
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
    if( "Iso" in str(xlabel) or "DR" in str(xlabel) or 'r9' in str(xlabel) or 's4' in str(xlabel)   ):
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
    ax[1].set_ylim(0.5, 1.5)
    if( 'mva' in xlabel ):
        ax[1].set_ylim(0.75, 1.25)

    ax[0].legend(
        loc="upper right", fontsize=24
    )

    hep.cms.label(data=True, ax=ax[0], loc=0, label="Private Work", com=13.6, lumi=21.7)

    #plt.subplots_adjust(hspace=0.03)

    plt.tight_layout()

    fig.savefig(output_filename)

    return 0

# This is a plot distribution suitable for dataframes, if one wants to plot tensors see "plot_distributions_for_tensors()"
def plot_distributions( path, data_df, mc_df, mc_weights, variables_to_plot ):

    for set in variables_to_plot:

        for key in set:

            mean = np.mean( np.array(data_df[key]) )
            std  = np.std(  np.array(data_df[key]) )

            if( 'Iso' in key or 'DR' in key   ):
                data_hist            = hist.Hist(hist.axis.Regular(100, 0.0, 1.5))
                mc_hist              = hist.Hist(hist.axis.Regular(100, 0.0, 1.5))
                mc_rw_hist           = hist.Hist(hist.axis.Regular(100, 0.0, 1.5))
            elif( 'hoe' in  key ):
                data_hist            = hist.Hist(hist.axis.Regular(100, 0.0, 0.02))
                mc_hist              = hist.Hist(hist.axis.Regular(100, 0.0, 0.02))
                mc_rw_hist           = hist.Hist(hist.axis.Regular(100, 0.0, 0.02))
            else:
                data_hist            = hist.Hist(hist.axis.Regular(70, mean - 2.0*std, mean + 2.0*std))
                mc_hist              = hist.Hist(hist.axis.Regular(70, mean - 2.0*std, mean + 2.0*std))
                mc_rw_hist           = hist.Hist(hist.axis.Regular(70, mean - 2.0*std, mean + 2.0*std))

            data_hist.fill( np.array(data_df[key]   )  )
            mc_hist.fill( np.array(  mc_df[key]) )

            #print( np.shape( np.array(drell_yan_df[key]) ) , np.shape( mc_weights  ) )

            mc_rw_hist.fill( np.array(mc_df[key]) , weight = mc_weights )

            plott( data_hist , mc_hist, mc_rw_hist , 'plots/' +  str(key) +".png", xlabel = str(key)  )

def plot_distributions_for_tensors( data_tensor, mc_tensor, flow_samples, mc_weights ):

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


    for i in range( np.shape( data_tensor )[1] ):

            mean = np.mean( np.array(data_tensor[:,i]) )
            std  = np.std(  np.array(data_tensor[:,i]) )

            data_hist            = hist.Hist(hist.axis.Regular(70, mean - 2.0*std, mean + 2.0*std))
            mc_hist              = hist.Hist(hist.axis.Regular(70, mean - 2.0*std, mean + 2.0*std))
            mc_rw_hist           = hist.Hist(hist.axis.Regular(70, mean - 2.0*std, mean + 2.0*std))

            data_hist.fill( np.array(data_tensor[:,i]   )  )
            mc_hist.fill(  np.array( mc_tensor[:,i]),  weight = 1e6*mc_weights )

            #print( np.shape( np.array(drell_yan_df[key]) ) , np.shape( mc_weights  ) )

            mc_rw_hist.fill( np.array( flow_samples[:,i]) , weight = 1e6*mc_weights )

            plott( data_hist , mc_hist, mc_rw_hist , 'plots/results/' +  str(var_list[i]) +".png", xlabel = str(var_list[i])  )

def plot_loss_cruve(training,validation):

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

        plt.savefig('plots/loss_plot.png') 
        plt.close()

def plot_mvaID_curve(mc_inputs,data_inputs,nl_inputs, mc_conditions, data_conditions,mc_weights, data_weights):
    
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

    # now, we create and fill the histograms with the mvaID distributions
    mc_mva      = hist.Hist(hist.axis.Regular(60, -0.9, 1.0))
    nl_mva      = hist.Hist(hist.axis.Regular(60, -0.9, 1.0))
    data_mva    = hist.Hist(hist.axis.Regular(60, -0.9, 1.0))

    mc_mva.fill( mc_mvaID, weight = (1e6)*mc_weights )
    nl_mva.fill( nl_mvaID, weight = (1e6)*mc_weights )
    data_mva.fill( data_mvaID, weight = (1e6)*data_weights  )

    plott( data_mva , mc_mva, nl_mva , 'plots/results/mvaID_barrel.png', xlabel = "Barrel mvaID"  )


def plot_mvaID_curve_endcap(mc_inputs,data_inputs,nl_inputs, mc_conditions, data_conditions,mc_weights, data_weights):
    
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

    # now, we create and fill the histograms with the mvaID distributions
    mc_mva      = hist.Hist(hist.axis.Regular(60, -0.9, 1.0))
    nl_mva      = hist.Hist(hist.axis.Regular(60, -0.9, 1.0))
    data_mva    = hist.Hist(hist.axis.Regular(60, -0.9, 1.0))

    mc_mva.fill( mc_mvaID, weight = (1e6)*mc_weights )
    nl_mva.fill( nl_mvaID, weight = (1e6)*mc_weights )
    data_mva.fill( data_mvaID, weight = (1e6)*data_weights  )

    plott( data_mva , mc_mva, nl_mva , 'plots/results/mvaID_endcap.png', xlabel = "End cap mvaID"  )
