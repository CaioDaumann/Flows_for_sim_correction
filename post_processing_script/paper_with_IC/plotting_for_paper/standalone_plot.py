# script made to plot the main validation and resulting distributions

# importing modules
import torch
import numpy as np
import matplotlib.pyplot as plt
import mplhep, hist
plt.style.use([mplhep.style.CMS])
import mplhep as hep
import xgboost
import matplotlib.patches as patches

# Names of the used variables, I copied it here only so it is easier to use it acess the labels and names of teh distirbutions
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


# some variables have value of zero in barrel, so we must exclude them. I created this matrix so it is easier to do that!
var_list_matrix_barrel = ["energyRaw",
                          r'$p_{t}$',
                          "r9", 
                          "sieie",
                          "etaWidth",
                          "phiWidth",
                          "sieip",
                          "s4",
                          "hoe",
                          "ecalPFIso",
                          "trkSumPtDR03",
                          "trkSumPtDR04",
                          "ChIsoWorstVtx",
                          "EffSigmaRR",
                          "EOverRawE",
                          "hcalPFIso"]

"""
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
                    "probe_pfChargedIsoWorstVtx",
                    "probe_esEffSigmaRR",
                    "probe_esEnergyOverRawE",
                    "probe_hcalPFClusterIso",
                    "probe_energyErr"
                    ]
"""
corr_mva_list     = [   "probe_energyRaw",
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
                    "probe_ScEta",
                    "fixedGridRhoAll"]

# The next three functions are related to the plotting of the profiles of the MVA as a function of the kinematical variables
# This function calculate the means of the input quantiles! - Weighted mean, of couse.
def weighted_quantiles_interpolate(values, weights, quantiles=0.5):
    i = np.argsort(values)
    c = np.cumsum(weights[i])
    return values[i[np.searchsorted(c, np.array(quantiles) * c[-1])]]

# this is the main plotting function, all the other will basically set up something to call this one in the end!
def plott(data_hist,mc_hist,mc_rw_hist ,output_filename,xlabel, zmmg = None , postEE = True, endcap = False ):

    plt.close()
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
    
    # Check if ax[0] is indeed a matplotlib Axes object
    if not isinstance(ax[0], plt.Axes):
        raise ValueError("ax[0] must be a matplotlib Axes object")
    
    # Fill the area with hatched grey lines until 0.25 on the x-axis
    if( "mva" in str(xlabel) and 1 == 2  ):
        
            
        # Define the x and y ranges for the greyed out region
        x_start, x_end = -0.9, 0.25
        #y_start, y_end = 0, 2.0  # Adjust the y range as needed
        # Get the current y-axis limits
        y_start, y_end = ax[0].get_ylim()
            
        # Add a hatched rectangle to the plot
        # Add a hatched rectangle to the plot
        #rect = patches.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start,
        #                        linewidth=1, edgecolor='grey', facecolor='none', hatch='//', alpha=0.5)
            
        rect = patches.Rectangle((x_start, y_start), x_end - x_start, 1, transform=ax[0].get_xaxis_transform(), 
                          edgecolor='grey', linewidth=1, alpha=0.5, facecolor='none', hatch='///')
            
        ax[0].add_patch(rect)

        # Define the x and y ranges for the greyed out region
        x_start, x_end = -0.9, 0.25
        y_start, y_end = 0.5, 1.5  # Adjust the y range as needed

        # Add a hatched rectangle to the plot
        rect_ = patches.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start,
                                linewidth=1, edgecolor='grey', facecolor='none', hatch='///', alpha=0.5)
        ax[1].add_patch(rect_)    
    
    hep.histplot(
                mc_hist,
                density=True,
                label = r'Simulation',
                color = "blue",
                linewidth=3,
                ax=ax[0],
                flow='sum'
            )

    hep.histplot(
                mc_rw_hist,
                density=True,
                label=r'Simulation (Corr)',
                color = "None",
                linewidth=3,
                linestyle='--',
                ax=ax[0],
                flow='sum',
                histtype='fill',
                hatch='//',
                edgecolor='green',
                
            )

    # Apply hatching to the filled area
    #ax.patches[0].set_hatch('/')  # Apply hatching pattern to the filled area

    hep.histplot(
            data_hist,
            density=True,
            label = r'Data',
            yerr=True,
            xerr=False,
            color="black",
            #linewidth=3,
            histtype='errorbar',
            markersize=12,
            elinewidth=3,
            ax=ax[0],
            flow='sum'
        )


    ax[0].set_xlabel('')
    ax[0].margins(y=0.15)
    ax[0].set_ylim(0, 1.15*ax[0].get_ylim()[1])
    ax[0].tick_params(labelsize=22)

    #log scale for Iso variables
    if( "Iso" in str(xlabel) or "DR" in str(xlabel) or "esE" in str(xlabel)   ): # or 'r9' in str(xlabel) or 's4' in str(xlabel)
        ax[0].set_yscale('log')
        #ax[0].set_ylim(0.001,( np.max(data_hist)/1.5e6 ))
        ax[0].set_ylim(0.001, 12.05*ax[0].get_ylim()[1])
        

    # line at 1
    ax[1].axhline(1, 0, 1, label=None, linestyle='--', color="black", linewidth=1)#, alpha=0.5)

    #ratio
    data_hist_numpy = data_hist.to_numpy()
    mc_hist_numpy   = mc_hist.to_numpy()
    mc_hist_rw_numpy   = mc_rw_hist.to_numpy()

    integral_data = data_hist.sum().value * (data_hist_numpy[1][1] - data_hist_numpy[1][0])
    integral_mc = mc_hist.sum().value * (mc_hist_numpy[1][1] - mc_hist_numpy[1][0])

    #ratio betwenn normalizng flows prediction and data
    ratio = (data_hist_numpy[0] / integral_data) / ( (mc_hist_numpy[0] + 1e-15 ) / integral_mc)
    ratio = np.nan_to_num(ratio)

    integral_mc_rw = mc_rw_hist.sum().value * (mc_hist_rw_numpy[1][1] - mc_hist_rw_numpy[1][0])
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
        ax=ax[1],
        xerr=True,
    )


    hep.histplot(
        ratio_rw,
        bins=data_hist_numpy[1],
        label=None,
        color="green",
        histtype='errorbar',
        yerr=errors_nom,
        markersize=12,
        elinewidth=3,
        alpha=1,
        ax=ax[1],
        xerr=True,
    )

    #ax[0].text(0.001, 0.99, "Barrel",
    #     fontsize=26)

    bin_width = round(data_hist.axes[0].edges[1] - data_hist.axes[0].edges[0],2)
    if "hoe" in str(xlabel):
        bin_width = round(data_hist.axes[0].edges[1] - data_hist.axes[0].edges[0],3)
    
    if( "Err" in str(xlabel) ):
        #ax[0].set_ylabel(f"Fraction of events / ({bin_width} GeV)", fontsize=26)
        ax[0].set_ylabel("a.u", fontsize=30)
    else:
        #ax[0].set_ylabel(f"Fraction of events / {bin_width}", fontsize=26)
        ax[0].set_ylabel("a.u", fontsize=30)
    
    #ax[1].set_ylabel("Data / MC", fontsize=26)
    #ax.set_xlabel( str(xlabel), fontsize=26)
    ax[1].set_ylabel("Data / MC", fontsize=26)
    
    if( "mva" in str(xlabel)  ):
        xlabel_new = "Photon identification BDT score"
        ax[1].set_xlabel( str(xlabel_new) , fontsize=26)
    elif( "Err" in str(xlabel)  ):
        xlabel_new = r'\sigma_{E}'
        ax[1].set_xlabel(  r'$\sigma_{E}$ [GeV]' , fontsize=26)
    elif( "hoe" in str(xlabel)  ):
        xlabel_new = r'\sigma_{E}'
        ax[1].set_xlabel(  'H/E' , fontsize=26)
    else:
        xlabel_new = xlabel.replace("probe_", "")
        ax[1].set_xlabel( str(xlabel_new) , fontsize=26)
    
    ax[0].tick_params(labelsize=24)
    #ax.set_ylim(0., 1.1*ax.get_ylim()[1])
    ax[1].set_ylim(0.79, 1.21)
    if( 'mva' in xlabel ):
        ax[1].set_ylim(0.79, 1.21)

    
    #ax[0].legend(
    #    loc="upper right", fontsize=20
    #)

    # Create a custom legend handle to show a line
    from matplotlib.lines import Line2D
    line = Line2D([0], [0], color='blue', linewidth=3)

    # Get existing legend handles and labels
    handles, labels = ax[0].get_legend_handles_labels()

    # Replace the handle for the first histogram with the custom line
    handles[1] = line

    # Add the legend with the modified handles
    ax[0].legend(handles=handles, labels=labels, loc="upper right", fontsize=20)

    ax[0].text(0.05, 0.96, r'$\mathcal{Z}\rightarrow e^{+}e^{-}$', transform=ax[0].transAxes, fontsize=20, verticalalignment='top')
    if( 'mva' in xlabel ):
        if( endcap ):
            #ax[0].text(0.05, 0.95, r'|$\eta$| > 1.566', transform=ax[0].transAxes, fontsize=20, verticalalignment='top')
            ax[0].text(0.05, 0.9, r'EE photons', transform=ax[0].transAxes, fontsize=20, verticalalignment='top')
        else:
            ax[0].text(0.05, 0.90, r'EB photons', transform=ax[0].transAxes, fontsize=20, verticalalignment='top')
            #ax[0].text(0.05, 0.95, r'|$\eta$| < 1.442', transform=ax[0].transAxes, fontsize=20, verticalalignment='top')
    else:
        ax[0].text(0.05, 0.9, r'EB+EE photons', transform=ax[0].transAxes, fontsize=20, verticalalignment='top')
        #ax[0].text(0.05, 0.95, r'|$\eta$| < 2.5', transform=ax[0].transAxes, fontsize=20, verticalalignment='top')

    #mplhep.cms.text( loc = 1, ax = ax[0],  )
 
    hep.cms.label(data=True, ax=ax[0], loc=0, label = "Preliminary", com=13.6, lumi = 27.0)

    # Remove the space between the subplots
    plt.subplots_adjust(hspace=0)

    ax[0].margins(x=0)
    ax[1].margins(x=0)

    # Adjust the tight_layout to not add extra padding
    fig.tight_layout(h_pad=0, w_pad=0)


    fig.savefig(output_filename)


# This is a plot distribution suitable for dataframes, if one wants to plot tensors see "plot_distributions_for_tensors()"
def plot_distributions( path, data_df, mc_df, mc_weights, variables_to_plot, weights_befores_rw = False ):

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
            
            if( len(weights_befores_rw)  ):
                mc_hist.fill( np.array(  mc_df[key]), weight = weights_befores_rw )
            else:
                mc_hist.fill( np.array(  mc_df[key]) )

            #print( np.shape( np.array(drell_yan_df[key]) ) , np.shape( mc_weights  ) )

            mc_rw_hist.fill( np.array(mc_df[key]) , weight = mc_weights )

            plott( data_hist , mc_hist, mc_rw_hist , 'plots_AR_model/' +  str(key) +".png", xlabel = str(key)  )

def plot_distributions_for_tensors( data_tensor, mc_tensor, flow_samples, mc_weights, plot_path ):

    for i in range( np.shape( data_tensor )[1] ):

            mean = np.mean( np.array(data_tensor[:,i]) )
            std  = np.std(  np.array(data_tensor[:,i]) )

            if( 'Iso' in str(var_list[i]) or 'DR' in str(var_list[i]) or 'esE' in str(var_list[i]) or 'hoe' in str(var_list[i]) or 'energy' in str(var_list[i])  ):
                data_hist            = hist.Hist(hist.axis.Regular(50, 0.0 , mean + 2.0*std))
                mc_hist              = hist.Hist(hist.axis.Regular(50, 0.0 , mean + 2.0*std))
                mc_rw_hist           = hist.Hist(hist.axis.Regular(50, 0.0 , mean + 2.0*std))
            else:
                data_hist            = hist.Hist(hist.axis.Regular(50, mean - 2.5*std, mean + 2.5*std))
                mc_hist              = hist.Hist(hist.axis.Regular(50, mean - 2.5*std, mean + 2.5*std))
                mc_rw_hist           = hist.Hist(hist.axis.Regular(50, mean - 2.5*std, mean + 2.5*std))

            if( 'DR04' in str(var_list[i])  ):
                data_hist            = hist.Hist(hist.axis.Regular(50, 0.0 , 5.0))
                mc_hist              = hist.Hist(hist.axis.Regular(50, 0.0 , 5.0))
                mc_rw_hist           = hist.Hist(hist.axis.Regular(50, 0.0 , 5.0))

            data_hist.fill( np.array(data_tensor[:,i]   )  )
            mc_hist.fill(  np.array( mc_tensor[:,i]),  weight = 1e6*mc_weights )
            mc_rw_hist.fill( np.array( flow_samples[:,i]) , weight = 1e6*mc_weights )

            plott( data_hist , mc_hist, mc_rw_hist , plot_path +  str(var_list[i]) +".png", xlabel = str(var_list[i])  )

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

def plot_mvaID_curve(mc_inputs,data_inputs,nl_inputs, mc_conditions, data_conditions,mc_weights, data_weights, plot_path, zmmg = False, postEE = True):
    
    model_path = '/net/scratch_cms3a/daumann/PhD/EarlyHgg/simulation_to_data_corrections/run3_mvaID_models/'

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

    # Lets not do this one for now!
    #plot_profile_barrel( nl_mvaID, mc_mvaID ,mc_conditions,  data_mvaID, data_conditions, mc_weights, data_weights, plot_path, zmmg = zmmg)

    # now, we create and fill the histograms with the mvaID distributions
    
    if( zmmg ):
        mc_mva      = hist.Hist(hist.axis.Regular(7, -0.9, 1.0))
        nl_mva      = hist.Hist(hist.axis.Regular(7, -0.9, 1.0))
        data_mva    = hist.Hist(hist.axis.Regular(7, -0.9, 1.0))
    else:
        mc_mva      = hist.Hist(hist.axis.Regular(8, -0.9, 1.0))
        nl_mva      = hist.Hist(hist.axis.Regular(8, -0.9, 1.0))
        data_mva    = hist.Hist(hist.axis.Regular(8, -0.9, 1.0))

    if( postEE ):
        mc_mva      = hist.Hist(hist.axis.Regular(12, -0.9, 1.0))
        nl_mva      = hist.Hist(hist.axis.Regular(12, -0.9, 1.0))
        data_mva    = hist.Hist(hist.axis.Regular(12, -0.9, 1.0))
    else:
        mc_mva      = hist.Hist(hist.axis.Regular(8, -0.9, 1.0))
        nl_mva      = hist.Hist(hist.axis.Regular(8, -0.9, 1.0))
        data_mva    = hist.Hist(hist.axis.Regular(8, -0.9, 1.0))

    mc_mva.fill( mc_mvaID,     weight = mc_weights )
    nl_mva.fill( nl_mvaID,     weight = mc_weights )
    data_mva.fill( data_mvaID, weight = data_weights  )

    if( zmmg ):
        plott( data_mva , mc_mva, nl_mva , plot_path + '/mvaID_barrel.png', xlabel = "mvaID" , zmmg = True, postEE=postEE )
    else:
        plott( data_mva , mc_mva, nl_mva , plot_path + '/mvaID_barrel_test.png', xlabel = "mvaID", postEE = postEE  )


def plot_mvaID_curve_endcap(mc_inputs,data_inputs,nl_inputs, mc_conditions, data_conditions,mc_weights, data_weights, plot_path, postEE = True):
    
    model_path = '/net/scratch_cms3a/daumann/PhD/EarlyHgg/simulation_to_data_corrections/run3_mvaID_models/'

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

    #plot_profile_endcap( nl_mvaID, mc_mvaID ,mc_conditions,  data_mvaID, data_conditions, mc_weights, data_weights, plot_path)

    # now, we create and fill the histograms with the mvaID distributions
    mc_mva      = hist.Hist(hist.axis.Regular(6, -0.9, 1.0))
    nl_mva      = hist.Hist(hist.axis.Regular(6, -0.9, 1.0))
    data_mva    = hist.Hist(hist.axis.Regular(6, -0.9, 1.0))

    mc_mva.fill( mc_mvaID, weight     = mc_weights )
    nl_mva.fill( nl_mvaID, weight     = mc_weights )
    data_mva.fill( data_mvaID, weight = data_weights)

    plott( data_mva , mc_mva, nl_mva , plot_path + '/mvaID_endcap.png', xlabel = "mvaID", postEE=postEE )

def plot_distributions_after_transformations(training_inputs, training_conditions, training_weights):

    # two masks to separate events betwenn mc and data
    data_mask = training_conditions[:,-1] == 1
    mc_mask   = training_conditions[:,-1] == 0

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


# Down here are some ploting functions I still need to implement!
def plot_correlation_matrix_diference_barrel(data, data_conditions,data_weights,mc , mc_conditions,mc_weights,mc_corrected,path):

    # lets do this for barrel only for now ...
    mask_mc,mask_data = np.abs(mc_conditions[:,1]) < 2.4222, np.abs(data_conditions[:,1]) < 2.44222

    # apply the barrel only condition
    data, mc, mc_corrected              = data[mask_data]            , mc[mask_mc]             ,mc_corrected[mask_mc]
    data_conditions,mc_conditions      = data_conditions[mask_data] , mc_conditions[mask_mc]
    data_weights,mc_weights            = data_weights[mask_data]    , mc_weights[mask_mc]

    energy_err_data = data[:,-1:]
    energy_err_mc = mc[:,-1:]
    energy_err_mc_corrected = mc_corrected[:,-1:]

    mc           = mc[:,1: int( data.size()[1]  - 0 ) ]
    mc_corrected = mc_corrected[:,1: int( data.size()[1]  - 0 ) ]
    data         = data[:,1: int( data.size()[1]  - 0 ) ]

    # Lets only use the mvaID inputs in the correlation matrix!
    #data = torch.cat( [  data, energy_err_data .view(-1,1)  ], axis = 1 )
    #mc = torch.cat( [  mc, energy_err_mc .view(-1,1)  ], axis = 1 )
    #mc_corrected = torch.cat( [  mc_corrected, energy_err_mc_corrected .view(-1,1)  ], axis = 1 )

    # Some weights can of course be negative, so I had to use the abs here, since it does not accept negative weights ...
    data_corr         = torch.cov( data.T         , aweights = torch.Tensor( abs(data_weights) ))
    mc_corr           = torch.cov( mc.T           , aweights = torch.Tensor( abs(mc_weights)   ))
    mc_corrected_corr = torch.cov( mc_corrected.T , aweights = torch.Tensor( abs(mc_weights)   ))

    #from covariance to correlation matrices
    data_corr         = torch.inverse( torch.sqrt( torch.diag_embed( torch.diag(data_corr))) ) @ data_corr @  torch.inverse( torch.sqrt(torch.diag_embed(torch.diag(data_corr))) ) 
    mc_corr           = torch.inverse( torch.sqrt( torch.diag_embed(torch.diag(mc_corr)) )) @ mc_corr @  torch.inverse( torch.sqrt(torch.diag_embed(torch.diag(mc_corr))) ) 
    mc_corrected_corr = torch.inverse( torch.diag_embed(torch.sqrt(torch.diag(mc_corrected_corr)) )) @ mc_corrected_corr @  torch.inverse( torch.sqrt(torch.diag_embed(torch.diag(mc_corrected_corr))) ) 
    # end of matrix evaluations, now the plotting part!


    fig, ax = plt.subplots(figsize=(64,64))
    ax.matshow( 100*( data_corr - mc_corr ), cmap = 'bwr', vmin=-35, vmax=35)   
    
    #ploting the cov matrix values
    mean,count = 0,0
    for (i, j), z in np.ndenumerate(100*( data_corr - mc_corr )):
        mean = mean + abs(z)
        count = count + 1
        if( abs(z) < 1  ):
            pass
        else:
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', fontsize = 85)  
    mean = mean/count   

    ax.set_xticks(np.arange(len(var_list_matrix_barrel)-1))
    ax.set_yticks(np.arange(len(var_list_matrix_barrel)-1))
    
    ax.set_xticklabels(var_list_matrix_barrel[1:],fontsize = 85,rotation=90)
    ax.set_yticklabels(var_list_matrix_barrel[1:],fontsize = 85,rotation=0)

    ax.set_xlabel(r' $(MC - data)$ - Metric: ' + str(mean), loc = 'center' ,fontsize = 120)

    plt.savefig(path + '/correlation_matrix_barrel.pdf')

    #plloting part
    fig, ax = plt.subplots(figsize=(64,64))
    ax.matshow( 100*( data_corr - mc_corrected_corr ), cmap = 'bwr', vmin=-35, vmax=35)

    #ploting the cov matrix values
    mean,count = 0,0
    for (i, j), z in np.ndenumerate( 100*( data_corr - mc_corrected_corr )):
        if( abs(z) < 1.15  ):
            pass 
        elif(abs(z) > 5):
            if( z > 0 ):
                z = z-5
                ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', fontsize = 85)  
            else:
                z = z+3.0
                ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', fontsize = 85)  
        else:
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', fontsize = 85)    
    
        mean = mean + abs(z)
        count = count + 1

    mean = mean/count
    ax.set_xlabel(r'($MC^{Corrected}$-data)  - Metric: ' + str(mean), loc = 'center' ,fontsize = 120)
    #plt.tight_layout()
    plt.title( mean )
    
    #var_list_matrix_barrel=var_list_matrix_barrel.replace('probe_', '')

    ax.set_xticks(np.arange(len(var_list_matrix_barrel)-1))
    ax.set_yticks(np.arange(len(var_list_matrix_barrel)-1))
    
    ax.set_xticklabels(var_list_matrix_barrel[1:],fontsize = 85 ,rotation=90)
    ax.set_yticklabels(var_list_matrix_barrel[1:],fontsize = 85 ,rotation=0)

    plt.savefig(path + '/correlation_matrix_corrected_barrel.pdf')
    plt.close()




def plot_correlation_matrix_diference_endcap(data, data_conditions,data_weights,mc , mc_conditions,mc_weights,mc_corrected,path):

    # Selecting only end-cap events
    mask_mc,mask_data = np.abs(mc_conditions[:,1]) > 1.56, np.abs(data_conditions[:,1]) > 1.56

    # apply the barrel only condition
    data, mc, mc_corrected              = data[mask_data]            , mc[mask_mc]             ,mc_corrected[mask_mc]
    data_conditions,mc_conditions      = data_conditions[mask_data] , mc_conditions[mask_mc]
    data_weights,mc_weights            = data_weights[mask_data]    , mc_weights[mask_mc]
    
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
    
    ax.set_xticks(np.arange(len(var_list)))
    ax.set_yticks(np.arange(len(var_list)))
    
    ax.set_xticklabels(var_list,fontsize = 45 ,rotation=90)
    ax.set_yticklabels(var_list,fontsize = 45 ,rotation=0)


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

    ax.set_xticks(np.arange(len(var_list)))
    ax.set_yticks(np.arange(len(var_list)))
    
    ax.set_xticklabels(var_list,fontsize = 45,rotation=90)
    ax.set_yticklabels(var_list,fontsize = 45,rotation=0)

    ax.set_xlabel(r'(Corr_MC-Corr_data  - Metric: ' + str(mean), loc = 'center' ,fontsize = 70)

    plt.savefig(path + '/correlation_matrix_endcap.png')


#The events are binned in bins of equal number of events of each profilling variable, than the median is calculated!
def plot_profile_barrel( nl_mva_ID, mc_mva_id ,mc_conditions,  data_mva_id, data_conditions, mc_weights, data_weights,path, zmmg = False):

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
    plot_mvaID_profile_barrel( nl_mva_ID,mc_mva_id,mc_conditions[:,0],data_mva_id,data_conditions[:,0],mc_weights,data_weights , path, zmmg, var = 'pt'  )
    plot_mvaID_profile_barrel( nl_mva_ID,mc_mva_id,mc_conditions[:,1],data_mva_id,data_conditions[:,1],mc_weights,data_weights , path, zmmg, var = 'eta' )
    plot_mvaID_profile_barrel( nl_mva_ID,mc_mva_id,mc_conditions[:,2],data_mva_id,data_conditions[:,2],mc_weights,data_weights , path, zmmg, var = 'phi' )
    plot_mvaID_profile_barrel( nl_mva_ID,mc_mva_id,mc_conditions[:,3],data_mva_id,data_conditions[:,3],mc_weights,data_weights , path, zmmg, var = 'rho' )



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
def plot_mvaID_profile_barrel( nl_mva_ID,mc_mva_id,var_mc,data_mva_id,var_data,mc_weights,data_weights,path, zmmg,  var = 'pt' ):
    
    nl_mva_ID    = np.array(nl_mva_ID)
    mc_mva_id    = np.array(mc_mva_id)
    data_mva_id  = np.array(data_mva_id)
    mc_weights   = np.array(mc_weights)
    data_weights = np.array(data_weights)

    if( zmmg ):
        if 'pt' in var:
            bins = np.linspace( 20.0, 40.0, 6 )
        elif 'phi' in var:
            bins = np.linspace( -3.1415, 3.1415, 6)
        elif 'eta' in var:
            bins = np.linspace( -1.442, 1.442, 6 )
        elif 'rho' in var:
            bins = np.linspace( 10.0, 45.0, 6 )       
    else:
        if 'pt' in var:
            bins = np.linspace( 25.0, 60.0, 13 )
        elif 'phi' in var:
            bins = np.linspace( -3.1415, 3.1415, 13)
        elif 'eta' in var:
            bins = np.linspace( -1.442, 1.442, 13 )
        elif 'rho' in var:
            bins = np.linspace( 10.0, 45.0, 13 )

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
        print('ue')
        bins = np.linspace( [[-2.5,-1.442],  [1.442, 2.5] ], 20 )
        print(bins)
    elif 'rho' in var:
        bins = np.linspace( 5.0, 50.0, 20 )

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

    plt.savefig( path + '/profile_endcap_' + str(var) +'.png' )

    plt.close()

def tackle_mva_ID(data_df,mc_df,var_list, corr_mva_list, mc_weights, plot_path):
    
    model_path = '/net/scratch_cms3a/daumann/PhD/EarlyHgg/simulation_to_data_corrections/run3_mvaID_models/'

    photonid_mva = xgboost.Booster()
    photonid_mva.load_model( model_path + "model.json" )

    data_df = data_df[np.abs(  np.array(data_df[ "probe_ScEta" ])) < 1.5]
    mask_abrrel = [np.abs(  np.array(mc_df[ "probe_ScEta" ])) < 1.5]
    mc_df   = mc_df[np.abs(  np.array(mc_df[ "probe_ScEta" ])) < 1.5]
    #nf_df   = nf_df[np.abs(  np.array(nf_df[ "probe_ScEta" ])) < 1.5]

    # mva_id for data,mc and simulation
    data_mvaID = plot_higgsdna_like_mva_ID( data_df  , data_df["fixedGridRhoAll"] , var_list , photonid_mva )
    mc_mvaID   = plot_higgsdna_like_mva_ID( mc_df    , mc_df["fixedGridRhoAll"]   , var_list , photonid_mva )
    nf_mvaID   = plot_higgsdna_like_mva_ID( mc_df    , mc_df["fixedGridRhoAll"]   , corr_mva_list , photonid_mva )    
    

    mc_mva      = hist.Hist(hist.axis.Regular(42, -0.9, 1.0))
    nl_mva      = hist.Hist(hist.axis.Regular(42, -0.9, 1.0))
    data_mva    = hist.Hist(hist.axis.Regular(42, -0.9, 1.0))

    print( mc_weights )
    print( mask_abrrel )

    mc_mva.fill( mc_mvaID, weight = (1e6)*mc_weights[mask_abrrel] )
    nl_mva.fill( nf_mvaID, weight = (1e6)*mc_weights[mask_abrrel] )
    data_mva.fill( data_mvaID )


    plott( data_mva , mc_mva, nl_mva , plot_path + '/mvaID_barrel_2.png', xlabel = "Barrel mvaID"  )

    return 0

def plot_higgsdna_like_mva_ID(photon, rho, var_order, photonid_mva):
    
    bdt_inputs = {}
    bdt_inputs = np.column_stack(
        [np.array(photon[name]) for name in var_order]
    )

    tempmatrix = xgboost.DMatrix(bdt_inputs)

    mvaID = photonid_mva.predict(tempmatrix)

    # Only needed to compare to TMVA
    mvaID = 1.0 - 2.0 / (1.0 + np.exp(2.0 * mvaID))

    return mvaID
