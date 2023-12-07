# script made to plot the main validation and resulting distributions

# importing modules
import torch
import numpy as np
import matplotlib.pyplot as plt
import mplhep, hist
plt.style.use([mplhep.style.CMS])
import mplhep as hep

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
        label = r'$Z\rightarrow ee$ w/ rw',
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

    ax[0].legend(
        loc="upper right", fontsize=24
    )

    hep.cms.label(data=True, ax=ax[0], loc=0, label="Private Work", com=13.6, lumi=21.7)

    #plt.subplots_adjust(hspace=0.03)

    plt.tight_layout()

    fig.savefig(output_filename)

    return 0

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