# importing modules
import torch
import numpy as np
import matplotlib.pyplot as plt
import mplhep, hist
plt.style.use([mplhep.style.CMS])
import mplhep as hep
import xgboost
from matplotlib.lines import Line2D


#plot the diference in correlations betwenn mc and data and data and flow
def plot_correlation_matrices(data,mc,mc_corrected, mc_weights, var_names, path):

    # lets not use all ...
    data = data[:,2:-5]
    mc = mc[:,2:-5]
    mc_corrected = mc_corrected[:,2:-5]
    var_names = var_names[2:-5]
    
    mc_weights = len(data)*mc_weights/torch.sum(mc_weights)

    # Only for debugging
    #print( mc_weights )
    #print( data.size(), mc.size(), mc_corrected.size(), torch.tensor(mc_weights).size() )
    #exit()

    #calculating the covariance matrix of the pytorch tensors
    data_corr         = torch.cov( data.T  )
    mc_corr           = torch.cov( mc.T           , aweights = torch.Tensor( abs(mc_weights)  ))
    mc_corrected_corr = torch.cov( mc_corrected.T , aweights = torch.Tensor( abs(mc_weights)  ))

    #from covariance to correlation matrices
    data_corr         = torch.inverse( torch.sqrt( torch.diag_embed( torch.diag(data_corr))) ) @ data_corr @  torch.inverse( torch.sqrt(torch.diag_embed(torch.diag(data_corr))) ) 
    mc_corr           = torch.inverse( torch.sqrt( torch.diag_embed( torch.diag(mc_corr)) )) @ mc_corr @  torch.inverse( torch.sqrt(torch.diag_embed(torch.diag(mc_corr))) ) 
    mc_corrected_corr = torch.inverse( torch.sqrt( torch.diag_embed( torch.diag(mc_corrected_corr)) )) @ mc_corrected_corr @  torch.inverse( torch.sqrt(torch.diag_embed(torch.diag(mc_corrected_corr))) ) 

    # matrices setup ended! Now plotting part!
    fig, ax = plt.subplots(figsize=(41,41))
    cax = ax.matshow( 100*( data_corr - mc_corrected_corr ), cmap = 'bwr', vmin = -35, vmax = 35)
    cbar = fig.colorbar(cax,fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize = 70)
    cbar.set_label(r'Difference in correlation coefficient $[\%]$', rotation=90, loc = 'center', fontsize = 110, labelpad=60)

    # ploting the cov matrix values
    mean,count = 0,0
    for (i, j), z in np.ndenumerate( 100*( data_corr - mc_corrected_corr )):
        mean = mean + abs(z)
        count = count + 1
        if( abs(z) < 1  ):
            pass
        else:
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', fontsize = 65)    
    
    ax.yaxis.labelpad = 20
    ax.xaxis.labelpad = 20
    mean = mean/count
    #ax.set_xlabel(r'$100 \cdot (Corr^{Data}[X_{i},X_{J}] - Corr^{Simulation^{Corr}}[X_{i},X_{J}]) $ ' , loc = 'center' ,fontsize = 100, labelpad=40)
    plt.title( r'$\rho$(data) - $\rho$(corrected simulation)', fontweight='bold', fontsize = 130 , pad = 60 )
    
    ax.set_xticks(np.arange(len(var_names)))
    ax.set_yticks(np.arange(len(var_names)))
    
    # Apply the replace method to each element of the list
    cleaned_var_names = [name.replace("probe_", "").replace("raw_", "").replace("trkSum","").replace("ChargedIso","").replace("es","").replace("Cone","") for name in var_names]
    
    ax.set_xticklabels(cleaned_var_names, fontsize = 45 , rotation=90 )
    ax.set_yticklabels(cleaned_var_names, fontsize = 45 , rotation=0  )

    ax.tick_params(axis='both', which='major', pad=30)
    plt.tight_layout()

    plt.savefig(path + '/correlation_matrix_corrected.pdf')

    ####################################
    # Nominal MC vs Data
    #####################################
    fig, ax = plt.subplots(figsize=(41,41))
    cax = ax.matshow( 100*( data_corr - mc_corr ), cmap = 'bwr', vmin = -35, vmax = 35)
    cbar = fig.colorbar(cax,fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize = 90)
    cbar.set_label(r'Difference in correlation coefficient $[\%]$', rotation=90, loc = 'center', fontsize = 110,labelpad=60)
    
    #ploting the cov matrix values
    mean,count = 0,0
    for (i, j), z in np.ndenumerate( 100*( data_corr - mc_corr )):
        mean = mean + abs(z)
        count = count + 1
        if( abs(z) < 1  ):
            pass
        else:
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', fontsize = 55)    
    
    mean = mean/count
    #ax.set_xlabel(r'$100 \cdot  (Corr^{Data}[X_{i},X_{J}] - Corr^{Simulation}[X_{i},X_{J}]) $ ' , loc = 'center' ,fontsize = 100, labelpad=40)
    plt.title( r'$\rho$(data) - $\rho$(nominal simulation)',fontweight='bold', fontsize = 140 , pad = 60 )
    
    ax.set_xticks(np.arange(len(var_names)))
    ax.set_yticks(np.arange(len(var_names)))
        
    ax.set_xticklabels(cleaned_var_names, fontsize = 45 , rotation=90 )
    ax.set_yticklabels(cleaned_var_names, fontsize = 45 , rotation=0  )

    ax.tick_params(axis='both', which='major', pad=30)
    plt.tight_layout()

    plt.savefig(path + '/correlation_matrix_nominal.pdf')
    

# function to calculate the weighted profiles quantiles!
def weighted_quantiles_interpolate(values, weights, quantiles=0.5):
    i = np.argsort(values)
    c = np.cumsum(weights[i])
    return values[i[np.searchsorted(c, np.array(quantiles) * c[-1])]]

# Calculates the weighted median stat errors for the profile plots 
# more details on: https://stats.stackexchange.com/questions/59838/standard-error-of-the-median/61759#61759

def weighted_quantiles_std(values, weights, quantiles=0.5):
    
    i = np.argsort(values)
    c = np.cumsum(weights[i])
    
    n_events = [i[np.searchsorted(c, np.array(quantiles) * c[-1])]]
    events = values[:int(n_events[0])]

    # Ensure weights sum to 1 (normalize if they don't)
    w_normalized = weights[i][:int(n_events[0])] / np.sum(weights[i][:int(n_events[0])])

    # Calculate weighted mean
    weighted_mean = np.sum(w_normalized * events)

    # Calculate weighted variance
    weighted_variance = np.sum(w_normalized * (events - weighted_mean)**2)

    # Calculate weighted standard deviation
    weighted_std = np.sqrt(weighted_variance)

    error = 1.253*weighted_std/np.sqrt(len(events))
    return error

# Now the profile plots!!!
def calculate_bins_position(array, num_bins=12):

    array_sorted = np.sort(array)  # Ensure the array is sorted
    n = len(array)
    
    # Calculate the exact number of elements per bin
    elements_per_bin = n // num_bins
    
    # Adjust bin_indices to accommodate for numpy's 0-indexing and avoid out-of-bounds access
    bin_indices = [i*elements_per_bin for i in range(1, num_bins)]
    bin_indices.append(n-1)  # Ensure the last index is included for the last bin
    
    # Find the array values at these adjusted indices
    bin_edges = array_sorted[bin_indices]

    bin_edges = np.insert(bin_edges, 0, 0)
    
    return bin_edges

# Lets do this separatly, first, we do the plots at the barrel only!
def plot_mvaID_profile_barrel( nl_mva_ID,mc_mva_id,var_mc,data_mva_id,var_data,mc_weights,data_weights,path,var = 'pt' ):
    
    if 'pt' in var:
        bins = calculate_bins_position( var_data[ var_data < 70 ], num_bins = 14 ) #np.linspace( 25.0, 65.0, 14 )
        bins[0] = 22.0
    elif 'phi' in var:
        bins = np.linspace( -3.1415, 3.1415, 14)
    elif 'eta' in var:
        #bins = np.linspace( -1.442, 1.442, 14 )
        #bins = np.linspace( 10.0, 45.0, 14 )
        bins = calculate_bins_position( var_data[ var_data < 1.5 ], num_bins = 14 ) #np.linspace( 25.0, 65.0, 14 )
        print( bins )
        bins[0] = - 1.5
    elif 'rho' in var:
        #bins = np.linspace( 10.0, 45.0, 14 )
        bins = calculate_bins_position( var_data[ var_data < 60 ], num_bins = 14 ) #np.linspace( 25.0, 65.0, 14 )
        print( bins )
        bins[0] = 8.0
        
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

        nl_mean_q25.append(   weighted_quantiles_interpolate( mva_nl_window  , mc_weights_window     , quantiles= 0.30  ) )   #np.median(  mva_nl_window )   )
        mc_mean_q25.append(   weighted_quantiles_interpolate( mva_mc_window  , mc_weights_window     , quantiles= 0.30  ) )
        data_mean_q25.append( weighted_quantiles_interpolate( mva_data_window, data_weights_window   , quantiles= 0.30  ) )

        nl_mean_q75.append(   weighted_quantiles_interpolate( mva_nl_window  , mc_weights_window     , quantiles = 0.70 ) )   #np.median(  mva_nl_window )   )
        mc_mean_q75.append(   weighted_quantiles_interpolate( mva_mc_window  , mc_weights_window     , quantiles = 0.70 ) )
        data_mean_q75.append( weighted_quantiles_interpolate( mva_data_window, data_weights_window   , quantiles = 0.70 ) )

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

    plt.ylabel( 'Photon MVA ID', fontsize = 18 )
    #plt.legend(fontsize=15)

    # Adding the title with the legends
    #title_text = 'MC corrected: Red, MC nominal: Blue, Data: Green'
    #plt.title(title_text, fontsize = 16)

    # Adding the legend
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize = 16)

    # Adding the legend
    #plt.legend(handles=legend_handles)

    # Constructing the title with the legends
    #title_text = " ".join([handle.get_label() for handle in legend_handles])
    #plt.title(title_text, fontsize=16)

    plt.ylim( 0.3 , 0.9 )

    if 'eta' in var:
        plt.ylim( 0.2 , 0.9 )
    if 'phi' in var:
        plt.ylim( 0.3 , 0.9 )
    if 'rho' in var:
        plt.ylim( 0.3 , 0.9 )

    plt.tight_layout()

    plt.savefig( path + '/profile_' + str(var) +'.png' )

    plt.close()


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