# importing modules
import torch
import numpy as np
import matplotlib.pyplot as plt
import mplhep, hist
plt.style.use([mplhep.style.CMS])
import mplhep as hep
import xgboost
from matplotlib.lines import Line2D

# Converting covariance to correlation matrices
def cov_to_corr(cov_matrix):
    stddev = torch.sqrt(torch.diag(cov_matrix))
    stddev_matrix = torch.diag_embed(stddev)
    corr_matrix = torch.inverse(stddev_matrix) @ cov_matrix @ torch.inverse(stddev_matrix)
    return corr_matrix

#plot the diference in correlations betwenn mc and data and data and flow
def plot_correlation_matrices(data,mc,mc_corrected, mc_weights, var_names, path):

    # Making the plot for barrel and end-cap
    eta_regions    = ['barrel', 'endcap']
    eta_mc_masks   = [  np.abs(mc[:,-3]) < 1.442, np.abs(mc[:,-3]) > 1.566 ]
    eta_data_masks = [  np.abs(data[:,-3]) < 1.442, np.abs(data[:,-3]) > 1.566 ] 

    for eta_region, mc_eta_mask, data_eta_mask in zip(eta_regions, eta_mc_masks, eta_data_masks):

        # lets not use all ...
        if 'barrel' in eta_region:
            data_  = torch.cat( [data[:,2:-8][data_eta_mask], data[:,-6][data_eta_mask].view(-1,1) ], axis = 1)
            mc_    = torch.cat( [mc[:,2:-8][mc_eta_mask], mc[:,-6][mc_eta_mask].view(-1,1)  ] , axis =1)
            mc_corrected_ = torch.cat( [mc_corrected[:,2:-8][mc_eta_mask], mc_corrected[:,-6][mc_eta_mask].view(-1,1)], axis =1 )
            mc_weights_   = mc_weights[mc_eta_mask]
            var_names_ = var_names[2:-8] + [var_names[-6]] 
        else:
            data_ = data[:,2:-5][data_eta_mask]
            mc_ = mc[:,2:-5][mc_eta_mask]
            mc_corrected_ = mc_corrected[:,2:-5][mc_eta_mask]
            mc_weights_ = mc_weights[mc_eta_mask]
            var_names_ = var_names[2:-5]
        
        mc_weights_ = len(data_)*mc_weights_/torch.sum(mc_weights_)

        #calculating the covariance matrix of the pytorch tensors
        data_cov         = torch.cov( data_.T  )
        mc_cov          = torch.cov( mc_.T           , aweights = torch.Tensor( abs(mc_weights_)  ))
        mc_corrected_cov = torch.cov( mc_corrected_.T , aweights = torch.Tensor( abs(mc_weights_)  ))

        #from covariance to correlation matrices
        #data_corr         = torch.inverse( torch.sqrt( torch.diag_embed( torch.diag(data_corr))) ) @ data_corr @  torch.inverse( torch.sqrt(torch.diag_embed(torch.diag(data_corr))) ) 
        #mc_corr           = torch.inverse( torch.sqrt( torch.diag_embed( torch.diag(mc_corr)) )) @ mc_corr @  torch.inverse( torch.sqrt(torch.diag_embed(torch.diag(mc_corr))) ) 
        #mc_corrected_corr = torch.inverse( torch.sqrt( torch.diag_embed( torch.diag(mc_corrected_corr)) )) @ mc_corrected_corr @  torch.inverse( torch.sqrt(torch.diag_embed(torch.diag(mc_corrected_corr))) ) 
        
        data_corr = cov_to_corr(data_cov)
        mc_corr = cov_to_corr(mc_cov)
        mc_corrected_corr = cov_to_corr(mc_corrected_cov)

        # matrices setup ended! Now plotting part!
        fig, ax = plt.subplots(figsize=(41,41))
        cax = ax.matshow( 100*( data_corr - mc_corrected_corr ), cmap = 'bwr', vmin = -35, vmax = 35)
        cbar = fig.colorbar(cax,fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize = 70)
        cbar.set_label(r'Difference in correlation coefficient $[\%]$', rotation=90, loc = 'center', fontsize = 110, labelpad=60)

        # ploting the cov matrix values
        factors_sum = 0
        mean,count = 0,0
        for (i, j), z in np.ndenumerate( 100*( data_corr - mc_corrected_corr )):
            mean = mean + abs(z)
            count = count + 1
            factors_sum = factors_sum + abs(z)
            if( abs(z) < 1  ):
                pass
            else:
                ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', fontsize = 65)    
        
        ax.yaxis.labelpad = 20
        ax.xaxis.labelpad = 20
        mean = mean/count
        #ax.set_xlabel(r'$100 \cdot (Corr^{Data}[X_{i},X_{J}] - Corr^{Simulation^{Corr}}[X_{i},X_{J}]) $ ' , loc = 'center' ,fontsize = 100, labelpad=40)
        plt.title( r'$\rho$(data) - $\rho$(corrected simulation)' + f' [{eta_region}]', fontweight='bold', fontsize = 130 , pad = 60 )
        
        ax.set_xticks(np.arange(len(var_names_)))
        ax.set_yticks(np.arange(len(var_names_)))
        
        # Apply the replace method to each element of the list
        cleaned_var_names = [name.replace("probe_", "").replace("raw_", "").replace("trkSum","").replace("ChargedIso","").replace("es","").replace("Cone","").replace("PF","").replace("Over","") for name in var_names_]
         
        ax.set_xticklabels(cleaned_var_names, fontsize = 50 , rotation=90 )
        ax.set_yticklabels(cleaned_var_names, fontsize = 50 , rotation=0  )

        # Add text below the plot
        plt.figtext(0.5, 0.04, f'Mean Absolute Sum of Coefficients - {round(factors_sum/(2.*count),2)}', ha="center", fontsize= 85)

        ax.tick_params(axis='both', which='major', pad=30)
        plt.tight_layout()

        plt.savefig(path + f'/correlation_matrix_corrected_{eta_region}.pdf')

        ####################################
        # Nominal MC vs Data
        #####################################
        fig, ax = plt.subplots(figsize=(41,41))
        cax = ax.matshow( 100*( data_corr - mc_corr ), cmap = 'bwr', vmin = -35, vmax = 35)
        cbar = fig.colorbar(cax,fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize = 90)
        cbar.set_label(r'Difference in correlation coefficient $[\%]$', rotation=90, loc = 'center', fontsize = 110,labelpad=60)
        
        #ploting the cov matrix values
        factors_sum = 0
        mean,count = 0,0
        for (i, j), z in np.ndenumerate( 100*( data_corr - mc_corr )):
            mean = mean + abs(z)
            count = count + 1
            factors_sum = factors_sum + abs(z)
            if( abs(z) < 1  ):
                pass
            else:
                ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center', fontsize = 55)    
        
        mean = mean/count
        #ax.set_xlabel(r'$100 \cdot  (Corr^{Data}[X_{i},X_{J}] - Corr^{Simulation}[X_{i},X_{J}]) $ ' , loc = 'center' ,fontsize = 100, labelpad=40)
        plt.title( r'$\rho$(data) - $\rho$(nominal simulation)' + f' [{eta_region}]',fontweight='bold', fontsize = 140 , pad = 60 )
        
        ax.set_xticks(np.arange(len(var_names_)))
        ax.set_yticks(np.arange(len(var_names_)))
            
        ax.set_xticklabels(cleaned_var_names, fontsize = 50 , rotation=90 )
        ax.set_yticklabels(cleaned_var_names, fontsize = 50 , rotation=0  )

        # Add text below the plot
        plt.figtext(0.5, 0.04, f'Mean Absolute Sum of Coefficients - {round(factors_sum/(2.*count),2)}', ha="center", fontsize= 85)

        ax.tick_params(axis='both', which='major', pad=30)
        plt.tight_layout()

        plt.savefig(path + f'/correlation_matrix_nominal_{eta_region}.pdf')
    

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
def plot_mvaID_profile_barrel( nl_mva_ID,mc_mva_id,var_mc,data_mva_id,var_data,mc_weights,data_weights,path,var = 'pt', IsBarrel = True):
    
    if 'pt' in var:
        if IsBarrel:
            bins = calculate_bins_position( var_data[ var_data < 70 ], num_bins = 14 ) #np.linspace( 25.0, 65.0, 14 )
            bins[0] = 22.0
        else:
            bins = calculate_bins_position( var_data[ var_data < 70 ], num_bins = 8 ) #np.linspace( 25.0, 65.0, 14 )
            bins[0] = 22.0    
    elif 'phi' in var:
        if IsBarrel:
            bins = np.linspace( -3.1415, 3.1415, 14)
        else:
            bins = np.linspace( -3.1415, 3.1415, 9)
    elif 'eta' in var:
        if IsBarrel:
            #bins = calculate_bins_position( var_data[ var_data < 1.5 ], num_bins = 14 ) #np.linspace( 25.0, 65.0, 14 )
            #bins[0] = - 1.5
            
            bins = calculate_bins_position( var_data[ var_data < 2.5 ], num_bins = 14 ) #np.linspace( 25.0, 65.0, 14 )
            bins[0] = - 2.5
        else:
            bins = calculate_bins_position( var_data[ var_data < 2.5 ], num_bins = 8 ) #np.linspace( 25.0, 65.0, 14 )
            bins[0] = - 2.5
    elif 'rho' in var:
        if IsBarrel:
            bins = calculate_bins_position( var_data[ var_data < 60 ], num_bins = 14 ) #np.linspace( 25.0, 65.0, 14 )
            bins[0] = 8.0
        else:
            bins = calculate_bins_position( var_data[ var_data < 60 ], num_bins = 8 ) #np.linspace( 25.0, 65.0, 14 )
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
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot( position , nl_mean ,  linewidth  = 3 , color = 'red' , label = 'MC corrected' )
    plt.plot( position , mc_mean ,  linewidth  = 3 , color = 'blue'  , label = 'MC nominal'   )
    plt.plot( position , data_mean , linewidth = 3 , color = 'green', label = 'Data'  )


    plt.plot( position , nl_mean_q25 ,  linewidth  = 3 , linestyle='dashed', color = 'red'  )
    plt.plot( position , mc_mean_q25 ,  linewidth  = 3 , linestyle='dashed',color = 'blue'    )
    plt.plot( position , data_mean_q25 , linewidth = 3 , linestyle='dashed', color = 'green' )

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

    # Custom text entry
    #custom_text = plt.Line2D([], [], color='w', label=f'$Z \rightarrow ee - EE photons$')

    # Adding the legend
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=3, fontsize = 15)

    # Add text to the plot
    """ 
    if IsBarrel:
        ax.text(0.05, 0.96, r'$\mathcal{Z}\rightarrow e^{+}e^{-}$', transform=ax.transAxes, fontsize=18, verticalalignment='top')
        #plt.text(0.1, 0.98, r'$Z \rightarrow ee - EB photons$', ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)
    else:
        ax.text(0.05, 0.96, r'$\mathcal{Z}\rightarrow e^{+}e^{-}$', transform=ax.transAxes, fontsize=18, verticalalignment='top')
        #plt.text(0.1, 0.98, r'$Z \rightarrow ee - EE photons$', ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)

    if( IsBarrel and  'eta' not in var):
        #ax[0].text(0.05, 0.95, r'|$\eta$| > 1.566', transform=ax[0].transAxes, fontsize=20, verticalalignment='top')
        ax.text(0.05, 0.88, r'EB photons', transform=ax.transAxes, fontsize=16, verticalalignment='top')
    elif 'eta' in var:
        ax.text(0.05, 0.88, r'EB+EE photons', transform=ax.transAxes, fontsize=16, verticalalignment='top')
    else:
        ax.text(0.05, 0.88, r'EE photons', transform=ax.transAxes, fontsize=16, verticalalignment='top')
    """

    # Adding the legend
    #plt.legend(handles=legend_handles)

    # Constructing the title with the legends
    #title_text = " ".join([handle.get_label() for handle in legend_handles])
    #plt.title(title_text, fontsize=16)

    plt.ylim( 0.3 , 0.95 )

    if 'eta' in var:
        plt.ylim( 0.2 , 1.0 )
    if 'phi' in var:
        plt.ylim( 0.3 , 1.0 )
    if 'rho' in var:
        plt.ylim( 0.3 , 1.0 )

    hep.cms.label(data=True, ax=ax, loc=0, label = "Preliminary", com=13.6, lumi = 27.0)

    plt.tight_layout()

    if( IsBarrel ):
        plt.savefig( path + '/profile_' + str(var) +'_barrel.png' )
    else:
        plt.savefig( path + '/profile_' + str(var) +'_endcap.png' )

    plt.close()

#The events are binned in bins of equal number of events of each profilling variable, than the median is calculated!
def plot_profile_barrel( nl_mva_ID, mc_mva_id ,mc_conditions,  data_mva_id, data_conditions, mc_weights, data_weights,path):

    # Making the plot for barrel and end-cap
    eta_regions = ['barrel', 'endcap']
    eta_mc_masks   = [ np.abs(mc_conditions[:,1]) < 1.442   , np.abs(mc_conditions[:,1]) > 1.566   ]
    eta_data_masks = [ np.abs(data_conditions[:,1]) < 1.442 , np.abs(data_conditions[:,1]) > 1.566 ] 

    for eta_region, mc_eta_mask, data_eta_mask in zip(eta_regions, eta_mc_masks, eta_data_masks):


        nl_mva_ID_ = nl_mva_ID
        mc_mva_id_ = mc_mva_id
        mc_weights_ = mc_weights
        mc_conditions_ = mc_conditions

        data_mva_id_     = data_mva_id
        data_conditions_ = data_conditions
        data_weights_    = data_weights

        IsBarrel = True
        if 'barrel' in eta_region:
           IsBarrel = True
        else:
            IsBarrel = False 

        #lets call the function ...
        plot_mvaID_profile_barrel( nl_mva_ID_[mc_eta_mask] ,mc_mva_id_[mc_eta_mask] ,mc_conditions_[mc_eta_mask][:,0],data_mva_id_[data_eta_mask] ,data_conditions_[data_eta_mask][:,0],mc_weights_[mc_eta_mask] ,data_weights_[data_eta_mask]  , path, var = 'pt'  , IsBarrel = IsBarrel)
        plot_mvaID_profile_barrel( nl_mva_ID_ ,mc_mva_id_ ,mc_conditions_[:,1],data_mva_id_ ,data_conditions_[:,1],mc_weights_ ,data_weights_  , path, var = 'eta' , IsBarrel = IsBarrel)
        plot_mvaID_profile_barrel( nl_mva_ID_[mc_eta_mask] ,mc_mva_id_[mc_eta_mask] ,mc_conditions_[mc_eta_mask][:,2],data_mva_id_[data_eta_mask] ,data_conditions_[data_eta_mask][:,2],mc_weights_[mc_eta_mask] ,data_weights_[data_eta_mask]  , path ,var = 'phi' , IsBarrel = IsBarrel)
        plot_mvaID_profile_barrel( nl_mva_ID_[mc_eta_mask] ,mc_mva_id_[mc_eta_mask] ,mc_conditions_[mc_eta_mask][:,3],data_mva_id_[data_eta_mask] ,data_conditions_[data_eta_mask][:,3],mc_weights_[mc_eta_mask] ,data_weights_[data_eta_mask]  , path ,var = 'rho' , IsBarrel = IsBarrel)