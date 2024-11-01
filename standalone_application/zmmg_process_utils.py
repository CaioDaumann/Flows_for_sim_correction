# Script to test the application of normalizing flows outisde of the main code enviroment

# python libraries import
import os 
import numpy as np
import glob
import torch
import pandas as pd
import zuko

print('vamo')

# This is only needed when using the DF pre-processing options!
def zmmg_kinematics_reweighting(data_array, data_weights, mc_array, mc_weights):
    
    # As a first step, we will normalize the sum of weights to one!
    data_weights = np.array(data_weights/np.sum(data_weights))
    mc_weights   = np.array(mc_weights/np.sum( mc_weights ))

    # I cannot slice pandas df, so I am making them into numpy arrays
    data_array = np.array( data_array )
    mc_array   = np.array( mc_array )

    variable_names = [ "pt", "eta", "phi", "rho" ]

    #defining the range where the rw will be performed in each variables
    pt_min,pt_max   = 20, 50
    rho_min,rho_max = 10.0,50.0
    eta_min,eta_max = -2.501,2.501
    phi_min, phi_max = -3.15,3.15

    # now the number of bins in each distribution
    pt_bins,rho_bins,eta_bins,phi_bins = 4, 10, 4, 2 #8, 25, 6, 1 #was 20, 30, 10, 4

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

def zmmg_selection(data_df,mc_df):

    vars = ["photon_ScEta","mmy_mass","dimuon_mass","muon_far_pt","muon_near_pt","photon_muon_near_dR","photon_pt", "photon_trkSumPtSolidConeDR04", "photon_hoe"]

    data = np.array(data_df[vars])
    mc   = np.array(mc_df[vars])

    mask_data = np.logical_and(  data[:,1] > 80 ,  data[:,1] < 100 ) #mmy mass [80,100]
    mask_data = np.logical_and( mask_data, data[:,2] > 35 )  # mumu mass  < 35
    mask_data = np.logical_and( mask_data, data[:,2] +  data[:,1] < 180 ) # mass mmy + mm < 180 GeV 
    mask_data = np.logical_and( mask_data, data[:,3] > 20 )
    mask_data = np.logical_and( mask_data, data[:,5] < 0.8 )
    mask_data = np.logical_and( mask_data, data[:,6] > 20 )
    #mask_data = np.logical_and( mask_data, data[:,5] > 0.4 )
    #mask_data = np.logical_and( mask_data, data[:,7] < 9 ) #trk sum cut
    #mask_data = np.logical_and( mask_data, data[:,8] < 0.025 )
    
    mask_mc = np.logical_and(  mc[:,1] > 80 ,  mc[:,1] < 100 ) #mmy mass [80,100]
    mask_mc = np.logical_and( mask_mc, mc[:,2] > 35 )  # mumu mass  < 35
    mask_mc = np.logical_and( mask_mc, mc[:,2] +  mc[:,1] < 180 ) # mass mmy + mm < 180 GeV 
    mask_mc = np.logical_and( mask_mc, mc[:,3] > 20 )
    mask_mc = np.logical_and( mask_mc, mc[:,5] < 0.8 )
    mask_mc = np.logical_and( mask_mc, mc[:,6] > 20 )
    #mask_mc = np.logical_and( mask_mc, mc[:,5] > 0.4 )
    #mask_mc = np.logical_and( mask_mc, mc[:,7] < 9 )     # trk sum cut
    #mask_mc = np.logical_and( mask_mc, mc[:,8] < 0.025 )

    return mask_data, mask_mc

