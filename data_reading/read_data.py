# Loads and treat the Z->ee data and simulation
#Author: Caio Cesar Daumann : caio.cesar.cern.ch

# importing libraries
import os 
import numpy as np
import glob
import torch
import pandas as pd

# importing other scripts
import plot.plot_utils        as plot_utils
import data_reading.zmmg_utils as zmmg_utils


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
    pt_bins  = calculate_bins_position(np.array(simulation_df["probe_pt"]), 30)
    eta_bins = calculate_bins_position(np.array(simulation_df["probe_ScEta"]), 30)
    rho_bins = calculate_bins_position(np.nan_to_num(np.array(simulation_df["fixedGridRhoAll"])), 30) #np.linspace( 5,65, 30) #calculate_bins_position(np.nan_to_num(np.array(simulation_df["fixedGridRhoAll"])), 70)

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

def perform_zee_selection(df):
    # Define individual selection criteria
    mass_criteria = (df["mass"] > 80) & (df["mass"] < 100)
    tag_pt_criteria = df["tag_pt"] > 40
    probe_pt_criteria = df["probe_pt"] > 20
    eta_criteria = np.abs(df["probe_ScEta"]) < 2.5
    phi_criteria = np.abs(df["probe_phi"]) < np.pi
    tag_mvaID_criteria = df["tag_mvaID"] > 0.0
    
    # Combine all criteria into a single mask
    mask = (
        mass_criteria &
        tag_pt_criteria &
        probe_pt_criteria &
        eta_criteria &
        phi_criteria &
        tag_mvaID_criteria
    )
    
    # Apply mask to filter DataFrame
    return df[mask]

# This function prepare teh data to be used bu pytorch!
def separate_training_data( data_df, mc_df, mc_weights, data_weights, input_vars, condition_vars):

    mc_inputs     = torch.tensor(np.array( np.nan_to_num(mc_df[input_vars])))
    mc_conditions = torch.tensor(np.concatenate( [  np.array( np.nan_to_num(mc_df[condition_vars])), 0*np.ones( len( mc_weights )  ).reshape(-1,1)  ], axis = 1  ) )

    input_vars_data = input_vars
    for i in range( len(input_vars) ):
        input_vars_data[i] = input_vars_data[i].replace('_raw','')

    # creating the inputs and conditions tensors! - adding the bollean to the conditions tensors
    data_inputs     = torch.tensor(np.nan_to_num(np.array(data_df[input_vars_data])))
    data_conditions = torch.tensor(np.concatenate( [ np.nan_to_num(np.array( data_df[condition_vars])), np.ones( len( data_weights )  ).reshape(-1,1)  ], axis = 1  ) )

    #first we shuffle all the arrays!
    permutation = np.random.permutation(len(data_weights))
    data_inputs      = torch.tensor(data_inputs[permutation])
    data_conditions  = torch.tensor(data_conditions[permutation])
    data_weights     = torch.tensor(data_weights[permutation])

    mc_permutation = np.random.permutation(len(mc_weights))
    mc_inputs      = torch.tensor(mc_inputs[mc_permutation])
    mc_conditions  = torch.tensor(mc_conditions[mc_permutation])
    mc_weights     = torch.tensor(mc_weights[mc_permutation])

    assert abs(torch.sum(mc_weights) - torch.sum(data_weights)) < torch.sum(data_weights)*(5/100)

    # Now, in order not to bias the network we choose make sure the tensors of data and simulation have the same number of events
    try:
        mc_inputs = mc_inputs[ :len(data_inputs) ]
        mc_conditions = mc_conditions[ :len(data_inputs) ]
        mc_weights = mc_weights[ :len(data_inputs) ]
        
        assert len( mc_weights )    == len( data_weights )
        assert len( mc_conditions ) == len(data_conditions)
    except:
        data_inputs = data_inputs[ :len(mc_inputs) ]
        data_conditions = data_conditions[ :len(mc_inputs) ]
        data_weights = data_weights[ :len(mc_inputs) ]
        
        assert len( mc_weights )    == len( data_weights )
        assert len( mc_conditions ) == len(data_conditions)

    # lets normalize the weights here again just to be sure ....
    mc_weights = len(data_weights)*mc_weights/torch.sum(mc_weights)

    assert abs(torch.sum(mc_weights) - torch.sum(data_weights)) < 1

    print( 'Number of MC events after equiparing! ', len( mc_conditions ), ' Number of data events: ', len(data_conditions))

    training_percent = 0.7
    validation_percent = 0.03 + training_percent
    testing_percet = 1 - training_percent - validation_percent

    # Now, the fun part! - Separating everyhting into trainnig, validation and testing datasets!
    data_training_inputs     = data_inputs[: int( training_percent*len(data_inputs  )) ] 
    data_training_conditions = data_conditions[: int( training_percent*len(data_inputs  )) ] 
    data_training_weights    = data_weights[: int( training_percent*len(data_inputs  )) ] 

    mc_training_inputs     = mc_inputs[: int( training_percent*len(mc_inputs  )) ] 
    mc_training_conditions = mc_conditions[: int( training_percent*len(mc_inputs  )) ] 
    mc_training_weights    = mc_weights[: int( training_percent*len(mc_inputs  )) ] 

    # now, the validation dataset
    data_validation_inputs     = data_inputs[int(training_percent*len(data_inputs  )):int( validation_percent*len(data_inputs  )) ] 
    data_validation_conditions = data_conditions[int(training_percent*len(data_inputs  )):int( validation_percent*len(data_inputs  )) ] 
    data_validation_weights    = data_weights[int(training_percent*len(data_inputs  )):int( validation_percent*len(data_inputs  )) ] 

    mc_validation_inputs     = mc_inputs[int(training_percent*len(mc_inputs  )):int( validation_percent*len(mc_inputs  )) ] 
    mc_validation_conditions = mc_conditions[int(training_percent*len(mc_inputs  )):int( validation_percent*len(mc_inputs  )) ] 
    mc_validation_weights    = mc_weights[int(training_percent*len(mc_inputs  )):int( validation_percent*len(mc_inputs  )) ] 

    # now for the grand finalle, the test tensors
    data_test_inputs     = data_inputs[int( validation_percent*len(data_inputs  )): ] 
    data_test_conditions = data_conditions[int( validation_percent*len(data_inputs  )): ] 
    data_test_weights    = data_weights[int( validation_percent*len(data_inputs  )): ] 

    mc_test_inputs     = mc_inputs[int( validation_percent*len(mc_inputs  )): ] 
    mc_test_conditions = mc_conditions[int( validation_percent*len(mc_inputs  )):] 
    mc_test_weights    = mc_weights[int( validation_percent*len(mc_inputs  )):] 

    # now, all the tensors are saved so they can be read by the training class
    path_to_save_tensors = "/net/data_cms3a-1/daumann/PhD/EarlyHgg/simulation_to_data_corrections/data_reading/saved_tensors/zee_tensors/"

    torch.save( data_training_inputs       , path_to_save_tensors + 'data_training_inputs.pt' )
    torch.save( data_training_conditions   , path_to_save_tensors + 'data_training_conditions.pt')
    torch.save( data_training_weights      , path_to_save_tensors + 'data_training_weights.pt') 

    torch.save( mc_training_inputs       , path_to_save_tensors + 'mc_training_inputs.pt' )
    torch.save( mc_training_conditions   , path_to_save_tensors + 'mc_training_conditions.pt')
    torch.save( mc_training_weights      , path_to_save_tensors + 'mc_training_weights.pt') 

    # now the validation tensors
    torch.save( data_validation_inputs       , path_to_save_tensors + 'data_validation_inputs.pt' )
    torch.save( data_validation_conditions   , path_to_save_tensors + 'data_validation_conditions.pt')
    torch.save( data_validation_weights      , path_to_save_tensors + 'data_validation_weights.pt') 

    torch.save( mc_validation_inputs       , path_to_save_tensors + 'mc_validation_inputs.pt' )
    torch.save( mc_validation_conditions   , path_to_save_tensors + 'mc_validation_conditions.pt')
    torch.save( mc_validation_weights      , path_to_save_tensors + 'mc_validation_weights.pt') 

    # now the test tensors
    torch.save( data_test_inputs       , path_to_save_tensors + 'data_test_inputs.pt' )
    torch.save( data_test_conditions   , path_to_save_tensors + 'data_test_conditions.pt')
    torch.save( data_test_weights      , path_to_save_tensors + 'data_test_weights.pt') 

    torch.save( mc_test_inputs       , path_to_save_tensors + 'mc_test_inputs.pt' )
    torch.save( mc_test_conditions   , path_to_save_tensors + 'mc_test_conditions.pt')
    torch.save( mc_test_weights      , path_to_save_tensors + 'mc_test_weights.pt')     

    return path_to_save_tensors

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def _read_parquet_file(path: str) -> pd.DataFrame:
    """Helper to read a single Parquet file."""
    return pd.read_parquet(path)


def read_parquet_files(patterns: list[str], max_workers: int | None = None) -> pd.DataFrame:
    """
    Parallel reading of multiple Parquet files matching given glob patterns.
    Returns a concatenated DataFrame, with a progress bar.
    """
    # Expand all glob patterns into real file paths
    files = [f for pat in patterns for f in glob.glob(pat)]
    if not files:
        return pd.DataFrame()

    dfs: list[pd.DataFrame] = []
    # Read in parallel with progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for df in tqdm(
            executor.map(_read_parquet_file, files),
            total=len(files),
            desc="Reading parquet files",
        ):
            dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def read_and_norm_mc(
    pattern: str,
    lumi_norm: float,
    max_workers = 24,
) -> pd.DataFrame:
    """
    Read MC samples matching the pattern, apply luminosity normalization,
    and return concatenated DataFrame.
    """
    df = read_parquet_files([pattern], max_workers)
    if df.empty:
        return df
    df["weight"] = df["weight"] * lumi_norm
    return df


def read_zee_data(
    var_list: list,
    conditions_list: list,
    data_samples_path: list[str],
    mc_samples_path: list[str],
    mc_samples_lumi_norm: list[float],
    DokinematicsRW: bool,
    max_workers: int | None = None,
) -> None:
    """
    Reads data and MC samples in parallel, applies selections and (optional) reweighting,
    normalizes weights, plots distributions for validation, and splits into train/val/test,
    with progress bars for I/O.
    """
    print('Begin data reading...')
    data_df = read_parquet_files(data_samples_path, max_workers)
    print(f'Read {len(data_df)} data events.')

    print('Begin MC reading...')
    mc_dfs: list[pd.DataFrame] = []
    for pattern, lumi in tqdm(
        zip(mc_samples_path, mc_samples_lumi_norm),
        total=len(mc_samples_path),
        desc="Reading MC samples",
    ):
        df = read_and_norm_mc(pattern, lumi, max_workers)
        mc_dfs.append(df)
    drell_yan_df = pd.concat(mc_dfs, ignore_index=True)
    print(f'Read total {len(drell_yan_df)} MC events.')

    # Apply loose selection to reduce background
    data_df = perform_zee_selection(data_df)
    drell_yan_df = perform_zee_selection(drell_yan_df)

    # Save MC weights before reweighting for validation
    mc_weights_before = drell_yan_df["weight"].to_numpy()

    # Kinematic reweighting
    if DokinematicsRW:
        data_w, mc_w = perform_reweighting(drell_yan_df, data_df)
        data_df["weight"], drell_yan_df["weight"] = data_w, mc_w
    else:
        data_df["weight"] = 1.0

    # Normalize weights so each dataset sums to number of data events
    total_events = len(data_df)
    def normalize(arr: np.ndarray) -> np.ndarray:
        return arr * total_events / arr.sum()

    data_df["weight"] = normalize(data_df["weight"].to_numpy())
    drell_yan_df["weight"] = normalize(drell_yan_df["weight"].to_numpy())
    mc_weights_before = normalize(mc_weights_before)

    # Sanity checks
    diff_sum = abs(drell_yan_df["weight"].sum() - data_df["weight"].sum())
    assert diff_sum <= total_events * 0.02, (
        f"Weight-sum difference {diff_sum:.2f} exceeds 2% tolerance."
    )

    mc_weights = drell_yan_df["weight"].to_numpy()
    data_weights = data_df["weight"].to_numpy()
    print(f'Sum MC weights: {mc_weights.sum():.2f}')
    print(f'Sum data weights: {data_weights.sum():.2f}')

    # Plot for validation
    plot_dir = "./plot/validation_plots/reweighting_validation/"
    plot_utils.plot_distributions(
        plot_dir,
        data_df,
        drell_yan_df,
        data_weights,
        mc_weights,
        [var_list, conditions_list],
        weights_before_rw=mc_weights_before,
    )

    # Split into train/val/test and save
    separate_training_data(
        data_df,
        drell_yan_df,
        mc_weights,
        data_weights,
        var_list,
        conditions_list,
    )

    print('End of data reading. No errors encountered.')
    print(
        f'Number of MC events: {len(drell_yan_df)}, '
        f'Number of data events: {len(data_df)}'
    )
