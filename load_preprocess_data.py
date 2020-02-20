# load dependencies
import h5py
import utils
import numpy as np
import glob
import pickle
import os

import xarray # for organizing and storing the data 
import pandas as pd

def load(fdir, fs, trial_start_end_seconds, conditions, num_avg_groups):
    
    """
    Takes in a numpy 2d array and a subplot location, and plots a heatmap at the subplot location without axes

    Parameters
    ----------
    fdir : string
        root file directory. Needs to have a "_framenumberforevents.pkl" file that corresponds to the session!!

    fs : float
        Sampling rate of the recording

    trial_start_end_seconds : list 
        list with two float entries. First 
    
    conditions : list
        list of strings that correspond to the behavioral conditions to be analyzed 
    
    num_avg_groups : int
        Number of segments to split and average the trials over. Ie. Because single trial plots in state space is noisy, 
        we break the trials up into groups and average to get less noisier signal.

    Returns
    -------
    data_dict : dictionary
            1st level of dict keys: individual conditions + condition combined data
                2nd level of keys :
                    data : numpy 4d array with dimensions (trials,y,x,samples)
                    num_samples : number of samples (time) in a trial
                    num_trials : total number of trials in the condition

    """

    data_snip = utils.load_h5(sima_h5_path)

    data_dims = data_snip.shape
    tvec = np.linspace(0, data_dims[2]/fs, data_dims[2])

    #load behavioral data and trial info
    try:
        glob_frame_files = glob.glob(fdir + "framenumberforevents*") # look for a file in specified directory
        frame_events = pickle.load( open( glob_frame_files[0], "rb" ), encoding="latin1" ) # latin1 b/c original pickle made in python 2
    except:
        print('Cannot find behavioral data file or file path is incorrect; utils.extract_trial_data will throw error.')
        
    # with trial start/end samples, 
    trial_window_samp = trial_start_end_seconds*fs # turn trial start/end times to samples
    data_dict= utils.extract_trial_data(data_snip, trial_window_samp[0], trial_window_samp[1],
                                                           frame_events, conditions)

    """let's load data into xarray format, which has numerous 
    advantages over using numpy arrays, one of which is the ability 
    to assign names to dimensions rather than indexing by ints """

    for condition in conditions:
        
        # create index vectors for data dimensions; xarrayy stores these indices (eg. encodes time in seconds in place of samples)
        ypix_vec = range(0,data_dims[0])
        xpix_vec = range(0,data_dims[1])
        flattenpix_vec = range(0,data_dims[0]*data_dims[1])
        trials_vec = range(data_dict[condition]['num_trials'])
        data_dict['trial_tvec'] = np.linspace(trial_start_end_seconds[0], trial_start_end_seconds[1], data_dict[condition]['num_samples'])


        # xarray with dimensions: x,y,trial,samples
        data_dict[condition]['xarr_data'] = xarray.DataArray(data_dict[condition]['data'], coords=[trials_vec, ypix_vec, xpix_vec, data_dict['trial_tvec']], dims=['trial', 'y', 'x', 'time'])

        # flatten x and y pixels into one dimension
        # reshape data and make xarray with dims: x-y,trial,samples
        flatten_pix_trial_data = np.reshape(data_dict[condition]['data'], (len(trials_vec), data_dims[0]*data_dims[1], len(data_dict['trial_tvec'])))
        data_dict[condition]['xarr_flatten_xy'] = xarray.DataArray( flatten_pix_trial_data, # this flattens only the x,y dimensions
                                           coords=[trials_vec, flattenpix_vec, data_dict['trial_tvec']],
                                           dims=['trial', 'yx', 'time'])

        # average across trials
        data_dict[condition]['xarr_flatten_pix_trialAvg'] = data_dict[condition]['xarr_flatten_xy'].mean(dim = 'trial')

        ### https://stackoverflow.com/questions/43015638/xarray-reshape-data-split-dimension
        # unstack trials into groups and average across trials (avged trials grouped by time)

        num_trials_to_avg = data_dict[condition]['num_trials']/num_avg_groups

        # need to create a pandas multi-index to tell xarray the target dimensions to unpack into
        ind = pd.MultiIndex.from_product([np.arange(0, num_trials_to_avg), np.arange(0, num_avg_groups)],
                                         names=['trials', 'trial_groups'])[np.arange(0, data_dict[condition]['num_trials'])] 
        # last arange cuts the index list if the number of trials per group does divide evenly into total num trials

        data_dict[condition]['xarr_flatten_xy_group_trials'] = data_dict[condition]['xarr_flatten_xy'].assign_coords(trial=ind).unstack('trial').mean(dim = 'trials').transpose('trial_groups', 'yx', 'time')
        ###

    # pull out all trial-avged data for each cond, then average across conditions
    data_dict['all_cond'] = {}
    
    # make an array with dimensions trials, xy_pixels, samples where trials from all conditions are stacked in the first dimension
    stacked_data = np.stack([data_dict[condition]['xarr_flatten_xy'].data 
                                                    for condition in conditions], axis = 0)
    data_shape = stacked_data.shape
    data_dict['all_cond']['flattenpix'] = stacked_data.reshape(data_shape[0]*data_shape[1], data_shape[2], data_shape[3])

    data_dict['all_cond']['flattenpix_trial_cond_avg'] = np.average( [data_dict[condition]['xarr_flatten_pix_trialAvg'].data 
                                           for condition in conditions], axis=0)

    return data_dict




