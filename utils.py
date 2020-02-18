import h5py
import numpy as np
import matplotlib.pyplot as plt

def load_h5(fpath):
    data_h5file = h5py.File(fpath, 'r')
    # load a snippit of data and get rid of un-needed singleton dimensions
    data_snip = np.squeeze(np.array(data_h5file['imaging'], dtype=int))

    """ typically it's good to have time as the last dimension because one doesn't usually iterate through time, so we'll
     reorganize the data dimension order"""
    return data_snip.transpose(1, 2, 0)

def plot_single_img(to_plot, frame_num):
    plt.figure(figsize=(7, 7))
    plt.imshow(to_plot, cmap='gray')
    plt.title(f'Frame {frame_num}', fontsize=20)
    plt.axis('off')

def subplot_heatmap(axs, title, image, cmap="seismic", clims=None, zoom_window=None):
    """
        Takes in a numpy 2d array and a subplot location, and plots a heatmap at the subplot location without axes

        Parameters
        ----------
        axs : matplotlib AxesSubplot object
            Specific subplot from the ax output of pyplot.subplots()

        title : string
            Title name of the plot

        image : numpy 2d array

        cmap : string or colormap
            Colormap desired; default is seismic

        Optional Parameters
        -------------------

        clims : list
            List with entries: [minimum_colorbar_limit, maximum_colorbar_limit] . This is for setting the color ranges
            for the heatmap

        zoom_window : list
            List with entries: [xmin, xmax, ymin, ymax] . This is for zooming into the specific window dictated by the
            x min and max, and y min and max locations

        Returns
        -------
        im : ndarray
            imshow AxesImage object. Used to reference the dataset for a colorbar (eg. fig.colorbar(im) )

        """

    im = axs.imshow(image, cmap)
    axs.set_title(title, fontsize=20)

    if zoom_window is not None:
        axs.axis(zoom_window)
        axs.invert_yaxis()

    if clims is not None:
        im.set_clim(vmin=clims[0], vmax=clims[1])

    axs.axis('off')

    return im  # for colorbar

def dict_key_len(dict_, key):
    return len(dict_[key])

def make_tile(start, end, num_rep):
    """
    Makes indices for tiles.

    Parameters
    ----------
    start_end : int
        List with two items where first int is start sample relative to trial onset.
        Second int is end sample relative to trial onset.

    num_rep : int
        Number of times to repeat the sample vector in the y axis

    Returns
    -------
    tile_array : ndarray
        Array with shape (num_rep, samples), where samples is number of samples between
        the items in start_end input

    """

    samp_vec = np.arange(start, end + 1)  # grab all samples between start/end

    tile_array = np.tile(samp_vec, (num_rep, 1))

    return tile_array

def remove_trials_out_of_bounds(data_end, these_frame_events, start_samp, end_samp):

    after_start_bool = (these_frame_events + start_samp) > start_samp
    before_end_bool = (these_frame_events + end_samp) < data_end

    return these_frame_events[after_start_bool*before_end_bool]

def extract_trial_data(data, start_samp, end_samp, frame_events, conditions):
    """
        Takes a 3d video (across a whole session) and cuts out trials based on event times.
        Also groups trial data by condition

        Parameters
        ----------
        data : numpy 3d array
            3d video data where dimensions are (y_pixel, x_pixel, samples)

        start_samp : int
            Number of samples before the event time for trial start

        end_samp : int
            Number of samples after the event time for trial end

        frame_events : dictionary of np 1d arrays (vectors)
            Dictionary where keys are the conditions in the session and values are numpy 1d vectors that contain
            event occurrences as samples

        conditions : list of strings
            Each entry in the list is a condition to extract trials from; must correspond to keys in frame_events

        Returns
        -------
        data_dict : dictionary
            1st level of dict keys: individual conditions
                2nd level of keys :
                    data : numpy 4d array with dimensions (trials,y,x,samples)
                    num_samples : number of samples (time) in a trial
                    num_trials : total number of trials in the condition

        """

    data_dict = {}

    for idx, condition in enumerate(conditions):

        data_dict[condition] = {}

        # get rid of trials that are outside of the session bounds with respect to time
        data_end_sample = data.shape[-1]
        frame_events[condition] = remove_trials_out_of_bounds(data_end_sample, frame_events[condition], start_samp, end_samp)

        # convert window time bounds to samples and make a trial sample vector
        # make an array where the sample indices are repeated in the y axis for n number of trials
        num_trials_cond = len(frame_events[condition])
        svec_tile = make_tile(start_samp, end_samp, num_trials_cond)
        num_trial_samps = svec_tile.shape[1]

        # now make a repeated matrix of each trial's ttl on sample in the x dimension
        ttl_repmat = np.repeat(frame_events[condition][:, np.newaxis], num_trial_samps, axis=1).astype('int')

        trial_sample_mat = ttl_repmat + svec_tile

        # extract frames in trials and reshape the data to be: y,x,trials,samples
        # basically unpacking the last 2 dimensions
        reshape_dim = data.shape[:-1] + (svec_tile.shape)
        extracted_trial_dat = data[:, :, np.ndarray.flatten(trial_sample_mat)].reshape(reshape_dim)

        # reorder dimensions and put trial as first dim
        data_dict[condition]['data'] = extracted_trial_dat.transpose((2, 0, 1, 3))
        data_dict[condition]['num_samples'] = num_trial_samps
        data_dict[condition]['num_trials'] = num_trials_cond

    return data_dict


def time_to_samples(trial_tvec, analysis_window, fs):
    """Takes in a numpy 2d array and a subplot location, and plots a heatmap at the subplot location without axes

    Parameters
    ----------
    trial_tvec : np 1d vector
        Vector of times in seconds that correspond to the samples in the data

    analysis_window : np 1d vector , entries are floats
        First entry is the window start time in seconds, second entry is the window end time
        in seconds.

    fs : float
        Sampling rate in Hz

    Returns
    -------
    analysis_svec : np 1d vector
        Vector of samples that are the corresponding samples in the trial_tvec between the start and end times

    """

    analysis_win_start_samp = np.argmin(abs(trial_tvec - analysis_window[0]))
    analysis_win_end_samp = np.argmin(abs(trial_tvec - analysis_window[1]))

    analysis_svec = np.arange(analysis_win_start_samp, analysis_win_end_samp)

    return analysis_svec