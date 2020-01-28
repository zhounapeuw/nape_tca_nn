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

def dict_key_len(dict_, key):
    return len(dict_[key])

def make_tile(start_end, num_rep):
    """
    Makes indices for tiles.

    Parameters
    ----------
    start_end : list
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

    samp_vec = np.arange(start_end[0], start_end[1] + 1)  # grab all samples between start/end

    tile_array = np.tile(samp_vec, (num_rep, 1))

    return tile_array