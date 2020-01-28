import h5py
import numpy as np

def load_h5(fpath):
    data_h5file = h5py.File(fpath, 'r')
    # load a snippit of data and get rid of un-needed singleton dimensions
    data_snip = np.squeeze(np.array(data_h5file['imaging'], dtype=int))

    """ typically it's good to have time as the last dimension because one doesn't usually iterate through time, so we'll
     reorganize the data dimension order"""
    return data_snip.transpose(1, 2, 0)