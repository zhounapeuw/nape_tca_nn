import numpy as np
import utils

def test_make_tile_start_end_vectors():
    start_end = [-5, 15]
    num_reps = 20
    test_tile = utils.make_tile(start_end, num_reps)

    start_rep = np.repeat(start_end[0], num_reps)
    end_rep = np.repeat(start_end[1], num_reps)
    assert test_tile[:, 0] == start_rep # make sure first column is a repeated vector of start samples
