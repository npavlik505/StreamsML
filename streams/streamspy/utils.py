from config import Config
#from .config import Config
import numpy as np
from globals import is_master
#from .globals import is_master

# print, but only print on the master process
def hprint(*args):
    if is_master:
        print(*args)

# in place, write the averages of `streams_data` to `span_average`. We also need `temp_field`
# so we can compute the divisions for (for example) rho*u / rho 
def calculate_span_averages(config: Config, span_average: np.ndarray, temp_field: np.ndarray, streams_data: np.ndarray):
    # for rhou, rhov, rhow, and rhoE
    for data_idx in [1,2,3,4]:
        # divide rho * VALUE by rho so we exclusively have VALUE
        temp_field[:] = np.divide(streams_data[data_idx, :, :, :], streams_data[0, :, :, :], out = temp_field[:])

        # sum the VALUE across the z direction
        span_average[data_idx, :, :] = np.sum(temp_field, axis=2, out=span_average[data_idx, :, :])
        # divide the sum by the number of points in the z direction to compute the average
        span_average[data_idx, :, :] /= config.grid.nz

    span_average[0, :, :] = np.sum(streams_data[0, :, :, :], axis=2, out=span_average[0, :, :])
    span_average[0, :, :] /= config.grid.nz
