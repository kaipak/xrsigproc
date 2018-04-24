from astropy.convolution import Box2DKernel, Tophat2DKernel
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import convolve

import xarray as xr

mask_vars = {('YC', 'XC'): 'hFacC',
             ('YC', 'XG'): 'hFacW',
             ('YG', 'XC'): 'hFacS'}

def gaussian_smooth(data, sigma=5, mask=False):
    xdim = 'XC' if 'XC' in data.dims else 'XG'
    ydim = 'YC' if 'YC' in data.dims else 'YG'
    dims = (ydim, xdim)

    sc_gaussian_nd = lambda data: gaussian_filter(data, sigma, mode='wrap')

    if mask:
        data_masked = data.where(ds[mask_vars[dims]])
    else:
        data_masked = data.fillna(0.)

    return xr.apply_ufunc(sc_gaussian_nd, data_masked,
                          vectorize=True,
                          dask='parallelized',
                          input_core_dims = [dims],
                          output_core_dims = [dims],
                          output_dtypes=[data.dtype])


def box2D_smooth(data, size=90, mask=False):
    xdim = 'XC' if 'XC' in data.dims else 'XG'
    ydim = 'YC' if 'YC' in data.dims else 'YG'
    dims = (ydim, xdim)
    
    box2d_kernel = Box2DKernel(size).array
    sc_convolve = lambda data: convolve(data, box2d_kernel, mode='same')
    
    if mask:
        data_masked = data.where(ds[mask_vars[dims]])
    else:
        data_masked = data.fillna(0.)
        
    return xr.apply_ufunc(sc_convolve, data_masked,
                          vectorize=True,
                          dask='parallelized',
                          input_core_dims = [dims],
                          output_core_dims = [dims],
                          output_dtypes=[data.dtype])

def tophat2D_smooth(data, size=90, mask=False):
    xdim = 'XC' if 'XC' in data.dims else 'XG'
    ydim = 'YC' if 'YC' in data.dims else 'YG'
    dims = (ydim, xdim)
    
    tophat2d_kernel = Tophat2DKernel(size).array
    sc_convolve = lambda data: convolve(data, tophat2d_kernel, mode='same')
    
    if mask:
        data_masked = data.where(ds[mask_vars[dims]])
    else:
        data_masked = data.fillna(0.)
        
    return xr.apply_ufunc(sc_convolve, data_masked,
                          vectorize=True,
                          dask='parallelized',
                          input_core_dims = [dims],
                          output_core_dims = [dims],
                          output_dtypes=[data.dtype])



def germano_tau(f, g, operator, size=45, mask=False):
    """Calculute small-scale variance of a signal according to Germano 1992 paper
       Note, for box2d, size is the total width; for tophat2d, it's the radius; 
       for a gaussian filter, size is sigma.

    """
    return operator(f * g, size=size, mask=mask) - (operator(f, size=size, mask=mask) * 
                    operator(g, size=size, mask=mask))

def gaussian_tau(f, g, operator, sigma=5, mask=False):
    """Likely will get removed in future release"""
    return operator(f * g, sigma=sigma, mask=mask) - (operator(f, sigma=sigma, mask=mask) *
                              operator(g, sigma=sigma, mask=mask))
