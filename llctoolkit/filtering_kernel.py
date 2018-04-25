from astropy.convolution import Box2DKernel, Tophat2DKernel
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import convolve

import xarray as xr

mask_vars = {('YC', 'XC'): 'hFacC',
             ('YC', 'XG'): 'hFacW',
             ('YG', 'XC'): 'hFacS'}

def gaussian_smooth(data, scale=5, mask=False):
    xdim = 'XC' if 'XC' in data.dims else 'x_km'
    ydim = 'YC' if 'YC' in data.dims else 'y_km'
    dims = (ydim, xdim)

    sc_gaussian_nd = lambda data: gaussian_filter(data, scale, mode='wrap')

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

def box2D_smooth(data, scale=90, mask=False):
    xdim = 'XC' if 'XC' in data.dims else 'XG'
    ydim = 'YC' if 'YC' in data.dims else 'YG'
    dims = (ydim, xdim)

    box2d_kernel = Box2DKernel(scale).array
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

def tophat2D_smooth(data, scale=45, mask=False):
    xdim = 'XC' if 'XC' in data.dims else 'x_km'
    ydim = 'YC' if 'YC' in data.dims else 'y_km'
    dims = (ydim, xdim)

    tophat2d_kernel = Tophat2DKernel(scale).array
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

def germano_tau(f, g, operator, scale=45, mask=False):
    """Small scale variance according to M. Germano 1990 paper where
       <tau>_ss = <f * g> - <f> * <g> and bracket terms are convolved inputs

       Scale is analogous to sigma if using a gaussian kernel.
    """
    return operator(f * g, scale=scale, mask=mask) - (operator(f, scale=scale, mask=mask) *
                    operator(g, scale=scale, mask=mask))
