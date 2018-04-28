from astropy.convolution import Box2DKernel, Tophat2DKernel
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import convolve
from matplotlib import pyplot as plt

import numpy as np
import xarray as xr

import re
import traceback

mask_vars = {('YC', 'XC'): 'hFacC',
             ('YC', 'XG'): 'hFacW',
             ('YG', 'XC'): 'hFacS'}

__all__ = ["plot_spectrum", "d2k_tangent_plane", "germano_tau", "tophat2D_smooth",
           "boxcar2D_smooth", "gaussian_smooth"]

def plot_spectrum(spectrum, ax=None, logx=False, logy=False, vmax=None, 
                  cmap='viridis', label=None):
    """Plot spectra with commonly used matplotlib options. Expects 
       spectra with x-axis as freq_x_km
    
    Parameters:
        spectrum (xarray.data): Data to plot
        logx (bool): Option to plot x-axis in logarithmic scale
        logy (bool): Option to plot y-axis in logarithmic scale vmax (float): Maximum range of plot
        cmap (str): Matplotlib color map to use
    
    
    """
    if ax is None:
        fig = plt.figure(figsize=(12,8))
        ax  = fig.add_subplot(111)
        
    spectrum.sel(freq_x_km=slice(0,None)).plot(ax=ax, label=label, 
                                               cmap=plt.cm.get_cmap(cmap), 
                                               vmax=vmax)
    if (logx == True):
        ax.set_xscale('log')
    if (logy == True):
        ax.set_yscale('log')

    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    return ax


def d2k_tangent_plane(da, lon0=None): 
    """Convert longitudinal degree coordinates into kilometers and evenly 
       space by average distance on a plane tangent to center point of 
       grid.
    
    Parameters:
        da  (dask.dataset): Data to have XC, YC (longitude) converted to km 
        lon_0 (float): If distance from a different longitude is required.
    
    Returns:
        dac (dask.dataset): Converted Array with longitudes replaced with 
                            km from Prime Meridian.

    """
    REQ     = 6378.1370 # Radius of Earth to the Equator
    RPO     = 6356.7523 # Radius of Earth to the Poles 
    XC_mean = (da.coords['XC'].values[-1] - da.coords['XC'].values[0]) / da.coords['XC'].count()
    YC_mean = (da.coords['YC'].values[-1] - da.coords['YC'].values[0]) / da.coords['YC'].count()
    
    # Get distance between grid points according to coordinates in center of input
    XC_mp = (da.coords['XC'].values[-1] - da.coords['XC'].values[0]) / 2.0
    YC_mp = (da.coords['YC'].values[-1] - da.coords['YC'].values[0]) / 2.0
    
    # Basis of even spaced grid we'll produce
    XC_even = np.zeros(da.coords['XC'].shape)
    XC_even[0] = da.coords['XC'].values[0]
    YC_even    = np.zeros(da.coords['YC'].shape)
    YC_even[0] = da.coords['YC'].values[0]
    
    for i in range(0, XC_even.size - 1): # X and Y coordinates assumed to be same size
        XC_even[i+1] = XC_even[i] + XC_mean
        YC_even[i+1] = YC_even[i] + YC_mean
    
    dac = da.copy()
    dac.coords['XC'] = np.pi * REQ * np.abs(np.cos(YC_mp)) * XC_even / 180.0 
    dac.coords['YC'] = np.pi * RPO * YC_even / 180.0 
    
    # Convert to km and return a copy of dataset
    
    dac    = dac.rename({'XC': 'x_km'}) # Change label
    dac    = dac.rename({'YC': 'y_km'})
    
    return dac


def gaussian_smooth(data, scale=5, mask=False, mode='reflect'):
    """Apply gaussian kernel to convolution. Uses Scipy 
       gaussian_filter method.

       Parameters:
       mode (str): {‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’}
                   What to do at edges of matrix input. See Scipy docs
                   for details on what these do.

    """
    xdim = 'XC' if 'XC' in data.dims else 'x_km'
    ydim = 'YC' if 'YC' in data.dims else 'y_km'
    dims = (ydim, xdim)

    sc_gaussian_nd = lambda data: gaussian_filter(data, scale, mode=mode)

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


def boxcar2D_smooth(data, scale=90, mask=False):
    xdim = 'XC' if 'XC' in data.dims else 'x_km'
    ydim = 'YC' if 'YC' in data.dims else 'y_km'
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
