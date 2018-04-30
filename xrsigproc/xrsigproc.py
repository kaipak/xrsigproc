from astropy.convolution import Box2DKernel, Tophat2DKernel, TrapezoidDisk2DKernel
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import convolve
from matplotlib import pyplot as plt

import matplotlib.colors as colors
import numpy as np
import xarray as xr

import re
import traceback

mask_vars = {('YC', 'XC'): 'hFacC',
             ('YC', 'XG'): 'hFacW',
             ('YG', 'XC'): 'hFacS'}

__all__ = ["plot_spectrum", "d2k_tangent_plane", "germano_tau", "tophat2D_smooth",
           "boxcar2D_smooth", "cone2D_smooth", "gaussian_smooth"]


def _get_dims(data):
    """Get primary x-y dimensions of dataset

    """
    if 'XC' in data.dims:
        xdim = 'XC'
    elif 'XG' in data.dims:
        xdim = 'XG'
    else:
        xdim = 'x_km'

    if 'YC' in data.dims:
        ydim = 'YC'
    elif 'YG' in data.dims:
        ydim = 'YG'
    else:
        ydim = 'y_km'

    return (ydim, xdim)


def plot_spectrum(spectrum, ax=None, logx=False, logy=False, logc=False, 
                            vmin=None, vmax=None, cmap='viridis', label=None):
    """Plot 1 or 2D spectra with commonly used matplotlib options. Expects 
       spectra with x-axis as freq_x_km
    
    Parameters:
        spectrum (xarray.data): Data to plot
        logx (bool): Option to plot x-axis in logarithmic scale
        logy (bool): Option to plot y-axis in logarithmic scale 
        logc (bool): Colormap will be plotted in logarithmic scale
        vmin (float): Minimum range of plot (applies only to 2D)
        vmax (float): Maximum range of plot (applies only to 2D)
        cmap (str): Matplotlib color map to use
        label (str): Useful label for plot
    
    """
    dims = len(spectrum.dims)
    
    if ax is None:
        fig = plt.figure(figsize=(12,8))
        ax  = fig.add_subplot(111)
    
    if (logx == True):
        ax.set_xscale('log')
    if (logy == True):
        ax.set_yscale('log')
    
    if dims > 1:
        if logc:
            spectrum.sel(freq_x_km=slice(0,None)).plot(ax=ax, label=label, 
                                                       cmap=plt.cm.get_cmap(cmap), 
                                                       vmin=vmin, vmax=vmax, norm=colors.LogNorm())
        else:
            spectrum.sel(freq_x_km=slice(0,None)).plot(ax=ax, label=label, 
                                                       cmap=plt.cm.get_cmap(cmap), 
                                                       vmin=vmin, vmax=vmax)
    else:
        spectrum.sel(freq_x_km=slice(0,None)).plot(ax=ax, label=label)

    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
    
    return ax


def d2k_tangent_plane(data, lon0=None): 
    """Convert longitudinal degree coordinates into kilometers and evenly 
       space by average distance on a plane tangent to center point of 
       grid.
    
    Parameters:
        data  (dask.dataset): Data to have XC, YC (longitude) converted to km 
        lon_0 (float): If distance from a different longitude is required.
    
    Returns:
        data_copy (dask.dataset): Converted Array with longitudes replaced with 
                            km from Prime Meridian.

    """
    REQ     = 6378.1370 # Radius of Earth to the Equator
    RPO     = 6356.7523 # Radius of Earth to the Poles 
    dims  = _get_dims(data)
    Y_coord = dims[0]
    X_coord = dims[1]
    X_mean = (data.coords[X_coord].values[-1] - 
              data.coords[X_coord].values[0]) / data.coords[X_coord].count()
    Y_mean = (data.coords[Y_coord].values[-1] - 
              data.coords[Y_coord].values[0]) / data.coords[Y_coord].count()
    
    # Get distance between grid points according to coordinates in center of input
    X_mp = (data.coords[X_coord].values[-1] - data.coords[X_coord].values[0]) / 2.0
    Y_mp = (data.coords[Y_coord].values[-1] - data.coords[Y_coord].values[0]) / 2.0
    
    # Basis of even spaced grid we'll produce
    X_even = np.zeros(data.coords[X_coord].shape)
    X_even[0] = data.coords[X_coord].values[0]
    Y_even    = np.zeros(data.coords[Y_coord].shape)
    Y_even[0] = data.coords[Y_coord].values[0]
    
    for i in range(0, X_even.size - 1): # X and Y coordinates assumed to be same size
        X_even[i+1] = X_even[i] + X_mean
        Y_even[i+1] = Y_even[i] + Y_mean
    
    data_copy = data.copy()
    data_copy.coords[X_coord] = np.pi * REQ * np.abs(np.cos(Y_mp)) * X_even / 180.0 
    data_copy.coords[Y_coord] = np.pi * RPO * Y_even / 180.0 
    
    # Convert to km and return a copy of dataset
    
    data_copy = data_copy.rename({X_coord: 'x_km'}) # Change label
    data_copy = data_copy.rename({Y_coord: 'y_km'})
    
    return data_copy


def gaussian_smooth(data, scale=5, mask=False, mode='reflect'):
    """Apply gaussian kernel to convolution. Uses Scipy 
       gaussian_filter method.

       Parameters:
       mode (str): {‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’}
                   What to do at edges of matrix input. See Scipy docs
                   for details on what these do.

    """
    dims = _get_dims(data)

    sc_gaussian_nd = lambda data: gaussian_filter(data, scale, mode=mode)

    if mask:
        data_masked = data.where(data[mask_vars[dims]])
    else:
        data_masked = data.fillna(0.)

    return xr.apply_ufunc(sc_gaussian_nd, data_masked,
                          vectorize=True,
                          dask='parallelized',
                          input_core_dims = [dims],
                          output_core_dims = [dims],
                          output_dtypes=[data.dtype])


def boxcar2D_smooth(data, scale=90, mask=False):
    dims = _get_dims(data)

    box2d_kernel = Box2DKernel(scale).array
    sc_convolve = lambda data: convolve(data, box2d_kernel, mode='same')

    if mask:
        data_masked = data.where(data[mask_vars[dims]])
    else:
        data_masked = data.fillna(0.)

    return xr.apply_ufunc(sc_convolve, data_masked,
                          vectorize=True,
                          dask='parallelized',
                          input_core_dims = [dims],
                          output_core_dims = [dims],
                          output_dtypes=[data.dtype])


def cone2D_smooth(data, scale=90, mask=False):
    """Create a cone shaped kernel. Uses a 2D trapezoid disk from Astropy
       to create a one grid wide point with a slope that is inverse of 
       desired size.
    
       Parameters:
       data (Xarray dset): Set of data to convolve
       scale (int): Radius of base of 2D cone
    
    """
    dims = _get_dims(data)
    slope = 1.0 / scale
    cone2d_kernel = TrapezoidDisk2DKernel(scale, slope).array
    sc_convolve = lambda data: convolve(data, cone2d_kernel, mode='same')

    if mask:
        data_masked = data.where(data[mask_vars[dims]])
    else:
        data_masked = data.fillna(0.)

    return xr.apply_ufunc(sc_convolve, data_masked,
                          vectorize=True,
                          dask='parallelized',
                          input_core_dims = [dims],
                          output_core_dims = [dims],
                          output_dtypes=[data.dtype])


def tophat2D_smooth(data, scale=45, mask=False):
    dims = _get_dims(data)

    tophat2d_kernel = Tophat2DKernel(scale).array
    sc_convolve = lambda data: convolve(data, tophat2d_kernel, mode='same')

    if mask:
        data_masked = data.where(data[mask_vars[dims]])
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
