import matplotlib as plt

def plot_spectrum(spectrum, ax=None, logx=False, logy=False, vmax=None, 
                  cmap='viridis', label=None):
    """Plot spectra with commonly used matplotlib options. Expects 
       spectra with x-axis as freq_x_km
    
    Parameters:
        spectrum (xarray.data): Data to plot
        logx (bool): Option to plot x-axis in logarithmic scale
        logy (bool): Option to plot y-axis in logarithmic scale
        vmax (float): Maximum range of plot
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
    #vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
    
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(plotAxFormatter))
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
    XC_mean = (ds.coords['XC'].values[-1] - ds.coords['XC'].values[0]) / ds.coords['XC'].count()
    YC_mean = (ds.coords['YC'].values[-1] - ds.coords['YC'].values[0]) / ds.coords['YC'].count()
    
    # Get distance between grid points according to coordinates in center of input
    XC_mp = (ds.coords['XC'].values[-1] - ds.coords['XC'].values[0]) / 2.0
    YC_mp = (ds.coords['YC'].values[-1] - ds.coords['YC'].values[0]) / 2.0
    
    # Basis of even spaced grid we'll produce
    XC_even = np.zeros(ds.coords['XC'].shape)
    XC_even[0] = ds.coords['XC'].values[0]
    YC_even    = np.zeros(ds.coords['YC'].shape)
    YC_even[0] = ds.coords['YC'].values[0]
    
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
