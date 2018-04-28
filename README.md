# xrsigproc - Data and Signal Processing Tools for Xarray Datasets

Package primarily focuses on filtering signals into large and small-scale components
using convolution kernels of various types (currently 2D tophat, 2D boxcar, and Gaussian). 
This package is created specifically for processing of model output of physical oceanography 
simulation model output---in particular from LLC4320 MITgcm---so there are tools related to
processing these datasets.

## Installation

```
pip install xrsigproc
```

## xrsigproc
General purpose tools for handling signal processing of LLC4320 data. 

### `plot_spectrum`
Wrapper for matplotlib with commonly used options for plotting spectra.

### `d2k_tangent_plane`
LLC4320 data has uneven grids and xrft will complain about this. This function takes
an input grid dataset and evens out the spacing calculating latitude distance by taking
the midpoint of the plane and calculating a common arc length from there.
