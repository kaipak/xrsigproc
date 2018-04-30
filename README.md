# xrsigproc - Data and Signal Processing Tools for Xarray Datasets

Package primarily focuses on filtering signals into large and small-scale components
using convolution kernels of various types (currently 2D tophat, 2D boxcar, 2D cone, and Gaussian). 
This package is created specifically for processing of model output of physical oceanography 
simulation model output---in particular from LLC4320 MITgcm---so there are tools related to
processing these datasets.

## Installation

```
pip install xrsigproc
```


### `plot_spectrum`
Wrapper for matplotlib with commonly used options for plotting spectra.

### `d2k_tangent_plane`
LLC4320 data has uneven grids and xrft will complain about this. This function takes
an input grid dataset and evens out the spacing calculating latitude distance by taking
the midpoint of the plane and calculating a common arc length from there.

## Examples on applying a convolution kernel

There's a variety of kernels included in this package including:

* `gaussian_smooth`
* `boxcar2D_smooth`
* `cone2D_smooth`
* `tophat2D_smooth`

Simply choose the size of the kernel and apply it to your dataset:

```
import xrsigproc as sp

sp.gaussian_smooth(dataset, scale=5)

```

For the boxcar kernel, scale refers to the total width, for the round kernels, it refers to the
radius, and for the gaussian kernel, it refers to sigma. The functions will use Dask parallelization
where it can. 

There's a helper function to compute small-scale variance according to M. Germano's 1990 paper, 
*Turbulence: the filtering approach*, where he defines small-scale variance as tau_ss = \<f*g\> - \<f\> * \<g\>.
Angle brackets here denote a convolution operator. 

```
sp.germano_tau(dataset, dataset, gaussian_smooth, scale=5)

```