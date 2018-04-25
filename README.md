# llctoolkit - Data and Signal Processing Tools for LLC4320 Datasets

## llctoolkit
General purpose tools for handling signal processing of LLC4320 data. 

### `plot_spectrum`
Wrapper for matplotlib with commonly used options for plotting spectra.

### `d2k_tangent_plane`
LLC4320 data has uneven grids and xrft will complain about this. This function takes
an input grid dataset and evens out the spacing calculating latitude distance by taking
the midpoint of the plane and calculating a common arc length from there.

