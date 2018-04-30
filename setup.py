from distutils.core import setup

setup(
    name = 'xrsigproc',
    author = 'Kai Pak',
    author_email = 'kai@kaipak.org',
    classifiers = [],
    description = 'Tools for signal processing of Xarray datasets using Python scientific libraries.',
    download_url = 'https://github.com/kaipak/xrsigproc.git/tarball/0.1.3/',
    install_requires =[
        'astropy',
        'matplotlib',
        'numpy',
        'scipy',
    ],
    keywords = ['FFT', 'fourier transform', 'tangent plane', 'plotting'], 
    license = 'MIT',
    long_description = 'Apply convolution to signals using various kernels to filter\
                        signals into large and small scale components.',
    packages = ['xrsigproc'],
    version = '0.1.3',
    url = 'https://github.com/kaipak/xrsigproc.git',
)
