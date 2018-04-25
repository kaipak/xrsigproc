from distutils.core import setup

setup(
    name = 'llctoolkit',
    author = 'Kai Pak',
    author_email = 'kai@kaipak.org',
    classifiers = [],
    description = 'Tools for analysis of LLC4320 datasets using Python scientific libraries.',
    download_url = 'https://github.com/kaipak/llctoolkit.git/tarball/0.1.4/',
    install_requires =[
        'astropy',
        'matplotlib',
        'numpy',
        'scipy',
    ],
    keywords = ['FFT', 'fourier transform', 'tangent plane', 'plotting'], 
    license = 'MIT',
    long_description = 'Tools for analyzing LLC4320 datasets.',
    packages = ['llctoolkit'],
    version = '0.0.1',
    url = 'https://github.com/kaipak/llctoolkit.git',
)
