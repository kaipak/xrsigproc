from distutils.core import setup

setup(
    name = 'llctoolkit',
    author = 'Kai Pak',
    author_email = 'kai@kaipak.org',
    classifiers = [],
    description = 'Tools for analysis of LLC4320 datasets using Python scientific libraries.',
    download_url = 'https://github.com/kaipak/llctoolkit.git/tarball/0.1/',
    install_requires =[
        'astropy',
        'matplotlib',
        'numpy',
        're',
        'scipy',
        'traceback',
    ],
    keywords = ['FFT', 'fourier transform', 'tangent plane', 'plotting'], 
    license = 'MIT',
    packages = ['llctoolkit'], # this must be the same as the name above
    version = '0.1.2',
    url = 'https://github.com/kaipak/llctoolkit.git',
)
