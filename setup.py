from distutils.core import setup

setup(
    name = 'llcpak',
    author = 'Kai Pak',
    author_email = 'kai@kaipak.org',
    description = 'Tools for analysis of LLC4320 datasets using Python scientific libraries.'
    download_url = 'https://github.com/kaipak/llcpak.git/tarball/0.1/',
    install_requires =[
        'matplotlib',
        'numpy',
    ],
    keywords = ['FFT', 'fourier transform', 'tangent plane', 'plotting'], 
    license = 'MIT',
    packages = ['llcpak'], # this must be the same as the name above
    version = '0.1',
    url = 'https://github.com/kaipak/llcpak.git'
    classifiers = [],
)
