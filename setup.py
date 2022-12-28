import setuptools
import sys

if sys.version_info.major != 3 and sys.version_info.minor < 6:
    raise TypeError('This Python is only compatible with Python >= 3.6, but you are running '
                    'Python {}. The installation will likely fail.'.format(
                        sys.version_info.major))

setuptools.setup(
    name="robustness",  # this is the name displayed in 'pip list'
    version="0.1",
    description="Computing the robustness of control agents.",
    install_requires=[
        'cma',
        'moarchiving',
        'numpy',
        'scipy',
        'matplotlib',
        'signal_temporal_logic',
        'imageio',
        'pygmo',
    ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)