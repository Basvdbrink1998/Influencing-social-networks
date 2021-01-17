# Influencing social networks using strategic subborn agents

## Authors


# Introduction


# Requirements

The project uses Python.
Code is multiplatform and should run on any standard operating system (i.e. Linux/Unix, MacOS, Windows).

## Python dependencies

 - Python3.6+
 - Other dependencies are specified in the `requirements.txt` file.

   To install them run `pip install -r requirements.txt`.

 - It is a good practice to install python dependencies in an isolated virtual environment.

# Content of the repository

## Scripts


## Python modules

  - `_.py`
    - General module with routines for running simulations.
  - `da.py`
    - Module with utilities for data analysis.
  - `sdnet`
    - Module with implementation of SDA and SDC models and related utilities. Detais are documented in the source files.
  - `tailest`
    - Module implementing methods from Voitalov et al. (2018). Details are documented in the source files.

## Directories


# Computation time

Simulations are comptationally expensive and may take several hours even when run on multiple cores in parallel.
