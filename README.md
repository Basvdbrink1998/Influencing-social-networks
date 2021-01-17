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


## Format of the results

Simulation results are saved as data frames in [feather format](https://blog.rstudio.com/2016/03/29/feather/)

## CoMSES repository

Feather files with original simulation results are available in CoMSES version of the repository.
They are stored in standard `results` subdirectory.
They have to moved to `code/data` directory for R scripts to work without any changes.

# Computation time

Simulations are comptationally expensive and may take several hours even when run on multiple cores in parallel.
