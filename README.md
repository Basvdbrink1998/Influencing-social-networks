# Influencing social networks using strategic subborn agents

## Authors

  * Bas van den Brink
    * University of Amsterdam (UvA)

  Based on work by:
  * Szymon Talaga
    * The Robert Zajonc Institute for Social Studies, University of Warsaw
  * Andrzej Nowak
    * The Robert Zajonc Institute for Social Studies, University of Warsaw

# Introduction

This is a code repository for a thesis "Influencing social networks" based on the code repository for the paper "Homophily as a process generating social networks: insights from Social Distance Attachment model".
It contains all code necessary to replicate the simulation presented in the thesis. all files taken from the codebase of the paper is acknowledged in the contents section of this document.


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

- `DeGroot.py`
  - Module with implementation of the DeGroot model.
- `Enviroment.py`
  - Module which simulates networks using both the SDA and the DeGroot model.
- `_.py`
  - This module implements different strategies to influence a social network.

Code from Szymon Talaga and Andrzej Nowak:
- `_.py`
  - General module with routines for running simulations.
- `da.py`
  - Module with utilities for data analysis.
- `sdnet`
  - Module with implementation of SDA and SDC models and related utilities. Detais are documented in the source files.
- `tailest`
  - Module implementing methods from Voitalov et al. (2018). Details are documented in the source files.

## Directories

- `Influence`
  - Here all modules which simulate the social networks are located.
- `Simple_plots`
  - Here all scripts used to plot graphs which do not utilize the simulation are located.

# Computation time

Simulations are comptationally expensive and may take several hours even when run on multiple cores in parallel.
