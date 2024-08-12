# Bayesian Blocks MATLAB Implementation

This directory contains an implementation of the **Bayesian Blocks** algorithm. The algorithm is used to identify change points in time series data, allowing for an adaptive binning of the data based on its underlying structure. This implementation is based on the methodology described by Scargle et al. in their paper, which can be accessed [here](https://arxiv.org/abs/1207.5578).

## Files

- **`main.m`**: The main script to run the Bayesian Blocks algorithm. It utilizes the `prepare_data` function to process input data and then calls the `find_blocks` function to detect change points. Results are displayed at the end of the script.

- **`prepare_data.m`**: This script prepares the input data for the Bayesian Blocks algorithm. It loads the data, processes it, and formats it appropriately for further analysis.


## Credits

The algorithm and its theoretical foundation are credited to Scargle et al. (2013). For more detailed information, please refer to their paper: [Studies in Astronomical Time Series Analysis. VI. Bayesian Block Representations](https://arxiv.org/abs/1207.5578).