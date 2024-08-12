# Bayesian Blocks for Time Series Analysis

## Overview

This Python implementation of the `bayesian_blocks` algorithm has been specifically adapted for use within the AGILE RTA (Real-Time Analysis) framework. The primary modification in this version is the final extraction of `change_points`, which has been adjusted to meet AGILE RTA requirements by excluding the first and last excess blocks. These adjustments ensure consistency with the MATLAB implementation used within the AGILE RTA context.

This code is built upon the Bayesian Blocks implementation originally found in the Astropy library. The core logic and structure remain intact, with additional modifications to better suit the specific needs of AGILE RTA users. The original Astropy implementation can be found [here](https://docs.astropy.org/en/latest/_modules/astropy/stats/bayesian_blocks.html).

## Features

- **Optimal Histogram Binning**: Finds an optimal histogram with adaptive bin widths, making it useful for analyzing time series data with varying event rates.
- **Segmentation of Time Series Data**: Identifies optimal segments within time series data by detecting change points.
- **Inflection Point Detection**: Detects inflection points in event data, allowing for the identification of significant changes in rate.

## BBlocks Class

This implementation introduces a new class named `BBlocks`. The `BBlocks` class is designed to encapsulate both the input and output of the `bayesian_blocks` function. Additionally, it provides methods for generating preliminary plots of the obtained results, facilitating the visualization of change points and other key metrics. These capabilities streamline the workflow and enable a more comprehensive examination of data within the AGILE RTA framework.

### Key Features of BBlocks:

- **Encapsulation of Input and Output**: Simplifies data management by grouping related information together.
- **Plotting Methods**: Includes built-in methods for visualizing the results of the Bayesian Blocks analysis, making it easier to interpret the change points and other significant findings.

## Usage

The primary interface to these routines is the `bayesian_blocks` function. The module supports various data types through different fitness functions:

- **Irregularly-Spaced Event Data**: Handled by the `Events` class.
- **Regularly-Spaced Event Data**: Handled by the `RegularEvents` class.
- **Irregularly-Spaced Point Measurements**: Handled by the `PointMeasures` class.

For advanced users, it is possible to define custom fitness functions by subclassing `FitnessFunc` and passing them to the `bayesian_blocks` function.

### Common Applications

- **Adaptive-Width Histograms**: Determine optimal histogram bins using the Bayesian Blocks algorithm.
- **Time Series Segmentation**: Automatically detect significant changes in event rates within time series data.

## References

This implementation is based on the following works:

1. [Scargle et al., 2013](https://ui.adsabs.harvard.edu/abs/2013ApJ...764..167S)
2. [AstroML Project](https://www.astroml.org/) - [GitHub](https://github.com/astroML/astroML/)
3. Bellman, R.E., Dreyfus, S.E., 1962. *Applied Dynamic Programming*. Princeton University Press, Princeton. [Link](https://press.princeton.edu/books/hardcover/9780691651873/applied-dynamic-programming)
4. Bellman, R., Roth, R., 1969. *Curve fitting by segmented straight lines*. J. Amer. Statist. Assoc. 64, 1079–1084. [Link](https://www.tandfonline.com/doi/abs/10.1080/01621459.1969.10501038)
5. [Astropy Bayesian Blocks Implementation](https://docs.astropy.org/en/latest/_modules/astropy/stats/bayesian_blocks.html)

## License

This code is distributed under the same license as Astropy. For more details, refer to the Astropy documentation.
