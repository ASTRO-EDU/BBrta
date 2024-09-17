# This implementation of `bayesian_blocks` has been adapted to fit the 
# context of AGILE RTA (Real-Time Analysis). Specifically, we have modified 
# the final extraction of `change_points` to align with the AGILE RTA 
# requirements by removing the first and last excess blocks, which are not 
# present in MATLAB.

# The backbone of this code was derived from the Bayesian Blocks implementation 
# in Astropy, which can be found at the following link:
# https://docs.astropy.org/en/latest/_modules/astropy/stats/bayesian_blocks.html

# Additionally, we have introduced a new class to encapsulate the results 
# of the Bayesian Blocks analysis. This class facilitates further analysis 
# by offering methods to plot the change points and other relevant metrics, 
# streamlining the workflow and enabling a more comprehensive examination 
# of the data within the AGILE RTA framework.

#datamode=1 -> class event, unbinned data: t -> for each t, x is set to 1 - Fitness function eq 19 - Statistics: Bernoulli
#datamode=2 -> class event, binned data: t, x -> x contains an integer number (the value of the bins) - Fitness function eq 19 - statistics Poisson
#datamode=2 -> class RegularEvent, binned data: t, x -> x can be only 0 or 1 (no duplicated time teg) - Fitness function Eq. C23 
#datamode=3 -> class PointMeasures - Fitness function eq. 41 from Scargle 2013 - statistics Gaussian

#NB: for class event, if p0 is passed, the ncp_prior is computed as function p0_prior (see eq. 21 in Scargle (2013)), for all cases and datamodes. Be carefull, this should be applicable only for datamode=1 (check in the paper)

"""Bayesian Blocks for Time Series Analysis.

Bayesian Blocks for Time Series Analysis
========================================

Dynamic programming algorithm for solving a piecewise-constant model for
various datasets. This is based on the algorithm presented in Scargle
et al 2013 [1]_. This code was ported from the astroML project [2]_, and 
the backbone of this implementation was derived from the Bayesian Blocks 
implementation in Astropy [5]_.

Applications include:

- finding an optimal histogram with adaptive bin widths
- finding optimal segmentation of time series data
- detecting inflection points in the rate of event data

The primary interface to these routines is the :func:`bayesian_blocks`
function. This module provides fitness functions suitable for three types
of data:

- Irregularly-spaced event data via the :class:`Events` class
- Regularly-spaced event data via the :class:`RegularEvents` class
- Irregularly-spaced point measurements via the :class:`PointMeasures` class

For more fine-tuned control over the fitness functions used, it is possible
to define custom :class:`FitnessFunc` classes directly and use them with
the :func:`bayesian_blocks` routine.

One common application of the Bayesian Blocks algorithm is the determination
of optimal adaptive-width histogram bins. This uses the same fitness function
as for irregularly-spaced time series events. The easiest interface for
creating Bayesian Blocks histograms is the :func:`astropy.stats.histogram`
function.

References
----------
.. [1] https://ui.adsabs.harvard.edu/abs/2013ApJ...764..167S
.. [2] https://www.astroml.org/ https://github.com//astroML/astroML/
.. [3] Bellman, R.E., Dreyfus, S.E., 1962. Applied Dynamic
   Programming. Princeton University Press, Princeton.
   https://press.princeton.edu/books/hardcover/9780691651873/applied-dynamic-programming
.. [4] Bellman, R., Roth, R., 1969. Curve fitting by segmented
   straight lines. J. Amer. Statist. Assoc. 64, 1079–1084.
   https://www.tandfonline.com/doi/abs/10.1080/01621459.1969.10501038
.. [5] https://docs.astropy.org/en/latest/_modules/astropy/stats/bayesian_blocks.html
"""


import warnings
from inspect import signature
import json

import numpy as np
import matplotlib.pyplot as plt

from astropy.utils.exceptions import AstropyUserWarning

from bayesian_blocks_utils import BBlocksUtils

#####################################################################################################

class BBlocks:
    def __init__(self):
        """
        Initialize two dictionaries to store input and output data for the Bayesian Blocks analysis.
        
        Attributes:
        -----------
        data_in : dict
            Dictionary to store input parameters like the data series and prior information.
        data_out : dict
            Dictionary to store the results of the Bayesian Blocks algorithm, such as the 
            identified change points and segment statistics.
        """
        self.data_in = {}
        self.data_out = {}
        
    def set_argsIn(self, **kwargs):
        """
        Update the 'data_in' dictionary with the provided keyword arguments.

        This method allows you to dynamically add or update input parameters 
        for the Bayesian Blocks algorithm. The keyword arguments are passed 
        in as key-value pairs, which are then directly added to the 'data_in' 
        dictionary. This is a flexible way to manage input data without having 
        to specify each individual parameter explicitly.
        """
        self.data_in.update(kwargs)
        
    def set_result(self, **kwargs):
        """
        Update the 'data_out' dictionary with the provided keyword arguments.

        This method allows you to store or update the output results of the 
        Bayesian Blocks algorithm, such as change points and segment statistics. 
        The keyword arguments are passed in as key-value pairs and are added 
        directly to the 'data_out' dictionary.

        """
        self.data_out.update(kwargs)

    # Get methods
    def get_data_in(self):
        """
        Retrieve the input data dictionary 'data_in'.
        
        Returns:
        --------
        dict
            The 'data_in' dictionary containing the input parameters such as the data series, 
            prior probabilities, fitness function, and other related input data for the 
            Bayesian Blocks analysis.
        """
        return self.data_in
    
    def get_data_out(self):
        """
        Retrieve the output data dictionary 'data_out'.
        
        Returns:
        --------
        dict
            The 'data_out' dictionary containing the results of the Bayesian Blocks algorithm, 
            such as the identified change points, edge points, mean values of blocks, 
            and other derived statistics or processed data.
        """
        return self.data_out

    # JSON methods
    def load_from_json(self, file_path):
        """
        Load 'data_in' and 'data_out' from a JSON file and handle special numeric values.
        
        Parameters:
        -----------
        file_path : str
            The path to the JSON file to load.
        """
        try:
            #####
            def convert_from_serializable(obj):
                """Convert serialized values back to their original form (e.g., 'Infinity' -> np.inf)."""
                if isinstance(obj, str):
                    if obj == "inf":
                        return np.inf
                    elif obj == "-inf":
                        return -np.inf
                    elif obj == "nan":
                        return np.nan
                elif isinstance(obj, list):
                    return np.array([convert_from_serializable(i) for i in obj])
                elif isinstance(obj, dict):
                    return {k: convert_from_serializable(v) for k, v in obj.items()}
                return obj
            #####
            with open(file_path, 'r') as f:
                data = json.load(f)
            # Convert special values (like 'Infinity' or 'NaN') back to their numerical form
            if 'data_in' in data:
                self.data_in.update(convert_from_serializable(data['data_in']))
            if 'data_out' in data:
                self.data_out.update(convert_from_serializable(data['data_out']))
            self.data_out.update(self.__compute_statistics(
                x=self.data_in["x"], t=self.data_in["t"], 
                change_points=self.data_out["change_points"], N=self.data_out["N"], 
                data_cells=self.data_out["data_cells"], rate=self.data_in["rate"]))
            print("Data successfully loaded from JSON file.")
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from file '{file_path}'.")

    def save_to_json(self, file_path):
        """
        Save 'data_in' and 'data_out' to a JSON file, converting NumPy arrays, NumPy types, and infinite values.

        Parameters:
        -----------
        file_path : str
            The path where the JSON file will be saved.
        """
        try:
            # Convert any NumPy arrays, NumPy-specific types, and handle infinite values
            def convert_to_serializable(obj):
                if isinstance(obj, float) and (np.isinf(obj) or np.isnan(obj)):  # Handle infinities and NaN
                    return str(obj)  # Converts to "Infinity", "-Infinity", or "NaN"
                elif isinstance(obj, np.number):  # Handle NumPy types
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, np.ndarray)):
                    return [convert_to_serializable(i) for i in obj]
                return obj

            data = {
                'data_in': convert_to_serializable(self.data_in),
                'data_out': convert_to_serializable(self.data_out)
            }

            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)

            print(f"Data successfully saved to '{file_path}'.")
        except IOError:
            print(f"Error: Could not write to file '{file_path}'.")

    # Plot methods
    def plot_blocks(self, t_delta=True, y_err=True, edge_points=True, data_cells=True, mean_blocks=True, sum_blocks=False, alg='custom'):
        """
        Plot the results of the Bayesian Blocks analysis, including the light curve, 
        detected blocks, and optionally the mean value of each block.
        
        Parameters:
        -----------
        t_delta : bool, optional
            If True, includes error bars based on time resolution (dt) and Poisson uncertainties 
            in the plot. Default is True.
        edge_points : bool, optional
            If True, plots the positions of the edge points between blocks. Default is True.
        data_cells : bool, optional
            If True, plots the positions of data cells or segments identified by the algorithm. 
            Default is True.
        mean_blocks : bool, optional
            If True, plots the mean value within each identified block. Default is True.
        sum_blocks : bool, optional
            If True, plots the sum within each identified block. Default is False.
        alg : str, optional
            The algorithm source, which determines the color of the plot lines. Supported options 
            are 'custom', 'astropy', and 'matlab'. Default is 'custom'.
        
        Raises:
        -------
        Exception:
            If an unsupported value is provided for 'alg'.
        """
        
        # A small constant to avoid division by zero or other numerical issues
        eps = 1e+5
        
        # Set the color of the edge points based on the algorithm type
        if alg == 'custom': 
            color = 'green'
        elif alg == 'astropy': 
            color = 'orange'
        elif alg == 'matlab': 
            color = 'red'
        else: 
            raise Exception(f"Not supported argument for alg: {alg}")
        
        # Determine error bars: Use dt for time and sqrt of counts for Poisson error if t_delta is True
        if t_delta:
            try:
                xerr = self.data_in['t_delta'] / 2.0
            except:
                xerr = np.full_like(self.data_in['x'], self.data_in['dt'] / 2.0)

            if y_err:
                #yerr = np.sqrt(self.data_in['x'])
                yerr = self.data_in['sigma']
            else:
                yerr = np.zeros_like(self.data_in['x'])
        else:
            xerr = np.zeros_like(self.data_in['x'])
            yerr = np.zeros_like(self.data_in['x'])
              

        
        # Create the figure for plotting
        plt.figure(figsize=(10, 6))
        
        # Plot the edge points if specified
        if edge_points:
            plt.vlines(self.data_out['edge_points'], 0, max(self.data_in['x'] + yerr), 
                       label='Edge blocks', color=color, ls='--', 
                       linewidth=1.5)
        
        # Plot the data cells if specified
        if data_cells:
            plt.vlines(self.data_out['data_cells'], 0, max(self.data_in['x'] + yerr), 
                       label='Data cell', color='gray', ls='--', 
                       linewidth=0.5)
        
        # Plot the actual data with error bars
        plt.errorbar(self.data_in['t'], self.data_in['x'], 
                     xerr=xerr, yerr=yerr, fmt="o", color='tab:blue', 
                     label='light curve')
        
        # Plot the mean values of the blocks if specified
        if mean_blocks:
            means = []
            i_edge = 0
            # Calculate mean for each block
            for t in self.data_out['data_cells']:
                means.append(self.data_out['mean_blocks'][i_edge])
                if i_edge < len(self.data_out['edge_points']) and t >= self.data_out['edge_points'][i_edge]:
                    i_edge += 1
            plt.step(self.data_out['data_cells'], means, color='purple', label='blocks mean')

        # Plot the mean values of the blocks if specified
        if sum_blocks:
            sumb = []
            i_edge = 0
            # Calculate sum for each block
            for t in self.data_out['data_cells']:
                sumb.append(self.data_out['sum_blocks'][i_edge])
                if i_edge < len(self.data_out['edge_points']) and t >= self.data_out['edge_points'][i_edge]:
                    i_edge += 1
            plt.step(self.data_out['data_cells'], sumb, color='red', label='blocks sum')
        
        # Label the axes
        plt.xlabel('Time')
        plt.ylabel('Count')
        
        # Add a legend to the plot
        plt.legend()
        
        # Rotate the x-axis labels for better readability
        #plt.xticks(self.data_in['t'], self.data_in['t'], rotation=90)
        from matplotlib.ticker import MaxNLocator
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=False, prune='both', nbins=14))
        plt.xticks(rotation=90)
        plt.tight_layout()

        # Set the title of the plot
        plt.title(f'Bayesian Blocks Analysis on Time Series ({alg})')
        # Uncomment plt.show() if running outside of a script that automatically renders plots
        # plt.show()
           
    def plot_blocks_with_rate(self, t_delta=True, y_err=True, edge_points=True, data_cells=True, mean_blocks=True, sum_blocks=False, alg='custom'):
        """
        Plot the results of the Bayesian Blocks analysis, including both the light curve with 
        block means and a rate vector showing changes over time.
        
        Parameters:
        -----------
        t_delta : bool, optional
            If True, includes error bars based on time resolution (dt) and Poisson uncertainties 
            in the plot. Default is True.
        edge_points : bool, optional
            If True, plots the positions of the edge points between blocks. Default is True.
        data_cells : bool, optional
            If True, plots the positions of data cells or segments identified by the algorithm. 
            Default is True.
        mean_blocks : bool, optional
            If True, plots the mean value within each identified block. Default is True.
        sum_blocks : bool, optional
            If True, plots the sum within each identified block. Default is True.
        alg : str, optional
            The algorithm source, which determines the color of the plot lines. Supported options 
            are 'custom', 'astropy', and 'matlab'. Default is 'custom'.
        
        Raises:
        -------
        Exception:
            If an unsupported value is provided for 'alg'.
        """
        
        # A small constant to avoid division by zero or other numerical issues
        eps = 1e+5
        
        # Set the color of the edge points based on the algorithm type
        if alg == 'custom': 
            color = 'green'
        elif alg == 'astropy': 
            color = 'orange'
        elif alg == 'matlab': 
            color = 'red'
        else: 
            raise Exception(f"Not supported argument for alg: {alg}")

        # Determine error bars: Use dt for time and sqrt of counts for Poisson error if t_delta is True
        if t_delta:
            try:
                xerr = self.data_in['t_delta'] / 2.0
            except:
                xerr = np.full_like(self.data_in['x'], self.data_in['dt'] / 2.0)
            if y_err:
                #yerr = np.sqrt(self.data_in['x'])
                yerr = self.data_in['sigma']
            else:
                yerr = np.zeros_like(self.data_in['x'])
        else:
            xerr = np.zeros_like(self.data_in['x'])
            yerr = np.zeros_like(self.data_in['x'])


        # Create the figure and two subplots
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))

        # First subplot: Light curve with block means
        if edge_points:
            axs[0].vlines(self.data_out['edge_points'], 0, max(self.data_in['x'] + yerr), 
                          label='Edge blocks', color=color, ls='--', 
                          linewidth=1.5)
        if data_cells:
            axs[0].vlines(self.data_out['data_cells'], 0, max(self.data_in['x'] + yerr), 
                          label='Data cell', color='gray', ls='--', 
                          linewidth=0.5)

        if mean_blocks or sum_blocks:
            means = []
            sumb = []
            i_edge = 0
            # Calculate mean and rate for each block
            for t in self.data_out['data_cells']:
                means.append(self.data_out['mean_blocks'][i_edge])
                sumb.append(self.data_out['sum_blocks'][i_edge])
                
                if i_edge < len(self.data_out['edge_points']) and t >= self.data_out['edge_points'][i_edge]:
                    i_edge += 1
                    
        if mean_blocks:
            axs[0].step(self.data_out['data_cells'], means, color='purple', label='blocks mean')
        if sum_blocks:
            axs[0].step(self.data_out['data_cells'], sumb, color='red', label='blocks sum')


        axs[0].errorbar(self.data_in['t'], self.data_in['x'], 
                        xerr=xerr, yerr=yerr, fmt="o", color='tab:blue', 
                        label='light curve')
                        
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Count')
        axs[0].legend()
        axs[0].set_xticks(self.data_in['t'])
        #axs[0].set_xticklabels(self.data_in['t'], rotation=90)
        axs[0].set_title(f'Bayesian Blocks Analysis on Time Series ({alg})')

        # Rotate the x-axis labels for better readability
        #plt.xticks(self.data_in['t'], self.data_in['t'], rotation=90)
        from matplotlib.ticker import MaxNLocator
        axs[0].xaxis.set_major_locator(MaxNLocator(integer=False, prune='both', nbins=20))
        axs[0].tick_params(axis='x', rotation=90)


        # Second subplot: Rate vector over time

        rates = []
        rateblocks = []
        i_edge = 0
        # Calculate mean and rate for each block
        for t in self.data_out['data_cells']:
            rate_this = self.data_out['eventrate_vec'][i_edge]
            rates.append(rate_this if rate_this != np.inf else 0)
            rate_this = self.data_out['blockrate_vec'][i_edge]
            rateblocks.append(rate_this if rate_this != np.inf else 0)
            
            if i_edge < len(self.data_out['edge_points']) and t >= self.data_out['edge_points'][i_edge]:
                i_edge += 1   

        if edge_points:
            axs[1].vlines(self.data_out['edge_points'], 0, max(rateblocks), 
                          label='Edge blocks', color=color, ls='--', 
                          linewidth=1.5)
        if data_cells:
            axs[1].vlines(self.data_out['data_cells'], 0, max(rateblocks), 
                          label='Data cell', color='gray', ls='--', 
                          linewidth=0.5)
        #axs[1].step(self.data_out['data_cells'], rates, color='tab:orange', label='eventrate', ls='--')
        #axs[1].step(self.data_out['data_cells'], rateblocks, color='tab:blue', label='blockrate_vec')
        axs[1].step(self.data_out['data_cells'], rateblocks, color='tab:blue', label='blockrate_vec')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Rate')
        axs[1].legend()
        axs[1].set_xticks(self.data_in['t'])
        #axs[1].set_xticklabels(self.data_in['t'], rotation=90)
        axs[1].set_title('Rate Vector over Time')

        axs[1].xaxis.set_major_locator(MaxNLocator(integer=False, prune='both', nbins=20))
        axs[1].tick_params(axis='x', rotation=90)

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()

    def prepare_input(self, t, x=None, sigma=None, input_data_cells=None, rate=None):
        #TODOAB this and the input routines must be merged
        #TODOAB sort also sigma, t_delta, input edges
        """Validate inputs to the model.

        Parameters
        ----------
        t : array-like
            times of observations
        x : array-like, optional
            values observed at each time
        sigma : float or array-like, optional
            errors in values x
        input_data_cells: TODO:
        rate: TODO:

        Returns
        -------
        t, x, sigma : array-like, float or None
            validated and perhaps modified versions of inputs
        """
        # validate array input
        t = np.asarray(t, dtype=float)

        # find unique values of t
        t = np.array(t)
        if t.ndim != 1:
            raise ValueError("t must be a one-dimensional array")
        unq_t, unq_ind, unq_inv = np.unique(t, return_index=True, return_inverse=True)

        # if x is not specified, x will be counts at each time
        if x is None:
            if sigma is not None:
                raise ValueError("If sigma is specified, x must be specified")
            else:
                sigma = 1

            if len(unq_t) == len(t):
                x = np.ones_like(t)
            else:
                x = np.bincount(unq_inv)
            dt = 0
            t = unq_t

        # if x is specified, then we need to simultaneously sort t and x
        else:
            # TODO: allow broadcasted x?
            x = np.asarray(x, dtype=float)

            if x.shape not in [(), (1,), (t.size,)]:
                raise ValueError("x does not match shape of t")
            x += np.zeros_like(t)

            if len(unq_t) != len(t):
                raise ValueError(
                    "Repeated values in t not supported when x is specified"
                )
            dt = min((t[1:]-t[:-1])/2)
            t = unq_t
            x = x[unq_ind]
            # TODO: input_datacells
            if rate is not None:
                rate = rate[unq_ind]

        # verify the given sigma value
        if sigma is None:
            sigma = 1
        else:
            sigma = np.asarray(sigma, dtype=float)
            if sigma.shape not in [(), (1,), (t.size,)]:
                raise ValueError("sigma does not match the shape of x")
        
        # Store in results data 
        self.set_argsIn(x=x, t=t, sigma=sigma, dt=dt)
        
        return t, x, sigma, input_data_cells, rate

    def bayesian_blocks(self, t, x=None, sigma=None, input_data_cells=None, rate=None, fitness="events", **kwargs):
        """Compute optimal segmentation of data with Scargle's Bayesian Blocks.

        This is a flexible implementation of the Bayesian Blocks algorithm
        described in Scargle 2013 [1]_.

        Parameters
        ----------
        t : array-like
            data times (one dimensional, length N)
        x : array-like, optional
            data values
        sigma : array-like or float, optional
            data errors
        fitness : str or object
            the fitness function to use for the model.
            If a string, the following options are supported:

            - 'events' : binned or unbinned event data.  Arguments are ``gamma``,
            which gives the slope of the prior on the number of bins, or
            ``ncp_prior``, which is :math:`-\ln({\tt gamma})`.
            - 'regular_events' : non-overlapping events measured at multiples of a
            fundamental tick rate, ``dt``, which must be specified as an
            additional argument.  Extra arguments are ``p0``, which gives the
            false alarm probability to compute the prior, or ``gamma``, which
            gives the slope of the prior on the number of bins, or ``ncp_prior``,
            which is :math:`-\ln({\tt gamma})`.
            - 'measures' : fitness for a measured sequence with Gaussian errors.
            Extra arguments are ``p0``, which gives the false alarm probability
            to compute the prior, or ``gamma``, which gives the slope of the
            prior on the number of bins, or ``ncp_prior``, which is
            :math:`-\ln({\tt gamma})`.

            In all three cases, if more than one of ``p0``, ``gamma``, and
            ``ncp_prior`` is chosen, ``ncp_prior`` takes precedence over ``gamma``
            which takes precedence over ``p0``.

            Alternatively, the fitness parameter can be an instance of
            :class:`FitnessFunc` or a subclass thereof.

        **kwargs :
            any additional keyword arguments will be passed to the specified
            :class:`FitnessFunc` derived class.

        Returns
        -------
        edges : ndarray
            array containing the (N+1) edges defining the N bins

        Examples
        --------
        .. testsetup::

            >>> np.random.seed(12345)

        Event data:

        >>> t = np.random.normal(size=100)
        >>> edges = bayesian_blocks(t, fitness='events', p0=0.01)

        Event data with repeats:

        >>> t = np.random.normal(size=100)
        >>> t[80:] = t[:20]
        >>> edges = bayesian_blocks(t, fitness='events', p0=0.01)

        Regular event data:

        >>> dt = 0.05
        >>> t = dt * np.arange(1000)
        >>> x = np.zeros(len(t))
        >>> x[np.random.randint(0, len(t), len(t) // 10)] = 1
        >>> edges = bayesian_blocks(t, x, fitness='regular_events', dt=dt)

        Measured point data with errors:

        >>> t = 100 * np.random.random(100)
        >>> x = np.exp(-0.5 * (t - 50) ** 2)
        >>> sigma = 0.1
        >>> x_obs = np.random.normal(x, sigma)
        >>> edges = bayesian_blocks(t, x_obs, sigma, fitness='measures')

        References
        ----------
        .. [1] Scargle, J et al. (2013)
        https://ui.adsabs.harvard.edu/abs/2013ApJ...764..167S

        .. [2] Bellman, R.E., Dreyfus, S.E., 1962. Applied Dynamic
        Programming. Princeton University Press, Princeton.
        https://press.princeton.edu/books/hardcover/9780691651873/applied-dynamic-programming

        .. [3] Bellman, R., Roth, R., 1969. Curve fitting by segmented
        straight lines. J. Amer. Statist. Assoc. 64, 1079–1084.
        https://www.tandfonline.com/doi/abs/10.1080/01621459.1969.10501038

        See Also
        --------
        astropy.stats.histogram : compute a histogram using bayesian blocks
        """
        FITNESS_DICT = {
            "events": Events,
            "regular_events": RegularEvents,
            "measures": PointMeasures,
        }
        fitness = FITNESS_DICT.get(fitness, fitness)

        if type(fitness) is type and issubclass(fitness, FitnessFunc):
            fitfunc = fitness(**kwargs)
        elif isinstance(fitness, FitnessFunc):
            fitfunc = fitness
        else:
            raise ValueError("fitness parameter not understood")
        
        t_valid, x_valid, sigma_valid, input_data_cells_valid, rate_valid = self.prepare_input(t, x, sigma, input_data_cells, rate)
        self.set_argsIn(t=t_valid, x=x_valid, sigma=sigma_valid, 
                        input_data_cells=input_data_cells, rate=rate_valid,
                        **kwargs)
        res = fitfunc.fit(t_valid, x_valid, sigma_valid, input_data_cells_valid, rate_valid)
        self.set_result(**res)
        # ----------------------------------------------------------------
        # Now compute the height of each block and rate_vector
        # ----------------------------------------------------------------
        self.data_out.update(
            self.__compute_statistics(
                x=self.data_in["x"], t=self.data_in["t"], 
                change_points=self.data_out["change_points"], N=self.data_out["N"], 
                data_cells=self.data_out["data_cells"]
            )
        )
        return res
        
    def __compute_statistics(self, **kwargs):
        # Get the data
        x = kwargs["x"]
        t = kwargs["t"]
        change_points = kwargs["change_points"]
        N = kwargs["N"]
        data_cells = kwargs["data_cells"] 
        if "rate" in kwargs.keys():
            rate = kwargs["rate"]
        else:
            rate = None
        # Compute the number of changepoints and blocks
        num_changepoints = len(change_points)
        num_blocks = num_changepoints + 1
        # Define the statistics 
        blockrate_vec    = np.zeros( num_blocks )
        blockrate_vec_vec    = np.zeros( num_blocks )
        eventrate_vec    = np.zeros( num_blocks )
        mean_blocks = np.zeros( num_blocks )
        sum_blocks  = np.zeros( num_blocks )
        num_vec     = np.zeros( num_blocks )
        dt_event_vec      = np.zeros( num_blocks )
        dt_block_vec      = np.zeros( num_blocks )
        tt_1_vec    = np.zeros( num_blocks )
        tt_2_vec    = np.zeros( num_blocks )
        
        cpt_use = np.zeros(num_changepoints+2, dtype=np.int32)
        cpt_use[1:-1] = change_points
        cpt_use[-1] = N

        for i in range(num_blocks):
            ii_1 = cpt_use[i] # start
            ii_2 = cpt_use[i + 1]
            # Get the data 
            block_vec = x[ii_1:ii_2].copy()
            tt_vec = t[ii_1:ii_2].copy()
            if rate is not None:
                rate_vec = rate[ii_1:ii_2].copy()
                blockrate_vec_vec[i] = rate_vec.mean()
            
            edge_vec = data_cells[ii_1:ii_2+1].copy()
            # Compute mean_blocks
            mean_blocks[i] = block_vec.mean()
            # Compute sum_blocks
            sum_blocks[i] = block_vec.sum()
            # Compute dt_eventl_vec, i.e. the difference between first and last time of data into blocks
            dt_event_vec[i] = (tt_vec[-1]-tt_vec[0])
            # Compute dt_vec of blocks
            dt_block_vec[i] = (edge_vec[-1]-edge_vec[0])
            # Compute rate_vec of blocks based on size of blocks
            blockrate_vec[i] = sum_blocks[i]/dt_block_vec[i]
            # Compute rate_vec of blocks based on difference between first and last event of data
            eventrate_vec[i] = sum_blocks[i]/dt_event_vec[i]
        # Return the result dictionary 
        return {
            "edge_vec":      edge_vec,
            "sum_blocks":    sum_blocks,
            "mean_blocks":   mean_blocks,
            "dt_event_vec":  dt_event_vec,
            "dt_block_vec":  dt_block_vec,
            "blockrate_vec": blockrate_vec,
            "eventrate_vec": eventrate_vec
        }
#####################################################################################################

# TODO: implement other fitness functions from appendix C of Scargle 2013

__all__ = ["FitnessFunc", "Events", "RegularEvents", "PointMeasures"]



class FitnessFunc:
    """Base class for bayesian blocks fitness functions.

    Derived classes should overload the following method:

    ``fitness(self, **kwargs)``:
      Compute the fitness given a set of named arguments.
      Arguments accepted by fitness must be among ``[T_k, N_k, a_k, b_k, c_k]``
      (See [1]_ for details on the meaning of these parameters).

    Additionally, other methods may be overloaded as well:

    ``__init__(self, **kwargs)``:
      Initialize the fitness function with any parameters beyond the normal
      ``p0`` and ``gamma``.

    ``validate_input(self, t, x, sigma)``:
      Enable specific checks of the input data (``t``, ``x``, ``sigma``)
      to be performed prior to the fit.

    ``compute_ncp_prior(self, N)``: If ``ncp_prior`` is not defined explicitly,
      this function is called in order to define it before fitting. This may be
      calculated from ``gamma``, ``p0``, or whatever method you choose.

    ``p0_prior(self, N)``:
      Specify the form of the prior given the false-alarm probability ``p0``
      (See [1]_ for details).

    For examples of implemented fitness functions, see :class:`Events`,
    :class:`RegularEvents`, and :class:`PointMeasures`.

    References
    ----------
    .. [1] Scargle, J et al. (2013)
       https://ui.adsabs.harvard.edu/abs/2013ApJ...764..167S
    """
    def __init__(self, p0=0.05, gamma=None, ncp_prior=None):
        self.p0 = p0
        self.gamma = gamma
        self.ncp_prior = ncp_prior
        self.data_out = {}
        # self.bblocks.set_argsIn(p0=p0, gamma=gamma)
        self.verbose = False
 
    def fitness(self, **kwargs):
        raise NotImplementedError()
 
    def p0_prior(self, N):
        """Empirical prior, parametrized by the false alarm probability ``p0``.

        See eq. 21 in Scargle (2013).

        Note that there was an error in this equation in the original Scargle
        paper (the "log" was missing). The following corrected form is taken
        from https://arxiv.org/abs/1304.2818
        """
        return 4 - np.log(73.53 * self.p0 * (N**-0.478))

    # The fitness_args property will return the list of arguments accepted by
    # the method fitness().  This allows more efficient computation below.
    @property
    def _fitness_args(self):
        return signature(self.fitness).parameters.keys()
 
    def compute_ncp_prior(self, N):
        """
        If ``ncp_prior`` is not explicitly defined, compute it from ``gamma``
        or ``p0``.
        """
        
        if self.gamma is not None:
            return -np.log(self.gamma)
        elif self.p0 is not None:
            return self.p0_prior(N)
        else:
            raise ValueError(
                "``ncp_prior`` cannot be computed as neither "
                "``gamma`` nor ``p0`` is defined."
            )

    def validate_input(self, t, x, sigma):
        raise NotImplementedError()
        
    def fit(self, t, x=None, sigma=None, input_data_cells=None, rate=None) -> BBlocks:
        """Fit the Bayesian Blocks model given the specified fitness function.

        Parameters
        ----------
        t : array-like
            data times (one dimensional, length N)
        x : array-like, optional
            data values
        sigma : array-like or float, optional
            data errors

        Returns
        -------
        edges : ndarray
            array containing the (M+1) edges defining the M optimal bins
        """
        t, x, sigma = self.validate_input(t, x, sigma)
        # compute values needed for computation, below
        if "a_k" in self._fitness_args:
            ak_raw = np.ones_like(x) / sigma**2
        if "b_k" in self._fitness_args:
            bk_raw = x / sigma**2
        if "c_k" in self._fitness_args:
            ck_raw = x * x / sigma**2
        # Create length-(N + 1) array of cell edges
        edges = np.concatenate([t[:1], 
                                0.5 * (t[1:] + t[:-1]), 
                                t[-1:]])
        if input_data_cells is not None:
            # custum data cells
            edges = input_data_cells
        # Store data_cells
        data_cells = edges
        self.data_out["data_cells"] = data_cells
        # Define blocks lenght
        block_length = t[-1] - edges
        if self.verbose == True:
            print(block_length)
        # arrays to store the best configuration
        N = len(t)
        best = np.zeros(N, dtype=float)
        last = np.zeros(N, dtype=int)
        # Compute ncp_prior if not defined
        if self.ncp_prior is None:
            ncp_prior = self.compute_ncp_prior(N)
        else:
            ncp_prior = self.ncp_prior
        self.data_out["ncp_prior"] = ncp_prior
        self.data_out["N"] = N
        # ----------------------------------------------------------------
        # Start with first data cell; add one cell at each iteration
        # ----------------------------------------------------------------
        for R in range(N):
            if self.verbose == True:
                print("### R", R, block_length[: (R + 1)], block_length[R + 1])
            # Compute fit_vec : fitness of putative last block (end at R)
            kwds = {}
            # T_k: width/duration of each block
            if "T_k" in self._fitness_args:
                kwds["T_k"] = block_length[: (R + 1)] - block_length[R + 1]
            # N_k: number of elements in each block
            if "N_k" in self._fitness_args:
                kwds["N_k"] = np.cumsum(x[: (R + 1)][::-1])[::-1]
            # a_k: eq. 31
            if "a_k" in self._fitness_args:
                kwds["a_k"] = 0.5 * np.cumsum(ak_raw[: (R + 1)][::-1])[::-1]
            # b_k: eq. 32
            if "b_k" in self._fitness_args:
                kwds["b_k"] = -np.cumsum(bk_raw[: (R + 1)][::-1])[::-1]
            # c_k: eq. 33
            if "c_k" in self._fitness_args:
                kwds["c_k"] = 0.5 * np.cumsum(ck_raw[: (R + 1)][::-1])[::-1]
            # evaluate fitness function
            fit_vec = self.fitness(**kwds)
            # Compute fitness blocks with dynamic programming 
            A_R = fit_vec - ncp_prior
            A_R[1:] += best[:R]
            # NOTE: When you have log(0)-log(0) the result will be a Not A
            # Number (NaN). If you want to have a MATLAB-like behavior in 
            # Python to implement Bayesian blocks, you can use the numpy 
            # nanmax function to evaluate the argmax of the fitness vector. 
            # However, if the array contains only NaNs, nanmax will raise a
            # ValueError exception. In this case you can use the classic 
            # numpy argmax function.
            try:
                i_max = np.nanargmax(A_R)
            except ValueError:
                i_max = np.argmax(A_R)
            last[R] = i_max
            best[R] = A_R[i_max]
            if self.verbose == True:
                print("A_R ",A_R)
                print("best ", best)
                print("last ", last)
                print("i_max ", i_max)
        # ----------------------------------------------------------------
        # Now find changepoints by iteratively peeling off the last block
        # ----------------------------------------------------------------
        if self.verbose == True:
            print("CHANGE POINTS")
        change_points = np.zeros(N, dtype=int)
        # NOTE: Replaced the final block of the function to follow the
        # behavior of the MATLAB implementation, avoiding to return also  
        # the first and last edge among the results
        index = last[N-1]
        change_points = []
        while index > 0:
            change_points.insert(0, index)
            if self.verbose == True:
                print(index, last[index - 1])
            index = last[index - 1]
        # Store egde_points and change_points
        self.data_out["change_points"] = change_points
        self.data_out["edge_points"] = edges[change_points]
        self.data_out["N"] = N
        
        return self.data_out



class Events(FitnessFunc):
    r"""Bayesian blocks fitness for binned or unbinned events.

    Parameters
    ----------
    p0 : float, optional
        False alarm probability, used to compute the prior on
        :math:`N_{\rm blocks}` (see eq. 21 of Scargle 2013). For the Events
        type data, ``p0`` does not seem to be an accurate representation of the
        actual false alarm probability. If you are using this fitness function
        for a triggering type condition, it is recommended that you run
        statistical trials on signal-free noise to determine an appropriate
        value of ``gamma`` or ``ncp_prior`` to use for a desired false alarm
        rate.
    gamma : float, optional
        If specified, then use this gamma to compute the general prior form,
        :math:`p \sim {\tt gamma}^{N_{\rm blocks}}`.  If gamma is specified, p0
        is ignored.
    ncp_prior : float, optional
        If specified, use the value of ``ncp_prior`` to compute the prior as
        above, using the definition :math:`{\tt ncp\_prior} = -\ln({\tt
        gamma})`.
        If ``ncp_prior`` is specified, ``gamma`` and ``p0`` is ignored.
    """
    
    def __init__(self, p0=0.05, gamma=None, ncp_prior=None):
        super().__init__(p0, gamma, ncp_prior)
 
    def fitness(self, N_k, T_k):
        # eq. 19 from Scargle 2013
        # log_args = N_k / T_k
        # log_args[log_args <= 0] = 1e-3
        # return N_k * (np.log(log_args))
        log_args_Tk = T_k.copy()
        log_args_Tk[T_k <= 0] = np.inf
        return N_k * (np.log(N_k) - np.log(T_k))
 
    def validate_input(self, t, x, sigma):
        if x is not None and np.any(x % 1 > 0):
            raise ValueError("x must be integer counts for fitness='events'")
        return t, x, sigma



class RegularEvents(FitnessFunc):
    r"""Bayesian blocks fitness for regular events.

    This is for data which has a fundamental "tick" length, so that all
    measured values are multiples of this tick length.  In each tick, there
    are either zero or one counts.

    Parameters
    ----------
    dt : float
        tick rate for data
    p0 : float, optional
        False alarm probability, used to compute the prior on :math:`N_{\rm
        blocks}` (see eq. 21 of Scargle 2013). If gamma is specified, p0 is
        ignored.
    ncp_prior : float, optional
        If specified, use the value of ``ncp_prior`` to compute the prior as
        above, using the definition :math:`{\tt ncp\_prior} = -\ln({\tt
        gamma})`.  If ``ncp_prior`` is specified, ``gamma`` and ``p0`` are
        ignored.
    """
    def __init__(self, dt, p0=0.05, gamma=None, ncp_prior=None):
        self.dt = dt
        super().__init__(p0, gamma, ncp_prior)
 
    def validate_input(self, t, x, sigma):
        if not np.all((x == 0) | (x == 1)):
            raise ValueError("Regular events must have only 0 and 1 in x")
        return t, x, sigma
 
    def fitness(self, T_k, N_k):
        # Eq. C23 of Scargle 2013
        M_k = T_k / self.dt
        N_over_M = N_k / M_k
        eps = 1e-8
        if np.any(N_over_M > 1 + eps):
            warnings.warn(
                "regular events: N/M > 1.  Is the time step correct?",
                AstropyUserWarning,
            )

        one_m_NM = 1 - N_over_M
        N_over_M[N_over_M <= 0] = 1
        one_m_NM[one_m_NM <= 0] = 1

        return N_k * np.log(N_over_M) + (M_k - N_k) * np.log(one_m_NM)



class PointMeasures(FitnessFunc):
    r"""Bayesian blocks fitness for point measures.

    Parameters
    ----------
    p0 : float, optional
        False alarm probability, used to compute the prior on :math:`N_{\rm
        blocks}` (see eq. 21 of Scargle 2013). If gamma is specified, p0 is
        ignored.
    ncp_prior : float, optional
        If specified, use the value of ``ncp_prior`` to compute the prior as
        above, using the definition :math:`{\tt ncp\_prior} = -\ln({\tt
        gamma})`.  If ``ncp_prior`` is specified, ``gamma`` and ``p0`` are
        ignored.
    """

    def __init__(self, p0=0.05, gamma=None, ncp_prior=None):
        super().__init__(p0, gamma, ncp_prior)
 
    def fitness(self, a_k, b_k):
        # eq. 41 from Scargle 2013
        return (b_k * b_k) / (4 * a_k)
 
    def validate_input(self, t, x, sigma):
        if x is None:
            raise ValueError("x must be specified for point measures")
        return t, x, sigma