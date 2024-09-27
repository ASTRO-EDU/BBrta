import pandas as pd
import numpy as np
from astropy.time import Time
from bayesian_blocks import BBlocks
import os

class BaseBBlocks:
    """
    A base class to handle and analyze Bayesian Blocks data.

    Attributes:
    -----------
    df_detections : DataFrame
        A DataFrame storing detection information, such as start and stop times.
    event_id: str
        The ID of the selected event.
    df_event : DataFrame
        A DataFrame containing data for the selected event.
    res_blocks: BBlocks
        An instance of the BBlocks class used to manage and process Bayesian blocks.
    """
    
    def __init__(self, detections_csv_path: str = None):
        """
        Initialize the BaseBBlocks class by loading detections from a CSV file.

        Parameters:
        -----------
        detections_csv_path : str
            The path to the CSV file containing detection data time ranges.
        """
        if detections_csv_path is not None:
            # Load the detections data from the provided CSV file.
            self.df_detections = pd.read_csv(detections_csv_path)
            # Set the index of the DataFrame to be the flare_id.
            self.df_detections.index = self.df_detections["flare_id"]
            # Keep only the start and stop times (in MJD format).
            self.df_detections = self.df_detections[["mjd_start", "mjd_stop"]]
            # Convert the MJD times to TT time relative to the AGILE epoch.
            #self.df_detections['mjd_start'] = self.df_detections['mjd_start'].apply(self.__mjd_to_tt)
            #self.df_detections['mjd_stop'] = self.df_detections['mjd_stop'].apply(self.__mjd_to_tt)

    def select_event(self, *args, **kwargs):
        """
        Abstract method to select an event. Should be implemented in the derived class.

        Raises:
        -------
        NotImplementedError
        """
        raise NotImplementedError("`select_event` must be implemented in the derived class.")

    def head_detections(self, n:int = 5):
        """
        Return the first `n` rows of the detections DataFrame.

        Parameters:
        -----------
        n : int
            The number of rows to return (default is 5).
        
        Returns:
        --------
        DataFrame
            The first `n` rows of the detections DataFrame.
        """
        # Return the first `n` rows of the detections DataFrame.
        return self.df_detections.head(n)
    
    def head_event(self, n:int = 5):
        """
        Return the first `n` rows of the event DataFrame.

        Parameters:
        -----------
        n : int
            The number of rows to return (default is 5).
        
        Returns:
        --------
        DataFrame
            The first `n` rows of the event DataFrame.
        """
        # Return the first `n` rows of the event DataFrame.
        return self.df_event.head(n)
    
    def plot_data(self, yerr=True):
        """
        Plot the Bayesian blocks data based on the data mode.
        """
        # Plot the data using the BBlocks object based on the data mode.
        if self.datamode == 2:
            self.bblocks.plot_blocks(t_delta=True, y_err=yerr, edge_points=False, 
                                        data_cells=False, mean_blocks=False)
        elif self.datamode == 1:
            self.bblocks.plot_blocks(t_delta=False, y_err=yerr, edge_points=False, 
                                        data_cells=False, mean_blocks=False)

    def bayesian_blocks(self, fitness='events', p0=None, gamma=None, useerror=False):
        """
        Compute the Bayesian blocks using the given parameters and plot the result.

        Parameters:
        -----------
        fitness : str
            The fitness function to use ('events' by default).
        p0 : float
            Prior on the number of blocks (optional). Calculated with eq. 21 in Scargle (2013)
        gamma : float
            Regularization parameter (optional).
        useerror : Bool
            Flag for using error for computing blocks or not.
        """
        # Compute the Bayesian blocks with the given parameters and plot the result.
        sigma = self.sigma if useerror else None
        self.bblocks = BBlocks()
        self.bblocks.bayesian_blocks(self.t_c, self.x, sigma=sigma, fitness=fitness, input_data_cells=self.data_cells, rate=self.rate, p0=p0, gamma=gamma)
        if self.datamode == 2:
            self.bblocks.plot_blocks(t_delta=True, edge_points=True, data_cells=True, mean_blocks=True)
        elif self.datamode == 1:
            self.bblocks.plot_blocks(t_delta=False, edge_points=True, data_cells=True, mean_blocks=False)

    def plot_bblocks(self):
        """
        Plot the Bayesian blocks with the appropriate time delta setting based on data mode.
        """
        # Plot the Bayesian blocks with appropriate time delta setting based on data mode.
        if self.datamode == 2:
            self.bblocks.plot_blocks(t_delta=True, edge_points=True, data_cells=True, mean_blocks=True)
        elif self.datamode == 1:
            self.bblocks.plot_blocks(t_delta=False, edge_points=True, data_cells=True, mean_blocks=False)
    
    def plot_blocks_with_rate(self):
        """
        Plot the results of the Bayesian Blocks analysis, including both the light curve with 
        block means and a rate vector showing changes over time.
        """
        # Plot the data using the BBlocks object based on the data mode.
        if self.datamode == 2:
            self.bblocks.plot_blocks_with_rate(t_delta=True, edge_points=True, data_cells=True, mean_blocks=True, sum_blocks=False)
        elif self.datamode == 1:
            self.bblocks.plot_blocks_with_rate(t_delta=False, edge_points=True, data_cells=True, mean_blocks=False, sum_blocks=True)
    
    def get_data_out(self):
        return self.bblocks.get_data_out() if self.bblocks else None
    
    def get_data_in(self):
        return self.bblocks.get_data_in() if self.bblocks else None
    
