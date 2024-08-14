import pandas as pd
import numpy as np
import os
from astropy.time import Time

from bayesian_blocks import BBlocks, bayesian_blocks

class AGILE_BBlocks:
    """
    A class to handle and analyze AGILE Bayesian Blocks data.

    Attributes:
    -----------
    datamode : int
        Indicates the mode of data, either binned light curve (2) or TTE (3).
    df_detections : DataFrame
        A DataFrame storing detection information, such as start and stop times.
    event_id: str
        The ID of the selected event.
    df_event : DataFrame
        A DataFrame containing data for the selected event.
    res_blocks: BBlocks
        An instance of the BBlocks class used to manage and process Bayesian blocks.
    """
    
    def __init__(self, detections_csv_path:str=None):
        """
        Initialize the AGILE_BBlocks class by loading detections from a CSV file.

        Parameters:
        -----------
        detections_csv_path : str
            The path to the CSV file containing detection data time ranges.
        """
        if detections_csv_path is None:
            raise ValueError("`detections_csv_path` must not be None!")
        # Load the detections data from the provided CSV file.
        self.df_detections = pd.read_csv(detections_csv_path)
        # Set the index of the DataFrame to be the flare_id.
        self.df_detections.index = self.df_detections["flare_id"]
        # Keep only the start and stop times (in MJD format).
        self.df_detections = self.df_detections[["mjd_start", "mjd_stop"]]
        # Convert the MJD times to Unix time relative to the AGILE epoch.
        self.df_detections['mjd_start'] = self.df_detections['mjd_start'].apply(self.__mjd_to_unix)
        self.df_detections['mjd_stop'] = self.df_detections['mjd_stop'].apply(self.__mjd_to_unix)
    

    def select_event(self, ap_path:str=None, ph_path:str=None, event_id: str='E05'):
        """
        Select an event by specifying its ID and data paths.

        Parameters:
        -----------
        ap_path : str
            The path to the binned light curve data file.
        ph_path : str
            The path to the TTE data file.
        event_id : str
            The ID of the event to select (default is 'E05').
        """
        if ap_path is None and ph_path is None:
            raise ValueError("`ap_path` and `ph_path` must not be both None!")
        # Determine the data mode based on the provided paths.
        if not ap_path is None:   # If a path to binned light curve data is provided
            self.datamode = 2
        elif not ph_path is None: # If a path to TTE data is provided
            self.datamode = 3
        else:
            raise ValueError("Other `datamode` different from 2 or 3 are not supported!")
        
        # Store the event ID.
        self.event_id = event_id
        # Extract the start and stop times for the selected event.
        estart = self.df_detections.loc[self.event_id]["mjd_start"]
        estop = self.df_detections.loc[self.event_id]["mjd_stop"]
        
        if self.datamode == 2:
            print("Binned light curve selected...")
            # Load the binned light curve data from the CSV file.
            df_ap = pd.read_csv(ap_path, delim_whitespace=True, header=None)
            df_ap.columns = ['lwtime', 'uptime', 'exposure', 'counts']
            # Filter the data to only include rows within the event time range.
            self.df_event = df_ap[(df_ap["lwtime"] >= estart) & (df_ap["uptime"] <= estop)]
            # Exclude rows with zero exposure.
            self.df_event = self.df_event[self.df_event["exposure"] != 0]
            print(f"Total number of rows for the event {self.event_id} (no zero-exposure): {len(self.df_event)}")
            print(f"Total number of photons for the event {self.event_id} (no zero-exposure): {self.df_event['counts'].sum()}")
            # Extract the relevant columns for Bayesian blocks processing.
            t_i = self.df_event['lwtime'].to_numpy()  # Lower bound time
            t_f = self.df_event['uptime'].to_numpy()  # Upper bound time
            # Calculate the midpoint time and time delta.
            self.t_delta = (t_f - t_i)/2
            self.t_c = t_i + self.t_delta
            self.x = self.df_event['counts'].to_numpy()
            self.sigma = np.sqrt(self.x)  # Standard deviation as square root of counts
        elif self.datamode == 3:
            print("TTE...")
            # Load the TTE data from the CSV file.
            df_tte = pd.read_csv(ph_path, delim_whitespace=True, header=None)
            # Assign column names.
            df_tte.columns = ['time', 'l', 'b', '_', '_c', '_d', '_e', '_f', '_g']
            # Filter the data to only include rows within the event time range.
            self.df_event = df_tte[(df_tte['time'] >= estart) & (df_tte['time'] <= estop)]
            print("Number of photons in this event is:", len(self.df_event))
            # Set time-related variables for Bayesian blocks processing.
            self.t_delta = 0
            self.t_c = self.df_event['time'].to_numpy()  # Event times
            self.x = np.ones_like(self.t_c)  # Set counts to 1 for each event
            self.sigma = np.zeros_like(self.t_c)  # Set sigma to 0
        else:
            raise ValueError("Ciao")
        
        # Initialize the BBlocks object and set the data for Bayesian blocks analysis.
        self.resbblocks = BBlocks()
        self.resbblocks.set_data(self.x, self.t_c, None, self.t_delta)
        

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
    
    def plot_data(self):
        """
        Plot the Bayesian blocks data based on the data mode.
        """
        # Plot the data using the BBlocks object based on the data mode.
        if self.datamode == 2:
            self.resbblocks.plot_blocks(t_delta=True, edge_points=False, 
                                        data_cells=False, mean_blocks=False)
        elif self.datamode == 3:
            self.resbblocks.plot_blocks(t_delta=False, edge_points=False, 
                                        data_cells=False, mean_blocks=False)

    def bayesian_blocks(self, fitness='events', p0=None, gamma=None):
        """
        Compute the Bayesian blocks using the given parameters and plot the result.

        Parameters:
        -----------
        fitness : str
            The fitness function to use ('events' by default).
        p0 : float
            Prior on the number of blocks (optional).
        gamma : float
            Regularization parameter (optional).
        """
        # Compute the Bayesian blocks with the given parameters and plot the result.
        self.resbblocks = bayesian_blocks(self.t_c, self.x, sigma=self.sigma, fitness=fitness, p0=p0, gamma=gamma)
        if self.datamode == 2:
            self.resbblocks.plot_blocks(t_delta=True, edge_points=True, data_cells=True, mean_blocks=True)
        elif self.datamode == 3:
            self.resbblocks.plot_blocks(t_delta=False, edge_points=False, data_cells=True, mean_blocks=False)

    def plot_bblocks(self):
        """
        Plot the Bayesian blocks with the appropriate time delta setting based on data mode.
        """
        # Plot the Bayesian blocks with appropriate time delta setting based on data mode.
        if self.datamode == 2:
            self.resbblocks.plot_blocks(t_delta=True, edge_points=True, data_cells=True, mean_blocks=True)
        elif self.datamode == 3:
            self.resbblocks.plot_blocks(t_delta=False, edge_points=False, data_cells=True, mean_blocks=False)
    
    def plot_blocks_with_rate(self):
        """
        Plot the results of the Bayesian Blocks analysis, including both the light curve with 
        block means and a rate vector showing changes over time.
        """
        # Plot the data using the BBlocks object based on the data mode.
        if self.datamode == 2:
            self.resbblocks.plot_blocks_with_rate(t_delta=True, edge_points=True, data_cells=True, mean_blocks=True)
        elif self.datamode == 3:
            self.resbblocks.plot_blocks_with_rate(t_delta=False, edge_points=True, data_cells=True, mean_blocks=True)

    def __mjd_to_unix(self, mjd):
        """
        Convert Modified Julian Date (MJD) to Unix time relative to the AGILE epoch.

        Parameters:
        -----------
        mjd : float
            The Modified Julian Date to convert.
        
        Returns:
        --------
        float
            The Unix time relative to the AGILE epoch.
        """
        # Convert Modified Julian Date (MJD) to Unix time relative to the AGILE epoch.
        mjd_date = Time(mjd, format='mjd', scale='utc')
        # Define the AGILE epoch date.
        agile_epoch = Time('2004-01-01T00:00:00', scale='utc')
        # Return the difference in seconds from the AGILE epoch.
        return mjd_date.unix - agile_epoch.unix 
