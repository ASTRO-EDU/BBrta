import pandas as pd
import numpy as np
from astropy.time import Time
from bayesian_blocks import BBlocks
import os
from base_bblocks import BaseBBlocks 


class AGILE_BBlocks(BaseBBlocks):
    """
    A class to handle and analyze AGILE Bayesian Blocks data, derived from BaseBBlocks.

    This class adds functionality specific to AGILE data, including selecting events
    and processing time formats unique to AGILE.
    """
    
    def __init__(self, detections_csv_path: str = None):
        """
        Initialize the AGILE_BBlocks class by loading detections from a CSV file.

        Parameters:
        -----------
        detections_csv_path : str
            The path to the CSV file containing detection data time ranges.
        """
        super().__init__(detections_csv_path)
    
    def select_event(self, mle_path=None, ap_path:str=None, ph_path:str=None, rate_path:str=None, event_id: str=None, tstart=None, tstop=None, rate=False, ratefactor=0):
        """
        Select an event by specifying its ID and data paths.

        Parameters:
        -----------
        ap_path : str
            The path to the binned light curve data file. 
        ph_path : str
            The path to the TTE data file.
        event_id : str
            The ID of the event to select (default is None).
        tstart 
            Time start in MJD. This is used if event_id is None
        tstop 
            Time stop in MJD. This is used if event_id is Non
        rate:
            For ap_path. Enable the evaluation of the rate multiplied for a ratefactor and converted into integer. If ratefactor == 0 use the mean of the exposure
        ratefactor: scale a rate to int. Usualy set it as the a mean effective area * dt
            For ap_path: A scale factor of the cts/exp values to obtain an integer after conversion. Typical value 1e7
            For rate_path: ratefactor=0 scale to int with a mean rate scaling, ratefactor=-1 show row data, scalefactor > 1 give your own scale factor
        """
        if ap_path is None and ph_path is None and rate_path is None and mle_path is None:
            raise ValueError("`ap_path` and `ph_path` and `rate_path` and mle_path must not be both None!")
        # Determine the data mode based on the provided paths.
        if not ap_path is None:   # If a path to binned light curve data is provided 'lwtime', 'uptime', 'exposure', 'counts'
            self.filemode = 2
            if rate == True:
                self.filemode = 3
        elif not mle_path is None:
            self.filemode = 6
        elif not ph_path is None: # If a path to TTE data is provided
            self.filemode = 4
        elif not rate_path is None:
            self.filemode = 5 #'lwtime', 'uptime', 'rate'
        else:
            raise ValueError("Other `datamode` different from 2 or 3 are not supported!")
        
        # Store the event ID.
        self.event_id = event_id
        # Extract the start and stop times for the selected event.
        if event_id is not None:
            estart = self.df_detections.loc[self.event_id]["mjd_start"]
            estop = self.df_detections.loc[self.event_id]["mjd_stop"]
        else:
            estart = tstart
            estop = tstop

        self.data_cells = None
        self.rate = None
        self.cts = None
        self.exp = None

        if self.filemode == 2 or self.filemode == 3:
            print(f"Binned light AGILE AP curve selected {self.filemode}...")
            # Load the binned light curve data from the CSV file.
            df_ap = pd.read_csv(ap_path, delim_whitespace=True, header=None)
            df_ap.columns = ['lwtime', 'uptime', 'exposure', 'counts']
            #convert in MJD
            df_ap['lwtime'] = df_ap['lwtime'].apply(self.__tt_to_mjd)
            df_ap['uptime'] = df_ap['uptime'].apply(self.__tt_to_mjd)
            # Filter the data to only include rows within the event time range.
            self.df_event = df_ap[(df_ap["lwtime"] >= estart) & (df_ap["uptime"] <= estop)]
            # Exclude rows with zero exposure.
            self.df_event = self.df_event[self.df_event["exposure"] != 0]
            print(f"Total number of rows for the event {self.event_id} (no zero-exposure): {len(self.df_event)}")
            print(f"Total number of photons for the event {self.event_id} (no zero-exposure): {self.df_event['counts'].sum()}")
            # Extract the relevant columns for Bayesian blocks processing.
            t_i = self.df_event['lwtime'].to_numpy()  # Lower bound time
            t_f = self.df_event['uptime'].to_numpy()  # Upper bound time
            #calculate custum data cells
            self.data_cells = np.append(t_i, t_f[-1])
            # Calculate the midpoint time and time delta.
            self.t_delta = (t_f - t_i)
            self.dt = min(self.t_delta)
            self.t_c = t_i + self.t_delta/2
            self.exp = self.df_event['exposure'].to_numpy()
            self.cts = self.df_event['counts'].to_numpy()
            self.rate = self.cts / self.exp
            if self.filemode == 2:
                self.x = self.cts
                self.sigma = np.sqrt(self.x)  # Standard deviation as square root of counts
            if self.filemode == 3:
                #print((cts / self.exp))
                if ratefactor == 0:
                    expmean = self.exp.mean()
                    self.x = ((self.cts / self.exp) * expmean).astype(int)
                else:
                    self.x = ((self.cts / self.exp) * ratefactor).astype(int)
                self.sigma = np.sqrt(self.x)  # Standard deviation as square root of counts
            self.datamode = 2 #lc
        elif self.filemode == 6:
            print(f"Binned light AGILE MLE curve selected {self.filemode}...")
            # Load the binned light curve data from the CSV file.
            df_ap = pd.read_csv(mle_path, delim_whitespace=True)
            df_ap = df_ap.sort_values(by='time_start_tt')
            # Filter the data to only include rows within the event time range.
            df_ap['lwtime'] = df_ap['time_start_tt'].apply(self.__tt_to_mjd)
            df_ap['uptime'] = df_ap['time_end_tt'].apply(self.__tt_to_mjd)
            self.df_event = df_ap[(df_ap["lwtime"] >= estart) & (df_ap["uptime"] <= estop)]
            # Exclude rows with zero exposure.
            self.df_event = self.df_event[self.df_event["exposure"] != 0]
            self.df_event = self.df_event[self.df_event["sqrt(ts)"] > 3]
            print(f"Total number of rows for the event {self.event_id} (no zero-exposure): {len(self.df_event)}")
            print(f"Total number of photons for the event {self.event_id} (no zero-exposure): {self.df_event['counts'].sum()}")
            # Extract the relevant columns for Bayesian blocks processing.
            t_i = self.df_event['lwtime'].to_numpy()  # Lower bound time
            t_f = self.df_event['uptime'].to_numpy()  # Upper bound time
            #calculate custum data cells
            self.data_cells = np.append(t_i, t_f[-1])
            # Calculate the midpoint time and time delta.
            self.t_delta = (t_f - t_i)
            self.dt = min(self.t_delta)
            self.t_c = t_i + self.t_delta/2
            self.exp = self.df_event['exposure'].to_numpy()
            self.cts = self.df_event['counts'].to_numpy().astype(int)
            self.rate = self.df_event['flux'].to_numpy()
            self.x = self.cts
            self.sigma = np.sqrt(self.x)  # Standard deviation as square root of counts
            self.datamode = 2 #lc
        elif self.filemode == 4:
            print("TTE...")
            self.exp = None
            # Load the TTE data from the CSV file.
            df_tte = pd.read_csv(ph_path, delim_whitespace=True, header=None)
            # Assign column names.
            df_tte.columns = ['time', 'l', 'b', '_', '_c', '_d', '_e', '_f', '_g']
            # Filter the data to only include rows within the event time range.
            df_tte['time'] = df_tte['time'].apply(self.__tt_to_mjd)
            self.df_event = df_tte[(df_tte['time'] >= estart) & (df_tte['time'] <= estop)]
            print("Number of photons in this event is:", len(self.df_event))
            # Set time-related variables for Bayesian blocks processing.
            self.data_cells = None
            self.dt = 0
            self.t_delta = None
            self.t_c = self.df_event['time'].to_numpy()  # Event times
            self.x = np.ones_like(self.t_c)  # Set counts to 1 for each event
            self.sigma = np.zeros_like(self.t_c)  # Set sigma to 0
            self.datamode = 1 #tte
        elif self.filemode == 5:
            print("Binned light AGILE RATE light curve selected...")
            # Load the binned rate light curve data from the CSV file.
            df_ap = pd.read_csv(rate_path, delim_whitespace=True, header=None)
            df_ap.columns = ['lwtime', 'uptime', 'rate']
            # Filter the data to only include rows within the event time range.
            self.df_event = df_ap[(df_ap["lwtime"] >= estart) & (df_ap["uptime"] <= estop)]
            # Extract the relevant columns for Bayesian blocks processing.
            t_i = self.df_event['lwtime'].to_numpy()  # Lower bound time
            t_f = self.df_event['uptime'].to_numpy()  # Upper bound time
            #calculate custum data cells
            self.data_cells = np.append(t_i, t_f[-1])
            self.exp = None
            # Calculate the midpoint time and time delta.
            self.t_delta = (t_f - t_i)
            self.dt = min(self.t_delta)
            self.t_c = t_i + self.t_delta/2
            self.x = self.df_event['rate'].to_numpy()
            if ratefactor == 0:
                xmean = self.x.mean() / 5.0
                print(xmean)
                self.x = (self.x / xmean).astype(int)
                print(self.x)
                self.sigma = np.sqrt(self.x)  # Standard deviation as square root of rate
            elif ratefactor == -1:
                self.x = self.x
                self.sigma = np.zeros_like(self.x)
                print(self.sigma)
            else:
                self.x = (self.x  * ratefactor).astype(int)
                self.sigma = np.sqrt(self.x)  # Standard deviation as square root of rate
            self.datamode = 2 #lc
        else:
            raise ValueError("No filemode")
        
        # Initialize the BBlocks object and set the data for Bayesian blocks analysis.
        self.bblocks = BBlocks()
        self.bblocks.set_argsIn(x=self.x, t=self.t_c, sigma=self.sigma, dt=self.dt, datamode=self.datamode, t_delta=self.t_delta, cts = self.cts, exp=self.exp, data_cells=self.data_cells, rate=self.rate)
            

    def __mjd_to_tt(self, mjd):
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


    def __tt_to_mjd(self, tt_seconds):
        """
        Convert Terrestrial Time (TT) seconds to Modified Julian Date (MJD).

        Parameters:
        - tt_seconds (float): Number of seconds elapsed since a reference epoch in Terrestrial Time (TT).

        Returns:
        - mjd (float): The resulting Modified Julian Date (MJD).
        """


        time_unix = np.array(tt_seconds) + 1072915200
        t = Time(time_unix, format="unix")
        return t.mjd
