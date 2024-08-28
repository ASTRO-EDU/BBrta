import pandas as pd
import numpy as np
from astropy.time import Time
from bayesian_blocks import BBlocks, bayesian_blocks
import os
from base_bblocks import BaseBBlocks 
from datetime import datetime, timedelta

class FERMI_BBlocks(BaseBBlocks):
    """
    A class to handle and analyze FERMI Bayesian Blocks data, derived from BaseBBlocks.

    This class adds functionality specific to FERMI data, including selecting events
    and processing time formats unique to FERMI.
    """
    
    def __init__(self, detections_csv_path: str = None):
        """
        Initialize the FERMI_BBlocks class by loading detections from a CSV file.

        Parameters:
        -----------
        detections_csv_path : str
            The path to the CSV file containing detection data time ranges.
        """
        super().__init__(detections_csv_path)
    
    def select_event(self, ap_path:str=None, mle_path:str=None, event_id: str=None, tstart=None, tstop=None, rate=False, ratefactor=0):
        """
        Select an event by specifying its ID and data paths.

        Parameters:
        -----------
        ap_path : str
            The path to the AP binned light curve data file. 
        mle_path : str
            The path to the MLE binned light curve data file.
        event_id : str
            The ID of the event to select (default is None).
        tstart 
            Time start in MJD. This is used if event_id is None
        tstop 
            Time stop in MJD. This is used if event_id is None
        rate:
            For ap_path. Enable the evaluation of the rate multiplied for a ratefactor and converted into integer. If ratefactor == 0 use the mean of the exposure
        ratefactor: scale a rate to int. Usualy set it as the a mean effective area * dt
            For ap_path: A scale factor of the cts/exp values to obtain an integer after conversion. Typical value 1e7
            For rate_path: ratefactor=0 scale to int with a mean rate scaling, ratefactor=-1 show row data, scalefactor > 1 give your own scale factor
        """
        if ap_path is None and mle_path is None:
            raise ValueError("`ap_path` and `mle_path` must not be both None!")
        # Determine the data mode based on the provided paths.
        if not ap_path is None:   # If a path to binned light curve data is provided 'lwtime', 'uptime', 'exposure', 'counts'
            self.filemode = 2
            if rate == True:
                self.filemode = 3
        elif not mle_path is None: 
            self.filemode = 4
        
        # Store the event ID.
        self.event_id = event_id
        # Extract the start and stop times for the selected event.
        if event_id is not None:
            estart = self.df_detections.loc[self.event_id]["mjd_start"]
            estop = self.df_detections.loc[self.event_id]["mjd_stop"]
        else:
            estart = tstart
            estop = tstop

        if self.filemode == 2 or self.filemode == 3:
            print("Binned light FERMI AP curve selected...")
            #aperture.pl modified with additional counts column
            #$time $rate $rerr $timedel $prerr $exposure $counts
            #Note time is in MET

            # Load the binned light curve data from the CSV file.
            df_ap = pd.read_csv(ap_path, delim_whitespace=True, header=None)
            df_ap.columns = ['time', 'rate', 'rerr', 'timedel', 'prerr', 'exposure', 'counts']
            # Apply the convert function to each element in the 'time' column
            df_ap['time'] = df_ap['time'].apply(self.fermi_seconds_to_mjd)
            # Filter the data to only include rows within the event time range.
            self.df_event = df_ap[(df_ap["time"] >= estart) & (df_ap["time"] <= estop)]
            # Exclude rows with zero exposure.
            self.df_event = self.df_event[self.df_event["exposure"] != 0]
            print(f"Total number of rows (no zero-exposure): {len(self.df_event)}")
            print(f"Total number of photons (no zero-exposure): {self.df_event['counts'].sum()}")
            self.t_c = self.df_event['time'].to_numpy()
            self.exp = self.df_event['exposure'].to_numpy()
            self.t_delta = 0
            if self.filemode == 2:
                self.x = self.df_event['counts'].to_numpy()
                self.sigma = np.sqrt(self.x)  # Standard deviation as square root of counts
            if self.filemode == 3:
                cts = self.df_event['counts'].to_numpy()
                print((cts / self.exp))
                if ratefactor == 0:
                    expmean = self.exp.mean()
                    self.x = ((cts / self.exp) * expmean).astype(int)
                else:
                    self.x = ((cts / self.exp) * ratefactor).astype(int)
                self.sigma = np.sqrt(self.x)  # Standard deviation as square root of counts
            self.datamode = 2 #lc
        elif self.filemode == 4:
            print("Binned light FERMI RATE light curve selected...")
            #
        else:
            raise ValueError("No filemode")
        
        # Initialize the BBlocks object and set the data for Bayesian blocks analysis.
        self.resbblocks = BBlocks()
        if self.filemode != 2:
            self.resbblocks.set_data(self.x, self.t_c, self.sigma, self.t_delta)
        if self.filemode == 2:
            self.resbblocks.set_data(self.x, self.t_c, self.sigma, self.t_delta, exp=self.exp)
            
    def fermi_seconds_to_mjd(self, fermi_seconds):
        """
        Converti i secondi Fermi dal 2001.0 UTC in Modified Julian Date (MJD), tenendo conto dei leap seconds.

        Parametri:
        - fermi_seconds (float): Numero di secondi trascorsi dal 1 gennaio 2001 00:00:00 UTC.

        Ritorna:
        - mjd (float): La Modified Julian Date (MJD) risultante.
        """

        # Definire le date dei leap seconds dal 2001
        leap_seconds_dates = [
            datetime(2006, 1, 1),
            datetime(2009, 1, 1),
            datetime(2012, 7, 1),
            datetime(2015, 7, 1),
            datetime(2017, 1, 1)
        ]

        # Data di riferimento: 1 gennaio 2001
        reference_date = datetime(2001, 1, 1)

        # MJD di riferimento per il 1 gennaio 2001
        reference_mjd = 51910

        # Calcola la data corrispondente ai secondi Fermi dal 2001
        target_date = reference_date + timedelta(seconds=fermi_seconds)

        # Conta il numero di leap seconds fino alla data target
        leap_seconds_count = sum(1 for leap_date in leap_seconds_dates if target_date >= leap_date)

        # Secondi di Fermi meno i leap seconds
        adjusted_fermi_seconds = fermi_seconds - leap_seconds_count

        # Converti i secondi Fermi in giorni
        days_since_2001 = adjusted_fermi_seconds / 86400  # 86400 secondi in un giorno

        # Calcola il MJD
        mjd = reference_mjd + days_since_2001

        return mjd

