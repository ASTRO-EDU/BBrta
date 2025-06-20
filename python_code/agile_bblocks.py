import pandas as pd
import numpy as np
from astropy.time import Time
from bayesian_blocks import BBlocks
from enum import Enum
import os
from base_bblocks import BaseBBlocks 


class DataMode(Enum):
    UNBINNED = 1
    BINNED   = 2
    
class FileMode(Enum):
    AGILE_PH  = 1
    AGILE_AP  = 2
    AGILE_MLE = 3
    CUSTOM_LC = 4

class AGILE_BBlocks(BaseBBlocks):
    """
    A class to handle and analyze AGILE Bayesian Blocks data, derived from BaseBBlocks.

    This class adds functionality specific to AGILE data, including selecting events
    and processing time formats unique to AGILE.
    """
    
    def __init__(self):
        """
        Initialize the AGILE_BBlocks class by loading events from a text file.
        """
        super().__init__()
        self.df_event = None   # Data Frame with selected data
        self.data_cells = None # Time Edges of the data
        self.t_c = None        # Reference Time of the Events: Flux Points (binned) or Photons(unbinned)
        self.t_delta = None    # Time Errors
        self.dt = None         # Minimum time error
        self.cts = None        # Counts Array
        self.exp = None        # Exposure Array
        self.rate = None       # "Rate" array (flux) = cts/exp corrected by ratecorrection.        
        self.x = None          # Value or the Events (Flux or 1 for Photons)
        self.sigma = None      # Errors on the Value of the Events
    
    
    @property
    def datamode(self):
        return self._datamode.value
    
    @property
    def filemode(self):
        return self._filemode.value
    
    def _parse_file_mode(self, mode):
        """
        Convert a input string or int into a FileMode Enum.
        
        Parameters
        ----------
        mode : int, str or FileMode
            Define the FileMode
            
        Return
        ------
        mode : FileMode
            Filemode as Enum object.
        
        Raises
        ------
        ValueError : if provided a value that cannot be converted into a FileMode.
        TypeError : if provided a type different than int, str or FileMode.
        """
        if isinstance(mode, FileMode):
            return mode
        elif isinstance(mode, int):
            try:
                return FileMode(mode)
            except ValueError:
                raise ValueError(f"Invalid mode value: {mode}. Must be 1 (AGILE_PH), 2 (AGILE_AP), 3 (AGILE_MLE) or 4 (CUSTOM_LC).")
        elif isinstance(mode, str):
            try:
                return FileMode[mode.upper()]
            except KeyError:
                raise ValueError(f"Invalid mode string: '{mode}'. Must be \'AGILE_PH\', \'AGILE_AP\', \'AGILE_MLE\' or \'CUSTOM_LC\'.")
        else:
            raise TypeError(f"Mode must be a FileMode, int, or str, not {type(mode).__name__}")
    
    
    def _read_agile_ap(self, file_path, tstart_mjd, tstop_mjd, ratecorrection):
        """
        Read the input file, provided in AGILE_AP format.
        
        Parameters
        ----------
        file_path : str
            The path to the data file. 
        tstart_mjd, tstop_mjd : float 
            Time start, stop in MJD to select events. If None, no selection is applied.
        ratecorrection: None or 0 or float
            Multiplication factor to convert rates (cts/exp) into int used with AGILE_AP data.
            - Set to None to not use.
            - Set to -1 to apply BB on the counts instead of flux points (e.g. for Ratemeter data).
            - Set to 0 to use the mean exposure, e.g. to apply BB on flux points scaled to int.
            - Set to a float value to multiply the rates (cts/exp) accordingly.
        """
        print(f"Read input file {file_path}")
        # Load the binned light curve data from the CSV file.
        df_in = pd.read_csv(file_path, delim_whitespace=True, header=None)
        df_in.columns = ['lwtime_tt', 'uptime_tt', 'exposure', 'counts']
        
        # Convert in MJD
        df_in['lwtime_mjd'] = df_in['lwtime_tt'].apply(self.__tt_to_mjd)
        df_in['uptime_mjd'] = df_in['uptime_tt'].apply(self.__tt_to_mjd)
        
        # Filter the data to only include rows within the event time range.
        if tstart_mjd is not None:
            df_in = df_in[df_in["lwtime_mjd"] >= tstart_mjd]
        if tstop_mjd is not None:
            df_in = df_in[df_in["uptime_mjd"] <= tstop_mjd]
        
        # Filter rows with zero exposure and set sef.df_event
        self.df_event = df_in[df_in["exposure"] > 0]
        print(f"Time Range selected (MJD): {self.df_event['lwtime_mjd'].min()}-{self.df_event['uptime_mjd'].max()}")
        print(f"Total number of rows selected (no zero-exposure): {len(self.df_event)}")
        print(f"Total number of counts selected (no zero-exposure): {self.df_event['counts'].sum()}")
        
        # Extract the relevant columns for Bayesian blocks processing.
        t_i = self.df_event['lwtime_tt'].to_numpy() # Lower bound time
        t_f = self.df_event['uptime_tt'].to_numpy() # Upper bound time
        
        # Calculate custum data cells (Time edges)
        self.data_cells = np.append(t_i, t_f[-1])
        
        # Calculate the midpoint time and time delta.
        self.t_delta = (t_f - t_i)
        self.dt = min(self.t_delta)
        self.t_c = t_i + self.t_delta/2
        
        # Compute Flux
        self.exp = self.df_event['exposure'].to_numpy()
        self.cts = self.df_event['counts'  ].to_numpy().astype(int)
        self.rate= self.cts / self.exp
        
        # Apply Flux Correction, set x
        if ratecorrection is None:
            print(f"Apply Bayesian Blocks on AP FLUX = CTS / EXP with no scaling")
            self.x = self.rate
        elif ratecorrection==-1:
            print(f"Apply Bayesian Blocks on CTS (integers)")
            self.x = self.cts
        elif ratecorrection==0:
            expmean = self.exp.mean()
            print(f"Apply Bayesian Blocks on AP FLUX (= CTS/EXP) scaled to integers with a mean exposure factor={expmean:.3g}")
            self.x = (self.rate * expmean).astype(int)
        else:
            print(f"Apply Bayesian Blocks on AP FLUX (= CTS/EXP) scaled to integers with a correction factor={ratecorrection:.3g}")
            self.x = (self.rate * ratecorrection).astype(int)
        
        # Set Error using Poisson Errors: sqrt of self.x (already scaled)
        self.sigma = np.sqrt(self.x)
        
        return None
    
    
    def _read_agile_mle(self, file_path, tstart_mjd, tstop_mjd, ratecorrection):
        """
        Read the input file, provided in AGILE_MLE format.
        
        Parameters
        ----------
        file_path : str
            The path to the data file. 
        tstart_mjd, tstop_mjd : float 
            Time start, stop in MJD to select events. If None, no selection is applied.
        ratecorrection: None or 0 or float
            Multiplication factor to convert rates (cts/exp) into int used with AGILE_AP data.
            - Set to None to not use.
            - Set to -1 to apply BB on the counts instead of flux points (e.g. for Ratemeter data).
            - Set to 0 to use the mean exposure, e.g. to apply BB on flux points scaled to int.
            - Set to a float value to multiply the rates (cts/exp) accordingly.
        """
        print(f"Read input file {file_path}")
        # Load the binned light curve data from the file.
        df_in = pd.read_csv(file_path, delim_whitespace=True)
        df_in = df_in.sort_values(by='time_start_tt')
        
        # Convert in MJD
        df_in['lwtime_mjd'] = df_in['time_start_tt'].apply(self.__tt_to_mjd)
        df_in['uptime_mjd'] = df_in['time_end_tt'  ].apply(self.__tt_to_mjd)
        
        # Filter the data to only include rows within the event time range.
        if tstart_mjd is not None:
            df_in = df_in[df_in["lwtime_mjd"] >= tstart_mjd]
        if tstop_mjd is not None:
            df_in = df_in[df_in["uptime_mjd"] <= tstop_mjd]
        
        # Filter rows with zero exposure and set sef.df_event
        self.df_event = df_in[df_in["exposure"] > 0]
        # Filter on Detected Flux Points
        self.df_event = self.df_event[self.df_event["sqrt(ts)"] > 3]
        print(f"Time Range selected (MJD): {self.df_event['lwtime_mjd'].min()}-{self.df_event['uptime_mjd'].max()}")
        print(f"Total number of rows selected (no zero-exposure, 3sigma): {len(self.df_event)}")
        print(f"Total number of source photons selected (no zero-exposure, sqrt(ts)>3): {self.df_event['counts'].sum()}")
        
        # Extract the relevant columns for Bayesian blocks processing.
        t_i = self.df_event['time_start_tt'].to_numpy() # Lower bound time
        t_f = self.df_event['time_end_tt'  ].to_numpy() # Upper bound time
        
        # Calculate custum data cells (Time edges)
        self.data_cells = np.append(t_i, t_f[-1])
        
        # Calculate the midpoint time and time delta.
        self.t_delta = (t_f - t_i)
        self.dt = min(self.t_delta)
        self.t_c = t_i + self.t_delta/2
        
        # Compute Flux
        self.exp = self.df_event['exposure'].to_numpy()
        self.cts = self.df_event['counts'  ].to_numpy().astype(int)
        self.rate= self.df_event['flux'    ].to_numpy()
        rate_err = self.df_event['flux_err'].to_numpy()
        
        # Apply Flux Correction, set x and sigma
        if ratecorrection is None:
            print(f"Apply Bayesian Blocks on MLE FLUX with no scaling")
            self.x = self.rate
            self.sigma = rate_err
        elif ratecorrection==-1:
            print(f"Apply Bayesian Blocks on CTS (integers)")
            self.x = self.cts
            self.sigma = np.sqrt(self.x)
        elif ratecorrection==0:
            expmean = self.exp.mean()
            print(f"Apply Bayesian Blocks on MLE FLUX scaled to integers with a mean exposure factor={expmean:.3g}")
            self.x = (self.rate * expmean).astype(int)
            self.sigma = (rate_err * expmean).astype(int)
        else:
            print(f"Apply Bayesian Blocks on MLE FLUX scaled to integers with a correction factor={ratecorrection:.3g}")
            self.x = (self.rate * ratecorrection).astype(int)
            self.sigma = (rate_err * ratecorrection).astype(int)
        
        return None
    
    
    def _read_custom_lc(self, file_path, tstart, tstop, ratecorrection):
        """
        Read the input file, provided in CUSTOM_LC format.
        
        Parameters
        ----------
        file_path : str
            The path to the data file. 
        tstart, tstop : float 
            Time start, stop in to select events. If None, no selection is applied.
        ratecorrection: None or 0 or float
            Multiplication factor to convert rates (cts/exp) into int used with AGILE_AP data.
            - Set to None to not use.
            - Set to -1 to apply BB on the counts instead of flux points (e.g. for Ratemeter data).
            - Set to 0 to use the mean exposure, e.g. to apply BB on flux points scaled to int.
            - Set to a float value to multiply the rates (cts/exp) accordingly.
        """
        print(f"Read input file {file_path}")
        # Load the binned light curve data from the file.
        df_in = pd.read_csv(file_path, delim_whitespace=True, header=None)
        df_in.columns = ['lwtime', 'uptime', 'rate']
        
        # Filter the data to only include rows within the event time range.
        if tstart is not None:
            df_in = df_in[df_in["lwtime"] >= tstart]
        if tstop is not None:
            df_in = df_in[df_in["uptime"] <= tstop]
        
        # Set self.df_event
        self.df_event = df_in
        print(f"Time Range selected (MJD): {self.df_event['lwtime'].min()}-{self.df_event['uptime'].max()}")
        print(f"Total number of rows selected: {len(self.df_event)}")
        
        # Extract the relevant columns for Bayesian blocks processing.
        t_i = self.df_event['lwtime'].to_numpy() # Lower bound time
        t_f = self.df_event['uptime'].to_numpy() # Upper bound time
        
        # Calculate custum data cells (Time edges)
        self.data_cells = np.append(t_i, t_f[-1])
        
        # Calculate the midpoint time and time delta.
        self.t_delta = (t_f - t_i)
        self.dt = min(self.t_delta)
        self.t_c = t_i + self.t_delta/2
        
        # Compute Flux
        self.exp = None
        self.cts = None
        self.rate= self.df_event['rate'].to_numpy()

        # Apply Flux Correction, set x
        if ratecorrection is None:
            print(f"Apply Bayesian Blocks on RATE with no scaling, assume no Error")
            self.x = self.rate
            self.sigma = np.zeros_like(self.x)
        elif ratecorrection == -1:
            print(f"Apply Bayesian Blocks on RATE with no scaling, assume Poisson Error")
            self.x = self.x
            self.sigma = np.sqrt(self.x)
        elif ratecorrection == 0:
            xmean = self.rate.mean() / 5.0
            print(f"Apply Bayesian Blocks on RATE scaled to integers with a factor={xmean:.3g}, assume Poisson Error")
            self.x = (self.x / xmean).astype(int)
            self.sigma = np.sqrt(self.x)
        else:
            print(f"Apply Bayesian Blocks on RATE scaled to integers with a factor={ratecorrection:.3g}, assume Poisson Error")
            self.x = (self.x * ratecorrection).astype(int)
            self.sigma = np.sqrt(self.x)
        
        return None
    
    
    def _read_agile_ph(self, file_path, tstart, tstop):
        """
        Read the input file, provided in CUSTOM_LC format.
        
        Parameters
        ----------
        file_path : str
            The path to the data file. 
        tstart, tstop : float 
            Time start, stop in to select events. If None, no selection is applied.
        """
        print(f"Read input file {file_path}")
        # Load the time-tagged events data from the file.
        df_in = pd.read_csv(file_path, delim_whitespace=True, header=None, usecols=[0])
        df_in.columns = ['time']
        
        # Filter the data to only include rows within the event time range.
        if tstart is not None:
            df_in = df_in[df_in["time"] >= tstart]
        if tstop is not None:
            df_in = df_in[df_in["time"] <= tstop]
        
        # Set self.df_event
        self.df_event = df_in
        print(f"Number of photons selected: {len(self.df_event)}")
        
        # Extract the relevant columns for Bayesian blocks processing.
        self.data_cells = None
        self.t_delta = None
        self.dt = 0
        self.t_c = self.df_event['time'].to_numpy() # Event times
        
        self.x = np.ones_like(self.t_c) # Set counts to 1 for each event
        self.sigma = np.zeros_like(self.t_c) # Set sigma to 0
        
        return None
    
    
    def select_event(self, file_path : str, file_mode, tstart=None, tstop=None, ratecorrection=0):
        """
        Read the input data according to a specified format, apply selection and eventual correction.
        
        Allowed data formats:
        - BINNED data: AGILE_AP, AGILE_MLE, CUSTOM_LC
        - UNBINNED data: AGILE_PH 

        Parameters:
        -----------
        file_path : str
            The path to the data file. 
        file_mode : str, int or FileMode
            FileMode to set read function and DataMode.
        tstart, tstop : float 
            Time start, stop in MJD to select events. If None, no selection is applied.
        ratecorrection: None or 0 or float
            Multiplication factor to convert rates (cts/exp) into int used with AGILE_AP data.
            - Set to None to not use.
            - Set to 0 to use the mean exposure (effective area * dt), typical value 1e7.
            - Set to a float value to use it directly.
        """
        # Determine FileMode
        self._filemode = self._parse_file_mode(file_mode)
        
        # Determine DataMode
        if self._filemode == FileMode.AGILE_PH:
            self._datamode = DataMode.UNBINNED
        else:
            self._datamode = DataMode.BINNED # FileMode.AGILE_AP, FileMode.AGILE_MLE, FileMode.CUSTOM_LC

        print(f"Select Events. FileMode={self._filemode.name}, DataMode={self._datamode.name}")
        
        # Read File according to File Mode
        if self._filemode == FileMode.AGILE_PH:
            self._read_agile_ph(file_path=file_path, tstart=tstart, tstop=tstop)
        elif self._filemode == FileMode.AGILE_AP:
            self._read_agile_ap(file_path=file_path, tstart_mjd=tstart, tstop_mjd=tstop, ratecorrection=ratecorrection)
        elif self._filemode == FileMode.AGILE_MLE:
            self._read_agile_mle(file_path=file_path, tstart_mjd=tstart, tstop_mjd=tstop, ratecorrection=ratecorrection)
        elif self._filemode == FileMode.CUSTOM_LC:
            self._read_custom_lc(file_path=file_path, tstart=tstart, tstop=tstop, ratecorrection=ratecorrection)
        else:
            raise ValueError("No filemode")
        
        # Set the data for Bayesian blocks analysis.
        self.bblocks.set_argsIn(x=self.x, t=self.t_c, sigma=self.sigma, dt=self.dt, datamode=self.datamode, t_delta=self.t_delta, cts = self.cts, exp=self.exp, data_cells=self.data_cells, rate=self.rate)
        
        return None
            

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
