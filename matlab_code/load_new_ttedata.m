function [ times, channels, detectors ] = load_new_ttedata( file_name )
%==========================================================================
% Title: Studies in Astronomical Time Series Analysis. 
%        VI. Bayesian Block Representations
% Authors: Jeffrey D. Scargle, Jay P. Norris, Bard Jackson, James Chaing 
%==========================================================================
%
% Code name: load_new_ttedata.m
%
% Language: MatLab
%
% Code tested under Mac OS X 10.5.8
%
% Description of input data: file_name - name of BATSE TTE data files 
%
% Description of output data:     times -- photon times
%                              channels -- energy channel markers
%                             detectors -- detector markers
%
% System requirements: as needed to run MatLab R2009 or later
%
% Calls to external routines: none
%
% Additional comments: data from NASA website 
% ftp://legacy.gsfc.nasa.gov/compton/data/batse/ascii_data/batse_tte/
%
%==========================================================================


% load_new_ttedata.m
% Open and read data from TTE files.

file_in = [ file_name ];

[ fid message ] = fopen( file_in, 'r');

if fid == -1 
   fprintf(1,['Error opening file ' file_in  '\n'] )
else
   fprintf(1,['Successfully opened file ' file_in  '\n'] )
end

%--------------------------
%  Read the File Headder
%--------------------------

format1 = '%s';
for ijk = 1:5
    aa(ijk).line = fgetl( fid );
end

npts = aa( 2 ).line;
npts = eval( npts( 9: length( npts ) ) );

%----------------------------------------------------------
%  Now read the data: times, channels, detectors
%----------------------------------------------------------

[ times, count_times ] = fscanf(fid,'%f', npts);
[ channels, count_channels ] = fscanf(fid,'%f', npts);
[ detectors, count_detectors ] = fscanf(fid,'%f', npts);

%---------------------------
%  carry out some checks
%---------------------------

ii1 = find( channels ~= 1 & channels ~= 2 & channels ~= 3 & channels ~= 4);
if ~isempty(ii1) %%  ~= []
   fprintf(1,'Warning: %4.0f channels are not 1,2,3 or 4!\n', length(ii1) )
end

[ a, count ] = fscanf(fid,format1,1);
if count ~= 0
   fprintf(1,'Error; End of file not reached!\n')
end

%-----------------
%  Close the File 
%-----------------

message = fclose( fid );
if message ~= 0
   fprintf(1,['Error closing file ' file_in  '\n'] )
end

%---------------------------------------------------------------
% strip off last few points if they are discrepant, and rescale
%---------------------------------------------------------------
max_strip = 10;
min_discrep = 10;

% Establish baseline of "good data"

n_times = length(times);
baseline_size = fix( 0.01 * n_times );

i2 = n_times - max_strip;
i1 = i2 - baseline_size + 1;
if i1 < 1,i1 = 1; end % Unlikely!

% Remove any points at the end that have a
%   relative value that is much greater than
%   the average value in the baseline region.

baseline = mean( times(i1:i2 ) );
discrep = ( times( n_times ) - baseline) / baseline;

count = 0;

while (discrep > min_discrep) & (count < max_strip)

     count = count + 1;
   n_times = length(times)-1;
     times = times(1:n_times);
   discrep = (times( n_times ) - baseline) / baseline;

end
%...........................................
n_times = length(times); % Just to be sure
times = times(1:n_times);
channels = channels(1:n_times);


