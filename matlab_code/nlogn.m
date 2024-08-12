function  ff = nlogn( nn, dt  )
%==========================================================================
% Title: Studies in Astronomical Time Series Analysis. 
%        VI. Bayesian Block Representations
% Authors: Jeffrey D. Scargle, Jay P. Norris, Bard Jackson, James Chaing 
%==========================================================================
%
% Code name: nlogn.m
%
% Language: MatLab
%
% Code tested under Mac OS X 10.5.8
%
% Description of input data: nn (cell population array)
%                            dt (cell width array)
%
% Description of output data: ff
%
% System requirements: as needed to run MatLab R2009 or later
%
% Calls to external routines: none
%
% Additional comments: evaluate cost N log( N / dT ) with correct limit
%                      for zero cell population
%
%=========================================================================

ff = nn .* ( log( nn ) - log( dt ) );
ff( find( nn == 0 ) ) = 0; % limit as nn and dt --> 0