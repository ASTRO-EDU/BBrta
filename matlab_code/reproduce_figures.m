%==========================================================================
% Title: Studies in Astronomical Time Series Analysis. 
%        VI. Bayesian Block Representations
% Authors: Jeffrey D. Scargle, Jay P. Norris, Bard Jackson, James Chaing 
%==========================================================================
%
% Code name: reproduce_figures.m
%
% Language: MatLab
%
% Code tested under Mac OS X 10.5.8
%
% Description of input data: none
%
% Description of output data: *.eps  (*.mat)
%
% System requirements: as needed to run MatLab R2009 or later
%
% Calls to external routines: all figure and data scripts 
%                            
% Additional comments: Reproduce all 12 figures in this paper.
%                      Option: recompute intermediate data files too.
%                      Plot script is:         figure_XXXX.m
%                      Data generation script         XXXX.m 
%                      The data file                  XXXX.mat
%                      (Exception: circle_data.mat has data for both 
%                       figures circle_plot and circle_hist).
%
%==========================================================================

erase_figs = 1;
if erase_figs == 1
    ! rm *eps
    ! rm *pdf
end

recompute_data = 0;
if recompute_data == 1
   
    % Figure 7: ex_tte
    ! rm ex_tte.mat
    ex_tte
    
    % Figure 1: cross_valid
    ! rm cross_valid.mat
    cross_valid
    
    % Figures 10 & 11: circle_plot & circle_hist
    ! rm circle_data.mat
    circle_plot
    
    % Figure 5: calibrate_bins
    ! rm calibrate_bins.mat
    calibrate_bins
   
    % Figure 6: calibrate_gauss
    ! rm calibrate_gauss.mat
    calibrate_gauss
    
    % Figure 4: ac_limit_2
    % YYY = 32, 64,128, 256,512, 1024
    ! rm ac_limit_1_*.mat ac_limit_2_*.mat
    ac_limit_2
    
end

figure_cross_valid
figure_mult_pic
figure_ac_limit_1
figure_ac_limit_2
figure_calibrate_bins
figure_cal_gauss
figure_ex_tte
figure_mult
figure_circle_plot
figure_circle_hist
figure_cell
figure_cp_error

