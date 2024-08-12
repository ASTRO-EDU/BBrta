  function data_out = find_blocks_mult( data_in )
%==========================================================================
% Title: Studies in Astronomical Time Series Analysis. 
%        VI. Bayesian Block Representations
% Authors: Jeffrey D. Scargle, Jay P. Norris, Bard Jackson, James Chaing 
%==========================================================================
%
% Code name: find_blocks_mult.m
%
% Language: MatLab
%
% Code tested under Mac OS X 10.5.8
%
% Description of input data:  data_in (data structure):
%                             
%                             data_in.fp_rate 
%                             data_in.do_iter
%                             data_in.data_all  --> data_all
%                                                   data_all(*).data_series
%
% Description of output data: data_out (data_structure):
%                             data_out.change_points
%                             data_out.tt
%                             data_out.ncp_prior_vec
%                             data_out.data_matrix
%                             data_out.best
%                             data_out.last
%                             data_out.index_vec
%                             data_out.data_mode_vec
%
% System requirements: as needed to run MatLab R2009 or later
%
% Calls to external routines: none
%
% Additional comments: Find the optimal block representation for
%                        multi-variate data.
%
%=========================================================================

   data_all = data_in.data_all;
num_series  = length( data_all ); % number of time series
   
if isfield( data_in, 'do_iter')
    do_iter = 1;
    iter_max = data_in.do_iter; % halt at maximum number of iterations
else
    do_iter = 0; % Default: do not iterate on ncp_prior
end

if isfield( data_in, 'fp_rate')
    fp_rate = data_in.fp_rate;
else
    fp_rate = .01; % Default value of false positive rate
end

%============================================================
%  First pass to collect information to set up data matrix
%============================================================

data_mode_vec = zeros( num_series, 1 ); 
ncp_prior_vec = zeros( num_series, 1 );
 tt_start_vec = zeros( num_series, 1 );
  tt_stop_vec = zeros( num_series, 1 );
 id_start_vec = zeros( num_series, 1 );

       tt = []; % Initialize master time array
 ii_start =  1; % Initialize marker for parsing the data
row_count =  1; % First row is array containing all times

for id_series = 1: num_series
  
    data_series = data_all( id_series ).data_series;
    
    %--------------------------------------------------------
    %         Identify and store data mode 
    %--------------------------------------------------------
  
    if isfield( data_series, 'cell_data')
        data_mode = 3;      % This series is point measurements
    elseif isfield( data_series, 'nn_vec')
        data_mode = 2;      % This series is binned data
    else
        data_mode = 1;      % This series is time-tagged event data
    end
    row_count = row_count + 2; % these modes all take two rows
    
    data_mode_vec( id_series ) = data_mode;
  
    %--------------------------------------------------------
    % Process time markers (all data modes must have one)
    %--------------------------------------------------------
    
    tt_this = data_series.tt;
    dt_this = diff( tt_this );
    
    if find( dt_this < 0 )
        error('Data must be time-ordered.')
    end
    
    if data_mode == 1
        
        % Combine any duplicate time_tags:
        dt_min = 0; % set to small value >0 to combine nearly equal times
        num_skip = 3;
        nn_vec = ones( size( tt_this ) );
        ii_dupe = find( dt_this <= dt_min ); % indices of small intervals
       
        while ~isempty( ii_dupe )
            
            iu = ii_dupe( 1: num_skip: end ); 

            % replace with average of the two identical (or close) times
            tt_this( iu ) = 0.5 * ( tt_this( iu ) + tt_this( iu + 1 ) );

            % replace with sum of the two corresponding cell populations
            nn_vec( iu ) = nn_vec( iu ) + nn_vec( iu + 1 );

            % Remove (null) second member of these pairs
            tt_this( iu + 1 ) = []; 
             nn_vec( iu + 1 ) = []; 

            % any more duplicates?  If so, go again; if not, you are done.
            dt_this = diff( tt_this );
            ii_dupe = find( dt_this <= dt_min );

        end

        % store adjusted data back into master data structure  data_all
        data_series.tt     = tt_this;
        data_series.nn_vec = nn_vec;
        data_all( id_series ).data_series = data_series;

    end
    
    num_points_this = length( tt_this );
    tt = [ tt tt_this' ]; % concetanate all times
   
    % This array keeps track of index ranges of entries for each series:
    ii_start_vec( id_series ) = ii_start;
    ii_start = ii_start + num_points_this; % update to start of next series
     
    if isfield( data_series, 'tt_start')
        tt_start = data_series.tt_start;
    else
        tt_start = tt(1) - 0.5 * median( diff( tt ) );% default start time
    end
    
    tt_start_vec( id_series ) = tt_start;
    
    if isfield( data_series, 'tt_stop')
        tt_stop = data_series.tt_stop;
    else
        tt_stop = tt(end) + 0.5 *  median( diff( tt ) );% default stop time
    end
    
    tt_stop_vec( id_series ) = tt_stop;
     
    %--------------------------------------------------------
    % Store ncp_prior if present; if not, use default
    %--------------------------------------------------------
  
    if isfield( data_series, 'ncp_prior')
        ncp_prior = data_series.ncp_prior;
    else
        ncp_prior = 4 - log( fp_rate / (0.0136*num_points_this .^ (0.478 ) ) );
    end
    ncp_prior_vec( id_series ) = ncp_prior;
 
end % for id_series

%==========================================================================
%        Construct data matrix  (Figure 2: Top Panel)
%==========================================================================

num_rows  = row_count;
num_data  = length( tt ); % total number of data points
data_matrix = zeros( num_data, num_rows );
index_vec   = zeros( num_data, 1 );

row_count = 1; % reset row counter for staging of data
data_matrix( :, row_count ) = tt; % First row contains all times (unordered)

for id_series = 1: num_series
    
    data_series = data_all( id_series ).data_series;
 
    %----------------------------------------------------
    %    Get index range for data for this series
    %----------------------------------------------------
    
    ii_start  =  ii_start_vec( id_series );
    data_mode = data_mode_vec( id_series );
    
    if id_series == num_series
        ii_stop = num_data;
    else
        ii_stop  = ii_start_vec( id_series + 1 ) - 1;
    end
    
    index_vec( ii_start: ii_stop ) = id_series; % keep track of series
 
    %----------------------------------------------------
    %     Compute mode-dependent fitness data 
    %----------------------------------------------------
    
    if data_mode == 1 | data_mode == 2
        
        nn_vec  = data_series.nn_vec;
        tt_this = data_series.tt;
        tt_start = tt_start_vec( id_series );
        tt_stop  =  tt_stop_vec( id_series );
        
        dt_start =           0.5 * ( tt_this(2) + tt_this(1) ) - tt_start;
        dt_stop  = tt_stop - 0.5 * ( tt_this(end-1) + tt_this(end) );
        delt_tt = [ dt_start ( tt_this(3:end) - tt_this(1:end-2) )' / 2  dt_stop ];
      
        row_count = row_count + 1;
        data_matrix( ii_start: ii_stop, row_count ) = delt_tt;
        
        row_count = row_count + 1;
        data_matrix( ii_start: ii_stop, row_count ) = nn_vec;
        
    elseif data_mode == 3
        
        cd = data_series.cell_data;
        row_count = row_count + 1;
        data_matrix( ii_start: ii_stop, row_count ) = cd(:,1);% (x/sig^2 )
        
        row_count = row_count + 1;
        data_matrix( :, row_count ) = 1;% non-zero denominator
        data_matrix( ii_start: ii_stop, row_count ) = cd(:,2);%  (1/sig^2)
        
    end
    
end

%  Redistribute data according to time order (Figure 2: Bottom Panel):
num_points = length( tt );
dm = data_matrix;
[ tt, ii_sort ] = sort( data_matrix( :, 1 ) ); % time order index
data_matrix( :, : ) = data_matrix( ii_sort, : ); % ro-order everything
index_vec = index_vec( ii_sort );

%==========================================================================
% Now apply the basic dynamic programming algorithm
%==========================================================================

tt_start = min( tt_start_vec );
tt_stop  = max( tt_stop_vec ); % min( tt_stop      );

block_length = ... % Make array of lentths of the "last blocks"
    tt_stop - [ tt_start 0.5*( tt(2:end) + tt(1:end-1) )' tt_stop ];

iter_count = 0;

while 1 % If iterating, continue until maximum number reached
       
    best = [];
    last = [];
    cpu_0 = cputime;
    
    for R = 1:num_points 
        
        fit_vec = zeros( 1, R ); % Initlize last-block fitness array
        row_count = 1; % initialize
        
        for id_series = 1: num_series
            
            data_mode = data_mode_vec( id_series );
            ncp_prior = ncp_prior_vec( id_series );
            
            if data_mode == 1 | data_mode == 2
                
                row_count = row_count + 1;
                delt_tt = data_matrix( 1:R, row_count );
                
                row_count = row_count + 1;
                nn_vec = data_matrix( 1:R, row_count )';
                
                 nn_cum_vec = reverse( cumsum( reverse( nn_vec  ) ) );
                arg_log_vec = reverse( cumsum( reverse( delt_tt ) ) );
                arg_log_vec( find( arg_log_vec <= 0 ) ) = Inf;
                
 fit_vec_this = nn_cum_vec .* ( log( nn_cum_vec) - log( arg_log_vec' ) );
        
            elseif data_mode == 3 % Measurements, normal errors

                row_count = row_count + 1;
                cd = data_matrix( :, row_count );
                sum_x_1 = cumsum( cd( R:-1:1, 1 ) ); % sum( x / sig^2 )
                
                row_count = row_count + 1;
                cd = data_matrix( :, row_count );
                sum_x_0 = cumsum( cd( R:-1:1, 1 ) ); % sum[ 1 / sig^2 )
 
                fit_vec_this = ...
                    ( (( sum_x_1(R:-1:1) ) .^ 2 ) ./ ...
                    ( 4 * sum_x_0(R:-1:1) ) )';

            end
            
            ii_bad = find( isnan( fit_vec_this ) );
            fit_vec_this( ii_bad ) = 0;
            fit_vec = fit_vec + fit_vec_this - ncp_prior_vec( id_series );

        end
        
        [ best(R), last(R)] = max( [ 0 best ] + fit_vec  );
        
    end
    
    %-------------------------------------------------------------------------
    % Now find changepoints by iteratively peeling off the last block
    %----------------------------------------------------------------------
    
    index = last( num_points );
    change_points = [];

    while index > 1
        change_points = [ index change_points ];
        index = last( index - 1 );
    end
    
    if do_iter == 0
        break % done; not iterating on ncp_prior
    else
        iter_count = iter_count + 1;
        num_cp = length( change_points );
        
        if num_cp < 1
            num_cp = 1;
        end

        if exist('cpt_old' )

            if num_cp == length( cpt_old ) % compare with previous iteration
                err_this = sum( abs( change_points - cpt_old ) );
            else
                err_this = Inf;
            end

            if err_this == 0
                fprintf(1,'Converged at %3.0f\n', iter_count )
                break
            end

            if iter_count > iter_max
                fprintf(1,'Did not converge at %3.0f\n', iter_count )
                break
            end

        end

        fp_rate = 1 - ( 1 - fp_rate ) .^ ( 1 / num_cp );
        ncp_prior_old = ncp_prior;
        ncp_prior_new = 4 - log( fp_rate / ( 0.0136 * num_points .^ (0.478 ) ) );
        ncp_prior_vec( id_series ) = ncp_prior_new;
        cpt_old    = change_points;
        [ ncp_prior_old ncp_prior_new num_cp ]
        
    end
    
end

data_out.change_points = change_points;
data_out.tt_all = tt;
data_out.ncp_prior_vec = ncp_prior_vec;
data_out.data_matrix = data_matrix;
data_out.last = last;
data_out.best = best;
data_out.index_vec = index_vec;
data_out.data_mode_vec = data_mode_vec;
