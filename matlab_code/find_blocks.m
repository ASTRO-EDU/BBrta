function data_out = find_blocks( data_in )
%==========================================================================
% Title: Studies in Astronomical Time Series Analysis. 
%        VI. Bayesian Block Representations
% Authors: Jeffrey D. Scargle, Jay P. Norris, Bard Jackson, James Chaing 
%==========================================================================
%
% Code name: find_blocks.m
%
% Language: MatLab
%
% Code tested under Mac OS X 10.5.8
%
% Description of input data:  data_in (data structure):
%                             data_in.cell_data 
%                             data_in.nn_vec 
%                             data_in.tt 
%                             data_in.fp_rate 
%                             data_in.ncp_prior 
%                             data_in.do_iter
%                             data_in.tt_start 
%                             data_in.tt_stop 
%                             data_in.ncp_prior 
%                            
% Description of output data: data_out (data_structure):
%                             data_out.change_points
%                             data_out.num_vec
%                             data_out.rate_vec
%                             data_out.best
%                             data_out.last
%                             data_out.ncp_prior
%                             data_out.nn
%                             data_out.block_length
%
% System requirements: as needed to run MatLab R2009 or later
%
% Calls to external routines: nlogn.m
%
% Additional comments: Basic algorithm for single time series;
%                      includes "empty block" option for TTE data
%
%==========================================================================
%data
% Discover data mode and unpack the input data structure
if isfield( data_in, 'cell_data')
    data_mode = 3;                   % POINT MEASUREMENTS
    cell_data = data_in.cell_data;
    [ num_points, dummy ] = size( cell_data );
    tt = 1: num_points; % nominal evenly spaced time points
    nn_vec = [];
elseif isfield( data_in, 'nn_vec')
    data_mode = 2;                   % BINNED DATA
    nn_vec = data_in.nn_vec;
    if isfield( data_in, 'tt')
       tt = data_in.tt;
    else
       tt = 1:length( nn_vec ); % nominal evenly spaced times
    end
else
    data_mode = 1;                   % TIME-TAGGED EVENT DATA
    if isfield( data_in, 'tt')
       tt = data_in.tt;
    else
       error('times not specified')
    end
    nn_vec = ones( size( tt' ) );
end
disp('DATA MODE IS:');
disp(data_mode);

num_points = length( tt );
dt = diff( tt );
if min( dt ) < 0
    error('Points must be ordered')
end

if isfield( data_in, 'fp_rate')
    fp_rate = data_in.fp_rate;
else
    fp_rate = .05; % Default value
end

if isfield( data_in, 'ncp_prior')
    ncp_prior = data_in.ncp_prior;
else
    ncp_prior = 4 - log( fp_rate / ( 0.0136 * num_points .^ (0.478 ) ) );
end

if isfield( data_in, 'do_iter')
    do_iter = data_in.do_iter;
else
    do_iter = 0; % Default: do not iterate on ncp_prior
end
  
if isfield( data_in, 'tt_start')
    tt_start = data_in.tt_start;
else
    tt_start = tt(1); % default start
end

if isfield( data_in, 'tt_stop')
    tt_stop = data_in.tt_stop; % default stop
else
    tt_stop = tt(end);
end

%===========================================
count_vec = [];
if data_mode ~= 3
    edges = [ tt_start 0.500000 *( tt(2:end) + tt(1:end-1) )' tt_stop ];
    block_length = tt_stop - edges;
end
iter_count = 0;
iter_max = 10;
while 1
    
    best = [];
    last = [];
    cpu_0 = cputime;
    
    for R = 1:num_points 

        if data_mode == 3 % Measurements, normal errors
           
            sum_x_1 = cumsum( cell_data( R:-1:1, 1 ) )'; % sum( x    / sig^2 )
            sum_x_0 = cumsum( cell_data( R:-1:1, 2 ) )'; % sum[ 1    / sig^2 )
            fit_vec = ( ( sum_x_1(R:-1:1) ) .^ 2 ) ./ ( 4 * sum_x_0(R:-1:1) ); %   ...
            
        else
            % Takes the first R elements of block_length and subtracts
            % from each of them the element at position R+1
            arg_log = block_length(1:R) - block_length(R+1);
            % Identifies elements of arg_log that are less than or equal to
            % 0 and replaces them with Inf (infinity). The use of Inf may 
            % be to avoid logarithm problems of non-positive numbers in the 
            % next line.
            arg_log( find( arg_log <= 0 ) ) = Inf;
            % Computes the cumulative sum of the elements of nn_vec, but 
            % here the sum is computed over the elements taken from last to
            % first
            nn_cum_vec = cumsum( nn_vec(R:-1:1) );
            nn_cum_vec = nn_cum_vec(R:-1:1);

            fit_vec = nn_cum_vec .* ( log( nn_cum_vec ) - log( arg_log ) );
            %fit_vec = nn_cum_vec .* ( log( nn_cum_vec + 1e-3) - log( arg_log + 1e-3 ) );
            %fit_vec = nn_cum_vec .* ( log( nn_cum_vec / arg_log ) );
            %%fit_vec = -nn_cum_vec .* log( arg_log );
            %if R > 2
            %disp(arg_log)
            %disp(nn_cum_vec)
            %    error('stop')
            %end
        end
       
        [ best(R), last(R)] = max( [ 0 best ] + fit_vec - ncp_prior );
       
    end

    %----------------------------------------------------------------------
    % Now find changepoints by iteratively peeling off the last block
    %----------------------------------------------------------------------
    
    index = last( num_points );
    change_points = [];

    while index > 1
        change_points = [ index change_points ];
        index = last( index - 1 );
    end
    
    %----------------------------------------------------------------------
    %            Iterate if desired
    %----------------------------------------------------------------------
    if do_iter == 0
        break
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
        ncp_prior = 4 - log( fp_rate / ( 0.0136 * num_points .^ (0.478 ) ) );
        cpt_old    = change_points;
        
    end
    
end

num_changepoints = length( change_points );
num_blocks = num_changepoints + 1;

 rate_vec = zeros( num_blocks, 1 );
  num_vec = zeros( num_blocks, 1 );
   dt_vec = zeros( num_blocks, 1 );
 tt_1_vec = zeros( num_blocks, 1 );
 tt_2_vec = zeros( num_blocks, 1 );

cpt_use = [ 1 change_points ];

for id_block = 1: num_blocks
    
    ii_1 = cpt_use( id_block ); % start
    if id_block < num_blocks
        ii_2 = cpt_use( id_block + 1 ) - 1;
    else
        ii_2 = num_points;
    end
    
    if data_mode == 3
        xx_this = cell_data( ii_1:ii_2, 1);
        wt_this = cell_data( ii_1:ii_2, 2); 
        rate_vec( id_block ) = sum( wt_this .* xx_this ) / sum( wt_this );
        
    else
        num_this = sum( nn_vec( ii_2: ii_2 ) );
        delta_tt = tt( ii_2 ) - tt( ii_1 );
        num_vec( id_block ) = num_this;
        rate_this = num_this / delta_tt;
        rate_vec( id_block ) = rate_this;
    end
    
end

if data_mode == 1 % Empty bin check
    
    for id_cp = 1: num_changepoints 

        num_left = num_vec( id_cp );
        dt_left  =  dt_vec( id_cp );

        num_right = num_vec( id_cp + 1 );
        dt_right  =  dt_vec( id_cp + 1 );

        fitness_left  = nlogn( num_left, dt_left );
        fitness_right = nlogn( num_right, dt_right );
        fitness = fitness_left + fitness_right;

        % move first cp of right block to left?

        num_left_movedown = num_left + 1;
        dt_left_movedown = dt_left;
        num_right_movedown = num_right - 1;
        dt_right_movedown = dt_right;

        fitness_left_movedown  = nlogn( num_left_movedown, dt_left_movedown );
        fitness_right_movedown = nlogn( num_right_movedown, dt_right_movedown );
        fitness_movedown = fitness_left_movedown + fitness_right_movedown;

        % move last cp of left block to right?
        num_left_moveup = num_left - 1;
        dt_left_moveup = dt_left;
        num_right_moveup = num_right + 1;
        dt_right_moveup = dt_right;

        fitness_left_moveup  = nlogn( num_left_moveup, dt_left_moveup );
        fitness_right_moveup = nlogn( num_right_moveup, dt_right_moveup );
        fitness_moveup = fitness_left_moveup + fitness_right_moveup;

        if fitness_movedown > fitness_moveup
            if fitness_movedown > fitness

                %fprintf(1,'move point at cp %4.0f down\n', id_cp )
                num_vec( id_cp ) = num_left_movedown;
                num_vec( id_cp + 1 ) = num_right_movedown;

            end
        else
            if fitness_moveup > fitness

                %fprintf(1,'move point at cp %4.0f up\n', id_cp )
                num_vec( id_cp ) = num_left_moveup;
                num_vec( id_cp + 1 ) = num_right_moveup;

            end
        end
    end
    
end

data_out.edge_points   = edges(change_points);
data_out.change_points = change_points;
data_out.num_vec       = num_vec;
data_out.rate_vec      = rate_vec;
data_out.best = best;
data_out.last = last;
data_out.ncp_prior     = ncp_prior;
data_out.nn = nn_vec;
    
if data_mode ~= 3
    data_out.block_length = block_length;
end

