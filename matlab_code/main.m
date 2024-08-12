% Main script to use prepare_data and find_blocks
% Carica la funzione prepare_data
data_in = prepare_data('tte');

% Eseguire la funzione find_blocks con la struttura data_in
data_out = find_blocks(data_in);

% Visualizza i risultati (opzionale)
data_out.change_points = data_out.change_points - 1;
disp('Change Points:');
disp(data_out.change_points);
disp('Edge Points:');
disp(data_out.edge_points);

edgepoints = load("output.txt")';
res = data_out.edge_points - edgepoints;

disp('Difference edgepoints MATLAB vs Python')
disp(res);
%disp('Number of Events per Block:');
%disp(data_out.num_vec);
%disp('Rate per Block:');
%disp(data_out.rate_vec);
