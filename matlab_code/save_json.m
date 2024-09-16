function save_json(data_in, data_out, json_file_path)
    % Combine the structs and encode as JSON
    json_data = struct( ...
        'data_in', data_in, ...
        'data_out', data_out);
    json_str = jsonencode(json_data);

    % Write to file
    fid = fopen(json_file_path, 'w');
    fprintf(fid, json_str);
    fclose(fid);
end