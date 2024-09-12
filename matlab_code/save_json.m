function save_json(data_in, data_out, json_file_path)
    % Convert to structs
    data_in_struct = struct('data_in', data_in);
    data_out_struct = struct('data_out', data_out);

    % Combine the structs and encode as JSON
    json_data = struct('data_in', data_in_struct, 'data_out', data_out_struct);
    json_str = jsonencode(json_data);

    % Write to file
    fid = fopen(json_file_path, 'w');
    fprintf(fid, json_str);
    fclose(fid);
end