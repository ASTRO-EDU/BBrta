function data_in = prepare_data(mod)
    % mod = 'binned' | 'tte'
    curve_filename = "curve.txt";
    times_filename = "times.txt";
    tte_filename = "tte.txt";
 
    % Definire la struttura data_in
    data_in = struct;

    if mod == "binned"
        % Carica i dati dal file curve.txt
        data = load(curve_filename);
        times = load(times_filename);
        % Vettore dei dati della curva di luce
        data_in.x = data';
    elseif mod == "tte"
        times = load(tte_filename);
    else
        error("No `mod` recognized. mod = 'binned' or 'tte'")
    end
    % Vettore dei tempi (assumiamo tempi uniformemente distribuiti)
    %data_in.tt = (0:length(data)-1)';
    data_in.t = times;

    % Impostare gli altri campi con valori predefiniti o calcolati
    p0 = 2.;
    data_in.p0 = p0; % Tasso di falsi positivi predefinito
    data_in.do_iter = 0; % Non iterare per default
    data_in.rate = NaN;
end
