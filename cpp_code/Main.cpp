#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>  // Per std::atof
#include <map>
#include <string>

// #include "BayesianBlocks.h"
#include "fitnessfunc.hpp"
#include "events.hpp"
#include "bayesblocks.hpp"
#include "utils.hpp"

int main(int argc, char* argv[]) {   
    // Verifica che l'utente abbia fornito il valore di p0
    if (argc != 3) {
        std::cerr << "Uso: " << argv[0] << " <p0> <path/to/save/changing_points.txt>" << std::endl;
        return 1;
    }
    // Estrai il valore di p0 dagli argomenti della linea di comando
    double p0 = std::atof(argv[1]);
    double ncp_prior = -1;
    double gamma = -1;
    // Estrai il percorso del file di output dagli argomenti della linea di comando
    std::string path_out = argv[2];
    
    // Dichiara i vettori e il file di input
    std::ifstream infile("../Data/curve.txt");
    std::vector<double> x;
    std::vector<double> t;
    
    // Importa valori dal file al Vector
    double value;
    int i = 0; 
    while (infile >> value) {
        x.push_back(value);
        t.push_back(i);
        i++;
    }
    // Chiusura del file dopo averlo usato
    infile.close();
    
    // Stampa l'array
    // printVector(x, "x: ");
    // printVector(t, "t: ");
    
    try {

        std::vector<double>  change_points = bayesian_blocks(t, x, -1, p0, ncp_prior, gamma, "events");
        // printVector(ris1);
        // Apri il file per scrivere i changing points
        std::ofstream outfile(path_out);
        if (!outfile.is_open()) {
            std::cerr << "Impossibile aprire il file per la scrittura." << std::endl;
            return 1;
        }
        // Scrivi i changing points sul file
        for (const auto& point : change_points) {
            outfile << point << " ";
        }
        outfile.close();

    } catch (const std::invalid_argument& e) {
        std::cerr << e.what() << std::endl;
    }
    
    return 0;
}
