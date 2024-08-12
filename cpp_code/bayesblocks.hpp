#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

#include "fitnessfunc.hpp"
#include "events.hpp"

// Funzione bayesian_blocks equivalente
std::vector<double> bayesian_blocks(std::vector<double>& t, std::vector<double>& x, 
                                    double sigma = -1.0, double p0 = -1.0, double ncp_prior = -1.0,
                                    double gamma = -1.0,
                                    std::string fitness = "events") {
    // Mappatura dei nomi fitness alle classi C++ corrispondenti
    std::map<std::string, std::string> FITNESS_DICT = {
        {"events", "Events"},
        {"regular_events", "RegularEvents"},
        {"measures", "PointMeasures"},
    };

    // Ottenere il tipo di fitness corrispondente dal dizionario
    std::string fitnessClassName = FITNESS_DICT[fitness];

    // Gestione dinamica della classe FitnessFunc
    FitnessFunc* fitfunc = nullptr;

    // Verifica se il tipo di fitness è valido
    if (fitnessClassName == "Events") {
        fitfunc = new Events(p0, gamma, ncp_prior);
    } 
    // else if (fitnessClassName == "RegularEvents") {
    //     fitfunc = new RegularEvents();
    // } 
    // else if (fitnessClassName == "PointMeasures") {
    //     fitfunc = new PointMeasures();
    // } 
    else {
        throw std::invalid_argument("Fitness parameter not understood");
    }

    // Chiamare il metodo fit sulla classe selezionata
    std::vector<double> result = fitfunc->fit(t, x, sigma);

    // Liberare la memoria allocata per fitfunc
    delete fitfunc;

    return result; 
}