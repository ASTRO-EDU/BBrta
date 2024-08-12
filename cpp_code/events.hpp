#ifndef EVENTS_H
#define EVENTS_H

#include "fitnessfunc.hpp"
#include <vector>
#include <tuple>
#include <cmath>
#include <stdexcept>

class Events : public FitnessFunc {
public:
    Events(double p0 = 0.05, double gamma = 0.0, double ncp_prior = 0.0);

    std::tuple<std::vector<double>, 
               std::vector<double>, 
               double> validateInput(std::vector<double>& t, std::vector<double>& x, double sigma);

    std::vector<std::string> fitnessArgs();

    std::vector<double> fitness(std::map<std::string, std::vector<double>>& kwd) override;
    
};

#endif // EVENTS_H
