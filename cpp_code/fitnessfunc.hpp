#ifndef FITNESS_FUNC_H
#define FITNESS_FUNC_H

#include <iostream>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <map>
#include <cmath>
#include <unordered_map>

class FitnessFunc {
public:
    FitnessFunc(double p0=0.05, double gamma=-1.0, double ncp_prior=-1.0)
        : p0(p0), gamma(gamma), ncpPrior(ncp_prior) {}

    virtual ~FitnessFunc() {}


    std::tuple<std::vector<double>, 
               std::vector<double>, 
               double> validateInput(std::vector<double>& t, 
                                     std::vector<double>& x, 
                                     double sigma);
    
    virtual double p0Prior(int N);
    
    virtual double computeNcpPrior(int N);
    
    virtual std::vector<std::string> fitnessArgs() = 0;

    virtual std::vector<double> fitness(std::map<std::string, std::vector<double>>& kwd) = 0;
    
    virtual std::vector<double> fit(std::vector<double>& t, std::vector<double>& x, double sigma);
    
protected:
    double p0;
    double gamma;
    double ncpPrior;
};

#endif // FITNESS_FUNC_H
