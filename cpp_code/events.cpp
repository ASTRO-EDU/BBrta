#include <algorithm>
#include <vector>
#include <numeric>
#include <tuple>
#include <stdexcept>

#include "events.hpp"
#include "fitnessfunc.hpp"

Events::Events(double p0, double gamma, double ncp_prior): FitnessFunc(p0, gamma, ncp_prior) {
    // std::cout << p0;
    // std::cout << gamma;
    // std::cout << ncp_prior;
}

std::tuple<std::vector<double>, 
           std::vector<double>, 
           double> Events::validateInput(std::vector<double>& t, 
                                         std::vector<double>& x, 
                                         double sigma) {
    // Validate input and get validated vectors
    std::vector<double> t_valid, x_valid;
    double sigma_valid;
    std::tie(t_valid, x_valid, sigma_valid) = FitnessFunc::validateInput(t, x, sigma);

    if (!x_valid.empty()) {
        for (double value : x_valid) {
            if (std::fmod(value, 1.0) != 0.0) {
                throw std::invalid_argument("x must be integer counts for fitness='events'");
            }
        }
    }

    return std::make_tuple(t_valid, x_valid, sigma_valid);
}

std::vector<std::string> Events::fitnessArgs() {
    return {"N_k", "T_k"};
}

std::vector<double> Events::fitness(std::map<std::string, std::vector<double>>& kwd) {
    std::vector<double> N_k = kwd["N_k"];
    std::vector<double> T_k = kwd["T_k"];
    // Eq. 19 from Scargle 2013
    if (N_k.size() != T_k.size()) {
        throw std::invalid_argument("N_k and T_k must have the same size");
    }
    std::vector<double> results;
    for (size_t i = 0; i < N_k.size(); ++i) {
        if (T_k[i] == 0) {
            throw std::invalid_argument("T_k elements must not be zero");
        }

        double result = N_k[i] * std::log(N_k[i] / T_k[i]);

        results.push_back(result);
    }
    return results;
}