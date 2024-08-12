#include "fitnessfunc.hpp"
#include "utils.hpp"

#include <algorithm>
#include <vector>
#include <numeric>
#include <tuple>
#include <stdexcept>

std::tuple<std::vector<double>, 
           std::vector<double>, 
           double> FitnessFunc::validateInput(std::vector<double>& t, 
                                              std::vector<double>& x, 
                                              double sigma) {
    /*
    Validate inputs to the model.

    Parameters
    ----------
    t : array-like
        times of observations
    x : array-like, optional
        values observed at each time
    sigma : float, optional errors 
        in values x

    Returns
    -------
    t, x, sigma : array-like, float or None
        validated and perhaps modified versions of inputs
    */
    // find unique values of t
    std::vector<double> t_sorted;
    for (size_t i = 0; i < t.size(); ++i) {
        t_sorted.push_back(t[i]);
    }
    std::sort(t_sorted.begin(), t_sorted.end());
    std::vector<double> unq_t;
    std::vector<size_t> unq_ind;
    for (size_t i = 0; i < t_sorted.size(); ++i) {
        if (i == 0 || t_sorted[i] != t_sorted[i - 1]) {
            unq_t.push_back(t_sorted[i]);
            unq_ind.push_back(i);
        }
    }
    // if x is specified, then we need to simultaneously sort t and x
    std::vector<double> x_sorted(unq_ind.size());
    if (!x.empty()) {
        for (size_t i = 0; i < unq_ind.size(); ++i) {
            x_sorted[i] = x[unq_ind[i]];
        }
    }
    // verify the given sigma value
    if (sigma < 0) {
        sigma = 1.0;
    } 
    return std::make_tuple(unq_t, x_sorted, sigma);
}

std::vector<double> FitnessFunc::fitness(std::map<std::string, std::vector<double>>& kwd){
    throw "NotImplementedError(FitnessFunc::fitness)";
}

double FitnessFunc::p0Prior(int N) {
    /*
    Empirical prior, parametrized by the false alarm probability ``p0``.

    See eq. 21 in Scargle (2013).

    Note that there was an error in this equation in the original Scargle
    paper (the "log" was missing). The following corrected form is taken
    from https://arxiv.org/abs/1304.2818
    */
    return 4 - log(73.53 * p0 * pow(N, -0.478));
}

std::vector<std::string> FitnessFunc::fitnessArgs() {
    /*
    the fitness_args property will return the list of arguments accepted by
    the method fitness().  This allows more efficient computation below.
    */
    throw "NotImplementedError(FitnessFunc::fitness)";
}

double FitnessFunc::computeNcpPrior(int N) {
    /*
    If ``ncp_prior`` is not explicitly defined, compute it from ``gamma``
    or ``p0``.
    */
    if (gamma > 0) {
        return -log(gamma);
    } else if (p0 > 0) {
        return p0Prior(N);
    } else {
        throw std::invalid_argument("ncp_prior cannot be computed as neither gamma nor p0 is defined.");
    }
}

std::vector<double> FitnessFunc::fit(std::vector<double>& t, std::vector<double>& x, double sigma) {
    /*
    Fit the Bayesian Blocks model given the specified fitness function.

    Parameters
    ----------
    t : array-like
        data times (one dimensional, length N)
    x : array-like, optional
        data values
    sigma : array-like or float, optional
        data errors

    Returns
    -------
    edges : ndarray
        array containing the (M+1) edges defining the M optimal bins
    */
    std::vector<double> t_valid, x_valid;
    double sigma_valid;
    std::tie(t_valid, x_valid, sigma_valid) = validateInput(t, x, sigma);
    
    // compute values needed for computation, below
    std::vector<double> ak_raw, bk_raw, ck_raw;
    std::vector<std::string> fitness_args = fitnessArgs();
    // NOTE: this part is not used in Events
    if (std::find(fitness_args.begin(), fitness_args.end(), "a_k") != fitness_args.end()) {
        ak_raw.resize(x_valid.size());
        std::transform(x_valid.begin(), x_valid.end(), ak_raw.begin(), [sigma_valid](double xi) { return 1.0 / (sigma_valid * sigma_valid); });
    }
    if (std::find(fitness_args.begin(), fitness_args.end(), "b_k") != fitness_args.end()) {
        bk_raw.resize(x_valid.size());
        std::transform(x_valid.begin(), x_valid.end(), bk_raw.begin(), [sigma_valid](double xi) { return xi / (sigma_valid * sigma_valid); });
    }
    if (std::find(fitness_args.begin(), fitness_args.end(), "c_k") != fitness_args.end()) {
        ck_raw.resize(x_valid.size());
        std::transform(x_valid.begin(), x_valid.end(), ck_raw.begin(), [sigma_valid](double xi) { return (xi * xi) / (sigma_valid * sigma_valid); });
    }

    // create length-(N + 1) array of cell edges
    std::vector<double> edges(t_valid.size() + 1);
    edges[0] = t_valid[0];
    for (size_t i = 1; i < t_valid.size(); ++i) {
        edges[i] = 0.5 * (t_valid[i] + t_valid[i - 1]);
    }
    edges[t_valid.size()] = t_valid[t_valid.size()-1];

    std::vector<double> block_length(t_valid.size());
    for (size_t i = 0; i < t_valid.size(); ++i) {
        block_length[i] = t_valid.back() - edges[i];
    }

    // arrays to store the best configuration
    int N = t_valid.size();
    std::vector<float> best(N, 0.0);
    std::vector<int> last(N, 0);

    // Compute ncp_prior if not defined
    float ncp_prior_val = ncpPrior < 0 ? computeNcpPrior(N): ncpPrior;
    // NOTE: until here it is all ok

    // ----------------------------------------------------------------
    // Start with first data cell; add one cell at each iteration
    // ----------------------------------------------------------------
    for (int R = 0; R < N; ++R) {
        // Compute fit_vec : fitness of putative last block (end at R)
        std::map<std::string, std::vector<double>> kwds;
        // T_k: width/duration of each block
        if (std::find(fitness_args.begin(), fitness_args.end(), "T_k") != fitness_args.end()) {
            std::vector<double> T_k(R + 1);
            for (int i = 0; i <= R; ++i) {
                T_k[i] = block_length[i] - block_length[R + 1];
            }
            kwds["T_k"] = T_k;
        }
        // N_k: number of elements in each block
        if (std::find(fitness_args.begin(), fitness_args.end(), "N_k") != fitness_args.end()) {
            std::vector<double> N_k(R + 1);
            std::vector<double> x_valid_cut(R + 1);
            for (size_t i = 0; i < R+1; i++){
                x_valid_cut[i] = x_valid[i];
            }
            partial_sum(x_valid_cut.rbegin(), x_valid_cut.rbegin() + R + 1, N_k.rbegin());
            kwds["N_k"] = N_k;
        }
        // a_k: eq. 31
        if (std::find(fitness_args.begin(), fitness_args.end(), "a_k") != fitness_args.end()) {
            // FIXME: stessa cosa del caso di N_k istruzione vera: kwds["a_k"] = 0.5 * np.cumsum(ak_raw[: (R + 1)][::-1])[::-1]
            std::vector<double> a_k(R + 1);
            std::partial_sum(ak_raw.rbegin(), ak_raw.rbegin() + R + 1, a_k.rbegin());
            std::transform(a_k.begin(), a_k.end(), a_k.begin(), [](double val) { return 0.5 * val; });
            kwds["a_k"] = a_k;
        }
        // b_k: eq. 32
        if (std::find(fitness_args.begin(), fitness_args.end(), "b_k") != fitness_args.end()) {
            // FIXME: stessa cosa del caso di N_k istruzione vera: kwds["b_k"] = -np.cumsum(bk_raw[: (R + 1)][::-1])[::-1]
            std::vector<double> b_k(R + 1);
            std::partial_sum(bk_raw.rbegin(), bk_raw.rbegin() + R + 1, b_k.rbegin());
            std::transform(b_k.begin(), b_k.end(), b_k.begin(), [](double val) { return -val; });
            kwds["b_k"] = b_k;
        }
        // c_k: eq. 33
        if (std::find(fitness_args.begin(), fitness_args.end(), "c_k") != fitness_args.end()) {
            // FIXME: stessa cosa del caso di N_k istruzione vera: kwds["c_k"] = 0.5 * np.cumsum(ck_raw[: (R + 1)][::-1])[::-1]
            std::vector<double> c_k(R + 1);
            std::partial_sum(ck_raw.rbegin(), ck_raw.rbegin() + R + 1, c_k.rbegin());
            std::transform(c_k.begin(), c_k.end(), c_k.begin(), [](double val) { return 0.5 * val; });
            kwds["c_k"] = c_k;
        }
        // evaluate fitness function
        std::vector<double> fit_vec = fitness(kwds);

        std::vector<double> A_R(R + 1);
        std::transform(fit_vec.begin(), fit_vec.end(), A_R.begin(), [ncp_prior_val](double fit) { return fit - ncp_prior_val; });

        for(size_t i = 1; i < fit_vec.size(); i++){
            A_R[i] += best[i-1];
        }

        // if (R > 0) {
        //     std::transform(A_R.begin() + 1, A_R.end(), best.begin(), A_R.begin() + 1, std::plus<double>());
        // }

        auto max_it = std::max_element(A_R.begin(), A_R.end());
        last[R] = std::distance(A_R.begin(), max_it);
        best[R] = *max_it;
    }
    // printVector(last);

        // ----------------------------------------------------------------
    // Now find changepoints by iteratively peeling off the last block
    // ---------------------------------------------------------------
//     std::vector<int> change_points(N, 0);
//     int i_cp = N;
//     int ind = N;
//     while (i_cp > 0) {
//         i_cp--;
//         change_points[i_cp] = ind;
//         if (ind == 0) 
//             break;
//         ind = last[ind - 1];
//     }
//     if (i_cp == 0) 
//         change_points[i_cp] = 0;

//     change_points = std::vector<int>(change_points.begin() + i_cp, change_points.end());
    int index = last[N-1];
    std::vector<int> change_points;
    while(index > 0){
        change_points.insert(change_points.begin(), index);
        index = last[index - 1];
    }
            
    std::vector<double> result_edges(change_points.size());
    for (size_t i = 0; i < change_points.size(); ++i) {
        result_edges[i] = edges[change_points[i]];
    }

    return result_edges;
}
