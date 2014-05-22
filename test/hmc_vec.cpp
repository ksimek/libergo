// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

/**
 * @file test/hmc_vec.cpp
 * Test of Hamiltonian Monte Carlo algorithm on a vector model.
 */

#include <ergo/hmc.h>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <functional>
#include <boost/random.hpp>
#include <boost/math/distributions/normal.hpp>

using namespace ergo;

typedef std::vector<double> vec_t;

static const double GAUSSIAN_MEAN = 2.0;
static const double GAUSSIAN_SDV = 2.0;
static const double ERROR_THRESHOLD = 0.05;
static const bool VERBOSE = true;
static const size_t NUM_DYNAMICS_STEPS = 100;
static const double MOMENTUM_ALPHA = 0.0;
static const size_t NUM_ITERATIONS = 50000;
static const size_t NUM_BURN_IN = 1000;
std::vector<double> STEP_SIZE(1, 0.1);

/** @brief  Compare two doubles upto a threshold. */
inline
bool fequal(double op1, double op2, double threshold)
{
    return fabs(op1 - op2) < threshold;
}

/** @brief  log-pdf of normal distribution. */
inline
double target_distribution(const vec_t& x)
{
    using namespace boost::math;

    static normal_distribution<> G(GAUSSIAN_MEAN, GAUSSIAN_SDV);
    return log(pdf(G, x[0]));
}

/** @brief  Compute the gradient of the normal pdf. */
vec_t gradient(const vec_t& x)
{
    double df = (GAUSSIAN_MEAN - x[0]) / (GAUSSIAN_SDV * GAUSSIAN_SDV);
    return std::vector<double>(1, df);
}

/** @brief  Main, baby! */
int main(int argc, char** argv)
{
    std::vector<double> samples(NUM_ITERATIONS);
    std::vector<double> densities(NUM_ITERATIONS);

    hmc_step<vec_t> step(
            target_distribution,
            gradient,
            STEP_SIZE,
            NUM_DYNAMICS_STEPS,
            MOMENTUM_ALPHA);

    vec_t cur_model(1, -10);
    double cur_target = target_distribution(cur_model);

    for(size_t i = 0; i < NUM_BURN_IN; ++i)
        step(cur_model, cur_target);

    size_t accepted_count = 0;
    double mean_accept_prob = 0;

    for(size_t i = 0; i < NUM_ITERATIONS; ++i)
    {
        step(cur_model, cur_target);
        accepted_count += step.accepted() ? 1 : 0;
        mean_accept_prob += step.acceptance_probability();
        samples[i] = cur_model[0];
        densities[i] = cur_target;
    }

    double mean = std::accumulate(samples.begin(), samples.end(), 0.0);
    mean /= NUM_ITERATIONS;

    // COMPUTE STANDARD DEV
    // subtract out the mean
    std::transform(
        samples.begin(),
        samples.end(),
        samples.begin(),
        std::bind2nd(std::minus<double>(), mean));

    // squared sum of all values
    double sdv = std::inner_product(
                            samples.begin(),
                            samples.end(),
                            samples.begin(),
                            0.0);

    sdv /= (NUM_ITERATIONS - 1);
    sdv = sqrt(sdv);

    double accept_rate = static_cast<double>(accepted_count) / NUM_ITERATIONS;
    mean_accept_prob /= NUM_ITERATIONS;

    if(VERBOSE)
    {
        std::cout << "mean: " << mean << " std dev: " << sdv << std::endl;
        std::cout << "accept rate: " << accepted_count << '/'
                  << NUM_ITERATIONS << " = " << accept_rate << std::endl;
        std::cout << "mean accept probability: " << mean_accept_prob
                  << std::endl;
    }

    bool success = true;

    if(!fequal(mean, GAUSSIAN_MEAN, ERROR_THRESHOLD))
    {
        std::cerr << "FAILED!  Means not equal" << std::endl;
        success = false;
    }

    if(!fequal(sdv, GAUSSIAN_SDV, ERROR_THRESHOLD))
    {
        std::cerr << "FAILED!  Variances not equal" << std::endl;
        success = false;
    }

    if(success)
    {
        std::cout << "All tests passed." << std::endl;
        return 0;
    }

    return 1;
}

