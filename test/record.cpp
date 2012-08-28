// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

/**
 * @file test/record.cpp
 * Test of recorders for the Hamiltonian Monte Carlo algorithm.
 */

#define TEST

#include <ergo/hmc.h>
#include <ergo/record.h>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <functional>
#include <boost/random.hpp>
#include <boost/math/distributions/normal.hpp>

using namespace ergo;

typedef double Real;

static const double GAUSSIAN_MEAN = 0.0;
static const double GAUSSIAN_VARIANCE = 1.0;
static const double ERROR_THRESHOLD = 0.005;
static const bool VERBOSE = true;
static const size_t NUM_DYNAMICS_STEPS = 100;
static const double MOMENTUM_ALPHA = 0.0;
static const size_t NUM_ITERATIONS = 10000;
std::vector<double> STEP_SIZE(1, 0.1);

/** @brief  log-pdf of normal distribution. */
inline
double target_distribution(const Real& x)
{
    static boost::math::normal_distribution<> G(GAUSSIAN_MEAN,
                                                GAUSSIAN_VARIANCE);
    return log(boost::math::pdf(G, x));
}

/** @brief  Compute the gradient of the normal pdf. */
std::vector<double> gradient(const Real& x)
{
    std::vector<double> out(1);
    out[0] = -x;

    return out;
}

/** @brief  Adapt a double into a VectorModel. */
struct Real_vector_adapter
{
    double get(const Real* x, size_t i) const
    {
        return *x;
    }

    void set(Real* x, size_t i, double value) const
    {
        *x = value;
    }

    size_t size(const Real* x) const
    {
        return 1;
    }
};

/** @brief  Main, baby! */
int main(int argc, char** argv)
{
    std::vector<Real> samples_true(NUM_ITERATIONS);
    std::vector<double> densities_true(NUM_ITERATIONS);
    std::vector<Real> proposed_true(NUM_ITERATIONS);

    std::vector<Real> samples(NUM_ITERATIONS);
    std::vector<double> densities(NUM_ITERATIONS);
    std::vector<Real> proposed(NUM_ITERATIONS);
    std::vector<step_detail> details(NUM_ITERATIONS);
    std::vector<Real> best_samples(NUM_ITERATIONS);
    std::vector<double> best_densities(NUM_ITERATIONS);
    Real best_sample;
    double best_density;

    // create step
    hmc_step<Real> step(
            Real_vector_adapter(),
            target_distribution,
            gradient,
            STEP_SIZE,
            NUM_DYNAMICS_STEPS,
            MOMENTUM_ALPHA);

    // add recorders
    step.add_recorder(make_sample_recorder(samples.begin()));
    step.add_recorder(make_target_recorder(densities.begin()));
    step.add_recorder(make_proposed_recorder(proposed.begin()));
    step.add_recorder(make_hmc_detail_recorder(details.begin()));

    step.add_recorder(make_best_sample_recorder(best_samples.begin()));
    step.add_recorder(make_best_sample_recorder(&best_sample).replace());

    step.add_recorder(make_best_target_recorder(best_densities.begin()));
    step.add_recorder(make_best_target_recorder(&best_density).replace());

    // run sampler
    Real cur_model = -10;
    double cur_target = target_distribution(cur_model);
    step.store_proposed();

    for(size_t i = 0; i < NUM_ITERATIONS; ++i)
    {
        step(cur_model, cur_target);
        samples_true[i] = cur_model;
        densities_true[i] = cur_target;
        proposed_true[i] = *step.proposed_model();
    }

    bool success = true;

    // test sample recorder
    if(samples != samples_true)
    {
        std::cerr << "FAILED! Recorded samples are off." << std::endl;
        success = false;
    }

    // test density recorder
    if(densities != densities_true)
    {
        std::cerr << "FAILED! Recorded log-target values are off." << std::endl;
        success = false;
    }

    // test best density recorders
    typedef std::vector<double>::const_iterator vdci;
    vdci bd_p = std::max_element(densities_true.begin(), densities_true.end());
    size_t best_density_idx = bd_p - densities_true.begin();

    if(VERBOSE)
    {
        std::cout << "best recorded density: " << best_density << std::endl
                  << "last recorded 'best density': "
                  << best_densities.back() << std::endl
                  << "best found (true) density: "
                  << densities_true[best_density_idx] << std::endl;
    }

    if(best_density != densities_true[best_density_idx])
    {
        std::cerr << "FAILED! Best recorded density is off." << std::endl;
        success = false;
    }

    if(best_density != best_densities.back())
    {
        std::cerr << "FAILED! Best recorded densities are off." << std::endl;
        success = false;
    }

    // test best sample recorders
    if(VERBOSE)
    {
        std::cout << "best recorded sample: " << best_sample << std::endl
                  << "last recorded 'best sample': "
                  << best_samples.back() << std::endl
                  << "best found (true) sample: "
                  << samples_true[best_density_idx] << std::endl;
    }

    if(best_sample != samples_true[best_density_idx])
    {
        std::cerr << "FAILED! Best recorded sample is off." << std::endl;
        success = false;
    }

    if(best_sample != best_samples.back())
    {
        std::cerr << "FAILED! Best recorded samples are off." << std::endl;
        success = false;
    }

    // test proposed recorder
    if(proposed != proposed_true)
    {
        std::cerr << "FAILED! Recorded proposed models are off." << std::endl;
        success = false;
    }

    for(size_t i = 0; i < NUM_ITERATIONS; i++)
    {
        if(boost::get<bool>(details[i].details["accepted"]))
        {
            if(proposed[i] != samples[i])
            {
                std::cerr << "FAILED! Recorded proposed models do not match "
                          << "accepted samples when accepted." << std::endl;
                success = false;
            }
        }
    }

    if(success)
    {
        std::cout << "All tests passed." << std::endl;
        return 0;
    }

    return 1;
}

