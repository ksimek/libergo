// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

/**
 * @file test/mh.cpp
 * Test of Metropolis-hastings sampling.
 */

#include <ergo/mh.h>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <functional>
#include <boost/random.hpp>
#include <boost/math/distributions/normal.hpp>

using namespace ergo;

typedef double Real;
typedef boost::mt19937 base_generator_type;

static const double GAUSSIAN_MEAN = 0.0;
static const double GAUSSIAN_SDV = 1.0;
static const double ERROR_THRESHOLD = 0.005;
static const bool VERBOSE = true;
static const size_t NUM_ITERATIONS = 1000000;
static const size_t NUM_BURN_IN = 5000;

base_generator_type base_rng;

/** @brief  Compare two doubles upto a threshold. */
inline
bool fequal(double op1, double op2, double threshold)
{
    return fabs(op1 - op2) < threshold;
}

/** @brief  Generate normal random number. */
inline
double nrand()
{   
    typedef boost::normal_distribution<> Distribution_type;

    Distribution_type ndist(0, 1);
    return ndist(base_rng);
}

/** @brief  log-pdf of normal distribution. */
inline
double target_distribution(const Real& x)
{
    using namespace boost::math;

    static normal_distribution<> G(GAUSSIAN_MEAN, GAUSSIAN_SDV);
    return log(pdf(G, x));
}

/** @brief  Random-walk Metropolis proposal. */
inline
mh_proposal_result propose(const Real& in, Real& out)
{
    out = in;
    out += nrand();

    // symmetric proposal, default contrstructor is okay
    return mh_proposal_result(0.0, 0.0);
}

/** @brief  Main, baby! */
int main(int argc, char** argv)
{

    std::vector<Real> samples(NUM_ITERATIONS);
    std::vector<Real> densities(NUM_ITERATIONS);

    mh_step<Real, base_generator_type> step(
                                        target_distribution,
                                        propose,
                                        base_rng);

    Real cur_model = -10;
    double cur_target = target_distribution(cur_model);

    for(size_t i = 0; i < NUM_BURN_IN; ++i)
        step(cur_model, cur_target);

    for(size_t i = 0; i < NUM_ITERATIONS; ++i)
    {
        step(cur_model, cur_target);
        samples[i] = cur_model;
        densities[i] = cur_target;
    }

    double mean = std::accumulate(samples.begin(), samples.end(), 0.0);
    mean /= NUM_ITERATIONS;

    // COMPUTE VARIANCE
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

    if(VERBOSE)
    {
        std::cout << "mean: " << mean << " std dev: " << sdv << std::endl;
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

