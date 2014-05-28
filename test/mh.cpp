// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4

/**
 * @file test/mh.cpp
 * Test of Metropolis-hastings sampling.
 */

#include <ergo/mh.h>
#include <ergo/def.h>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <functional>
#include <boost/random.hpp>
#include <boost/math/distributions/normal.hpp>

using namespace ergo;

typedef double Real;
typedef ergo::default_rng_t base_generator_t;
typedef mh_step<Real, base_generator_t> mh_step_real;

static const double GAUSSIAN_MEAN = 0.0;
static const double GAUSSIAN_SDV = 1.0;
static const double ERROR_THRESHOLD = 0.005;
static const bool VERBOSE = true;
static const size_t NUM_ITERATIONS = 1000000;
static const size_t NUM_BURN_IN = 5000;

base_generator_t base_rng;

/** @brief  Compare two doubles upto a threshold. */
inline
bool fequal(double op1, double op2, double threshold)
{
    return fabs(op1 - op2) < threshold;
}

ergo::normal_rand<base_generator_t> nrand(&base_rng, 0, 1);

/** @brief  log-pdf of normal distribution. */
inline
double target_distribution(const Real& x)
{
    static boost::math::normal_distribution<> G(GAUSSIAN_MEAN, GAUSSIAN_SDV);
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
//    base_rng.seed(std::time(0));
#if defined(HAVE_CXX11)
    std::cout << "using c++11 for rand" << std::endl;
#elif BOOST_VERSION >= 104700
    std::cout << "using modern boost for rand" << std::endl;
#else
    std::cout << "using old boost for rand" << std::endl;
#endif
    std::cout << __cplusplus << std::endl;
    std::cout << BOOST_VERSION << std::endl;

    std::vector<Real> samples;
    std::vector<Real> densities(NUM_ITERATIONS);
    size_t thinning = 1;

    samples.reserve(NUM_ITERATIONS);

    // test constructors 
    mh_step_real step_cpy(target_distribution, propose, base_rng);
    mh_step_real step_ptr(target_distribution, propose, &base_rng);
    mh_step_real step_smart_ptr(
                            target_distribution,
                            propose,
                            make_shared<base_generator_t>(base_rng));

    // copy constructor
    mh_step<Real, base_generator_t> step = step_ptr;

    // assignment
    step = step_ptr;

    Real cur_model = -10;
    double cur_target = target_distribution(cur_model);

    for(size_t i = 0; i < NUM_BURN_IN; ++i)
        step(cur_model, cur_target);

    for(size_t i = 0; i < NUM_ITERATIONS; ++i)
    {
        step(cur_model, cur_target);
        if(i % thinning == 0)
            samples.push_back(cur_model);
        densities[i] = cur_target;
    }

    double mean = std::accumulate(samples.begin(), samples.end(), 0.0);
    mean /= samples.size();

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

    sdv /= samples.size()-1;
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

