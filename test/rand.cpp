/**
 * @file    test/rand.cpp
 * Test of RNG strategy used in MH step.
 */

#include <ergo/def.h>
#include <ergo/mh.h>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <functional>
#include <iterator>
#include <boost/random.hpp>
#include <boost/math/distributions/normal.hpp>

using namespace ergo;

typedef double Real;
typedef boost::mt19937 rng_t;

static const double GAUSSIAN_MEAN = 0.0;
static const double GAUSSIAN_SDV = 1.0;
static const bool VERBOSE = true;
static const size_t NUM_ITERATIONS = 20;

/** @brief  Compare two doubles upto a threshold. */
inline
bool fequal(double op1, double op2, double threshold)
{
    return fabs(op1 - op2) < threshold;
}

/** @brief  log-pdf of normal distribution. */
inline
double log_target(const Real& x)
{
    using namespace boost::math;

    static boost::math::normal_distribution<> G(GAUSSIAN_MEAN, GAUSSIAN_SDV);
    return log(pdf(G, x));
}

/** @brief  Deterministic proposer to test RNG. */
inline
mh_proposal_result propose(const Real& in, Real& out)
{
    out = in;
    out += 0.1;
    return mh_proposal_result(0.0, 0.0);
}

void test_uniform_rand(rng_t& rng)
{
    ergo::uniform_rand<rng_t> rand(&rng);

    double x = rand();

    // check for NaN bug
    assert(!(x != x));
#ifdef HAVE_Cxx11
    assert(!std::isnan(x));
#endif
}

void test_normal_rand(rng_t& rng)
{
    ergo::normal_rand<rng_t> rand(&rng, 0, 1);

    double x = rand();

    // check for NaN bug
    assert(!(x != x));
#ifdef HAVE_Cxx11
    assert(!std::isnan(x));
#endif
}

/** @brief  Main, baby! */
int main(int argc, char** argv)
{
    rng_t rng;
    rng_t rng2;
    rng2.seed(12345);

    test_uniform_rand(rng);
    test_normal_rand(rng);

    // these steps share the global RNG
    mh_step<Real, rng_t> step0(log_target, propose);
    mh_step<Real, rng_t> step1(log_target, propose);

    // these steps share the same local RNG
    mh_step<Real, rng_t> step2(log_target, propose, boost::ref(rng));
    mh_step<Real, rng_t> step3(log_target, propose, boost::ref(rng));

    // this step owns its own RNG
    mh_step<Real, rng_t> step4(log_target, propose, rng2);
    mh_step<Real, rng_t> step5(log_target, propose, rng2);

    // prepare for running and recording
    Real cur_models[6];
    double cur_targets[6];
    std::vector<Real> samples[6];

    std::fill(cur_models, cur_models + 6, -10.0);
    std::transform(cur_models, cur_models + 6, cur_targets, log_target);
    std::fill(samples, samples + 6, std::vector<Real>(NUM_ITERATIONS));

    // run steps
    for(size_t i = 0; i < NUM_ITERATIONS; ++i)
    {
        step0(cur_models[0], cur_targets[0]);
        step1(cur_models[1], cur_targets[1]);
        step2(cur_models[2], cur_targets[2]);
        step3(cur_models[3], cur_targets[3]);
        step4(cur_models[4], cur_targets[4]);
        step5(cur_models[5], cur_targets[5]);

        for(size_t j = 0; j < 6; j++)
        {
            samples[j][i] = cur_models[j];
        }
    }

    // start testing results
    bool success = true;

    if(!equal(samples[0].begin(), samples[0].end(), samples[2].begin()))
    {
        std::cerr << "FAILED! Shared steps not equal." << std::endl;
        success = false;
    }

    if(!equal(samples[1].begin(), samples[1].end(), samples[3].begin()))
    {
        std::cerr << "FAILED! Shared steps not equal." << std::endl;
        success = false;
    }

    if(!equal(samples[4].begin(), samples[4].end(), samples[5].begin()))
    {
        std::cerr << "FAILED! Copy steps not equal." << std::endl;
        success = false;
    }

    if(VERBOSE)
    {
        for(size_t j = 0; j < 6; j++)
        {
            std::copy(
                samples[j].begin(),
                samples[j].end(),
                std::ostream_iterator<Real>(std::cout, " "));
            std::cout << std::endl;
        }
    }

    if(success)
    {
        std::cout << "All tests passed." << std::endl;
        return 0;
    }

    return 1;
}

