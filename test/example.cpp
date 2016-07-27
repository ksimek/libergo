#include <ergo/mh.h>
#include <ergo/record.h>
#include <ergo/rand.h>
#include <boost/bind.hpp>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>

using namespace std;
using namespace ergo;

const double w1 = 0.1;
const double w2 = 0.9;
const double mu1 = -1.0;
const double mu2 = 0.5;
const double s1 = 0.5;
const double s2 = 3.0;

default_rng_t rng;
normal_rand<default_rng_t> nrnd(&rng);

/** @brief  Compare two doubles upto a threshold. */
inline
bool fequal(double op1, double op2, double threshold)
{
    return fabs(op1 - op2) < threshold;
}

/** @brief  Compute mix of two Gaussians log pdf. */
double mix_of_gauss(const double& x)
{
    return log(
            w1*(1/sqrt(2*s1*M_PI))*exp(-(x-mu1)*(x-mu1)/(2*s1)) +
            w2*(1/sqrt(2*s2*M_PI))*exp(-(x-mu2)*(x-mu2)/(2*s2)));
}

/** @brief  Random-walk Gaussian proposal. */
mh_proposal_result propose(const double& x, double& xp)
{
    double u = nrnd();
    xp = x + u;

    return mh_proposal_result(0.0, 0.0, "uniform_proposer");
}

/** @brief  Main function. */
int main(int argc, char** argv)
{
    // create MH step
    mh_step<double> step(mix_of_gauss, propose);
    step.store_proposed(true);

    double cur_x = 0.0;
    double cur_lt = mix_of_gauss(cur_x);

    // prepare recorder objects
    const size_t N = 500000;
    vector<double> samples(N);
    vector<double> targets(N);
    double bsample;
    double btarget;

    typedef mh_step<double>::record_t rec_t;
    vector<rec_t> recs;

    recs.push_back(make_sample_recorder(samples.begin()));
    recs.push_back(make_target_recorder(targets.begin()));
    recs.push_back(make_best_sample_recorder(&bsample).replace());
    recs.push_back(make_best_target_recorder(&btarget).replace());

    // run sampler
    for(size_t i = 1; i <= N; ++i)
    {
        step(cur_x, cur_lt);

        for_each(
            recs.begin(),
            recs.end(),
            boost::bind(&rec_t::operator(), _1, step, cur_x, cur_lt));
    }

    // compare mean and variance
    double mt = w1*mu1 + w2*mu2;
    double st = w1*s1 + w2*s2 + w1*mu1*mu1 + w2*mu2*mu2 - mt*mt;

    double mi = accumulate(samples.begin(), samples.end(), 0.0)/samples.size();

    vector<double> temp(samples.size());
    transform(
        samples.begin(),
        samples.end(),
        temp.begin(),
        boost::bind(minus<double>(), _1, mi));
    transform(
        temp.begin(),
        temp.end(),
        temp.begin(),
        boost::bind(multiplies<double>(), _1, _1));
    double si = accumulate(temp.begin(), temp.end(), 0.0)/(temp.size() - 1);

    const double err_thresh = 1e-2;
    bool success = true;

    if(!fequal(mt, mi, err_thresh))
    {
        std::cerr << "FAILED!  Means not equal" << std::endl;
        success = false;
    }

    if(!fequal(sqrt(st), sqrt(si), err_thresh))
    {
        std::cerr << "FAILED!  Std devs not equal" << std::endl;
        success = false;
    }

    if(success)
    {
        std::cout << "All tests passed." << std::endl;
        return 0;
    }

    return 1;
}

