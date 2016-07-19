#include <ergo/mh.h>
#include <cmath>

using namespace std;
using namespace ergo;

double mix_of_gauss(const double& x)
{
    const double w1 = 0.1;
    const double w2 = 0.9;
    const double mu1 = -1;
    const double mu2 = 0.5;

    return log(w1 * exp(-(x-mu1)*(x-mu1)) + w2 * exp(-(x-mu2)*(x-mu2)));
}

mh_proposal_result propose(const double& x, double& xp)
{
    double u = 0.5 - std::rand();
    xp = x + u;

    return mh_proposal_result(0.0, 0.0, "uniform_proposer");
}

int main(int argc, char** argv)
{
    mh_step<double> step(mix_of_gauss, propose);
    step.store_proposed(true);

    double cur_x = 0.0;
    double cur_lt = mix_of_gauss(cur_x);

    vector<double> samples(1000);
    vector<double> targets(1000);
    double bsample;

    typedef mh_step<double>::record_t rec_t;
    std::vector<rec_t> recs;

    recs.push_back(make_sample_recorder(samples.begin()));
    recs.push_back(make_target_recorder(targets.begin()));
    recs.push_back(make_best_sample_recorder(&bsample).replace());
    recs.push_back(make_best_target_recorder(&btarget).replace());
    for(size_t i = 1; i <= 1000; ++i)
    {
        step(cur_x, cur_lt);

        std::for_each(
            recs.begin(),
            recs.end(),
            boost::bind(&rec_t::operator(), _1, *this, cur_x, cur_lt));
    }
}

