#ifndef ERGO_MH_H
#define ERGO_MH_H

#include <ergo/rand.h>
#include <ergo/record.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <boost/function.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/optional.hpp>
#include <boost/bind.hpp>

/**
 * @file    mh_step.h
 * @brief   Contains classes representing Metropolis-Hastings steps.
 */

namespace ergo {

/**
 * @struct  mh_proposal_result
 * @brief   Contains information about a MH proposal.
 *
 * This class represents the result of proposing a value using a MH proposal
 * distribution. Any proposers used with mh_step should return this. It 
 * contains the forward and reverse densities/probabilities.
 */
struct mh_proposal_result
{
    double fwd;
    double rev;
    std::string name;

    mh_proposal_result(double fp, double rp, const std::string& nm = "") :
        fwd(fp), rev(rp), name(nm) {}
};

/**
 * @class   mh_step
 * @brief   Basic Metropolist-Hastings step.
 *
 * This class implements the canonical MH step logic. To use it, you must
 * have a target distribution and a proposal mechanism.
 */
template <class Model>
class mh_step
{
private:
    // typedefs
    typedef boost::mt19937 rng_t;
    typedef boost::function1<double, const Model&> evaluate_t;
    typedef boost::function2<mh_proposal_result, const Model&, Model&>
            propose_t;
    typedef boost::function3<void, const mh_step&, const Model&, double>
            record_t;

public:
    /**
     * @brief  Construct a MH step object.
     *
     * Construct a Metropolis-Hastings sampling step with the given target
     * distribution and proposal distribution.
     *
     * @tparam  Evaluate    A unary function type; receives a Model by
     *                      const-ref and returns a double.
     *
     * @tparam  Propose     The proposer type; must comply with MH proposal
     *                      concept.
     */
    template <class Evaluate, class Propose>
    mh_step(const Evaluate& log_target, const Propose& propose) :
        log_target_(log_target),
        propose_(propose),
        temperature_(1.0),
        name_("generic-mh-step"),
        store_proposed_(false),
        uni_dist_(0, 1),
        uni_rand(&rng<rng_t>(), uni_dist_)
    {}

    /**
     * @brief   Executes this step.
     *
     * Runs this step, which generates a new sample.
     *
     * @param m     The current state of the sampler. Note that this function
     *              will overwrite its value with the new sample.
     *
     * @param lt    The current value of the log-target. Note that this
     *              function will overwrite its value with the new value.
     */
    void operator()(Model& m, double& lt) const;

    /**
     * @brief   log probability density of the current model under the 
     *          target distribution. Applies to the previously-executed
     *          step only.
     */
    double current_target_density() const { return current_target_; }

    /**
     * @brief   log probability density of the proposed model under the 
     *          target distribution. Applies to the previously-executed
     *          step only.
     */
    double proposed_target_density() const { return proposed_target_; }

    /**
     * @brief   log probability density of the current model given the
     *          proposed model under the proposal distribution.  
     *          Applies to the previously-executed step only.
     */
    double reverse_proposal_density() const { return rev_proposal_prob_; }

    /**
     * @brief   log probability density of the proposed model given the
     *          current model under the proposal distribution.
     *          Applies to the previously-executed step only.
     */
    double forward_proposal_density() const { return fwd_proposal_prob_; }

    /**
     * @brief   Metropolis-hastings acceptance probability of the previous
     *          step.
     */
    double acceptance_probability() const { return accept_prob_; }

    /**
     * @brief   Metropolis-hastings proposed model.
     *
     * This function returns an optional to the last model proposed by this
     * step, regardless of acceptance. Note that, if step is not instructed
     * to save proposed models (via store_proposed()), or if the step has not
     * been executed yet, this function returns boost::none.
     */
    boost::optional<const Model&> proposed_model() const
    {
        if(store_proposed_)
        {
            return boost::optional<const Model&>(*proposed_model_);
        }

        return boost::none;
    }

    /** @brief  Was the previous step accepted? */
    bool accepted() const { return accepted_; }

    /** @brief  Set the temperature (for annealing). */
    const double& temperature() const { return temperature_; }

    /** @brief  Set the temperature (for annealing). */
    double& temperature() { return temperature_; }

    /** @brief  Returns the name of this step. */
    const std::string& name() const { return name_; }

    /** @brief  Returns the name of this step. */
    std::string& name() { return name_; }

    /** @brief  Toggle whether this step should store the proposed model. */
    void store_proposed(bool store = true) { store_proposed_ = store; }

    /** @brief  Add a recorder to this step. */
    template <class Recorder>
    void add_recorder(const Recorder& rec) { recorders_.push_back(rec); }

private:
    evaluate_t log_target_;
    propose_t propose_;

    mutable double accept_prob_;
    mutable double fwd_proposal_prob_;
    mutable double rev_proposal_prob_;
    mutable double current_target_;
    mutable double proposed_target_;
    mutable bool accepted_;

    double temperature_;
    mutable std::string name_;
    mutable boost::optional<Model> proposed_model_;
    bool store_proposed_;

    boost::uniform_real<> uni_dist_;
    mutable boost::variate_generator<rng_t*, boost::uniform_real<> > uni_rand;

    mutable std::vector<record_t> recorders_;
};

/* \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ */

template <class Model>
void mh_step<Model>::operator()(Model& m, double& log_target) const
{
    current_target_ = log_target;

    // Use member so that:
    // (a) no default construction is necessary
    // (b) the proposed object is only allocated once per the lifetime
    //     of the object
    if(!proposed_model_)
    {
        // copy constructor is called
        proposed_model_ = m;
    }

    // m_p is proposed model
    Model& m_p = *proposed_model_;

    // propose model and compute probabilities/densities
    mh_proposal_result p_res = propose_(m, m_p);
    fwd_proposal_prob_ = p_res.fwd;
    rev_proposal_prob_ = p_res.rev;

    // TODO: do we really want to overwrite this step's name?
    if(p_res.name != "")
    {
        name_ = p_res.name;
    }

    // get log-target distribution of the proposed model
    proposed_target_ = log_target_(m_p);

    // compute acceptance probability
    accept_prob_ = (proposed_target_ - current_target_
                    + rev_proposal_prob_ - fwd_proposal_prob_) / temperature_;

    // accept sample?
    double u = std::log(uni_rand());
    if(u < accept_prob_)
    {
        // Model type should specialize swap to get best performance 
        using std::swap;
        swap(m, m_p);
        log_target = proposed_target_;
        accepted_ = true;

        // if asked to keep the proposed model, copy it into m_p (which is a
        // ref to proposed_model_)
        if(store_proposed_)
        {
            m_p = m;
        }
    }
    else
    {
        accepted_ = false;

        // if(store_proposed_) do nothing; m_p already contains proposed model
    }

    // call recorders
    std::for_each(
        recorders_.begin(),
        recorders_.end(),
        boost::bind(&record_t::operator(), _1, *this, m, log_target));
}

/**
 * @class   mh_detail_recorder
 * @brief   Records details about a MH step.
 */
template <class OutputIterator>
class mh_detail_recorder
{
public:
    typedef step_detail record_type;

    mh_detail_recorder(OutputIterator it) : it_(it) 
    {}

    template <class Model>
    void operator()(const mh_step<Model>& step, const Model&, double log_target)
    {
        step_detail detail;
        detail.type = "mh";
        detail.name = step.name();
        detail.log_target = log_target;
        detail.details["cur_lt"] = step.current_target_density();
        detail.details["prop_lt"] = step.proposed_target_density();
        detail.details["fwd_q"] = step.forward_proposal_density();
        detail.details["rev_q"] = step.reverse_proposal_density();
        detail.details["p_accept"] = step.acceptance_probability();
        detail.details["accepted"] = step.accepted();

        *it_++ = detail;
    }

private:
    OutputIterator it_;
};

/**
 * @brief   Convenience function to create a mh_detail_recorder.
 */
template <class OutputIterator>
inline
mh_detail_recorder<OutputIterator> make_mh_detail_recorder(OutputIterator it)
{
    return mh_detail_recorder<OutputIterator>(it);
}

} // namespace ergo

#endif //ERGO_MH_H

