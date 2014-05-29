#ifndef ERGO_MH_H
#define ERGO_MH_H

#include <ergo/rand.h>
#include <ergo/record.h>
#include <ergo/util.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <boost/function.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/optional.hpp>
#include <boost/bind.hpp>
#include <boost/ref.hpp>

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
template <class Model, class Rng = default_rng_t>
class mh_step
{
public:
    // typedefs
    typedef boost::function1<double, const Model&> evaluate_t;
    typedef boost::function2<mh_proposal_result, const Model&, Model&>
            propose_t;
    typedef boost::function3<void, const mh_step&, const Model&, double>
            record_t;
    typedef Rng rng_t;

public:
    /**
     * @brief   Construct a MH step object.
     *
     * Construct a Metropolis-Hastings sampling step with the given target
     * distribution, proposal distribution, and random number generator.
     * The RNG passed is not copied, but a reference to it is held inside this
     * object. If copy is desired, use constructor without boost::ref. If
     * no RNG is provided, this function will use a global one which is
     * shared amongst all objects.
     *
     * @tparam  Evaluate    A unary function type; receives a Model by
     *                      const-ref and returns a double.
     *
     * @tparam  Propose     The proposer type; must comply with MH proposal
     *                      concept.
     */
//    template <class Evaluate, class Propose>
//    mh_step
//    (
//        const Evaluate& log_target,
//        const Propose& propose,
//        boost::reference_wrapper<rng_t> rngr = boost::ref(global_rng<rng_t>())
//    ) :
//        log_target_(log_target),
//        propose_(propose),
//        temperature_(1.0),
//        name_("generic-mh-step"),
//        store_proposed_(false),
//        p_res_(0.0, 0.0),
//        rng_(rngr.get()),
//        uni_dist_(&rng_)
//    {}

    /**
     * @brief  Construct a MH step object.
     *
     * Construct a Metropolis-Hastings sampling step with the given target
     * distribution, proposal distribution, and random number generator.
     * The RNG will be copied to the mh_step object; if reference semantics
     * are desired, use constructor with boost::ref.
     *
     * @param rng           A random number generator object.  Can receive
     *                      const T&, T*, or shared_ptr<T>, which chooses
     *                      copy, external ownershp, or shared ownership semantics,
     *                      respectively.
     *
     * @tparam  Evaluate    A unary function type; receives a Model by
     *                      const-ref and returns a double.
     *
     * @tparam  Propose     The proposer type; must comply with MH proposal
     *                      concept.
     */
    template <class Evaluate, class Propose>
    mh_step
    (
        const Evaluate& log_target,
        const Propose& propose,
        copy_or_ref<rng_t> rng = &global_rng<rng_t>()
    ) :
        log_target_(log_target),
        propose_(propose),
        temperature_(1.0),
        name_("generic-mh-step"),
        store_proposed_(false),
        p_res_(0.0, 0.0),
        rng_ptr_(rng.get()),
        uni_dist_(rng_ptr_.get())
    {}

    /** @brief  Set the temperature (for annealing). */
    double temperature() const { return temperature_; }

    /** @brief  Set the temperature (for annealing). */
    void set_temperature(double temp) { temperature_ = temp; }

    /** @brief  Returns the name of this step. */
    const std::string& name() const { return name_; }

    /** @brief  Returns the name of this step. */
    void rename(const std::string& name) { name_ = name; }

    /** @brief  Toggle whether this step should store the proposed model. */
    void store_proposed(bool store = true) { store_proposed_ = store; }

    /** @brief  Add a recorder to this step. */
    template <class Recorder>
    void add_recorder(const Recorder& rec) { recorders_.push_back(rec); }

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
    double current_target_density() const { return cur_target_; }

    /**
     * @brief   log probability density of the proposed model under the
     *          target distribution. Applies to the previously-executed
     *          step only.
     */
    double proposed_target_density() const { return prop_target_; }

    /**
     * @brief   Proposal result of latest call to step.
     */
    const mh_proposal_result& proposal_result() const { return p_res_; }

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

private:
    evaluate_t log_target_;
    propose_t propose_;
    double temperature_;
    std::string name_;
    bool store_proposed_;

    mutable double accept_prob_;
    mutable double cur_target_;
    mutable double prop_target_;
    mutable mh_proposal_result p_res_;
    mutable bool accepted_;

    mutable boost::optional<Model> proposed_model_;
//
//    rng_t rng_own_;
//    rng_t& rng_;
    
    shared_ptr<rng_t> rng_ptr_;
    mutable uniform_rand<rng_t> uni_dist_;

    mutable std::vector<record_t> recorders_;
};

/* \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ */

template <class Model, class Rng>
void mh_step<Model, Rng>::operator()(Model& m, double& log_target) const
{
    cur_target_ = log_target;

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
    p_res_ = propose_(m, m_p);
    double fwd = p_res_.fwd;
    double rev = p_res_.rev;

    // get log-target distribution of the proposed model
    prop_target_ = log_target_(m_p);

    // compute acceptance probability
    accept_prob_ = (prop_target_ - cur_target_ + rev - fwd) / temperature_;

    // accept sample?
    double u = std::log(uni_dist_());
    if(u < accept_prob_)
    {
        // Model type should specialize swap to get best performance
        using std::swap;
        swap(m, m_p);
        log_target = prop_target_;
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

    template <class Model, class Rng>
    void operator()
    (
        const mh_step<Model, Rng>& step,
        const Model&,
        double log_target
    )
    {
        step_detail detail;
        const mh_proposal_result& pres = step.proposal_result();

        detail.type = "mh";
        detail.name = step.name() + ":" + pres.name;
        detail.log_target = log_target;
        detail.details["cur_lt"] = step.current_target_density();
        detail.details["prop_lt"] = step.proposed_target_density();
        detail.details["fwd_q"] = pres.fwd;
        detail.details["rev_q"] = pres.rev;
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

