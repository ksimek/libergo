// vim: tabstop=4 shiftwidth=4 foldmethod=marker

#ifndef ERGO_HMC_H
#define ERGO_HMC_H

#include <ergo/util.h>
#include <ergo/exception.h>
#include <ergo/rand.h>
#include <ergo/record.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <string>
#include <boost/function.hpp>
#include <boost/optional.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/uniform.hpp>
#include <boost/bind.hpp>

namespace ergo {

/**
 * @class hmc_step
 *
 * hmc_step is a functor that runs a single iteration of the hybrid monte carlo
 * algorithm.
 *
 * Boolean template parameters control whether certain optimizations are on or
 * off.
 *
 * @tparam Model            The model type.
 * @tparam ACCEPT_STEP      Do a accept/reject step? If this is true 
 *                          this becomes a true HMC step.  If false,
 *                          several evaluations of the target distribtuion
 *                          can be eliminated, but convergence to the 
 *                          target distribution is not guaranteed.
 * @tparam REVERSIBLE       Whether or not it's important for the leapfrog steps
 *                          to be reversible.  For true MCMC, this should be 
 *                          true, but passing false allows for fewer gradient 
 *                          evaluations, in some configurations (i.e. if we
 *                          accept_step = false, and alpha > 0).
 */
template<typename Model, bool ACCEPT_STEP = true, bool REVERSIBLE = true>
class hmc_step
{
public:
    // typedefs
    typedef std::vector<double> vec_t;
    typedef boost::mt19937 rng_t;
    typedef boost::function1<double, const Model&> evaluate_t;
    typedef boost::function1<vec_t, const Model&> gradient_t;
    typedef boost::function3<void, const hmc_step&, const Model&, double>
            record_t;

public:

    /** @brief  Document!!! */
    template <class VectorAdapter, class Evaluate, class Gradient>
    hmc_step(
        const VectorAdapter& adapter,
        const Evaluate& log_target, 
        const Gradient& gradient,
        const vec_t& step_sizes,
        int num_dynamics_steps,
        double alpha = 0.0
    ) :
        get_(),
        set_(),
        size_(),
        log_target_(log_target),
        gradient_(gradient),
        step_sizes_(step_sizes),
        num_dynamics_steps_(num_dynamics_steps),
        alpha_(alpha),
        p_(),
        first_p_full_(is_first_p_full_(alpha, ACCEPT_STEP, REVERSIBLE)),
        last_p_ignore_(is_last_p_ignored_(alpha, ACCEPT_STEP, REVERSIBLE)),
        temperature_(1.0),
        name_("generic-hmc-step"),
        lower_bounds_(),
        upper_bounds_(),
        uni_dist_(0, 1),
        uni_rand_(rng<rng_t>(), uni_dist_),
        norm_dist_(0, 1),
        norm_rand_(rng<rng_t>(), norm_dist_),
        store_proposed_(false)
    {
        boost::shared_ptr<VectorAdapter> a_p(new VectorAdapter(adapter));
        get_ = boost::bind<double>(&VectorAdapter::get, a_p, _1, _2);
        set_ = boost::bind<void>(&VectorAdapter::set, a_p, _1, _2, _3);
        size_ = boost::bind<size_t>(&VectorAdapter::size, a_p, _1);
    }

    /** @brief  Document!!! */
    template <class Evaluate, class Gradient>
    hmc_step(
        const Evaluate& log_target, 
        const Gradient& gradient,
        const vec_t& step_sizes,
        int num_dynamics_steps,
        double alpha = 0.0
    ) :  
        get_(boost::bind<double>(vector_get, _1, _2)),
        set_(boost::bind<void>(vector_set, _1, _2, _3)),
        size_(boost::bind<size_t>(&Model::size, _1)),
        log_target_(log_target),
        gradient_(gradient),
        step_sizes_(step_sizes),
        num_dynamics_steps_(num_dynamics_steps),
        alpha_(alpha),
        p_(),
        first_p_full_(is_first_p_full_(alpha, ACCEPT_STEP, REVERSIBLE)),
        last_p_ignore_(is_last_p_ignored_(alpha, ACCEPT_STEP, REVERSIBLE)),
        temperature_(1.0),
        name_("generic-hmc-step"),
        lower_bounds_(),
        upper_bounds_(),
        uni_dist_(0, 1),
        uni_rand_(rng<rng_t>(), uni_dist_),
        norm_dist_(0, 1),
        norm_rand_(rng<rng_t>(), norm_dist_),
        store_proposed_(false)
    {}

    /** 
     * @brief Runs a step of Hybrid Monte Carlo (HMC) on a model m.  
     *
     * If the step is rejected, m remains unchanged; if the step is 
     * accepted, m will hold the new state.  
     *
     * @note This method uses swap(Model&, Model&); for best 
     * performance, swap() should be specialized for Model.
     */
    void operator()(Model& m, double& lt_m) const;

    /**
     * @brief   Potential energy (i.e. negative log target density) of initial
     *          model in previous step.
     */
    double initial_potential() const { return U_; }

    /**
     * @brief   Potential energy (i.e. negative log target density) of final
     *          model in previous step
     */
    double final_potential() const { return U_star_; }

    /**
     * @brief   Kinetic energy (i.e. squared magnitude of momentum) of initial
     *          model in previous step
     */
    double initial_kinetic() const { return K_; }

    /**
     * @brief   Kinetic energy (i.e. squared magnitude of momentum) of final
     *          model in previous step
     */
    double final_kinetic() const { return K_star_; }

    /** @brief  Acceptance probability of previous step */
    double acceptance_probability() const { return accept_prob_; }

    /**
     * @brief   HMC proposed model (i.e., model after dynamics).
     *
     * This function returns an optional to the last model proposed by this
     * step, regardless of acceptance. Note that, if step is not instructed
     * to save proposed models (via store_proposed()), or if the step has not
     * been executed yet, this function returns boost::none.
     */
    const boost::optional<Model>& proposed_model() const
    {
        if(!store_proposed_)
        {
            static boost::optional<Model> empty_model;
            return empty_model;
        }
    
        return proposed_model_;
    }

    /** @brief  Was previous step accepted? */
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

    /** @brief  Sets the lower bounds for variables of the model. */
    void set_lower_bounds(const vec_t& lower_bounds)
    {
        lower_bounds_ = lower_bounds;
    }

    /** @brief  Sets the upper bounds for variables of the model. */
    void set_upper_bounds(const vec_t& upper_bounds)
    {
        upper_bounds_ = upper_bounds;
    }

    /** @brief  Add a recorder to this step. */
    template <class Recorder>
    void add_recorder(const Recorder& rec) { recorders_.push_back(rec); }

private:
    /**
     * @brief   Determine whether the full leapfrog step is a
     *          full-step (otherwise half-step)
     */
    bool is_first_p_full_(
            const bool alpha,
            const bool accept_step,
            const bool reversible)
    {
        // We'd like to avoid the final momentum update, if possible.  It 
        // isn't avoidable if we have an accept step.  If we discard
        // the accept step, we have two options:
        // There are two ways to skip the final momentum update:
        // (a) If the momentum is completely replaced in every step
        //     (i.e. alpha == 0), the final momentum doesn't matter, so 
        //     we can discard it without sacrificing anything.
        // (b) If we perform partial replacement of momenta (alpha > 0), 
        //     we can't discard the momentum update, so we'll roll it into 
        //     the first momentum update, sacrificing reversibility.
        //
        // If you use partial momentum updates, and reversibility must be
        // maintined, no optimizations may be used.  
        //
        // We can summarize this logic in a truth table, which we build below.
        // 
        // Inputs are encoded as follows:
        //     A = 1: accept/reject step is enabled
        //     A = 0: accept/reject step is skipped
        // 
        //     R = 1: reversibility must be maintained
        //     R = 0: reversibilty may be sacrificed.
        //
        //     P = 1: partial momenta updates are performed
        //     P = 0: momenta are fully replaced at each iteration
        //
        // Outputs:
        //     f = 0: first momenta update is a half-update
        //     f = 1:   "     "       "     "   full-update
        //     l = 0: last momemnta update is a half-update
        //     l = 1:  "     "        "     "   ignored
        //
        //  In other words, f=0 and l=0 means "use standard mcmc".  Any
        //  other combination of outputs implies an optimization is used.
        //
        //    A  R  P  ||  f  l    comments
        //   -------------------------------
        //    1  x  x  ||  0  0     true HMC
        //    0  0  0  ||  0  1     still reversible
        //    0  1  0  ||  0  1     
        //    0  0  1  ||  1  1     not reversible
        //    0  1  1  ||  0  0     
        //
        // This results in the following boolean expressions:
        //
        //    f = !A  ^ !(R ^ P)
        //    l = !A ^ !R ^ P
        //
        // Here, 'v' means "or",  '^' means "and".

        // Which of these methods to use 
        const bool partial_updates = (alpha > 0.0);
        return !accept_step && !reversible && partial_updates;
    }

    /**
     * @brief   Determine whether to ignore the last momentum
     *          leapfrog step (otherwise take a final half-step).
     */
    bool is_last_p_ignored_(
            const bool alpha,
            const bool accept_step,
            const bool reversible)
    {
        const bool partial_updates = (alpha > 0.0);
        // there's a truth table under is_first_p_full that explains this 
        // boolean logic.
        return !accept_step && !(reversible && partial_updates);
    }

protected:
    /** @brief  Function object that gets the i-th element of the model. */
    boost::function2<double, const Model*, size_t> get_;

    /** @brief  Function object that sets the i-th element of the model. */
    boost::function3<void, Model*, size_t, double> set_;

    /** @brief  Function object that gets the size of the model. */
    boost::function1<size_t, const Model*> size_;

    /** @brief  Function object that computes the (log) target density. */
    evaluate_t log_target_;

    /** @brief  Function object that computes the gradient. */
    gradient_t gradient_;

    /** @brief  Vector of step sizes for the leap-frog algorithm. */
    vec_t step_sizes_;

    /**
     * @brief   The number of leapfrog steps in the trajectory.
     *
     * @note    According to XXX, "choosing a suitable trajectory length
     *          is crucial if HMC is to explore the state space systematically,
     *          rather than by a random walk". "For a problem thought to be
     *          fairly difficult, a trajectory with m_length = 100 might be
     *          a suitable starting point". 
     */
    const int num_dynamics_steps_;

    /** @brief  Amount of stochastic update to apply to momentum. */
    double alpha_;

    /** @brief  Momentum vector. */
    mutable vec_t p_;

    /** @brief  Optimization constant. Set to false for true MCMC. */
    bool first_p_full_;

    /** @brief  Optimization constant. Set to false for true MCMC. */
    bool last_p_ignore_;

    /** @brief  Temperature is used for annealing. */
    double temperature_;

    /** @brief  Name is used for identifying this step. */
    std::string name_;

    /** @brief  State vector will never be smaller than this. */
    vec_t lower_bounds_;

    /** @brief  State vector will never be larger than this. */
    vec_t upper_bounds_;

    /** @brief  Uniform distribution -- used for generating uniform samples. */
    boost::uniform_real<> uni_dist_;

    /** @brief  Generates uniform random numbers. */
    mutable boost::variate_generator<rng_t&, boost::uniform_real<> > uni_rand_;

    /** @brief  Normal distribution -- used for generating normal samples. */
    boost::normal_distribution<> norm_dist_;

    /** @brief  Generates normal random numbers. */
    mutable
    boost::variate_generator<rng_t&, boost::normal_distribution<> > norm_rand_;

    /** @brief  Potential energy of model before simulating dynamics */
    mutable double U_;

    /** @brief  Potential energy of model after simulating dynamics */
    mutable double U_star_;

    /** @brief  Kinetic energy of model before simulating dynamics */
    mutable double K_;

    /** @brief  Kinetic energy of model after simulating dynamics */
    mutable double K_star_;

    /** @brief  Acceptance probabiliy of the model after simulating dynamics */
    mutable double accept_prob_;

    /** @brief  Whether or not the proposed model was accepted */
    mutable bool accepted_;

    /** @brief  Store proposed model here. */
    mutable boost::optional<Model> proposed_model_;

    /** @brief  Keep tabs on proposed model? */
    bool store_proposed_;

    /** @brief  Vector of recorders. */
    mutable std::vector<record_t> recorders_;
};

/* \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ */

template<class Model, bool ACCEPT_STEP, bool REVERSIBLE>
void hmc_step<Model, ACCEPT_STEP, REVERSIBLE>::operator()(
    Model& q,
    double& lt_q) const
{
    using namespace std;

    // ===============================================================
    // This variant of HMC is outlined in Horowitz (1991):
    //  "A generalized guided monte carlo algorithm."
    //  It allows us to choose between partial or complete stochastic 
    //  replacement of momenta, without changing the algorithm.  Neal's 
    //  version exhibits random walk when using "partial updates," so 
    //  we opted against it.
    // --------------------------------------------------------------- 
    
    const bool CONSTRAINED_TARGET = !lower_bounds_.empty();
    const size_t hmc_dim = size_(&q);

    // make sure dimension of p is the same as of the model
    if(p_.size() != hmc_dim)
    {
        p_.resize(hmc_dim);
        fill(p_.begin(), p_.end(), 0.0);
    }

    // Sample new momentum -- using a partial update.
    // Note: when alpha = 0 this algorithm becomes regular HMC;
    // however, it performs a bunch of unnecessary computations,
    // so when alpha = 0, we avoid them and generate the momentum
    // directly
    if(alpha_ == 0.0 && temperature_ == 1.0)
    {
        generate(p_.begin(), p_.end(), norm_rand_);
    }
    else
    {
        transform(
            p_.begin(),
            p_.end(),
            p_.begin(),
            bind1st(multiplies<double>(), alpha_));

        vec_t nvec(hmc_dim);
        generate(nvec.begin(), nvec.end(), norm_rand_);

        double stochastic_weight =
            sqrt(1 - alpha_ * alpha_) * sqrt(temperature_);
        
        transform(
            nvec.begin(),
            nvec.end(),
            nvec.begin(),
            bind1st(multiplies<double>(), stochastic_weight));

        transform(
            p_.begin(),
            p_.end(),
            nvec.begin(),
            p_.begin(),
            plus<double>());
    }

    // Compute grad_U (which is -grad(log_target))
    // and multiply times epsilon (grad_U never appears alone)
    vec_t grad_x_eps = gradient_(q);

    transform(
        grad_x_eps.begin(),
        grad_x_eps.end(),
        grad_x_eps.begin(),
        bind1st(multiplies<double>(), -1.0));

    transform(
        grad_x_eps.begin(),
        grad_x_eps.end(),
        step_sizes_.begin(),
        grad_x_eps.begin(),
        multiplies<double>());

    // ===============================================================
    //  LEAP-FROG algorithm
    // --------------------------------------------------------------- 

    // copy cur model to proposed model
    proposed_model_ = q;

    // q_star is a ref to the member propoosed_model_
    Model& q_star = *proposed_model_;

    vec_t p_star = p_;

    if(first_p_full_)
    {
        // OPTIMIZATION BRANCH
        // We perform a full-step of the momentum.
        //
        // The normal algorithm calls for a 1/2 step here, but 
        // we offer an optimization that makes this a full step,
        // and eliminates the final 1/2 step later on.  This saves
        // one gradient evaluation per iteration, which may be significant.
        // Note: using this optimization, the algorithm is no longer true
        // MCMC, because the step is not volume-preserving (and thus not
        // reversible)
        transform(
            p_star.begin(),
            p_star.end(),
            grad_x_eps.begin(),
            p_star.begin(),
            minus<double>());
    }
    else
    {
        // We perform a half-step of the momentum 
        // (as per the normal leapfrog algorithm).
        vec_t tmp_vec(grad_x_eps.size());
        transform(
            grad_x_eps.begin(),
            grad_x_eps.end(),
            tmp_vec.begin(),
            bind2nd(divides<double>(), 2.0));

        transform(
            p_star.begin(),
            p_star.end(),
            tmp_vec.begin(),
            p_star.begin(),
            minus<double>());
    }

    // Alternate full steps for position and momentum   
    vec_t etp(step_sizes_.size());

    for(int i = 0; i < num_dynamics_steps_; i++)
    {
        // Perform a full update of the parameters
        // First compute epsilon x p_star
        transform(
            step_sizes_.begin(),
            step_sizes_.end(),
            p_star.begin(),
            etp.begin(),
            multiplies<double>());

        for(size_t d = 0; d < hmc_dim; d++)
        {
            double mpb;
            if(!CONSTRAINED_TARGET)
            {
                mpb = etp[d];
            }
            else
            {
                if(lower_bounds_.size() != hmc_dim)
                {
                    throw dimension_mismatch(__FILE__, __LINE__);
                }

                // HANDLING CONSTRAINTS
                // We need to fix the position and momentum until they stop
                // violating constraints. See Neal for details.
                double q_d_p, q_d;
                do
                {
                    q_d = get_(&q_star, d);
                    q_d_p = q_d + etp[d];
                    if(q_d_p < lower_bounds_[d])
                    {
                        q_d_p = (2 * lower_bounds_[d] - q_d_p);
                        p_star[d] *= -1;
                    }

                    if(q_d_p > upper_bounds_[d])
                    {
                        q_d_p = (2 * upper_bounds_[d] - q_d_p);
                        p_star[d] *= -1;
                    }
                } while(q_d_p < lower_bounds_[d] || q_d_p > upper_bounds_[d]);

                mpb = q_d_p - q_d;
            }

            double tmp = get_(&q_star, d);
            set_(&q_star, d, tmp + mpb);
        }

        // if (last_iteration && don't care about final value of p)
        if(i == num_dynamics_steps_ - 1 && last_p_ignore_) 
        {
            /* do nothing */

            // OPTIMIZATION BRANCH
            // Don't bother performing the final gradient evaluation, because
            // either
            // (a) the final momentum will be discarded, or 
            // (b) the final half-update of momentum was rolled into a full
            //     initial momentum update.
            // In either case, this is no longer true MCMC, but could be
            // "close enough," and the benefits to running time may be worth it.
        }
        else
        {
            // update grad_U x epsilon.
            grad_x_eps = gradient_(q_star);

            transform(
                grad_x_eps.begin(),
                grad_x_eps.end(),
                grad_x_eps.begin(),
                bind1st(multiplies<double>(), -1));

            transform(
                grad_x_eps.begin(),
                grad_x_eps.end(),
                step_sizes_.begin(),
                grad_x_eps.begin(),
                multiplies<double>());
        }

        // Make a full step for the momentum, except at the end of the trajectory 
        if(i != num_dynamics_steps_ - 1)
        {
            transform(
                p_star.begin(),
                p_star.end(),
                grad_x_eps.begin(),
                p_star.begin(),
                minus<double>());
        }
    }

    if(last_p_ignore_)
    {
        /* Do nothing */
        // OPTIMIZATION BRANCH (see above)
    }
    else
    {
        // Make a half step for momentum at the end. 
        vec_t tmp_vec(grad_x_eps.size());
        transform(
            grad_x_eps.begin(),
            grad_x_eps.end(),
            tmp_vec.begin(),
            bind2nd(divides<double>(), 2.0));

        transform(
            p_star.begin(),
            p_star.end(),
            tmp_vec.begin(),
            p_star.begin(),
            minus<double>());
    }

    // Negate momentum at end of trajectory to make the proposal symmetric
    transform(
        p_star.begin(),
        p_star.end(),
        p_star.begin(),
        bind1st(multiplies<double>(), -1));


    // ===============================================================
    //  Accept step
    // --------------------------------------------------------------- 

    U_star_ = -log_target_(q_star);
    U_ = -lt_q;

    // TODO: may need to divide by step size here...
    K_star_ = inner_product(
                    p_star.begin(),
                    p_star.end(),
                    p_star.begin(),
                    0.0) / 2.0;

    K_ = inner_product(p_.begin(), p_.end(), p_.begin(), 0.0) / 2.0;

    if(ACCEPT_STEP)
    {
        // compute acceptance probability
        // not totall sure if dividing by temperature is valid here,
        // since we're mutliplying 
        // momentum by temperature earlier in the file.  Might be either/or.
        // ksimek, April 26, 2012
        accept_prob_ = (U_ - U_star_ + K_ - K_star_) / temperature_;

#ifdef TEST
        assert(accept_prob_ == accept_prob_);
        assert(accept_prob_ > -numeric_limits<double>::max());
#endif
    }
    else
    {
        // at least 0 ensures acceptance
        accept_prob_ = 0.0;
    }

    double u = log(uni_rand_());

    if(u < accept_prob_)
    {
        // update the position and momentum
        // Note: specialize swap to get best performance 
        using std::swap;
        swap(q, q_star);
        swap(p_, p_star);
        accepted_ = true;

        // update log(target)
        lt_q = -U_star_;

        // if asked to keep the proposed model, copy it into q_star (which is a
        // ref to proposed_model_)
        if(store_proposed_)
        {
            q_star = q;
        }
    }
    else
    {
        // Everything stays the same
        accepted_ = false;
    }

    // Negate momentum to avoid random walk.
    // true reversal of momentum only occurs when rejection occurs.
    // note: if alpha is not 0, this makes the step non-reversible
    transform(
        p_.begin(),
        p_.end(),
        p_.begin(),
        bind1st(multiplies<double>(), -1));

    // call recorders
    for_each(
        recorders_.begin(),
        recorders_.end(),
        boost::bind(&record_t::operator(), _1, *this, q, lt_q));
}

/**
 * @class   hmc_detail_recorder
 * @brief   Records details about a HMC step.
 */
template <class OutputIterator>
class hmc_detail_recorder
{
public:
    typedef step_detail record_type;

    hmc_detail_recorder(OutputIterator it) : it_(it) 
    {}

    template <class Model>
    void operator()(const hmc_step<Model>& step, const Model&, double log_target)
    {
        step_detail detail;
        detail.type = "hmc";
        detail.name = step.name();
        detail.log_target = log_target;
        detail.details["initial_potential"] = step.initial_potential();
        detail.details["final_potential"] = step.final_potential();
        detail.details["initial_kinetic"] = step.initial_kinetic();
        detail.details["final_kinetic"] = step.final_kinetic();
        detail.details["p_accept"] = step.acceptance_probability();
        detail.details["accepted"] = step.accepted();

        *it_++ = detail;
    }

private:
    OutputIterator it_;
};

/**
 * @brief   Convenience function to create a hmc_detail_recorder.
 */
template <class OutputIterator>
inline
hmc_detail_recorder<OutputIterator> make_hmc_detail_recorder(OutputIterator it)
{
    return hmc_detail_recorder<OutputIterator>(it);
}

} // namespace ergo

#endif // ERGO_HMC_H

