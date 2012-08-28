// vim: tabstop=4 shiftwidth=4 foldmethod=marker

#ifndef ERGO_RECORD_H
#define ERGO_RECORD_H

#include <map>
#include <string>
#include <iterator>
#include <iostream>
#include <limits>
#include <boost/variant.hpp>
#include <boost/io/ios_state.hpp>
#include <boost/optional.hpp>
//#include <boost/type_traits.hpp>
//#include <boost/static_assert.hpp>

namespace ergo {
 
/**
 * @struct  step_detail
 * @brief   Stores assorted details about the execution of a sampling step.
 *
 * This class acknowledges that each type of step is likely to have different 
 * types of details, (e.g. a Metropolis-hastings step would indicate whether 
 * it was accepted, while a Gibbs step would not).  Therefore, this
 * class provides a freeform generic "details map", which associates 
 * key string to values from one of several basic data types. 
 *
 * This class is intended for writing to a "Detail report" file, in which each 
 * line is the details from a sampling step.  When passed to a stream, this 
 * object is represented in the following space-delimited format:
 *
 *    <step-name> <step-type> <log-target> key1=value1 key2=value2 ...
 *
 * To ensure easy parsing, user should ensure step-name and step-type are 
 * always one token each (i.e. containing at least one character, no spaces).
 *
 * Usage:
 *  
 * All steps provided by the library have an associated XXX_detail_recorder
 * type that can be attached to the step to record step_detail's.
 * XXX_detail_recorder is usually wrapped with ostream_recorder_wrapper to
 * write all details to a file. User-defined steps may write their own
 * XXX_detail_recorder, or may use the library-provided default_detail_recorder
 * object if no additional details are needed.
 *
 * @sa  default_detail_recorder, mh_detail_recorder, hmc_detail_recorder
 */
struct step_detail
{
    typedef boost::variant<double, std::string, bool, int, size_t> detail_t;

    typedef std::map<std::string, detail_t > detail_map_t;

    /** @brief type of the executed step (e.g. mh, hmc, etc) */
    std::string type;

    /** @brief name of the executed step (e.g. my_mh_step_1) */
    std::string name;

    /** @brief log target density of the resulting sample */
    double log_target;

    /** @brief generic details about the executed step */
    detail_map_t details;
};

/** @brief  Stream a step_detail. */
inline
std::ostream& operator<<(std::ostream& ost, const step_detail& detail)
{
    static const char DELIM = ' ';

    // this will restore the "precision" state of ost after leaving scope
    boost::io::ios_flags_saver flag_saver( ost );

    // enable full precision printing of double-values
    ost.precision(std::numeric_limits<double>::digits10);

    ost << detail.name;
    ost << DELIM << detail.type;
    ost << DELIM << detail.log_target;

    typedef step_detail::detail_map_t::const_iterator iterator;

    for(iterator it = detail.details.begin(); it != detail.details.end(); ++it)
    {
        ost << DELIM << it->first << '=' << it->second;
    }

    return ost;
}

/**
 * @defgroup recorders Recorders
 * @{
 *
 * These recorder classes collect data from the most recent sampling 
 * iteration.  
 *
 * Several recorders pass the recorded data to a user-supplied iterator, 
 * allowing live streaming of data using an std::ostream_iterator, or saving
 * to a collection using std::inserter or std::back_inserter.  See syntax 
 * examples below..
 * 
 * In addition to these general-purpose recorders, most library-provided 
 * steps have one or more step-specific "detail recorders" named 
 * XXX_detail_recorder.
 *
 * Examples:
 *  
 *     // create the step
 *     mh_step<Model> step(...)
 *
 *     // write all posteriors to cout
 *     step.add_recorder(make_target_recorder(ostream_iterator<Model>(cout)));
 *
 *     // save all model samples to a vector
 *     std::vector<Model>& all_samples;
 *     step.add_recorder(make_sample_recorder(back_inserter(samples)));
 *
 *     // save all proposals to an std::list
 *     std::list<step_detail> all_proposals;
 *     step.add_recorder(make_mh_detail_recorder(back_inserter(all_proposals)));
 *
 */

/**
 * @class   default_detail_recorder
 * @brief   Records the step type, the step name and the value of the
 *          log-target distribution.
 *
 * This is a very basic "detail recorder". It records the only detail that is
 * common to all steps, the log-target density. Each step should have its
 * corresponding detail recorder which knows about the details of that
 * particular step and records them appropriately.
 */
template <class OutputIterator>
class default_detail_recorder
{
public:
    typedef step_detail record_type;

    /** @brief  Construct a recorder. */
    default_detail_recorder(
        const std::string& type,
        OutputIterator it
    ) :
        step_detail_(),
        it_(it)
    {
        //using std::iterator_traits;
        //typedef typename iterator_traits<OutputIterator>::value_type value_type;
        //BOOST_STATIC_ASSERT_MSG(
        //        typename boost::is_convertible<step_detail, value_type>::value, 
        //        "step_detail must be convertible to OutputIterator::value_type");

        step_detail_.type = type;
    }

    /** @brief  Records a step. */
    template <class Step, class Model>
    void operator()(const Step& step, const Model& /*model*/, double log_target)
    {
        step_detail_.log_target = log_target;
        step_detail_.name = step.name();
        *it_++ = step_detail_;
    }

private:
    step_detail step_detail_;
    OutputIterator it_;
};

/** @brief  Convenience function to create a default_detail_recorder. */
template <class OutputIterator>
inline
default_detail_recorder<OutputIterator> make_default_detail_recorder(
    const std::string& type,
    OutputIterator it
)
{
    return default_detail_recorder<OutputIterator>(type, it);
}

/**
 * @class   target_recorder
 * @brief   Records the value of the log-target distribution.
 */
template <class OutputIterator>
class target_recorder
{
public:
    typedef double record_type;

    /** @brief  Constructs a recorder. */
    target_recorder(OutputIterator it) : it_(it)
    { 
        //using std::iterator_traits;
        //typedef typename iterator_traits<OutputIterator>::value_type value_type;
        //BOOST_STATIC_ASSERT_MSG(
        //        boost::is_convertible<double, value_type>::value, 
        //        "double must be convertible to OutputIterator::value_type");
    }

    /** @brief  Records a step. */
    template <class Step, class Model>
    void operator()(const Step&, const Model&, double log_target)
    {
        *it_++ = log_target;
    }

private:
    OutputIterator it_;
};

/** @brief  Convenience function to create a target_recorder. */
template <class OutputIterator>
inline
target_recorder<OutputIterator> make_target_recorder(OutputIterator it)
{
    return target_recorder<OutputIterator>(it);
}

/**
 * @class   sample_recorder
 * @brief   Records the model sampled by the step.
 */
template <class Model, class OutputIterator>
class sample_recorder
{
public:
    typedef Model record_type;

    /** @brief  Construct a recorder. */
    sample_recorder(OutputIterator it) : it_(it)
    { 
        //using std::iterator_traits;
        //typedef typename iterator_traits<OutputIterator>::value_type value_type;
        //BOOST_STATIC_ASSERT_MSG(
        //        boost::is_convertible<Model, value_type>::value, 
        //        "Model must be convertible to Iterator::value_type");
    }

    /** @brief  Record a step. */
    template <class Step>
    void operator()(const Step&, const Model& model, double)
    {
        *it_++ = model;
    }

private:
    OutputIterator it_;
};

/** @brief  Convenience function to create a sample_recorder. */
template <class OutputIterator>
inline
sample_recorder<
    typename std::iterator_traits<OutputIterator>::value_type,
    OutputIterator>
make_sample_recorder(OutputIterator it)
{
    typedef typename std::iterator_traits<OutputIterator>::value_type Model;
    return sample_recorder<Model, OutputIterator>(it);
}

/**
 * @class   best_target_recorder
 * @brief   Records the best (so far) value of the log-target distribution.
 */
template <class OutputIterator>
class best_target_recorder
{
public:
    typedef double record_type;

    /** @brief  Constructs a recorder. */
    best_target_recorder(OutputIterator it) :
        it_(it), initialized_(false), increment_(true)
    {}

    /** @brief  Force replacing of recorded value every time. */
    best_target_recorder& replace()
    {
        increment_ = false;
        return *this;
    }

    /** @brief  Records a step. */
    template <class Step, class Model>
    void operator()(const Step&, const Model&, double log_target)
    {
        if(!initialized_ || log_target > best_log_target_)
        {
            *it_ = log_target;
            best_log_target_ = log_target;
        }
        else
        {
            *it_ = best_log_target_;
        }

        if(increment_) it_++;

        initialized_ = true;
    }

private:
    OutputIterator it_;
    double best_log_target_;
    bool initialized_;
    bool increment_;
};

/** @brief  Convenience function to create a best_target_recorder. */
template <class OutputIterator>
inline
best_target_recorder<OutputIterator> make_best_target_recorder(OutputIterator it)
{
    return best_target_recorder<OutputIterator>(it);
}

/**
 * @class   best_sample_recorder
 * @brief   Records the best (so far) model sampled by the step.
 */
template <class Model, class OutputIterator>
class best_sample_recorder
{
public:
    typedef Model record_type;

    /** @brief  Construct a recorder. */
    best_sample_recorder(OutputIterator it) : it_(it), increment_(true)
    {}

    /** @brief  Force replacing of recorded value every time. */
    best_sample_recorder& replace()
    {
        increment_ = false;
        return *this;
    }

    /** @brief  Record a step. */
    template <class Step>
    void operator()(const Step&, const Model& model, double log_target)
    {
        if(!best_model_ || log_target > best_log_target_)
        {
            *it_ = model;
            best_log_target_ = log_target;

            if(!best_model_ || increment_) best_model_ = model;
        }
        else
        {
            if(increment_) *it_ = *best_model_;
        }

        if(increment_) it_++;
    }

private:
    OutputIterator it_;
    double best_log_target_;
    boost::optional<Model> best_model_;
    bool increment_;
};

/** @brief  Convenience function to create a best_sample_recorder. */
template <class OutputIterator>
inline
best_sample_recorder<
    typename std::iterator_traits<OutputIterator>::value_type,
    OutputIterator>
make_best_sample_recorder(OutputIterator it)
{
    typedef typename std::iterator_traits<OutputIterator>::value_type Model;
    return best_sample_recorder<Model, OutputIterator>(it);
}

/**
 * @class   proposed_recorder
 * @brief   Records the proposed model; can be used with any steps
 *          that have the proposed_model() member defined.
 */
template <class Model, class OutputIterator>
class proposed_recorder
{
public:
    typedef Model record_type;

    /** @brief  Construct a recorder. */
    proposed_recorder(OutputIterator it) : it_(it)
    {}

    /** @brief  Record a step. */
    template <class Step>
    void operator()(const Step& step, const Model&, double)
    {
        *it_++ = *step.proposed_model();
    }

private:
    OutputIterator it_;
};

/** @brief  Convenience function to create a best_sample_recorder. */
template <class OutputIterator>
inline
proposed_recorder<
    typename std::iterator_traits<OutputIterator>::value_type,
    OutputIterator>
make_proposed_recorder(OutputIterator it)
{
    typedef typename std::iterator_traits<OutputIterator>::value_type Model;
    return proposed_recorder<Model, OutputIterator>(it);
}

/** @} */

} // namespace ergo

#endif // ERGO_RECORD_H

