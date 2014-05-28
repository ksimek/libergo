#ifndef ERGO_RAND_H
#define ERGO_RAND_H

#include <ergo/def.h>
#include <boost/random/mersenne_twister.hpp>
#include <boost/version.hpp>

#ifdef HAVE_CXX11
#include <random>
#else 
#if BOOST_VERSION < 104700
#include <boost/random/variate_generator.hpp>
#endif
#include <boost/random/uniform_01.hpp>
#include <boost/random/normal_distribution.hpp>
#endif

namespace ergo {

#ifdef HAVE_CXX11
typedef std::mt19937 default_rng_t;
#else
typedef boost::mt19937 default_rng_t;
#endif

/**
 * @brief   Initializes and returns random number generator for sampling.
 *
 * The first time this is called, it creates a random number generator
 * of type Rng and returns it. Any subsequent calls to this will simply
 * return the previously-created random number generator.
 *
 * @ptype Rng   The random number generator type. Must comply with Boost's
 *              concept of the same.
 *
 * @return      A reference to the RNG of that type.
 */
template <class Rng>
inline
Rng& global_rng()
{
    static Rng r;

    return r;
}

inline default_rng_t& default_rng()
{
    return global_rng<default_rng_t>();
}


namespace detail
{
#if !defined(HAVE_CXX11) && BOOST_VERSION < 104700
    template <class Engine, class Dist>
    class legacy_distribution_wrapper
    {
        typedef Engine engine_t;
        typedef Dist dist_t;
        typedef typename boost::variate_generator<engine_t*, dist_t>
                variate_generator_t;

    public:
        legacy_distribution_wrapper(engine_t* rng, const dist_t& dist) :
            variate_generator_(rng, dist)
        { }

        double operator()()
        {
            variate_generator_.distribution().reset();
            return variate_generator_();
        }

    private:
        variate_generator_t variate_generator_;
    };
#endif

    template <class Engine, class Dist>
    class modern_distribution_wrapper
    {
        typedef Engine engine_t;
        typedef Dist dist_t;

    public:
        modern_distribution_wrapper(engine_t* rng, const dist_t& dist) :
            rng_(rng),
            dist_(dist)
        { }

        double operator()()
        {
            return dist_(*rng_);
        }

    private:
        engine_t* rng_;
        dist_t dist_;
    }; 

} // namespace detail

/**
 * A class that generates uniformly-distributed random numbers.
 *
 * This class exists to provide backward compatibility with old
 * implementations of boost and forward compatibility with C++11.
 */
template <class Engine>
class uniform_rand 
{
    typedef Engine engine_t;

#if defined(HAVE_CXX11)
    typedef std::uniform_real_distribution<> dist_t;
#elif BOOST_VERSION >= 104700
    typedef boost::random::uniform_01<> dist_t;
#else
    typedef boost::uniform_01<> dist_t;
#endif

public:
    uniform_rand(engine_t* rng) :
        wrapper_(rng, dist_t())
    {}

    double operator()()
    {
        return wrapper_();
    }

private:  
#if defined(HAVE_CXX11) || BOOST_VERSION >= 104700
    detail::modern_distribution_wrapper<engine_t, dist_t> wrapper_;
#else  // boost < 1.47
    detail::legacy_distribution_wrapper<engine_t, dist_t> wrapper_;
#endif
};

/**
 * A class that generates normally-distributed random numbers.
 *
 * This class exists to provide backward compatibility with old
 * implementations of boost and forward compatibility with C++11.
 */
template <class Engine>
class normal_rand 
{
    typedef Engine engine_t;

#if defined(HAVE_CXX11)
    typedef std::normal_distribution<> dist_t;
#elif BOOST_VERSION >= 104700
    typedef boost::random::normal_distribution<> dist_t;
#else
    typedef boost::normal_distribution<> dist_t;
#endif

public:
    normal_rand(engine_t* rng, double mean = 0.0, double stddev = 1.0) :
        wrapper_(rng, dist_t(mean, stddev))
    {}

    double operator()()
    {
        return wrapper_();
    }
private:  
#if defined(HAVE_CXX11) || BOOST_VERSION >= 104700
    detail::modern_distribution_wrapper<engine_t, dist_t> wrapper_;

#else  // boost < 1.47
    detail::legacy_distribution_wrapper<engine_t, dist_t> wrapper_;
#endif
};

} //namespace ergo

#endif //ERGO_RAND_H

