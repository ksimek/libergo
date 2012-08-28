#ifndef ERGO_RAND_H
#define ERGO_RAND_H

namespace ergo {

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
Rng& rng()
{
    static Rng r;
    return r;
}

} //namespace ergo

#endif //ERGO_RAND_H

