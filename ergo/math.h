#ifndef ERGO_MATH_H
#define ERGO_MATH_H

#include <vector>
#include <cmath>

/**
 * @file    math.h
 * @brief   This file contains all math functionality needed to sample.
 */

namespace ergo {

/**
 * @brief   Gets the squared magnitude of a vector.
 */
inline
double magnitude_squared(const std::vector<double>& v)
{
    return std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
}

/**
 * @brief   Gets the magnitude of a vector.
 */
inline
double magnitude(const std::vector<double>& v)
{
    return std::sqrt(magnitude_squared(v));
}

} // namespace ergo

#endif //ERGO_MATH_H

