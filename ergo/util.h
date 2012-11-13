#ifndef ERGO_UTIL_H
#define ERGO_UTIL_H

#include <vector>

/**
 * @file    util.h
 * @brief   File for utility classes and functions.
 */

namespace ergo {

/**
 * @brief   Helper function that modifies an element of a vector.
 */
template <class VectorType>
inline
double vector_get(const VectorType* v, size_t i)
{
    return (*v)[i];
}

/**
 * @brief   Helper function that modifies an element of a vector.
 */
template <class VectorType>
inline
void vector_set(VectorType* v, size_t i, double x)
{
    (*v)[i] = x;
}

} // namespace ergo

#endif //ERGO_UTIL_H

