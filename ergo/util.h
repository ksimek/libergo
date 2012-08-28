#ifndef ERGO_UTIL_H
#define ERGO_UTIL_H

#include <vector>
#include <iostream>

/**
 * @file    util.h
 * @brief   File for utility classes and functions.
 */

namespace ergo {

/**
 * @brief   Helper function that modifies an element of a vector.
 */
inline
double vector_get(const std::vector<double>* v, size_t i)
{
    return (*v)[i];
}

/**
 * @brief   Helper function that modifies an element of a vector.
 */
inline
void vector_set(std::vector<double>* v, size_t i, double x)
{
    (*v)[i] = x;
}

} // namespace ergo

#endif //ERGO_UTIL_H

