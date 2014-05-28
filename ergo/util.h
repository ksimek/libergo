#ifndef ERGO_UTIL_H
#define ERGO_UTIL_H

#include <cstddef>
#include "ergo/def.h"
#include <boost/ref.hpp>

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

/**
 * Utility class for receiving parameters by copy, reference, or shared
 * ownership, depending on whether received by const-reference, pointer,
 * or shared_ptr. This allows ownership semantics to be decided by the
 * caller, rather than the receiving function. It does this by internally
 * storing as a shared_ptr, which the get() function returns.  For
 * external ownership, the shared_ptr's delete functionality is disabled to
 * avoid a double-free.
 *
 * It can be used to replace three overloads:
 *
 * \code{.cpp}
 *     void foo(const T& t); // copy (internal ownership)
 *     void foo(T* t); // reference (external ownership)
 *     void foo(shared_ptr<T> t); // reference (shared ownership)
 * \endcode
 *
 * with one signature
 *
 * \code{.cpp}
 *     void foo(copy_or_ref<T> t_in)
 *     {
 *          T& t = *t_in.get();
 *          t.do_stuff();
 *     }
 * \endcode
 *
 * without changing any calling code.
 */
template <class T>
class copy_or_ref
{
private:
    struct null_deleter
    {
        void operator()(void const *) const
        { }
    };

public:
    copy_or_ref(const T& obj) :
        obj_(new T(obj))
    { }

    copy_or_ref(T* obj) :
        obj_(obj, null_deleter())
    { }

    copy_or_ref(shared_ptr<T> obj) :
        obj_(obj)
    { }

    // @deprecated
    copy_or_ref(boost::reference_wrapper<T> obj) :
        obj_(obj.get_pointer(), null_deleter())
    { }

    /** @brief  Get underlying shared pointer representation. */
    shared_ptr<T> get() { return obj_; }

    /**
     * @brief   Get underlying shared pointer representation.
     * @note    Returning by const-value? Is this a mistake? --Ernesto
     */
    const shared_ptr<T> get() const { return obj_; }

private:
    shared_ptr<T> obj_;
};

} // namespace ergo

#endif //ERGO_UTIL_H

