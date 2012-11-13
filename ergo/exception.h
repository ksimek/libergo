/* $Id: exception.h 65 2012-08-27 16:50:44Z ernesto $ */

// vim: tabstop=4 shiftwidth=4 foldmethod=marker

#ifndef ERGO_EXCEPTION_H
#define ERGO_EXCEPTION_H

#include <stdexcept>

#include <sstream>

namespace ergo
{

/**
 * Exception for errors in which the dimension of two objects are not equal.
 */
class dimension_mismatch : public std::logic_error
{
    typedef std::logic_error base_t;

public:
    dimension_mismatch() :
        base_t(default_msg_())
    { }

    dimension_mismatch(const char* file, unsigned int line) :
        base_t(make_error_string_(default_msg_(), file, line))
    { }

    dimension_mismatch(
        const std::string& msg,
        const char* file,
        unsigned int line
    ) :
        base_t(make_error_string_(msg, file, line))
    { }
    
    virtual ~dimension_mismatch() throw() {}

private:
    static const char* default_msg_()
    {
        static const char* msg = "Dimension mismatch.";
        return msg;
    }

    static std::string make_error_string_(
                        const std::string& msg,
                        const char* file,
                        unsigned int line
    )
    {
        std::ostringstream oss;
        oss << msg << '(' << file << ':' << line << ')' << std::endl;

        return oss.str();
    }

}; // class dimension mismatch

} // namespace ergo

#endif //ERGO_EXCEPTION_H

