/* $Id: exception.h 65 2012-08-27 16:50:44Z ernesto $ */
/* {{{=========================================================================== *
   |
   |  Copyright (c) 1994-2012 by Kobus Barnard (author)
   |
   |  Personal and educational use of this code is granted, provided that this
   |  header is kept intact, and that the authorship is not misrepresented, that
   |  its use is acknowledged in publications, and relevant papers are cited.
   |
   |  For other use contact the author (kobus AT cs DOT arizona DOT edu).
   |
   |  Please note that the code in this file has not necessarily been adequately
   |  tested. Naturally, there is no guarantee of performance, support, or fitness
   |  for any particular task. Nonetheless, I am interested in hearing about
   |  problems that you encounter.
   |
   |  Author:  Kyle Simek
 * =========================================================================== }}}*/

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

