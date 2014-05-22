#ifndef ERGO_EXCEPTION_H
#define ERGO_EXCEPTION_H

#include <stdexcept>
#include <sstream>
#include <string>
#include <iostream>

namespace ergo {

/**
 * Exception for errors in which the dimension of two objects are not equal.
 */
class dimension_mismatch : public std::logic_error
{
private:
    typedef std::logic_error base_t;

public:
    /** @brief  Constructor. */
    dimension_mismatch() :
        base_t(default_msg_())
    {}

    /** @brief  Constructor with file and line number. */
    dimension_mismatch(const char* file, unsigned int line) :
        base_t(make_error_string_(default_msg_(), file, line))
    {}

    /** @brief  Constructor with custom message, and file and line number. */
    dimension_mismatch
    (
        const std::string& msg,
        const char* file,
        unsigned int line
    ) :
        base_t(make_error_string_(msg, file, line))
    {}

    /** @brief  Destructor. */
    virtual ~dimension_mismatch() throw() {}

private:
    /** @brief  Default message. */
    static const char* default_msg_()
    {
        static const char* msg = "Dimension mismatch.";
        return msg;
    }

    /** @brief  Message with file and line number. */
    static std::string make_error_string_
    (
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

