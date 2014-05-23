#ifndef ERGO_DEF_H
#define ERGO_DEF_H

#if __cplusplus > 199711L
#define HAVE_CXX11 1
#endif

#ifdef HAVE_CXX11
#include <memory>
namespace ergo 
{
    using std::shared_ptr;
    using std::make_shared;
}
#else
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
namespace ergo 
{
    using ::boost::shared_ptr;
    using ::boost::make_shared;
}
#endif

#endif // ERGO_DEF_H

