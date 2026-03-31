/**
 * Compatibility for building with Ceres and glog 0.7.x (conda).
 * Include glog correctly (export.h first), then make MakeCheckOpValueString
 * visible in google::logging:: so template instantiation from Ceres finds it.
 */
#ifndef VOLDOR_GLOG_CERES_COMPAT_H
#define VOLDOR_GLOG_CERES_COMPAT_H

#include <glog/export.h>
#include <glog/logging.h>

namespace google {
namespace logging {
using internal::MakeCheckOpValueString;
}
}

#endif
