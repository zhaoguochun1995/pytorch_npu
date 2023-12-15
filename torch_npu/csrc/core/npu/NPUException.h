#pragma once

#include <iostream>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <third_party/acl/inc/acl/acl_base.h>
#include "torch_npu/csrc/core/npu/NPUErrorCodes.h"
#include "torch_npu/csrc/core/npu/npu_log.h"
#include "torch_npu/csrc/core/npu/NPUMacros.h"
#include "torch_npu/csrc/core/npu/interface/AclInterface.h"

#define C10_NPU_SHOW_ERR_MSG()                                     \
do {                                                               \
  std::cout<< c10_npu::c10_npu_get_error_message() << std::endl;   \
} while (0)

#define NPU_CHECK_ERROR(err_code)                                    \
  do {                                                               \
    auto Error = err_code;                                           \
    static c10_npu::acl::AclErrorCode err_map;                       \
    if ((Error) != ACL_ERROR_NONE) {                                 \
      TORCH_CHECK(                                                   \
        false,                                                       \
        __func__,                                                    \
        ":",                                                         \
        __FILE__,                                                    \
        ":",                                                         \
        __LINE__,                                                    \
        " NPU error, error code is ", Error,                         \
        (err_map.error_code_map.find(Error) !=                       \
        err_map.error_code_map.end() ?                               \
        "\n[Error]: " + err_map.error_code_map[Error] : "."),        \
        "\n", c10_npu::c10_npu_get_error_message());                 \
    }                                                                \
  } while (0)

#define NPU_CHECK_SUPPORTED_OR_ERROR(err_code)                         \
  do {                                                                 \
    auto Error = err_code;                                             \
    static c10_npu::acl::AclErrorCode err_map;                         \
    if ((Error) != ACL_ERROR_NONE) {                                   \
      if ((Error) == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {               \
        static auto feature_not_support_warn_once = []() {             \
          NPU_LOGW(Feature is not supported and the possible cause is  \
                    that drivers and firmware packages do not match.); \
          return true;                                                 \
        }();                                                           \
      } else {                                                         \
        TORCH_CHECK(                                                   \
          false,                                                       \
          __func__,                                                    \
          ":",                                                         \
          __FILE__,                                                    \
          ":",                                                         \
          __LINE__,                                                    \
          " NPU error, error code is ", Error,                         \
          (err_map.error_code_map.find(Error) !=                       \
          err_map.error_code_map.end() ?                               \
          "\n[Error]: " + err_map.error_code_map[Error] : "."),        \
          "\n", c10_npu::c10_npu_get_error_message());                 \
      }                                                                \
    }                                                                  \
  } while (0)

#define NPU_CHECK_WARN(err_code)                                     \
  do {                                                               \
    auto Error = err_code;                                           \
    static c10_npu::acl::AclErrorCode err_map;                       \
    if ((Error) != ACL_ERROR_NONE) {                                 \
      TORCH_WARN("NPU warning, error code is ", Error,               \
        "[Error]: ",                                                 \
        (err_map.error_code_map.find(Error) !=                       \
        err_map.error_code_map.end() ?                               \
        "\n[Error]: " + err_map.error_code_map[Error] : "."),        \
        "\n", c10_npu::c10_npu_get_error_message());                 \
    }                                                                \
  } while (0)

void warn_(const ::c10::Warning& warning);

#define TORCH_NPU_WARN(...)                                  \
  warn_(::c10::Warning(                                       \
      ::c10::UserWarning(),                                  \
      {__func__, __FILE__, static_cast<uint32_t>(__LINE__)}, \
      ::c10::str(__VA_ARGS__),                               \
      false));

#define TORCH_NPU_WARN_ONCE(...)                                          \
  C10_UNUSED static const auto C10_ANONYMOUS_VARIABLE(TORCH_NPU_WARN_ONCE_) = \
      [&] {                                                               \
        TORCH_NPU_WARN(__VA_ARGS__);                                      \
        return true;                                                      \
      }()

namespace c10_npu {

C10_NPU_API const char *c10_npu_get_error_message();

} // namespace c10_npu
