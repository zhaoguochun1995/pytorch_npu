// Copyright (c) 2020 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <c10/util/Exception.h>

#include "torch_npu/csrc/core/npu/register/FunctionLoader.h"
#include "torch_npu/csrc/framework/interface/AclInterface.h"

namespace at_npu {
namespace native {

#undef LOAD_FUNCTION
#define LOAD_FUNCTION(funcName) \
  REGISTER_FUNCTION(libascendcl, funcName)
#undef GET_FUNC
#define GET_FUNC(funcName)              \
  GET_FUNCTION(libascendcl, funcName)

REGISTER_LIBRARY(libascendcl)
LOAD_FUNCTION(aclGetRecentErrMsg)
LOAD_FUNCTION(aclrtCreateEventWithFlag)
LOAD_FUNCTION(aclrtQueryEventWaitStatus)
LOAD_FUNCTION(aclprofCreateStepInfo)
LOAD_FUNCTION(aclprofGetStepTimestamp)
LOAD_FUNCTION(aclprofDestroyStepInfo)
LOAD_FUNCTION(aclprofInit)
LOAD_FUNCTION(aclprofStart)
LOAD_FUNCTION(aclprofStop)
LOAD_FUNCTION(aclprofFinalize)
LOAD_FUNCTION(aclprofCreateConfig)
LOAD_FUNCTION(aclprofDestroyConfig)

aclprofStepInfoPtr init_stepinfo() {
  typedef aclprofStepInfoPtr(*npdInitFunc)();
  static npdInitFunc func = nullptr;
  if(func == nullptr) {
      func = (npdInitFunc)GET_FUNC(aclprofCreateStepInfo);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclprofCreateStepInfo");
  auto ret = func();
  return ret;
}

NpdStatus destroy_stepinfo(aclprofStepInfoPtr stepInfo) {
  typedef NpdStatus(*npdDestroyFunc)(aclprofStepInfoPtr);
  static npdDestroyFunc func = nullptr;
  if(func == nullptr) {
      func = (npdDestroyFunc)GET_FUNC(aclprofDestroyStepInfo);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclprofDestroyStepInfo");
  auto ret = func(stepInfo);
  return ret;
}

NpdStatus start_deliver_op(aclprofStepInfoPtr stepInfo, aclprofStepTag stepTag, aclrtStream stream) {
  typedef NpdStatus(*npdStartProfiling)(aclprofStepInfoPtr, aclprofStepTag, aclrtStream);
  static npdStartProfiling func = nullptr;
  if(func == nullptr) {
      func = (npdStartProfiling)GET_FUNC(aclprofGetStepTimestamp);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclprofGetStepTimestamp");
  auto ret = func(stepInfo, stepTag, stream);
  return ret;
}

NpdStatus stop_deliver_op(aclprofStepInfoPtr stepInfo, aclprofStepTag stepTag, aclrtStream stream) {
  typedef NpdStatus(*npdStopProfiling)(aclprofStepInfoPtr, aclprofStepTag, aclrtStream);
  static npdStopProfiling func = nullptr;
  if(func == nullptr) {
      func = (npdStopProfiling)GET_FUNC(aclprofGetStepTimestamp);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclprofGetStepTimestamp");
  auto ret = func(stepInfo, stepTag, stream);
  return ret;
}

const char *AclGetErrMsg()
{
  typedef const char *(*aclGetErrMsg)();
  static aclGetErrMsg func = nullptr;
  if (func == nullptr) {
    func = (aclGetErrMsg)GET_FUNC(aclGetRecentErrMsg);
  }
  if (func != nullptr) {
    return func();
  }
  return "";
}

aclError AclrtCreateEventWithFlag(aclrtEvent *event, uint32_t flag) {
  typedef aclError(*AclrtCreateEventWithFlagFunc)(aclrtEvent*, uint32_t);
  static AclrtCreateEventWithFlagFunc func = nullptr;
  if (func == nullptr) {
    func = (AclrtCreateEventWithFlagFunc)GET_FUNC(aclrtCreateEventWithFlag);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclrtCreateEventWithFlag");
  return func(event, flag);
}

aclError AclProfilingInit(const char *profilerResultPath, size_t length) {
  typedef aclError (*AclProfInitFunc) (const char *, size_t);
  static AclProfInitFunc func = nullptr;
  if (func == nullptr) {
    func = (AclProfInitFunc)GET_FUNC(aclprofInit);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclprofInit");
  return func(profilerResultPath, length);
}

aclError AclProfilingStart(const aclprofConfig *profilerConfig) {
  typedef aclError (*AclProfStartFunc) (const aclprofConfig *);
  static AclProfStartFunc func = nullptr;
  if (func == nullptr) {
    func = (AclProfStartFunc)GET_FUNC(aclprofStart);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclprofStart");
  return func(profilerConfig);
}

aclError AclProfilingStop(const aclprofConfig *profilerConfig) {
  typedef aclError (*AclProfStopFunc) (const aclprofConfig*);
  static AclProfStopFunc func = nullptr;
  if (func == nullptr) {
    func = (AclProfStopFunc)GET_FUNC(aclprofStop);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclprofStop");
  return func(profilerConfig);
}

aclError AclProfilingFinalize() {
  typedef aclError (*AclProfFinalizeFunc) ();
  static AclProfFinalizeFunc func = nullptr;
  if (func == nullptr) {
    func = (AclProfFinalizeFunc)GET_FUNC(aclprofFinalize);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclprofFinalize");
  return func();
}

aclprofConfig *AclProfilingCreateConfig(
    uint32_t *deviceIdList,
    uint32_t deviceNums,
    aclprofAicoreMetrics aicoreMetrics,
    aclprofAicoreEvents *aicoreEvents,
    uint64_t dataTypeConfig) {
  typedef aclprofConfig *(*AclProfCreateConfigFunc) \
    (uint32_t *, uint32_t, aclprofAicoreMetrics, aclprofAicoreEvents *, uint64_t);
  static AclProfCreateConfigFunc func = nullptr;
  if (func == nullptr) {
    func = (AclProfCreateConfigFunc)GET_FUNC(aclprofCreateConfig);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclprofCreateConfig");
  return func(deviceIdList, deviceNums, aicoreMetrics, aicoreEvents, dataTypeConfig);
}

aclError AclProfilingDestroyConfig(const aclprofConfig *profilerConfig) {
  typedef aclError (*AclProfDestroyConfigFunc) (const aclprofConfig *);
  static AclProfDestroyConfigFunc func = nullptr;
  if (func == nullptr) {
    func = (AclProfDestroyConfigFunc)GET_FUNC(aclprofDestroyConfig);
  }
  TORCH_CHECK(func, "Failed to find function ", "aclprofDestroyConfig");
  return func(profilerConfig);
}

#undef LOAD_ASCEND_DUMP_FUNCTION
#define LOAD_ASCEND_DUMP_FUNCTION(funcName) \
  REGISTER_FUNCTION(libascend_dump, funcName)

#undef GET_ASCEND_DUMP_FUNC
#define GET_ASCEND_DUMP_FUNC(funcName) \
  GET_FUNCTION(libascend_dump, funcName)

REGISTER_LIBRARY(libascend_dump)
LOAD_ASCEND_DUMP_FUNCTION(aclopStartDumpArgs)
LOAD_ASCEND_DUMP_FUNCTION(aclopStopDumpArgs)

aclError AclopStartDumpArgs(uint32_t dumpType, const char *path) {
  typedef aclError(*AclopStartDumpArgsFunc)(uint32_t, const char *);
    static AclopStartDumpArgsFunc func = nullptr;
    if (func == nullptr) {
        func = (AclopStartDumpArgsFunc)GET_ASCEND_DUMP_FUNC(aclopStartDumpArgs);
        if (func == nullptr) {
            return ACL_ERROR_FEATURE_UNSUPPORTED;
        }
    }
    TORCH_CHECK(func, "Failed to find function ", "aclopStartDumpArgs");
    return func(dumpType, path);
}

aclError AclopStopDumpArgs(uint32_t dumpType) {
  typedef aclError(*AclopStopDumpArgsFunc)(uint32_t);
    static AclopStopDumpArgsFunc func = nullptr;
    if (func == nullptr) {
        func = (AclopStopDumpArgsFunc)GET_ASCEND_DUMP_FUNC(aclopStopDumpArgs);
        if (func == nullptr) {
            return ACL_ERROR_FEATURE_UNSUPPORTED;
        }
    }
    TORCH_CHECK(func, "Failed to find function ", "aclopStopDumpArgs");
    return func(dumpType);
}

} // namespace native
} // namespace at_npu
