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

#include <string>

#include "torch_npu/csrc/core/npu/register/OptionRegister.h"
#include "torch_npu/csrc/core/npu/register/OptionsManager.h"

namespace c10_npu {
namespace option {

using namespace std;

bool OptionsManager::IsResumeModeEnable() {
  const static bool isResumeModeEnable = []() -> bool {
    int32_t enable = OptionsManager::GetBoolTypeOption("RESUME_MODE_ENABLE", 0);
    return enable != 0;
  }();
  return isResumeModeEnable;
}

bool OptionsManager::IsMultiStreamMemoryReuse() {
  const static bool hcclRealTimeMemoryReuse = []() -> bool {
    int32_t enable = OptionsManager::GetBoolTypeOption("MULTI_STREAM_MEMORY_REUSE", 0);
    return enable != 0;
  }();
  return hcclRealTimeMemoryReuse;
}

bool OptionsManager::CheckInfNanModeEnable() {
  const static bool checkInfNanModeEnable = []() -> bool {
    int32_t enable = OptionsManager::GetBoolTypeOption("INF_NAN_MODE_ENABLE", 0);
    return enable != 0;
  }();
  return checkInfNanModeEnable;
}

bool OptionsManager::CheckBlockingEnable() {
  const static bool checkBlockingEnable = []() -> bool {
    int32_t blocking_enable = OptionsManager::GetBoolTypeOption("ASCEND_LAUNCH_BLOCKING", 0);
    return blocking_enable != 0;
  }();
  return checkBlockingEnable;
}

bool OptionsManager::CheckQueueEnable() {
  if (CheckBlockingEnable()) {
    return false;
  }
  const static bool checkQueueEnable = []() -> bool {
    int32_t queue_enable = OptionsManager::GetBoolTypeOption("TASK_QUEUE_ENABLE", 1);
    return queue_enable != 0;
  }();
  return checkQueueEnable;
}

bool OptionsManager::CheckCombinedOptimizerEnable() {
  const static bool checkCombinedOptimizerEnable = []() -> bool {
    int32_t combined_optimize = OptionsManager::GetBoolTypeOption("COMBINED_ENABLE");
    return combined_optimize != 0;
  }();
  return checkCombinedOptimizerEnable;
}

bool OptionsManager::CheckAclDumpDateEnable() {
  const static bool checkAclDumpDateEnable = []() -> bool {
    int32_t acl_dump_data = OptionsManager::GetBoolTypeOption("ACL_DUMP_DATA");
    return acl_dump_data != 0;
  }();
  return checkAclDumpDateEnable;
}

bool OptionsManager::CheckDisableAclopComAndExe() {
  const static bool checkDisableAclopComAndExe = []() -> bool {
    int32_t disable_aclop_com_exe = OptionsManager::GetBoolTypeOption("DISABLE_ACLOP_COM_EXE");
    return disable_aclop_com_exe != 0;
  }();
  return checkDisableAclopComAndExe;
}

bool OptionsManager::CheckSwitchMMOutputEnable() {
  static int switchMMOutputEnable = -1;
  if (switchMMOutputEnable == -1) {
    switchMMOutputEnable = GetBoolTypeOption("SWITCH_MM_OUTPUT_ENABLE");
  }
  return (switchMMOutputEnable == 1);
}

int OptionsManager::GetBoolTypeOption(const char* env_str, int defaultVal) {
  char* env_val = std::getenv(env_str);
  int64_t envFlag = (env_val != nullptr) ? strtol(env_val, nullptr, 10) : defaultVal;
  return (envFlag != 0) ? 1 : 0;
}

uint32_t OptionsManager::GetHCCLExecTimeout() {
  char* env_val = std::getenv("HCCL_EXEC_TIMEOUT");
  int64_t envFlag = (env_val != nullptr) ? strtol(env_val, nullptr, 10) : 0;
  return static_cast<uint32_t>(envFlag);
}

bool OptionsManager::CheckUseNpuLogEnable() {
  static int useNpuLog = -1;
  if (useNpuLog == -1) {
    useNpuLog = GetBoolTypeOption("NPU_LOG_ENABLE");
  }

  return (useNpuLog == 1);
}

int32_t OptionsManager::GetACLExecTimeout() {
  char* env_val = std::getenv("ACL_STREAM_TIMEOUT");
  int64_t envFlag = (env_val != nullptr) ? strtol(env_val, nullptr, 10) : -1;
  return static_cast<int32_t>(envFlag);
}

uint32_t OptionsManager::CheckUseHcclAsyncErrorHandleEnable() {
  char* asyncErrorHandling_val = std::getenv("HCCL_ASYNC_ERROR_HANDLING");
  int64_t asyncErrorHandlingFlag = (asyncErrorHandling_val != nullptr) ? strtol(asyncErrorHandling_val, nullptr, 10) : 0;
  return static_cast<uint32_t>(asyncErrorHandlingFlag);
}

uint32_t OptionsManager::CheckUseDesyncDebugEnable() {
  char* desyncDebug_val = std::getenv("HCCL_DESYNC_DEBUG");
  int64_t desyncDebugFlag = (desyncDebug_val != nullptr) ? strtol(desyncDebug_val, nullptr, 10) : 0;
  return static_cast<uint32_t>(desyncDebugFlag);
}

} // namespace option
} // namespace c10_npu