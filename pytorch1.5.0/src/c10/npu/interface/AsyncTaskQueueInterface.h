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

#ifndef __C10_NPU_INTERFACE_ASYNCTASKQUEUEINTERFACE__
#define __C10_NPU_INTERFACE_ASYNCTASKQUEUEINTERFACE__

#include "c10/core/Storage.h"
#include "c10/npu/NPUStream.h"
#include "third_party/acl/inc/acl/acl_rt.h"

namespace c10 {
namespace npu {
namespace queue {
struct CopyParas {
  void *dst = nullptr;
  size_t dstLen = 0;
  void *src = nullptr;
  size_t srcLen = 0;
  aclrtMemcpyKind kind = ACL_MEMCPY_HOST_TO_HOST;
  void Copy(CopyParas& other);
};

enum EventAllocatorType {
  HOST_ALLOCATOR_EVENT = 1,
  NPU_ALLOCATOR_EVENT = 2,
  RESERVED = -1,
};

struct EventParas {
  explicit EventParas(aclrtEvent aclEvent, EventAllocatorType allocatorType) :
      event(aclEvent), eventAllocatorType(allocatorType) {}
  aclrtEvent event = nullptr;
  void Copy(EventParas& other);
  EventAllocatorType eventAllocatorType = RESERVED;
};

enum QueueParamType {
  COMPILE_AND_EXECUTE = 1,
  ASYNC_MEMCPY = 2,
  RECORD_EVENT = 3,
  WAIT_EVENT = 4,
  LAZY_DESTROY_EVENT = 5,
};

struct QueueParas {
  QueueParas(QueueParamType type, size_t len, void *val) : paramType(type), paramLen(len), paramVal(val) {}
  aclrtStream paramStream = nullptr;
  QueueParamType paramType = COMPILE_AND_EXECUTE;
  size_t paramLen = 0;
  void* paramVal = nullptr;
};

aclError LaunchAsyncCopyTask(void* dst, size_t dstLen, void* src, size_t srcLen, aclrtMemcpyKind kind);

aclError HostAllocatorLaunchRecordEventTask(aclrtEvent event,
                                            at::npu::NPUStream npuStream);

aclError NpuAllocatorLaunchRecordEventTask(aclrtEvent event,
                                           at::npu::NPUStream npuStream);

aclError LaunchRecordEventTask(aclrtEvent event, at::npu::NPUStream npuStream);

aclError LaunchWaitEventTask(aclrtEvent event, at::npu::NPUStream npuStream);

aclError LaunchLazyDestroyEventTask(aclrtEvent event);
} // namespace queue
} // namespace npu
} // namespace c10

#endif // __C10_NPU_INTERFACE_ASYNCTASKQUEUEINTERFACE__