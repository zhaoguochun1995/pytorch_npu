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
#include <c10/core/Scalar.h>
#include <ATen/record_function.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "op_plugin/OpInterface.h"

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/StorageDescHelper.h"
#include "torch_npu/csrc/aten/common/InnerNpuNativeFunction.h"
#include "torch_npu/csrc/core/npu/interface/AsyncTaskQueueInterface.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/framework/graph/util/GraphModeGuard.h"
#include "torch_npu/csrc/core/npu/NPURunMode.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

namespace at_npu {
namespace native {

// the dst and src are same dtype
// the dst and src have same elemsize
// if exceptCopySize is not defined, we will copy dst storage size
// so: caller should make sure that the storage size of src and dst are reasonable.
void copy_d2d_by_memcpy(at::Tensor& dst, const at::Tensor& src, int64_t exceptSize) {
  int64_t size = exceptSize;
  auto dst_mem_size = StorageDescHelper::GetMemorySize(dst);
  if (exceptSize == 0) {
    size = dst_mem_size;
  }

  if (c10_npu::NpuRunMode::IsGraphMode()) {
    if (dst_mem_size != size ||
        dst_mem_size != StorageDescHelper::GetMemorySize(src)) {
      // In graph mode, using Viewcopy to copy part data of src.
      op_plugin::npu_view_copy(dst, src, true);
      return;
    }

    /*
    In single op mode, the current interface may copy tensors between different
    shapes. So in the graph mode, only Reshape can be used to complete the copy
    of the complete memory block, not Identity.

    Refer to the following case:
    a [3,4,5,6] [3,4,30]
    b [3,4,5,6] [3,4,5,6]
    a.copy_(b)

    We should ensure that after copying, the shape of a is still [3,4,5,6] [3,4,30].

    In single op mode, it is always satisfied. But in graph mode, it is
    only satisfied when doing Reshape operations based on base_sizes_ of dst.
    */

    // In graph mode, using Reshape to copy whole data of src.
    c10::SmallVector<int64_t, 5> self_base_sizes_5 =
        torch_npu::NPUBridge::GetNpuStorageImpl(dst.storage().unsafeGetStorageImpl())->get_npu_desc().base_sizes_;
    c10::SmallVector<int64_t, 32> self_base_sizes_32(self_base_sizes_5.begin(), self_base_sizes_5.end());
    OpCommand cmd;
    cmd.Name("Reshape")
        .InputWithoutContiguous(src)
        .Input(self_base_sizes_32)
        .Output(dst)
        .Run();
    return;
  }

  if(!dst.data_ptr()) {
    TORCH_WARN("copy_d2d_by_memcpy, dst.data_ptr() is null.");
    return;
  }

  if(!src.data_ptr()) {
    TORCH_WARN("copy_d2d_by_memcpy, src.data_ptr() is null.");
    return;
  }

  if (dst.data_ptr() == src.data_ptr() && dst.element_size() == src.element_size()) {
    return;
  }

  // The current logic is only used in single op mode.
  aclError error = c10_npu::queue::LaunchAsyncCopyTask(
      dst.data_ptr(),
      size * dst.element_size(),
      src.data_ptr(),
      size * dst.element_size(),
      ACL_MEMCPY_DEVICE_TO_DEVICE);
  if (error != ACL_ERROR_NONE) {
    AT_ERROR("async copy device to device error.");
    return;
  }
}
} // namespace native
} // namespace at_npu
