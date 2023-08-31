// Copyright (c) 2023, Huawei Technologies.All rights reserved.
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

#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::max_unpool3d_out(
    const at::Tensor& self,
    const at::Tensor& indices,
    at::IntArrayRef output_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnMaxUnpool3d, NPUNativeFunctions::max_unpool3d_out(self, indices, output_size,
                                                                          stride, padding, result));
  auto out_shape = max_pool3d_output_size(self, output_size);
  OpPreparation::check_tensor({self, indices}, result, self.scalar_type(), out_shape);
  EXEC_NPU_CMD(aclnnMaxUnpool3d, self, indices, output_size, stride, padding, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::max_unpool3d(
    const at::Tensor& self,
    const at::Tensor& indices,
    at::IntArrayRef output_size,
    at::IntArrayRef stride,
    at::IntArrayRef padding) {
  DO_COMPATIBILITY(aclnnMaxUnpool3d, NPUNativeFunctions::max_unpool3d(self, indices, output_size, stride, padding));
  auto out_shape = max_pool3d_output_size(self, output_size);
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self, out_shape);
  EXEC_NPU_CMD(aclnnMaxUnpool3d, self, indices, output_size, stride, padding, result);
  return result;
}

} // namespace native
} // namespace at_npu

