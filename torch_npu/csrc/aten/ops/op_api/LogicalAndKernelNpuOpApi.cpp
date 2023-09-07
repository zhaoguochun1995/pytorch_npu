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

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::logical_and_out(const at::Tensor& self, const at::Tensor& other,
                                                     at::Tensor& result) {
  DO_COMPATIBILITY(aclnnLogicalAnd, NPUNativeFunctions::logical_and_out(self, other, result));
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  OpPreparation::CheckOut({self}, result, CalcuOpUtil::GetTensorNpuFormat(self), result.scalar_type(), outputSize);
  EXEC_NPU_CMD(aclnnLogicalAnd, self, other, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::logical_and(const at::Tensor& self, const at::Tensor& other) {
  DO_COMPATIBILITY(aclnnLogicalAnd, NPUNativeFunctions::logical_and(self, other));
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(outputSize, self.options().dtype(at::kBool));
  EXEC_NPU_CMD(aclnnLogicalAnd, self, other, result);
  return result;
}

at::Tensor &NPUNativeOpApiFunctions::logical_and_(at::Tensor &self, const at::Tensor &other) {
  DO_COMPATIBILITY(aclnnLogicalAnd, NPUNativeFunctions::logical_and_(self, other));
  EXEC_NPU_CMD(aclnnInplaceLogicalAnd, self, other);
  return self;
}

}  // namespace native
}  // namespace at_npu