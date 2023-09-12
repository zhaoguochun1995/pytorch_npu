// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {
at::Tensor& NPUNativeOpApiFunctions::threshold_out(
    const at::Tensor& self,
    const at::Scalar& threshold,
    const at::Scalar& value,
    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnThreshold, NPUNativeFunctions::threshold_out(self, threshold, value, result));
  auto res_type = at::result_type(self, result);
  OpPreparation::CheckOut({self}, result, res_type, self.sizes());
  EXEC_NPU_CMD(aclnnThreshold, self, threshold, value, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::threshold(const at::Tensor& self, const at::Scalar& threshold, const at::Scalar& value) {
  DO_COMPATIBILITY(aclnnThreshold, NPUNativeFunctions::threshold(self, threshold, value));
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self);
  NPUNativeOpApiFunctions::threshold_out(self, threshold, value, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::threshold_(at::Tensor& self, const at::Scalar& threshold, const at::Scalar& value) {
  DO_COMPATIBILITY(aclnnInplaceThreshold, NPUNativeFunctions::threshold_(self, threshold, value));
  EXEC_NPU_CMD(aclnnInplaceThreshold, self, threshold, value);
  return self;
}

} // namespace native 
} // namespace at_npu 