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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {
at::Tensor NPUNativeOpApiFunctions::rsub(const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  DO_COMPATIBILITY(aclnnRsub, NPUNativeFunctions::rsub(self, other, alpha));
  auto output_size = broadcast_ops_npu_output_size(self, other);
  at::ScalarType result_type = at::native::result_type(self, other);
  auto result = OpPreparation::ApplyTensorWithoutFormat(output_size, self.options().dtype(result_type));
  EXEC_NPU_CMD(aclnnRsub, self, other, alpha, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::rsub(const at::Tensor& self, const at::Scalar& other, const at::Scalar& alpha) {
  DO_COMPATIBILITY(aclnnRsubs, NPUNativeFunctions::rsub(self, other, alpha));
  auto output_size = input_same_output_size(self);
  at::ScalarType result_type = at::native::result_type(self, other);
  auto result = OpPreparation::ApplyTensorWithoutFormat(output_size, self.options().dtype(result_type));
  EXEC_NPU_CMD(aclnnRsubs, self, other, alpha, result);
  return result;
}

} // namespace native
} // namespace at_npu