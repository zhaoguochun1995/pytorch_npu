// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2023, Facebook CORPORATION.
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

#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::silu_out(const at::Tensor& self, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnSilu, NPUNativeFunctions::silu_out(self, result));
  OpPreparation::CheckOut({self}, result, self);
  EXEC_NPU_CMD(aclnnSilu, self, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::silu(const at::Tensor& self) {
  DO_COMPATIBILITY(aclnnSilu, NPUNativeFunctions::silu(self));
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self);
  EXEC_NPU_CMD(aclnnSilu, self, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::silu_(at::Tensor& self) {
  DO_COMPATIBILITY(aclnnSilu, NPUNativeFunctions::silu_(self));
  NPUNativeOpApiFunctions::silu_out(self, self);
  return self;
}

} // namespace native
} // namespace at_npu