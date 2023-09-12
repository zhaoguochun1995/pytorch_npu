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

#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::sinh_out(const at::Tensor& self, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnSinh, NPUNativeFunctions::sinh_out(self, result));
  OpPreparation::CheckOut({self}, result, result.scalar_type(), self.sizes());
  EXEC_NPU_CMD(aclnnSinh, self, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::sinh_(at::Tensor& self) {
  DO_COMPATIBILITY(aclnnInplaceSinh, NPUNativeFunctions::sinh_(self));
  EXEC_NPU_CMD(aclnnInplaceSinh, self);
  return self;
}

at::Tensor NPUNativeOpApiFunctions::sinh(const at::Tensor& self) {
  DO_COMPATIBILITY(aclnnSinh, NPUNativeFunctions::sinh(self));
  auto output_options = (isIntegralType(self.scalar_type(), true)) ? self.options().dtype(at::kFloat) : self.options();
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self.sizes(), output_options);
  EXEC_NPU_CMD(aclnnSinh, self, result);
  return result;
}

} // namespace native
} // namespace at_npu