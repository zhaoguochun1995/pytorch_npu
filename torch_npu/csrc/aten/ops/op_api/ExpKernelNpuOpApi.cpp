// Copyright (c) 2020 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::exp_out(const at::Tensor& self, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnExp, NPUNativeFunctions::exp_out(self, result));
  OpPreparation::CheckOut({self}, result, result, self.sizes());
  CalcuOpUtil::CheckMemoryOverLaps({self}, {result});

  EXEC_NPU_CMD(aclnnExp, self, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::exp_(at::Tensor& self) {
  DO_COMPATIBILITY(aclnnInplaceExp, NPUNativeFunctions::exp_(self));
  EXEC_NPU_CMD(aclnnInplaceExp, self);
  return self;
}

at::Tensor NPUNativeOpApiFunctions::exp(const at::Tensor& self) {
  DO_COMPATIBILITY(aclnnExp, NPUNativeFunctions::exp(self));
  at::Tensor result;
  if (self.scalar_type() == at::ScalarType::Bool || self.scalar_type() == at::ScalarType::Long) {
    result = OpPreparation::ApplyTensor(self, self.options().dtype(at::kFloat));
  } else {
    result = OpPreparation::ApplyTensor(self);
  }
  EXEC_NPU_CMD(aclnnExp, self, result);
  return result;
}

}  // namespace native
}  // namespace at_npu