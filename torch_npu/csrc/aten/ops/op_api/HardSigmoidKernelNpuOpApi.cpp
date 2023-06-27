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

#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::hardsigmoid_out(const at::Tensor& self, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnHardsigmoid, NPUNativeFunctions::hardsigmoid_out(self, result));
  OpPreparation::CheckOut(
      {self},
      result,
      self);
  EXEC_NPU_CMD(aclnnHardsigmoid, self, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::hardsigmoid(const at::Tensor &self) {
  DO_COMPATIBILITY(aclnnHardsigmoid, NPUNativeFunctions::hardsigmoid(self));
  auto out_size = input_same_output_size(self);
  auto result = OpPreparation::ApplyTensor(out_size, self.options(), self);
  EXEC_NPU_CMD(aclnnHardsigmoid, self, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::hardsigmoid_(at::Tensor &self) {
  DO_COMPATIBILITY(aclnnInplaceHardsigmoid, NPUNativeFunctions::hardsigmoid_(self));
  EXEC_NPU_CMD(aclnnInplaceHardsigmoid, self);
  return self;
}
} // namespace native
} // namespace at_npu

