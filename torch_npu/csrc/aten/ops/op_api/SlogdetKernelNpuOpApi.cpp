// Copyright (c) 2020, Huawei Technologies.All rights reserved.
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
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUGeneratorImpl.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

namespace at_npu {
namespace native {

tuple<at::Tensor, at::Tensor> NPUNativeOpApiFunctions::linalg_slogdet(const at::Tensor& self) {
  DO_COMPATIBILITY(aclnnSlogdet, NPUNativeFunctions::linalg_slogdet(self));
  // calculate the output size
  auto outputSize = array_to_small_vector(self.sizes());
  outputSize.erase(outputSize.end() - 2, outputSize.end());
  // construct the output tensor of the NPU
  at::Tensor sign = OpPreparation::ApplyTensor(self, outputSize);
  at::Tensor log = OpPreparation::ApplyTensor(self, outputSize);
  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnSlogdet, self, sign, log);

  return std::tie(sign, log);
}

tuple<at::Tensor &, at::Tensor &> NPUNativeOpApiFunctions::linalg_slogdet_out(const at::Tensor& self,
    at::Tensor& sign, at::Tensor& log) {
  DO_COMPATIBILITY(aclnnSlogdet, NPUNativeFunctions::linalg_slogdet_out(self, sign, log));
  EXEC_NPU_CMD(aclnnSlogdet, self, sign, log);

  return std::tie(sign, log);
}

} // namespace native
} // namespace at_npu