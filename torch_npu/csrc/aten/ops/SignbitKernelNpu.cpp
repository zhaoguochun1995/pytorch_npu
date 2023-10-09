// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeFunctions::signbit_out(const at::Tensor& self, at::Tensor& result) {
  TORCH_NPU_WARN_ONCE(
    "Warning: kernel [signbit] is not supported by NPU currently. Now this kernel is running on CPU.");
  OpPreparation::CheckOut({self}, result, at::ScalarType::Bool, self.sizes());
  const auto self_cpu = self.cpu();
  auto result_cpu = result.cpu();
  at::signbit_out(result_cpu, self_cpu);
  result.copy_(result_cpu);
  return result;
}

} // namespace native
} // namespace at_npu