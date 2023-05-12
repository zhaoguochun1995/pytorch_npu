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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeFunctions::one_hot(const at::Tensor& self, int64_t num_classes) {
  at::Scalar on_value = 1;
  at::Scalar off_value = 0;
  int64_t axis = -1;
  int64_t depth;
  auto self_temp = NPUNativeFunctions::npu_dtype_cast(self, at::kFloat);

  TORCH_CHECK(
      self_temp.dim() < 8, "NPU error,can not support the input tensor's dim bigger than 7.");
  if (self.numel() == 0) {
    if (num_classes <= 0) {
      AT_ERROR("Can not infer total number of classes from empty tensor.");
    } else {
      depth = num_classes;
    }
  }
  
  if (num_classes == -1) {
    depth = self_temp.max().item().toLong() + 1;
  } else {
    depth = num_classes;
  }

  auto outputSize = array_to_small_vector(self.sizes());
  outputSize.emplace_back(depth);
  at::Tensor result = OpPreparation::ApplyTensor(
      outputSize,
      self.options(),
      self);
  at::Scalar depthCp = depth;

  OpCommand cmd;
  cmd.Name("OneHot")
    .Input(self)
    .Input(depthCp, self.scalar_type(), CompileType::MEMORY_HOST_COMPILE_DEPENDENT)
    .Input(on_value, self.scalar_type())
    .Input(off_value, self.scalar_type())
    .Output(result)
    .Attr("axis", axis)
    .Run();
  return result;
}
} // namespace native
} // namespace at_npu
