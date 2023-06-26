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
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& repeat_interleave_out_npu(at::Tensor& result, const at::Tensor& self, int64_t repeats) {
  at::Scalar repeat = repeats;
  OpCommand cmd;
  cmd.Name("RepeatInterleave")
    .Input(self)
    .Input(repeat, at::kLong)
    .Output(result)
    .Attr("axis", (int64_t)0)
    .Run();

  return result;
}

at::Tensor& repeat_interleave_out_npu(at::Tensor& result, const at::Tensor& self, const at::Tensor& repeats) {
  OpCommand cmd;
  cmd.Name("RepeatInterleave")
    .Input(self)
    .Input(repeats)
    .Output(result)
    .Attr("axis", (int64_t)0)
    .Run();

  return result;
}

at::Tensor NPUNativeFunctions::repeat_interleave(
    const at::Tensor& self,
    int64_t repeats,
    c10::optional<int64_t> dim,
    c10::optional<int64_t> output_size) {
  int64_t real_dim = dim.value_or(0);
  int64_t self_dim = self.dim();
  TORCH_CHECK(
      (real_dim >= -self_dim) && (real_dim <= self_dim - 1),
      "dim value should be in the range of [-x, x-1], x is the dimension number of input tensor.");
  TORCH_CHECK(
      repeats >= 1,
      "repeats can not be negative.");
  at::Tensor self_tensor = self;
  if (!dim.has_value()) {
    self_tensor = at::flatten(self_tensor);
  }
  if (repeats == 1) {
    return self_tensor;
  }

  if (self_dim > 1 && real_dim != 0) {
    self_tensor = self_tensor.transpose(0, real_dim);
  }

  auto result_size = repeat_interleave_npu_output_size(self_tensor, repeats, 0);
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(self_tensor, result_size, ACL_FORMAT_ND);
  repeat_interleave_out_npu(result, self_tensor, repeats);
  if (self_dim > 1 && real_dim != 0) {
    result = result.transpose(0, real_dim);
  }
  return result;
}

at::Tensor NPUNativeFunctions::repeat_interleave(
    const at::Tensor& self,
    const at::Tensor& repeats,
    c10::optional<int64_t> dim,
    c10::optional<int64_t> output_size) {
  int64_t real_dim = dim.value_or(0);
  int64_t self_dim = self.dim();
  TORCH_CHECK(
      (real_dim >= -self_dim) && (real_dim <= self_dim - 1),
      "dim value should be in the range of [-x, x-1], x is the dimension number of input tensor.");

  at::Tensor self_tensor = self;
  at::Tensor repeats_tensor = repeats;
  if (repeats.dim() == 0) {
    repeats_tensor.unsqueeze_(0);
  }
  if (!dim.has_value()) {
    self_tensor = at::flatten(self_tensor);
  }

  TORCH_CHECK(
      repeats.size(0) == self_tensor.size(real_dim),
      "repeats must have the same size as input along dim.");

  if (self_dim > 1 && real_dim != 0) {
    self_tensor = self_tensor.transpose(0, real_dim);
  }

  repeats_tensor = NPUNativeFunctions::npu_dtype_cast(repeats_tensor, at::ScalarType::Int);
  repeats_tensor = NPUNativeFunctions::npu_dtype_cast(repeats_tensor, at::ScalarType::Float);
  auto result_size = repeat_interleave_npu_output_size(self_tensor, repeats_tensor, 0);

  at::Tensor result = OpPreparation::ApplyTensorWithFormat(self_tensor, result_size, ACL_FORMAT_ND);
  repeat_interleave_out_npu(result, self_tensor, repeats);
  if (self_dim > 1 && real_dim != 0) {
    result = result.transpose(0, real_dim);
  }
  return result;
}

} // namespace native
} // namespace at_npu