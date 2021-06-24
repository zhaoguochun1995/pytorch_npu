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

#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& bitwise_and_out_npu_nocheck(
    Tensor& result,
    const Tensor& self,
    const Scalar other) {
  // executing the NPU operator
  string real_op_name = (self.dtype() == ScalarType::Bool) ? "LogicalAnd" : "BitwiseAnd";

  OpCommand cmd;
  cmd.Name(real_op_name)
      .Input(self)
      .Input(other, self.scalar_type())
      .Output(result)
      .Run();

  return result;
}

Tensor& bitwise_and_out_npu(
    Tensor& result,
    const Tensor& self,
    const Scalar other) {
  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::get_tensor_npu_format(self),
      self.scalar_type(),
      self.sizes());

  bitwise_and_out_npu_nocheck(result, self, other);

  return result;
}

Tensor& bitwise_and_out_npu_nocheck(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  auto unified_result = OpPreparation::binary_op_check(result, self, other, true);
  if (other.dim() == 0 && !other.is_npu()) {
    bitwise_and_out_npu(result, self, other.item());
  } else if (self.dim() == 0 && !self.is_npu()) {
    bitwise_and_out_npu(result, other, self.item());
  } else {
    // executing the NPU operator
    string real_op_name = (self.dtype() == ScalarType::Bool) ? "LogicalAnd" : "BitwiseAnd";

    OpCommand cmd;
    cmd.Name(real_op_name)
        .Expect(unified_result)
        .Input(self)
        .Input(other)
        .Output(result)
        .Run();
  }

  return result;
}

Tensor& bitwise_and_out_npu(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  bool isSelfWrapped = CalcuOpUtil::is_scalar_wrapped_to_tensor(self);

  Tensor outputTensor;
  if (not isSelfWrapped) {
    outputTensor = self;
  } else {
    outputTensor = other;
  }

  auto outputSize = broadcast_ops_npu_output_size(self, other);

  OpPreparation::CheckOut(
      {self},
      result,
      CalcuOpUtil::get_tensor_npu_format(outputTensor),
      outputTensor.scalar_type(),
      outputSize);

  bitwise_and_out_npu_nocheck(result, self, other);

  return result;
}

Tensor bitwise_and_npu(const Tensor& self, const Tensor& other) {
  // calculate the output size
  bool isSelfWrapped = CalcuOpUtil::is_scalar_wrapped_to_tensor(self);

  Tensor outputTensor;
  if (not isSelfWrapped) {
    outputTensor = self;
  } else {
    outputTensor = other;
  }

  auto outputSize = broadcast_ops_npu_output_size(self, other);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize,
      outputTensor.options(),
      CalcuOpUtil::get_tensor_npu_format(outputTensor));

  // calculate the output result of the NPU
  bitwise_and_out_npu_nocheck(result, self, other);

  return result;
}

Tensor bitwise_and_npu(const Tensor& self, Scalar other) {
  // calculate the output size
  auto outputSize = input_same_output_size(self);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  // calculate the output result of the NPU
  bitwise_and_out_npu_nocheck(result, self, other);

  return result;
}

Tensor& bitwise_and_npu_(Tensor& self, const Tensor& other) {
  SmallVector<Tensor, N> inputs = {self, other};
  SmallVector<Tensor, N> outputs = {self};
  CalcuOpUtil::check_memory_over_laps(inputs, outputs);

  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = bitwise_and_out_npu_nocheck(contiguousSelf, contiguousSelf, other);
    NpuUtils::format_fresh_view(self, result);
  } else {
    bitwise_and_out_npu_nocheck(self, self, other);
  }

  return self;
}

Tensor& bitwise_and_npu_(Tensor& self, Scalar other) {
  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    Tensor result = bitwise_and_out_npu_nocheck(contiguousSelf, contiguousSelf, other);
    NpuUtils::format_fresh_view(self, result);
  } else {
    bitwise_and_out_npu_nocheck(self, self, other);
  }

  return self;
}
} // namespace native
} // namespace at