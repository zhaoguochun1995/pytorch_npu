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

#include "ATen/native/npu/utils/KernelNpuOutputSize.h"
#include "ATen/native/npu/utils/OpTemplate.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor __and___dest_output(const Tensor& self, const Tensor& other) {
  bool isSelfWrapped = CalcuOpUtil::is_scalar_wrapped_to_tensor(self);

  if (not isSelfWrapped) {
    return self;
  } else {
    return other;
  }
}

Tensor& __and___out_npu(
    Tensor& result,
    const Tensor& self,
    const Scalar other) {
  OpCommand cmd;
  cmd.Name((self.scalar_type() == ScalarType::Bool) ? "LogicalAnd" : "BitwiseAnd")
     .Input(self)
     .Input(other,self.scalar_type())
     .Output(result)
     .Run();

  return result;
}

Tensor& __and___out_npu(
    Tensor& result,
    const Tensor& self,
    const Tensor& other) {
  if (other.dim() == 0 && !other.is_npu()) {
    __and___out_npu(result, self, other.item());
  } else if (self.dim() == 0 && !self.is_npu()) {
    __and___out_npu(result, other, self.item());
  } else {
    OpCommand cmd;
    cmd.Name((self.scalar_type() == ScalarType::Bool) ? "LogicalAnd" : "BitwiseAnd")
       .Input(self)
       .Input(other)
       .Output(result)
       .Run(); 
  }

  return result;
}

Tensor __and___npu(const Tensor& self, const Tensor& other) {
  // calculate the output size
  Tensor outputTensor = __and___dest_output(self, other);
  auto outputSize = broadcast_ops_npu_output_size(self, other);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize,
      outputTensor.options(),
      CalcuOpUtil::get_tensor_npu_format(outputTensor));

  // calculate the output result of the NPU
  __and___out_npu(result, self, other);
  return result;
}

Tensor __and___npu(const Tensor& self, Scalar other) {
  // calculate the output size
  auto outputSize = input_same_output_size(self);

  // construct the output tensor of the NPU
  Tensor result = at::empty_with_format(
      outputSize, self.options(), CalcuOpUtil::get_tensor_npu_format(self));

  // calculate the output result of the NPU
  __and___out_npu(result, self, other);

  return result;
}

} // namespace native
} // namespace at
