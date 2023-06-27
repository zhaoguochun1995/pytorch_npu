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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"


namespace at_npu {
namespace native {

at::Tensor &lt_out_npu_nocheck(at::Tensor &result, const at::Tensor &self, const at::Tensor &other)
{
  at::Tensor selfCast = self;
  at::Tensor otherCast = other;
  if (self.dtype() == at::ScalarType::Int || other.dtype() == at::ScalarType::Int ||
      self.dtype() == at::ScalarType::Bool || other.dtype() == at::ScalarType::Bool ||
      self.dtype() == at::ScalarType::Long || other.dtype() == at::ScalarType::Long)
  {
    selfCast = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Float);
    otherCast = NPUNativeFunctions::npu_dtype_cast(other, at::ScalarType::Float);
  }
  auto unified_result = OpPreparation::comparison_op_check(result, selfCast, otherCast, true);
  OpCommand cmd;
  cmd.Name("Less")
      .Expect(unified_result)
      .Input(selfCast)
      .Input(otherCast)
      .Output(result)
      .Run();

  return result;
}

at::Tensor &NPUNativeFunctions::lt_out(const at::Tensor &self, const at::Tensor &other, at::Tensor &result)
{
  at::Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  at::Tensor formatCastOfOther = OpPreparation::CastBackToOriFormat(other);
  auto outputSize = broadcast_ops_npu_output_size(formatCastOfSelf, formatCastOfOther);

  OpPreparation::CheckOut(
      {self},
      result,
      ACL_FORMAT_ND,
      at::kBool,
      outputSize);

  lt_out_npu_nocheck(result, formatCastOfSelf, formatCastOfOther);
  return result;
}

at::Tensor &lt_out_npu_nocheck(at::Tensor &result, const at::Tensor &self, at::Scalar other)
{
  at::Tensor selfCast = self;
  if (self.dtype() == at::ScalarType::Int ||
      self.dtype() == at::ScalarType::Long || self.dtype() == at::ScalarType::Bool)
  {
    selfCast = NPUNativeFunctions::npu_dtype_cast(self, at::ScalarType::Float);
  }
  OpCommand cmd;
  cmd.Name("Less")
      .Input(selfCast)
      .Input(other, selfCast.scalar_type())
      .Output(result)
      .Run();

  return result;
}

at::Tensor &NPUNativeFunctions::lt_out(const at::Tensor &self, const at::Scalar& other, at::Tensor &result)
{
  at::Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  auto outputSize = formatCastOfSelf.sizes();
  OpPreparation::CheckOut(
      {self},
      result,
      ACL_FORMAT_ND,
      at::kBool,
      outputSize);

  lt_out_npu_nocheck(result, formatCastOfSelf, other);
  return result;
}

at::Tensor NPUNativeFunctions::lt(const at::Tensor &self, const at::Tensor &other)
{
  if (OpPreparation::IsCPUScalar(other)) {
    return NPUNativeFunctions::lt(self, other.item());
  } else if (OpPreparation::IsCPUScalar(self)) {
    return NPUNativeFunctions::gt(other, self.item());
  } else {
    TORCH_CHECK(self.device() == other.device(),
        "Expected all tensors to be on the same device, but found at least two devices, ",
        (self.device().type() == at_npu::key::NativeDeviceType ? "npu" : "cpu"),
        " and ",
        (other.device().type() == at_npu::key::NativeDeviceType ? "npu! " : "cpu! "));
    at::Tensor format_cast_of_self = OpPreparation::CastBackToOriFormat(self);
    at::Tensor format_cast_of_other = OpPreparation::CastBackToOriFormat(other);
    // calculate the output size
    auto output_size = broadcast_ops_npu_output_size(format_cast_of_self, format_cast_of_other);

    // construct the output tensor of the NPU
    at::Tensor result = OpPreparation::ApplyTensorWithSizes(
        output_size,
        format_cast_of_self.options().dtype(at::kBool));

    // calculate the output result of the NPU
    lt_out_npu_nocheck(result, format_cast_of_self, format_cast_of_other);
    return result;
  }
}

at::Tensor NPUNativeFunctions::lt(const at::Tensor &self, const at::Scalar& other)
{
  at::Tensor formatCastOfSelf = OpPreparation::CastBackToOriFormat(self);
  // calculate the output size
  auto outputSize = input_same_output_size(formatCastOfSelf);

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithSizes(
      outputSize,
      formatCastOfSelf.options().dtype(at::kBool));

  // calculate the output result of the NPU
  lt_out_npu_nocheck(result, formatCastOfSelf, other);
  return result;
}

at::Tensor &NPUNativeFunctions::lt_(at::Tensor &self, const at::Tensor &other)
{
  if (OpPreparation::IsCPUScalar(other)) {
    return NPUNativeFunctions::lt_(self, other.item());
  } else {
    TORCH_CHECK(self.device() == other.device(),
        "Expected all tensors to be on the same device, but found at least two devices, ",
        (self.device().type() == at_npu::key::NativeDeviceType ? "npu" : "cpu"),
        " and ",
        (other.device().type() == at_npu::key::NativeDeviceType ? "npu! " : "cpu! "));
    OpPreparation::CastBackToOriFormat(self);
    OpPreparation::CastBackToOriFormat(other);
    OpPreparation::CheckMemory({self, other}, {self});

    at::Tensor result = OpPreparation::ApplyTensorWithFormat(
        self.sizes(),
        self.options().dtype(at::ScalarType::Byte),
        CalcuOpUtil::GetTensorNpuFormat(self));

    if (!NpuUtils::check_match(&self)) {
      at::Tensor contiguous_self = NpuUtils::format_contiguous(self);
      lt_out_npu_nocheck(result, contiguous_self, other);
    } else {
      lt_out_npu_nocheck(result, self, other);
    }

    // uint8 to self dtype
    self.copy_(result);

    return self;
  }
}

at::Tensor &NPUNativeFunctions::lt_(at::Tensor &self, const at::Scalar& other)
{
  OpPreparation::CastBackToOriFormat(self);
  c10::SmallVector<at::Tensor, N> inputs = {self};
  c10::SmallVector<at::Tensor, N> outputs = {self};
  CalcuOpUtil::CheckMemoryOverLaps(inputs, outputs);

  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      self.sizes(),
      self.options().dtype(at::ScalarType::Byte),
      CalcuOpUtil::GetTensorNpuFormat(self));

  if (!NpuUtils::check_match(&self))
  {
    at::Tensor contiguousSelf = NpuUtils::format_contiguous(self);
    lt_out_npu_nocheck(result, contiguousSelf, other);
  }
  else
  {
    lt_out_npu_nocheck(result, self, other);
  }

  // uint8 to self dtype
  self.copy_(result);

  return self;
}

} // namespace native
} // namespace at_npu
