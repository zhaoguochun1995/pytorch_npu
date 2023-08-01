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

#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {
// pow.Tensor_Tensor_out
at::Tensor& NPUNativeOpApiFunctions::pow_out(const at::Tensor& self, const at::Tensor& exp, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnPowTensorTensor, NPUNativeFunctions::pow_out(self, exp, result));
  auto outputSize = broadcast_ops_npu_output_size(self, exp);
  OpPreparation::CheckOut({self, exp}, result, result, outputSize);

  CalcuOpUtil::CheckMemoryOverLaps({self, exp}, {result});
  EXEC_NPU_CMD(aclnnPowTensorTensor, self, exp, result);
  return result;
}

// pow.Tensor_Scalar_out
at::Tensor& NPUNativeOpApiFunctions::pow_out(const at::Tensor& self, const at::Scalar& exp, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnPowTensorScalar, NPUNativeFunctions::pow_out(self, exp, result));
  auto resultType = at::result_type(self, exp);
  OpPreparation::CheckOut({self}, result, resultType, self.sizes());
  CalcuOpUtil::CheckMemoryOverLaps({self}, {result});
  EXEC_NPU_CMD(aclnnPowTensorScalar, self, exp, result);
  return result;
}

// pow.Scalar_out
at::Tensor &NPUNativeOpApiFunctions::pow_out(const at::Scalar& self, const at::Tensor &exp, at::Tensor &result) {
  DO_COMPATIBILITY(aclnnPowScalarTensor, NPUNativeFunctions::pow_out(self, exp, result));
  OpPreparation::CheckOut({exp}, result, result.scalar_type(), exp.sizes());
  EXEC_NPU_CMD(aclnnPowScalarTensor, self, exp, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::pow(const at::Tensor& self, const at::Tensor& exp) {
  DO_COMPATIBILITY(aclnnPowTensorTensor, NPUNativeFunctions::pow(self, exp));
  // calculate the output size
  auto output_size = broadcast_ops_npu_output_size(self, exp);
  at::ScalarType result_type = at::result_type(self, exp);
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(output_size, self.options().dtype(result_type));
  EXEC_NPU_CMD(aclnnPowTensorTensor, self, exp, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::pow(const at::Tensor& self, const at::Scalar& exp) {
  DO_COMPATIBILITY(aclnnPowTensorScalar, NPUNativeFunctions::pow(self, exp));
  auto outputSize = input_same_output_size(self);
  auto resultType = at::result_type(self, exp);
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(outputSize, self.options().dtype(resultType));
  EXEC_NPU_CMD(aclnnPowTensorScalar, self, exp, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::pow(const at::Scalar& self, const at::Tensor& exp) {
  DO_COMPATIBILITY(aclnnPowScalarTensor, NPUNativeFunctions::pow(self, exp));
  at::ScalarType result_type = at::result_type(self, exp);
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(exp.sizes(), exp.options().dtype(result_type));
  EXEC_NPU_CMD(aclnnPowScalarTensor, self, exp, result);
  return result;
}

at::Tensor &NPUNativeOpApiFunctions::pow_(at::Tensor &self, const at::Tensor &exp) {
  DO_COMPATIBILITY(aclnnInplacePowTensorTensor, NPUNativeFunctions::pow_(self, exp));
  EXEC_NPU_CMD(aclnnInplacePowTensorTensor, self, exp);
  return self;
}

at::Tensor &NPUNativeOpApiFunctions::pow_(at::Tensor &self, const at::Scalar& exp) {
  DO_COMPATIBILITY(aclnnInplacePowTensorScalar, NPUNativeFunctions::pow_(self, exp));
  EXEC_NPU_CMD(aclnnInplacePowTensorScalar, self, exp);
  return self;
}

}  // namespace native
}  // namespace at_npu