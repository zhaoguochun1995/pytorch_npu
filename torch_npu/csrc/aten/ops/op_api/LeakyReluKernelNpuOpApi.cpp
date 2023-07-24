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
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {
at::Tensor& NPUNativeOpApiFunctions::leaky_relu_out(const at::Tensor& self, const at::Scalar& negval,
                                                    at::Tensor& result) {
  DO_COMPATIBILITY(aclnnLeakyRelu, NPUNativeFunctions::leaky_relu_out(self, negval, result));

  OpPreparation::CheckOut({self}, result, self.scalar_type(), self.sizes());
  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnLeakyRelu, self, negval, result);

  return result;
}

at::Tensor NPUNativeOpApiFunctions::leaky_relu(const at::Tensor& self, const at::Scalar& negval) {
  DO_COMPATIBILITY(aclnnLeakyRelu, NPUNativeFunctions::leaky_relu(self, negval));

  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self);
  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnLeakyRelu, self, negval, result);

  return result;
}

at::Tensor& NPUNativeOpApiFunctions::leaky_relu_(at::Tensor& self, const at::Scalar& negval) {
  DO_COMPATIBILITY(aclnnInplaceLeakyRelu, NPUNativeFunctions::leaky_relu_(self, negval));
  EXEC_NPU_CMD(aclnnInplaceLeakyRelu, self, negval);
  return self;
}

} // namespace native
} // namespace at_npu
