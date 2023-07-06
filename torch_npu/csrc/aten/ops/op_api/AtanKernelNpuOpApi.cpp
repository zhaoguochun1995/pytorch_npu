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
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
namespace at_npu { 
namespace native {
at::Tensor& NPUNativeOpApiFunctions::atan_out(const at::Tensor& self, at::Tensor& result) { 
  DO_COMPATIBILITY(aclnnAtan, NPUNativeFunctions::atan_out(self, result));
  TORCH_CHECK(!isIntegralType(result.scalar_type(), true), "result dtype can't be cast to the desired output type.\n");
  auto outputSize = self.sizes();
  OpPreparation::CheckOut({self}, result, result.scalar_type(), outputSize);
  EXEC_NPU_CMD(aclnnAtan, self, result);
  return result;  
}
 
at::Tensor NPUNativeOpApiFunctions::atan(const at::Tensor& self) { 
  DO_COMPATIBILITY(aclnnAtan, NPUNativeFunctions::atan(self));
  auto outputSize = self.sizes();
  auto outDtype = self.dtype();
  if (isIntegralType(self.scalar_type(), true)) {
    outDtype = at::kFloat;
  }
  at::Tensor result = OpPreparation::ApplyTensorWithSizes(outputSize, self.options().dtype(outDtype));
  EXEC_NPU_CMD(aclnnAtan, self, result);

  return result;
} 
 
at::Tensor& NPUNativeOpApiFunctions::atan_(at::Tensor& self) { 
  DO_COMPATIBILITY(aclnnInplaceAtan, NPUNativeFunctions::atan_(self));
  TORCH_CHECK(!isIntegralType(self.scalar_type(), true), "result dtype can't be cast to the desired output type.\n");
  EXEC_NPU_CMD(aclnnInplaceAtan, self);
  return self;
}

}} // namespace at_npu::native