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
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::addcmul_out(const at::Tensor& self, const at::Tensor& tensor1,
                                                 const at::Tensor& tensor2, const at::Scalar& value,
                                                 at::Tensor& result) {
  DO_COMPATIBILITY(aclnnAddcmul, NPUNativeFunctions::addcmul_out(self, tensor1, tensor2, value, result));
  auto mul_output_size = broadcast_ops_npu_output_size(tensor1, tensor2);
  auto output_size = broadcast_ops_npu_output_size(self.sizes(), mul_output_size);
  OpPreparation::CheckOut(
      {self},
      result,
      self,
      output_size);

  EXEC_NPU_CMD(aclnnAddcmul, self, tensor1, tensor2, value, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::addcmul(const at::Tensor& self, const at::Tensor& tensor1,
                                            const at::Tensor& tensor2, const at::Scalar& value) {
  DO_COMPATIBILITY(aclnnAddcmul, NPUNativeFunctions::addcmul(self, tensor1, tensor2, value));
  auto mul_output_size = broadcast_ops_npu_output_size(tensor1, tensor2);
  auto output_size = broadcast_ops_npu_output_size(self.sizes(), mul_output_size);

  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self, output_size);

  EXEC_NPU_CMD(aclnnAddcmul, self, tensor1, tensor2, value, result);
  return result;
}

at::Tensor& NPUNativeOpApiFunctions::addcmul_(at::Tensor& self, const at::Tensor& tensor1, const at::Tensor& tensor2,
                                              const at::Scalar& value) {
  DO_COMPATIBILITY(aclnnInplaceAddcmul, NPUNativeFunctions::addcmul_(self, tensor1, tensor2, value));
  EXEC_NPU_CMD(aclnnInplaceAddcmul, self, tensor1, tensor2, value);
  return self;
}

}  // namespace native
}  // namespace at_npu