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
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include <third_party/acl/inc/acl/op_api/aclnn_op.h>

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::maximum_out(
    const at::Tensor& self,
    const at::Tensor& other,
    at::Tensor& result) {
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  OpPreparation::CheckOut({self, other}, result, result, outputSize);
  EXEC_NPU_CMD(aclnnMaximum, self, other, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::maximum(
    const at::Tensor& self,
    const at::Tensor& other) {
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  at::ScalarType high_type = at::native::result_type(self, other);
  at::Tensor self_copy = (self.scalar_type() != high_type && !CalcuOpUtil::IsScalarWrappedToTensor(self)) ?
      NPUNativeFunctions::npu_dtype_cast(self, high_type) : self;
  at::Tensor other_copy = (other.scalar_type() != high_type && !CalcuOpUtil::IsScalarWrappedToTensor(other)) ?
      NPUNativeFunctions::npu_dtype_cast(other, high_type) : other;
  at::Tensor result = OpPreparation::ApplyTensor(self_copy, outputSize);
  EXEC_NPU_CMD(aclnnMaximum, self_copy, other_copy, result);
  return result;
}

} // namespace native
} // namespace at_npu 
