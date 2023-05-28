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
#include <third_party/acl/inc/acl/op_api/aclnn_op.h>
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeOpApiFunctions::sqrt(const at::Tensor &self)
{
  // return at::threshold(self, 0, 0);
  // calculate the output size
  auto outputSize = input_same_output_size(self);

  // construct the output tensor of the NPU
  at::Tensor result = OpPreparation::ApplyTensorWithFormat(
      outputSize, self.options(), CalcuOpUtil::GetTensorNpuFormat(self));

  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnSqrt, self, result);
  return result;
}
at::Tensor& NPUNativeOpApiFunctions::sqrt_out(const at::Tensor& self, at::Tensor& result) {
  EXEC_NPU_CMD(aclnnSqrt, self, result);
  return result;
}
at::Tensor &NPUNativeOpApiFunctions::sqrt_(at::Tensor &self)
{
  EXEC_NPU_CMD(aclnnInplaceSqrt, self);
  return self;
}

} // namespace native
} // namespace at_npu