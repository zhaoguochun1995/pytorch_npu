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
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::adaptive_avg_pool3d_backward_out(const at::Tensor& grad_output,
                                                                      const at::Tensor& self,
                                                                      at::Tensor& result) {
  DO_COMPATIBILITY(aclnnAdaptiveAvgPool3dBackward,
                   NPUNativeFunctions::adaptive_avg_pool3d_backward_out(grad_output, self, result));
  EXEC_NPU_CMD(aclnnAdaptiveAvgPool3dBackward, grad_output, self, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::_adaptive_avg_pool3d_backward(const at::Tensor& grad_output,
                                                                  const at::Tensor& self) {
  DO_COMPATIBILITY(aclnnAdaptiveAvgPool3dBackward,
                   NPUNativeFunctions::_adaptive_avg_pool3d_backward(grad_output, self));
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat(self);
  EXEC_NPU_CMD(aclnnAdaptiveAvgPool3dBackward, grad_output, self, result);
  return result;
}

} // native
} // at_npu