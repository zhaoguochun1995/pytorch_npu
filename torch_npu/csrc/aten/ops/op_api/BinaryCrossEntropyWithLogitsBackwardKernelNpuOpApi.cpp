// Copyright (c) 2023 Huawei Technologies Co., Ltd
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
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {
at::Tensor NPUNativeOpApiFunctions::binary_cross_entropy_with_logits_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight_opt,
    const c10::optional<at::Tensor>& pos_weight_opt,
    int64_t reduction) {
  DO_COMPATIBILITY(aclnnBinaryCrossEntropyWithLogitsBackward,
      NPUNativeFunctions::binary_cross_entropy_with_logits_backward(grad_output, self, target, 
      weight_opt, pos_weight_opt, reduction));
  at::Tensor grad_input = OpPreparation::ApplyTensorWithoutFormat(target);
  // calculate the output result of the NPU
  EXEC_NPU_CMD(aclnnBinaryCrossEntropyWithLogitsBackward, 
      grad_output, self, target, weight_opt, pos_weight_opt, reduction, grad_input);
  return grad_input;
}
} // namespace native
} // namespace at_npu