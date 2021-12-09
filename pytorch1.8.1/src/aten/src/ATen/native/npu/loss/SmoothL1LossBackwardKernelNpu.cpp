// Copyright (c) 2020 Huawei Technologies Co., Ltd
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

#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& smooth_l1_loss_backward_out_npu(
    const Tensor& grad_out,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    double beta,
    Tensor& grad_input) {
  string reductionStr(CalcuOpUtil::get_reduction_str(reduction));

  OpPreparation::CheckMemory({self, grad_out, target}, {grad_input});
  OpCommand cmd;
  cmd.Name("SmoothL1LossGradV2")
      .Input(self)
      .Input(target)
      .Input(grad_out)
      .Output(grad_input)
      .Attr("reduction", reductionStr)
      .Attr("sigma", static_cast<float>(1.0))
      .Run();
  return grad_input;
}

Tensor smooth_l1_loss_backward_npu(
    const Tensor& grad_out,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    double beta) {
  Tensor grad_input = OpPreparation::ApplyTensor(self);

  smooth_l1_loss_backward_out_npu(
      grad_out, self, target, reduction, beta, grad_input);
  return grad_input;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("smooth_l1_loss_backward", TORCH_FN(smooth_l1_loss_backward_npu));
  m.impl("smooth_l1_loss_backward.grad_input", TORCH_FN(smooth_l1_loss_backward_out_npu));
}
} // namespace native
} // namespace at
