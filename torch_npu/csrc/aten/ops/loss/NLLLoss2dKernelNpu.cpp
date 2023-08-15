// Copyright (c) 2020, Huawei Technologies.All rights reserved.
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

#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

namespace {

tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>> nll_loss2d_npu_output_size(
    const at::Tensor& self,
    const at::Tensor& target,
    int64_t reduction,
    int64_t ignore_index) {
  c10::SmallVector<int64_t, SIZE> output_size;
  c10::SmallVector<int64_t, SIZE> total_weight_size;

  if (reduction == at::Reduction::None) {
    output_size = {self.size(0)};
  }

  return tuple<c10::SmallVector<int64_t, SIZE>, c10::SmallVector<int64_t, SIZE>>(
      output_size, total_weight_size);
}
} // namespace

tuple<at::Tensor&, at::Tensor&> NPUNativeFunctions::nll_loss2d_forward_out(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    at::Tensor& result,
    at::Tensor& total_weight) {
  at::Tensor weight = c10::value_or_else(weight_opt, [] {return at::Tensor();});
  at::Tensor weight_tensor;
  if (weight.defined()) {
    weight_tensor = NpuUtils::format_contiguous(weight);
  } else {
    weight_tensor = at::ones(self.size(1), self.options());
  }

  if (ignore_index >= 0 && ignore_index < self.size(-1)) {
    at::Tensor zero = at::zeros(1, self.options());
    CalcuOpUtil::AclrtMemcpyAsync(
        {weight_tensor, ignore_index},
        weight_tensor.itemsize(),
        {zero, 0},
        weight_tensor.itemsize(),
        ACL_MEMCPY_DEVICE_TO_DEVICE);
  }

  OpPreparation::CheckMemory({self, target, weight_tensor}, {result, total_weight});

  auto reductionStr = CalcuOpUtil::GetReductionStr(reduction);
  OpCommand cmd;
  cmd.Name("NLLLoss")
      .Input(self)
      .Input(target)
      .Input(weight_tensor)
      .Attr("reduction", reductionStr)
      .Attr("ignore_index", ignore_index)
      .Output(result)
      .Output(total_weight)
      .Run();

  NPUNativeFunctions::npu_reshape_out(result, result.sizes(), true, result);
  return tuple<at::Tensor&, at::Tensor&>(result, total_weight);
}

tuple<at::Tensor, at::Tensor> NPUNativeFunctions::nll_loss2d_forward(
    const at::Tensor& self,
    const at::Tensor& target,
    const c10::optional<at::Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index) {
  // Check Target Dtype
  auto scalar_type = target.scalar_type();
  TORCH_CHECK(scalar_type == at::kLong || scalar_type == at::kInt, 
      "Expected object of scalar type ", at::kLong, " or ", at::kInt, " but got scalar type ", scalar_type,
      " for argument 'target'  in call to nll_loss2d_forward");
  at::Tensor target_cast = (scalar_type == at::kLong) ?
      NPUNativeFunctions::npu_dtype_cast(target, at::kInt) : target;

  auto self_input = self.contiguous();
  self_input = NPUNativeFunctions::npu_format_cast(self_input, ACL_FORMAT_ND);
  self_input = self_input.permute({0, 2, 3, 1});
  self_input = self_input.reshape({-1, self.size(1)});

  auto target_input = target_cast.contiguous();
  target_input = target_cast.reshape({-1});

  // calculate the output size
  auto output_sizes =
      nll_loss2d_npu_output_size(self_input, target, reduction, ignore_index);

  // construct the output tensor of the NPU
  at::Tensor result =
      OpPreparation::ApplyTensor(self_input, std::get<0>(output_sizes));
  at::Tensor total_weight =
      OpPreparation::ApplyTensor(self_input, std::get<1>(output_sizes));

  // calculate the output result of the NPU
  NPUNativeFunctions::nll_loss2d_forward_out(
      self_input,
      target_input,
      weight_opt,
      reduction,
      ignore_index,
      result,
      total_weight);

  if (reduction == at::Reduction::None) {
    result.resize_({self.size(0), self.size(2), self.size(3)});
  }
  return tuple<at::Tensor, at::Tensor>(result, total_weight);
}

} // namespace native
} // namespace at_npu