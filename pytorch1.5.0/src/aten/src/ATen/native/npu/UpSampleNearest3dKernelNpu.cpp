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

#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& upsample_nearest3d_out_npu_nocheck(
    Tensor& result,
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  TORCH_CHECK(
      output_size.size() == 3,
      "It is expected output_size equals to 3, but got size ",
      output_size.size());

  int64_t output_depth = output_size[0];
  int64_t output_height = output_size[1];
  int64_t output_width = output_size[2];

  int64_t nbatch = input.size(0);
  int64_t channels = input.size(1);
  int64_t input_depth = input.size(2);
  int64_t input_height = input.size(3);
  int64_t input_width = input.size(4);

  result.resize_({nbatch, channels, output_depth, output_height, output_width});

  Tensor inputCopy = input;
  Tensor resultCopy = result;
  if (input.scalar_type() == ScalarType::Half) {
    inputCopy = inputCopy.npu_dtype_cast(ScalarType::Float);
    resultCopy = resultCopy.npu_dtype_cast(ScalarType::Float);
  }

  OpCommand cmd;
  cmd.Name("UpsampleNearest3d")
    .Input(inputCopy)
    .Output(resultCopy)
    .Attr("output_size", output_size)
    .Run();
  if (input.scalar_type() == ScalarType::Half) {
    resultCopy = resultCopy.npu_dtype_cast(ScalarType::Half);
  }
  result.copy_(resultCopy);
  return result;
}

Tensor& upsample_nearest3d_out_npu(
    Tensor& result,
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  OpPreparation::CheckOut(
      {input},
      result,
      input,
      {1});
  
  if (!NpuUtils::check_match(&result)) {
    Tensor contiguousResult = NpuUtils::format_contiguous(result);
    upsample_nearest3d_out_npu_nocheck(
        contiguousResult, input, output_size, scales_d, scales_h, scales_w);
    NpuUtils::format_fresh_view(result, contiguousResult);
  } else {
    upsample_nearest3d_out_npu_nocheck(
        result, input, output_size, scales_d, scales_h, scales_w);
  }
  return result;
}

Tensor upsample_nearest3d_npu(
    const Tensor& input,
    IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {

  Tensor result = OpPreparation::ApplyTensor(input, {1});

  upsample_nearest3d_out_npu_nocheck(result, input, output_size, scales_d, scales_h, scales_w);

  return result;
}

} // namespace native
} // namespace at
