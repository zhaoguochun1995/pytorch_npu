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

#include <ATen/Tensor.h>
#include <c10/util/SmallVector.h>

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include "torch_npu/csrc/framework/utils/KernelNpuOutputSize.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

namespace at_npu {
namespace native {

static inline c10::SmallVector<int64_t, N> expand_dim(at::IntArrayRef list_param, const char *param_name,
                                                      int64_t expected_dim) {
  if (list_param.size() == 1) {
    c10::SmallVector<int64_t, N> expand_dim_param_vec;
    for (int64_t i = 0; i < expected_dim; i++) {
      expand_dim_param_vec.emplace_back(list_param[0]);
    }
    return expand_dim_param_vec;
  } else {
    return CalcuOpUtil::ConvertIntArrayRefToSmallVector(list_param);
  }
}

at::Tensor NPUNativeOpApiFunctions::convolution(const at::Tensor &input, const at::Tensor &weight,
                                                const c10::optional<at::Tensor> &bias, at::IntArrayRef stride,
                                                at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
                                                at::IntArrayRef output_padding, int64_t groups) {
  DO_COMPATIBILITY(aclnnAdd, NPUNativeFunctions::convolution(input, weight, bias, stride, padding, dilation, transposed,
                                                             output_padding, groups));

  int64_t k = weight.ndimension();
  int64_t dim = k - 2;
  bool unBatch = false;

  // Groups > 1 and 3D scenes are currently not supported (binary operator problem), and path 3 implementation is
  // temporarily called
  if (dim == 3 || groups > 1) {
    return at::_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, false,
                            false, false);
  }

  c10::SmallVector<int64_t, N> stride_expand = expand_dim(stride, "stride", dim);
  stride = at::IntArrayRef(stride_expand);

  c10::SmallVector<int64_t, N> padding_expand = expand_dim(padding, "padding", dim);
  padding = at::IntArrayRef(padding_expand);

  c10::SmallVector<int64_t, N> dilation_expand = expand_dim(dilation, "dilation", dim);
  dilation = at::IntArrayRef(dilation_expand);

  c10::SmallVector<int64_t, N> output_padding_expend = expand_dim(output_padding, "output_padding", dim);
  output_padding = at::IntArrayRef(output_padding_expend);

  c10::SmallVector<int64_t, SIZE> out_size;

  if (!transposed) {
    if (dim == 1) {
      out_size = conv1d_npu_output_size(input, weight, padding, stride, dilation);
    } else if (dim == 2) {
      out_size = conv2d_npu_output_size(input, weight, padding, stride, dilation);
    } else {
      return at::Tensor();
    }

  } else {
    const at::Tensor &bias_tensor = c10::value_or_else(bias, [] { return at::Tensor(); });
    if (dim == 1) {
      out_size = conv_transpose1d_npu_output_size(input, weight, bias_tensor, padding, output_padding, stride, dilation,
                                                  groups);
    } else if (dim == 2) {
      if (input.ndimension() == 3) {
        unBatch = true;
        c10::SmallVector<int64_t, SIZE> unsqueeze_size = {1, input.size(0), input.size(1), input.size(2)};
        input.resize_(unsqueeze_size);
      }

      out_size = conv_transpose2d_npu_output_size(input, weight, bias_tensor, padding, output_padding, stride, dilation,
                                                  groups);
    } else {
      return at::Tensor();
    }
  }
  auto output = OpPreparation::ApplyTensorWithSizes(out_size, input.options());
  EXEC_NPU_CMD(aclnnConvolution, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups,
               output);

  if (unBatch) {
    c10::SmallVector<int64_t, SIZE> squeeze_size = {output.size(1), output.size(2), output.size(3)};
    output.resize_(squeeze_size);
  }

  return output;
}
}  // namespace native
}  // namespace at_npu
