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

#include "ATen/native/npu/utils/OpAdapter.h"
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor cross_dest_output(const Tensor& self, const Tensor& other) {
  bool isSelfWrapped = CalcuOpUtil::is_scalar_wrapped_to_tensor(self);
  return isSelfWrapped ? other : self;
}

int64_t cross_real_dim(optional<int64_t> dim) {
  // -65530 is the default value of dim
  return dim.has_value() ? dim.value() : -65530;
}

Tensor& cross_out_npu(
    Tensor& result, 
    const Tensor& self,
    const Tensor& other,
    optional<int64_t> dim) {
  int64_t realDim = cross_real_dim(dim);
  OpCommand cmd;
  cmd.Name("Cross")
    .Input(self)
    .Input(other)
    .Output(result)
    .Attr("dim", realDim)
    .Run();
  return result;
}

Tensor cross_npu(
    const Tensor& self, 
    const Tensor& other,
    optional<int64_t> dim) {
  auto outputSize = broadcast_ops_npu_output_size(self, other);
  Tensor outputTensor = cross_dest_output(self, other);
  Tensor result = at::empty_with_format(
    outputSize, 
    self.options(),
    CalcuOpUtil::get_tensor_npu_format(outputTensor));
  cross_out_npu(result, self, other, dim);
  return result;
}

} // namespace native
} // namespace at
