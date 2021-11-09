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

#include "ATen/native/npu/utils/CalcuOpUtil.h"
#include "ATen/native/npu/utils/OpAdapter.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& pdist_out_npu_nocheck(   
    Tensor& result, 
    const Tensor& self,
    float p) {
  OpCommand cmd;
  cmd.Name("Pdist")
      .Input(self)
      .Attr("p", p)
      .Output(result)
      .Run();

  return result;
}

Tensor _pdist_forward_npu(const Tensor& self, double p) {
  Tensor result;
  if (self.size(0) <= 1) {
    result = OpPreparation::ApplyTensor(self, {0});
  } else {
    // double is not supported in NPU,  type of P needs to be converted from double to float.
    float p_float;
    if (std::isinf(p)) {
      p_float = std::numeric_limits<float>::infinity();
    } else {
      TORCH_CHECK(p <= std::numeric_limits<float>::max(), "npu dose not support float64" );
      p_float = static_cast<float>(p);
    }
    auto outputSize = pdist_npu_output_size(self, p_float);
    result = OpPreparation::ApplyTensor(self, outputSize);
    if(self.size(1) == 0){
      result.fill_(0);
    } else {
      pdist_out_npu_nocheck(result, self, p_float);
    }  
  }
  return result;
}

Tensor pdist_npu(const Tensor& self, double p) {
  TORCH_CHECK(self.dim() == 2,
      "pdist only supports 2D tensors, got: ", self.dim(), "D");
  TORCH_CHECK(at::isFloatingType(self.scalar_type()), "pdist only supports floating-point dtypes");
  TORCH_CHECK(p >= 0, "pdist only supports non-negative p values");

  return at::_pdist_forward(self, p);
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("pdist", TORCH_FN(pdist_npu));
  m.impl("_pdist_forward", TORCH_FN(_pdist_forward_npu));
}
} // namespace native
} // namespace at
