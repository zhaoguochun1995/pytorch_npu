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
#include "ATen/native/npu/utils/CalcuOpUtil.h"

namespace at {
namespace native {
using namespace at::native::npu;

Tensor& scatter_npu_src_nocheck(
    Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& src) {
  OpCommand cmd;
  cmd.Name("ScatterElements")
     .Input(self)
     .Input(index)
     .Input(src)
     .Output(self)
     .Attr("axis", dim)
     .Run();
  return self;
}

Tensor& scatter_npu_src_impl(
    Tensor& self,
    int64_t dim,
    const Tensor& index_ex,
    const Tensor& src) {
  ScalarType selfType = self.scalar_type();
  if (selfType == ScalarType::Half) {
    self = self.npu_dtype_cast(ScalarType::Float);
  }

  Tensor index(index_ex);
  if (index.scalar_type() == ScalarType::Half) {
    index = index.npu_dtype_cast(ScalarType::Float);
  }

  if (!NpuUtils::check_match(&self)) {
    Tensor contiguousSelf = NpuUtils::format_contiguous(self);

    scatter_npu_src_nocheck(contiguousSelf, dim, index, src);
    NpuUtils::format_fresh_view(self, contiguousSelf);
  } else {
    scatter_npu_src_nocheck(self, dim, index, src);
  }

  if(self.scalar_type() != selfType){
    self = self.npu_dtype_cast(ScalarType::Half);
  }

  return self;
}

Tensor& scatter_npu_src(
    Tensor& self,
    int64_t dim,
    const Tensor& index_ex,
    const Tensor& src_ex) {
  Tensor src(src_ex);
  if (src.scalar_type() != self.scalar_type()) {
    src = src.npu_dtype_cast(self.scalar_type());
  }

  scatter_npu_src_impl(self, dim, index_ex, src);
  return self;
}

Tensor& scatter_npu_value(
    Tensor& self,
    int64_t dim,
    const Tensor& index_ex,
    Scalar src) {
  Tensor srcTensor = scalar_to_tensor(src).to(ScalarType::Float);
  srcTensor = CalcuOpUtil::copy_tensor_host_to_device(srcTensor);
  Tensor srcTensor_broadcast = at::npu_broadcast(srcTensor, array_to_small_vector(index_ex.sizes()));

  if (srcTensor_broadcast.scalar_type() != self.scalar_type()) {
    srcTensor_broadcast = srcTensor_broadcast.npu_dtype_cast(self.scalar_type());
  }

  scatter_npu_src_impl(self, dim, index_ex, srcTensor_broadcast);
  return self;
}

TORCH_LIBRARY_IMPL(aten, NPU, m) {
  m.impl("scatter_.src", TORCH_FN(scatter_npu_src));
  m.impl("scatter_.value", TORCH_FN(scatter_npu_value));
}

} // namespace native
} // namespace at