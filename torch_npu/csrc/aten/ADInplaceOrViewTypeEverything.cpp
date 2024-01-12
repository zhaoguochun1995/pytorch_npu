// Copyright (c) 2023 Huawei Technologies Co., Ltd
// Copyright (c) 2022, Facebook CORPORATION.
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

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <torch/csrc/autograd/VariableTypeUtils.h>

#include <torch/library.h>
// @generated from /hwtest/zhaoguochun/op-plugin/build/pytorch/codegen/autograd/templates/ADInplaceOrViewType.cpp

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Operators.h>

#endif
#include "op_plugin/OpInterface.h"

using namespace at;
using torch::autograd::CreationMeta;
using torch::autograd::as_view;
using torch::autograd::increment_version;

namespace at_npu {

namespace ADInplaceOrView {

namespace {
at::Tensor & npu_silu_(c10::DispatchKeySet ks, at::Tensor & self) {
  {
    at::AutoDispatchBelowADInplaceOrView guard;
    op_plugin::npu_silu_(self);
  }
  increment_version(self);
  return self;
}
}  // namespace
}  // namespace ADInplaceOrView

namespace {

TORCH_LIBRARY_IMPL(aten, ADInplaceOrView, m) {
  m.impl("npu_silu_",
         TORCH_FN(ADInplaceOrView::npu_silu_)
  );
}

}  // namespace
} // namespace at_npu
