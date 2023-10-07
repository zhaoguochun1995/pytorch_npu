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

#include <ciso646>
#include <algorithm>
#include <numeric>
#include <functional>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/LegacyBatchedTensorImpl.h>
#include <ATen/ScalarOps.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/Utils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/grad_mode.h>
#include <ATen/native/Activation.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/OptionalArrayRef.h>
#include <c10/util/SmallBuffer.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>

#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/functions/basic_ops.h>

#include "FunctionsManual.h"

#include "op_plugin/OpInterface.h"

// Helper functions for autogenerated code
// These used to be inlined into the codegened Functions.cpp

namespace at_npu {
namespace autograd {
namespace generated {
namespace details {

using at::Tensor;
using at::Scalar;
using at::IntArrayRef;
using at::TensorList;
using at::areAnyTensorSubclassLike;

Tensor apply_loss_reduction(const Tensor& unreduced, int64_t reduction) {
  if (reduction == at::Reduction::Mean) {
    return unreduced.mean();
  } else if (reduction == at::Reduction::Sum) {
    return unreduced.sum();
  }
  return unreduced;
}

bool any_variable_defined(const variable_list& variables) {
  for (const auto& variable : variables) {
    if (variable.defined()) {
      return true;
    }
  }
  return false;
}

bool isDefined(const c10::optional<Tensor>& t) {
  return t.has_value() && t->defined();
}

Tensor toNonOptTensor(const c10::optional<Tensor>& t) {
  return t.has_value() ? *t : Tensor();
}

Tensor toNonOptFwGrad(const c10::optional<Tensor>& t) {
  return (t.has_value() && t->defined()) ? t->_fw_grad(/*level */ 0) : Tensor();
}

Tensor toNonOptPrimal(const c10::optional<Tensor>& t) {
  return (t.has_value() && t->defined()) ? t->_fw_primal(/*level */ 0) : Tensor();
}

void copy_range(variable_list& out, IndexRange range, const Tensor& t) {
  AT_ASSERT(range.second <= out.size());
  AT_ASSERTM(range.second - range.first == 1, "inconsistent range for Tensor output");
  out[range.first] = t;
}

void copy_range(variable_list& out, IndexRange range, at::ArrayRef<Tensor> t) {
  AT_ASSERT(range.second <= out.size());
  AT_ASSERTM(range.second - range.first == t.size(), "inconsistent range for TensorList output");
  std::copy(t.begin(), t.end(), out.begin() + range.first);
}

template <typename T>
T not_implemented_base(const char* name, const char* reason) {
  std::string msg = c10::str("the derivative for '", name, "' is not implemented.");
  if (strlen(reason) > 0) {
    msg = c10::str(msg, " ", reason);
  };
  TORCH_CHECK_NOT_IMPLEMENTED(false, msg);
}

Tensor not_implemented(const char* name, const char* reason) {
  return not_implemented_base<Tensor>(name, reason);
}

std::vector<Tensor> not_implemented_list(const char* name, const char* reason) {
  return not_implemented_base<std::vector<Tensor>>(name, reason);
}

Tensor maybe_multiply(const Tensor& t, const Scalar& s) {
  bool is_one = false;
  if (s.isFloatingPoint()) {
    is_one = s.toDouble() == 1;
  } else if (s.isIntegral(true)) {
    is_one = s.toLong() == 1;
  }

  if (is_one) {
    return t;
  } else {
    return t * s;
  }
}

} // namespace details
} // namespace generated
} // namespace autograd
} // namespace at_npu
