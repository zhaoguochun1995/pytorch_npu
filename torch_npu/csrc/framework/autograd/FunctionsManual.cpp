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

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,
at::Tensor, at::Tensor, at::Tensor, at::Tensor> multi_head_attention_backward(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const at::Tensor& query_weight, const at::Tensor& key_weight, const at::Tensor& value_weight,
    const at::Tensor& out_proj_weight, const c10::optional<at::Tensor>& query_bias_opt,
    const c10::optional<at::Tensor>& key_bias_opt, const c10::optional<at::Tensor>& value_bias_opt,
    const c10::optional<at::Tensor>& out_proj_bias_opt, const at::Tensor& query_res,
    const at::Tensor& key_res, const at::Tensor& value_res,
    const at::Tensor& attn_scores, const at::Tensor& attn_res, const at::Tensor& context,
    const at::Tensor& y_grad, const at::Tensor& dropout_mask,
    int64_t attn_head_num, int64_t attn_dim_per_head,
    int64_t src_len, int64_t tgt_len,
    double dropout_prob, bool softmax_use_float) {
  return op_plugin::npu_multi_head_attention_backward(
      query, key, value, query_weight, key_weight, value_weight, out_proj_weight, query_bias_opt, key_bias_opt,
      value_bias_opt, out_proj_bias_opt, query_res, key_res, value_res, attn_scores, attn_res, context, y_grad,
      dropout_mask, attn_head_num, attn_dim_per_head, src_len, tgt_len, dropout_prob, softmax_use_float);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> gru_backward(
    const c10::optional<at::Tensor>& grady_opt,
    const c10::optional<at::Tensor>& gradh_opt,
    const at::Tensor& input,
    const at::Tensor& weight_input,
    const at::Tensor& weight_hidden,
    const at::Tensor& bias_input,
    const at::Tensor& bias_hidden,
    const at::Tensor& seq_length,
    const at::Tensor& init_h,
    const at::Tensor& output_y,
    const at::Tensor& output_h,
    const at::Tensor& output_updata,
    const at::Tensor& output_reset,
    const at::Tensor& output_new,
    const at::Tensor& hidden_new) {
  return op_plugin::npu_gru_backward(
      grady_opt, gradh_opt, input, weight_input, weight_hidden, bias_input, bias_hidden, seq_length, init_h, output_y,
      output_h, output_updata, output_reset, output_new, hidden_new);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> lstm_backward(
    const c10::optional<at::Tensor>& grady_opt,
    const c10::optional<at::Tensor>& gradh_opt,
    const c10::optional<at::Tensor>& gradc_opt,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& init_h,
    const at::Tensor& init_c,
    const at::Tensor& y,
    const at::Tensor& h,
    const at::Tensor& c,
    const at::Tensor& i,
    const at::Tensor& j,
    const at::Tensor& f,
    const at::Tensor& o,
    const at::Tensor& tanhc){
  return op_plugin::npu_lstm_backward(
      grady_opt, gradh_opt, gradc_opt, input, weight, bias, init_h, init_c, y, h, c, i, j, f, o, tanhc);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> lstm_cell_backward(
    const c10::optional<at::Tensor>& grad_y_opt_,
    const c10::optional<at::Tensor>& grad_h_opt_,
    const c10::optional<at::Tensor>& grad_c_opt_,
    const at::Tensor& input,
    const at::Tensor& w_ih,
    const at::Tensor& w_hh,
    const at::Tensor& h,
    const at::Tensor& c,
    const at::Tensor& y_output,
    const at::Tensor& h_output,
    const at::Tensor& c_output,
    const at::Tensor& i,
    const at::Tensor& j,
    const at::Tensor& f,
    const at::Tensor& o,
    const at::Tensor& tanhc){
  return op_plugin::npu_lstm_cell_backward(
      grad_y_opt_, grad_h_opt_, grad_c_opt_, input, w_ih, w_hh, h, c, y_output, h_output, c_output, i, j, f, o, tanhc);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> lstm_data_backward(
    const c10::optional<at::Tensor>& grady_opt,
    const c10::optional<at::Tensor>& gradh_opt,
    const c10::optional<at::Tensor>& gradc_opt,
    const at::Tensor& input,
    const at::Tensor& batch_sizes,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& init_h,
    const at::Tensor& init_c,
    const at::Tensor& y,
    const at::Tensor& h,
    const at::Tensor& c,
    const at::Tensor& i,
    const at::Tensor& j,
    const at::Tensor& f,
    const at::Tensor& o,
    const at::Tensor& tanhc,
    bool flag_direction) {
  return op_plugin::npu_lstm_data_backward(grady_opt, gradh_opt, gradc_opt, input,
      batch_sizes, weight, bias, init_h, init_c, y, h, c, i, j, f, o, tanhc, flag_direction);
}

} // namespace details
} // namespace generated
} // namespace autograd
} // namespace at_npu
