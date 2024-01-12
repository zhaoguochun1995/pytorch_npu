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

#include "torch_npu/csrc/framework/autograd/FunctionsManual.h"
#include "op_plugin/OpInterface.h"
// @generated from /hwtest/zhaoguochun/op-plugin/build/pytorch/codegen/autograd/templates/Functions.cpp

// The manual function definitions that used to be here are now in torch/csrc/autograd/FunctionsManual.cpp
// This speeds up re-compilation and allow to share these implementations so that they can be
// used for forward mode AD formulas as well.

using namespace at_npu::autograd::generated::details;
using namespace op_plugin;
using at::Tensor;
using at::Scalar;
using at::IntArrayRef;
using at::TensorList;

namespace at_npu { namespace autograd { namespace generated {

variable_list NpuMultiHeadAttentionBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto query_ix = gen.range(1);
  auto key_ix = gen.range(1);
  auto value_ix = gen.range(1);
  auto query_weight_ix = gen.range(1);
  auto key_weight_ix = gen.range(1);
  auto value_weight_ix = gen.range(1);
  auto out_proj_weight_ix = gen.range(1);
  auto query_bias_ix = gen.range(1);
  auto key_bias_ix = gen.range(1);
  auto value_bias_ix = gen.range(1);
  auto out_proj_bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto query = query_.unpack();
  auto key = key_.unpack();
  auto value = value_.unpack();
  auto query_weight = query_weight_.unpack();
  auto key_weight = key_weight_.unpack();
  auto value_weight = value_weight_.unpack();
  auto out_proj_weight = out_proj_weight_.unpack();
  auto query_bias = query_bias_.unpack();
  auto key_bias = key_bias_.unpack();
  auto value_bias = value_bias_.unpack();
  auto out_proj_bias = out_proj_bias_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  auto result4 = result4_.unpack(shared_from_this());
  auto result5 = result5_.unpack(shared_from_this());
  auto result6 = result6_.unpack(shared_from_this());
  auto result7 = result7_.unpack(shared_from_this());
  if (should_compute_output({ query_weight_ix, key_weight_ix, value_weight_ix, out_proj_weight_ix, query_ix, key_ix, value_ix, query_bias_ix, key_bias_ix, value_bias_ix, out_proj_bias_ix })) {
  
    auto grad_result = npu_multi_head_attention_backward(query, key, value, query_weight, key_weight, value_weight, out_proj_weight, query_bias, key_bias, value_bias, out_proj_bias, result2, result3, result4, result5, result6, result7, grad, result1, attn_head_num, attn_dim_per_head, src_len, tgt_len, dropout_prob, softmax_use_float);
      if (should_compute_output({ query_weight_ix })) {
        copy_range(grad_inputs, query_weight_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ key_weight_ix })) {
        copy_range(grad_inputs, key_weight_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ value_weight_ix })) {
        copy_range(grad_inputs, value_weight_ix, std::get<2>(grad_result));
      }
      if (should_compute_output({ out_proj_weight_ix })) {
        copy_range(grad_inputs, out_proj_weight_ix, std::get<3>(grad_result));
      }
      if (should_compute_output({ query_ix })) {
        copy_range(grad_inputs, query_ix, std::get<4>(grad_result));
      }
      if (should_compute_output({ key_ix })) {
        copy_range(grad_inputs, key_ix, std::get<5>(grad_result));
      }
      if (should_compute_output({ value_ix })) {
        copy_range(grad_inputs, value_ix, std::get<6>(grad_result));
      }
      if (should_compute_output({ query_bias_ix })) {
        copy_range(grad_inputs, query_bias_ix, std::get<7>(grad_result));
      }
      if (should_compute_output({ key_bias_ix })) {
        copy_range(grad_inputs, key_bias_ix, std::get<8>(grad_result));
      }
      if (should_compute_output({ value_bias_ix })) {
        copy_range(grad_inputs, value_bias_ix, std::get<9>(grad_result));
      }
      if (should_compute_output({ out_proj_bias_ix })) {
        copy_range(grad_inputs, out_proj_bias_ix, std::get<10>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list NpuFusionAttentionBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto query_ix = gen.range(1);
  auto key_ix = gen.range(1);
  auto value_ix = gen.range(1);
  auto pse_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto query = query_.unpack();
  auto key = key_.unpack();
  auto value = value_.unpack();
  auto pse = pse_.unpack();
  auto padding_mask = padding_mask_.unpack();
  auto atten_mask = atten_mask_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  if (should_compute_output({ query_ix, key_ix, value_ix, pse_ix })) {
  
    auto grad_result = npu_fusion_attention_grad(query, key, value, grad, head_num, input_layout, pse, padding_mask, atten_mask, result1, result2, result3, result0, scale, keep_prob, pre_tockens, next_tockens, inner_precise, result4, result5, result6, gen_mask_parallel, sync);
      if (should_compute_output({ query_ix })) {
        copy_range(grad_inputs, query_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ key_ix })) {
        copy_range(grad_inputs, key_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ value_ix })) {
        copy_range(grad_inputs, value_ix, std::get<2>(grad_result));
      }
      if (should_compute_output({ pse_ix })) {
        copy_range(grad_inputs, pse_ix, std::get<3>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list NpuGruBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto hx_ix = gen.range(1);
  auto weight_input_ix = gen.range(1);
  auto weight_hidden_ix = gen.range(1);
  auto bias_input_ix = gen.range(1);
  auto bias_hidden_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto input = input_.unpack();
  auto hx = hx_.unpack();
  auto weight_input = weight_input_.unpack();
  auto weight_hidden = weight_hidden_.unpack();
  auto bias_input = bias_input_.unpack();
  auto bias_hidden = bias_hidden_.unpack();
  auto seq_length = seq_length_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  auto result4 = result4_.unpack(shared_from_this());
  auto result5 = result5_.unpack(shared_from_this());
  if (should_compute_output({ weight_input_ix, weight_hidden_ix, input_ix, bias_input_ix, bias_hidden_ix, hx_ix })) {
  
    auto grad_result = npu_gru_backward(grads[0], grads[1], input, weight_input, weight_hidden, bias_input, bias_hidden, seq_length, hx, result0, result1, result2, result3, result4, result5);
      if (should_compute_output({ weight_input_ix })) {
        copy_range(grad_inputs, weight_input_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ weight_hidden_ix })) {
        copy_range(grad_inputs, weight_hidden_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<2>(grad_result));
      }
      if (should_compute_output({ bias_input_ix })) {
        copy_range(grad_inputs, bias_input_ix, std::get<3>(grad_result));
      }
      if (should_compute_output({ bias_hidden_ix })) {
        copy_range(grad_inputs, bias_hidden_ix, std::get<4>(grad_result));
      }
      if (should_compute_output({ hx_ix })) {
        copy_range(grad_inputs, hx_ix, std::get<5>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list NpuLstmBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  auto h_ix = gen.range(1);
  auto c_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto input = input_.unpack();
  auto weight = weight_.unpack();
  auto bias = bias_.unpack();
  auto h = h_.unpack();
  auto c = c_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  auto result4 = result4_.unpack(shared_from_this());
  auto result5 = result5_.unpack(shared_from_this());
  auto result6 = result6_.unpack(shared_from_this());
  auto result7 = result7_.unpack(shared_from_this());
  if (should_compute_output({ input_ix, weight_ix, bias_ix, h_ix, c_ix })) {
  
    auto grad_result = npu_lstm_backward(grads[0], grads[1], grads[2], input, weight, bias, h, c, result0, result1, result2, result3, result4, result5, result6, result7);
      if (should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
      if (should_compute_output({ h_ix })) {
        copy_range(grad_inputs, h_ix, std::get<3>(grad_result));
      }
      if (should_compute_output({ c_ix })) {
        copy_range(grad_inputs, c_ix, std::get<4>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list NpuLstmDataBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  auto h_ix = gen.range(1);
  auto c_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto input = input_.unpack();
  auto batch_sizes = batch_sizes_.unpack();
  auto weight = weight_.unpack();
  auto bias = bias_.unpack();
  auto h = h_.unpack();
  auto c = c_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  auto result4 = result4_.unpack(shared_from_this());
  auto result5 = result5_.unpack(shared_from_this());
  auto result6 = result6_.unpack(shared_from_this());
  auto result7 = result7_.unpack(shared_from_this());
  if (should_compute_output({ input_ix, weight_ix, bias_ix, h_ix, c_ix })) {
  
    auto grad_result = npu_lstm_data_backward(grads[0], grads[1], grads[2], input, batch_sizes, weight, bias, h, c, result0, result1, result2, result3, result4, result5, result6, result7, direction);
      if (should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
      if (should_compute_output({ h_ix })) {
        copy_range(grad_inputs, h_ix, std::get<3>(grad_result));
      }
      if (should_compute_output({ c_ix })) {
        copy_range(grad_inputs, c_ix, std::get<4>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list NpuLstmCellBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto w_ih_ix = gen.range(1);
  auto w_hh_ix = gen.range(1);
  auto h_ix = gen.range(1);
  auto c_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  auto input = input_.unpack();
  auto w_ih = w_ih_.unpack();
  auto w_hh = w_hh_.unpack();
  auto h = h_.unpack();
  auto c = c_.unpack();
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  auto result3 = result3_.unpack(shared_from_this());
  auto result4 = result4_.unpack(shared_from_this());
  auto result5 = result5_.unpack(shared_from_this());
  auto result6 = result6_.unpack(shared_from_this());
  auto result7 = result7_.unpack(shared_from_this());
  if (should_compute_output({ input_ix, w_ih_ix, w_hh_ix, bias_ix, h_ix, c_ix })) {
  
    auto grad_result = npu_lstm_cell_backward(grads[0], grads[1], grads[2], input, w_ih, w_hh, h, c, result0, result1, result2, result3, result4, result5, result6, result7);
      if (should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ w_ih_ix })) {
        copy_range(grad_inputs, w_ih_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ w_hh_ix })) {
        copy_range(grad_inputs, w_hh_ix, std::get<2>(grad_result));
      }
      if (should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<3>(grad_result));
      }
      if (should_compute_output({ h_ix })) {
        copy_range(grad_inputs, h_ix, std::get<4>(grad_result));
      }
      if (should_compute_output({ c_ix })) {
        copy_range(grad_inputs, c_ix, std::get<5>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list DropoutWithByteMaskBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto result1 = result1_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (_dropout_with_byte_mask_backward(grad, result1, p)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list NpuDropoutBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto result1 = result1_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (npu_dropout_backward(grad, result1, p)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list NpuDropoutDoMaskBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto result1 = result1_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (npu_dropout_backward(grad, result1, p)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list NpuCiouBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto gtboxes_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto gtboxes = gtboxes_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  if (should_compute_output({ self_ix, gtboxes_ix })) {
  
    auto grad_result = npu_ciou_backward(grad, self, gtboxes, result1, trans, is_cross, mode);
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ gtboxes_ix })) {
        copy_range(grad_inputs, gtboxes_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list NpuFusedAttentionScoreFwdBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto query_layer_ix = gen.range(1);
  auto key_layer_ix = gen.range(1);
  auto value_layer_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto query_layer = query_layer_.unpack();
  auto key_layer = key_layer_.unpack();
  auto value_layer = value_layer_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  auto result2 = result2_.unpack(shared_from_this());
  if (should_compute_output({ query_layer_ix, key_layer_ix, value_layer_ix })) {
  
    auto grad_result = npu_fused_attention_score_backward(grad, result1, query_layer, key_layer, value_layer, result2, scale, keep_prob, query_transpose, key_transpose, value_transpose, dx_transpose);
      if (should_compute_output({ query_layer_ix })) {
        copy_range(grad_inputs, query_layer_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ key_layer_ix })) {
        copy_range(grad_inputs, key_layer_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ value_layer_ix })) {
        copy_range(grad_inputs, value_layer_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list NpuBmmv2Backward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto mat2_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto mat2 = mat2_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ mat2_ix })) {
    auto grad_result = any_grad_defined ? (npu_bmm_v2_mat2_backward(grad, self, mat2, mat2.sym_sizes())) : Tensor();
    copy_range(grad_inputs, mat2_ix, grad_result);
  }
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (npu_bmm_v2_mat1_backward(grad, self, mat2, self.sym_sizes())) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list NpuMaxBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto indices = indices_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (npu_max_backward(grad, dim, indices, self.sym_sizes(), keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list NpuMinBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto indices = indices_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (npu_min_backward(grad, dim, indices, self.sym_sizes(), keepdim)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list NpuPsRoiPoolingBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto rois = rois_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (npu_ps_roi_pooling_backward(grad, rois, spatial_scale, group_size, output_dim, {self.sym_size(2), self.sym_size(3)})) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list NpuConfusionTransposeBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (npu_confusion_transpose_backward(grad, perm, self.sym_sizes(), !transpose_first)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list SeluBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (selu_backward(grad, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list NpuSiluBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (npu_silu_backward(grad, self, result)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list FastGeluBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (npu_fast_gelu_backward(grad, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list NpuRotaryMulBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto r1_ix = gen.range(1);
  auto r2_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto r1 = r1_.unpack();
  auto r2 = r2_.unpack();
  if (should_compute_output({ self_ix, r1_ix, r2_ix })) {
  
    auto grad_result = npu_rotary_mul_backward(grad, self, r1, r2);
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ r1_ix })) {
        copy_range(grad_inputs, r1_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ r2_ix })) {
        copy_range(grad_inputs, r2_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list NpuDiouBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto gtboxes_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto gtboxes = gtboxes_.unpack();
  if (should_compute_output({ self_ix, gtboxes_ix })) {
  
    auto grad_result = npu_diou_backward(grad, self, gtboxes, trans, is_cross, mode);
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ gtboxes_ix })) {
        copy_range(grad_inputs, gtboxes_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list NpuGiouBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto gtboxes_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto gtboxes = gtboxes_.unpack();
  if (should_compute_output({ self_ix, gtboxes_ix })) {
  
    auto grad_result = npu_giou_backward(grad, self, gtboxes, trans, is_cross, mode);
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ gtboxes_ix })) {
        copy_range(grad_inputs, gtboxes_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list NpuMishBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (npu_mish_backward(grad, self)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list NpuSoftmaxCrossEntropyWithLogitsBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto labels = labels_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (npu_softmax_cross_entropy_with_logits_backward(grad, self, labels)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list NpuLinearBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto input = input_.unpack();
  auto weight = weight_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ bias_ix })) {
    auto grad_result = any_grad_defined ? (maybe_multiply(grad, 1)) : Tensor();
    copy_range(grad_inputs, bias_ix, grad_result);
  }
  if (should_compute_output({ input_ix, weight_ix })) {
  
    auto grad_result = npu_linear_backward(grad, input, weight);
      if (should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list NpuDropoutWithAddSoftmaxBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  auto x1_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto result0 = result0_.unpack(shared_from_this());
  auto result1 = result1_.unpack(shared_from_this());
  if (should_compute_output({ self_ix, x1_ix })) {
  
    auto grad_result = npu_dropout_with_add_softmax_backward(grad, result0, result1, alpha, prob, dim);
      if (should_compute_output({ self_ix })) {
        copy_range(grad_inputs, self_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ x1_ix })) {
        copy_range(grad_inputs, x1_ix, std::get<1>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list NpuDtypeCastBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (npu_dtype_cast_backward(grad, self_scalar_type)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list NpuConvolutionTransposeBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto input = input_.unpack();
  auto weight = weight_.unpack();
  if (should_compute_output({ input_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ input_ix }),
        should_compute_output({ weight_ix }),
        should_compute_output({ bias_ix }),
      };
    auto grad_result = npu_convolution_transpose_backward(input, grad, weight, padding, output_padding, stride, dilation, groups, grad_input_mask);
      if (should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list NpuConvolutionBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto input = input_.unpack();
  auto weight = weight_.unpack();
  if (should_compute_output({ input_ix, weight_ix, bias_ix })) {
      auto grad_input_mask = std::array<bool, 3>{
        should_compute_output({ input_ix }),
        should_compute_output({ weight_ix }),
        should_compute_output({ bias_ix }),
      };
    auto grad_result = npu_convolution_backward(input, grad, weight, stride, padding, dilation, groups, grad_input_mask);
      if (should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<2>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list NpuDeformableConv2DBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto input_ix = gen.range(1);
  auto weight_ix = gen.range(1);
  auto offset_ix = gen.range(1);
  auto bias_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto input = input_.unpack();
  auto weight = weight_.unpack();
  auto offset = offset_.unpack();
  auto result1 = result1_.unpack(shared_from_this());
  if (should_compute_output({ input_ix, weight_ix, offset_ix, bias_ix })) {
  
    auto grad_result = npu_deformable_conv2dbk(input, grad, result1, weight, offset, kernel_size, stride, padding, dilation, groups, deformable_groups, modulated);
      if (should_compute_output({ input_ix })) {
        copy_range(grad_inputs, input_ix, std::get<0>(grad_result));
      }
      if (should_compute_output({ weight_ix })) {
        copy_range(grad_inputs, weight_ix, std::get<1>(grad_result));
      }
      if (should_compute_output({ offset_ix })) {
        copy_range(grad_inputs, offset_ix, std::get<2>(grad_result));
      }
      if (should_compute_output({ bias_ix })) {
        copy_range(grad_inputs, bias_ix, std::get<3>(grad_result));
      }
  }
  return grad_inputs;
}
variable_list NpuScaledMaskedSoftmaxBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto x_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto mask = mask_.unpack();
  auto result = result_.unpack(shared_from_this());
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ x_ix })) {
    auto grad_result = any_grad_defined ? (npu_scaled_masked_softmax_backward(grad, result, mask, scale, fixed_triu_mask)) : Tensor();
    copy_range(grad_inputs, x_ix, grad_result);
  }
  return grad_inputs;
}
variable_list BinaryCrossEntropyWithLogitsBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto target = target_.unpack();
  auto weight = weight_.unpack();
  auto pos_weight = pos_weight_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (npu_binary_cross_entropy_with_logits_backward(grad, self, target, weight, pos_weight, reduction)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list KlDivBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto target = target_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (kl_div_backward(grad, self, target, reduction, log_target)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list L1LossBackward0::apply(variable_list&& grads) {
  std::lock_guard<std::mutex> lock(mutex_);

  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  auto self = self_.unpack();
  auto target = target_.unpack();
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (l1_loss_backward(grad, self, target, reduction)) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}
variable_list NpuFormatCastBackward0::apply(variable_list&& grads) {


  IndexRangeGenerator gen;
  auto self_ix = gen.range(1);
  variable_list grad_inputs(gen.size());
  const auto& grad = grads[0];
  bool any_grad_defined = any_variable_defined(grads);
  if (should_compute_output({ self_ix })) {
    auto grad_result = any_grad_defined ? (grad) : Tensor();
    copy_range(grad_inputs, self_ix, grad_result);
  }
  return grad_inputs;
}

}}} // namespace at_npu::autograd::generated
