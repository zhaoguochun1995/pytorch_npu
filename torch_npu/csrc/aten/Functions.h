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

#pragma once

// @generated from /hwtest/zhaoguochun/op-plugin/build/pytorch/codegen/autograd/templates/Functions.h

#include <ATen/ATen.h>
#include <ATen/core/functional.h>
#include <ATen/TensorGeometry.h>

#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <torch/csrc/Export.h>

using namespace torch::autograd;

namespace at_npu { namespace autograd { namespace generated {

using at::Scalar;
using at::Tensor;
using at::IntArrayRef;
using at::ArrayRef;
using at::Type;
using at::TensorGeometry;
using at::ScalarType;
using c10::optional;
using c10::fmap;

inline std::vector<Tensor> unpack_list(at::ArrayRef<SavedVariable> xs) {
  // NB: we must explicitly do the conversion in the lambda, otherwise template
  // deduction will give a Tensor of Variable which is not convertible
  return fmap(xs, [](const SavedVariable& x) {
    return static_cast<Tensor>(x.unpack());
  });
}

inline c10::List<c10::optional<Tensor>> unpack_opt_list(at::ArrayRef<SavedVariable> xs) {
  torch::List<c10::optional<Tensor>> result;
  result.reserve(xs.size());
  for (const SavedVariable& v : xs) {
    result.push_back(v.unpack());
  }
  return result;
}

struct TypeAndSize {
    TypeAndSize() : options(at::TensorOptions()) {}
    /* implicit */
    TypeAndSize(const Tensor & t)
        : sizes(t.sizes().vec())
        , options(t.options()) {}

    Tensor zeros() { return at::zeros(sizes, options); }

private:
    std::vector<int64_t> sizes;
    at::TensorOptions options;
};

struct TORCH_API NpuMultiHeadAttentionBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NpuMultiHeadAttentionBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    query_.reset_data();
    key_.reset_data();
    value_.reset_data();
    query_weight_.reset_data();
    key_weight_.reset_data();
    value_weight_.reset_data();
    out_proj_weight_.reset_data();
    query_bias_.reset_data();
    key_bias_.reset_data();
    value_bias_.reset_data();
    out_proj_bias_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
    result3_.reset_data();
    result4_.reset_data();
    result5_.reset_data();
    result6_.reset_data();
    result7_.reset_data();
  }

  SavedVariable query_;
  SavedVariable key_;
  SavedVariable value_;
  SavedVariable query_weight_;
  SavedVariable key_weight_;
  SavedVariable value_weight_;
  SavedVariable out_proj_weight_;
  SavedVariable query_bias_;
  SavedVariable key_bias_;
  SavedVariable value_bias_;
  SavedVariable out_proj_bias_;
  int64_t attn_head_num = 0;
  int64_t attn_dim_per_head = 0;
  int64_t src_len = 0;
  int64_t tgt_len = 0;
  double dropout_prob;
  bool softmax_use_float;
  SavedVariable result1_;
  SavedVariable result2_;
  SavedVariable result3_;
  SavedVariable result4_;
  SavedVariable result5_;
  SavedVariable result6_;
  SavedVariable result7_;

};
struct TORCH_API NpuFusionAttentionBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NpuFusionAttentionBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    query_.reset_data();
    key_.reset_data();
    value_.reset_data();
    pse_.reset_data();
    padding_mask_.reset_data();
    atten_mask_.reset_data();
    result0_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
    result3_.reset_data();
  }

  SavedVariable query_;
  SavedVariable key_;
  SavedVariable value_;
  int64_t head_num = 0;
  std::string input_layout;
  SavedVariable pse_;
  SavedVariable padding_mask_;
  SavedVariable atten_mask_;
  double scale;
  double keep_prob;
  int64_t pre_tockens = 0;
  int64_t next_tockens = 0;
  int64_t inner_precise = 0;
  bool gen_mask_parallel;
  bool sync;
  SavedVariable result0_;
  SavedVariable result1_;
  SavedVariable result2_;
  SavedVariable result3_;
  int64_t result4 = 0;
  int64_t result5 = 0;
  int64_t result6 = 0;

};
struct TORCH_API NpuGruBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NpuGruBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    hx_.reset_data();
    weight_input_.reset_data();
    weight_hidden_.reset_data();
    bias_input_.reset_data();
    bias_hidden_.reset_data();
    seq_length_.reset_data();
    result0_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
    result3_.reset_data();
    result4_.reset_data();
    result5_.reset_data();
  }

  SavedVariable input_;
  SavedVariable hx_;
  SavedVariable weight_input_;
  SavedVariable weight_hidden_;
  SavedVariable bias_input_;
  SavedVariable bias_hidden_;
  SavedVariable seq_length_;
  SavedVariable result0_;
  SavedVariable result1_;
  SavedVariable result2_;
  SavedVariable result3_;
  SavedVariable result4_;
  SavedVariable result5_;

};
struct TORCH_API NpuLstmBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NpuLstmBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    weight_.reset_data();
    bias_.reset_data();
    h_.reset_data();
    c_.reset_data();
    result0_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
    result3_.reset_data();
    result4_.reset_data();
    result5_.reset_data();
    result6_.reset_data();
    result7_.reset_data();
  }

  SavedVariable input_;
  SavedVariable weight_;
  SavedVariable bias_;
  SavedVariable h_;
  SavedVariable c_;
  SavedVariable result0_;
  SavedVariable result1_;
  SavedVariable result2_;
  SavedVariable result3_;
  SavedVariable result4_;
  SavedVariable result5_;
  SavedVariable result6_;
  SavedVariable result7_;

};
struct TORCH_API NpuLstmDataBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NpuLstmDataBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    batch_sizes_.reset_data();
    weight_.reset_data();
    bias_.reset_data();
    h_.reset_data();
    c_.reset_data();
    result0_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
    result3_.reset_data();
    result4_.reset_data();
    result5_.reset_data();
    result6_.reset_data();
    result7_.reset_data();
  }

  SavedVariable input_;
  SavedVariable batch_sizes_;
  SavedVariable weight_;
  SavedVariable bias_;
  SavedVariable h_;
  SavedVariable c_;
  bool direction;
  SavedVariable result0_;
  SavedVariable result1_;
  SavedVariable result2_;
  SavedVariable result3_;
  SavedVariable result4_;
  SavedVariable result5_;
  SavedVariable result6_;
  SavedVariable result7_;

};
struct TORCH_API NpuLstmCellBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NpuLstmCellBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    w_ih_.reset_data();
    w_hh_.reset_data();
    h_.reset_data();
    c_.reset_data();
    result0_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
    result3_.reset_data();
    result4_.reset_data();
    result5_.reset_data();
    result6_.reset_data();
    result7_.reset_data();
  }

  SavedVariable input_;
  SavedVariable w_ih_;
  SavedVariable w_hh_;
  SavedVariable h_;
  SavedVariable c_;
  SavedVariable result0_;
  SavedVariable result1_;
  SavedVariable result2_;
  SavedVariable result3_;
  SavedVariable result4_;
  SavedVariable result5_;
  SavedVariable result6_;
  SavedVariable result7_;

};
struct TORCH_API DropoutWithByteMaskBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "DropoutWithByteMaskBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result1_.reset_data();
  }

  double p;
  SavedVariable result1_;

};
struct TORCH_API NpuDropoutBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NpuDropoutBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result1_.reset_data();
  }

  double p;
  SavedVariable result1_;

};
struct TORCH_API NpuDropoutDoMaskBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NpuDropoutDoMaskBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result1_.reset_data();
  }

  double p;
  SavedVariable result1_;

};
struct TORCH_API NpuCiouBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NpuCiouBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    gtboxes_.reset_data();
    result1_.reset_data();
  }

  SavedVariable self_;
  SavedVariable gtboxes_;
  bool trans;
  bool is_cross;
  int64_t mode = 0;
  SavedVariable result1_;

};
struct TORCH_API NpuFusedAttentionScoreFwdBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NpuFusedAttentionScoreFwdBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    query_layer_.reset_data();
    key_layer_.reset_data();
    value_layer_.reset_data();
    result1_.reset_data();
    result2_.reset_data();
  }

  SavedVariable query_layer_;
  SavedVariable key_layer_;
  SavedVariable value_layer_;
  at::Scalar scale;
  double keep_prob;
  bool query_transpose;
  bool key_transpose;
  bool value_transpose;
  bool dx_transpose;
  SavedVariable result1_;
  SavedVariable result2_;

};
struct TORCH_API NpuBmmv2Backward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NpuBmmv2Backward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    mat2_.reset_data();
  }

  SavedVariable self_;
  SavedVariable mat2_;

};
struct TORCH_API NpuMaxBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NpuMaxBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    indices_.reset_data();
  }

  SavedVariable self_;
  int64_t dim = 0;
  bool keepdim;
  SavedVariable indices_;

};
struct TORCH_API NpuMinBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NpuMinBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    indices_.reset_data();
  }

  SavedVariable self_;
  int64_t dim = 0;
  bool keepdim;
  SavedVariable indices_;

};
struct TORCH_API NpuPsRoiPoolingBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NpuPsRoiPoolingBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    rois_.reset_data();
  }

  SavedVariable self_;
  SavedVariable rois_;
  double spatial_scale;
  int64_t group_size = 0;
  int64_t output_dim = 0;

};
struct TORCH_API NpuConfusionTransposeBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NpuConfusionTransposeBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;
  std::vector<int64_t> perm;
  bool transpose_first;

};
struct TORCH_API SeluBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "SeluBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result_.reset_data();
  }

  SavedVariable result_;

};
struct TORCH_API NpuSiluBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NpuSiluBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    result_.reset_data();
  }

  SavedVariable self_;
  SavedVariable result_;

};
struct TORCH_API FastGeluBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "FastGeluBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API NpuRotaryMulBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NpuRotaryMulBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    r1_.reset_data();
    r2_.reset_data();
  }

  SavedVariable self_;
  SavedVariable r1_;
  SavedVariable r2_;

};
struct TORCH_API NpuDiouBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NpuDiouBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    gtboxes_.reset_data();
  }

  SavedVariable self_;
  SavedVariable gtboxes_;
  bool trans;
  bool is_cross;
  int64_t mode = 0;

};
struct TORCH_API NpuGiouBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NpuGiouBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    gtboxes_.reset_data();
  }

  SavedVariable self_;
  SavedVariable gtboxes_;
  bool trans;
  bool is_cross;
  int64_t mode = 0;

};
struct TORCH_API NpuMishBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NpuMishBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
  }

  SavedVariable self_;

};
struct TORCH_API NpuSoftmaxCrossEntropyWithLogitsBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NpuSoftmaxCrossEntropyWithLogitsBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    labels_.reset_data();
  }

  SavedVariable self_;
  SavedVariable labels_;

};
struct TORCH_API NpuLinearBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NpuLinearBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    weight_.reset_data();
  }

  SavedVariable input_;
  SavedVariable weight_;

};
struct TORCH_API NpuDropoutWithAddSoftmaxBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NpuDropoutWithAddSoftmaxBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    result0_.reset_data();
    result1_.reset_data();
  }

  at::Scalar alpha;
  double prob;
  int64_t dim = 0;
  SavedVariable result0_;
  SavedVariable result1_;

};
struct TORCH_API NpuDtypeCastBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NpuDtypeCastBackward0"; }
  void release_variables() override {


  }

  at::ScalarType self_scalar_type;

};
struct TORCH_API NpuConvolutionTransposeBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NpuConvolutionTransposeBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    weight_.reset_data();
  }

  SavedVariable input_;
  SavedVariable weight_;
  std::vector<int64_t> padding;
  std::vector<int64_t> output_padding;
  std::vector<int64_t> stride;
  std::vector<int64_t> dilation;
  int64_t groups = 0;

};
struct TORCH_API NpuConvolutionBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NpuConvolutionBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    weight_.reset_data();
  }

  SavedVariable input_;
  SavedVariable weight_;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  int64_t groups = 0;

};
struct TORCH_API NpuDeformableConv2DBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NpuDeformableConv2DBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    input_.reset_data();
    weight_.reset_data();
    offset_.reset_data();
    result1_.reset_data();
  }

  SavedVariable input_;
  SavedVariable weight_;
  SavedVariable offset_;
  std::vector<int64_t> kernel_size;
  std::vector<int64_t> stride;
  std::vector<int64_t> padding;
  std::vector<int64_t> dilation;
  int64_t groups = 0;
  int64_t deformable_groups = 0;
  bool modulated;
  SavedVariable result1_;

};
struct TORCH_API NpuScaledMaskedSoftmaxBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NpuScaledMaskedSoftmaxBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    mask_.reset_data();
    result_.reset_data();
  }

  SavedVariable mask_;
  at::Scalar scale;
  bool fixed_triu_mask;
  SavedVariable result_;

};
struct TORCH_API BinaryCrossEntropyWithLogitsBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "BinaryCrossEntropyWithLogitsBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    target_.reset_data();
    weight_.reset_data();
    pos_weight_.reset_data();
  }

  SavedVariable self_;
  SavedVariable target_;
  SavedVariable weight_;
  SavedVariable pos_weight_;
  int64_t reduction = 0;

};
struct TORCH_API KlDivBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "KlDivBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    target_.reset_data();
  }

  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;
  bool log_target;

};
struct TORCH_API L1LossBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "L1LossBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    self_.reset_data();
    target_.reset_data();
  }

  SavedVariable self_;
  SavedVariable target_;
  int64_t reduction = 0;

};
struct TORCH_API NpuFormatCastBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "NpuFormatCastBackward0"; }
  void release_variables() override {


  }



};

}}} // namespace at_npu::autograd::generated
