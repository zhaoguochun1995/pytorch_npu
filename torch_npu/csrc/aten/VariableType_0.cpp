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

#include <torch/csrc/autograd/VariableTypeUtils.h>

#include <ATen/RedispatchFunctions.h>
#include <torch/library.h>

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/VariableType.h"
#include "torch_npu/csrc/framework/autograd/FunctionsManual.h"


// @generated from /hwtest/zhaoguochun/op-plugin/build/pytorch/codegen/autograd/templates/VariableType.cpp

// NOTE [Sharded File]: on this file's split-into-shards state
//
// Back in the good old days, VariableType.cpp was generated as one
// file with every function in it, and everything was great and
// simple.
//
// However, this file was also very large (over 36,000 lines), and
// compiling it was very slow, and in fact was a significant
// bottleneck for incremental rebuilds. To address this, we now
// generate the file split across multiple shards, named
// VariableType_0.cpp and so on, which can be compiled in parallel.
//
// For ease of inspection and debugging, so that it's not necessary to
// go rooting around in multiple files, we also generate all the
// functions together in VariableTypeEverything.cpp. This generated
// file is only for convenience; it's not actually used in the
// build. If the file you're looking at now is one of the shards, you
// may want to switch over to the Everything variant to make you
// grepping smoother.

using namespace at;
using namespace at_npu::autograd::generated;
using namespace at_npu::autograd::generated::details;

namespace at_npu { namespace autograd {

namespace VariableType {
namespace {
  C10_UNUSED void reset_grad_accumulator(Variable & self) {
    AutogradMeta* meta = torch::autograd::impl::get_autograd_meta(self);
    if (meta != nullptr) {
      meta->grad_accumulator_.reset();
    }
  }
}

namespace {
at::Tensor binary_cross_entropy_with_logits(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & pos_weight, int64_t reduction) {
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  
  (void)_any_requires_grad;
  check_no_requires_grad(target, "target", "binary_cross_entropy_with_logits");
  check_no_requires_grad(weight, "weight", "binary_cross_entropy_with_logits");
  check_no_requires_grad(pos_weight, "pos_weight", "binary_cross_entropy_with_logits");
  std::shared_ptr<BinaryCrossEntropyWithLogitsBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<BinaryCrossEntropyWithLogitsBackward0>(new BinaryCrossEntropyWithLogitsBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->weight_ = SavedVariable(weight, false);
    grad_fn->pos_weight_ = SavedVariable(pos_weight, false);
    grad_fn->reduction = reduction;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::binary_cross_entropy_with_logits(ks & c10::after_autograd_keyset, self_, target_, weight, pos_weight, reduction);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  if (result.has_storage()) AT_ASSERT(result.storage().use_count() == 1, "function: binary_cross_entropy_with_logits");
  AT_ASSERT(result.use_count() <= 1, "function: binary_cross_entropy_with_logits");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "binary_cross_entropy_with_logits");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(target) || isFwGradDefined(weight) || isFwGradDefined(pos_weight)), "Trying to use forward AD with binary_cross_entropy_with_logits that does not support it.");
  return result;
}
at::Tensor kl_div(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, int64_t reduction, bool log_target) {
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  
  (void)_any_requires_grad;
  check_no_requires_grad(target, "target", "kl_div");
  std::shared_ptr<KlDivBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<KlDivBackward0>(new KlDivBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->reduction = reduction;
    grad_fn->log_target = log_target;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::kl_div(ks & c10::after_autograd_keyset, self_, target_, reduction, log_target);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  if (result.has_storage()) AT_ASSERT(result.storage().use_count() == 1, "function: kl_div");
  AT_ASSERT(result.use_count() <= 1, "function: kl_div");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "kl_div");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(target)), "Trying to use forward AD with kl_div that does not support it.");
  return result;
}
at::Tensor selu(c10::DispatchKeySet ks, const at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  
  (void)_any_requires_grad;
  std::shared_ptr<SeluBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SeluBackward0>(new SeluBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::selu(ks & c10::after_autograd_keyset, self_);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (result.has_storage()) AT_ASSERT(result.storage().use_count() == 1, "function: selu");
  AT_ASSERT(result.use_count() <= 1, "function: selu");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "selu");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with selu that does not support it.");
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(result, true);
  }
  return result;
}
at::Tensor & selu_(c10::DispatchKeySet ks, at::Tensor & self) {
  auto& self_ = unpack(self, "self", 0);
  auto _any_requires_grad = compute_requires_grad( self );
  
  (void)_any_requires_grad;
  check_inplace(self, _any_requires_grad);
  c10::optional<at::Tensor> original_self;
  std::shared_ptr<SeluBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<SeluBackward0>(new SeluBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  {
    at::AutoDispatchBelowAutograd guard;
    at::redispatch::selu_(ks & c10::after_autograd_keyset, self_);
  }
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  #endif
  if (grad_fn) {
      rebase_history(flatten_tensor_args( self ), grad_fn);
  }
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self)), "Trying to use forward AD with selu_ that does not support it.");
  if (grad_fn) {
    grad_fn->result_ = SavedVariable(self, true, self.is_view());
  }
  return self;
}
at::Tensor l1_loss(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & target, int64_t reduction) {
  auto& self_ = unpack(self, "self", 0);
  auto& target_ = unpack(target, "target", 1);
  auto _any_requires_grad = compute_requires_grad( self );
  
  (void)_any_requires_grad;
  check_no_requires_grad(target, "target", "l1_loss");
  std::shared_ptr<L1LossBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<L1LossBackward0>(new L1LossBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
    grad_fn->self_ = SavedVariable(self, false);
    grad_fn->target_ = SavedVariable(target, false);
    grad_fn->reduction = reduction;
  }
  #ifndef NDEBUG
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  c10::optional<Storage> target__storage_saved =
    target_.has_storage() ? c10::optional<Storage>(target_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> target__impl_saved;
  if (target_.defined()) target__impl_saved = target_.getIntrusivePtr();
  #endif
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::l1_loss(ks & c10::after_autograd_keyset, self_, target_, reduction);
  })();
  auto result = std::move(_tmp);
  #ifndef NDEBUG
  if (self__storage_saved.has_value())
    AT_ASSERT(self__storage_saved.value().is_alias_of(self_.storage()));
  if (self__impl_saved) AT_ASSERT(self__impl_saved == self_.getIntrusivePtr());
  if (target__storage_saved.has_value())
    AT_ASSERT(target__storage_saved.value().is_alias_of(target_.storage()));
  if (target__impl_saved) AT_ASSERT(target__impl_saved == target_.getIntrusivePtr());
  if (result.has_storage()) AT_ASSERT(result.storage().use_count() == 1, "function: l1_loss");
  AT_ASSERT(result.use_count() <= 1, "function: l1_loss");
  #endif
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  throw_error_for_complex_autograd(result, "l1_loss");
  TORCH_CHECK_NOT_IMPLEMENTED(!(isFwGradDefined(self) || isFwGradDefined(target)), "Trying to use forward AD with l1_loss that does not support it.");
  return result;
}
}
} // namespace VariableType

namespace {

TORCH_LIBRARY_IMPL(aten, AutogradXLA, m) {
  m.impl("binary_cross_entropy_with_logits",
         TORCH_FN(VariableType::binary_cross_entropy_with_logits)
  );
  m.impl("kl_div",
         TORCH_FN(VariableType::kl_div)
  );
  m.impl("selu",
         TORCH_FN(VariableType::selu)
  );
  m.impl("selu_",
         TORCH_FN(VariableType::selu_)
  );
  m.impl("l1_loss",
         TORCH_FN(VariableType::l1_loss)
  );
}

}

}} // namespace at_npu::autograd
