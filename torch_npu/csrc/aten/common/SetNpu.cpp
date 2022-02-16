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

// #pragma once
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <TH/THTensor.hpp>

#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/aten/common/ResizeNpu.h"
#include "torch_npu/csrc/framework/StorageDescHelper.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"

namespace at_npu {
namespace native {

c10::StorageImpl* storage_new_npu(caffe2::TypeMeta data_type) {
  c10::StorageImpl* storage =
      c10::make_intrusive<c10::StorageImpl>(
          c10::StorageImpl::use_byte_size_t(),
          0,
          c10_npu::get(),
          true)
          .release();
  return storage;
}

void set_storage_nd_npu(
    at::TensorImpl* self,
    c10::StorageImpl* storage,
    ptrdiff_t storageOffset,
    int nDimension,
    const int64_t* size,
    const int64_t* stride) {
  if (THTensor_getStoragePtr(self) != storage) {
    if (!THTensor_getStoragePtr(self)) {
      AT_ERROR("at::Tensor: invalid null storage");
    }
    auto data_type = self->dtype();
    if (storage != nullptr) {
      c10::raw::intrusive_ptr::incref(storage);
      THTensor_stealAndSetStoragePtr(self, storage);
    } else {
      THTensor_stealAndSetStoragePtr(self, storage_new_npu(data_type));
    }
  }

  if (storageOffset < 0) {
    AT_ERROR("at::Tensor: invalid storage offset");
  }
  self->set_storage_offset(storageOffset);
  resize_nd_npu(self, nDimension, size, stride);
}

void set_storage_npu_(
    at::TensorImpl* self,
    c10::StorageImpl* storage_,
    ptrdiff_t storageOffset_,
    c10::IntArrayRef size_,
    c10::IntArrayRef stride_) {
  if (stride_.data()) {
  }
  set_storage_nd_npu(
      self,
      storage_,
      storageOffset_,
      size_.size(),
      size_.data(),
      stride_.data());
}

at::Tensor& NPUNativeFunctions::set_(at::Tensor& self, c10::Storage src) {
  int64_t new_size = static_cast<int64_t>(src.nbytes() / self.dtype().itemsize());
  set_storage_npu_(
      self.unsafeGetTensorImpl(),
      src.unsafeGetStorageImpl(),
      0,
      {new_size},
      {});
  StorageDescHelper::CopyDesc(self, src);
  return self;
}

at::Tensor& NPUNativeFunctions::set_(
    at::Tensor& self,
    c10::Storage src,
    long storage_offset,
    c10::IntArrayRef size,
    c10::IntArrayRef stride) {
  set_storage_npu_(
      self.unsafeGetTensorImpl(),
      src.unsafeGetStorageImpl(),
      storage_offset,
      size,
      stride);
  StorageDescHelper::CopyDesc(self, src);
  return self;
}

at::Tensor& set_format_npu_(
    at::Tensor& self,
    c10::Storage src,
    long storage_offset,
    long npu_format,
    c10::IntArrayRef size,
    c10::IntArrayRef stride) {
  set_storage_npu_(
      self.unsafeGetTensorImpl(),
      src.unsafeGetStorageImpl(),
      storage_offset,
      size,
      stride);

  StorageDescHelper::SetDesc(self, size, stride, (aclFormat)npu_format);
  return self;
}

at::Tensor& NPUNativeFunctions::set_(at::Tensor& self) {
  set_storage_npu_(self.unsafeGetTensorImpl(), NULL, 0, {0}, {});
  StorageDescHelper::SetDesc(self);
  return self;
}

at::Tensor& NPUNativeFunctions::set_(at::Tensor& self, const at::Tensor& src) {
  at::TensorImpl* self_ = self.unsafeGetTensorImpl();
  at::TensorImpl* src_ = src.unsafeGetTensorImpl();
  if (self_ != src_) {
    set_storage_nd_npu(
        self_,
        THTensor_getStoragePtr(src_),
        src_->storage_offset(),
        src_->dim(),
        THTensor_getSizePtr(src_),
        THTensor_getStridePtr(src_));
  }
  StorageDescHelper::CopyDesc(self, src);
  return self;
}

} // namespace native
} // namespace at_npu
