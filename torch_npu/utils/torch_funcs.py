# Copyright (c) 2020 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

import torch_npu
from torch_npu.utils.device_guard import torch_device_guard

def raise_error_empty_with_format(*args, **kwargs):
    raise RuntimeError(f"torch.empty_with_format is deprecated and will be removed in future version. "
                       f"Use torch_npu.empty_with_format instead.")

def raise_error_npu_dropout_gen_mask(*args, **kwargs):
    raise RuntimeError(f"torch.npu_dropout_gen_mask is deprecated and will be removed in future version. "
                       f"Use torch_npu.npu_dropout_gen_mask instead.")

@torch_device_guard
def _tensor(*args, **kwargs):
    return torch_npu.tensor(*args, **kwargs)


@torch_device_guard
def _full(*args, **kwargs):
    return torch_npu.full(*args, **kwargs)


@torch_device_guard
def _randint(*args, **kwargs):
    return torch_npu.randint(*args, **kwargs)


@torch_device_guard
def _range(*args, **kwargs):
    return torch_npu.range(*args, **kwargs)


@torch_device_guard
def _arange(*args, **kwargs):
    return torch_npu.arange(*args, **kwargs)


@torch_device_guard
def _empty_with_format(*args, **kwargs):
    return torch_npu._C._VariableFunctions.empty_with_format(*args, **kwargs)


@torch_device_guard
def _npu_dropout_gen_mask(*args, **kwargs):
    return torch_npu._C._VariableFunctions.npu_dropout_gen_mask(*args, **kwargs)


@torch_device_guard
def _new_device(*args, **kwargs):
    return torch_npu._C.device(*args, **kwargs)

def _generator(*args, **kwargs):
  if 'npu' in str(args) or 'npu' in str(kwargs):
      raise AssertionError(f"Please use torch_npu._C.Generator for npu device.")
  return torch._C.Generator(*args, **kwargs)

def jit_script(obj, optimize=None, _frames_up=0, _rcb=None):
    # (Ascend) Disable extension of torch.jit.script
    return obj


def _as_tensor(*args, **kwargs):
    if isinstance(args[0], torch.Tensor):
        dst_device = args[0].device
    else:
        dst_device = "cpu"

    if kwargs and "device" in kwargs:
        dst_device = kwargs.pop("device")

    return torch._C._VariableFunctions.as_tensor(*args, **kwargs).to(dst_device)



@torch_device_guard
def __efficientzerotensor(*args, **kwargs):
    return torch_npu._efficientzerotensor(*args, **kwargs)


@torch_device_guard
def __pin_memory(*args, **kwargs):
    return torch_npu._pin_memory(*args, **kwargs)


@torch_device_guard
def _bartlett_window(*args, **kwargs):
    return torch_npu.bartlett_window(*args, **kwargs)


@torch_device_guard
def _blackman_window(*args, **kwargs):
    return torch_npu.blackman_window(*args, **kwargs)


@torch_device_guard
def _empty(*args, **kwargs):
    return torch_npu.empty(*args, **kwargs)


@torch_device_guard
def _empty_like(*args, **kwargs):
    return torch_npu.empty_like(*args, **kwargs)


@torch_device_guard
def _empty_strided(*args, **kwargs):
    return torch_npu.empty_strided(*args, **kwargs)


@torch_device_guard
def _eye(*args, **kwargs):
    return torch_npu.eye(*args, **kwargs)


@torch_device_guard
def _from_file(*args, **kwargs):
    return torch_npu.from_file(*args, **kwargs)


@torch_device_guard
def _full_like(*args, **kwargs):
    return torch_npu.full_like(*args, **kwargs)


@torch_device_guard
def _hamming_window(*args, **kwargs):
    return torch_npu.hamming_window(*args, **kwargs)


@torch_device_guard
def _hann_window(*args, **kwargs):
    return torch_npu.hann_window(*args, **kwargs)


@torch_device_guard
def _kaiser_window(*args, **kwargs):
    return torch_npu.kaiser_window(*args, **kwargs)


@torch_device_guard
def _linspace(*args, **kwargs):
    return torch_npu.linspace(*args, **kwargs)


@torch_device_guard
def _logspace(*args, **kwargs):
    return torch_npu.logspace(*args, **kwargs)


@torch_device_guard
def _normal(*args, **kwargs):
    return torch_npu.normal(*args, **kwargs)


@torch_device_guard
def _ones(*args, **kwargs):
    return torch_npu.ones(*args, **kwargs)


@torch_device_guard
def _ones_like(*args, **kwargs):
    return torch_npu.ones_like(*args, **kwargs)


@torch_device_guard
def _rand(*args, **kwargs):
    return torch_npu.rand(*args, **kwargs)


@torch_device_guard
def _rand_like(*args, **kwargs):
    return torch_npu.rand_like(*args, **kwargs)


@torch_device_guard
def _randint_like(*args, **kwargs):
    return torch_npu.randint_like(*args, **kwargs)


@torch_device_guard
def _randn(*args, **kwargs):
    return torch_npu.randn(*args, **kwargs)


@torch_device_guard
def _randn_like(*args, **kwargs):
    return torch_npu.randn_like(*args, **kwargs)


@torch_device_guard
def _randperm(*args, **kwargs):
    return torch_npu.randperm(*args, **kwargs)


@torch_device_guard
def _scalar_tensor(*args, **kwargs):
    return torch_npu.scalar_tensor(*args, **kwargs)


@torch_device_guard
def _tril_indices(*args, **kwargs):
    return torch_npu.tril_indices(*args, **kwargs)


@torch_device_guard
def _triu_indices(*args, **kwargs):
    return torch_npu.triu_indices(*args, **kwargs)


@torch_device_guard
def _zeros(*args, **kwargs):
    return torch_npu.zeros(*args, **kwargs)


@torch_device_guard
def _zeros_like(*args, **kwargs):
    return torch_npu.zeros_like(*args, **kwargs)

def add_torch_funcs():
    torch.tensor = _tensor
    torch.full = _full
    torch.randint = _randint
    torch.range = _range
    torch.arange = _arange
    torch.empty_with_format = raise_error_empty_with_format
    torch_npu.empty_with_format = _empty_with_format
    torch.npu_dropout_gen_mask = raise_error_npu_dropout_gen_mask
    torch_npu.npu_dropout_gen_mask = _npu_dropout_gen_mask
    torch.jit.script = jit_script
    torch.as_tensor = _as_tensor
    torch.new_device = _new_device
    torch.Generator = _generator

    torch._efficientzerotensor = __efficientzerotensor
    torch._pin_memory = __pin_memory
    torch.bartlett_window = _bartlett_window
    torch.blackman_window = _blackman_window
    torch.empty = _empty
    torch.empty_like = _empty_like
    torch.empty_strided = _empty_strided
    torch.eye = _eye
    torch.from_file = _from_file
    torch.full_like = _full_like
    torch.hamming_window = _hamming_window
    torch.hann_window = _hann_window
    torch.kaiser_window = _kaiser_window
    torch.linspace = _linspace
    torch.logspace = _logspace
    torch.normal = _normal
    torch.ones = _ones
    torch.ones_like = _ones_like
    torch.rand = _rand
    torch.rand_like = _rand_like
    torch.randint_like = _randint_like
    torch.randn = _randn
    torch.randn_like = _randn_like
    torch.randperm = _randperm
    torch.scalar_tensor = _scalar_tensor
    torch.tril_indices = _tril_indices
    torch.triu_indices = _triu_indices
    torch.zeros = _zeros
    torch.zeros_like = _zeros_like