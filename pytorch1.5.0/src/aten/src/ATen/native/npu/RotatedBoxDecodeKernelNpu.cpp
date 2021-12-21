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

namespace at {
namespace native {
using namespace at::native::npu;
    
Tensor rotated_box_decode_npu(
    const Tensor& self, 
    const Tensor& deltas, 
    const Tensor& weight){
  Tensor result = OpPreparation::ApplyTensor(self);
  Tensor weightContiguous = weight.to(Device(at::kCPU), at::kFloat);
  ArrayRef<float> weightList(weightContiguous.data_ptr<float>(), weightContiguous.numel());  
  
  OpCommand cmd;
  cmd.Name("RotatedBoxDecode")
      .Input(self)
      .Input(deltas)
      .Output(result)
      .Attr("weight", weightList)
      .Run();   
  return result;  
}    

}}
