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
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"
#include <third_party/acl/inc/acl/op_api/aclnn_op.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"

namespace at_npu {
namespace native {

at::Tensor NPUNativeOpApiFunctions::scatter_add(
    const at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src) {
        auto selfClone = self.clone(at::MemoryFormat::Contiguous);
        OpPreparation::CheckMemory({selfClone, index, src},{selfClone});
        EXEC_NPU_CMD(aclnnScatterAdd, selfClone, dim, index, src, selfClone);
        return selfClone;
}

at::Tensor& NPUNativeOpApiFunctions::scatter_add_(
    at::Tensor& self,
    int64_t dim,
    const at::Tensor& index,
    const at::Tensor& src) {
        OpPreparation::CheckMemory({self, index, src}, {self});
        EXEC_NPU_CMD(aclnnScatterAdd, self, dim, index, src, self);
        return self;
}

at::Tensor NPUNativeOpApiFunctions::scatter_add(
    const at::Tensor& self,
    at::Dimname dim,
    const at::Tensor& index,
    const at::Tensor& src) {
        return scatter_add(self, dimname_to_position(self, dim), index, src);
}

} // namaspace native
} // namespace at_npu