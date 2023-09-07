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
// See the License for the specific languacontiguousResultge governing permissions and
// limitations under the License.

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/NPUNativeOpApiFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {

at::Tensor& NPUNativeOpApiFunctions::histc_out(const at::Tensor& self, int64_t bins, const at::Scalar& min, 
                                               const at::Scalar& max, at::Tensor& result) {
  DO_COMPATIBILITY(aclnnHistc, NPUNativeFunctions::histc_out(self, bins, min, max, result));
  OpPreparation::CheckOut({self}, result, self.scalar_type(), {bins});
  EXEC_NPU_CMD(aclnnHistc, self, bins, min, max, result);
  return result;
}

at::Tensor NPUNativeOpApiFunctions::histc(const at::Tensor& self, int64_t bins, const at::Scalar& min, 
                                          const at::Scalar& max) {
  DO_COMPATIBILITY(aclnnHistc, NPUNativeFunctions::histc(self, bins, min, max));
  at::ScalarType out_type = self.scalar_type();
  at::Tensor result = OpPreparation::ApplyTensorWithoutFormat({bins}, self.options().dtype(out_type));
  EXEC_NPU_CMD(aclnnHistc, self, bins, min, max, result);
  return result;
}

}  // namespace native
}  // namespace at_npu