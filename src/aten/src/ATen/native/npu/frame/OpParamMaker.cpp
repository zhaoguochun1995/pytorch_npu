// Copyright (c) 2020 Huawei Technologies Co., Ltd
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

#include "OpParamMaker.h"
#include <c10/npu/OptionsManager.h>
#include "c10/npu/NPUQueue.h"
#include <torch/csrc/autograd/record_function.h>
#include "ATen/native/npu/utils/DynamicShapeUtil.h"
#include "ATen/native/npu/aoe/AutoTune.h"
#include "ATen/native/npu/utils/NpuFuzzyBlacklist.h"
#include "ATen/native/GlobalStep.h"

namespace at {
namespace native {
namespace npu {

void OpCommandImpl::Run() {
  InitAttr();
  NPU_LOGD("Op %s Run.", opName.c_str());
  RECORD_FUNCTION(opName, std::vector<c10::IValue>({}));

  ACL_REQUIRE_OK_OP(InnerRun(opName, execParam), opName.c_str());
}

aclError OpCommandImpl::InnerRun(string name, AclExecParam& params) {
  AutoTune::GetInstance()->Do(name, params.graph);
  auto stream = c10::npu::getCurrentNPUStream();
  auto inputSize = params.inBuffer.size();
  auto outputSize = params.outBuffer.size();
  bool reset_flag = false;
  if (check_fuzz_enable() &&
      FuzzyCompileBlacklist::GetInstance().IsInBlacklist(name)) {
    aclopSetCompileFlag(aclOpCompileFlag::ACL_OP_COMPILE_DEFAULT);
    reset_flag = true;
  }
  auto ret = aclopCompileAndExecute(
    name.c_str(),
    inputSize,
    params.inDesc.data(),
    params.inBuffer.data(),
    outputSize,
    params.outDesc.data(),
    params.outBuffer.data(),
    params.attr,
    ACL_ENGINE_SYS,
    ACL_COMPILE_SYS,
    NULL,
    stream);
  if (reset_flag) {
    aclopSetCompileFlag(aclOpCompileFlag::ACL_OP_COMPILE_FUZZ);
  }
  return ret;
}

int ExecFunc(void* in, aclrtStream stream) {
  auto cur_paras = (ExecuteParas*)in;
  NPU_LOGD("Op %s Run.", cur_paras->opType.c_str());

  aclError ret;
  if (c10::npu::OptionsManager::CheckDynamicEnable()) {
    ret = DynamicRun(*cur_paras, stream);
  } else {
    bool reset_flag = false;
    if (check_fuzz_enable() &&
        FuzzyCompileBlacklist::GetInstance().IsInBlacklist(cur_paras->opType)) {
      aclopSetCompileFlag(aclOpCompileFlag::ACL_OP_COMPILE_DEFAULT);
      reset_flag = true;
    }
    ret = aclopCompileAndExecute(
        (cur_paras->opType).c_str(),
        cur_paras->paras.input_num,
        cur_paras->paras.input_desc,
        cur_paras->paras.input_data_buf,
        cur_paras->paras.output_num,
        cur_paras->paras.output_desc,
        cur_paras->paras.output_data_buf,
        cur_paras->attr,
        ACL_ENGINE_SYS,
        ACL_COMPILE_SYS,
        nullptr,
        stream);
    if (reset_flag) {
      aclopSetCompileFlag(aclOpCompileFlag::ACL_OP_COMPILE_FUZZ);
    }
    if (ret != ACL_ERROR_NONE) {
      C10_NPU_SHOW_ERR_MSG();
    }
  }

  if (ret != 0) {
    std::cout << "---OpName--- " << cur_paras->opType << std::endl;
  }
  return ret;
}

void CopyFunc(void* dst, void* src) {
  auto dstPtr = (ExecuteParas*)dst;
  auto srcPtr = (ExecuteParas*)src;
  dstPtr->Copy(*srcPtr);
}

void ReleaseFunc(void* ptr) {
  auto cur_paras = (ExecuteParas*)ptr;
  if (cur_paras->opDynamicType != "") {
    cur_paras->DynamicRelease();
    cur_paras->opDynamicType = "";
  }
  cur_paras->Release();
}

void* NewFunc(int caption, int& size) {
  size = sizeof(ExecuteParas);
  return (void*)new ExecuteParas[caption];
}

void DeleteFunc(void* ptr) {
  delete[](ExecuteParas*) ptr;
}

REGISTER_QUEUE_FUNC(ExecFunc, CopyFunc, ReleaseFunc, NewFunc, DeleteFunc)

OpCommandImpls* OpCommandImpls::GetInstance() {
  static OpCommandImpls impl;
  return &impl;
}

void OpCommandImpls::Push(OpCommandImpl*& ptr) {
  offset += 1;
  if (objs.size() <= offset) {
    OpCommandImpl impl;
    objs.push_back(impl);
  }
  TORCH_CHECK(
      objs.size() > offset,
      "OpCommand size (",
      objs.size(),
      ") is smaller than offset (",
      offset,
      ")");
  ptr = &objs[offset];
}

void OpCommandImpls::Pop() {
  TORCH_CHECK(
      offset >= 0, "OpCommand current offset should not be less than ", offset);
  offset -= 1;
}

} // namespace npu
} // namespace native
} // namespace at