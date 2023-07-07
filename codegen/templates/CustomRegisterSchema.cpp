#include <c10/core/TensorImpl.h>
#include <c10/core/Allocator.h>
#include <ATen/DeviceGuard.h>
#include <ATen/NativeFunctions.h>
#include <ATen/MetaFunctions.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/Dispatch.h>
#include <c10/util/Half.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/util/Optional.h>
#include <ATen/Tensor.h>
#include <ATen/Functions.h>
#include <ATen/native/Resize.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

#include <ATen/Config.h>
#include <torch/library.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/jit/frontend/tracer.h>

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/VariableType.h"

#ifdef USE_OPPLUGIN
#include "op_plugin/ops/OpInterface.h"
#endif

namespace at_npu {

namespace native {

${custom_trace_definitions}

namespace {

TORCH_LIBRARY(npu, m) {


  ${custom_trace_registrations}
}

} // anonymous namespace

} // namespace native

} // namespace at_npu
