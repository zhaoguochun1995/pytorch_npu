#ifndef __PULGIN_NATIVE_NPU_UTILS_OP_PREPARATION__
#define __PULGIN_NATIVE_NPU_UTILS_OP_PREPARATION__

#include "torch_npu/csrc/framework/utils/NPUDefinition.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace at_npu {
namespace native {

class OpPreparation {
public:
  static UnifiedResult binary_op_check(at::Tensor &out, const at::Tensor &a, const at::Tensor &b,
                                       bool check_mem_overlap);
  static UnifiedResult binary_op_check(at::Tensor &out, const at::Tensor &a, const c10::Scalar b,
                                       bool check_mem_overlap);
  static UnifiedResult comparison_op_check(at::Tensor &out, const at::Tensor &a, const at::Tensor &b,
                                           bool check_mem_overlap);
  static UnifiedResult unary_op_check(at::Tensor &out, const at::Tensor &a, bool check_mem_overlap);
  static void nullary_op(at::Tensor &out);
  static UnifiedResult reduce_op_check(at::Tensor &out, const at::Tensor &a);
  static UnifiedResult reduce_op_check(at::Tensor &out1, at::Tensor &out2, const at::Tensor &a);
  // check output tensor
  static void check_tensor(const std::initializer_list<at::Tensor> &src_list, at::Tensor &dst,
                           at::ScalarType expect_dtype, c10::IntArrayRef expect_size);
  static void check_tensor(const std::initializer_list<at::Tensor> &src_list, at::Tensor &dst,
                           const at::Tensor &expect_tensor);
  static void check_tensor(const std::initializer_list<at::Tensor> &src_list, at::Tensor &dst,
                           c10::IntArrayRef expect_size);
  // check memory overlaps
  static void check_memory(const std::initializer_list<at::Tensor> &inputs,
                           const std::initializer_list<at::Tensor> &outputs);
  // cast format
  static at::Tensor cast_to_ori_format(const at::Tensor &tensor);
  static at::Tensor &cast_to_ori_format(at::Tensor &tensor);
  // used to apply output tensor
  static at::Tensor apply_tensor(const at::Tensor &src);
  static at::Tensor apply_tensor(const at::Tensor &src, c10::IntArrayRef sizes);
  static at::Tensor apply_tensor(const at::Tensor &src, const c10::TensorOptions &options);
  static at::Tensor apply_tensor(c10::IntArrayRef sizes, const c10::TensorOptions &options, const at::Tensor &src);
  static at::Tensor apply_tensor_with_format(const at::Tensor &src, int64_t format, bool keep_format = false);
  static at::Tensor apply_tensor_with_format(const at::Tensor &src, c10::IntArrayRef sizes, int64_t format,
                                             bool keep_format = false);
  static at::Tensor apply_tensor_with_format(c10::IntArrayRef sizes, const c10::TensorOptions &options, int64_t format,
                                             bool keep_format = false);
  static at::Tensor apply_tensor_with_sizes(c10::IntArrayRef sizes, const c10::TensorOptions &options);

  // DEPRECATED: CheckOut will be deprecated, please use check_tensor to check output tensor instead.
  static void CheckOut(const std::initializer_list<at::Tensor> &inputs, at::Tensor &output, at::Tensor dst);
  static void CheckOut(const std::initializer_list<at::Tensor> &inputs, at::Tensor &output, at::Tensor dst,
                       c10::IntArrayRef shape);
  static void CheckOut(const std::initializer_list<at::Tensor> &input, at::Tensor &output, int64_t format,
                       at::ScalarType dtype, c10::IntArrayRef shape);
  // DEPRECATED: CastBackToOriFormat will be deprecated, please use cast_to_ori_format instead.
  static at::Tensor CastBackToOriFormat(const at::Tensor &tensor);
  static at::Tensor &CastBackToOriFormat(at::Tensor &tensor);
  // DEPRECATED: ApplyTensor will be deprecated, please use apply_tensor instead.
  static at::Tensor ApplyTensor(const at::Tensor &src);
  static at::Tensor ApplyTensor(const at::Tensor &src, c10::IntArrayRef sizes);
  static at::Tensor ApplyTensor(const at::Tensor &src, const c10::TensorOptions &options);
  static at::Tensor ApplyTensor(c10::IntArrayRef sizes, const c10::TensorOptions &options, const at::Tensor &src);
  // DEPRECATED: ApplyTensorWithFormat will be deprecated, please use apply_tensor_with_format instead.
  static at::Tensor ApplyTensorWithFormat(const at::Tensor &src, int64_t format, bool keep_format = false);
  static at::Tensor ApplyTensorWithFormat(const at::Tensor &src, c10::IntArrayRef sizes, int64_t format,
                                          bool keep_format = false);
  static at::Tensor ApplyTensorWithFormat(c10::IntArrayRef sizes, const c10::TensorOptions &options, int64_t format,
                                          bool keep_format = false);
  // DEPRECATED: ApplyTensorWithSizes will be deprecated, please use apply_tensor_with_sizes instead.
  static at::Tensor ApplyTensorWithSizes(c10::IntArrayRef sizes, const c10::TensorOptions &options);
  // DEPRECATED: CheckMemory will be deprecated, please use check_memory instead.
  static void CheckMemory(const std::initializer_list<at::Tensor> &inputs,
                          const std::initializer_list<at::Tensor> &outputs);
  static bool IsCPUScalar(const at::Tensor &tensor);
};  // namespace OpPreparation
}  // namespace native
}  // namespace at_npu

#endif
