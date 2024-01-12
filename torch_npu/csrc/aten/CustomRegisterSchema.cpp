#include <torch/library.h>

namespace at_npu {

namespace native {


namespace {

TORCH_LIBRARY_FRAGMENT(aten, m) {

  m.def("npu_change_data_ptr(Tensor dst, Tensor src, int index) -> int");
  m.def("get_npu_format(Tensor self) -> int");
  m.def("npu_format_cast.Tensor(Tensor self, Tensor dst) -> Tensor");
  m.def("npu_format_cast_.acl_format(Tensor(a!) self, int acl_format) -> Tensor(a!)");
  m.def("npu_format_cast_(Tensor(a!) self, Tensor src) -> Tensor(a!)");
  m.def("empty_with_format(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, int acl_format=2) -> Tensor");
  m.def("unsafe_empty_with_format(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, int acl_format=2, bool keep_format=False) -> Tensor");
  m.def("empty_with_format.names(int[] size, Dimname[]? names, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, int acl_format=2) -> Tensor");
  m.def("copy_memory_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)");
  m.def("format_contiguous(Tensor self) -> Tensor");
  m.def("check_match(Tensor self) -> bool");
  m.def("check_memory_overlaps(Tensor[] inputs, Tensor[] outputs) -> ()");
  m.def("get_storage_size(Tensor self) -> int");
  m.def("npu_format_cast(Tensor self, int acl_format) -> Tensor");
  m.def("_npu_format_cast(Tensor self, int acl_format) -> Tensor");
  m.def("npu_view_copy(Tensor(a!) self, Tensor other, bool non_blocking) -> Tensor(a!)");
  m.def("npu_transpose(Tensor self, int[] perm, bool require_contiguous=True) -> Tensor");
  m.def("npu_transpose.out(Tensor self, int[] perm, bool require_contiguous=True, *, Tensor(a!) out) -> Tensor(a!)");
  m.def("npu_broadcast(Tensor self, int[] size) -> Tensor");
  m.def("npu_broadcast.out(Tensor self, int[] size, *, Tensor(a!) out) -> Tensor(a!)");
  m.def("npu_dtype_cast_(Tensor(a!) self, Tensor src) -> Tensor(a!)");
  m.def("npu_alloc_float_status(Tensor self) -> Tensor");
  m.def("npu_get_float_status(Tensor self) -> Tensor");
  m.def("npu_clear_float_status(Tensor self) -> Tensor");
  m.def("one_(Tensor(a!) self) -> Tensor(a!)");
  m.def("fast_gelu(Tensor self) -> Tensor");
  m.def("npu_fast_gelu_backward(Tensor grad, Tensor self) -> Tensor");
  m.def("_amp_foreach_non_finite_check(Tensor[] scaled_grads) -> bool");
  m.def("npu_sign_bits_pack(Tensor self, int size) -> Tensor");
  m.def("npu_bert_apply_adam(Scalar lr, Scalar beta1, Scalar beta2, Scalar epsilon, Tensor grad, Scalar max_grad_norm, Scalar global_grad_norm, Scalar weight_decay, Scalar? step_size=None, int adam_mode=0) -> (Tensor var, Tensor m, Tensor v)");
  m.def("npu_bert_apply_adam.out(Scalar lr, Scalar beta1, Scalar beta2, Scalar epsilon, Tensor grad, Scalar max_grad_norm, Scalar global_grad_norm, Scalar weight_decay, Scalar? step_size=None, int adam_mode=0, *, Tensor(a!) var, Tensor(b!) m, Tensor(c!) v) -> (Tensor(a!), Tensor(b!), Tensor(c!))");
  m.def("npu_conv_transpose2d_backward(Tensor input, Tensor grad_output, Tensor weight, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)");
  m.def("npu_conv_transpose3d_backward(Tensor input, Tensor grad_output, Tensor weight, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)");
  m.def("npu_convolution_backward(Tensor input, Tensor grad_output, Tensor weight, int[] stride, int[] padding, int[] dilation, int groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)");
  m.def("npu_conv_transpose2d(Tensor input, Tensor weight, Tensor? bias, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups) -> Tensor");
  m.def("npu_conv2d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor");
  m.def("npu_conv2d.out(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups, *, Tensor(a!) out) -> Tensor(a!)");
  m.def("npu_conv2d_backward(Tensor input, Tensor grad_output, Tensor weight, int[] stride, int[] padding, int[] dilation, int groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)");
  m.def("npu_conv3d(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor");
  m.def("npu_conv3d.out(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups, *, Tensor(a!) out) -> Tensor(a!)");
  m.def("npu_conv3d_backward(Tensor input, Tensor grad, Tensor weight, int[] stride, int[] padding, int[] dilation, int groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)");
  m.def("npu_stride_add(Tensor self, Tensor other, Scalar offset1, Scalar offset2, Scalar c1_len) -> Tensor");
  m.def("npu_slice(Tensor self, int[] offsets, int[] size) -> Tensor");
  m.def("npu_slice.out(Tensor self, int[] offsets, int[] size, *, Tensor(a!) out) -> Tensor(a!)");
  m.def("npu_indexing(Tensor self, int[] begin, int[] end, int[] strides, int begin_mask=0, int end_mask=0, int ellipsis_mask=0, int new_axis_mask=0, int shrink_axis_mask=0) -> Tensor");
  m.def("npu_indexing.out(Tensor self, int[] begin, int[] end, int[] strides, int begin_mask=0, int end_mask=0, int ellipsis_mask=0, int new_axis_mask=0, int shrink_axis_mask=0, *, Tensor(a!) out) -> Tensor(a!)");
  m.def("npu_softmax_cross_entropy_with_logits_backward(Tensor grad, Tensor self, Tensor labels) -> Tensor");
  m.def("npu_stride_copy(Tensor self, int[] shape, int[] stride, Scalar storage_offset) -> Tensor");
  m.def("npu_stride_copy.out(Tensor self, int[] shape, int[] stride, Scalar storage_offset, *, Tensor(a!) out) -> Tensor(a!)");
  m.def("npu_roi_align(Tensor self, Tensor rois, float spatial_scale, int pooled_height, int pooled_width, int sample_num, int roi_end_mode) -> Tensor");
  m.def("npu_roi_alignbk(Tensor self, Tensor rois, int[] xdiff_shape, int pooled_width, int pooled_height, float spatial_scale, int sample_num, int? roi_end_mode=None) -> Tensor");
  m.def("npu_sort_v2.out(Tensor self, int dim=-1, bool descending=False, *, Tensor(a!) out) -> Tensor(a!)");
  m.def("npu_sort_v2(Tensor self, int dim=-1, bool descending=False) -> Tensor");
  m.def("npu_one_hot(Tensor self, int num_classes=-1, int depth=1, Scalar on_value=1, Scalar off_value=0) -> Tensor");
  m.def("npu_linear_backward(Tensor grad, Tensor input, Tensor weight) -> (Tensor, Tensor)");
  m.def("npu_anchor_response_flags(Tensor self, int[2] featmap_size, int[2] stride, int num_base_anchors) -> Tensor");
  m.def("npu_dropout_backward(Tensor grad_output, Tensor mask, float p) -> Tensor");
  m.def("npu_nms_rotated(Tensor self, Tensor scores, float iou_threshold, float scores_threshold=0, int max_output_size=-1, int mode=0) -> (Tensor, Tensor)");
  m.def("npu_masked_fill_range(Tensor self, Tensor start, Tensor end, Tensor value, int axis=-1) -> Tensor");
  m.def("npu_sub_sample(Tensor self, int per_images, float positive_fraction) -> Tensor");
  m.def("npu_yolo_boxes_encode(Tensor self, Tensor gt_bboxes, Tensor stride, bool performance_mode=False) -> Tensor");
  m.def("npu_scatter(Tensor self, Tensor indices, Tensor updates, int dim) -> Tensor");
  m.def("npu_layer_norm_eval(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05) -> Tensor");
  m.def("npu_rotated_box_encode(Tensor self, Tensor gt_bboxes, Tensor weight) -> Tensor");
  m.def("npu_rotated_box_decode(Tensor self, Tensor deltas, Tensor weight) -> Tensor");
  m.def("npu_rotated_overlaps(Tensor self, Tensor query_boxes, bool trans=False) -> Tensor");
  m.def("npu_silu_backward(Tensor grad_output, Tensor x0, Tensor x1) -> Tensor");
  m.def("npu_rotated_iou(Tensor self, Tensor query_boxes, bool trans=False, int mode=0, bool is_cross=True, float v_threshold=0.0, float e_threshold=0.0) -> Tensor");
  m.def("npu_nms_with_mask(Tensor input, Scalar iou_threshold) -> (Tensor, Tensor, Tensor)");
  m.def("npu_gru_backward(Tensor? grady, Tensor? gradh, Tensor input, Tensor weight_input, Tensor weight_hidden, Tensor bias_input, Tensor bias_hidden, Tensor seq_length, Tensor hx, Tensor y_output, Tensor h_output, Tensor output_updata, Tensor output_reset, Tensor output_new, Tensor hidden_new) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("npu_mish_backward(Tensor grad, Tensor input) -> Tensor");
  m.def("npu_reshape(Tensor self, int[] shape, bool can_refresh=False) -> Tensor");
  m.def("npu_reshape.out(Tensor self, int[] shape, bool can_refresh=False, *, Tensor(a!) out) -> Tensor(a!)");
  m.def("npu_batch_nms(Tensor self, Tensor scores, float score_threshold, float iou_threshold, int max_size_per_class, int max_total_size, bool change_coordinate_frame=False, bool transpose_box=False) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("npu_bounding_box_encode(Tensor anchor_box, Tensor ground_truth_box, float means0, float means1, float means2, float means3, float stds0, float stds1, float stds2, float stds3) -> Tensor");
  m.def("npu_bounding_box_decode(Tensor rois, Tensor deltas, float means0, float means1, float means2, float means3, float stds0, float stds1, float stds2, float stds3, int[1] max_shape, float wh_ratio_clip) -> Tensor");
  m.def("npu_apply_adam(Scalar beta1_power, Scalar beta2_power, Scalar lr, Scalar beta1, Scalar beta2, Scalar epsilon, Tensor grad, bool? use_locking, bool? use_nesterov) -> (Tensor, Tensor, Tensor)");
  m.def("npu_apply_adam.out(Scalar beta1_power, Scalar beta2_power, Scalar lr, Scalar beta1, Scalar beta2, Scalar epsilon, Tensor grad, bool? use_locking, bool? use_nesterov, *, Tensor(a!) var, Tensor(b!) m, Tensor(c!) v) -> (Tensor(a!), Tensor(b!), Tensor(c!))");
  m.def("npu_apply_adam_w(Scalar beta1_power, Scalar beta2_power, Scalar lr, Scalar weight_decay, Scalar beta1, Scalar beta2, Scalar epsilon, Tensor grad, Tensor? max_grad_norm, bool? amsgrad, bool? maximize) -> (Tensor, Tensor, Tensor)");
  m.def("npu_apply_adam_w.out(Scalar beta1_power, Scalar beta2_power, Scalar lr, Scalar weight_decay, Scalar beta1, Scalar beta2, Scalar epsilon, Tensor grad, Tensor? max_grad_norm, bool? amsgrad, bool? maximize, *, Tensor(a!) var, Tensor(b!) m, Tensor(c!) v) -> (Tensor(a!), Tensor(b!), Tensor(c!))");
  m.def("npu_deformable_conv2dbk(Tensor input, Tensor grad_output, Tensor offset_out, Tensor weight, Tensor offset, int[2] kernel_size, int[] stride, int[] padding, int[] dilation=[1,1,1,1], int groups=1, int deformable_groups=1, bool modulated=True) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("npu_giou_backward(Tensor grad, Tensor bboxes, Tensor gtboxes, bool trans=False, bool is_cross=False, int mode=0) -> (Tensor, Tensor)");
  m.def("npu_diou_backward(Tensor grad, Tensor bboxes, Tensor gtboxes, bool trans=False, bool is_cross=False, int mode=0) -> (Tensor, Tensor)");
  m.def("npu_iou(Tensor bboxes, Tensor gtboxes, int mode=0) -> Tensor");
  m.def("npu_nms_v4(Tensor self, Tensor scores, Scalar max_output_size, Tensor iou_threshold, Tensor scores_threshold, bool pad_to_max_output_size=False) -> (Tensor, Tensor)");
  m.def("npu_pad(Tensor input, int[] paddings) -> Tensor");
  m.def("npu_random_choice_with_mask(Tensor x, int count=256, int seed=0, int seed2=0) -> (Tensor, Tensor)");
  m.def("npu_normalize_batch(Tensor self, Tensor seq_len, int normalize_type=0) -> Tensor");
  m.def("npu_ptiou(Tensor bboxes, Tensor gtboxes, int mode=0) -> Tensor");
  m.def("npu_lstm_backward(Tensor? grady, Tensor? gradh, Tensor? gradc, Tensor input, Tensor weight, Tensor bias, Tensor hx, Tensor cx, Tensor y_output, Tensor h_output, Tensor c_output, Tensor i, Tensor j, Tensor f, Tensor o, Tensor tanhc) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("_dropout_with_byte_mask_backward(Tensor grad_output, Tensor mask, float p) -> Tensor");
  m.def("dropout_with_byte_mask(Tensor self, float p, bool train) -> Tensor");
  m.def("npu_dropout_with_add_softmax_backward(Tensor grad, Tensor mask, Tensor softmax_out, Scalar alpha, float prob, int dim) -> (Tensor, Tensor)");
  m.def("npu_multi_head_attention_backward(Tensor query, Tensor key, Tensor value, Tensor query_weight, Tensor key_weight, Tensor value_weight, Tensor out_proj_weight, Tensor? query_bias, Tensor? key_bias, Tensor? value_bias, Tensor? out_proj_bias, Tensor query_res, Tensor key_res, Tensor value_res, Tensor attn_scores, Tensor attn_res, Tensor context, Tensor y_grad, Tensor dropout_mask, int attn_head_num, int attn_dim_per_head, int src_len, int tgt_len, float dropout_prob, bool softmax_use_float) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("npu_dropout_gen_mask(int[] size, float p, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor");
  m.def("npu_ciou_backward(Tensor grad, Tensor bboxes, Tensor gtboxes, Tensor? atan_sub, bool trans=False, bool is_cross=True, int mode=0) -> (Tensor, Tensor)");
  m.def("npu_sign_bits_unpack(Tensor input, int size, ScalarType dtype) -> Tensor");
  m.def("decode_jpeg(Tensor self, int[] image_shape, int channels=3, bool try_recover_truncated=False) -> Tensor");
  m.def("crop_and_resize(Tensor self, float[]? boxes, int[] box_index, int[] crop_size, float extrapolation_value=0, str method=\"bilinear\") -> Tensor");
  m.def("reverse(Tensor self, int[] axis) -> Tensor");
  m.def("image_normalize(Tensor self, float[]? mean, float[]? variance, int dtype=0) -> Tensor");
  m.def("image_normalize_(Tensor(a!) self, float[]? mean, float[]? variance, int dtype=0) -> Tensor(a!)");
  m.def("img_to_tensor(Tensor self) -> Tensor");
  m.def("_conv_depthwise2d_backward(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool[2] output_mask) -> (Tensor grad_input, Tensor grad_weight)");
  m.def("slow_conv_dilated2d_backward(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)");
  m.def("slow_conv_transpose2d_backward(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] output_padding, int[2] dilation, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)");
  m.def("npu_lstm_cell_backward(Tensor? grady, Tensor? gradh, Tensor? gradc, Tensor input, Tensor w_ih, Tensor w_hh, Tensor h, Tensor c, Tensor y_output, Tensor h_output, Tensor c_output, Tensor i, Tensor j, Tensor f, Tensor o, Tensor tanhc) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("batch_norm_reduce(Tensor input, float eps) -> (Tensor, Tensor)");
  m.def("batch_norm_gather_stats_update(Tensor input, Tensor mean, Tensor invstd, Tensor? running_mean, Tensor? running_var, float momentum, float eps, Tensor counts) -> (Tensor, Tensor)");
  m.def("npu_fused_attention_score_backward(Tensor grad_output, Tensor softmax_output, Tensor query_layer, Tensor key_layer, Tensor value_layer, Tensor mask, Scalar scale, float keep_prob, bool query_transpose=False, bool key_transpose=False, bool value_transpose=False, bool dx_transpose=False) -> (Tensor, Tensor, Tensor)");
  m.def("npu_fused_attention_score_fwd(Tensor query_layer, Tensor key_layer, Tensor value_layer, Tensor attention_mask, Scalar scale, float keep_prob, bool query_transpose=False, bool key_transpose=False, bool bmm_score_transpose_a=False, bool bmm_score_transpose_b=False, bool value_transpose=False, bool dx_transpose=False) -> (Tensor, Tensor, Tensor)");
  m.def("npu_fused_attention_score_grad(Tensor grad_output, Tensor softmax_output, Tensor query_layer, Tensor key_layer, Tensor value_layer, Tensor mask, Scalar scale, float keep_prob, bool query_transpose=False, bool key_transpose=False, bool value_transpose=False, bool dx_transpose=False) -> (Tensor, Tensor, Tensor)");
  m.def("npu_fused_attention_qkv_grad(Tensor grad_output_query, Tensor grad_output_key, Tensor grad_output_value, Tensor query_kernel, Tensor key_kernel, Tensor value_kernel, Tensor hidden_states, Tensor grad_output_ln) -> Tensor[]");
  m.def("npu_fused_attention_layernorm_qkv_fwd(Tensor x, Tensor kernel_query, Tensor kernel_key, Tensor kernel_value, Tensor gamma, Tensor beta, Tensor? bias_query=None, Tensor? bias_key=None, Tensor? bias_value=None, int seq_len=128, int num_heads=12, float eps=1e-05) -> Tensor[]");
  m.def("npu_layernorm_grad(Tensor grad_out, Tensor input, int[] normalized_shape, Tensor mean, Tensor rstd, Tensor? weight, Tensor? bias) -> (Tensor, Tensor, Tensor)");
  m.def("npu_ifmr(Tensor data, Tensor data_min, Tensor data_max, Tensor cumsum, float min_percentile, float max_percentile, float search_start, float search_end, float search_step, bool with_offset) -> (Tensor, Tensor)");
  m.def("npu_grid_assign_positive(Tensor self, Tensor overlaps, Tensor box_responsible_flags, Tensor max_overlaps, Tensor argmax_overlaps, Tensor gt_max_overlaps, Tensor gt_argmax_overlaps, int num_gts, float pos_iou_thr, float min_pos_iou, bool gt_max_assign_all) -> Tensor");
  m.def("npu_rotary_mul(Tensor self, Tensor r1, Tensor r2) -> Tensor");
  m.def("npu_rotary_mul_backward(Tensor grad, Tensor self, Tensor r1, Tensor r2) -> (Tensor, Tensor, Tensor)");
  m.def("npu_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> Tensor");
  m.def("npu_convolution_transpose(Tensor input, Tensor weight, Tensor? bias, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups) -> Tensor");
  m.def("npu_confusion_transpose(Tensor self, int[] perm, int[] shape, bool transpose_first) -> Tensor");
  m.def("npu_ps_roi_pooling(Tensor self, Tensor rois, float spatial_scale, int group_size, int output_dim) -> Tensor");
  m.def("npu_linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor");
  m.def("_npu_dropout(Tensor self, float p) -> (Tensor, Tensor)");
  m.def("npu_softmax_cross_entropy_with_logits(Tensor self, Tensor labels) -> Tensor");
  m.def("npu_max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)");
  m.def("npu_max.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)");
  m.def("npu_bmmV2(Tensor self, Tensor mat2, int[] output_sizes) -> Tensor");
  m.def("npu_dtype_cast(Tensor self, ScalarType dtype) -> Tensor");
  m.def("npu_silu(Tensor self) -> Tensor");
  m.def("npu_silu_(Tensor(a!) self) -> Tensor(a!)");
  m.def("npu_gru(Tensor input, Tensor hx, Tensor weight_input, Tensor weight_hidden, Tensor bias_input, Tensor bias_hidden, Tensor seq_length, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("npu_mish(Tensor self) -> Tensor");
  m.def("npu_min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)");
  m.def("npu_min.names_dim(Tensor self, Dimname dim, bool keepdim=False) -> (Tensor values, Tensor indices)");
  m.def("npu_deformable_conv2d(Tensor input, Tensor weight, Tensor offset, Tensor? bias, int[2] kernel_size, int[] stride, int[] padding, int[] dilation=[1,1,1,1], int groups=1, int deformable_groups=1, bool modulated=True) -> (Tensor, Tensor)");
  m.def("npu_giou(Tensor self, Tensor gtboxes, bool trans=False, bool is_cross=False, int mode=0) -> Tensor");
  m.def("npu_diou(Tensor self, Tensor gtboxes, bool trans=False, bool is_cross=False, int mode=0) -> Tensor");
  m.def("npu_lstm(Tensor input, Tensor weight, Tensor bias, Tensor seq_mask, Tensor h, Tensor c, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first, bool flag_seq, bool direction) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("npu_lstm_data(Tensor input, Tensor batch_sizes, Tensor weight, Tensor bias, Tensor seq_mask, Tensor h, Tensor c, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first, bool flag_seq, bool direction) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("_dropout_with_byte_mask(Tensor self, float p) -> (Tensor, Tensor)");
  m.def("npu_dropout_with_add_softmax(Tensor self, Tensor x1, Scalar alpha, float prob, int dim) -> (Tensor, Tensor, Tensor)");
  m.def("npu_scaled_masked_softmax(Tensor x, Tensor mask, Scalar scale=1, bool fixed_triu_mask=False) -> Tensor");
  m.def("npu_multi_head_attention(Tensor query, Tensor key, Tensor value, Tensor query_weight, Tensor key_weight, Tensor value_weight, Tensor attn_mask, Tensor out_proj_weight, Tensor? query_bias, Tensor? key_bias, Tensor? value_bias, Tensor? out_proj_bias, Tensor? dropout_mask, int attn_head_num, int attn_dim_per_head, int src_len, int tgt_len, float dropout_prob, bool softmax_use_float) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("npu_fusion_attention(Tensor query, Tensor key, Tensor value, int head_num, str input_layout, Tensor? pse=None, Tensor? padding_mask=None, Tensor? atten_mask=None, float scale=1., float keep_prob=1., int pre_tockens=2147483647, int next_tockens=2147483647, int inner_precise=0, bool gen_mask_parallel=True, bool sync=False) -> (Tensor, Tensor, Tensor, Tensor, int, int, int)");
  m.def("npu_fusion_attention_grad(Tensor query, Tensor key, Tensor value, Tensor dy, int head_num, str input_layout, *, Tensor? pse=None, Tensor? padding_mask=None, Tensor? atten_mask=None, Tensor? softmax_max=None, Tensor? softmax_sum=None, Tensor? softmax_in=None, Tensor? attention_in=None, float scale_value=1., float keep_prob=1., int pre_tockens=2147483647, int next_tockens=2147483647, int inner_precise=0, int seed=0, int offset=0, int numels=0, bool gen_mask_parallel=True, bool sync=False) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def("npu_dropout_do_mask(Tensor self, Tensor mask, float p) -> (Tensor, Tensor)");
  m.def("npu_ciou(Tensor self, Tensor gtboxes, bool trans=False, bool is_cross=True, int mode=0, bool atan_sub_flag=False) -> Tensor");
  m.def("npu_lstm_cell(Tensor input, Tensor w_ih, Tensor w_hh, Tensor h, Tensor c, Tensor? bias=None) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("npu_fused_attention_score(Tensor query_layer, Tensor key_layer, Tensor value_layer, Tensor attention_mask, Scalar scale, float keep_prob, bool query_transpose=False, bool key_transpose=False, bool bmm_score_transpose_a=False, bool bmm_score_transpose_b=False, bool value_transpose=False, bool dx_transpose=False) -> Tensor");
  m.def("_npu_ciou(Tensor self, Tensor gtboxes, bool trans=False, bool is_cross=True, int mode=0, bool atan_sub_flag=False) -> (Tensor, Tensor)");
}

} // anonymous namespace

} // namespace native

} // namespace at_npu
