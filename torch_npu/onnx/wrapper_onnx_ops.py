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

from typing import Optional, List

import torch

import torch_npu


class NPUOneHotOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch_npu._C._VariableFunctionsClass.npu_one_hot(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: torch.Tensor, num_classses: int = -1, depth: int = 1,
                 on_value: int = 1, off_value: int = 0):
        return g.op("npu::NPUOneHot", self, num_classses_i=num_classses, depth_i=depth,
                    on_value_i=on_value, off_value_i=off_value)


class NPUSliceOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch_npu._C._VariableFunctionsClass.npu_slice(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: torch.Tensor, offsets: List[int], size: List[int]):
        return g.op("npu::NPUSlice", self, offsetss_i=offsets, sizes_i=size)


class NPURoiAlignOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch_npu._C._VariableFunctionsClass.npu_roi_align(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: torch.Tensor, rois: torch.Tensor, spatial_scale: float,
                 pooled_height: int, pooled_width: int, sample_num: int, roi_end_mode: int):
        return g.op("npu::NPURoiAlign", self, rois, spatial_scale_f=spatial_scale,
                    pooled_height_i=pooled_height, pooled_width_i=pooled_width,
                    sample_num_i=sample_num, roi_end_mode_i=roi_end_mode)


class NPUIouOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch_npu._C._VariableFunctionsClass.npu_iou(*args, **kwargs)

    @staticmethod
    def symbolic(g, bboxes: torch.Tensor, gtboxes: torch.Tensor, mode: int = 0):
        return g.op("npu::NPUIou", bboxes, gtboxes, mode_i=mode)


class NPUBatchNmsOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch_npu._C._VariableFunctionsClass.npu_batch_nms(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: torch.Tensor, scores: torch.Tensor, score_threshold: float,
                 iou_threshold: float, max_size_per_class: int, max_total_size: int,
                 change_coordinate_frame: bool = False, transpose_box: bool = False):
        return g.op("npu::NPUBatchNms", self, scores, score_threshold_f=score_threshold,
                    iou_threshold_f=iou_threshold, max_size_per_class_i=max_size_per_class,
                    max_total_size_i=max_total_size, change_coordinate_frame_i=change_coordinate_frame,
                    transpose_box_i=transpose_box, outputs=4)


class NPUFastGeluOP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, *args, **kwargs):
        return torch_npu._C._VariableFunctionsClass.fast_gelu(*args, **kwargs)

    @staticmethod
    def symbolic(g, self: torch.Tensor):
        return g.op("npu::NPUFastGelu", self)


def torch_wrapper_npu_one_hot(self: torch.Tensor, num_classses: int = -1, depth: int = 1,
                              on_value: int = 1, off_value: int = 0):
    return  NPUOneHotOP.apply(self, num_classses, depth, on_value, off_value)


def torch_wrapper_npu_slice(self: torch.Tensor, offsets: List[int], size: List[int]):
    return NPUSliceOP.apply(self, offsets, size)


def torch_wrapper_npu_roi_align(self: torch.Tensor, rois: torch.Tensor, spatial_scale: float,
                                pooled_height: int, pooled_width: int, sample_num: int, roi_end_mode: int):
    return NPURoiAlignOP.apply(self, rois, spatial_scale,
                                       pooled_height, pooled_width, sample_num, roi_end_mode)


def torch_wrapper_npu_iou(bboxes: torch.Tensor, gtboxes: torch.Tensor, mode: int = 0):
    return NPUIouOP.apply(bboxes, gtboxes, mode)


def torch_wrapper_npu_batch_nms(self: torch.Tensor, scores: torch.Tensor, score_threshold: float,
                                iou_threshold: float, max_size_per_class: int, max_total_size: int,
                                change_coordinate_frame: bool = False, transpose_box: bool = False):
    return NPUBatchNmsOP.apply(self, scores, score_threshold,
                                       iou_threshold, max_size_per_class, max_total_size,
                                       change_coordinate_frame, transpose_box)


def torch_wrapper_fast_gelu(self: torch.Tensor):
    return NPUFastGeluOP.apply(self)


def add_onnx_ops():
    torch_npu.npu_one_hot = torch_wrapper_npu_one_hot
    torch_npu.npu_slice = torch_wrapper_npu_slice
    torch_npu.npu_roi_align = torch_wrapper_npu_roi_align
    torch_npu.npu_iou = torch_wrapper_npu_iou
    torch_npu.npu_batch_nms = torch_wrapper_npu_batch_nms
    torch_npu.fast_gelu = torch_wrapper_fast_gelu
