# Copyright (c) 2020, Huawei Technologies.All rights reserved.
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
import numpy as np
import sys
import copy
import torch.nn as nn
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestMiopenConvolutionBackwardBias(TestCase):
    weight_grad = []
    input_grad = []
    bias_grad = []

    def getWeightGrad(self, grad):
        self.weight_grad.append(grad.to("cpu"))

    def getInputGrad(self, grad):
        self.input_grad.append(grad.to("cpu"))

    def getBiasGrad(self, grad):
        self.bias_grad.append(grad.to("cpu"))

    def op_exec_cpu(self, input, weight, bias, stride, padding, dilation, transposed, 
                    output_padding, groups, benchmark, deterministic, cudnn_enabled):

        input.requires_grad = True
        input.register_hook(lambda grad: self.getInputGrad(grad))
        weight.requires_grad = True
        weight.register_hook(lambda grad: self.getWeightGrad(grad))
        bias.requires_grad = True
        bias.register_hook(lambda grad: self.getBiasGrad(grad))

        cpu_res_forward = torch._convolution(input, weight, bias, stride, padding, dilation, transposed=False, output_padding=[0, 0], 
                                         groups=1, benchmark=False, deterministic=False, cudnn_enabled=False)

        tmp = torch.ones_like(cpu_res_forward).float()
        cpu_res_forward.backward(tmp, retain_graph=True)

        return cpu_res_forward
    
    def op_exec_npu(self, input, weight, bias, stride, padding, dilation, transposed, 
                    output_padding, groups, benchmark, deterministic, cudnn_enabled):

        input = input.to("npu")
        input.requires_grad = True
        input.register_hook(lambda grad: self.getInputGrad(grad))
        weight = weight.to("npu")
        weight.requires_grad = True
        weight.register_hook(lambda grad: self.getWeightGrad(grad))
        bias = bias.to("npu")
        bias.requires_grad = True
        bias.register_hook(lambda grad: self.getBiasGrad(grad))

        npu_res_forward = torch._convolution(input, weight, bias, stride, padding, dilation, transposed=False, output_padding=[0, 0],
                                             groups=1, benchmark=False, deterministic=False, cudnn_enabled=False)

        tmp = torch.ones_like(npu_res_forward).float()
        tmp = tmp.to("npu")
        npu_res_forward.backward(tmp, retain_graph=True)

        npu_res_forward = npu_res_forward.to("cpu")
        return npu_res_forward

    def test_miopen_convolution_backwrd_bias_float16_001(self, device):

        # input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, 
        # benchmark, deterministic, cudnn_enabled
        item = [[np.float16, 3, [256,128,7,7]], [np.float16, 3, (16,128,3,3)], [np.float16, 3, (16)], 
              [1, 1], [0, 0], [1, 1], False, [0, 0], 1, False, False, False]
        
        self.weight_grad.clear()
        self.input_grad.clear()
        self.bias_grad.clear()
        input_cpu, input_npu = create_common_tensor(item[0], -65500, 65500)
        if input_cpu.dtype == torch.float16:
            input_cpu = input_cpu.to(torch.float32)
        weight_cpu, weight_npu = create_common_tensor(item[1], -65500, 65500)
        if weight_cpu.dtype == torch.float16:
            weight_cpu = weight_cpu.to(torch.float32)
        bias_cpu, bias_npu = create_common_tensor(item[2], -65500, 65500)
        if bias_cpu.dtype == torch.float16:
            bias_cpu = bias_cpu.to(torch.float32)
        
        cpu_output = self.op_exec_cpu(input_cpu, weight_cpu, bias_cpu, stride=item[3], padding=item[4], dilation=item[5], transposed=item[6],
                                      output_padding=item[7], groups=item[8], benchmark=item[9], deterministic=item[10], cudnn_enabled=item[10])
        npu_output = self.op_exec_npu(input_npu, weight_npu, bias_npu, stride=item[3], padding=item[4], dilation=item[5], transposed=item[6],
                                      output_padding=item[7], groups=item[8], benchmark=item[9], deterministic=item[10], cudnn_enabled=item[10])
        cpu_output = cpu_output.to(npu_output.dtype)
    
        self.input_grad[0] = self.input_grad[0].to(self.input_grad[1].dtype)
        self.weight_grad[0] = self.weight_grad[0].to(self.weight_grad[1].dtype)
        self.bias_grad[0] = self.bias_grad[0].to(self.bias_grad[1].dtype)
        print("===bias_grad_float16_001===")
        print(self.bias_grad)   

        self.assertRtolEqual(self.bias_grad[0].numpy(), self.bias_grad[1].numpy())


instantiate_device_type_tests(TestMiopenConvolutionBackwardBias, globals(), except_for='cpu')
if __name__ == "__main__":
    torch.npu.set_device("npu:6")
    run_tests()
