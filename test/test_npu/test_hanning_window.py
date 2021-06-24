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
import numpy as np
import sys
import copy
from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestHannWindow(TestCase):

    def cpu_op_exec(self, window_length):
        output = torch.hann_window(window_length)
        output = output.numpy()
        return output

    def npu_op_exec(self, window_length):
        output = torch.hann_window(window_length, device='npu')
        output = output.to('cpu')
        output = output.numpy()
        return output

    def cpu_op_exec_periodic(self, window_length, periodic):
        output = torch.hann_window(window_length, periodic)
        output = output.numpy()
        return output

    def npu_op_exec_periodic(self, window_length, periodic):
        output = torch.hann_window(window_length, periodic, device='npu')
        output = output.to('cpu')
        output = output.numpy()
        return output


    def test_hann_window(self, device):
        shape_format = [
            [0, torch.float32],
            [1, torch.float32],
            [7, torch.float32],
            [12, torch.float32],
            [0, torch.int32],
            [1, torch.int32],
            [7, torch.int32],
            [12, torch.int32]]
        for item in shape_format:
            cpu_output = self.cpu_op_exec(item[0])
            npu_output = self.npu_op_exec(item[0])
            self.assertRtolEqual(cpu_output, npu_output)

    def test_hann_window_periodic(self, device):
        shape_format = [
            [0, False, torch.float32],
            [1, False, torch.float32],
            [7, False, torch.float32],
            [12, False, torch.float32],
            [0, False, torch.int32],
            [1, False, torch.int32],
            [7, False, torch.int32],
            [12, False, torch.int32]]
        for item in shape_format:
            cpu_output = self.cpu_op_exec_periodic(item[0], item[1])
            npu_output = self.npu_op_exec_periodic(item[0], item[1])
            self.assertRtolEqual(cpu_output, npu_output)
            
    
    

instantiate_device_type_tests(TestHannWindow, globals(), except_for='cpu')
if __name__ == "__main__":
    torch.npu.set_device("npu:6")
    run_tests()
