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

import time

import torch
import numpy as np

from common_utils import TestCase, run_tests
from common_device_type import dtypes, instantiate_device_type_tests
from util_test import create_common_tensor


class TestIndexingToContiguous(TestCase):
    def test_IndexingToContiguous(self, device):
        dtype_list = [ np.float16 ,np.float32 ]
        format_list = [0, 3, 29]
        shape_list = [[2,6,6,6]]
        shape_format = [
            [i, j, k] for i in dtype_list for j in format_list for k in shape_list
        ]
        start = time.time()
        for item in shape_format:    
            a1_cpu, a1_npu = create_common_tensor(item, 0, 100)
            npu_out = a1_npu[:,1:6:2,:,:].contiguous()
            cpu_out = a1_cpu[:,1:6:2,:,:].contiguous()
            self.assertRtolEqual(npu_out.to("cpu").numpy(), cpu_out.numpy()) 

            npu_out = a1_npu[:,:,1:6:3,:].contiguous()
            cpu_out = a1_cpu[:,:,1:6:3,:].contiguous()
            self.assertRtolEqual(npu_out.to("cpu").numpy(), cpu_out.numpy()) 

            npu_out = a1_npu[:,:,:,1:6:4].contiguous()
            cpu_out = a1_cpu[:,:,:,1:6:4].contiguous()
            self.assertRtolEqual(npu_out.to("cpu").numpy(), cpu_out.numpy())               
        end = time.time()
        print("indexing to contiguous uses: %.2f s"%(end-start))      
                
instantiate_device_type_tests(TestIndexingToContiguous, globals(), except_for='cpu')
if __name__ == "__main__":
    run_tests()