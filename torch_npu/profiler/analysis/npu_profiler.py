# Copyright (c) 2023, Huawei Technologies.
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

import os
from multiprocessing import Process

from .prof_common_func.constant import Constant
from .prof_view.view_parser_factory import ViewParserFactory
from .prof_common_func.path_manager import ProfilerPathManager
from ...utils.path_manager import PathManager


class NpuProfiler:

    @classmethod
    def analyse(cls, input_path: str, analysis_type: str = Constant.TENSORBOARD_TRACE_HANDLER, output_path: str = None,
                **kwargs):
        input_path = ProfilerPathManager.get_realpath(input_path)
        cls._check_input_path(input_path)
        profiler_path_list = ProfilerPathManager.get_profiler_path_list(input_path)
        if not profiler_path_list:
            return
        # 多个profiler用多进程处理
        process_list = []
        for profiler_path in profiler_path_list:
            PathManager.check_directory_path_writeable(profiler_path)
            process = Process(target=ViewParserFactory.create_view_parser_and_run,
                              args=(profiler_path, analysis_type, output_path, kwargs))
            process.start()
            process_list.append(process)

        for process in process_list:
            process.join()

    @classmethod
    def _check_input_path(cls, path: str):
        PathManager.check_input_directory_path(path)
        PathManager.check_path_owner_consistent(path)
