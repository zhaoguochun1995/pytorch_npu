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

from typing import Union

from ..prof_common_func.constant import Constant
from ..prof_common_func.constant import convert_us2ns


class EventBean:
    HOST_TO_DEVICE = "HostToDevice"
    START_FLOW = "s"
    END_FLOW = "f"

    def __init__(self, data: dict):
        self._origin_data = data

    @property
    def ts(self) -> int:
        # Time unit is ns
        ts_us = self._origin_data.get("ts", 0)
        ts_ns = convert_us2ns(ts_us)
        return ts_ns

    @property
    def pid(self) -> str:
        return self._origin_data.get("pid", "")

    @property
    def tid(self) -> int:
        return self._origin_data.get("tid", 0)

    @property
    def dur(self) -> Union[float, str]:
        # Time unit is us
        return self._origin_data.get("dur", 0)

    @property
    def name(self) -> str:
        return self._origin_data.get("name", "")

    @property
    def id(self) -> any:
        return self._origin_data.get("id")

    @property
    def unique_id(self) -> str:
        return "{}-{}-{}".format(self.pid, self.tid, self.ts)

    @property
    def is_ai_core(self) -> bool:
        args = self._origin_data.get("args")
        if args:
            return args.get("Task Type") == Constant.AI_CORE
        return False

    def is_flow_start_event(self) -> bool:
        return self._origin_data.get("cat") == self.HOST_TO_DEVICE and self._origin_data.get("ph") == self.START_FLOW

    def is_flow_end_event(self) -> bool:
        return self._origin_data.get("cat") == self.HOST_TO_DEVICE and self._origin_data.get("ph") == self.END_FLOW

    def is_x_event(self) -> bool:
        return self._origin_data.get("ph") == "X"
