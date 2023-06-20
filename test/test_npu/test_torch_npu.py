import contextlib
import collections
import time

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

class TorchNPUApiTestCase(TestCase):
    def test_npu_current_device(self):
        res = torch_npu.npu.current_device()
        self.assertIsInstance(res, int)

    def test_npu_current_stream(self):
        res = torch_npu.npu.current_stream()
        self.assertIsInstance(res, torch_npu.npu.streams.Stream)
    
    def test_npu_default_stream(self):
        res = torch_npu.npu.default_stream()
        self.assertIsInstance(res, torch_npu.npu.streams.Stream)

    def test_npu_device(self):
        res = torch_npu.npu.device("npu:0")
        self.assertIsInstance(res, torch_npu.npu.device)

    def test_npu_device_count(self):
        res = torch_npu.npu.device_count()
        self.assertIsInstance(res, int)

    def test_npu_device_of(self):
        x = torch.Tensor([1,2,3]).to("npu")
        res = torch_npu.npu.device_of(x)
        self.assertIsInstance(res, torch_npu.npu.device_of)

    def test_npu_init(self):
        res = torch_npu.npu.init()
        self.assertIsNone(res)

    def test_npu_is_available(self):
        res = torch_npu.npu.is_available()
        self.assertIsInstance(res, bool)

    def test_npu_is_initialized(self):
        res = torch_npu.npu.is_initialized()
        self.assertIsInstance(res, bool)

    def test_npu_stream(self):
        s = torch_npu.npu.current_stream()
        res = torch_npu.npu.stream(s)
        self.assertIsInstance(res, contextlib._GeneratorContextManager)

    def test_npu_synchronize(self):
        res = torch_npu.npu.synchronize()
        self.assertIsNone(res)

    def test_npu_stream_class(self):
        s = torch_npu.npu.Stream()
        self.assertIsInstance(s, torch_npu.npu.Stream)

    def test_npu_stream_record_evnet(self):
        s = torch_npu.npu.current_stream()
        res = s.record_event()
        self.assertIsInstance(res, torch_npu.npu.Event)

    def test_npu_stream_synchronize(self):
        s = torch_npu.npu.current_stream()
        res = s.synchronize()
        self.assertIsNone(res)

    def test_npu_stream_wait_event(self):
        s = torch_npu.npu.current_stream()
        e = torch_npu.npu.Event()
        res = s.wait_event(e)
        self.assertIsNone(res)

    def test_npu_event(self):
        res = torch_npu.npu.Event(enable_timing=True, blocking=True, interprocess=True)
        self.assertIsInstance(res, torch_npu.npu.Event)

    def test_npu_event_elapsed_time(self):
        start_event = torch_npu.npu.Event()
        end_event = torch_npu.npu.Event()
        start_event.record()
        end_event.record()
        res = start_event.elapsed_time(end_event)
        self.assertIsInstance(res, float)

    def test_npu_event_query(self):
        event = torch_npu.npu.Event()
        res = event.query()
        self.assertIsInstance(res, bool)

    def test_npu_event_record(self):
        event = torch_npu.npu.Event()
        res = event.record()
        self.assertIsNone(res)

    def test_npu_event_synchronize(self):
        event = torch_npu.npu.Event()
        res = event.synchronize()
        self.assertIsNone(res)

    def test_npu_event_wait(self):
        event = torch_npu.npu.Event()
        res = event.wait()
        self.assertIsNone(res)

    def test_npu_empty_cache(self):
        res = torch_npu.npu.empty_cache()
        self.assertIsNone(res)

    def test_npu_memory_stats(self):
        res = torch_npu.npu.memory_stats()
        self.assertIsInstance(res, collections.OrderedDict)

    def test_npu_memory_summary(self):
        res = torch_npu.npu.memory_summary()
        self.assertIsInstance(res, str)

    def test_npu_memory_snapshot(self):
        res = torch_npu.npu.memory_snapshot()
        self.assertIsInstance(res, list)

    def test_npu_memory_allocated(self):
        res = torch_npu.npu.memory_allocated()
        self.assertIsInstance(res, int)

    def test_npu_max_memory_allocated(self):
        res = torch_npu.npu.max_memory_allocated()
        self.assertIsInstance(res, int)

    def test_npu_reset_max_memory_allocated(self):
        res = torch_npu.npu.reset_max_memory_allocated()
        self.assertIsNone(res)

    def test_npu_memory_reserved(self):
        res = torch_npu.npu.memory_reserved()
        self.assertIsInstance(res, int)

    def test_npu_max_memory_reserved(self):
        res = torch_npu.npu.max_memory_reserved()
        self.assertIsInstance(res, int)

    def test_npu_memory_cached(self):
        res = torch_npu.npu.memory_cached()
        self.assertIsInstance(res, int)

    def test_npu_max_memory_cached(self):
        res = torch_npu.npu.max_memory_cached()
        self.assertIsInstance(res, int)

    def test_npu_reset_max_memory_cached(self):
        res = torch_npu.npu.reset_max_memory_cached()
        self.assertIsNone(res)

    def test_npu_get_device_name(self):
        res = torch_npu.npu.get_device_name(0)
        self.assertIsInstance(res, str)
        res = torch_npu.npu.get_device_name()
        self.assertIsInstance(res, str)
        res = torch_npu.npu.get_device_name("npu:0")
        self.assertIsInstance(res, str)
        device = torch.device("npu:0")
        res = torch_npu.npu.get_device_name(device)
        self.assertIsInstance(res, str)

    def test_npu_get_device_properties(self):
        name = torch_npu.npu.get_device_properties(0).name
        self.assertIsInstance(name, str)
        total_memory = torch_npu.npu.get_device_properties(0).total_memory
        self.assertIsInstance(total_memory, int)

    def test_npu_get_device_capability(self):
        res = torch.npu.get_device_capability()
        self.assertEqual(res, None)
        res = torch.npu.get_device_capability(0)
        self.assertEqual(res, None)
        name = torch.npu.get_device_properties(0).name
        res = torch.npu.get_device_capability(name)
        self.assertEqual(res, None)
        device = torch.npu.device("npu")
        res = torch.npu.get_device_capability(device)
        self.assertEqual(res, None)
        
    def test_npu_get_aclnn_version(self):
        res = torch.npu.aclnn.version()
        self.assertEqual(res, None)

    def test_npu_get_utilization(self):
        res = torch.npu.utilization()
        self.assertEqual(res, 0)

    def test_npu_get_utilization_runing(self):
        for i in range(200):
            tensora = torch.randn(64, 100, 64, 64, dtype=torch.float, device="npu:0")
            torch.sum(tensora)
        util_sum = 0
        for i in range(200):
            res = torch.npu.utilization()
            util_sum = util_sum + res
            time.sleep(0.2)
        self.assertNotEqual(util_sum, 0)

if __name__ == "__main__":
    run_tests()
