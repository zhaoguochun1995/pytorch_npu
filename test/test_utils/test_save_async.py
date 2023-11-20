import os
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.utils.path_manager import PathManager


class TestAsyncSave(TestCase):
    test_save_path = os.path.join(
        os.path.realpath(os.path.dirname(__file__)), "test_save_async")

    @classmethod
    def setUpClass(cls):
        PathManager.make_dir_safety(TestAsyncSave.test_save_path)

    @classmethod
    def tearDownClass(cls):
        time.sleep(5)
        PathManager.remove_path_safety(TestAsyncSave.test_save_path)

    def test_save_async_tensor(self):
        save_tensor = torch.rand(1024, dtype=torch.float32).npu()
        async_save_path = os.path.join(TestAsyncSave.test_save_path, "async_save_tensor.pt")
        torch_npu.utils.save_async(save_tensor, async_save_path)

        torch.npu.synchronize()
        tensor_async = torch.load(async_save_path)

        self.assertEqual(save_tensor, tensor_async)
    
    def test_save_async(self):
        loss1 = [1.6099495, 1.6099086, 1.6098710]
        loss2 = []
        model_list = []
        checkpoint_list = []
        model_origin = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
            nn.ReLU()
        )

        input_data = torch.ones(6400, 100).npu()
        labels = torch.arange(5).repeat(1280).npu()

        criterion = nn.CrossEntropyLoss()
        model = model_origin.npu()
        optimerizer = optim.SGD(model.parameters(), lr=0.1)
        for step in range(3):
            outputs = model(input_data)
            loss = criterion(outputs, labels)

            optimerizer.zero_grad()
            loss.backward()

            optimerizer.step()
            
            loss2.append(loss)
            checkpoint = {
                "model" : model.state_dict(),
                "optimizer" : optimerizer.state_dict()
            }
            checkpoint_list.append(copy.deepcopy(checkpoint))
            model_list.append(copy.deepcopy(model))
            checkpoint_async_path = os.path.join(TestAsyncSave.test_save_path, f"checkpoint_async_{step}.path")
            model_async_path = os.path.join(TestAsyncSave.test_save_path, f"model_async_{step}.path")
            torch_npu.utils.save_async(checkpoint, checkpoint_async_path, model=model)
            torch_npu.utils.save_async(model, model_async_path, model=model)

            time.sleep(1)

        torch.npu.synchronize()

        for i in range(3):
            self.assertEqual(loss1[i], loss2[i].item())
            checkpoint_async_path = os.path.join(TestAsyncSave.test_save_path, f"checkpoint_async_{i}.path")
            checkpoint_async = torch.load(checkpoint_async_path)
            self.assertEqual(checkpoint_list[i], checkpoint_async, prec=2e-3)
            model_async_path = os.path.join(TestAsyncSave.test_save_path, f"model_async_{i}.path")
            model_async = torch.load(model_async_path)

            state_dict_sync = model_list[i].state_dict()
            state_dict_async = model_async.state_dict()

            key_sync = sorted(state_dict_sync.keys())
            key_async = sorted(state_dict_async.keys())

            self.assertEqual(key_sync, key_async)
            for key in key_async:
                self.assertEqual(state_dict_async[key], state_dict_sync[key], prec=2e-3)

if __name__ == '__main__':
    torch.npu.set_device(0)
    run_tests()
