import unittest
from tinygrad import Tensor, dtypes
from tinygrad.device import Device

class TestAutoCopy(unittest.TestCase):
  def test_ones(self):
    self.assertEqual(Tensor.ones(4).autocopy().tolist(), [1, 1, 1, 1])

  def test_multi_assign(self):
    devices = tuple(f"PYTHON:{i}" for i in range(2))
    a = Tensor([1, 2, 3, 4], dtype=dtypes.float).shard(devices).realize().to(devices[0]).shard(devices, 0).autocopy()
    a.assign(a + 1).realize()
    self.assertEqual(a.tolist(), [2, 3, 4, 5])

  def test_shard_add(self):
    devices = tuple(f"PYTHON:{i}" for i in range(4))
    a = Tensor([1, 2, 3, 4]).shard(devices[:2], 0).realize().to(devices[0]).shard(devices[:2]).autocopy(1)
    b = Tensor([5, 6, 7, 8]).to(devices[2]).realize().autocopy()
    self.assertEqual((a + b).tolist(), [6, 8, 10, 12])

  def test_shard_gradient(self):
    devices = tuple(f"PYTHON:{i}" for i in range(4))
    a = Tensor([1, 2, 3, 4], dtype=dtypes.float).shard(devices[:2], 0).realize().autocopy(1).to(devices[0]).shard(devices[:2])
    b = Tensor([5, 6, 7, 8], dtype=dtypes.float).shard(devices[2:4], 0).realize().autocopy(0).to(devices[2]).shard(devices[2:4])
    a.assign(a - 2 * (a * b).sum().gradient(a)[0]).realize()
    self.assertEqual(a.tolist(), [-9.0, -10.0, -11.0, -12.0])

if __name__ == "__main__":
  unittest.main()