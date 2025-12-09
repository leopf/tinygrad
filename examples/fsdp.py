from tinygrad import Tensor
from tinygrad.device import Device
from tinygrad.engine.jit import TinyJit
import csv, tinygrad.nn as nn, tinygrad.helpers as helpers, typing

def fsdp(model: typing.Any, devices: list[tuple[str, ...]]):
  sd = nn.state.get_state_dict(model)
  bytes_per_dev = (sum(v.nbytes() for v in sd.values()) + 1) / len(devices)
  nbytes_on_dev = 0
  for v in sd.values():
    dev_idx = int(nbytes_on_dev / bytes_per_dev)
    v.replace(v.shard(devices[dev_idx], axis=0).realize().autocopy(1 + dev_idx).to(devices[dev_idx][0]).shard(devices[dev_idx]))
    nbytes_on_dev += v.nbytes()

data_path = helpers.fetch("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
with open(data_path) as fd:
  dict_data = list(csv.DictReader(fd))

classes = { name: idx for idx, name in enumerate(set(item["species"] for item in dict_data)) }

data_y = Tensor([ classes[item["species"]] for item in dict_data ]).autocopy()
data_x = Tensor([ [ float(v) for k, v in item.items() if k != "species" ] for item in dict_data ]).autocopy()

class Demo:
  def __init__(self) -> None:
    self.lin1 = nn.Linear(4, 4)
    self.lin2 = nn.Linear(4, 4)

  def __call__(self, x: Tensor) -> Tensor:
    return self.lin2(self.lin1(x).gelu())

model = Demo()

GPUS = [f"{Device.DEFAULT}:{i}" for i in range(helpers.getenv("GPUS", 1))]
assert len(GPUS) > 2, "fsdp demo required more than 2 devices!"
fsdp(model, [ tuple(GPUS[:2]), tuple(GPUS[2:4]) ])

optimizer = nn.optim.AdamW(nn.state.get_parameters(model), lr=0.1)

@TinyJit
def step(x: Tensor, y: Tensor):
  optimizer.zero_grad()
  y_pred = model(x)
  loss = y_pred.sparse_categorical_crossentropy(y)
  loss.backward()
  optimizer.step()
  return loss

Tensor.training = True
for i in range(101):
  loss = step(data_x, data_y)
  if i % 10 == 0:
    print(f"step: {i}, loss: {loss.item()}")
