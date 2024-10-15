import ctypes
import os
import numpy as np
import unittest
from extra.gguf import GGUFConverters, load_gguf
from tinygrad import Device, dtypes, fetch, Tensor
import ggml

np.random.seed(1337)
block_count = 4

GGML_TYPE_TO_NP_DTYPE = {
  ggml.GGML_TYPE_F16: np.float16, ggml.GGML_TYPE_F32:np.float32, ggml.GGML_TYPE_F64:np.float64,
  ggml.GGML_TYPE_I8:np.int8, ggml.GGML_TYPE_I16: np.int16, ggml.GGML_TYPE_I32: np.int32, ggml.GGML_TYPE_I64: np.int64,
}
NP_DTYPE_TO_CTYPE = { np.float16: ctypes.c_uint16 }

def ggml_tensor_to_numpy(tensor: ggml.ggml_tensor_p):
  ctx: ggml.ggml_context_p | None = None
  ggml_type, n_dims, n_els = tensor.contents.type, ggml.ggml_n_dims(tensor), ggml.ggml_nelements(tensor)
  shape = tuple(reversed(tensor.contents.ne[:n_dims]))
  if ggml_type not in GGML_TYPE_TO_NP_DTYPE:
    ctx = ggml.ggml_init(ggml.ggml_init_params(mem_size=n_els * 5 + 500, mem_buffer=None))
    ntensor = ggml.ggml_new_tensor(ctx, ggml.GGML_TYPE_F32, n_dims, tensor.contents.ne)
    type_traits = ggml.ggml_internal_get_type_traits(ggml_type)
    type_traits.to_float(ggml.ggml_get_data(tensor), ggml.ggml_get_data_f32(ntensor), n_els)
    tensor, ggml_type = ntensor, ggml.GGML_TYPE_F32

  np_type = GGML_TYPE_TO_NP_DTYPE[ggml_type]
  ctypes_type = NP_DTYPE_TO_CTYPE.get(np_type, None) or np.ctypeslib.as_ctypes_type(np_type)
  data = ggml.ggml_get_data(tensor)
  if data is None: raise ValueError("tensor data is None")
  arr = (ctypes_type * ggml.ggml_nelements(tensor)).from_address(data)
  strides = tuple(reversed(tensor.contents.nb[:n_dims]))
  output = np.ctypeslib.as_array(arr)
  output.dtype = np_type
  return np.lib.stride_tricks.as_strided(output, shape=shape, strides=strides), ctx

gguf_val_getters = [
  ggml.gguf_get_val_u8, ggml.gguf_get_val_i8, ggml.gguf_get_val_u16, ggml.gguf_get_val_i16,
  ggml.gguf_get_val_u32, ggml.gguf_get_val_i32, ggml.gguf_get_val_f32, ggml.gguf_get_val_bool,
  lambda *args: ggml.gguf_get_val_str(*args).decode("utf-8"), None,
  ggml.gguf_get_val_u64, ggml.gguf_get_val_i64, ggml.gguf_get_val_f64,
]

class TestGGML(unittest.TestCase):
  def setUp(self) -> None:
    params = ggml.ggml_init_params(mem_size=1024*1024, mem_buffer=None)
    self.ctx = ggml.ggml_init(params)
  def tearDown(self) -> None:
    ggml.ggml_free(self.ctx)

  def test_dequantization_q4_0(self): self._test_dequantization(ggml.GGML_TYPE_Q4_0)
  def test_dequantization_q4_1(self): self._test_dequantization(ggml.GGML_TYPE_Q4_1)
  def test_dequantization_q8_0(self): self._test_dequantization(ggml.GGML_TYPE_Q8_0)
  def test_dequantization_q6_k(self): self._test_dequantization(ggml.GGML_TYPE_Q6_K)
  def _test_dequantization(self, ttype: int):
    type_traits = ggml.ggml_internal_get_type_traits(ttype)
    n_el, n_bytes = block_count * type_traits.blck_size, block_count * type_traits.type_size

    data_in = (np.random.random((n_el,)).astype(np.float32) * 100 - 50).ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    c_q_data, c_dq_data = (ctypes.c_char * n_bytes)(0), (ctypes.c_float * n_el)(0)
    type_traits.from_float(data_in, c_q_data, n_el)
    type_traits.to_float(c_q_data, c_dq_data, n_el)

    q_tensor = Tensor(np.frombuffer(c_q_data, dtype=np.uint8, count=n_bytes))
    dq_tensor = GGUFConverters.converter_map[ttype](q_tensor, n_el).reshape(n_el)

    np.testing.assert_equal(dq_tensor.numpy(), np.frombuffer(c_dq_data, dtype=np.float32))

class TestGGUF(unittest.TestCase):
  def test_load_gpt2_q8_0(self): self._test_load_gguf("https://huggingface.co/PrunaAI/gpt2-GGUF-smashed/resolve/main/gpt2.Q8_0.gguf?download=true")
  def test_load_gpt2_q4_0(self): self._test_load_gguf("https://huggingface.co/PrunaAI/gpt2-GGUF-smashed/resolve/main/gpt2.Q4_0.gguf?download=true")
  def test_load_gpt2_q4_1(self): self._test_load_gguf("https://huggingface.co/PrunaAI/gpt2-GGUF-smashed/resolve/main/gpt2.Q4_1.gguf?download=true")
  def test_load_gpt2_q6_k(self): self._test_load_gguf("https://huggingface.co/PrunaAI/gpt2-GGUF-smashed/resolve/main/gpt2.Q6_K.gguf?download=true")
  def _test_load_gguf(self, url: str):
    fp = fetch(url)
    model_size = os.stat(fp).st_size
    gguf_tensor = Tensor.empty(model_size, dtype=dtypes.uint8, device=f"disk:{fp}").to(Device.DEFAULT)
    kv_data, tensors = load_gguf(gguf_tensor)

    params = ggml.ggml_init_params(mem_size=0, mem_buffer=None, no_alloc=False)
    ctx = ctypes.cast(ggml.ggml_init(params), ctypes.POINTER(ctypes.c_void_p))

    gguf_params = ggml.gguf_init_params(ctx=ctx, no_alloc=False)
    gguf_ctx = ggml.gguf_init_from_file(str(fp).encode("utf8"), gguf_params)
    param_ctx = gguf_params.ctx.contents.value

    for ggml_tensor_idx in range(ggml.gguf_get_n_tensors(gguf_ctx)):
      tensor_name = ggml.gguf_get_tensor_name(gguf_ctx, ggml_tensor_idx)
      ggml_tensor = ggml.ggml_get_tensor(param_ctx, tensor_name)
      ggml_tensor_numpy, temp_ctx = ggml_tensor_to_numpy(ggml_tensor)
      tensor = tensors.get(tensor_name.decode("utf-8"))
      np.testing.assert_equal(tensor.numpy(), ggml_tensor_numpy)
      if temp_ctx is not None: ggml.ggml_free(temp_ctx)

    for gguf_key_id in range(ggml.gguf_get_n_kv(gguf_ctx)):
      v = kv_data[ggml.gguf_get_key(gguf_ctx, gguf_key_id).decode("utf-8")]
      v_type = ggml.gguf_get_kv_type(gguf_ctx, gguf_key_id)
      if (get_fn := gguf_val_getters[v_type]) is not None: self.assertEqual(get_fn(gguf_ctx, gguf_key_id), v)

    ggml.gguf_free(gguf_ctx)
    ggml.ggml_free(ctx)

if __name__ == '__main__':
  unittest.main()