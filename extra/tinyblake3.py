import functools
import sys, time, math, unittest, blake3
from tinygrad import Tensor, Variable, dtypes, TinyJit
from tinygrad.helpers import Context

BLOCK_SIZE, CHUNK_SIZE = 64, 1024
IV = [ 0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19 ]
BLK_PERMUTATION = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]

def brot(x: Tensor, n: int) -> Tensor: return (x.mul(x.full_like(2**(x.dtype.itemsize*8-n)))) + (x.idiv(x.full_like(2**n)))

def g_partial(state: Tensor, m: Tensor, rot1: int, rot2: int, *selectors: int) -> Tensor:
    a, b, c, d = (state.shrink((None, (s, s+1))) for s in selectors)

    a = (a + b + m)
    d = brot(d ^ a, rot1)
    c = (c + d)
    b = brot(b ^ c, rot2)

    values = functools.reduce(lambda a, b: a + b, [ v.pad([None, (s, 15-s)], value=0) for v, s in zip((a, b, c, d), selectors) ])
    mask = functools.reduce(lambda a, b: a + b, [ v.full_like(1).pad([None, (s, 15-s)], value=0) for v, s in zip((a, b, c, d), selectors) ])

    return state.masked_fill(mask, values)

def g_full(state: Tensor, m: Tensor, *selectors: int) -> Tensor:
    state = g_partial(state, m.shrink((None, (0, 1))), 16, 12, *selectors)
    return g_partial(state, m.shrink((None, (1, 2))), 8, 7, *selectors)

def round(state: Tensor, m: Tensor) -> Tensor:
    for i in range(4): state = g_full(state, m.shrink((None, (i*2, 16))), i, i+4, i+8, i+12)
    for i in range(4): state = g_full(state, m.shrink((None, (i*2+8, 16))), i, (i+1)%4+4, (i+2)%4+8, (i+3)%4+12)
    return state

@TinyJit
def compress(state: Tensor, block: Tensor) -> Tensor:
    assert len(state.shape) == 2 and state.shape[1] == 16 and state.dtype == dtypes.uint32
    assert len(block.shape) == 2 and block.shape[1] == 16 and block.dtype == dtypes.uint32
    assert state.shape[0] == block.shape[0]

    for _ in range(6):
        state, block = round(state, block), Tensor.cat(*[ block.shrink([None, (i, i+1)]) for i in BLK_PERMUTATION ], dim=-1)

    state = round(state, block)
    return (state.shrink((None, (0, 8))) ^ state.shrink((None, (8, 16)))).contiguous()

def compress_chunk(iv0: Tensor, chunk: Tensor, counter: Tensor, is_root: bool, is_parent: bool):
    assert len(chunk.shape) == 3 and chunk.shape[2] <= CHUNK_SIZE and chunk.dtype == dtypes.uint8
    assert counter.shape == (chunk.shape[1],) and counter.dtype == dtypes.uint64

    uint32_kwargs = dict(dtype=dtypes.uint32, device=chunk.device)
    cbatch_var = Variable("cbatch", 0, 2**20, dtype=dtypes.int32).bind(math.prod(chunk.shape[:2]))

    counter_lower, counter_upper = counter.cast(dtypes.uint32).reshape(1, -1, 1), (counter >> 32).cast(dtypes.uint32).reshape(1, -1, 1)
    pupper_state = Tensor.cat(iv0[:4].reshape(1, 1, 4).expand(1, chunk.shape[1], 4), counter_lower, counter_upper, dim=-1).contiguous().realize()

    block_index = 0
    cv = iv0.reshape(1, 1, -1).expand(*chunk.shape[:2], -1)

    while (block_start:=block_index*BLOCK_SIZE) < chunk.shape[2]:
        is_chunk_end = block_start + BLOCK_SIZE >= chunk.shape[2]
        flags = int(block_index == 0 and not is_parent) | ((is_chunk_end and not is_parent) << 1) | ((is_parent) << 2) | ((is_chunk_end and is_root) << 3)

        block_len = min(chunk.shape[2] - block_start, BLOCK_SIZE)
        block = chunk[:,:,block_start:block_start+BLOCK_SIZE].pad([None, None, (0, BLOCK_SIZE - block_len)], value=0).bitcast(dtypes.uint32)

        block_len_tensor, flags_tensor = (Tensor(v, **uint32_kwargs).reshape(1, 1, 1).expand(*chunk.shape[:2], 1) for v in (block_len, flags))

        state = cv.cat(pupper_state.expand(*chunk.shape[:2], -1), block_len_tensor, flags_tensor, dim=-1).contiguous().reshape(cbatch_var, 16)
        block = block.contiguous().reshape(cbatch_var, 16)

        cv = compress(state, block).reshape(*chunk.shape[:2], 8).realize()

        block_index += 1

    return cv.bitcast(dtypes.uint8).realize() # NOTE: breaks if the realize is removed

def mblake3(t: Tensor, dim: int = -1):
    if t.dtype != dtypes.uint8: raise ValueError("Expected tensor of dtype uint8")

    d = t.reshape(math.prod(t.shape[:dim]), -1) # (batch, data)
    iv0 = Tensor(IV, dtype=dtypes.uint32, device=t.device)
    n_full_chunks = d.shape[-1] // CHUNK_SIZE
    cvs: list[Tensor] = []

    if n_full_chunks > 0:
        chunks = d[:,:n_full_chunks*CHUNK_SIZE].reshape(d.shape[0], n_full_chunks, -1)
        counter = Tensor.arange(n_full_chunks, dtype=dtypes.uint64, device=t.device)
        cvs.append(compress_chunk(iv0, chunks, counter, is_root=d.shape[1] == CHUNK_SIZE, is_parent=False))

    if n_full_chunks * CHUNK_SIZE < d.shape[1] or d.shape[1] == 0:
        chunks = d[:,n_full_chunks*CHUNK_SIZE:].reshape(d.shape[0], 1, -1)
        counter = Tensor(n_full_chunks, dtype=dtypes.uint64, device=t.device).reshape(1)
        cvs.append(compress_chunk(iv0, chunks, counter, is_root=d.shape[1] == chunks.shape[2], is_parent=False))

    cv = cvs[0] if len(cvs)==1 else cvs[0].cat(cvs[1], dim=1)

    while cv.shape[1] > 1:
        chunks, append_cv = (cv[:,:-1], cv[:,-1:]) if cv.shape[1] % 2 else (cv, None)
        chunks = chunks.reshape(cv.shape[0], -1, BLOCK_SIZE)
        counter = Tensor(0, dtype=dtypes.uint64, device=t.device).expand(chunks.shape[1])
        cv = compress_chunk(iv0, chunks, counter, is_root=cv.shape[1] == 2, is_parent=True)
        if append_cv is not None: cv = cv.cat(append_cv, dim=1)

    return cv.reshape(*t.shape[:dim], -1)

class BlakeTests(unittest.TestCase):
    def _test_data(self, data: bytes):
        tiny_hash = bytes(mblake3(Tensor(data)).data())
        ref_hash = blake3.blake3().update(data).digest()
        self.assertEqual(tiny_hash.hex(), ref_hash.hex())

    def test_len_65(self): self._test_data(b"1"*65)
    def test_chunk_1(self): self._test_data(b"1"*CHUNK_SIZE)
    def test_chunk_2(self): self._test_data(b"1"*CHUNK_SIZE*2)
    def test_chunk_3(self): self._test_data(b"1"*CHUNK_SIZE*3)
    def test_hello_world(self): self._test_data(b"hello, world")
    def test_long(self): self._test_data(b"abc" * 1000)
    def test_big(self): self._test_data(b"1" * 10**6)
    def test_chunk_p1(self): self._test_data(b"1"*(CHUNK_SIZE+1))
    # def test_empty(self): self._test_data(b"")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "perf":
            import matplotlib.pyplot as plt

            sizes = []
            total_times = []
            mb_per_sec = []

            mblake3(Tensor(b"1"*65)).realize()
            print("warmup done")
            for i in range(29):
                size = 2**i
                rnd_tensor = Tensor.randint(size, high=256, dtype=dtypes.uint8).realize()
                start_time = time.time()
                for i in range(5):
                    tiny_hash_tensor = mblake3(rnd_tensor).realize()
                end_time = time.time()
                total_time = (end_time - start_time) / 5
                mb_per_sec_value = (size / (1024 * 1024)) / total_time

                ref_hash = blake3.blake3().update(rnd_tensor.data()).digest()
                del rnd_tensor
                tiny_hash = bytes(tiny_hash_tensor.data())
                if tiny_hash.hex() != ref_hash.hex():
                    print(f"Size: {size} bytes, TinyHash: {tiny_hash.hex()}, RefHash: {ref_hash.hex()}")
                    break

                sizes.append(size)
                total_times.append(total_time)
                mb_per_sec.append(mb_per_sec_value)

                print(f"Size: {size} bytes, Total time: {total_time:.6f} seconds, MB/sec: {mb_per_sec_value:.6f}")

            # Plotting the graphs
            plt.figure(figsize=(12, 6))

            # Total time vs size
            plt.subplot(1, 2, 1)
            plt.plot(sizes, total_times, marker='o')
            plt.xscale('log')
            plt.xlabel('Size (bytes)')
            plt.ylabel('Total Time (seconds)')
            plt.title('Total Time vs Size')

            # MB/sec vs size
            plt.subplot(1, 2, 2)
            plt.plot(sizes, mb_per_sec, marker='o')
            plt.xscale('log')
            plt.xlabel('Size (bytes)')
            plt.ylabel('MB/sec')
            plt.title('MB/sec vs Size')

            plt.tight_layout()
            plt.show()
        elif sys.argv[1] == "kernel_g":
            with Context(DEBUG=0):
                state = Tensor.randint((1, 16), high=1000, dtype=dtypes.uint32).realize()
                m = Tensor.randint((1, 1), high=1000, dtype=dtypes.uint32).realize()

            rot1 = 16
            rot2 = 12
            selectors = (0, 4, 8, 12)

            result = g_partial(state, m, rot1, rot2, *selectors).realize()
            print("Result tensor:", state.tolist(), result.tolist())
        elif sys.argv[1] == "bigtest":
            with Context(DEBUG=0):
                mblake3(Tensor(b"1"*65)).realize()
                rnd_tensor = Tensor.randint(10**8, high=256, dtype=dtypes.uint8).realize()
                print("Warmup done")

            tiny_hash = bytes(mblake3(rnd_tensor).data())
            ref_hash = blake3.blake3().update(bytes(rnd_tensor.data())).digest()

            print("tiny_hash:", tiny_hash.hex())
            print("ref_hash: ", ref_hash.hex())
    else:
        unittest.main()
