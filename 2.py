import numpy as np

a = np.fromfile("pre-ffn3-b1.bin", dtype=np.float32)
b = np.fromfile("pre-ffn3-b1-chunk.bin", dtype=np.float32)

# [token, hidden]
a = a.reshape(35, 4096)
b = b.reshape(35, 4096)

ranges = [
    ("chunk0", 0, 8),
    ("chunk1", 8, 17),
    ("chunk2", 17, 26),
    ("chunk3", 26, 35),
]

for name, start, end in ranges:
    x = a[start:end]
    y = b[start:end]

    diff = np.abs(x - y)

    print(f"\n{name}: tokens [{start}, {end})")
    print("exact       :", np.array_equal(x, y))
    print("num != 0    :", np.count_nonzero(diff))
    print("max diff    :", diff.max())
    print("mean diff   :", diff.mean())
