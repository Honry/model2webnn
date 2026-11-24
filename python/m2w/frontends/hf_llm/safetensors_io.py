"""Minimal safetensors reader (numpy only): header parsing, memory-mapped tensor access, bf16 / f16 / f32 -> float32.
No torch and no `safetensors` package needed; works on single files and on sharded checkpoints (model.safetensors.index.json)."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

_DTYPES = {"F32": np.float32, "F16": np.float16, "BF16": None, "F64": np.float64, "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_}


class TensorStore:
    def __init__(self, files):
        self._index: dict = {}   # name -> (path, dtype, shape, begin, end, data_start)
        for path in map(Path, files):
            with open(path, "rb") as f:
                (header_size,) = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(header_size))
            start = 8 + header_size
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                begin, end = info["data_offsets"]
                self._index[name] = (path, info["dtype"], tuple(info["shape"]), begin, end, start)

    @classmethod
    def open(cls, model_dir) -> "TensorStore":
        d = Path(model_dir)
        index = d / "model.safetensors.index.json"
        if index.exists():
            files = sorted({d / f for f in json.loads(index.read_text())["weight_map"].values()})
        else:
            files = sorted(d.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no .safetensors files in {d}")
        return cls(files)

    def names(self):
        return list(self._index)

    def __contains__(self, name: str) -> bool:
        return name in self._index

    def shape(self, name: str) -> tuple:
        return self._index[name][2]

    def dtype(self, name: str) -> str:
        return self._index[name][1]

    def read(self, name: str, rows=None) -> np.ndarray:
        """The tensor (or rows ``rows = (start, stop)`` of its first axis) as float32."""
        path, dtype, shape, begin, end, data_start = self._index[name]
        if dtype not in _DTYPES:
            raise ValueError(f"unsupported safetensors dtype {dtype} for {name}")
        raw = np.memmap(path, mode="r", dtype=np.uint8, offset=data_start + begin, shape=(end - begin,))
        item = 2 if dtype in ("BF16", "F16") else np.dtype(_DTYPES[dtype]).itemsize
        if rows is not None:
            row_bytes = item * int(np.prod(shape[1:], dtype=np.int64))
            raw = raw[rows[0] * row_bytes: rows[1] * row_bytes]
            shape = (rows[1] - rows[0],) + shape[1:]
        if dtype == "BF16":
            return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
        return np.frombuffer(raw, dtype=_DTYPES[dtype]).astype(np.float32).reshape(shape)


def write_safetensors(path, tensors: dict, dtype: str = "F32") -> None:
    """Writer for tests / fixtures. tensors: name -> float32 array; stored as F32 / F16 / BF16."""
    header, blobs, offset = {}, [], 0
    for name, array in tensors.items():
        a = np.ascontiguousarray(array, dtype=np.float32)
        if dtype == "BF16":
            blob = (a.view(np.uint32) >> 16).astype(np.uint16).tobytes()   # truncation: fine for fixtures
        elif dtype == "F16":
            blob = a.astype(np.float16).tobytes()
        else:
            blob = a.tobytes()
        header[name] = {"dtype": dtype, "shape": list(a.shape), "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))
