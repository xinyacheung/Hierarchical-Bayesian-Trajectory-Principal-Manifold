"""Safe tensor-only reader for the nested state dicts in packaged PyTorch checkpoints."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import io
from pathlib import Path
import pickle
import zipfile

import numpy as np


@dataclass
class StorageRef:
    dtype: np.dtype
    key: str
    size: int


@dataclass
class TensorRef:
    storage: StorageRef
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]


def rebuild_tensor(storage, offset, size, stride, *unused):
    return TensorRef(storage, int(offset), tuple(int(v) for v in size), tuple(int(v) for v in stride))


class CheckpointUnpickler(pickle.Unpickler):
    storage_dtypes = {
        "FloatStorage": np.dtype("float32"), "DoubleStorage": np.dtype("float64"),
        "HalfStorage": np.dtype("float16"), "LongStorage": np.dtype("int64"),
        "IntStorage": np.dtype("int32"), "ShortStorage": np.dtype("int16"),
        "ByteStorage": np.dtype("uint8"), "BoolStorage": np.dtype("bool"),
    }

    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "torch._utils" and name in {"_rebuild_tensor_v2", "_rebuild_tensor"}:
            return rebuild_tensor
        if module == "torch._utils" and name.startswith("_rebuild_parameter"):
            return lambda tensor, *args: tensor
        if module == "torch" and name in self.storage_dtypes:
            return self.storage_dtypes[name]
        raise pickle.UnpicklingError(f"unsupported checkpoint global {module}.{name}")

    def persistent_load(self, saved_id):
        if not isinstance(saved_id, tuple) or saved_id[0] != "storage":
            raise pickle.UnpicklingError(f"unsupported persistent id {saved_id!r}")
        _, dtype, key, _location, size = saved_id[:5]
        return StorageRef(np.dtype(dtype), str(key), int(size))


def load_checkpoint_state_without_torch(path: Path) -> OrderedDict[str, np.ndarray]:
    """Load only arrays from a torch.save ZIP; no model code or torch import occurs."""
    with zipfile.ZipFile(path) as archive:
        pkl_name = next(name for name in archive.namelist() if name.endswith("data.pkl"))
        prefix = pkl_name[: -len("data.pkl")]
        value = CheckpointUnpickler(io.BytesIO(archive.read(pkl_name))).load()
        byteorder_name = prefix + "byteorder"
        byteorder = archive.read(byteorder_name).decode("ascii").strip() if byteorder_name in archive.namelist() else "little"
        storage_cache = {}

        def resolve(item):
            if isinstance(item, (dict, OrderedDict)):
                return OrderedDict((str(key), resolve(child)) for key, child in item.items())
            if isinstance(item, list):
                return [resolve(child) for child in item]
            if isinstance(item, tuple):
                return tuple(resolve(child) for child in item)
            if not isinstance(item, TensorRef):
                return item
            dtype = item.storage.dtype.newbyteorder("<" if byteorder == "little" else ">")
            cache_key = (item.storage.key, dtype.str)
            if cache_key not in storage_cache:
                raw = archive.read(prefix + "data/" + item.storage.key)
                storage_cache[cache_key] = np.frombuffer(raw, dtype=dtype, count=item.storage.size)
            storage = storage_cache[cache_key]
            if not item.shape:
                return np.asarray(storage[item.offset]).copy()
            strides = tuple(step * dtype.itemsize for step in item.stride)
            return np.asarray(np.lib.stride_tricks.as_strided(storage[item.offset:], shape=item.shape, strides=strides)).copy()

        resolved = resolve(value)
        state = resolved.get("model_state_dict", resolved) if isinstance(resolved, dict) else resolved
        if not isinstance(state, (dict, OrderedDict)):
            raise ValueError(f"checkpoint does not contain a tensor state dict: {path}")
        return OrderedDict(state)
