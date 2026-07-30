"""ctypes bridge for the Mojo calendar kernels."""

from __future__ import annotations

import ctypes
import operator
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get(
    "MOJO_PENDULUM_LIB", os.path.join(ROOT, "dist", "libmojo-pendulum.so")
)
I = ctypes.c_int64

_SIGNATURES = {
    "mp_parse_iso": ([ctypes.c_char_p, I, I], I),
    "mp_parse_iso_many": ([I] * 6, I),
    "mp_add_datetime": ([I] * 10, I),
    "mp_add_datetime_many": ([I] * 11, I),
    "mp_format_iso_many": ([I] * 5, I),
}

_library: ctypes.CDLL | None = None
_I64_MIN = -(1 << 63)
_I64_MAX = (1 << 63) - 1


class BuildError(RuntimeError):
    pass


def i64(value, name: str = "value") -> int:
    if type(value) is int:
        if not _I64_MIN <= value <= _I64_MAX:
            raise OverflowError(f"{name} does not fit in a signed 64-bit integer")
        return value
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if not _I64_MIN <= result <= _I64_MAX:
        raise OverflowError(f"{name} does not fit in a signed 64-bit integer")
    return result


def build(force: bool = False) -> str:
    source = os.path.join(ROOT, "src", "pendulum.mojo")
    if (
        not force
        and os.path.exists(LIB)
        and os.path.getmtime(LIB) >= os.path.getmtime(source)
    ):
        return LIB
    if os.environ.get("MOJO_PENDULUM_LIB"):
        if os.path.exists(LIB):
            return LIB
        raise BuildError(f"MOJO_PENDULUM_LIB does not exist: {LIB}")
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def addr(array: np.ndarray, dtype: np.dtype, *, ndim: int | None = None) -> int:
    """Return an address only for the exact native buffer layout expected by Mojo."""
    if not isinstance(array, np.ndarray):
        raise TypeError("native buffers must be numpy arrays")
    if array.dtype != np.dtype(dtype):
        raise TypeError(f"native buffer must have dtype {np.dtype(dtype)}")
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"native buffer must have {ndim} dimensions")
    if not array.flags.c_contiguous:
        raise ValueError("native buffers must be C-contiguous")
    if array.size == 0 or array.ctypes.data == 0:
        raise ValueError("native buffers must be non-empty")
    if not array.flags.aligned:
        raise ValueError("native buffers must be aligned")
    return i64(array.ctypes.data, "buffer address")
