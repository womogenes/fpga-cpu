#!/usr/bin/env python3

import time

import numpy as np


def timed(label, operation, work):
    operation()
    start = time.monotonic()
    result = operation()
    elapsed = time.monotonic() - start
    print(f"{label}: {elapsed:.3f} s, {work / elapsed / 1e6:.2f} Mops/s, checksum={float(np.sum(result)):.6g}")


def main():
    print(f"NumPy {np.__version__}")
    print("BLAS configuration:")
    np.show_config()
    length = 262_144
    left = np.linspace(-1.0, 1.0, length, dtype=np.float32)
    right = np.linspace(1.0, -1.0, length, dtype=np.float32)
    timed("float32 elementwise multiply-add", lambda: left * right + left, length * 2)
    side = 128
    matrix = np.arange(side * side, dtype=np.float32).reshape(side, side) / side
    vector = np.linspace(-1.0, 1.0, side, dtype=np.float32)
    timed("OpenBLAS float32 matrix-vector", lambda: matrix @ vector, side * side * 2)


if __name__ == "__main__":
    main()
