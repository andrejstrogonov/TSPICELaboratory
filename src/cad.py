"""Import triangulated CAD geometry exported as STL."""

from __future__ import annotations

import re
import struct
from pathlib import Path

import numpy as np

MAX_STL_TRIANGLES = 2_000_000
_BINARY_TRIANGLE_DTYPE = np.dtype(
    [
        ("normal", "<f4", (3,)),
        ("vertices", "<f4", (3, 3)),
        ("attribute", "<u2"),
    ]
)
_VERTEX_PATTERN = re.compile(
    rb"\bvertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)",
    re.IGNORECASE,
)


def load_stl(path: str | Path, *, unit_to_nm: float = 1_000_000.0):
    """Return STL vertices in nanometres and triangle indices."""
    if not np.isfinite(unit_to_nm) or unit_to_nm <= 0:
        raise ValueError("STL unit scale must be positive and finite.")
    path = Path(path)
    if path.stat().st_size > 256 * 1024 * 1024:
        raise ValueError("STL file exceeds the 256 MiB import limit.")
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError("STL file is truncated or does not contain any triangles.")

    triangle_count = struct.unpack_from("<I", data, 80)[0]
    expected_size = 84 + triangle_count * _BINARY_TRIANGLE_DTYPE.itemsize
    if expected_size == len(data):
        if triangle_count > MAX_STL_TRIANGLES:
            raise ValueError(
                f"STL contains more than {MAX_STL_TRIANGLES:,} triangles."
            )
        triangles = np.frombuffer(
            data,
            dtype=_BINARY_TRIANGLE_DTYPE,
            count=triangle_count,
            offset=84,
        )["vertices"].astype(np.float64, copy=True)
    else:
        vertex_count = sum(1 for _ in _VERTEX_PATTERN.finditer(data))
        if not vertex_count or vertex_count % 3:
            raise ValueError("STL file has an invalid or empty triangle list.")
        if vertex_count // 3 > MAX_STL_TRIANGLES:
            raise ValueError(
                f"STL contains more than {MAX_STL_TRIANGLES:,} triangles."
            )
        coordinates = np.fromiter(
            (
                float(component)
                for match in _VERTEX_PATTERN.finditer(data)
                for component in match.groups()
            ),
            dtype=np.float64,
            count=vertex_count * 3,
        )
        triangles = coordinates.reshape(-1, 3, 3)

    if not len(triangles):
        raise ValueError("STL file has no triangles.")
    if not np.isfinite(triangles).all():
        raise ValueError("STL geometry contains non-finite coordinates.")
    vertices = triangles.reshape(-1, 3)
    vertices *= unit_to_nm
    if np.any(np.ptp(vertices, axis=0) <= 0):
        raise ValueError("STL geometry must have non-zero extent along all axes.")
    faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    return vertices, faces
