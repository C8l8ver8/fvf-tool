# SPDX-License-Identifier: CC0-1.0
"""recontainer_lite.py -- builds one FVF frame record."""
import struct


def rec(vb, ab, pal=None):
    """One frame record: 24-byte header, u16 video length, video [+336B palette] [+u32 audio len + audio]."""
    n = 24 + 2 + len(vb); po = 0; ao = 0; body = struct.pack('<H', len(vb)) + vb
    if pal is not None: po = n; body += pal; n += len(pal)
    if ab is not None: ao = n; body += struct.pack('<I', len(ab)) + ab; n += 4 + len(ab)
    assert n <= 65535, n
    return struct.pack('<HHHIIIIH', 24, n, 0, 24, po, ao, 0, 0) + body
