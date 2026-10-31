# SPDX-License-Identifier: CC0-1.0
"""Replace STORED entries inside a zip IN PLACE (same size), keeping every other byte,
the entry order and all offsets identical. Updates CRC32 in local + central headers.
usage: python patch_zip.py ORIGINAL.ZIP OUTPUT.ZIP name1=file1 [name2=file2 ...]"""
import sys, struct, shutil, zlib, os


def patch(src, dst, repl):
    if os.path.abspath(src) == os.path.abspath(dst):
        raise SystemExit('the output zip must be a different file from the original')
    shutil.copyfile(src, dst)
    try:
        _patch_in_place(src, dst, repl)
    except BaseException:
        if os.path.exists(dst): os.remove(dst)      # never leave a partly patched zip behind
        raise


def _patch_in_place(src, dst, repl):
    with open(dst, 'r+b') as f:
        f.seek(0, 2); size = f.tell()
        tail = min(size, 70000); f.seek(size - tail); buf = f.read()
        i = buf.rfind(b'PK\x05\x06')
        if i < 0: raise SystemExit('not a zip file (no end-of-central-directory record)')
        _, _, _, _, nent, cdsize, cdoff, _ = struct.unpack('<IHHHHIIH', buf[i:i+22])
        if nent == 0xFFFF or cdoff == 0xFFFFFFFF: raise SystemExit('zip64 archive: not supported by this tool')
        f.seek(cdoff); cd = f.read(cdsize)
        p = 0; done = set()
        for _ in range(nent):
            sig, = struct.unpack('<I', cd[p:p+4])
            if sig != 0x02014b50: raise SystemExit('damaged zip: bad central directory entry')
            (method, ) = struct.unpack('<H', cd[p+10:p+12])
            crc, csz, usz, nl, el, cl = struct.unpack('<IIIHHH', cd[p+16:p+34])
            lho, = struct.unpack('<I', cd[p+42:p+46])
            name = cd[p+46:p+46+nl].decode('cp437')
            if name in repl:
                with open(repl[name], 'rb') as g: data = g.read()
                if method != 0: raise SystemExit(name + ' is not stored')
                if not len(data) == usz == csz: raise SystemExit('%s: size %d != archive entry %d' % (name, len(data), usz))
                f.seek(lho); lh = f.read(30)
                if lh[:4] != b'PK\x03\x04': raise SystemExit('damaged zip: bad local header for ' + name)
                lnl, lel = struct.unpack('<HH', lh[26:30])
                flags, = struct.unpack('<H', lh[6:8])
                if flags & 8: raise SystemExit('data descriptor present: not handled')
                newcrc = zlib.crc32(data) & 0xFFFFFFFF
                f.seek(lho + 30 + lnl + lel); f.write(data)                  # data in place
                f.seek(lho + 14); f.write(struct.pack('<I', newcrc))         # local CRC
                f.seek(cdoff + p + 16); f.write(struct.pack('<I', newcrc))   # central CRC
                print('patched %-12s crc %08X (was %08X) at offset %d' % (name, newcrc, crc, lho + 30 + lnl + lel))
                done.add(name)
            p += 46 + nl + el + cl
        if done != set(repl): raise SystemExit('not found: %s' % (set(repl) - done))
    print('done ->', dst, os.path.getsize(dst), 'bytes (source', os.path.getsize(src), ')')


if __name__ == '__main__':
    patch(sys.argv[1], sys.argv[2], dict(a.split('=', 1) for a in sys.argv[3:]))
