#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
"""Reference hashes for test_js.js: decodes each file with fvflib/decoder.py and stores, per frame,
[video length, audio length, flags, md5 of the RGB picture, md5 of the audio] in ref/<name>.json.
usage: python viewer_src/mkref.py FILE.fvf [...]"""
import sys, os, json, hashlib
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, '..', 'fvflib'))
import decoder as dec
os.makedirs(os.path.join(here, 'ref'), exist_ok=True)
for p in sys.argv[1:]:
    f = dec.FVFFile(p); f.read_next_block(); st = dec.DecoderState(); out = []
    for i in range(f.frame_count):
        vf, vl, af = dec.decode_next_frame(f); rgb = dec.decode_video_frame(st, vf, vl)
        out.append([vl, len(af) if af is not None else None, int.from_bytes(vf[2:4], 'little') if vf else None,
                    hashlib.md5(bytes(rgb)).hexdigest() if rgb is not None else None, hashlib.md5(bytes(af)).hexdigest() if af else None])
    json.dump(out, open(os.path.join(here, 'ref', os.path.basename(p) + '.json'), 'w'))
    print(os.path.basename(p), len(out), 'frames')
