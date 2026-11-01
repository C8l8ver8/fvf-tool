#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
"""check_fvf.py -- look for anything unusual in FVF video files, and compare a re-encoded file with its original.

  python check_fvf.py scan    FILE_OR_FOLDER [...] [--csv out.csv] [--report out.txt] [--no-decode] [--deep]
  python check_fvf.py compare ORIGINAL.fvf NEW.fvf [--csv per_frame.csv] [--thr MAD]

scan     reads every file's headers, blocks, frame records and video opcodes, decodes it with decoder.py, and lists
         anything that is unusual (see the report).  It never changes any file.
         --no-decode  skip the pixel decode (fast, structure and opcode statistics only)
         --deep       also decode a second time with the picture buffers pre-filled with 0x00 instead of 0xAB; any frame
                      that comes out different reads video memory that nothing has written yet (the encoder assumes 0xAB)
compare  decodes both files and reports, per frame, how far NEW is from ORIGINAL (average error, brightness bias,
         worst pixel), the opcode mix of both, sizes, and whether the audio is identical.
"""
import argparse, collections, contextlib, csv, glob, hashlib, io, os, struct, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'fvflib'))
sys.path.insert(0, HERE)
import numpy as np
import decoder as dec

MAXBLK = 63488
FILL_DEFAULT = 0xAB


def u16(d, p): return d[p] | (d[p + 1] << 8)
def u32(d, p): return d[p] | (d[p + 1] << 8) | (d[p + 2] << 16) | (d[p + 3] << 24)


# ------------------------------------------------------------------ container
def scan_container(data):
    """Parse the container of one file. Return a dict with header info, the block list, the frame list and warnings."""
    r = {'size': len(data), 'warn': [], 'info': []}
    try:
        f = dec.FVFFile.__new__(dec.FVFFile)
        f.data = data
        with contextlib.redirect_stdout(io.StringIO()):
            f.parse_header()
        r['frame_count'] = f.frame_count; r['data_offset'] = f.data_offset
    except Exception as e:
        r['header_error'] = '%s: %s' % (type(e).__name__, e)
        r['warn'].append('header differs from what the decoder assumes: ' + r['header_error'])
        return r
    pos = f.data_offset; blocks = []; frames = []
    prev_size = 0
    try:
        while pos < len(data):
            hs, fc, prv, cur, nxt = struct.unpack_from('<HHIII', data, pos)
            b = {'pos': pos, 'hs': hs, 'fc': fc, 'prev': prv, 'size': cur, 'next': nxt, 'used': 16}
            if cur < 16: r['warn'].append('block at %d: invalid size %d' % (pos, cur)); break
            if hs != 0x10: r['warn'].append('block at %d: header size %#x (expected 0x10)' % (pos, hs)); break
            if cur % 2048: r['warn'].append('block at %d: size %d is not a multiple of 2048' % (pos, cur))
            if cur > MAXBLK: r['warn'].append('block at %d: size %d exceeds %d' % (pos, cur, MAXBLK))
            if prv != prev_size:
                if not blocks: r['info'].append('first block has a non-zero prev-size field (%d); leftover value, harmless (also present in some original files)' % prv)
                else: r['warn'].append('block at %d: prev-size field %d != previous block size %d' % (pos, prv, prev_size))
            p = pos + 16
            for k in range(fc):
                hl, fl, unk, vo, po, ao, unk2, unk3 = struct.unpack_from('<HHHIIIIH', data, p)
                fr = {'block': len(blocks), 'pos': p, 'hl': hl, 'fl': fl, 'unk': unk, 'vo': vo, 'po': po, 'ao': ao,
                      'unk2': unk2, 'unk3': unk3, 'video': None, 'vlen': 0, 'palette': None, 'audio': None}
                sofar = hl
                if vo:
                    if vo != sofar: fr['layout'] = 'video offset %d' % vo
                    vl = u16(data, p + vo); fr['vlen'] = vl; fr['video'] = data[p + vo + 2:p + vo + 2 + vl]; sofar = vo + 2 + vl
                if po:
                    if po != sofar: fr['layout'] = 'palette offset %d (expected %d)' % (po, sofar)
                    fr['palette'] = data[p + po:p + po + 336]; sofar = po + 336
                if ao:
                    if ao != sofar: fr['layout'] = 'audio offset %d (expected %d)' % (ao, sofar)
                    al = u32(data, p + ao); fr['audio'] = data[p + ao + 4:p + ao + 4 + al]; sofar = ao + 4 + al
                if sofar != fl: fr['layout'] = 'record length %d but fields add up to %d' % (fl, sofar)
                frames.append(fr); b['used'] += fl; p += fl
            b['slack'] = cur - b['used']
            if b['slack'] < 0: r['warn'].append('block at %d: frames overflow the block by %d bytes' % (pos, -b['slack']))
            blocks.append(b); prev_size = cur; pos += cur
    except Exception as e:
        r['warn'].append('container walk stopped: %s: %s' % (type(e).__name__, e))
    for i in range(len(blocks) - 1):
        if blocks[i]['next'] != blocks[i + 1]['size']:
            r['warn'].append('block %d: next-size field %d != following block size %d' % (i, blocks[i]['next'], blocks[i + 1]['size']))
    if blocks and blocks[-1]['next'] != 0: r['warn'].append('last block has a non-zero next-size field (%d)' % blocks[-1]['next'])
    if pos != len(data): r['warn'].append('%d bytes left after the last block' % (len(data) - pos))
    if len(frames) != r['frame_count']: r['warn'].append('header says %d frames, blocks contain %d' % (r['frame_count'], len(frames)))
    lay = [i for i, fr in enumerate(frames) if 'layout' in fr]
    if lay: r['warn'].append('%d frame record(s) with an unusual layout, first: frame %d (%s)' % (len(lay), lay[0], frames[lay[0]]['layout']))
    if blocks:
        b0 = blocks[0]; v0 = frames[0]['vlen'] if frames else 0
        if b0['fc'] != 3 or b0['size'] > 61440 or v0 > 38010:
            r['info'].append('first block holds %d frame(s) in %d bytes with a %d-byte frame 0. The original files have 3 frames in the first block (at most 59,392 bytes); this tool allows up to 61,440 bytes' % (b0['fc'], b0['size'], v0))
        lo = u32(data, 20)
        if lo != blocks[-1]['pos']:
            r['warn'].append('header field at byte 20 is %d but the last block starts at %d. They are equal in every original file; if they differ the game may fail to load the file' % (lo, blocks[-1]['pos']))
    r['blocks'] = blocks; r['frames'] = frames
    return r


# ------------------------------------------------------------------ video opcode scanner
CLASSES = ['hdr', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6'] + ['c7_%d' % i for i in range(8)] + ['c7rec', 'c7_12', 'c7_15']


def scan_video(video, counts):
    """Walk one video chunk exactly as decoder.py does, without touching pixels.
    counts[name] += number of opcodes, counts['bytes:'+name] += bytes, counts['blk:'+name] += blocks covered.
    Return (leftover_bytes, overshoot_blocks), or raise ValueError."""
    f = bytes(video) + b'\0\0\0\0'; flen = len(video)
    def add(name, nbytes, blocks=0):
        counts[name] += 1; counts['bytes:' + name] += nbytes; counts['blk:' + name] += blocks
    nstore = u16(f, 4)
    counts['bytes:hdr'] += 6 + 2 * nstore
    if nstore: counts['storage_slots_max'] = max(counts['storage_slots_max'], nstore)
    pos = 6 + 2 * nstore; cid = 0; limit = 4000; pass1 = True
    while pass1 or cid < limit:
        if pos >= flen: raise ValueError('ran past the end of the video data at block %d' % cid)
        if cid >= limit:
            if not (pass1 and cid == limit): raise ValueError('block count overrun in the luma pass (%d)' % cid)
            limit = 6000; pass1 = False
        b0 = f[pos]; ct = b0 & 7
        if ct <= 3:
            add('c%d' % ct, 3, 1); pos += 3; cid += 1
        elif ct == 4:
            n = ((b0 >> 3) & 0x1F) + 1; add('c4', 1, n); pos += 1; cid += n
        elif ct == 5:
            if (b0 >> 3) & 1: n = f[pos + 1] + 2; add('c5', 2, n); counts['c5_long'] += 1; pos += 2
            else: n = 1; add('c5', 1, 1); pos += 1
            cid += n
        elif ct == 6:
            add('c6', 17, 1); pos += 17; cid += 1
        else:
            sub = (b0 & 0x78) >> 3
            if sub == 12:
                info = (u32(f, pos) & 0xFFFFFF) >> 8
                if b0 & 0x80: n = info + 0x120 + 1; nb = 3
                else: n = (info & 0xFF) + 0x20 + 1; nb = 2
                add('c7_12', nb, n); pos += nb; cid += n
            elif sub <= 7:
                rec = f[pos + 1] & 0x80; pos += 4
                if not rec:
                    add('c7_%d' % sub, 4, 4)
                else:
                    s2 = f[pos] & 7
                    nb = {0: 3, 1: 3, 2: 3, 3: 3, 4: 1, 5: 1, 6: 17}.get(s2)
                    if nb is None: raise ValueError('unhandled recursive sub-opcode 7')
                    add('c7rec', 4 + nb, 4); counts['c7rec_sub%d' % s2] += 1; pos += nb
                cid += 4
            elif sub == 15:
                add('c7_15', 7, 0); pos += 7
            else:
                raise ValueError('unknown opcode 7/%d' % sub)
    return flen - pos, max(0, cid - 6000)


def new_counts():
    return collections.defaultdict(int)


# ------------------------------------------------------------------ decoding
def decode_all(frames, fill=FILL_DEFAULT, keep=False, want_md5=True, progress=None, keep_idx=()):
    """Decode a file. Return (list of per-frame md5 hex digests or None, error string or None [, list of RGB frames])."""
    st = dec.DecoderState()
    if fill != FILL_DEFAULT:
        for buf in (st.real_front_buffer, st.real_back_buffer, st.real_colour_front_buffer, st.real_colour_back_buffer):
            buf[:] = bytes([fill]) * len(buf)
    out = []; imgs = []; last = None
    for i, fr in enumerate(frames):
        try:
            if fr['video'] is None:
                rgb = None
            else:
                rgb = dec.decode_video_frame(st, fr['video'], fr['vlen'])
        except Exception as e:
            return out, 'frame %d: %s: %s' % (i, type(e).__name__, e), imgs
        if rgb is not None: last = bytes(rgb)
        out.append(hashlib.md5(last).hexdigest() if (want_md5 and last is not None) else None)
        if keep: imgs.append(last)
        elif i in keep_idx: imgs.append((i, last))
        if progress and i % 400 == 0: progress(i, len(frames))
    return out, None, imgs


# ------------------------------------------------------------------ scan
def scan_file(path, decode=True, deep=False, quiet=False):
    t0 = time.time(); data = open(path, 'rb').read()
    res = scan_container(data); res['path'] = path; res['name'] = os.path.basename(path)
    res['opc'] = new_counts()
    if 'frames' not in res:
        return res
    frames = res['frames']; warn = res['warn']; info = res['info']
    # per-frame fields
    vl = [fr['vlen'] for fr in frames if fr['video'] is not None]
    res['vmin'], res['vmax'], res['vavg'] = (min(vl), max(vl), sum(vl) / len(vl)) if vl else (0, 0, 0)
    res['vodd'] = sum(1 for x in vl if x & 1)
    nov = [i for i, fr in enumerate(frames) if fr['video'] is None]
    if nov: warn.append('%d frame(s) without video, first at %d' % (len(nov), nov[0]))
    noa = [i for i, fr in enumerate(frames) if fr['audio'] is None]
    if noa: info.append('%d frame(s) without audio (first at %d, last at %d)' % (len(noa), noa[0], noa[-1]))
    al = collections.Counter(len(fr['audio']) for fr in frames[1:] if fr['audio'] is not None)
    res['audio_sizes'] = dict(al)
    if len(al) > 1: info.append('audio chunk sizes after frame 0: ' + ', '.join('%d B x%d' % kv for kv in sorted(al.items())))
    res['audio0'] = len(frames[0]['audio']) if frames and frames[0]['audio'] is not None else 0
    pals = [(i, hashlib.md5(fr['palette']).hexdigest()) for i, fr in enumerate(frames) if fr['palette'] is not None]
    res['palette_frames'] = len(pals); res['palette_distinct'] = len(set(h for _, h in pals))
    if pals and not (len(pals) == 1 and pals[0][0] == 0):
        info.append('palette chunks in %d frame(s), %d distinct (first frames: %s)' % (len(pals), res['palette_distinct'], ', '.join(str(i) for i, _ in pals[:6])))
    res['fh'] = collections.Counter((fr['unk'], fr['unk2'], fr['unk3']) for fr in frames)
    if len(res['fh']) > 1 or (0, 0, 0) not in res['fh']:
        info.append('frame header "unknown" fields (u16,u32,u16): ' + ', '.join('%s x%d' % (k, v) for k, v in res['fh'].most_common(5)))
    res['flags'] = collections.Counter(u16(fr['video'], 2) for fr in frames if fr['video'] is not None and len(fr['video']) >= 4)
    info.append('frame flags: ' + ', '.join('%#x x%d' % kv for kv in sorted(res['flags'].items())))
    slacks = [b['slack'] for b in res['blocks']]
    res['slack'] = (min(slacks), sum(slacks) / len(slacks), max(slacks)) if slacks else (0, 0, 0)
    res['block_frames'] = collections.Counter(b['fc'] for b in res['blocks'])
    # opcodes
    c = res['opc']; bad = []; trail = 0; over = 0; pad = collections.Counter()
    for i, fr in enumerate(frames):
        if fr['video'] is None: continue
        try:
            lo, ov = scan_video(fr['video'], c); trail += 1 if lo else 0; over += 1 if ov else 0; pad[lo] += 1
        except Exception as e:
            bad.append((i, str(e)))
    res['opc_errors'] = bad
    if bad: warn.append('%d frame(s) whose video the opcode scanner could not walk, first: frame %d (%s)' % (len(bad), bad[0][0], bad[0][1]))
    res['pad'] = dict(pad)
    if trail: info.append('unused bytes after the last opcode (bytes: frames): ' + ', '.join('%d: %d' % kv for kv in sorted(pad.items())))
    if over: warn.append('%d frame(s) cover more than 6000 blocks' % over)
    if c['c7rec']: info.append('uses recursive 8x8 blocks (7 with refinement) x%d' % c['c7rec'])
    if c['c7_15']: info.append('uses raw 2x2 writes (7/15) x%d' % c['c7_15'])
    if res['flags'].get(0x8) or any(f & 8 for f in res['flags']): info.append('contains skip frames (flag 0x8)')
    if any(f & 1 for f in res['flags']): info.append('uses the 16x repeat flag (0x1) in %d frame(s)' % sum(v for f, v in res['flags'].items() if f & 1))
    # decode
    res['decoded'] = None; res['decode_error'] = None; res['fill_dependent'] = None
    if decode:
        md5a, err, _ = decode_all(frames)
        res['decode_error'] = err; res['decoded'] = err is None
        if err: warn.append('decoding fails at ' + err)
        else:
            skip = sum(1 for fr in frames if fr['video'] is not None and len(fr['video']) >= 4 and u16(fr['video'], 2) & 8)
            if skip: info.append('%d skip frame(s) decoded' % skip)
            if deep:
                md5b, err2, _ = decode_all(frames, fill=0x00)
                diff = [i for i, (x, y) in enumerate(zip(md5a, md5b)) if x != y] if err2 is None else None
                if err2: warn.append('decoding with a different initial fill fails at ' + err2)
                else:
                    res['fill_dependent'] = diff
                    if not diff: info.append('output does not depend on the initial buffer fill')
                    else:
                        want = {diff[0], diff[-1]}
                        _, _, ia = decode_all(frames, keep_idx=want); _, _, ib = decode_all(frames, fill=0x00, keep_idx=want)
                        pc = {}
                        for (i, x), (_, y) in zip(ia, ib):
                            d = np.abs(np.frombuffer(x, np.uint8).astype(np.int16) - np.frombuffer(y, np.uint8).astype(np.int16))
                            pc[i] = (float(d.mean()), int(d.max()))
                        res['fill_pct'] = pc
                        f0, f1 = diff[0], diff[-1]
                        msg = ('output depends on the initial buffer fill in %d of %d frame(s); frame %d differs by %.2f on average (largest %d), '
                               'frame %d by %.2f (largest %d)' % (len(diff), len(frames), f0, pc[f0][0], pc[f0][1], f1, pc[f1][0], pc[f1][1]))
                        # a clip whose first frame iterates (flag 0x1) converges from any starting picture, so small residuals are normal
                        bad_end = f1 == len(frames) - 1 and pc[f1][1] > 16
                        (warn if bad_end else info).append(msg + (' - clearly visible at the end of the clip' if bad_end else ''))
    res['secs'] = time.time() - t0
    return res


def pct(res, names, what='bytes:'):
    tot = sum(v for k, v in res['opc'].items() if k.startswith('bytes:'))
    return 100.0 * sum(res['opc'].get(what + n, 0) for n in names) / tot if tot else 0.0


def opcode_table(res):
    c = res['opc']; tot = sum(v for k, v in c.items() if k.startswith('bytes:')) or 1
    rows = []
    for n in CLASSES:
        if c.get(n, 0) or c.get('bytes:' + n, 0):
            rows.append((n, c[n], c['bytes:' + n], 100.0 * c['bytes:' + n] / tot, c['blk:' + n]))
    return rows


NAMES = {'hdr': 'frame header + MV table', 'c0': 'affine 0', 'c1': 'affine 1', 'c2': 'affine 2', 'c3': 'affine 3', 'c4': 'copy run (skip)',
         'c5': 'motion-vector copy', 'c6': 'raw 4x4', 'c7rec': '8x8 + refinement', 'c7_12': 'long skip run', 'c7_15': 'raw 2x2'}
for _i in range(8): NAMES['c7_%d' % _i] = '8x8 transform %d' % _i


def fmt_report(res):
    L = []
    L.append('== %s  (%s bytes)' % (res['name'], format(res['size'], ',')))
    if 'frames' not in res:
        L.append('   ' + '\n   '.join(res['warn'])); return '\n'.join(L)
    L.append('   %d frames, %d blocks (frames/block: %s), block slack min/avg/max %d / %d / %d'
             % (len(res['frames']), len(res['blocks']), ', '.join('%d x%d' % kv for kv in sorted(res['block_frames'].items())), *[int(x) for x in res['slack']]))
    L.append('   video chunk bytes min/avg/max %d / %d / %d, odd lengths %d; first-frame audio %d B; palettes %d (%d distinct)'
             % (res['vmin'], res['vavg'], res['vmax'], res['vodd'], res['audio0'], res['palette_frames'], res['palette_distinct']))
    L.append('   opcode mix (share of video bytes):')
    for n, cnt, nb, share, blk in opcode_table(res):
        L.append('      %-8s %-24s %9d ops %11d B %5.1f %%  covers %d blocks' % (n, NAMES.get(n, n), cnt, nb, share, blk))
    for w in res['warn']: L.append('   WARNING: ' + w)
    for w in res['info']: L.append('   note: ' + w)
    if res.get('decoded'): L.append('   decodes without errors')
    return '\n'.join(L)


def cmd_scan(a):
    paths = []
    for p in a.paths:
        if os.path.isdir(p): paths += sorted(glob.glob(os.path.join(p, '*.fvf')) + glob.glob(os.path.join(p, '*.FVF')))
        else: paths += sorted(glob.glob(p)) or [p]
    seen = set(); paths = [p for p in paths if not (p.lower() in seen or seen.add(p.lower()))]
    if not paths: raise SystemExit('no .fvf files found')
    results = []; t0 = time.time()
    for k, p in enumerate(paths):
        print('[%d/%d] %s ...' % (k + 1, len(paths), os.path.basename(p)), flush=True)
        try: res = scan_file(p, decode=not a.no_decode, deep=a.deep)
        except Exception as e:
            res = {'path': p, 'name': os.path.basename(p), 'size': os.path.getsize(p), 'warn': ['scan failed: %s: %s' % (type(e).__name__, e)], 'info': [], 'opc': new_counts()}
        results.append(res)
    # summary table
    print()
    hdr = '%-14s %11s %6s %6s %7s %7s %7s %5s %5s %5s %5s' % ('file', 'bytes', 'frames', 'blocks', 'vid-max', 'raw%', '8x8%', 'mv%', 'pal', 'warn', 'dec')
    print(hdr); print('-' * len(hdr))
    for r in results:
        if 'frames' not in r:
            print('%-14s %11s  %s' % (r['name'], format(r['size'], ','), r['warn'][0])); continue
        c8 = pct(r, ['c7_%d' % i for i in range(8)] + ['c7rec'])
        print('%-14s %11s %6d %6d %7d %6.1f%% %6.1f%% %4.1f%% %5d %5d %5s' % (
            r['name'], format(r['size'], ','), len(r['frames']), len(r['blocks']), r['vmax'], pct(r, ['c6']), c8, pct(r, ['c5']), r['palette_frames'], len(r['warn']),
            {True: 'ok', False: 'FAIL', None: '-'}[r.get('decoded')]))
    tw = sum(len(r['warn']) for r in results)
    print('\n%d file(s) scanned in %.0f s, %d warning(s) in total.' % (len(results), time.time() - t0, tw))
    if tw:
        print('\nWarnings:')
        for r in results:
            for w in r['warn']: print('  %s: %s' % (r['name'], w))
    if a.report:
        with open(a.report, 'w', encoding='utf-8') as f:
            f.write('FVF scan report (%s)\n\n' % time.strftime('%Y-%m-%d %H:%M'))
            for r in results: f.write(fmt_report(r) + '\n\n')
        print('detailed report ->', a.report)
    if a.csv:
        with open(a.csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['file', 'bytes', 'frames', 'blocks', 'video_min', 'video_avg', 'video_max', 'video_odd', 'audio0', 'palette_frames', 'palette_distinct',
                        'decoded', 'warnings'] + ['bytes:' + n for n in CLASSES] + ['ops:' + n for n in CLASSES])
            for r in results:
                if 'frames' not in r: w.writerow([r['name'], r['size']] + [''] * 10 + ['; '.join(r['warn'])]); continue
                w.writerow([r['name'], r['size'], len(r['frames']), len(r['blocks']), r['vmin'], round(r['vavg'], 1), r['vmax'], r['vodd'], r['audio0'],
                            r['palette_frames'], r['palette_distinct'], r.get('decoded'), ' | '.join(r['warn'])] +
                           [r['opc'].get('bytes:' + n, 0) for n in CLASSES] + [r['opc'].get(n, 0) for n in CLASSES])
        print('table ->', a.csv)
    return results


# ------------------------------------------------------------------ compare
def cmd_compare(a):
    A = scan_container(open(a.original, 'rb').read()); B = scan_container(open(a.new, 'rb').read())
    for tag, X in (('ORIGINAL', A), ('NEW', B)):
        if 'frames' not in X: raise SystemExit('%s: %s' % (tag, '; '.join(X['warn'])))
    print('ORIGINAL %s: %s bytes, %d frames, %d blocks' % (os.path.basename(a.original), format(A['size'], ','), len(A['frames']), len(A['blocks'])))
    print('NEW      %s: %s bytes, %d frames, %d blocks' % (os.path.basename(a.new), format(B['size'], ','), len(B['frames']), len(B['blocks'])))
    ok = True
    def chk(cond, good, bad):
        nonlocal ok; ok &= bool(cond); print('  %s %s' % ('OK  ' if cond else 'FAIL', good if cond else bad))
    print('\nStructure')
    chk(A['size'] == B['size'], 'same file size (required by the game)', 'file sizes differ: %d vs %d - the game will not accept it' % (A['size'], B['size']))
    chk(len(A['frames']) == len(B['frames']), 'same number of frames', 'frame counts differ')
    n = min(len(A['frames']), len(B['frames']))
    aud = all((A['frames'][i]['audio'] or b'') == (B['frames'][i]['audio'] or b'') for i in range(n))
    chk(aud, 'audio identical in every frame', 'audio differs')
    ha = bytearray(open(a.original, 'rb').read(2048)); hb = bytearray(open(a.new, 'rb').read(2048))
    ha[20:24] = hb[20:24] = b'\0\0\0\0'   # bytes 20-23 hold the offset of the last block, which depends on the block layout
    chk(ha == hb, 'file header identical (apart from the last-block offset, bytes 20-23)', 'file header differs')
    chk(B['blocks'] and u32(open(a.new, 'rb').read(2048), 20) == B['blocks'][-1]['pos'], 'header last-block offset matches the new file', 'header last-block offset (bytes 20-23) does not match the last block; the game may fail to load the file')
    mxB = max(b['size'] for b in B['blocks'])
    chk(mxB <= MAXBLK and all(b['size'] % 2048 == 0 for b in B['blocks']), 'all blocks are multiples of 2048 and at most %d' % MAXBLK, 'block size problem')
    for w in B['warn']: print('  WARNING (new file): ' + w)
    # opcodes
    for X in (A, B):
        X['opc'] = new_counts()
        for fr in X['frames']:
            if fr['video'] is not None:
                try: scan_video(fr['video'], X['opc'])
                except Exception as e: X['warn'].append(str(e))
    print('\nVideo bytes by opcode class (share of all video bytes)')
    print('  %-8s %-24s %12s %6s   %12s %6s' % ('class', '', 'original', '', 'new', ''))
    ta = sum(v for k, v in A['opc'].items() if k.startswith('bytes:')) or 1; tb = sum(v for k, v in B['opc'].items() if k.startswith('bytes:')) or 1
    for nme in CLASSES:
        x, y = A['opc'].get('bytes:' + nme, 0), B['opc'].get('bytes:' + nme, 0)
        if x or y: print('  %-8s %-24s %12d %5.1f%%   %12d %5.1f%%' % (nme, NAMES.get(nme, nme), x, 100.0 * x / ta, y, 100.0 * y / tb))
    print('  %-8s %-24s %12d %6s   %12d' % ('total', '', ta, '', tb))
    va = [fr['vlen'] for fr in A['frames']]; vb = [fr['vlen'] for fr in B['frames']]
    print('\nVideo chunk size: original avg %d max %d, new avg %d max %d' % (sum(va) / len(va), max(va), sum(vb) / len(vb), max(vb)))
    # pixels
    print('\nDecoding both files ...', flush=True)
    _, ea, ia = decode_all(A['frames'][:n], keep=True, want_md5=False); _, eb, ib = decode_all(B['frames'][:n], keep=True, want_md5=False)
    if ea or eb: raise SystemExit('decode failed: %s %s' % (ea, eb))
    rows = []
    for i in range(n):
        x = np.frombuffer(ia[i], np.uint8).astype(np.int16).reshape(-1, 3); y = np.frombuffer(ib[i], np.uint8).astype(np.int16).reshape(-1, 3)
        d = y - x; ad = np.abs(d)
        rows.append((i, float(ad.mean()), *[float(v) for v in d.mean(axis=0)], int(ad.max()), 100.0 * float((ad.max(axis=1) > 8).mean()), va[i], vb[i]))
    mad = np.array([r[1] for r in rows])
    thr = a.thr if a.thr is not None else max(1.0, float(np.percentile(mad, 99)))
    print('\nPicture error (new vs original): mean MAD %.3f, median %.3f, 99th percentile %.3f, worst %.3f' % (mad.mean(), np.median(mad), np.percentile(mad, 99), mad.max()))
    gb = np.array([r[3] for r in rows])
    print('Brightness bias (new - original, green): mean %.3f, worst %.3f at frame %d' % (gb.mean(), gb[np.abs(gb).argmax()], int(np.abs(gb).argmax())))
    print('\nWorst 15 frames by MAD:')
    print('  %6s %7s %7s %8s %8s %6s %7s %7s %7s' % ('frame', 'time', 'MAD', 'R bias', 'G bias', 'max', '>8 %', 'orig B', 'new B'))
    for r in sorted(rows, key=lambda r: -r[1])[:15]:
        print('  %6d %6.2fs %7.3f %8.3f %8.3f %6d %7.2f %7d %7d' % (r[0], r[0] / 15, r[1], r[2], r[3], r[5], r[6], r[7], r[8]))
    flagged = [r[0] for r in rows if r[1] >= thr]
    print('\n%d frame(s) at or above MAD %.2f: %s%s' % (len(flagged), thr, ', '.join(map(str, flagged[:40])), ' ...' if len(flagged) > 40 else ''))
    if a.csv:
        with open(a.csv, 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['frame', 'mad', 'bias_r', 'bias_g', 'bias_b', 'max_err', 'pct_over_8', 'orig_video_bytes', 'new_video_bytes'])
            for r in rows: w.writerow([r[0]] + [round(v, 4) if isinstance(v, float) else v for v in r[1:]])
        print('per-frame table ->', a.csv)
    print('\nResult: %s' % ('structure OK' if ok else 'STRUCTURE PROBLEMS FOUND (see FAIL lines above)'))
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest='cmd', required=True)
    x = s.add_parser('scan'); x.add_argument('paths', nargs='+'); x.add_argument('--csv'); x.add_argument('--report')
    x.add_argument('--no-decode', action='store_true'); x.add_argument('--deep', action='store_true'); x.set_defaults(fn=cmd_scan)
    x = s.add_parser('compare'); x.add_argument('original'); x.add_argument('new'); x.add_argument('--csv'); x.add_argument('--thr', type=float); x.set_defaults(fn=cmd_compare)
    a = p.parse_args(argv); a.fn(a)


if __name__ == '__main__':
    main()
