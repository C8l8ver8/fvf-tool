#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
"""fvf.py -- decode, re-encode and repack FVF video files (Total Multimedia / SoftVideo)
from "Star Trek: The Next Generation - A Final Unity".

  python fvf.py info    FILE.fvf
  python fvf.py decode  FILE.fvf OUTBASE [--png DIR] [--no-mp4] [--frames N]
  python fvf.py encode  SRC.fvf OUT.fvf [--mode s10|fast] [--scale S] [--work DIR] [--jobs N] [--redo] [rate-control options]
  python fvf.py patch-zip ORIGINAL.ZIP OUT.ZIP name=file [name=file ...]
  python fvf.py replace ORIGINAL.ZIP OUT.ZIP name=src.fvf [...] [--mode ...] [rate-control options]
  python fvf.py view                                   (opens the viewer, fvf_viewer.html, in your web browser)
  python fvf.py check   FILE_OR_FOLDER [...] [--csv F] [--report F] [--no-decode] [--deep]
  python fvf.py compare ORIGINAL.fvf NEW.fvf [--csv F] [--thr MAD]

`encode` takes an ORIGINAL .fvf as the source: it is decoded, re-encoded, and
padded to exactly the original file size, which the game requires. Every command
accepts -v/--verbose (after the command name) to print decoder header and block details.
"""
import argparse, math, os, struct, subprocess, sys, time, wave

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'fvflib'))
sys.path.insert(0, HERE)

try:
    import numpy as np
except ImportError:
    raise SystemExit('numpy is required: python -m pip install numpy')
import decoder as dec
import encoder7

W, H = 320, 200
SECT, MAXBLK = 2048, 63488
FIRST_BLOCK_MAX = 61440   # largest first block (bytes) the encoder writes; it holds frames 0-2, as in the original files
F0_TARGET = 18010          # frame-0 video budget: 6000 affine blocks (18,007 B), the size of frame 0 in the original files
F0_MAX = 38006            # upper limit for the frame-0 video chunk (the largest found in the original files)


def _clamp12(sched, cap):
    """Cap the budgets of frames 1 and 2, which share the first block with frame 0, so that all three fit in it."""
    return [min(v, cap) if i in (1, 2) else v for i, v in enumerate(sched)]


def _floor_sched(base, olen, floor, fcap, skip_below=0, skip_mad=None, mad_thr=0.0, protect=0.0):
    """Raise every frame budget to at least floor * (original frame size), capped at fcap. The other frames are rescaled by one
    common factor (found by bisection) so the total stays the same. Frames planned below skip_below bytes, or whose pass-1 error
    is below mad_thr, are not raised; the latter are kept at protect * their planned size."""
    if floor <= 0: return list(base)
    # Nearly static frames (planned below skip_below bytes) are not raised: the extra bytes would be wasted.
    fl = [0.0 if b < skip_below else min(floor * o, fcap) for b, o in zip(base, olen)]; tot = sum(base); lo, hi = 0.05, 2.0
    if skip_mad is not None:
        # Frames that pass 1 codes almost perfectly are not raised, but they must not be cut either: their error rises sharply
        # when the budget drops well below what pass 1 spent. Keep them at `protect` times their plan.
        fl = [protect * b if m < mad_thr else f for f, m, b in zip(fl, skip_mad, base)]
    for _ in range(60):
        m = (lo + hi) / 2
        if sum(max(b * m, f) for b, f in zip(base, fl)) > tot: hi = m
        else: lo = m
    out = [max(b * lo, f) for b, f in zip(base, fl)]
    if sum(out) > tot * 1.002:
        log(f'WARNING: the floors need {sum(out):.0f} B but the plan has {tot:.0f} B: lower --floor (the file may not fit)')
    return out




def _mad_tilt(base, mad, tilt, tmax, tmin=1.0):
    """Byte-neutral reweighting of the frame budgets: each frame's plan is multiplied by
    (its pass-1 picture error / the 75th-percentile error) ** tilt, clipped to [tmin, tmax], and the result is rescaled to the
    same total. Frames that pass 1 codes poorly therefore get more bytes. With tmin = 1.0 no frame is reduced by the weighting itself
    (the common rescale takes the bytes back evenly). Returns the plan unchanged when tilt <= 0 or no pass-1 error list exists."""
    if tilt <= 0 or mad is None: return list(base)
    ms = sorted(mad); ref = max(ms[int(0.75 * (len(ms) - 1))], 1e-3)
    w = [min(max((max(m, 1e-4) / ref) ** tilt, tmin), tmax) for m in mad]
    t = sum(w[i] * base[i] for i in range(len(base)))
    return [base[i] * w[i] * sum(base) / t for i in range(len(base))]


def _key_boost(sched, rho):
    """Byte-neutral: every key frame (i % 4 == 0, i >= 4) takes the fraction rho of the budgets of the three frames that follow it.
    Those frames predict from the key frame, and so does the next key frame, so errors in a key frame are inherited."""
    if rho <= 0: return list(sched)
    s = list(map(float, sched))
    for k in range(4, len(s), 4):
        lv = [j for j in (k + 1, k + 2, k + 3) if j < len(s)]
        mv = rho * sum(s[j] for j in lv)
        for j in lv: s[j] *= 1 - rho
        s[k] += mv
    return s


_LOGF = None


def log(*a):
    line = ' '.join([time.strftime('%H:%M:%S')] + [str(x) for x in a])
    print(line, flush=True)
    if _LOGF:
        _LOGF.write(line + '\n'); _LOGF.flush()


def hms(sec):
    sec = int(max(0, sec)); return f'{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}' if sec >= 3600 else f'{sec // 60}:{sec % 60:02d}'


# ---------------------------------------------------------------- reading
def read_records(path, n=None):
    """Return (video_bitstreams, audio_chunks), one entry per frame record."""
    f = dec.FVFFile(path); f.read_next_block()
    n = f.frame_count if n is None else min(n, f.frame_count)
    vids, auds = [], []
    for _ in range(n):
        vf, vl, af = dec.decode_next_frame(f)
        vids.append(bytes(vf)); auds.append(None if af is None else bytes(af))
    return vids, auds


def palette_of(path):
    d = open(path, 'rb').read()
    f = dec.FVFFile(path); f.read_next_block()
    fs = f.cur_pos; po = struct.unpack_from('<I', d, fs + 10)[0]
    pal = d[fs + po:fs + po + 336]
    assert len(pal) == 336
    return d[:2048], pal


def decode_planes(path, n):
    """Decode with decoder.py; return per-frame (luma_plane, colour_plane) bytes and the original video chunk sizes."""
    f = dec.FVFFile(path); f.read_next_block(); st = dec.DecoderState()
    frames, olen = [], []
    while len(frames) < n:
        vf, vl, af = dec.decode_next_frame(f)
        flags = dec.read_le_u16(vf, 2)
        if flags & 0x8:
            raise SystemExit('skip frames are not supported by the encoder')
        wf, wc = st.front_buffer, st.colour_front_buffer
        dec.decode_video_frame_data(st, vf, vl)
        if flags & 1:
            wf, wc = st.front_buffer, st.colour_front_buffer
            for _ in range(15):
                dec.decode_video_frame_data(st, vf, vl)
        frames.append((bytes(wf), bytes(wc))); olen.append(vl)
    return frames, olen


def expected_rgb(g, c):
    g = np.frombuffer(g, np.uint8).reshape(H, W)
    c = np.frombuffer(c, np.uint8).reshape(H // 2, W)
    r = np.repeat(np.repeat(c[:, :W // 2], 2, 0), 2, 1)
    b = np.repeat(np.repeat(c[:, W // 2:], 2, 0), 2, 1)
    return np.stack([r, g, b], -1).astype(np.uint8).tobytes()


# ---------------------------------------------------------------- info
def open_fvf(path):
    """Open an FVF file for parsing; report a damaged or foreign file as a clear error instead of a traceback."""
    try:
        return dec.FVFFile(path)
    except (AssertionError, struct.error, IndexError, ValueError) as e:
        raise SystemExit(f'{path}: not a valid FVF file ({type(e).__name__}: {e})')


def cmd_info(a):
    d = open(a.file, 'rb').read(); pos = 2048; nb = 0; fcs = []
    while pos < len(d):
        _, fc, _, cs, _ = struct.unpack('<HHIII', d[pos:pos + 16]); fcs.append(fc); pos += cs; nb += 1
        if cs < 16: raise SystemExit(f'{a.file}: invalid block size {cs} in block {nb}; not a valid FVF file')
    f = open_fvf(a.file)
    print(f'{a.file}: {len(d)} bytes, {f.frame_count} frames, {nb} blocks, '
          f'{f.frame_count / 15:.1f}s @15fps, audio {f.audio_channels}ch {f.audio_sample_size}bit {f.audio_sample_rate}Hz')


# ---------------------------------------------------------------- decode
def cmd_decode(a):
    f = open_fvf(a.file); f.read_next_block(); st = dec.DecoderState()
    n = f.frame_count if a.frames is None else min(a.frames, f.frame_count)
    if a.png: os.makedirs(a.png, exist_ok=True)
    pcm = bytearray(); ff = None
    if not a.no_mp4:
        ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                               '-s', f'{W}x{H}', '-r', '15', '-i', '-', '-c:v', 'libx264', '-crf', '14',
                               '-pix_fmt', 'yuv420p', a.out + '_silent.mp4'], stdin=subprocess.PIPE)
    last = bytes(W * H * 3)
    for i in range(n):
        vf, vl, af = dec.decode_next_frame(f)
        rgb = dec.decode_video_frame(st, vf, vl)
        if rgb is not None: last = bytes(rgb)
        if af: pcm += af
        if ff: ff.stdin.write(last)
        if a.png:
            from PIL import Image
            Image.frombytes('RGB', (W, H), last).save(os.path.join(a.png, f'{i:05d}.png'))
        if i % 200 == 0: log(f'decoded {i}/{n}')
    if ff:
        ff.stdin.close(); ff.wait()
    wav = a.out + '.wav'
    with wave.open(wav, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(1); w.setframerate(22050); w.writeframes(bytes(pcm))
    if ff:
        subprocess.check_call(['ffmpeg', '-y', '-loglevel', 'error', '-i', a.out + '_silent.mp4', '-i', wav,
                               '-c:v', 'copy', '-c:a', 'aac', '-b:a', '128k', '-shortest', a.out + '.mp4'])
        os.remove(a.out + '_silent.mp4')
        log('wrote', a.out + '.mp4')
    log('wrote', wav)


# ---------------------------------------------------------------- encode
def detect_cuts(frames):
    def chg(a, b):
        g = (np.frombuffer(a[0], np.uint8).reshape(50, 4, 80, 4) != np.frombuffer(b[0], np.uint8).reshape(50, 4, 80, 4)).any(axis=(1, 3)).sum()
        c = (np.frombuffer(a[1], np.uint8).reshape(25, 4, 80, 4) != np.frombuffer(b[1], np.uint8).reshape(25, 4, 80, 4)).any(axis=(1, 3)).sum()
        return (g + c) / 6000
    cuts = []
    for i in range(1, len(frames)):
        if chg(frames[i], frames[i - 1]) > 0.8 and (not cuts or i - cuts[-1] >= 10): cuts.append(i)
    return cuts


def run_pass(frames, budgets, cuts, label, **kw):
    t0 = time.time(); total = len(frames); last = [t0]
    log(f'[{label}] starting: {total} frames')
    recent = [(t0, 0)]
    def prog(i, n, b, lam):
        now = time.time()
        if i == 0 or i == total - 1 or now - last[0] >= 10:
            last[0] = now; el = now - t0; done = i + 1
            recent.append((now, done))
            while len(recent) > 2 and now - recent[1][0] > 60: recent.pop(0)
            t1, d1 = recent[0]
            avg = done / el if el > 0 else 0.0
            cur = (done - d1) / (now - t1) if now - t1 > 0 else avg
            msg = f'[{label}] frame {done}/{total} ({100 * done // total}%)  elapsed {hms(el)}  avg {avg:.2f} fps  now {cur:.2f} fps'
            if done * 10 >= total and done < total and avg > 0:
                msg += f'  ~{hms((total - done) / avg)} left'
            log(msg)
    common = dict(chroma_weight=2.5, lam_floor=0.0, cuts=cuts, first_budget=46000, bad_thr=150,
                  max_global=600, dstride=2, allow_c7_chroma=True, allow_rec=True, progress=prog)
    common.update(kw)
    bs, rec = encoder7.build_encoded_sequence_budgeted(frames, budgets, **common)
    log(f'[{label}] done in {time.time() - t0:.0f}s, {sum(map(len, bs))} video bytes')
    return bs, rec


def plan_slack(vids, auds, pal, target):
    """Return the bytes left over (negative = overflow) when these video chunks are packed the way pack_to_size packs them,
    before the slack is spread over the blocks."""
    import recontainer_lite as rc
    recs = [len(rc.rec(v, a, pal if i == 0 else None)) for i, (v, a) in enumerate(zip(vids, auds))]
    bl, cur, sz, lim = [], [], 16, 3
    for r in recs:
        if cur and (sz + r > MAXBLK or len(cur) >= lim): bl.append(cur); cur = []; sz = 16; lim = 6
        cur.append(r); sz += r
    bl.append(cur)
    return target - 2048 - sum((16 + sum(b) + SECT - 1) // SECT * SECT for b in bl)


def pack_to_size(vids, auds, src_hdr, pal, target):
    """Pack the frame records into blocks as the original files do (3 frames in the first block, then at most 6), and spread the
    slack inside the blocks so the file is exactly `target` bytes. Return (file_bytes, 0), or (None, -overflow) if it does not fit."""
    import recontainer_lite as rc
    recs = [rc.rec(v, a, pal if i == 0 else None) for i, (v, a) in enumerate(zip(vids, auds))]
    bl, cur, sz, lim = [], [], 16, 3
    for r in recs:
        if cur and (sz + len(r) > MAXBLK or len(cur) >= lim): bl.append(cur); cur = []; sz = 16; lim = 6
        cur.append(r); sz += len(r)
    bl.append(cur)
    if len(bl[0]) != 3 or 16 + sum(map(len, bl[0])) > FIRST_BLOCK_MAX:
        raise SystemExit(f'internal error: the first block would hold {len(bl[0])} frames in {16 + sum(map(len, bl[0]))} bytes; '
                         f'the game needs 3 frames and at most {FIRST_BLOCK_MAX} bytes there. Please report this.')
    blocks = []
    for b in bl:
        s = (16 + sum(map(len, b)) + SECT - 1) // SECT * SECT
        blk = struct.pack('<HHIII', 16, len(b), 0, s, 0) + b''.join(b)
        blocks.append(bytearray(blk + bytes(s - len(blk))))
    big = [len(b) for b in blocks if len(b) > MAXBLK]
    if big:
        raise SystemExit(f'internal error: block of {big[0]} bytes exceeds the {MAXBLK}-byte limit the game accepts')
    slack = target - 2048 - sum(map(len, blocks))
    if slack < 0 or slack % SECT: return None, slack
    n = len(blocks); per = (slack // n) // SECT * SECT
    lim = lambda k: FIRST_BLOCK_MAX if k == 0 else MAXBLK
    for k in range(n):
        g = min(per, lim(k) - len(blocks[k]))
        if g > 0: blocks[k] += bytes(g); slack -= g
    k = n - 1
    while slack > 0 and k >= 0:
        g = min(slack, lim(k) - len(blocks[k]))
        if g > 0: blocks[k] += bytes(g); slack -= g
        k -= 1
    if slack: return None, -slack
    sizes = [len(b) for b in blocks]
    out = bytearray(src_hdr)
    # Header bytes 20..23 hold the file offset of the last block.
    struct.pack_into('<I', out, 20, target - sizes[-1])
    for k, b in enumerate(blocks):
        struct.pack_into('<III', b, 4, sizes[k - 1] if k else 0, sizes[k], sizes[k + 1] if k + 1 < n else 0)
        out += b
    assert len(out) == target
    return bytes(out), 0


def _cmd_encode_inner(a):
    nf = open_fvf(a.src).frame_count
    target = os.path.getsize(a.src)
    work = a.work or (a.out + '.work'); os.makedirs(work, exist_ok=True)
    global _LOGF
    _LOGF = open(a.out + '.log', 'a', encoding='utf-8')
    log(f'(progress is also written to {a.out}.log)')
    log(f'source {a.src}: {nf} frames, target size {target}')
    frames, olen = decode_planes(a.src, nf)
    cuts = detect_cuts(frames); log('cuts', cuts)
    hdr, pal = palette_of(a.src)
    _, auds = read_records(a.src, nf)
    # Frame 0 shares its block with the palette chunk and the audio pre-buffer. The block must stay <= MAXBLK, so frame 0's video
    # gets what is left (24+2 record header, palette, 4 audio length, audio).
    f0_budget = MAXBLK - 16 - (24 + 2 + len(pal) + 4 + len(auds[0]))
    # Largest video chunk a later frame may have so that its record still fits in one block (24+2 header, 4 audio length, audio).
    fcap = MAXBLK - 16 - (24 + 2 + 4 + max([len(x) for x in auds[1:] if x] or [0]))
    # The first block holds frames 0-2, as in the original files: at most FIRST_BLOCK_MAX bytes, with frame 0 no larger than F0_MAX.
    # `avail` is the space left for the three video chunks.
    avail = FIRST_BLOCK_MAX - 16 - (24 + 2 + len(pal) + 4 + len(auds[0])) \
        - sum(24 + 2 + (4 + len(auds[k]) if auds[k] else 0) for k in (1, 2) if k < nf) - 64
    # Frame 0 is limited to F0_TARGET; frames 1-2 get the rest of the first block (at least 6,000 B each when there is room).
    f0_budget = max(1500, min(f0_budget, F0_MAX, F0_TARGET, avail - 2 * 6000))
    cap12 = max(1500, (avail - f0_budget) // 2)
    log(f'frame 0 video budget {f0_budget} B (block limit {MAXBLK}, audio pre-buffer {len(auds[0])} B); frames 1-2 capped at {cap12} B so the first block holds 3 frames')

    if a.mode == 'fast':
        sched = _clamp12(list(olen), cap12)
        bs, rec = run_pass(frames, sched, cuts, 'fast', frame_cap=40000, reservoir_cap=300000, lam_floor=0.5, first_budget=f0_budget, pw=a.pw)
    else:
        # pass 1 -> how many bytes each frame gets / needs
        need_p = os.path.join(work, 'need.csv' if a.need == 'lossless' else 'need_slope.csv')   # the file name depends on --need
        if os.path.exists(need_p) and not a.redo:
            need = [float(x) for x in open(need_p).read().split()]
            log('reusing', need_p)
        elif a.need == 'lossless':
            # --need lossless: the bytes each frame needs to be coded (almost) without error.
            ubs, _ = run_pass(frames, _clamp12([int(min(o * 100, fcap)) for o in olen], cap12), cuts, 'pass1-need',
                              frame_cap=fcap, reservoir_cap=620000, first_budget=f0_budget)
            need = [float(len(b)) for b in ubs]
            open(need_p, 'w').write('\n'.join(map(str, need)))
        else:
            # --need slope: every frame is coded at the same Lagrange multiplier (error versus bytes). The multiplier is nudged only so that
            # the running total follows the original's. Each byte then buys the same error reduction wherever it is spent.
            # The search options are the same as in pass 2.
            tgt = [float(x) for x in _clamp12([int(o) for o in olen], cap12)]
            lc = dict(target=tgt, lam0=a.lam0, gain=0.3)
            ubs, rec1 = run_pass(frames, _clamp12([fcap] * nf, cap12), cuts, 'pass1-slope', frame_cap=fcap, reservoir_cap=620000,
                              reserve=0.0, g_search=True, g_thr=8, g_topk=24, first_budget=f0_budget, pw=a.pw, lam_ctrl=lc)
            need = [float(len(b)) for b in ubs]
            open(need_p, 'w').write('\n'.join(map(str, need)))
            open(need_p.replace('.csv', '_mad.csv'), 'w').write('\n'.join('%.4f' % float(np.abs(np.frombuffer(expected_rgb(*frames[i]), np.uint8).astype(np.int16) - np.frombuffer(expected_rgb(*rec1[i]), np.uint8).astype(np.int16)).mean()) for i in range(nf)))
            h = lc.get('hist') or [0]; log(f'pass 1: lambda min {min(h):.1f} median {sorted(h)[len(h) // 2]:.1f} max {max(h):.1f}')
        so, sn = sum(olen), sum(need)
        mad_p = need_p.replace('.csv', '_mad.csv'); pass1_mad = [float(x) for x in open(mad_p).read().split()] if os.path.exists(mad_p) else None
        scale = a.scale
        base0 = [min(need[i] * so / sn, fcap) for i in range(nf)]
        base0 = _mad_tilt(base0, pass1_mad, a.mad_tilt, a.mad_tilt_max, a.mad_tilt_min)
        base = _floor_sched(base0, olen, a.floor, fcap, a.floor_min, pass1_mad, a.floor_mad, a.protect)
        base = _key_boost(base, a.key_boost)

        def make_sched(sc):
            sch = _clamp12([int(base[i] * sc) for i in range(nf)], cap12)
            # Frames 1-2 refine the picture that the 16-pass frame 0 produced: give them at least the original size.
            for i in (1, 2):
                if i < nf: sch[i] = max(sch[i], min(olen[i], 9400, cap12))
            for i in (3, 4):   # the refinement continues in frames 3-4: same minimum
                if i < nf: sch[i] = max(sch[i], min(olen[i], fcap))
            return sch

        while True:
            sched = make_sched(scale)
            if a.autoscale:
                # Pass 2 never spends more than its schedule, so packing the schedule into blocks predicts whether the file fits
                # (block padding can make a plan at the original's total overflow). Lower the scale until the prediction leaves a margin.
                start = scale
                # The prediction error comes from block boundaries, so the margin grows with the number of blocks (frames / 6).
                margin = SECT * (2 + int(0.5 * math.sqrt(nf / 4.8)))
                while scale > 0.5:
                    pslack = plan_slack([bytes(v) for v in sched], auds, pal, target)
                    if pslack >= margin: break
                    scale -= 0.0025; sched = make_sched(scale)
                if scale != start: log(f'autoscale: predicted slack too small at scale {start:.4f}; starting at {scale:.4f} (margin {margin} B)')
            log(f'pass 2: scale {scale:.3f}, scheduled {sum(sched)} vs original {so}')
            bs, rec = run_pass(frames, sched, cuts, 'pass2', frame_cap=a.frame_cap, reservoir_cap=620000,
                               reserve=0.0, g_search=True, g_thr=8, g_topk=24, first_budget=f0_budget, pw=a.pw)
            data, slack = pack_to_size(bs, auds, hdr, pal, target)
            if data is not None: break
            # Lower the scale in proportion to the overshoot (plus a 25% margin) instead of in small fixed steps.
            step = max(0.005, (-slack) / max(1, sum(len(b) for b in bs)) * 1.25)
            log(f'file does not fit (short by {-slack} B); lowering scale by {step:.3f} and retrying')
            scale -= step
            if scale < 0.5: raise SystemExit('cannot fit')
    if a.mode == 'fast':
        data, slack = pack_to_size(bs, auds, hdr, pal, target)
        if data is None: raise SystemExit(f'does not fit (short by {-slack} B); use --mode s10 (auto-fit)')
    open(a.out, 'wb').write(data)
    log('wrote', a.out, len(data), 'bytes (target', target, ')')
    verify(a.out, frames, rec, auds)


def auto_jobs():
    n = os.cpu_count() or 1
    return max(1, min(8, n // 2))


def cmd_encode(a):
    jobs = getattr(a, 'jobs', None)
    jobs = auto_jobs() if jobs is None else max(1, jobs)
    encoder7.set_jobs(jobs)
    log(f'using {jobs} worker process(es) ({"parallel" if jobs > 1 else "serial"}; output is identical either way)')
    try:
        _cmd_encode_inner(a)
    finally:
        encoder7.set_jobs(1)


def verify(path, frames, rec, auds):
    f = dec.FVFFile(path); f.read_next_block(); st = dec.DecoderState()
    if f.frame_count != len(frames): raise RuntimeError('frame count mismatch')
    mads = []
    for i in range(len(frames)):
        vf, vl, af = dec.decode_next_frame(f)
        out = bytes(dec.decode_video_frame(st, vf, vl))
        if out != expected_rgb(*rec[i]): raise RuntimeError(f'decoder output != encoder model at frame {i}')
        if af != auds[i]: raise RuntimeError(f'audio mismatch at frame {i}')
        x = np.frombuffer(expected_rgb(*frames[i]), np.uint8).astype(np.int16)
        mads.append(float(np.abs(x - np.frombuffer(out, np.uint8).astype(np.int16)).mean()))
    log(f'VERIFIED: decoder matches encoder model, audio identical; mean MAD {np.mean(mads):.3f}, '
        f'median {np.median(mads):.3f}, max {max(mads):.2f}, frames>2: {sum(m > 2 for m in mads)}')


# ---------------------------------------------------------------- zip
def cmd_patch_zip(a):
    import patch_zip
    patch_zip.patch(a.orig, a.out, dict(kv.split('=', 1) for kv in a.pairs))


def cmd_check(a):
    import check_fvf
    check_fvf.cmd_scan(a)


def cmd_compare(a):
    import check_fvf
    if not check_fvf.cmd_compare(a): sys.exit(1)


def cmd_view(a):
    import webbrowser, pathlib
    p = os.path.join(HERE, 'fvf_viewer.html')
    if not os.path.exists(p): raise SystemExit('fvf_viewer.html was not found next to fvf.py')
    print('opening', p, 'in your web browser'); webbrowser.open(pathlib.Path(p).as_uri())


def cmd_replace(a):
    tmp = (a.work or (a.out + '.work')); os.makedirs(tmp, exist_ok=True)
    import patch_zip
    pairs = {}
    for kv in a.pairs:
        if '=' not in kv: raise SystemExit(f'expected name=file, got {kv!r}')
        name, src = kv.split('=', 1)
        if not name or os.path.basename(name) != name: raise SystemExit(f'invalid zip entry name {name!r}')
        out = os.path.join(tmp, name)
        ns = argparse.Namespace(src=src, out=out, mode=a.mode, scale=a.scale, work=os.path.join(tmp, name + '.work'),
                                redo=False, jobs=a.jobs, need=a.need, pw=a.pw, floor=a.floor, floor_min=a.floor_min, floor_mad=a.floor_mad, frame_cap=a.frame_cap, mad_tilt=a.mad_tilt, mad_tilt_max=a.mad_tilt_max, mad_tilt_min=a.mad_tilt_min, protect=a.protect, autoscale=a.autoscale, key_boost=a.key_boost, lam0=a.lam0)
        cmd_encode(ns); pairs[name] = out
    patch_zip.patch(a.orig, a.out, pairs)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    vp = argparse.ArgumentParser(add_help=False)
    vp.add_argument('-v', '--verbose', action='store_true', help='print decoder header and block details')
    s = p.add_subparsers(dest='cmd', required=True)
    x = s.add_parser('info', parents=[vp]); x.add_argument('file'); x.set_defaults(fn=cmd_info)
    x = s.add_parser('decode', parents=[vp]); x.add_argument('file'); x.add_argument('out'); x.add_argument('--png')
    x.add_argument('--no-mp4', action='store_true'); x.add_argument('--frames', type=int); x.set_defaults(fn=cmd_decode)
    def enc_opts(x):
        x.add_argument('--mode', choices=['s10', 'fast'], default='s10', help='s10 = two-pass best quality (slow); fast = single pass')
        x.add_argument('--scale', type=float, default=1.0, help='start budget scale (auto-lowered until the file fits)')
        x.add_argument('--work', help='work dir (caches the pass-1 result)')
        x.add_argument('--jobs', type=int, help='worker processes (default: half your CPU threads, max 8; 1 = serial)')
        x.add_argument('--need', choices=['slope', 'lossless'], default='slope',
                       help='pass 1 / budget shape: slope = every frame coded at one common lambda (equal-slope), lossless = bytes each frame needs to be coded without error')
        x.add_argument('--pw', type=float, default=2.0, help='key-frame persistence weight (0 = off): static blocks of key frames count 1+pw times as much')
        x.add_argument('--floor', type=float, default=0.0, help='every frame gets at least this fraction of the original frame size; the others pay (0 = off)')
        x.add_argument('--floor-min', dest='floor_min', type=float, default=0.0, help='with --floor: frames whose pass-1 plan is below this many bytes (near-static frames) are not lifted')
        x.add_argument('--floor-mad', dest='floor_mad', type=float, default=0.03, help='with --floor: frames whose pass-1 picture error (MAD) is below this are not lifted (needs the *_mad.csv written by pass 1)')
        x.add_argument('--frame-cap', dest='frame_cap', type=int, default=24000, help='largest video chunk pass 2 may spend on one frame (the originals never exceed 18,008 B)')
        x.add_argument('--mad-tilt', dest='mad_tilt', type=float, default=0.0, help='byte-neutral: weight frames by (pass-1 error / 75th percentile) ** this, so visibly worse frames get more bytes (0 = off; try 0.25)')
        x.add_argument('--mad-tilt-min', dest='mad_tilt_min', type=float, default=1.0, help='smallest weight --mad-tilt may give a frame (1.0 = never cut by the weighting)')
        x.add_argument('--protect', type=float, default=0.97, help='with --floor and a pass-1 error file: frames pass 1 codes almost perfectly keep at least this fraction of their plan')
        x.add_argument('--mad-tilt-max', dest='mad_tilt_max', type=float, default=1.6, help='largest weight --mad-tilt may give a frame')
        x.add_argument('--no-autoscale', dest='autoscale', action='store_false', help='do not lower the start scale until the packed plan is predicted to fit (default: do)')
        x.add_argument('--key-boost', type=float, default=0.0, dest='key_boost', help='byte-neutral: key frames take this fraction of their three leaf frames budgets')
        x.add_argument('--lam0', type=float, default=4.0, help='starting multiplier of the slope pass (it adapts)')
    x = s.add_parser('encode', parents=[vp]); x.add_argument('src'); x.add_argument('out'); enc_opts(x)
    x.add_argument('--redo', action='store_true'); x.set_defaults(fn=cmd_encode)
    x = s.add_parser('patch-zip', parents=[vp]); x.add_argument('orig'); x.add_argument('out'); x.add_argument('pairs', nargs='+'); x.set_defaults(fn=cmd_patch_zip)
    x = s.add_parser('replace', parents=[vp]); x.add_argument('orig'); x.add_argument('out'); x.add_argument('pairs', nargs='+'); enc_opts(x); x.set_defaults(fn=cmd_replace)
    x = s.add_parser('check', parents=[vp]); x.add_argument('paths', nargs='+'); x.add_argument('--csv'); x.add_argument('--report')
    x.add_argument('--no-decode', action='store_true'); x.add_argument('--deep', action='store_true'); x.set_defaults(fn=cmd_check)
    x = s.add_parser('compare', parents=[vp]); x.add_argument('original'); x.add_argument('new'); x.add_argument('--csv'); x.add_argument('--thr', type=float); x.set_defaults(fn=cmd_compare)
    x = s.add_parser('view', parents=[vp]); x.set_defaults(fn=cmd_view)
    a = p.parse_args()
    if a.verbose: os.environ['FVF_VERBOSE'] = '1'
    a.fn(a)


if __name__ == '__main__':
    main()
