# SPDX-License-Identifier: CC0-1.0
"""encoder7.py -- the frame encoder used by fvf.py.

Per-frame rate-distortion search: each 4x4 block is coded as a skip, a motion-vector copy (offsets chosen per frame), an affine
(fractal) copy, a raw block, or as part of an 8x8 group transform whose worst quadrant can be refined by a nested sub-opcode.
Supports a constant-slope mode (every frame coded at the same Lagrange multiplier), a context-aware byte model, key-frame
persistence weights, and a 16-pass fractal frame 0. Frames are searched in parallel worker processes when jobs > 1."""
import struct, sys, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import encoder4 as e4
from encoder4 import *
from encoder3 import _case7_group_base, _domain_base16_from_offset

COST_C7 = 4
CTX_COST = os.environ.get('FVF_CTX', '1') == '1'
GROUPS_L = []; GROUPS_C = []
for _b in range(0, LUMA_BLOCK_COUNT, 4):
    g = _case7_group_base(_b, LUMA_BLOCK_COUNT)
    if g: GROUPS_L.append((_b, g[1], False))
for _b in range(LUMA_BLOCK_COUNT, TOTAL_BLOCKS, 4):
    g = _case7_group_base(_b, TOTAL_BLOCKS)
    if g and (g[1] % 320) % 160 < 160 - 8 + 1: GROUPS_C.append((_b, g[1], True))

def _orient_windows(known2d, stride):
    D = known2d[::2, ::2]
    W = np.lib.stride_tricks.sliding_window_view(D, (8, 8))[::stride, ::stride]
    ny, nx = W.shape[:2]
    W = W.reshape(-1, 8, 8); T = W.transpose(0, 2, 1)
    var = [W, W[:, :, ::-1], W[:, ::-1, :], W[:, ::-1, ::-1], T, T[:, ::-1, :], T[:, :, ::-1], T[:, ::-1, ::-1]]
    base = [(((3 * v.reshape(len(v), 64).astype(np.int16)) + 2) >> 2) for v in var]
    ys, xs = np.divmod(np.arange(ny * nx), nx)
    return base, ys * stride, xs * stride

_QM = np.zeros((64, 4), np.float32)
for _p in range(64): _QM[_p, ((_p // 8) // 4) * 2 + ((_p % 8) // 4)] = 1
def search_c7(wins, t64, half=None, w=1.0):
    """Return (best_total, best_minus_worst); each is (sad_total, quads[4], sub, offset_src, modifier). half: None | 0 | 1 (chroma colour plane)."""
    base, wy, wx = wins
    if half is None: sel = None
    else:
        sel = np.nonzero((wx < 80 - 7) if half == 0 else (wx >= 80))[0]
        if len(sel) == 0: return None, None
    t = t64.astype(np.int16)[None, :]
    bt = bm = None
    for s in range(8):
        b = base[s] if sel is None else base[s][sel]
        yy = wy if sel is None else wy[sel]; xx = wx if sel is None else wx[sel]
        m = (t - b).mean(axis=1); i0 = np.rint(m / 4.0).astype(np.int16)
        # saturated targets (fades to white/black): the plain mean is dragged off by clipped pixels, so also centre on the
        # unclipped target pixels only, and try the two extreme brightness steps
        tm_ = t[0]; msk = (tm_ > 0) & (tm_ < 255)
        cands = [np.clip(i0 + d, -64, 63) for d in (-1, 0, 1)]
        if msk.any() and not msk.all():
            i1 = np.rint((t[:, msk] - b[:, msk]).mean(axis=1) / 4.0).astype(np.int16)
            cands += [np.clip(i1 + d, -64, 63) for d in (-1, 0, 1)]
        elif not msk.any():
            cands += [np.full_like(i0, 63), np.full_like(i0, -64)]
        for i in cands:
            err = e4.dist(np.clip(b + 4 * i[:, None], 0, 255) - t)               # (n,64)
            q = err.astype(np.float32) @ _QM                                     # TL,TR,BL,BR
            tot = q.sum(axis=1); mw = tot - q.max(axis=1)
            k = int(np.argmin(tot))
            if bt is None or tot[k] < bt[0]:
                bt = (float(tot[k]) * w, q[k] * w, s, 2 * int(xx[k]) + 320 * 2 * int(yy[k]), (int(i[k]) + 64) * 256)
            k = int(np.argmin(mw))
            if bm is None or mw[k] < bm[0]:
                bm = (float(tot[k]) * w, q[k] * w, s, 2 * int(xx[k]) + 320 * 2 * int(yy[k]), (int(i[k]) + 64) * 256)
    return bt, bm


def _g_domains(known2d):
    """Build all 4-orientation 4x4 domain vectors (after the 0.75 scale, before brightness) over every legal anchor.
    Return Qc (4N,16) float32 mean-removed, Q (4N,16) int16, and the meta arrays (ct, anchor_offset)."""
    h = known2d.shape[0]
    Dsub = known2d[::2, ::2]
    W = np.lib.stride_tricks.sliding_window_view(Dsub, (4, 4))            # (ny,nx,4,4)
    ny, nx = W.shape[:2]
    ay, ax = np.divmod(np.arange(ny * nx), nx)
    off = (2 * ay) * 320 + 2 * ax
    vs = [W, W[:, :, :, ::-1], W[:, :, ::-1, :], W[:, :, ::-1, ::-1]]
    Q = np.concatenate([(((3 * v.reshape(-1, 16).astype(np.int16)) + 2) >> 2) for v in vs])
    ct = np.repeat(np.arange(4), ny * nx); offs = np.tile(off, 4)
    Qf = Q.astype(np.float32); Qc = Qf - Qf.mean(axis=1, keepdims=True)
    return Qc, (Qc * Qc).sum(axis=1), Q, ct, offs

_ML = np.array(e4.dec.modify_lookup, dtype=np.int16)
def global_c3(known2d, T, topk=16, chunk=256):
    """For T (M,16) int16 targets, return the best exhaustive affine match of each as (sad, ct, src_block, modifier)."""
    Qc, qn, Q, ct, offs = _g_domains(known2d)
    out = []
    Tf = T.astype(np.float32); Tc = Tf - Tf.mean(axis=1, keepdims=True)
    for a in range(0, len(T), chunk):
        sc = qn[:, None] - 2.0 * (Qc @ Tc[a:a + chunk].T)                  # (4N, m)
        k = min(topk, sc.shape[0] - 1)
        top = np.argpartition(sc, k, axis=0)[:topk]                         # (topk, m)
        for m in range(top.shape[1]):
            t = T[a + m]; ids = top[:, m]
            q = Q[ids]                                                      # (k,16)
            tm = t.astype(np.int16)
            msk = (tm > 0) & (tm < 255)
            ref = (tm[None, :] - q)
            ref = ref[:, msk] if msk.any() else ref
            i0 = np.rint(ref.mean(axis=1) / 4.0).astype(np.int32)
            best = None
            for d in (-2, -1, 0, 1, 2):
                i = np.clip(i0 + d, -64, 63)
                p = np.clip(4 * i[:, None] + q, 0, 255)
                s_ = e4.dist(p - tm[None, :]).sum(axis=1)
                kk = int(np.argmin(s_))
                if best is None or s_[kk] < best[0]: best = (int(s_[kk]), int(ids[kk]), int(i[kk]))
            for i in (-64, 63):
                p = np.clip(4 * i + q, 0, 255); s_ = e4.dist(p - tm[None, :]).sum(axis=1); kk = int(np.argmin(s_))
                if s_[kk] < best[0]: best = (int(s_[kk]), int(ids[kk]), i)
            sad, gid, i = best
            o = int(offs[gid]); y, x = divmod(o, 320)
            out.append((sad, int(ct[gid]), x + 160 * y, (i + 64) * 256, o))
    return out

def _emit(flags, plan, chosen_offsets, target_g, target_c):
    slot_of = {o: i for i, o in enumerate(chosen_offsets)}
    out = bytearray(b'\x00\x00'); out += struct.pack('<H', flags)
    out += struct.pack('<H', len(chosen_offsets))
    for o in chosen_offsets: out += struct.pack('<h', o)
    bid = 0; last3 = False
    while bid < TOTAL_BLOCKS:
        last3 = False
        boundary = LUMA_BLOCK_COUNT if bid < LUMA_BLOCK_COUNT else TOTAL_BLOCKS
        if bid not in plan:
            end = bid; mx = min(bid + MAX_RUN_PER_CASE4, boundary)
            while end < mx and end not in plan: end += 1
            out.append(((end - bid - 1) << 3) | 4); bid = end; continue
        p = plan[bid]
        if p[0] == 'c7':
            _, sub, src16, mod, ref = p
            b23 = src16 & 0x3FFF
            if ref is not None: b23 |= ref[0] << 14
            out += bytes([7 | (sub << 3), ((mod >> 8) & 0x7F) | (0x80 if ref is not None else 0), b23 & 0xFF, b23 >> 8])
            if ref is not None:
                q, kind = ref[0], ref[1]
                if kind == 'copy': out.append(4)
                elif kind == 'c3': out += e4._pack_case03_opcode(ref[2], ref[3], ref[4]); last3 = True
                else:
                    plane = target_g if bid < LUMA_BLOCK_COUNT else target_c
                    out.append(6); out += e4._read_block(plane, bid + q)
            bid += 4
        elif p[0] == 'c5':
            end = bid + 1; mx = min(bid + 257, boundary)
            while end < mx and plan.get(end, ('',))[0] == 'c5' and plan[end][1] == p[1]: end += 1
            n = end - bid; s = slot_of[p[1]]
            if n > 1: out.append((s << 4) | 0x8 | 5); out.append(n - 2)
            else: out.append((s << 4) | 5)
            bid = end
        elif p[0] == 'c3':
            out += e4._pack_case03_opcode(p[1], p[2], p[3]); bid += 1; last3 = True
        else:
            plane = target_g if bid < LUMA_BLOCK_COUNT else target_c
            out.append(6); out += e4._read_block(plane, bid); bid += 1
    if last3: out.append(0)   # decoder reads a u32 at a case 0-3 opcode: keep 1 byte of slack
    return bytes(out)


# ---------------------------------------------------------------- optional multi-process helpers
# Every piece parallelised below is a pure function of one block (or one group of blocks); results are gathered in the
# original order, so the output is identical to the serial code (jobs=1).
import multiprocessing as _mp
_POOL = None; _NJOBS = 1

def set_jobs(n):
    """n<=1: serial. n>1: use a persistent pool of n worker processes."""
    global _POOL, _NJOBS
    if _POOL is not None:
        _POOL.terminate(); _POOL.join(); _POOL = None
    _NJOBS = max(1, int(n))
    if _NJOBS > 1:
        for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            os.environ[k] = '1'          # inherited by the workers: one BLAS thread each, no oversubscription
        _POOL = _mp.get_context('spawn').Pool(_NJOBS)

def _pmap(fn, tasks):
    if _POOL is None or len(tasks) < 2: return [fn(t) for t in tasks]
    return _POOL.map(fn, tasks, chunksize=1)

def _split(seq, parts):
    k = max(1, min(parts, len(seq))); q, r = divmod(len(seq), k); out = []; a = 0
    for i in range(k):
        b = a + q + (1 if i < r else 0); out.append(seq[a:b]); a = b
    return out

def _t_options(a):
    bids, known_g, known_c, kg2, kc2, tg2, tc2, radius5 = a
    out = []
    for bid in bids:
        luma = bid < LUMA_BLOCK_COUNT
        out.append(block_options(bid, known_g if luma else known_c, kg2 if luma else kc2, tg2 if luma else tc2, 200 if luma else 100, radius5))
    return out

def _t_global(a):
    k2, T, topk = a
    return global_c3(k2, T, topk)

def _t_c7(a):
    wins, items = a              # items: [(t64, half, w)]
    return [search_c7(wins, t64, half, w) for t64, half, w in items]

def _choose_c5(changed, kg2, kc2, tg2, tc2, sk, s3, s5, p5, chroma_weight, nslots=16, npop=40, grid=3):
    """Pick the frame's (up to) 16 shared offsets first (greedy, by total SAD saved over all blocks), then let every block
    choose among them. Candidates: the most popular per-block best offsets plus a small grid around zero."""
    n = len(changed)
    if n == 0: return
    base = np.minimum(sk, s3).astype(float)
    pop = {}
    for o in p5:
        if o is not None: pop[o] = pop.get(o, 0) + 1
    cands = set(sorted(pop, key=lambda o: -pop[o])[:npop])
    for dy in range(-grid, grid + 1):
        for dx in range(-grid, grid + 1):
            if dy or dx: cands.add(dy * 320 + dx)
    cands = sorted(cands)
    lum = np.array([changed[j] < LUMA_BLOCK_COUNT for j in range(n)])
    w = np.where(lum, 1.0, chroma_weight)
    Y0 = np.zeros(n, int); X0 = np.zeros(n, int)
    for j, bid in enumerate(changed): Y0[j], X0[j] = divmod(B2O[bid], 320)
    ar = np.arange(4)
    planes = []
    for is_l, k2, t2 in ((True, kg2, tg2), (False, kc2, tc2)):
        sel = np.nonzero(lum == is_l)[0]
        if len(sel) == 0: continue
        rows = Y0[sel][:, None] + ar; cols = X0[sel][:, None] + ar
        T = t2[rows[:, :, None], cols[:, None, :]].astype(np.int16)
        planes.append((sel, k2, T, k2.shape[0]))
    S = np.full((len(cands), n), float(INF))
    for ci, off in enumerate(cands):
        dy, dx = divmod(off + 160, 320); dx -= 160   # offset = dy*320+dx with |dx| < 160
        for sel, k2, T, h in planes:
            ys = Y0[sel] + dy; xs = X0[sel] + dx
            ok = (ys >= 0) & (ys <= h - 4) & (xs >= 0) & (xs <= 316)
            if not ok.any(): continue
            r = np.clip(ys, 0, h - 4)[:, None] + ar; c = np.clip(xs, 0, 316)[:, None] + ar
            K = k2[r[:, :, None], c[:, None, :]].astype(np.int16)
            sad = e4.dist(K - T).sum(axis=(1, 2)).astype(float) * w[sel]
            S[ci, sel[ok]] = sad[ok]
    b = base.copy(); chosen = []
    for _ in range(nslots):
        gain = np.maximum(0.0, b[None, :] - S).sum(axis=1)
        if chosen: gain[chosen] = 0
        k = int(np.argmax(gain))
        if gain[k] <= 0: break
        chosen.append(k); b = np.minimum(b, S[k])
    for j in range(n):
        if chosen:
            ci = min(chosen, key=lambda c: S[c, j])
            if S[ci, j] < INF: s5[j] = S[ci, j]; p5[j] = cands[ci]; continue
        s5[j] = float(INF); p5[j] = None

def build_budgeted_bitstream(known_g, known_c, target_g, target_c, budget, flags=0x4, radius5=8,
                             allow_c5=True, allow_c3=True, allow_c7=True, allow_c7_chroma=True, allow_rec=True,
                             chroma_weight=1.0, lam_floor=0.0, cut_frac=0.35, chroma_weight_cut=None,
                             dstride=2, force_all=False, max_global=1000, bad_thr=60, gate_on_skip=True, g_search=False, g_thr=16, g_topk=16,
                             wblock=None, lam_fixed=None):
    budget = min(budget, MAX_VIDEO_BODY_BYTES)
    changed = list(range(TOTAL_BLOCKS)) if force_all else sorted(diff_changed_blocks(known_g, known_c, target_g, target_c))
    if not changed: return _emit(flags, {}, [], target_g, target_c), 0.0
    kg2, kc2 = e4._as2d(known_g, 200), e4._as2d(known_c, 100)
    tg2, tc2 = e4._as2d(target_g, 200), e4._as2d(target_c, 100)
    n = len(changed); idx = {b: j for j, b in enumerate(changed)}
    if chroma_weight_cut is not None and n > cut_frac * TOTAL_BLOCKS: chroma_weight = chroma_weight_cut
    sk = np.zeros(n); s3 = np.full(n, float(INF)); s5 = np.full(n, float(INF)); p3 = [None] * n; p5 = [None] * n
    opts_all = [o for part in _pmap(_t_options, [(c, known_g, known_c, kg2, kc2, tg2, tc2, radius5) for c in _split(changed, _NJOBS)]) for o in part]
    for j, bid in enumerate(changed):
        luma = bid < LUMA_BLOCK_COUNT
        o = opts_all[j]
        w = 1.0 if luma else chroma_weight
        sk[j] = (float(INF) if force_all else o['sad_skip'] * w)
        if allow_c3 and o['c3']: s3[j] = o['c3'][0] * w; p3[j] = o['c3'][1:]
        if allow_c5 and o['c5'] and not (force_all and o['c5'][1] == 0): s5[j] = o['c5'][0] * w; p5[j] = o['c5'][1]
    if g_search and allow_c3:
        for plane_is_luma in (True, False):
            sel = [j for j in range(n) if (changed[j] < LUMA_BLOCK_COUNT) == plane_is_luma and min(sk[j], s3[j], s5[j]) > e4.thr_units(g_thr, 16) * (1.0 if plane_is_luma else chroma_weight)]
            if not sel: continue
            k2 = kg2 if plane_is_luma else kc2; t2 = tg2 if plane_is_luma else tc2
            T = np.stack([t2[y:y + 4, x:x + 4].reshape(16) for y, x in (divmod(B2O[changed[j]], 320) for j in sel)]).astype(np.int16)
            res = [r for part in _pmap(_t_global, [(k2, Tp, g_topk) for Tp in _split(T, _NJOBS)]) for r in part]
            w = 1.0 if plane_is_luma else chroma_weight
            for j, (sad, ct_, sb, mod, _o) in zip(sel, res):
                if sad * w < s3[j]: s3[j] = sad * w; p3[j] = (ct_, sb, mod)
    if allow_c5:
        _choose_c5(changed, kg2, kc2, tg2, tc2, sk, s3, s5, p5, chroma_weight)
    wfull = None
    if wblock is not None:
        # Per-block distortion weights (key-frame persistence, see _persist_weights): an error in a heavy block counts wblock[b] times as much.
        wfull = np.asarray(wblock, float)
        wj = np.array([wfull[b] for b in changed])
        sk = sk * wj; s3 = s3 * wj; s5 = s5 * wj
    raw_sad = np.zeros(n)
    g_info = []   # (gb, mem_js, [cand...], refine_opts[4])
    if allow_c7:
        best_blk = np.minimum(np.minimum(sk, s3), s5)
        cand = []
        for gb, o0, ch in GROUPS_L + (GROUPS_C if allow_c7_chroma else []):
            mem = [idx[b] for b in range(gb, gb + 4) if b in idx]
            if mem:
                # Gate on the skip error, not the best-option error: a block that c3 reproduces almost exactly still costs 3 B,
                # while an 8x8 c7 covers four of them for ~4 B (fades and flashes, where every block changes).
                bad = float(sum(sk[j] for j in mem)) if gate_on_skip else float(sum(best_blk[j] for j in mem))
                cand.append((bad, gb, o0, ch, mem))
        cand.sort(key=lambda c: -c[0] / (chroma_weight if c[3] else 1.0))   # rank luma and colour groups on the same (unweighted) scale
        wl = wc = None
        todo = [c for c in cand if c[0] >= e4.thr_units(bad_thr, 64)]
        if len(todo) > max_global: max_global = min(len(todo), max(max_global, 1600))   # whole-screen fades: do not starve luma groups
        pos = 0
        while pos < len(todo) and len(g_info) < max_global:
            batch = todo[pos:pos + (max_global - len(g_info))]; pos += len(batch)
            items = []
            for badness, gb, o0, ch, mem in batch:
                y0, x0 = divmod(o0, 320)
                if not ch:
                    if wl is None: wl = _orient_windows(kg2, dstride)
                    items.append((0, tg2[y0:y0 + 8, x0:x0 + 8].reshape(64), None, 1.0))
                else:
                    if wc is None: wc = _orient_windows(kc2, dstride)
                    items.append((1, tc2[y0:y0 + 8, x0:x0 + 8].reshape(64), 0 if x0 < 160 else 1, chroma_weight))
            res = [None] * len(batch)
            for plane, wins in ((0, wl), (1, wc)):
                idxs = [i for i, it in enumerate(items) if it[0] == plane]
                if not idxs: continue
                parts = _split(idxs, _NJOBS)
                outs = _pmap(_t_c7, [(wins, [items[i][1:] for i in part]) for part in parts])
                for part, o in zip(parts, outs):
                    for i, r in zip(part, o): res[i] = r
            for (badness, gb, o0, ch, mem), (bt, bm) in zip(batch, res):
                cands = [c for c in (bt, bm) if c is not None]
                if wfull is not None:
                    wq = np.array([wfull[gb + q] for q in range(4)])
                    cands = [(float((c[1] * wq).sum()), c[1] * wq) + tuple(c[2:]) for c in cands]
                if not cands: continue
                ref = []
                for q in range(4):
                    b = gb + q; opts = []
                    if b in idx:
                        j = idx[b]
                        opts.append((float(sk[j]), 1, ('copy',)))
                        if p3[j] is not None: opts.append((float(s3[j]), 3, ('c3',) + tuple(p3[j])))
                    else: opts.append((0.0, 1, ('copy',)))
                    opts.append((0.0, 17, ('raw',)))
                    ref.append(opts)
                g_info.append((gb, mem, cands, ref))
    # Context-aware byte model (FVF_CTX=1). The emitter makes a c5 block cost ~0.4 B on average (neighbours with the same offset slot
    # coalesce into one 2-byte run opcode), while a lone block inside a skip run costs its own opcode plus one extra skip-run opcode. A
    # constant per-mode cost cannot express that, so after the first argmin the costs are re-derived from the neighbours' choices
    # (emission order = block id order, per plane) and the argmin is repeated in checkerboard sweeps.
    barr = np.array(changed); lo_b = np.where(barr < LUMA_BLOCK_COUNT, 0, LUMA_BLOCK_COUNT); hi_b = np.where(barr < LUMA_BLOCK_COUNT, LUMA_BLOCK_COUNT, TOTAL_BLOCKS)
    prv, nxt = barr - 1, barr + 1; hasp, hasn = prv >= lo_b, nxt < hi_b
    slot = np.array([10 ** 9 if o is None else o for o in p5])
    def ctx_choice(lam, c5_0=1.0, sweeps=3):
        J = np.stack([sk, s5 + lam * c5_0, s3 + lam * COST_C3, raw_sad + lam * COST_RAW]); ch = J.argmin(axis=0)
        for _ in range(sweeps):
            for par in (0, 1):
                mode = np.zeros(TOTAL_BLOCKS + 2, int); sl = np.full(TOTAL_BLOCKS + 2, -10 ** 9)
                mode[barr] = ch; sl[barr[ch == 1]] = slot[ch == 1]
                mp = np.where(hasp, mode[prv], 1); mn = np.where(hasn, mode[nxt], 1)
                sp = np.where(hasp, sl[prv], -10 ** 9 - 1); sn = np.where(hasn, sl[nxt], -10 ** 9 - 1)
                same = ((mp == 1) & (sp == slot)) | ((mn == 1) & (sn == slot))
                split = ((mp == 0) & (mn == 0)) * 1.0
                J = np.stack([sk, s5 + lam * np.where(same, 0.3, 1.0 + split), s3 + lam * (COST_C3 + split), raw_sad + lam * (COST_RAW + split)])
                ch = np.where(barr % 2 == par, J.argmin(axis=0), ch)
        return ch, J.min(axis=0)
    def plan_for(lam):
        if CTX_COST:
            choice, Jmin = ctx_choice(lam); plan = {}
        else:
            J = np.stack([sk, s5 + lam * COST_C5, s3 + lam * COST_C3, raw_sad + lam * COST_RAW])
            choice = J.argmin(axis=0); Jmin = J.min(axis=0); plan = {}
        for gb, mem, cands, ref in g_info:
            jb = sum(Jmin[j] for j in mem); bestJ = jb; bestp = None
            for (s7, quads, sub, osrc, mod) in cands:
                Jp = s7 + lam * COST_C7
                if Jp < bestJ: bestJ, bestp = Jp, ('c7', sub, _domain_base16_from_offset(osrc), mod, None)
                if allow_rec:
                    for q in range(4):
                        e, c, data = min(ref[q], key=lambda o: o[0] + lam * o[1])
                        Jr = s7 - quads[q] + e + lam * (COST_C7 + c)
                        if Jr < bestJ: bestJ, bestp = Jr, ('c7', sub, _domain_base16_from_offset(osrc), mod, (q,) + data)
            if bestp is not None:
                plan[gb] = bestp
                for b in range(gb + 1, gb + 4): plan[b] = ('c7cov',)
                for j in mem: choice[j] = 0
        for j, c in enumerate(choice):
            b = changed[j]
            if b in plan: continue
            if c == 1: plan[b] = ('c5', p5[j])
            elif c == 2: plan[b] = ('c3',) + tuple(p3[j])
            elif c == 3: plan[b] = ('raw',)
        used = sorted({v[1] for v in plan.values() if v[0] == 'c5'})
        return plan, used
    def size_at(lam):
        plan, used = plan_for(lam); return _emit(flags, plan, used, target_g, target_c), plan
    if lam_fixed is not None:   # constant-slope mode: `budget` is only a safety cap (see build_encoded_sequence_budgeted, lam_ctrl)
        bsf, _ = size_at(lam_fixed)
        if len(bsf) <= budget: return bsf, lam_fixed
    bs0, _ = size_at(lam_floor)
    if len(bs0) <= budget: return bs0, lam_floor
    lo, hi = lam_floor, max(1.0, lam_floor * 2); bs_hi, _ = size_at(hi)
    while len(bs_hi) > budget and hi < 1e6: lo, hi = hi, hi * 4; bs_hi, _ = size_at(hi)
    best = bs_hi if len(bs_hi) <= budget else None
    for _ in range(14):
        mid = (lo + hi) / 2; bs, _ = size_at(mid)
        if len(bs) <= budget: hi, best = mid, bs
        else: lo = mid
    if best is None: best, _ = size_at(1e9)
    return best, hi



# ---------------------------------------------------------------- frame 0 coded the way the original files do it
# Every original file starts with a frame flagged "repeat 16x": the decoder runs the same frame data 16 times, each
# pass reading the previous pass's picture (starting from the flat 0xAB fill). That is iterated-fractal coding: every
# block is built from a (shrunk, brightness-mapped) copy of another part of the picture, and the picture converges.
# Blocks that copy the same position (skip) or use plain motion copies (no shrinking) do not converge from the fill,
# so frame 0 may only use affine blocks (opcodes 0-3), 8x8 transforms and raw blocks.
FRACTAL_FRAME0 = True
F0_TOPK = 64
F0_MIN_BUDGET = 18010   # 6000 x 3 B + 6 B header + 1 B slack = 18,007

TREE_REFERENCES = True   # the original files' reference structure (see build_encoded_sequence_budgeted)

_PW_IDX = None
def _persist_weights(frames, i, pw, thr=24, span=(1, 2, 3, 4)):
    """Key frames are the references: an error a key frame leaves in a block is inherited by the following frames that skip-copy it and by
    the next key frame. Weight each block of key frame i by 1 + pw * (fraction of the next 4 frames whose target block is nearly the same,
    SAD <= thr), so that static detail (stars, ship hulls) is repaired once in the key frame."""
    global _PW_IDX
    if _PW_IDX is None:
        li = np.array([divmod(B2O[b], 320) for b in range(LUMA_BLOCK_COUNT)]) // 4
        ci = np.array([divmod(B2O[b], 320) for b in range(LUMA_BLOCK_COUNT, TOTAL_BLOCKS)]) // 4
        _PW_IDX = (li, ci)
    li, ci = _PW_IDX
    cnt = [np.zeros((50, 80)), np.zeros((25, 80))]; n = 0
    for k in span:
        if i + k >= len(frames): continue
        n += 1
        for pi, h in ((0, 200), (1, 100)):
            a = np.frombuffer(frames[i][pi], np.uint8).reshape(h, 320).astype(np.int16)
            b = np.frombuffer(frames[i + k][pi], np.uint8).reshape(h, 320).astype(np.int16)
            cnt[pi] += (np.abs(a - b).reshape(h // 4, 4, 80, 4).sum(axis=(1, 3)) <= thr)
    if n == 0: return None
    w = np.ones(TOTAL_BLOCKS)
    w[:LUMA_BLOCK_COUNT] += pw * cnt[0][li[:, 0], li[:, 1]] / n
    w[LUMA_BLOCK_COUNT:] += pw * cnt[1][ci[:, 0], ci[:, 1]] / n
    return w

def build_encoded_sequence_budgeted(frames, budgets, flags=0x4, radius5=48, progress=None,
                                    reservoir_cap=60000, frame_cap=18008, debt_cap=0, min_frac=0.6,
                                    cuts=None, look=8, reserve=0.3, first_budget=None, tree=None,
                                    pw=0.0, lam_ctrl=None, **kw):
    """Frame 0 is coded as a 16x-repeat fractal frame. Other frames follow the reference structure of the original files:
    frame 0 has flags 5, every frame with i % 4 == 0 has flags 4 (the buffers swap, so it becomes the new reference) and the three
    frames after it have flags 0 (no swap: all three predict from that key frame, and the next key frame predicts from the previous
    key frame). tree=False gives every frame flags 4, so each predicts from the frame before it.
    pw: persistence weight for key frames (see _persist_weights). lam_ctrl={'target': per-frame bytes, 'lam0': L, 'gain': g}: constant-slope
    mode -- every frame is coded at the same Lagrange multiplier (budgets are only caps) and the multiplier is nudged by the cumulative
    byte ratio, so bytes go where they buy the most error reduction (result: lam_ctrl['hist'] = multiplier used per frame)."""
    if tree is None: tree = TREE_REFERENCES
    state = e4.dec.DecoderState()
    known_g, known_c = bytes(state.front_buffer), bytes(state.colour_front_buffer)
    bitstreams, recon, bank = [], [], 0
    lam_c = lam_ctrl['lam0'] if lam_ctrl else None; cum_t = cum_a = 0.0; lam_hist = []
    for i, ((tg, tc), b) in enumerate(zip(frames, budgets)):
        precut = bool(cuts) and any(0 < c - i <= look for c in cuts)
        avail = b + bank
        if debt_cap and bank >= 0: avail += debt_cap
        eff = int(max(min_frac * b, min(avail, max(b, frame_cap))))
        if precut: eff = int(b * (1 - reserve))
        first = (i == 0)
        if first and first_budget: eff = first_budget
        kwi = dict(kw)
        if pw and tree and i % 4 == 0 and not first:
            wb = _persist_weights(frames, i, pw)
            if wb is not None: kwi['wblock'] = wb
        if lam_ctrl and not first: kwi['lam_fixed'] = lam_c
        if first and FRACTAL_FRAME0:
            # Frame 0 consists of exactly 6000 affine blocks (opcodes 0-3), about 18,008 B. The 16-pass picture is the fixed point of those
            # transforms, so an exhaustive global search for every block (g_thr=0, not only for the blocks the local search left above g_thr)
            # finds them. Spending extra bytes on raw blocks gave a worse picture. A budget below about 6,006 B cannot hold 6000 affine blocks and the rate loop
            # would fall back to a degenerate plan, so the budget is clamped.
            kw0 = dict(kw); kw0.update(g_search=True, g_thr=0, g_topk=F0_TOPK)
            bs, lam = build_budgeted_bitstream(tg, tc, tg, tc, max(eff, F0_MIN_BUDGET), 0x5, radius5, force_all=True, allow_c5=False, gate_on_skip=False, **kw0)
        elif tree:
            # the reference is the decoder's back buffer as it stands before this call
            bs, lam = build_budgeted_bitstream(bytes(state.back_buffer), bytes(state.colour_back_buffer), tg, tc, eff,
                                               4 if i % 4 == 0 else 0, radius5, **kwi)
        else:
            bs, lam = build_budgeted_bitstream(known_g, known_c, tg, tc, eff, flags, radius5, **kwi)
        known_g, known_c = e4._decode_one_call(state, bs)
        # Frame 0 is sized by first_budget, outside the schedule. Whatever it leaves unspent must not become bank that frames 1-2 then
        # spend, because they share the first block with it, which must stay <= 61,440 B.
        if not first: bank = max(-debt_cap, min(reservoir_cap, bank + b - len(bs)))
        bitstreams.append(bs); recon.append((known_g, known_c))
        if lam_ctrl:
            lam_hist.append(lam_c)
            if not first:
                cum_t += lam_ctrl['target'][i]; cum_a += len(bs)
                lam_c = min(lam_ctrl.get('max', 60.0), max(lam_ctrl.get('min', 0.5), lam_c * (cum_a / max(1.0, cum_t)) ** lam_ctrl.get('gain', 0.3)))
        if progress: progress(i, len(bs), eff, lam)
    if lam_ctrl: lam_ctrl['hist'] = lam_hist
    return bitstreams, recon
