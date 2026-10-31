# SPDX-License-Identifier: CC0-1.0
"""
encoder4.py -- rate-controlled (size-matched) frame encoder: per-block rate-distortion choice under a byte budget.

Goal: one record per logical frame, each no bigger than a caller-supplied
byte budget (e.g. the original file's own per-frame size, ~8.1 KB), by
trading quality for bytes with rate-distortion optimisation.

Per frame:
  1. changed = blocks whose current decoder-side content != target.
  2. For each changed block compute its options and their error (SAD, sum
     of |diff| over the 16 bytes) vs the target:
        skip   cost 0   (leave decoder content as is; error = known vs target)
        case5  cost ~2  (best translated copy within +-R, from the
                         frame-global top-16 offsets that get storage slots)
        case03 cost 3   (best same-position fractal: 4 orientations x 128
                         brightness modifiers, full search, vectorised)
        raw    cost 17  (case_type=6, error 0)
  3. Pick per block argmin(error + lambda*cost). Bisect lambda so the REAL
     emitted bitstream is <= budget (the emitter is the cost oracle, so
     run-length coalescing and header costs are exact).
  4. Closed loop: the next frame's "known" state is what the real decoder
     reconstructs, so skipped/lossy blocks are re-evaluated every frame and
     refine progressively (this is what turns a hard cut into a multi-frame
     refinement instead of a giant spread).
"""
import struct, sys, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from encoder2 import (G_PLANE_SIZE, C_PLANE_SIZE, LUMA_BLOCK_COUNT, TOTAL_BLOCKS,
                      MAX_RUN_PER_CASE4, MAX_VIDEO_BODY_BYTES, PLANE_WIDTH, _read_block,
                      diff_changed_blocks)
from encoder3 import (_read_domain_block, _domain_block_in_bounds, _domain_base_from_offset,
                      _pack_case03_opcode, _decode_one_call, MAX_STORAGE_SLOTS)
import decoder as dec

ML = np.array(dec.modify_lookup, dtype=np.int16)
MODS = (np.arange(-64, 64) + 64) * 256           # (128,)
B2O = list(dec.block_id_to_offset)
INF = 1 << 30
COST_C5, COST_C3, COST_RAW = 2, 3, 17
COST_C5 = float(os.environ.get('FVF_COST_C5', COST_C5))   # modelled bytes of a c5 block (the real average is far lower: opcodes coalesce into runs)

# Distortion measure for every per-block decision (default SAD). FVF_DIST=sse: sum of squared differences; p15: sum |d|^1.5.
# Read from the environment so spawned worker processes see the same setting.
_DM = os.environ.get('FVF_DIST', 'sad').lower()
DIST_POW = {'sad': 1.0, 'sse': 2.0, 'p15': 1.5}[_DM]
def dist(d):
    """Per-pixel distortion of a difference array: |d| (SAD), d*d in int32 (SSE) or |d|^1.5."""
    if DIST_POW == 2.0:
        d = d.astype(np.int32); return d * d
    if DIST_POW != 1.0:
        return np.abs(d).astype(np.float32) ** DIST_POW
    return np.abs(d)
def thr_units(t, npix):
    """Convert a SAD threshold t over npix pixels into the active distortion unit (the value a flat error of t/npix per pixel gives)."""
    return t if DIST_POW == 1.0 else npix * (t / npix) ** DIST_POW


def _as2d(plane, h):
    return np.frombuffer(plane, dtype=np.uint8).reshape(h, PLANE_WIDTH)


def block_options(bid, known_bytes, known2d, target2d, height, radius5):
    """Return a dict with sad_skip, sad_c3 (plus its parameters) and sad_c5 (plus its offset)."""
    off = B2O[bid]; y0, x0 = divmod(off, PLANE_WIDTH)
    t = target2d[y0:y0 + 4, x0:x0 + 4].astype(np.int16)
    k = known2d[y0:y0 + 4, x0:x0 + 4].astype(np.int16)
    res = {'sad_skip': int(dist(t - k).sum()), 'c3': None, 'c5': None}
    # --- case5: best translated copy (excluding zero shift) ---
    ylo, yhi = max(0, y0 - radius5), min(height - 4, y0 + radius5)
    xlo, xhi = max(0, x0 - radius5), min(PLANE_WIDTH - 4, x0 + radius5)
    if yhi >= ylo and xhi >= xlo:
        reg = known2d[ylo:yhi + 4, xlo:xhi + 4].astype(np.int16)
        win = np.lib.stride_tricks.sliding_window_view(reg, (4, 4))   # (ny,nx,4,4)
        sad = dist(win - t).sum(axis=(2, 3))
        zy, zx = y0 - ylo, x0 - xlo
        sad[zy, zx] = INF                                            # same-position = case4, not case5
        iy, ix = np.unravel_index(int(np.argmin(sad)), sad.shape)
        if sad[iy, ix] < INF:
            res['c5'] = (int(sad[iy, ix]), (ylo + iy - y0) * PLANE_WIDTH + (xlo + ix - x0))
    # --- case03: same-position domain, 4 orientations x 128 modifiers ---
    if _domain_block_in_bounds(off, height):
        tf = t.reshape(16)
        best = None
        for ct in (0, 1, 2, 3):
            dom = np.frombuffer(_read_domain_block(known_bytes, off, ct),
                                dtype=np.uint8).astype(np.int32)
            pred = ML[MODS[:, None] + dom[None, :]]                  # (128,16)
            sad = dist(pred - tf).sum(axis=1)
            mi = int(np.argmin(sad))
            if best is None or sad[mi] < best[0]:
                best = (int(sad[mi]), ct, int(MODS[mi]))
        res['c3'] = (best[0], best[1], _domain_base_from_offset(off), best[2])   # (sad, case_type, src_block, modifier)
    return res


def _emit(flags, changed_plan, chosen_offsets, target_g, target_c):
    """changed_plan: bid -> ('raw',) | ('c3', ct, src_block, modifier) | ('c5', offset)."""
    slot_of = {o: i for i, o in enumerate(chosen_offsets)}
    out = bytearray(b'\x00\x00'); out += struct.pack('<H', flags)
    out += struct.pack('<H', len(chosen_offsets))
    for o in chosen_offsets: out += struct.pack('<h', o)
    bid = 0
    while bid < TOTAL_BLOCKS:
        boundary = LUMA_BLOCK_COUNT if bid < LUMA_BLOCK_COUNT else TOTAL_BLOCKS
        if bid not in changed_plan:
            end = bid; mx = min(bid + MAX_RUN_PER_CASE4, boundary)
            while end < mx and end not in changed_plan: end += 1
            out.append(((end - bid - 1) << 3) | 4); bid = end; continue
        p = changed_plan[bid]
        if p[0] == 'c5':
            end = bid + 1; mx = min(bid + 257, boundary)
            while end < mx and changed_plan.get(end, ('',))[0] == 'c5' and changed_plan[end][1] == p[1]:
                end += 1
            n = end - bid; s = slot_of[p[1]]
            if n > 1: out.append((s << 4) | 0x8 | 5); out.append(n - 2)
            else: out.append((s << 4) | 5)
            bid = end
        elif p[0] == 'c3':
            out += _pack_case03_opcode(p[1], p[2], p[3]); bid += 1
        else:
            plane = target_g if bid < LUMA_BLOCK_COUNT else target_c
            out.append(6); out += _read_block(plane, bid); bid += 1
    return bytes(out)


def build_budgeted_bitstream(known_g, known_c, target_g, target_c, budget, flags=0x4,
                             radius5=8, allow_c5=True, allow_c3=True, chroma_weight=1.0, lam_floor=0.0,
                             cut_frac=0.35, chroma_weight_cut=None):
    budget = min(budget, MAX_VIDEO_BODY_BYTES)
    changed = sorted(diff_changed_blocks(known_g, known_c, target_g, target_c))
    if not changed:
        return _emit(flags, {}, [], target_g, target_c), 0.0
    kg2, kc2 = _as2d(known_g, 200), _as2d(known_c, 100)
    tg2, tc2 = _as2d(target_g, 200), _as2d(target_c, 100)
    n = len(changed)
    if chroma_weight_cut is not None and n > cut_frac * TOTAL_BLOCKS:
        chroma_weight = chroma_weight_cut      # cut frame: colour blocks get priority
    sk = np.zeros(n); s3 = np.full(n, float(INF)); s5 = np.full(n, float(INF))
    p3 = [None] * n; p5 = [None] * n
    for j, bid in enumerate(changed):
        luma = bid < LUMA_BLOCK_COUNT
        o = block_options(bid, known_g if luma else known_c, kg2 if luma else kc2, tg2 if luma else tc2,
                          200 if luma else 100, radius5)
        w = 1.0 if luma else chroma_weight
        sk[j] = o['sad_skip'] * w
        if allow_c3 and o['c3']: s3[j] = o['c3'][0] * w; p3[j] = o['c3'][1:]
        if allow_c5 and o['c5']: s5[j] = o['c5'][0] * w; p5[j] = o['c5'][1]
    # choose the frame-global top-16 case5 offsets by total benefit
    benefit = {}
    base = np.minimum(sk, s3)
    for j in range(n):
        if p5[j] is not None and s5[j] < base[j]:
            benefit[p5[j]] = benefit.get(p5[j], 0) + (base[j] - s5[j])
    chosen = sorted(benefit, key=lambda o: -benefit[o])[:MAX_STORAGE_SLOTS]
    cs = set(chosen)
    for j in range(n):
        if p5[j] is not None and p5[j] not in cs: s5[j] = INF
    raw_sad = np.zeros(n)

    def plan_for(lam):
        J = np.stack([sk, s5 + lam * COST_C5, s3 + lam * COST_C3, raw_sad + lam * COST_RAW])
        choice = J.argmin(axis=0)
        plan = {}
        for j, c in enumerate(choice):
            if c == 1: plan[changed[j]] = ('c5', p5[j])
            elif c == 2: plan[changed[j]] = ('c3',) + tuple(p3[j])
            elif c == 3: plan[changed[j]] = ('raw',)
        used = sorted({v[1] for v in plan.values() if v[0] == 'c5'})
        return plan, used

    def size_at(lam):
        plan, used = plan_for(lam)
        bs = _emit(flags, plan, used, target_g, target_c)
        return bs, plan

    # lam = lam_floor gives the best quality worth its bytes; find the smallest lam >= lam_floor whose size fits the budget.
    bs0, _ = size_at(lam_floor)
    if len(bs0) <= budget:
        return bs0, lam_floor
    lo, hi = lam_floor, max(1.0, lam_floor * 2)
    bs_hi, _ = size_at(hi)
    while len(bs_hi) > budget and hi < 1e6:
        lo, hi = hi, hi * 4; bs_hi, _ = size_at(hi)
    best = bs_hi if len(bs_hi) <= budget else None
    for _ in range(14):
        mid = (lo + hi) / 2; bs, _ = size_at(mid)
        if len(bs) <= budget: hi, best = mid, bs
        else: lo = mid
    if best is None:                       # even all-skip is over budget (should not happen)
        best, _ = size_at(1e9)
    return best, hi


def build_encoded_sequence_budgeted(frames, budgets, flags=0x4, radius5=8, progress=None,
                                    reservoir_cap=60000, frame_cap=18008, debt_cap=0, min_frac=0.6,
                                    cuts=None, look=8, reserve=0.3, first_budget=None, **kw):
    """Cumulative size never exceeds sum(budgets) (+ at most debt_cap in flight, repaid by later frames).
    Frames finishing under budget bank the difference (up to reservoir_cap); a frame may spend banked bytes
    up to frame_cap. With debt_cap>0 the bank may go negative (a cut frame borrows from the frames after it);
    a frame's effective budget never drops below min_frac*its own budget."""
    state = dec.DecoderState()
    known_g, known_c = bytes(state.front_buffer), bytes(state.colour_front_buffer)
    bitstreams, recon, bank = [], [], 0
    for i, ((tg, tc), b) in enumerate(zip(frames, budgets)):
        precut = bool(cuts) and any(0 < c - i <= look for c in cuts)
        avail = b + bank                                  # bank < 0 means we are repaying debt
        if debt_cap and bank >= 0:
            avail += debt_cap                             # may borrow beyond the bank
        eff = int(max(min_frac * b, min(avail, max(b, frame_cap))))
        if precut:
            eff = int(b * (1 - reserve))                  # lookahead: underspend before a known cut, bank the rest
        if i == 0 and first_budget: eff = first_budget    # frame 0 (keyframe) always gets its own budget, even next to a cut
        bs, lam = build_budgeted_bitstream(known_g, known_c, tg, tc, eff, flags, radius5, **kw)
        known_g, known_c = _decode_one_call(state, bs)
        bank = max(-debt_cap, min(reservoir_cap, bank + b - len(bs)))
        bitstreams.append(bs); recon.append((known_g, known_c))
        if progress: progress(i, len(bs), eff, lam)
    return bitstreams, recon
