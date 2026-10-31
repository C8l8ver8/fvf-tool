# SPDX-License-Identifier: CC0-1.0
"""
encoder3.py -- block matching for the affine (fractal, case_type 0-3), motion-vector (case_type 5) and 8x8 group
(case_type 7) opcodes, and the opcode bitstream builders that use them.

For every block that changed since the previous picture, cheaper opcodes than a 17-byte raw literal are tried first:

  1. case_type 0-3 (3 bytes): an affine copy of a downsampled 8x8 domain region of the previous picture, with the format's
     fixed 0.75 contrast scale and a brightness offset of 4*i. Covers fades and lighting changes, and shrunk copies.
  2. case_type 5 (2-3 bytes): a copy of a translated region of the previous picture. At most 16 distinct offsets can be
     used per frame (the opcode has a 4-bit index into the frame's storage table), so offsets are chosen per frame.
  3. case_type 7 (8x8 group transform covering four blocks).

Blocks that match none of these are sent as case_type 6 raw literals. With tolerance 0 (the default) only exact matches are
accepted; a non-zero tolerance accepts approximate matches.

--------------------------------------------------------------------------
Domain-block addressing of case_type 0-3 (decoder.py, case_type 0-3 handler):

  X = offset_src % 320, Y = offset_src // 320   (both even for any 4-aligned block anchor)
  src_block = X + 160*Y
  The decoder's inverse is offset_src = get_odd_pixel_offset(src_block) = block_lookup[src_block // 2] = X + 320*Y.

  For row i and column j in 0..3 (orientation 0): src = offset_src + 640*i + 2*j, dest = offset_dest + 320*i + j.

The 640-byte row stride and 2-byte column step mean the domain region is 8x8 pixels, downsampled by 2 on both axes to fill the
4x4 destination block. case_type 1, 2 and 3 read the same region with the rows and/or columns reversed. `_read_domain_block()`
replicates the decoder's stepping exactly, so the predicted output equals what decode_video_frame_data produces.

Every source byte goes through the fixed lookup modify_lookup[(i+64)*256 + b] = clamp(4*i + round(0.75*b), 0, 255), with one
brightness index i in -64..63 per block (the opcode's `modifier` field). The 3-byte opcode packing (`_pack_case03_opcode`) is
the inverse of the decoder's extraction (`modifier = (V<<5)&0x7F00`, `src_block = (V>>9)&0x7FFE`).
--------------------------------------------------------------------------
case_type 5 (motion-vector copy) opcode layout:

  byte0: bits 0-2 = 5 (case_type), bit 3 = "explicit count byte follows", bits 4-7 = `info`, a 4-bit index into state.storage[]
         (so at most 16 distinct offsets per frame).
  if bit 3 is set: byte1 = count - 2 (the run covers count blocks, up to 257); otherwise the opcode covers one block.

`input_val`, the byte offset added to each covered block's own offset_dest to get its source position, is a signed 16-bit value
shared by every block in a run, so all blocks of a run must use the same shift.
--------------------------------------------------------------------------
"""
import struct
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from encoder2 import (  # noqa: E402
    G_PLANE_SIZE, C_PLANE_SIZE, LUMA_BLOCK_COUNT, CHROMA_BLOCK_COUNT,
    TOTAL_BLOCKS, MAX_RUN_PER_CASE4, MAX_VIDEO_BODY_BYTES,
    MAX_RAW_BLOCKS_PER_FRAME, PLANE_WIDTH, _read_block, diff_changed_blocks,
    _spread_update, estimate_bitstream_size,
)

import decoder as dec  # noqa: E402

LUMA_HEIGHT = 200
CHROMA_HEIGHT = 100
MAX_STORAGE_SLOTS = 16  # case_type 5's `info = byte >> 4` is a 4-bit index


# ---------------------------------------------------------------------
# case_type 0-3 (fractal transform) search
# ---------------------------------------------------------------------

def _domain_base_from_offset(offset_src: int) -> int:
    """Return the `src_block` value that encodes a flat plane offset (X + 320*Y, X and Y even); the inverse of
    get_odd_pixel_offset(). See the module docstring."""
    y = offset_src // PLANE_WIDTH
    x = offset_src % PLANE_WIDTH
    return x + 160 * y


def _read_domain_block(plane: bytes, offset_src: int, case_type: int) -> bytes:
    """Read the 16 source bytes that the decoder's case_type 0-3 handler reads for this offset_src and case_type, in its row and
    column order. This is not a contiguous 4x4 read: it samples an 8x8-pixel region at every second pixel (see module docstring)."""
    out = bytearray(16)
    idx = 0
    for i in range(4):
        src = offset_src
        if case_type == 0:
            src += 640 * i
        elif case_type == 1:
            src += 640 * i + 6
        elif case_type == 2:
            src += 640 * (3 - i)
        elif case_type == 3:
            src += 640 * (3 - i) + 6
        for j in range(4):
            out[idx] = plane[src]
            idx += 1
            if case_type in (0, 2):
                src += 2
            else:
                src -= 2
    return bytes(out)


def _domain_block_in_bounds(offset_src: int, height: int) -> bool:
    """Check that the domain read stays inside the plane. It touches columns [X, X+6] and rows [Y, Y+6] for
    (X, Y) = (offset_src % 320, offset_src // 320), whatever the orientation."""
    y0 = offset_src // PLANE_WIDTH
    x0 = offset_src % PLANE_WIDTH
    if x0 < 0 or x0 + 6 >= PLANE_WIDTH:
        return False
    if y0 < 0 or y0 + 6 >= height:
        return False
    return True


def find_case03_match(prev_plane: bytes, target_block: bytes, dest_offset: int,
                       height: int, search_radius: int = 0, tolerance: int = 0):
    """Search for a case_type 0-3 match for `target_block` (16 bytes read from the target plane at dest_offset).
    The block's own position in the previous picture is tried first (fades, flicker), then, if search_radius > 0, nearby
    4-aligned positions (a moving and recolouring object).

    `tolerance` is the largest allowed absolute error of any single output byte (maximum norm). An exact match is returned
    immediately; otherwise the candidate with the smallest maximum error is returned if that error is within `tolerance`.
    tolerance = 0 therefore accepts exact matches only.

    Returns (case_type, src_block, modifier), or None.
    """
    y0 = dest_offset // PLANE_WIDTH
    x0 = dest_offset % PLANE_WIDTH

    candidates = [(x0, y0)]
    if search_radius:
        for dy in range(-search_radius, search_radius + 1, 4):
            for dx in range(-search_radius, search_radius + 1, 4):
                if dx == 0 and dy == 0:
                    continue
                cx, cy = x0 + dx, y0 + dy
                if cx < 0 or cy < 0:
                    continue
                candidates.append((cx, cy))

    best = None  # (error, case_type, src_block, modifier)
    for cx, cy in candidates:
        offset_src = cx + PLANE_WIDTH * cy
        if not _domain_block_in_bounds(offset_src, height):
            continue
        for case_type in (0, 1, 2, 3):
            domain = _read_domain_block(prev_plane, offset_src, case_type)
            for i in range(-64, 64):
                modifier = (i + 64) * 256
                err = 0
                aborted = False
                for k in range(16):
                    d = dec.modify_lookup[modifier + domain[k]] - target_block[k]
                    if d < 0:
                        d = -d
                    if d > err:
                        err = d
                        if best is not None and err >= best[0]:
                            aborted = True
                            break
                if aborted:
                    continue
                if best is None or err < best[0]:
                    src_block = _domain_base_from_offset(offset_src)
                    best = (err, case_type, src_block, modifier)
                    if err == 0:
                        return case_type, src_block, modifier
    if best is not None and best[0] <= tolerance:
        return best[1], best[2], best[3]
    return None


def _pack_case03_opcode(case_type: int, src_block: int, modifier: int) -> bytes:
    """Pack the 3-byte case_type 0-3 opcode, the inverse of the decoder's field extraction:
      modifier = (V<<5) & 0x7F00      ->  V's bits 3-9 hold modifier>>8 (7 bits)
      src_block = (V>>9) & 0x7FFE     ->  V's bits 10-23 hold src_block's bits 1-14"""
    mod7 = (modifier >> 8) & 0x7F          # 7 bits
    src15 = (src_block >> 1) & 0x3FFF      # 14 bits (src_block's bit0 is always 0 -- X,Y both even)
    v = case_type | (mod7 << 3) | (src15 << 10)
    return bytes([v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF])


# ---------------------------------------------------------------------
# case_type 7, subtype 0-7 (8x8 group transform) search
# ---------------------------------------------------------------------
#
# Only the non-recursive subtypes 0-7 are used here: one affine transform covering a 2x2 group of 4x4 blocks, 4 bytes.
# Subtype 12 (large-span mark/copy) and the recursive refinement sub-opcode are written by encoder7.
#
# Bit layout (non-recursive), 4 bytes:
#   byte0: bits 0-2 = 7 (case_type), bits 3-6 = subtype (0-7, the eight orientations), bit 7 unused.
#   byte1: bits 0-6 = i + 64 for the modifier (i + 64) * 256 (same modify_lookup indexing as case_type 0-3), bit 7 = recursive flag (0).
#   bytes 2-3 (u16, little endian): bits 0-13 = src_block16 (_domain_base16_from_offset(offset_src)),
#          bits 14-15 = recursive_offset (0 when not recursive).
#
# Destination: case_type 7 writes a contiguous 8x8 region starting at block_id_to_offset[curr_block_id], the group's first
# (top-left) member, and then advances curr_block_id by 4. A group can therefore only be emitted where curr_block_id is its first
# member and the next three ids are the other three quadrants of the same region (checked by `_case7_group_base`).
#
# Source: an 8 x 8 sample loop over a 16x16-pixel domain region, downsampled by 2 on both axes (case_type 0-3 at double scale).
# Subtypes 0-3 mirror and flip the traversal as case_type 0-3 do; subtypes 4-7 are the transposed versions (the outer loop steps
# along columns, the inner loop along rows).

def _domain_base16_from_offset(offset_src: int) -> int:
    """Return the `src_block16` value that encodes a flat plane offset (X + 320*Y, X and Y even); the inverse of
    get_odd_pixel_offset(src_block16 * 2). Equal to _domain_base_from_offset divided by 2."""
    y = offset_src // PLANE_WIDTH
    x = offset_src % PLANE_WIDTH
    return (x + 160 * y) // 2


def _read_domain_block8x8(plane: bytes, offset_src: int, subtype: int) -> bytes:
    """Read the 64 source bytes that the decoder's case_type 7 handler (subtype 0-7) reads for this offset_src and subtype: an 8x8
    sample read over a 16x16-pixel footprint, returned in output (row, column) order. Subtypes 4-7 change which domain axis the
    outer and inner loops walk, not where a sample lands in the output."""
    out = bytearray(64)
    idx = 0
    for i in range(8):
        src = offset_src
        if subtype == 0:
            src += 640 * i
        elif subtype == 1:
            src += 640 * i + 14
        elif subtype == 2:
            src += 640 * (7 - i)
        elif subtype == 3:
            src += 640 * (7 - i) + 14
        elif subtype == 4:
            src += 2 * i
        elif subtype == 5:
            src += 2 * (7 - i)
        elif subtype == 6:
            src += 640 * 7 + 2 * i
        elif subtype == 7:
            src += 640 * 7 + 2 * (7 - i)
        for j in range(8):
            out[idx] = plane[src]
            idx += 1
            if subtype in (0, 2):
                src += 2
            elif subtype in (1, 3):
                src -= 2
            elif subtype in (4, 5):
                src += 640
            else:  # 6, 7
                src -= 640
    return bytes(out)


def _read_group8x8(plane: bytes, offset_dest: int) -> bytes:
    """Read the 8x8 destination content (64 bytes, row-major) that a case_type 7 write replaces, starting at the group's base offset
    (block_id_to_offset of its first member)."""
    out = bytearray(64)
    idx = 0
    for i in range(8):
        row_off = offset_dest + PLANE_WIDTH * i
        out[idx:idx + 8] = plane[row_off:row_off + 8]
        idx += 8
    return bytes(out)


def _domain_block8x8_in_bounds(offset_src: int, height: int) -> bool:
    """Check that the domain read stays inside the plane. It touches columns [X, X+14] and rows [Y, Y+14] for
    (X, Y) = (offset_src % 320, offset_src // 320), whatever the subtype."""
    y0 = offset_src // PLANE_WIDTH
    x0 = offset_src % PLANE_WIDTH
    if x0 < 0 or x0 + 14 >= PLANE_WIDTH:
        return False
    if y0 < 0 or y0 + 14 >= height:
        return False
    return True


def _case7_group_base(base_bid: int, boundary_end: int) -> list:
    """Check whether `base_bid` anchors a complete 2x2 group for case_type 7: base_bid % 4 == 0, all four ids lie inside the plane
    segment, and their offsets in block_id_to_offset form the (0,0), (+4,0), (0,+4), (+4,+4) pattern of one 8x8 region. This
    rejects, for example, the partial band at the bottom of a plane.

    Returns (member_ids, offset_dest) for the four ids and the base destination offset, or None."""
    if base_bid % 4 != 0:
        return None
    if base_bid + 3 >= boundary_end:
        return None
    ids = [base_bid, base_bid + 1, base_bid + 2, base_bid + 3]
    offs = [dec.block_id_to_offset[i] for i in ids]
    o0 = offs[0]
    expected = [o0, o0 + 4, o0 + 4 * PLANE_WIDTH, o0 + 4 * PLANE_WIDTH + 4]
    if offs != expected:
        return None
    return ids, o0


def find_case7_match(prev_plane: bytes, target_group: bytes, dest_offset: int,
                      height: int, search_radius: int = 0, tolerance: int = 0):
    """Search for a case_type 7 match (subtype 0-7, non-recursive) for `target_group` (64 bytes, from `_read_group8x8`).
    Same strategy and `tolerance` meaning as `find_case03_match`, over all 64 output bytes: the group's own position first,
    then, if search_radius > 0, nearby 8-aligned positions.

    Returns (subtype, src_block16, modifier), or None."""
    y0 = dest_offset // PLANE_WIDTH
    x0 = dest_offset % PLANE_WIDTH

    candidates = [(x0, y0)]
    if search_radius:
        for dy in range(-search_radius, search_radius + 1, 8):
            for dx in range(-search_radius, search_radius + 1, 8):
                if dx == 0 and dy == 0:
                    continue
                cx, cy = x0 + dx, y0 + dy
                if cx < 0 or cy < 0:
                    continue
                candidates.append((cx, cy))

    best = None  # (error, subtype, src_block16, modifier)
    for cx, cy in candidates:
        offset_src = cx + PLANE_WIDTH * cy
        if not _domain_block8x8_in_bounds(offset_src, height):
            continue
        for subtype in range(8):
            domain = _read_domain_block8x8(prev_plane, offset_src, subtype)
            for i in range(-64, 64):
                modifier = (i + 64) * 256
                err = 0
                aborted = False
                for k in range(64):
                    d = dec.modify_lookup[modifier + domain[k]] - target_group[k]
                    if d < 0:
                        d = -d
                    if d > err:
                        err = d
                        if best is not None and err >= best[0]:
                            aborted = True
                            break
                if aborted:
                    continue
                if best is None or err < best[0]:
                    src_block16 = _domain_base16_from_offset(offset_src)
                    best = (err, subtype, src_block16, modifier)
                    if err == 0:
                        return subtype, src_block16, modifier
    if best is not None and best[0] <= tolerance:
        return best[1], best[2], best[3]
    return None


def _pack_case7_opcode(subtype: int, src_block16: int, modifier: int) -> bytes:
    """Pack the 4-byte non-recursive case_type 7 opcode: byte0 = 7 | subtype << 3, byte1 = i + 64 (recursive flag clear),
    bytes 2-3 = src_block16 as u16 little endian (recursive_offset bits 14-15 left 0)."""
    i_plus_64 = (modifier >> 8) & 0x7F
    byte0 = 7 | (subtype << 3)
    byte1 = i_plus_64
    b23 = src_block16 & 0x3FFF
    return bytes([byte0 & 0xFF, byte1 & 0xFF, b23 & 0xFF, (b23 >> 8) & 0xFF])


# ---------------------------------------------------------------------
# case_type 5 (motion-vector copy) search
# ---------------------------------------------------------------------

def find_case5_offset(prev_plane: bytes, target_block: bytes, dest_offset: int,
                       height: int, search_radius: int, tolerance: int = 0):
    """Search a window of byte-offset translations for one at which the previous picture's 4x4 block at dest_offset + offset
    matches target_block. The read is 4 rows of 4 contiguous bytes at a constant offset from dest_offset (row stride 320).
    `tolerance` has the same meaning as in `find_case03_match`.

    Returns the integer `input_val` (may be negative), or None."""
    y0 = dest_offset // PLANE_WIDTH
    x0 = dest_offset % PLANE_WIDTH

    best = None  # (error, offset)
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            if dx == 0 and dy == 0:
                continue  # a zero shift is a case_type 4 skip, which is cheaper and handled elsewhere
            cx, cy = x0 + dx, y0 + dy
            if cx < 0 or cy < 0 or cx + 3 >= PLANE_WIDTH or cy + 3 >= height:
                continue
            src_offset = cx + PLANE_WIDTH * cy
            err = 0
            aborted = False
            for row in range(4):
                so = src_offset + row * PLANE_WIDTH
                do = row * 4
                for k in range(4):
                    d = prev_plane[so + k] - target_block[do + k]
                    if d < 0:
                        d = -d
                    if d > err:
                        err = d
                        if best is not None and err >= best[0]:
                            aborted = True
                            break
                if aborted:
                    break
            if aborted:
                continue
            offset = src_offset - dest_offset
            if best is None or err < best[0]:
                best = (err, offset)
                if err == 0:
                    return offset
    if best is not None and best[0] <= tolerance:
        return best[1]
    return None


# ---------------------------------------------------------------------
# Top-level per-frame bitstream builder
# ---------------------------------------------------------------------

def build_compressed_bitstream(prev_g: bytes, prev_c: bytes, cur_g: bytes, cur_c: bytes,
                                changed_block_ids: set, flags: int = 0x4,
                                search_radius_case5: int = 16,
                                search_radius_case03: int = 0,
                                search_radius_case7: int = 0,
                                attempt_case7: bool = True,
                                tolerance_case03: int = 0,
                                tolerance_case5: int = 0,
                                tolerance_case7: int = 0) -> bytes:
    """Build one frame's bitstream. Like encoder2's build_temporal_skip_bitstream, every unchanged run is a case_type 4 opcode;
    every changed block first tries case_type 0-3, then case_type 5 (which needs one of at most MAX_STORAGE_SLOTS per-frame
    offsets), and otherwise falls back to a case_type 6 raw literal.

    Before that, when `attempt_case7` is set, a group pass looks for 2x2-aligned groups of four blocks, at least one of which
    changed, that one case_type 7 transform can reproduce (4 bytes instead of up to 4 x 17). Members of a matched group are
    covered by the group opcode and are not considered individually.

    `tolerance_case03`, `tolerance_case5` and `tolerance_case7` are passed to the matching find_case* functions (0 = exact
    matches only). When any is non-zero, `cur_g` and `cur_c` are a target to approximate rather than a picture the bitstream
    reproduces exactly, so a caller encoding several frames must feed the decoder's reconstructed output, not `cur_g` and
    `cur_c`, into the next call (see build_encoded_sequence).
    """
    if len(cur_g) != G_PLANE_SIZE:
        raise ValueError(f"g_plane must be {G_PLANE_SIZE} bytes, got {len(cur_g)}")
    if len(cur_c) != C_PLANE_SIZE:
        raise ValueError(f"colour_plane must be {C_PLANE_SIZE} bytes, got {len(cur_c)}")

    # Pass 0: case_type 7 group search
    group_plan = {}    # base_bid -> (subtype, src_block16, modifier)
    covered_ids = set()  # every block_id absorbed into a matched group

    if attempt_case7:
        for boundary_start, boundary_end, is_luma in (
            (0, LUMA_BLOCK_COUNT, True),
            (LUMA_BLOCK_COUNT, TOTAL_BLOCKS, False),
        ):
            plane_prev = prev_g if is_luma else prev_c
            plane_cur = cur_g if is_luma else cur_c
            height = LUMA_HEIGHT if is_luma else CHROMA_HEIGHT
            for base_bid in range(boundary_start, boundary_end, 4):
                grp = _case7_group_base(base_bid, boundary_end)
                if grp is None:
                    continue
                member_ids, offset_dest = grp
                if not any(m in changed_block_ids for m in member_ids):
                    continue  # already free via the ordinary unchanged-run coalescing
                target_group = _read_group8x8(plane_cur, offset_dest)
                m7 = find_case7_match(plane_prev, target_group, offset_dest, height,
                                       search_radius=search_radius_case7,
                                       tolerance=tolerance_case7)
                if m7 is not None:
                    group_plan[base_bid] = m7
                    covered_ids.update(member_ids)

    # Pass 1: find the best match for every changed block
    block_plan = {}  # bid -> ('case03', case_type, src_block, modifier) | ('case5', offset) | ('raw',)
    case5_votes = {}  # offset -> number of blocks that could use it

    for bid in changed_block_ids:
        if bid in covered_ids:
            continue
        is_luma = bid < LUMA_BLOCK_COUNT
        plane_prev = prev_g if is_luma else prev_c
        plane_cur = cur_g if is_luma else cur_c
        height = LUMA_HEIGHT if is_luma else CHROMA_HEIGHT
        dest_offset = dec.block_id_to_offset[bid]
        target_block = _read_block(plane_cur, bid)

        m03 = find_case03_match(plane_prev, target_block, dest_offset, height,
                                 search_radius=search_radius_case03,
                                 tolerance=tolerance_case03)
        if m03 is not None:
            block_plan[bid] = ('case03',) + m03
            continue

        off5 = find_case5_offset(plane_prev, target_block, dest_offset, height,
                                  search_radius=search_radius_case5,
                                  tolerance=tolerance_case5)
        if off5 is not None:
            block_plan[bid] = ('case5', off5)
            case5_votes[off5] = case5_votes.get(off5, 0) + 1
            continue

        block_plan[bid] = ('raw',)

    # Keep only the MAX_STORAGE_SLOTS most useful case_type 5 offsets (the opcode can index 16 storage slots per frame).
    # A block whose only candidate is an offset outside that set falls back to raw.
    chosen_offsets = sorted(case5_votes, key=lambda o: -case5_votes[o])[:MAX_STORAGE_SLOTS]
    slot_of = {off: idx for idx, off in enumerate(chosen_offsets)}
    for bid, plan in list(block_plan.items()):
        if plan[0] == 'case5' and plan[1] not in slot_of:
            block_plan[bid] = ('raw',)

    # Pass 2: emit the bitstream in block_id order
    out = bytearray()
    out += b'\x00\x00'
    out += struct.pack('<H', flags)
    out += struct.pack('<H', len(chosen_offsets))
    for off in chosen_offsets:
        out += struct.pack('<h', off)

    bid = 0
    while bid < TOTAL_BLOCKS:
        boundary = LUMA_BLOCK_COUNT if bid < LUMA_BLOCK_COUNT else TOTAL_BLOCKS

        if bid in group_plan:
            subtype, src_block16, modifier = group_plan[bid]
            out += _pack_case7_opcode(subtype, src_block16, modifier)
            bid += 4
            continue

        if bid not in changed_block_ids:
            # Extend the unchanged run, but never past a group's base block_id: curr_block_id must land exactly
            # there for the group opcode to be emitted (see _case7_group_base).
            run_end = bid
            max_end = min(bid + MAX_RUN_PER_CASE4, boundary)
            while (run_end < max_end and run_end not in changed_block_ids
                   and run_end not in group_plan):
                run_end += 1
            count = run_end - bid
            out.append(((count - 1) << 3) | 4)
            bid = run_end
            continue

        plan = block_plan[bid]

        if plan[0] == 'case5':
            offset = plan[1]
            slot = slot_of[offset]
            # Extend a run of consecutive block_ids that all use this slot: the count field applies one shared input_val
            # to every block in the run.
            run_end = bid + 1
            max_end = min(bid + 257, boundary)  # count+2 fits a byte -> up to 257 blocks
            while (run_end < max_end and run_end in changed_block_ids
                   and block_plan.get(run_end, ('raw',))[0] == 'case5'
                   and block_plan[run_end][1] == offset):
                run_end += 1
            count = run_end - bid
            if count > 1:
                out.append((slot << 4) | 0x8 | 5)
                out.append(count - 2)
            else:
                out.append((slot << 4) | 5)
            bid = run_end

        elif plan[0] == 'case03':
            _, case_type, src_block, modifier = plan
            out += _pack_case03_opcode(case_type, src_block, modifier)
            bid += 1

        else:  # raw literal
            plane = cur_g if bid < LUMA_BLOCK_COUNT else cur_c
            out.append(6)
            out += _read_block(plane, bid)
            bid += 1

    return bytes(out)


def _decode_one_call(state, bitstream: bytes):
    """Decode one emitted record with decoder.py and return the (luma_plane, colour_plane) bytes it produces.
    The buffers are captured before the call, as decode_video_frame does, because the persistent state pointers may swap during
    the call. This is the picture the decoder really holds, which build_encoded_sequence needs as the reference for the next
    frame whenever a tolerance is non-zero. The 16x-repeat flag (0x1) is handled like decode_video_frame does.
    """
    flags = dec.read_le_u16(bitstream, 2)
    if flags & 0x8:
        # Skip frame: nothing is decoded or swapped, so the caller's known_g/known_c remain correct.
        state.curr_status = 2
        return None

    was_front = state.front_buffer
    was_colour_front = state.colour_front_buffer
    dec.decode_video_frame_data(state, bitstream, len(bitstream))
    if flags & 0x1:
        was_front = state.front_buffer
        was_colour_front = state.colour_front_buffer
        for _ in range(15):
            dec.decode_video_frame_data(state, bitstream, len(bitstream))
    return bytes(was_front), bytes(was_colour_front)


def build_encoded_sequence(frames, flags: int = 0x4, search_radius_case5: int = 16,
                            search_radius_case03: int = 0, search_radius_case7: int = 0,
                            attempt_case7: bool = True,
                            tolerance_case03: int = 0,
                            tolerance_case5: int = 0,
                            tolerance_case7: int = 0):
    """Encode a sequence: same contract as encoder2's build_encoded_sequence (returns (bitstreams, frame_record_counts)), using
    build_compressed_bitstream for each frame.

    A frame whose change cannot fit one record even in the best case (estimate_bitstream_size, which counts every changed block
    as raw) is sent with encoder2's `_spread_update`, which uses raw literals only and therefore reproduces its target exactly.

    The reference for each next frame is what the decoder actually reconstructs from the emitted records. A single persistent
    DecoderState decodes every emitted record through `_decode_one_call`, and its output becomes `known_g` and `known_c`. With
    all tolerances at 0 this equals the target; with a tolerance it is the approximation the decoder really shows.
    """
    if not frames:
        return [], []

    state = dec.DecoderState()
    # The initial picture is whatever DecoderState starts with (the buffers are filled with 0xAB).
    known_g = bytes(state.front_buffer)
    known_c = bytes(state.colour_front_buffer)
    bitstreams = []
    frame_record_counts = []

    for target_g, target_c in frames:
        changed = diff_changed_blocks(known_g, known_c, target_g, target_c)
        if estimate_bitstream_size(changed) <= MAX_VIDEO_BODY_BYTES:
            bs = build_compressed_bitstream(
                known_g, known_c, target_g, target_c, changed,
                flags=flags, search_radius_case5=search_radius_case5,
                search_radius_case03=search_radius_case03,
                search_radius_case7=search_radius_case7,
                attempt_case7=attempt_case7,
                tolerance_case03=tolerance_case03,
                tolerance_case5=tolerance_case5,
                tolerance_case7=tolerance_case7,
            )
            bitstreams.append(bs)
            frame_record_counts.append(1)
            recs = [bs]
        else:
            # _spread_update modifies bytearrays in place: pass copies, because known_g and known_c are set only by the
            # decode below.
            known_g_ba = bytearray(known_g)
            known_c_ba = bytearray(known_c)
            recs = _spread_update(known_g_ba, known_c_ba, target_g, target_c, flags=flags)
            bitstreams.extend(recs)
            frame_record_counts.append(len(recs))

        for rec in recs:
            decoded = _decode_one_call(state, rec)
            if decoded is not None:
                known_g, known_c = decoded

    return bitstreams, frame_record_counts
