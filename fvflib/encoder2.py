# SPDX-License-Identifier: CC0-1.0
"""
encoder2.py -- temporal-skip frame encoder and shared constants.

Each block is either unchanged since the previous picture, which is coded with the run-length "copy previous content" opcode
(case_type 4; one byte covers up to 32 blocks), or changed, which is re-sent as a raw literal (case_type 6, 17 bytes: opcode plus
16 pixels). encoder3 and encoder4 build on the helpers and constants defined here.

Facts about the format that this module relies on:

* The decoder reads `case_type = frame[data_pos] & 7` once per dispatch, so top-level opcodes are 0-7. The large-span
  "mark or copy" run is case_type 7 with subtype 12; this encoder does not use it and chains case_type 4 runs instead.
* A block whose content changes is always sent as a write opcode, which resets its block_status to 0. Runs of unchanged blocks can
  therefore be sent with case_type 4 whatever their length, together with flags = 0x1 (swap the buffers on every frame).
* A frame record is at most 65535 bytes (frame_length is a u16), which limits one frame to 3853 raw blocks. The decoder buffers
  start filled with 0xAB, so a first picture that needs more than that must be spread over several records.
  `_spread_update` does this for any transition: it covers the blocks that differ between what the decoder's front buffer holds
  (`known_*`) and the target, using as many records as the size cap requires. A spread inserts extra records, so the output can
  contain more records than there were input frames; `build_encoded_sequence` returns how many records each input frame produced.
"""
import struct
import sys
import os

import decoder as dec

PLANE_WIDTH = 320
G_PLANE_SIZE = 320 * 200
C_PLANE_SIZE = 320 * 100
LUMA_BLOCK_COUNT = 4000
CHROMA_BLOCK_COUNT = 2000
TOTAL_BLOCKS = LUMA_BLOCK_COUNT + CHROMA_BLOCK_COUNT  # 6000
MAX_RUN_PER_CASE4 = 32          # 5-bit count field: (count-1) fits 0..31
MAX_FRAME_RECORD_BYTES = 65535  # decoder.py parses frame_length as u16
# Space for the opcode stream: the frame record is reduced by its 24-byte header, the 2-byte video length and the
# 6-byte bitstream header (unused u16, flags u16, storage_count u16).
MAX_VIDEO_BODY_BYTES = MAX_FRAME_RECORD_BYTES - 24 - 2 - 6
MAX_RAW_BLOCKS_PER_FRAME = MAX_VIDEO_BODY_BYTES // 17  # 3853, floor


def _read_block(plane: bytes, bid: int) -> bytes:
    """Read the 4x4 content (16 bytes, row-major) of one block_id from a luma or colour plane, using the same block_id_to_offset
    and 320-byte row stride as decoder.py."""
    offset = dec.block_id_to_offset[bid]
    out = bytearray(16)
    for row in range(4):
        row_off = offset + row * 320
        out[row * 4:row * 4 + 4] = plane[row_off:row_off + 4]
    return bytes(out)


def diff_changed_blocks(prev_g: bytes, prev_c: bytes, cur_g: bytes, cur_c: bytes) -> set:
    """Return the set of block_ids whose 4x4 content differs between two pictures. Blocks are compared through block_id_to_offset,
    so the result does not depend on the planes' non-raster block numbering."""
    if len(prev_g) != G_PLANE_SIZE or len(cur_g) != G_PLANE_SIZE:
        raise ValueError(f"g_plane must be {G_PLANE_SIZE} bytes")
    if len(prev_c) != C_PLANE_SIZE or len(cur_c) != C_PLANE_SIZE:
        raise ValueError(f"colour_plane must be {C_PLANE_SIZE} bytes")

    changed = set()
    for bid in range(LUMA_BLOCK_COUNT):
        if _read_block(prev_g, bid) != _read_block(cur_g, bid):
            changed.add(bid)
    for bid in range(LUMA_BLOCK_COUNT, TOTAL_BLOCKS):
        if _read_block(prev_c, bid) != _read_block(cur_c, bid):
            changed.add(bid)
    return changed


def build_temporal_skip_bitstream(cur_g: bytes, cur_c: bytes, changed_block_ids: set,
                                   flags: int = 0x1) -> bytes:
    """Build one frame's video bitstream: a case_type 6 raw literal for every block_id in changed_block_ids and a case_type 4
    "unchanged" run for every other block_id, in block_id order 0..5999. Runs are split at the 5-bit count field of case_type 4
    (at most 32 blocks per opcode) and at the luma/chroma boundary at block_id 4000: the decoder only checks the plane limit
    between opcodes, so a run that crossed it would keep using the luma buffers."""
    if len(cur_g) != G_PLANE_SIZE:
        raise ValueError(f"g_plane must be {G_PLANE_SIZE} bytes, got {len(cur_g)}")
    if len(cur_c) != C_PLANE_SIZE:
        raise ValueError(f"colour_plane must be {C_PLANE_SIZE} bytes, got {len(cur_c)}")

    out = bytearray()
    out += b'\x00\x00'
    out += struct.pack('<H', flags)
    out += struct.pack('<H', 0)  # storage_count = 0: no case_type 5 (motion copy) in this encoder

    bid = 0
    while bid < TOTAL_BLOCKS:
        boundary = LUMA_BLOCK_COUNT if bid < LUMA_BLOCK_COUNT else TOTAL_BLOCKS
        if bid in changed_block_ids:
            plane = cur_g if bid < LUMA_BLOCK_COUNT else cur_c
            out.append(6)  # case_type=6, high bits free (written 0)
            out += _read_block(plane, bid)
            bid += 1
        else:
            # extend the unchanged run as far as possible, capped by the
            # per-opcode 32-block limit AND the luma/chroma boundary
            run_end = bid
            max_end = min(bid + MAX_RUN_PER_CASE4, boundary)
            while run_end < max_end and run_end not in changed_block_ids:
                run_end += 1
            count = run_end - bid  # 1..32
            out.append(((count - 1) << 3) | 4)  # case_type=4, count field = count-1
            bid = run_end

    return bytes(out)


def estimate_bitstream_size(changed_block_ids: set) -> int:
    """Return a cheap upper bound of the bitstream size (header plus worst-case run splitting) without building it."""
    n_changed = len(changed_block_ids)
    n_unchanged = TOTAL_BLOCKS - n_changed
    # Worst case: every run is split at 32 blocks. Real runs can only be fewer, so this never underestimates.
    n_run_opcodes = -(-n_unchanged // MAX_RUN_PER_CASE4) if n_unchanged else 0
    return 6 + n_changed * 17 + n_run_opcodes


def _spread_update(known_g: bytearray, known_c: bytearray,
                    target_g: bytes, target_c: bytes, flags: int = 0x1) -> list:
    """Cover every block_id whose content differs between `known_*` (what the decoder's front buffer holds) and `target_*`,
    using raw (case_type 6) blocks spread over as many frame records as the size cap requires: one record when the change fits.

    `known_g` and `known_c` are updated in place and equal the targets once all returned bitstreams have been played in order.
    Returns a list of one or more bitstreams in playback order.
    """
    changed = diff_changed_blocks(bytes(known_g), bytes(known_c), target_g, target_c)
    remaining = sorted(changed)
    bitstreams = []
    while True:
        # MAX_RAW_BLOCKS_PER_FRAME is not a safe per-record budget by itself: it counts only the 17 bytes of each raw block, not
        # the run opcodes that mark all other blocks unchanged in the same record. So the block count is reduced until the
        # bitstream that is actually built fits.
        n = min(MAX_RAW_BLOCKS_PER_FRAME, len(remaining))
        while True:
            budget_ids = remaining[:n]
            bs = build_temporal_skip_bitstream(target_g, target_c, set(budget_ids), flags=flags)
            if len(bs) <= MAX_VIDEO_BODY_BYTES or n <= 0:
                break
            n -= 50  # shrink in steps of 50 blocks and retry
        remaining = remaining[len(budget_ids):]
        bitstreams.append(bs)
        for bid in budget_ids:
            plane_known = known_g if bid < LUMA_BLOCK_COUNT else known_c
            plane_src = target_g if bid < LUMA_BLOCK_COUNT else target_c
            offset = dec.block_id_to_offset[bid]
            for row in range(4):
                row_off = offset + row * 320
                plane_known[row_off:row_off + 4] = plane_src[row_off:row_off + 4]
        if not remaining:
            break
    return bitstreams


def build_encoded_sequence(frames, flags: int = 0x1):
    """Encode a sequence. `frames` is a list of (luma_plane, colour_plane) tuples in playback order.
    Returns `(bitstreams, frame_record_counts)`: the per-record video bitstreams in playback order, and for every input frame the
    number of consecutive records it produced (normally 1; more when its change was too large for one record).

    A `known` reference picture starts as the 0xAB fill of DecoderState and is updated by `_spread_update` for every frame,
    frame 0 included.
    """
    if not frames:
        return [], []

    known_g = bytearray([0xAB]) * G_PLANE_SIZE
    known_c = bytearray([0xAB]) * C_PLANE_SIZE
    bitstreams = []
    frame_record_counts = []

    for target_g, target_c in frames:
        recs = _spread_update(known_g, known_c, target_g, target_c, flags=flags)
        bitstreams.extend(recs)
        frame_record_counts.append(len(recs))

    return bitstreams, frame_record_counts
