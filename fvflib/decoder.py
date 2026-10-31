# SPDX-License-Identifier: CC0-1.0
"""decoder.py -- decoder for the FVF video format (Total Multimedia / SoftVideo) used by
"Star Trek: The Next Generation - A Final Unity".

An FVF file is a 2048-byte header followed by blocks of frame records. Each record carries a video chunk, an optional
palette and an optional raw audio chunk (mono, 8-bit unsigned PCM, 22050 Hz). The 320x200 picture is stored as a luma plane
and a half-width colour plane, which are coded in 4x4 blocks with affine (fractal) copies of an earlier picture, motion-vector
copies, skip runs, raw blocks and 8x8 group transforms.

Main entry points:
    FVFFile(path)                      parse the header; read_next_block() then decode_next_frame(f) yield the raw chunks
    DecoderState()                     decoder state (picture buffers, block status table)
    decode_video_frame(state, v, n)    decode one video chunk and return the 320x200 RGB picture (bytes-like, 3 bytes per pixel)

Set the environment variable FVF_VERBOSE=1 to print header and block details while parsing.
See the README for acknowledgements.
"""
import os
import struct

WIDTH = 320
HEIGHT = 200

# ---------------------------------------------------------------- lookup tables

table1_x = [0, 4, 0, 4, 8, 0xc, 8, 0xc, 0, 4, 0, 4, 8, 0xc, 8, 0xc]
table1_y = [0, 0, 4, 4, 0, 0, 4, 4, 8, 8, 0xc, 0xc, 8, 8, 0xc, 0xc]
table2_x = [0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3]
table2_y = [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3]

def setup_subset_block_id_to_offset(ptr, off, counter, width, size, count, base, halved):
    ourcount = width // size
    if halved:
        ourcount //= 2
    for i in range(ourcount):
        for j in range(count):
            val = base + (i * size) + table1_x[j] + ((counter + table1_y[j]) * width)
            ptr[off[0]] = val
            off[0] += 1

def setup_block_id_to_offset(ptr):
    off = [0]
    counter = 0
    for counter in range(0, 192, 16):
        setup_subset_block_id_to_offset(ptr, off, counter, 320, 16, 16, 0, False)
    counter = 192
    setup_subset_block_id_to_offset(ptr, off, counter, 320, 16, 8, 0, False)
    for counter in range(0, 96, 8):
        setup_subset_block_id_to_offset(ptr, off, counter, 320, 16, 8, 0, True)
    counter = 96
    setup_subset_block_id_to_offset(ptr, off, counter, 320, 8, 2, 0, True)
    for counter in range(0, 96, 8):
        setup_subset_block_id_to_offset(ptr, off, counter, 320, 16, 8, 160, True)
    counter = 96
    setup_subset_block_id_to_offset(ptr, off, counter, 320, 8, 2, 160, True)
    return off[0]

def setup_subset_block_status_lookup(ptr, off, counter, width, size, count, base, halved):
    ourcount = width // (size * 4)
    if halved:
        ourcount //= 2
    for i in range(ourcount):
        for j in range(count):
            val = base + (i * size) + table2_x[j] + ((table2_y[j] + counter // 4) * width) // 4
            ptr[off[0]] = val
            off[0] += 1

def setup_block_status_lookup(ptr):
    off = [0]
    for counter in range(0, 192, 16):
        setup_subset_block_status_lookup(ptr, off, counter, 320, 4, 16, 0, False)
    counter = 192
    setup_subset_block_status_lookup(ptr, off, counter, 320, 4, 8, 0, False)
    for counter in range(200, 296, 8):
        setup_subset_block_status_lookup(ptr, off, counter, 320, 4, 8, 0, True)
    counter = 296
    setup_subset_block_status_lookup(ptr, off, counter, 320, 2, 2, 0, True)
    for counter in range(200, 296, 8):
        setup_subset_block_status_lookup(ptr, off, counter, 320, 4, 8, 40, True)
    counter = 296
    setup_subset_block_status_lookup(ptr, off, counter, 320, 2, 2, 40, True)
    return off[0]

# Build the tables and check their sizes
block_id_to_offset = [0] * 6000
n1 = setup_block_id_to_offset(block_id_to_offset)

block_status_lookup = [0] * 6000
n2 = setup_block_status_lookup(block_status_lookup)

modify_lookup = [0] * 32768
a = 0
for i in range(-64, 64):
    b = 0
    for j in range(256):
        x = (4 * i) + ((3 * b + 2) >> 2)
        b += 1
        if x < 0: x = 0
        elif x > 0xff: x = 0xff
        modify_lookup[a] = x
        a += 1
assert a == 32768
assert n1 == 6000 and n2 == 6000

block_lookup = [0] * ((320*200)//4)
for x in range(0, 320, 2):
    for y in range(0, 200, 2):
        block_lookup[(x + 160*y)//2] = x + 320*y

# ---------------------------------------------------------------- file parsing

def read_u16(data, pos):
    return struct.unpack('<H', data[pos:pos+2])[0]

def read_u32(data, pos):
    return struct.unpack('<I', data[pos:pos+4])[0]

class FVFFile:
    def __init__(self, path):
        with open(path, 'rb') as f:
            self.data = f.read()
        self.parse_header()

    def parse_header(self):
        data = self.data
        assert data[0:4] == b'FVF ', "bad magic"
        self.data_offset = read_u32(data, 16)
        self.header_offset = read_u32(data, 24)
        self.audio_header_offset = read_u32(data, 28)

        pos = self.header_offset
        unknown = read_u16(data, pos); pos += 2
        assert unknown == 40, f"expected 40, got {unknown}"
        unknown2 = read_u16(data, pos); pos += 2
        assert unknown2 == 1
        self.bpp = read_u16(data, pos); pos += 2
        self.width = read_u16(data, pos); pos += 2
        self.height = read_u16(data, pos); pos += 2
        self.us_per_frame = read_u32(data, pos); pos += 4
        assert self.bpp == 16
        assert self.width == 320
        assert self.height == 200
        assert self.us_per_frame == 66666

        padding = read_u32(data, pos); pos += 4
        assert padding == 0
        self.frame_count = read_u32(data, pos); pos += 4
        padding = read_u32(data, pos); pos += 4
        assert padding == 0
        unknown3 = read_u16(data, pos); pos += 2
        assert unknown3 == 0x18
        self.palette_entry_count = data[pos]; pos += 1
        assert self.palette_entry_count == 112
        self.header_pal_count = data[pos]; pos += 1
        assert self.header_pal_count == 15
        self.header_pal_offset = read_u32(data, pos); pos += 4

        # Audio header: ten little-endian u16 fields at audio_header_offset. Every known file has the same values
        # (mono, 8-bit unsigned PCM, 22050 Hz, uncompressed), so they are only checked, not used to branch.
        apos = self.audio_header_offset
        a_unknown1 = read_u16(data, apos); apos += 2
        assert a_unknown1 == 0
        a_unknown2 = read_u16(data, apos); apos += 2
        assert a_unknown2 == 1
        self.audio_channels = read_u16(data, apos); apos += 2
        assert self.audio_channels == 1, f"expected mono (1), got {self.audio_channels}"
        self.audio_sample_size = read_u16(data, apos); apos += 2
        assert self.audio_sample_size == 8, f"expected 8-bit samples, got {self.audio_sample_size}"
        self.audio_sample_rate = read_u16(data, apos); apos += 2
        assert self.audio_sample_rate == 22050, f"expected 22050Hz, got {self.audio_sample_rate}"
        a_unknown3 = read_u16(data, apos); apos += 2
        assert a_unknown3 == 0
        a_unknown4 = read_u16(data, apos); apos += 2
        assert a_unknown4 == 0
        a_unknown5 = read_u16(data, apos); apos += 2
        assert a_unknown5 == 0
        a_unknown6 = read_u16(data, apos); apos += 2
        assert a_unknown6 == 2

        if os.environ.get("FVF_VERBOSE"): print(f"Parsed header: {self.width}x{self.height} bpp={self.bpp} frames={self.frame_count}")
        if os.environ.get("FVF_VERBOSE"): print(f"Audio: {self.audio_channels}ch {self.audio_sample_size}-bit unsigned PCM @ {self.audio_sample_rate}Hz")
        if os.environ.get("FVF_VERBOSE"): print(f"data_offset={self.data_offset} header_pal_offset={self.header_pal_offset}")

        self.cur_pos = self.data_offset
        self.curr_block_remaining_frames = 0
        self.curr_block_remaining_size = 0

    def read_next_block(self):
        data = self.data
        pos = self.cur_pos
        curr_block_header_size = read_u16(data, pos); pos += 2
        assert curr_block_header_size == 0x10, f"expected 0x10, got {hex(curr_block_header_size)}"
        curr_block_frame_count = read_u16(data, pos); pos += 2
        prev_block_size = read_u32(data, pos); pos += 4
        curr_block_size = read_u32(data, pos); pos += 4
        next_block_size = read_u32(data, pos); pos += 4
        self.curr_block_remaining_frames = curr_block_frame_count
        self.curr_block_remaining_size = curr_block_size - curr_block_header_size
        self.cur_pos = pos
        if os.environ.get("FVF_VERBOSE"): print(f"  block: frame_count={curr_block_frame_count} size={curr_block_size}")

# ---------------------------------------------------------------- frame records

def decode_next_frame(fvf):
    data = fvf.data
    if fvf.curr_block_remaining_frames == 0:
        fvf.cur_pos += fvf.curr_block_remaining_size
        fvf.read_next_block()

    frame_start = fvf.cur_pos
    pos = frame_start
    header_length = read_u16(data, pos); pos += 2
    assert header_length == 0x18, f"expected 0x18, got {hex(header_length)}"
    frame_length = read_u16(data, pos); pos += 2
    assert frame_length <= fvf.curr_block_remaining_size
    unknown = read_u16(data, pos); pos += 2
    video_offset = read_u32(data, pos); pos += 4
    palette_offset = read_u32(data, pos); pos += 4
    audio_offset = read_u32(data, pos); pos += 4
    unknown2 = read_u32(data, pos); pos += 4
    unknown3 = read_u16(data, pos); pos += 2

    read_so_far = header_length
    video_frame = None
    video_length = 0
    audio_frame = None

    if video_offset != 0:
        video_offset -= read_so_far
        assert video_offset == 0, f"video_offset mismatch: {video_offset}"
        video_length = read_u16(data, frame_start + read_so_far)
        read_so_far += 2
        video_frame = data[frame_start + read_so_far : frame_start + read_so_far + video_length]
        read_so_far += video_length

    if palette_offset != 0:
        palette_offset -= read_so_far
        assert palette_offset == 0, f"palette_offset mismatch: {palette_offset}"
        read_so_far += fvf.palette_entry_count * 3

    if audio_offset != 0:
        audio_offset -= read_so_far
        assert audio_offset == 0, f"audio_offset mismatch: {audio_offset}"
        audio_length = read_u32(data, frame_start + read_so_far)
        read_so_far += 4
        # Raw 8-bit unsigned PCM payload.
        audio_frame = data[frame_start + read_so_far : frame_start + read_so_far + audio_length]
        read_so_far += audio_length

    assert read_so_far == frame_length, f"read_so_far={read_so_far} frame_length={frame_length}"

    fvf.curr_block_remaining_frames -= 1
    fvf.curr_block_remaining_size -= frame_length
    fvf.cur_pos = frame_start + frame_length

    return video_frame, video_length, audio_frame

# ---------------------------------------------------------------- video chunk decoding

def read_le_u16(buf, pos):
    return buf[pos] | (buf[pos+1] << 8)

def read_le_u32(buf, pos):
    return buf[pos] | (buf[pos+1]<<8) | (buf[pos+2]<<16) | (buf[pos+3]<<24)

class DecoderState:
    def __init__(self):
        self.storage = bytearray(256*2)
        self.curr_status = 2
        self.block_status = bytearray(6000)
        self.real_front_buffer = bytearray([0xAB]*64000)
        self.real_back_buffer = bytearray([0xAB]*64000)
        self.real_colour_front_buffer = bytearray([0xAB]*32000)
        self.real_colour_back_buffer = bytearray([0xAB]*32000)
        self.front_buffer = self.real_front_buffer
        self.back_buffer = self.real_back_buffer
        self.colour_front_buffer = self.real_colour_front_buffer
        self.colour_back_buffer = self.real_colour_back_buffer
        self.rgb_output = bytearray(WIDTH * HEIGHT * 3)

def get_odd_pixel_offset(inn):
    return block_lookup[inn // 2]

def decode_video_frame_data(state, frame, flen):
    our_front = state.front_buffer
    our_back = state.back_buffer
    our_colour_front = state.colour_front_buffer
    our_colour_back = state.colour_back_buffer

    flags = read_le_u16(frame, 2)  # u16 at byte offset 2

    if (flags & 0x1) or (flags & 0x4):
        # The swap only affects the next call: the local our_front/our_back/our_colour_front/our_colour_back
        # keep their pre-swap values for this call's rendering.
        state.front_buffer, state.back_buffer = state.back_buffer, state.front_buffer
        state.colour_front_buffer, state.colour_back_buffer = state.colour_back_buffer, state.colour_front_buffer

    data_pos = 6  # offset of the first opcode
    end_of_data = flen

    storage_count = read_le_u16(frame, 4)  # u16 at byte offset 4
    storage_ptr = 0
    while storage_count:
        val = read_le_u16(frame, data_pos)
        # sign extend as int16 then store as bytes (we just keep it as int16 semantics in storage)
        struct.pack_into('<h', state.storage, storage_ptr, struct.unpack('<h', struct.pack('<H', val))[0])
        storage_ptr += 2
        data_pos += 2
        storage_count -= 1

    curr_block_id = 0
    block_limit = 4000
    pass1 = True

    while pass1 or curr_block_id < block_limit:
        assert data_pos < end_of_data, f"data_pos={data_pos} end={end_of_data} curr_block_id={curr_block_id}"

        if curr_block_id >= block_limit:
            assert pass1
            assert curr_block_id == block_limit
            our_front = our_colour_front
            our_back = our_colour_back
            block_limit = 6000
            pass1 = False

        case_type = frame[data_pos] & 0x7

        if case_type in (0,1,2,3):
            info = read_le_u32(frame, data_pos)
            modifier = (info << 5) & 0x7F00
            src_block = (info >> 9) & 0x7FFE
            data_pos += 3

            offset_src = get_odd_pixel_offset(src_block)
            offset_dest = block_id_to_offset[curr_block_id]

            for i in range(4):
                dest = offset_dest + (320*i)
                src = offset_src
                if case_type == 0: src += (640*i)
                elif case_type == 1: src += (640*i) + 6
                elif case_type == 2: src += (640*(3-i))
                elif case_type == 3: src += (640*(3-i)) + 6
                for j in range(4):
                    our_front[dest] = modify_lookup[modifier + our_back[src]]
                    dest += 1
                    if case_type in (0,2): src += 2
                    else: src -= 2

            state.block_status[block_status_lookup[curr_block_id]] = 0
            curr_block_id += 1

        elif case_type == 4:
            count = (frame[data_pos] >> 3) & 0x1F
            data_pos += 1
            for j in range(count+1):
                offset = block_status_lookup[curr_block_id]
                if state.block_status[offset] & 2:
                    state.block_status[offset] = 3
                else:
                    state.block_status[offset] = state.curr_status
                    offset_both = block_id_to_offset[curr_block_id]
                    for i in range(0, 320*4, 320):
                        our_front[offset_both+i:offset_both+i+4] = our_back[offset_both+i:offset_both+i+4]
                curr_block_id += 1

        elif case_type == 5:
            info = frame[data_pos] >> 4
            input_val = struct.unpack('<h', state.storage[info*2:info*2+2])[0]
            count = 0
            if (frame[data_pos] >> 3) & 1:
                data_pos += 1
                count = frame[data_pos] + 1
                data_pos += 1
            else:
                data_pos += 1
            for j in range(count+1):
                offset_dest = block_id_to_offset[curr_block_id]
                offset_src = offset_dest + input_val
                for i in range(0, 320*4, 320):
                    our_front[offset_dest+i:offset_dest+i+4] = our_back[offset_src+i:offset_src+i+4]
                state.block_status[block_status_lookup[curr_block_id]] = 0
                curr_block_id += 1

        elif case_type == 6:
            data_pos += 1
            offset_dest = block_id_to_offset[curr_block_id]
            for i in range(0, 320*4, 320):
                our_front[offset_dest+i:offset_dest+i+4] = frame[data_pos:data_pos+4]
                data_pos += 4
            state.block_status[block_status_lookup[curr_block_id]] = 0
            curr_block_id += 1

        elif case_type == 7:
            subtype = (frame[data_pos] & 0x78) >> 3
            if subtype == 12:
                info = read_le_u32(frame, data_pos)
                info &= 0xFFFFFF
                info >>= 8
                if frame[data_pos] & 0x80:
                    data_pos += 3
                    info += 0x120
                else:
                    info &= 0xff
                    data_pos += 2
                    info += 0x20
                for j in range(info+1):
                    offset = block_status_lookup[curr_block_id]
                    if state.block_status[offset] & 2:
                        state.block_status[offset] = 3
                    else:
                        state.block_status[offset] = state.curr_status
                        offset_both = block_id_to_offset[curr_block_id]
                        for i in range(0, 320*4, 320):
                            our_front[offset_both+i:offset_both+i+4] = our_back[offset_both+i:offset_both+i+4]
                    curr_block_id += 1
            elif subtype <= 7:
                data_pos += 1
                modifier = frame[data_pos]
                data_pos += 1
                src_block16 = read_le_u16(frame, data_pos)
                data_pos += 2
                recursive = (modifier & 0x80) == 0x80
                modifier = (modifier << 8) & 0x7F00
                recursive_offset = 0
                if recursive:
                    recursive_offset = (src_block16 >> 8) >> 6
                src_block16 &= 0x3FFF
                offset_src = get_odd_pixel_offset(src_block16 * 2)
                offset_dest = block_id_to_offset[curr_block_id]

                for i in range(8):
                    src = offset_src
                    if subtype==0: src += 640*i
                    elif subtype==1: src += 640*i + 14
                    elif subtype==2: src += 640*(7-i)
                    elif subtype==3: src += 640*(7-i) + 14
                    elif subtype==4: src += 0 + 2*i
                    elif subtype==5: src += 0 + 2*(7-i)
                    elif subtype==6: src += 640*7 + 2*i
                    elif subtype==7: src += 640*7 + 2*(7-i)
                    dest = offset_dest + 320*i
                    for j in range(8):
                        our_front[dest] = modify_lookup[modifier + our_back[src]]
                        dest += 1
                        if subtype in (0,2): src += 2
                        elif subtype in (1,3): src -= 2
                        elif subtype in (4,5): src += 640
                        elif subtype in (6,7): src -= 640

                status_offset = block_status_lookup[curr_block_id]
                state.block_status[status_offset] = 0
                state.block_status[status_offset+1] = 0
                state.block_status[status_offset+80] = 0
                state.block_status[status_offset+81] = 0

                if not recursive:
                    curr_block_id += 4
                else:
                    temp_block_id = curr_block_id + recursive_offset
                    subsubtype = frame[data_pos] & 0x7
                    if subsubtype in (0,1,2,3):
                        info = read_le_u32(frame, data_pos)
                        data_pos += 3
                        rmod = (info << 5) & 0x7F00
                        rsrc_block = (info >> 9) & 0x7FFE
                        roff_src = get_odd_pixel_offset(rsrc_block)
                        roff_dest = block_id_to_offset[temp_block_id]
                        for i in range(4):
                            dest = roff_dest + 320*i
                            src = roff_src
                            if subsubtype==0: src += 640*i
                            elif subsubtype==1: src += 640*i + 6
                            elif subsubtype==2: src += 640*(3-i)
                            elif subsubtype==3: src += 640*(3-i) + 6
                            for j in range(4):
                                our_front[dest] = modify_lookup[rmod + our_back[src]]
                                dest += 1
                                if subsubtype in (0,2): src += 2
                                else: src -= 2
                    elif subsubtype == 4:
                        data_pos += 1
                        roff2 = block_status_lookup[temp_block_id]
                        state.block_status[roff2] |= state.curr_status
                        roff_both = block_id_to_offset[temp_block_id]
                        for i in range(0, 320*4, 320):
                            our_front[roff_both+i:roff_both+i+4] = our_back[roff_both+i:roff_both+i+4]
                    elif subsubtype == 5:
                        info = (frame[data_pos] >> 3) & 0x1f
                        data_pos += 1
                        roff2 = struct.unpack('<h', state.storage[info:info+2])[0]
                        roff_dest = block_id_to_offset[temp_block_id]
                        roff_src = roff_dest + roff2
                        for i in range(0, 320*4, 320):
                            our_front[roff_dest+i:roff_dest+i+4] = our_back[roff_src+i:roff_src+i+4]
                    elif subsubtype == 6:
                        data_pos += 1
                        roff_dest = block_id_to_offset[temp_block_id]
                        for i in range(0, 320*4, 320):
                            our_front[roff_dest+i:roff_dest+i+4] = frame[data_pos:data_pos+4]
                            data_pos += 4
                    else:
                        raise Exception("unhandled subsubtype 7")

                    if subsubtype not in (4,5):
                        state.block_status[block_status_lookup[temp_block_id]] = 0
                        temp_block_id += 1

                    curr_block_id += 4
            elif subtype == 15:
                # Raw 2x2 write: an absolute byte offset and two 16-bit rows are read straight from the bitstream.
                # This opcode does not advance curr_block_id. It does not occur in any known file.
                data_pos += 1
                offset = read_le_u16(frame, data_pos)
                data_pos += 2
                row0 = frame[data_pos:data_pos+2]
                data_pos += 2
                row1 = frame[data_pos:data_pos+2]
                data_pos += 2
                our_front[offset:offset+2] = row0
                our_front[offset+320:offset+322] = row1
            else:
                raise Exception(f"FVF: unknown block type {subtype} in video frame (case 7 subtypes 8-11, 13 and 14 are undefined)")

        elif case_type == 12:
            info = read_le_u32(frame, data_pos)
            info &= 0xFFFFFF
            info >>= 8
            if frame[data_pos] & 0x80:
                data_pos += 3
                info += 0x120
            else:
                info &= 0xff
                data_pos += 2
                info += 0x20
            for j in range(info+1):
                offset = block_status_lookup[curr_block_id]
                if state.block_status[offset] & 2:
                    state.block_status[offset] = 3
                else:
                    state.block_status[offset] = state.curr_status
                    offset_both = block_id_to_offset[curr_block_id]
                    for i in range(0, 320*4, 320):
                        our_front[offset_both+i:offset_both+i+4] = our_back[offset_both+i:offset_both+i+4]
                curr_block_id += 1

        elif case_type == 15:
            raise Exception("case 15 untested/unused")
        else:
            raise Exception(f"unknown case_type {case_type}")

    if not pass1:
        in_offset = 4000
        out_offset = 0
        for i in range(25):
            for j in range(40):
                x = (state.block_status[in_offset] & state.block_status[in_offset+40]) << 2
                state.block_status[out_offset] |= x
                state.block_status[out_offset+1] |= x
                state.block_status[out_offset+80] |= x
                state.block_status[out_offset+81] |= x
                out_offset += 2
                in_offset += 1
            out_offset += 80
            in_offset += 40

    return flags

# ---------------------------------------------------------------- full frame decode and RGB output

def decode_video_frame(state, frame, flen):
    flags = read_le_u16(frame, 2)
    if flags & 0x8:
        # Skip frame: nothing is decoded, but curr_status is reset. No known file contains one.
        state.curr_status = 2
        return None  # skip frame

    was_front = state.front_buffer
    was_colour_front = state.colour_front_buffer

    decode_video_frame_data(state, frame, flen)

    if flags & 0x1:
        was_front = state.front_buffer
        was_colour_front = state.colour_front_buffer
        for i in range(15):
            decode_video_frame_data(state, frame, flen)

    # Write RGB24 into the persistent output buffer, updating only the blocks whose block_status is not 0xf.
    target = state.rgb_output
    source = was_front
    source_colour = was_colour_front
    src_idx = 0
    src_col_idx = 0
    tgt_idx = 0
    blk_i = 0

    for y in range(HEIGHT // 4):
        for x in range(WIDTH // 4):
            if state.block_status[blk_i] != 0xf:
                targ = tgt_idx
                for j in range(4):
                    srcdata = src_idx + (j * 320)
                    src_col_ptr = src_col_idx
                    if j > 1:
                        src_col_ptr += 320

                    # The colour plane stores the red half at +0 and the blue half at +160, so the output
                    # pixel is [colour1 (red), luma (green), colour2 (blue)].
                    colour1 = source_colour[src_col_ptr + 0]
                    colour2 = source_colour[src_col_ptr + 160]
                    target[targ+0] = colour1
                    target[targ+1] = source[srcdata]; srcdata += 1
                    target[targ+2] = colour2
                    target[targ+3] = colour1
                    target[targ+4] = source[srcdata]; srcdata += 1
                    target[targ+5] = colour2
                    src_col_ptr += 1

                    colour1 = source_colour[src_col_ptr + 0]
                    colour2 = source_colour[src_col_ptr + 160]
                    target[targ+6] = colour1
                    target[targ+7] = source[srcdata]; srcdata += 1
                    target[targ+8] = colour2
                    target[targ+9] = colour1
                    target[targ+10] = source[srcdata]; srcdata += 1
                    target[targ+11] = colour2

                    targ += 320 * 3
            blk_i += 1
            src_idx += 4
            tgt_idx += 4 * 3
            src_col_idx += 2
        src_idx += 320 * 3
        tgt_idx += 320 * 3 * 3
        src_col_idx += 160 * 3

    state.curr_status = 2
    if flags & 0x4:
        state.curr_status = 3

    return target
