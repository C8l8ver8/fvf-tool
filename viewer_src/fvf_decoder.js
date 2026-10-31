/* fvf_decoder.js -- JavaScript port of fvflib/decoder.py (FVF video, "Star Trek: The Next Generation - A Final Unity").
 * Works in Node and in a browser. Its output is identical to decoder.py (compare with mkref.py and test_js.js).
 * SPDX-License-Identifier: CC0-1.0 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.FVF = factory();
})(typeof self !== 'undefined' ? self : this, function () {
'use strict';
const WIDTH = 320, HEIGHT = 200;

// ---------------------------------------------------------------- tables
const table1_x = [0, 4, 0, 4, 8, 12, 8, 12, 0, 4, 0, 4, 8, 12, 8, 12];
const table1_y = [0, 0, 4, 4, 0, 0, 4, 4, 8, 8, 12, 12, 8, 8, 12, 12];
const table2_x = [0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3];
const table2_y = [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3];

const block_id_to_offset = new Int32Array(6000);
const block_status_lookup = new Int32Array(6000);
const modify_lookup = new Uint8Array(32768);
const block_lookup = new Int32Array(16000);
(function buildTables() {
  let off = 0;
  function sub1(counter, width, size, count, base, halved) {
    let n = Math.floor(width / size); if (halved) n = Math.floor(n / 2);
    for (let i = 0; i < n; i++) for (let j = 0; j < count; j++)
      block_id_to_offset[off++] = base + i * size + table1_x[j] + (counter + table1_y[j]) * width;
  }
  for (let c = 0; c < 192; c += 16) sub1(c, 320, 16, 16, 0, false);
  sub1(192, 320, 16, 8, 0, false);
  for (let c = 0; c < 96; c += 8) sub1(c, 320, 16, 8, 0, true);
  sub1(96, 320, 8, 2, 0, true);
  for (let c = 0; c < 96; c += 8) sub1(c, 320, 16, 8, 160, true);
  sub1(96, 320, 8, 2, 160, true);
  off = 0;
  function sub2(counter, width, size, count, base, halved) {
    let n = Math.floor(width / (size * 4)); if (halved) n = Math.floor(n / 2);
    for (let i = 0; i < n; i++) for (let j = 0; j < count; j++)
      block_status_lookup[off++] = base + i * size + table2_x[j] + Math.floor(((table2_y[j] + (counter >> 2)) * width) / 4);
  }
  for (let c = 0; c < 192; c += 16) sub2(c, 320, 4, 16, 0, false);
  sub2(192, 320, 4, 8, 0, false);
  for (let c = 200; c < 296; c += 8) sub2(c, 320, 4, 8, 0, true);
  sub2(296, 320, 2, 2, 0, true);
  for (let c = 200; c < 296; c += 8) sub2(c, 320, 4, 8, 40, true);
  sub2(296, 320, 2, 2, 40, true);
  let a = 0;
  for (let i = -64; i < 64; i++) for (let b = 0; b < 256; b++) {
    let x = 4 * i + ((3 * b + 2) >> 2);
    modify_lookup[a++] = x < 0 ? 0 : x > 255 ? 255 : x;
  }
  for (let x = 0; x < 320; x += 2) for (let y = 0; y < 200; y += 2) block_lookup[(x + 160 * y) >> 1] = x + 320 * y;
})();

class FVFError extends Error {}
function need(cond, msg) { if (!cond) throw new FVFError(msg); }

// ---------------------------------------------------------------- container
class FVFFile {
  constructor(buf) {
    this.data = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
    this.dv = new DataView(this.data.buffer, this.data.byteOffset, this.data.byteLength);
    this.parseHeader();
  }
  u16(p) { need(p + 2 <= this.data.length, 'read past end of file at ' + p); return this.dv.getUint16(p, true); }
  u32(p) { need(p + 4 <= this.data.length, 'read past end of file at ' + p); return this.dv.getUint32(p, true); }
  parseHeader() {
    const d = this.data;
    need(d.length >= 2048 && d[0] === 0x46 && d[1] === 0x56 && d[2] === 0x46 && d[3] === 0x20, 'bad magic (not an FVF file)');
    this.dataOffset = this.u32(16);
    this.headerOffset = this.u32(24);
    this.audioHeaderOffset = this.u32(28);
    let p = this.headerOffset;
    need(this.u16(p) === 40, 'header: expected 40'); p += 2;
    need(this.u16(p) === 1, 'header: expected 1'); p += 2;
    this.bpp = this.u16(p); p += 2;
    this.width = this.u16(p); p += 2;
    this.height = this.u16(p); p += 2;
    this.usPerFrame = this.u32(p); p += 4;
    need(this.bpp === 16 && this.width === 320 && this.height === 200, 'header: unsupported size/depth');
    need(this.usPerFrame === 66666, 'header: unexpected frame time ' + this.usPerFrame);
    need(this.u32(p) === 0, 'header: padding'); p += 4;
    this.frameCount = this.u32(p); p += 4;
    need(this.u32(p) === 0, 'header: padding'); p += 4;
    need(this.u16(p) === 0x18, 'header: expected 0x18'); p += 2;
    this.paletteEntryCount = d[p++];
    need(this.paletteEntryCount === 112, 'header: palette count');
    this.headerPalCount = d[p++];
    need(this.headerPalCount === 15, 'header: header palette count');
    this.headerPalOffset = this.u32(p); p += 4;
    let a = this.audioHeaderOffset;
    need(this.u16(a) === 0, 'audio header'); a += 2;
    need(this.u16(a) === 1, 'audio header'); a += 2;
    this.audioChannels = this.u16(a); a += 2;
    this.audioSampleSize = this.u16(a); a += 2;
    this.audioSampleRate = this.u16(a); a += 2;
    need(this.audioChannels === 1 && this.audioSampleSize === 8 && this.audioSampleRate === 22050, 'audio: only mono 8-bit 22050 Hz is supported');
    this.curPos = this.dataOffset; this.blockRemFrames = 0; this.blockRemSize = 0;
    this.blockCount = 0;
  }
  readNextBlock() {
    let p = this.curPos;
    need(this.u16(p) === 0x10, 'block header size not 0x10 at ' + p); p += 2;
    const fc = this.u16(p); p += 2;
    this.prevBlockSize = this.u32(p); p += 4;
    this.blockSize = this.u32(p); p += 4;
    this.nextBlockSize = this.u32(p); p += 4;
    this.blockRemFrames = fc; this.blockRemSize = this.blockSize - 0x10; this.curPos = p;
    this.blockCount++;
  }
  /** -> {video:Uint8Array|null, videoLength, audio:Uint8Array|null, hasPalette, frameLength, block} */
  nextFrame() {
    if (this.blockRemFrames === 0) { this.curPos += this.blockRemSize; this.readNextBlock(); }
    const fs = this.curPos; let p = fs;
    need(this.u16(p) === 0x18, 'frame header size not 0x18'); p += 2;
    const frameLength = this.u16(p); p += 2;
    need(frameLength <= this.blockRemSize, 'frame longer than its block');
    p += 2;
    let videoOffset = this.u32(p); p += 4;
    let paletteOffset = this.u32(p); p += 4;
    let audioOffset = this.u32(p); p += 4;
    let sofar = 0x18, video = null, videoLength = 0, audio = null, hasPalette = false;
    if (videoOffset !== 0) {
      need(videoOffset - sofar === 0, 'video_offset mismatch');
      videoLength = this.u16(fs + sofar); sofar += 2;
      need(fs + sofar + videoLength <= this.data.length, 'video past end of file');
      video = this.data.subarray(fs + sofar, fs + sofar + videoLength); sofar += videoLength;
    }
    if (paletteOffset !== 0) { need(paletteOffset - sofar === 0, 'palette_offset mismatch'); hasPalette = true; sofar += this.paletteEntryCount * 3; }
    if (audioOffset !== 0) {
      need(audioOffset - sofar === 0, 'audio_offset mismatch');
      const al = this.u32(fs + sofar); sofar += 4;
      need(fs + sofar + al <= this.data.length, 'audio past end of file');
      audio = this.data.subarray(fs + sofar, fs + sofar + al); sofar += al;
    }
    need(sofar === frameLength, 'record size mismatch: read ' + sofar + ' of ' + frameLength);
    this.blockRemFrames--; this.blockRemSize -= frameLength; this.curPos = fs + frameLength;
    return { video, videoLength, audio, hasPalette, frameLength, block: this.blockCount - 1 };
  }
  /** parse every frame record (no pixel decoding). Returns an array of frame records. */
  readAll() {
    this.curPos = this.dataOffset; this.blockRemFrames = 0; this.blockRemSize = 0; this.blockCount = 0;
    this.readNextBlock();
    const out = [];
    for (let i = 0; i < this.frameCount; i++) out.push(this.nextFrame());
    return out;
  }
}

// ---------------------------------------------------------------- video decoder
class Decoder {
  constructor() {
    this.storageBuf = new ArrayBuffer(512);
    this.storageB = new Uint8Array(this.storageBuf);
    this.storageH = new Int16Array(this.storageBuf);
    this.currStatus = 2;
    this.blockStatus = new Uint8Array(6000);
    this.A = new Uint8Array(64000).fill(0xAB); this.B = new Uint8Array(64000).fill(0xAB);
    this.CA = new Uint8Array(32000).fill(0xAB); this.CB = new Uint8Array(32000).fill(0xAB);
    this.front = this.A; this.back = this.B; this.cfront = this.CA; this.cback = this.CB;
    this.rgb = new Uint8Array(WIDTH * HEIGHT * 3);
  }
  /** copy of the complete decoder state (for seeking) */
  snapshot() {
    return { storage: this.storageB.slice(), currStatus: this.currStatus, blockStatus: this.blockStatus.slice(),
      A: this.A.slice(), B: this.B.slice(), CA: this.CA.slice(), CB: this.CB.slice(),
      frontIsA: this.front === this.A, cfrontIsA: this.cfront === this.CA, rgb: this.rgb.slice() };
  }
  restore(s) {
    this.storageB.set(s.storage); this.currStatus = s.currStatus; this.blockStatus.set(s.blockStatus);
    this.A.set(s.A); this.B.set(s.B); this.CA.set(s.CA); this.CB.set(s.CB);
    this.front = s.frontIsA ? this.A : this.B; this.back = s.frontIsA ? this.B : this.A;
    this.cfront = s.cfrontIsA ? this.CA : this.CB; this.cback = s.cfrontIsA ? this.CB : this.CA;
    this.rgb.set(s.rgb);
  }
  /** decode one video bitstream. Returns the (persistent) RGB24 buffer, or null for a "skip" frame. */
  decodeFrame(frame, flen) {
    if (flen === undefined) flen = frame.length;
    need(flen >= 6, 'video chunk too short');
    const flags = frame[2] | (frame[3] << 8);
    if (flags & 8) { this.currStatus = 2; return null; }
    let wasFront = this.front, wasCol = this.cfront;
    this._data(frame, flen);
    if (flags & 1) {
      wasFront = this.front; wasCol = this.cfront;
      for (let i = 0; i < 15; i++) this._data(frame, flen);
    }
    this._render(wasFront, wasCol);
    this.currStatus = (flags & 4) ? 3 : 2;
    return this.rgb;
  }
  _data(frame, flen) {
    let ourFront = this.front, ourBack = this.back, colFront = this.cfront, colBack = this.cback;
    const flags = frame[2] | (frame[3] << 8);
    if (flags & 5) {
      let t = this.front; this.front = this.back; this.back = t;
      t = this.cfront; this.cfront = this.cback; this.cback = t;
    }
    let pos = 6;
    const end = flen, bs = this.blockStatus, storageH = this.storageH, storageB = this.storageB;
    let storageCount = frame[4] | (frame[5] << 8);
    need(storageCount <= 256 && 6 + 2 * storageCount <= flen, 'bad storage count');
    for (let i = 0; i < storageCount; i++) { storageB[2 * i] = frame[pos]; storageB[2 * i + 1] = frame[pos + 1]; pos += 2; }
    let id = 0, limit = 4000, pass1 = true;
    const cs = this.currStatus;
    const u32 = (p) => { need(p + 3 <= flen, 'read past end of video chunk'); return (frame[p] | (frame[p + 1] << 8) | (frame[p + 2] << 16) | ((frame[p + 3] || 0) << 24)) >>> 0; };  // the 4th byte is unused by every caller, so a 3-byte opcode may end the chunk
    const rd = (p) => { need(p < flen, 'read past end of video chunk'); return frame[p]; };
    let dst = ourFront, src = ourBack;
    const copy4 = (dOff, sOff, from) => {   // 4x4 copy of `from` (back) into dst
      need(sOff >= 0 && sOff + 3 * 320 + 4 <= from.length, 'copy source out of range');
      for (let i = 0; i < 1280; i += 320) { const d = dOff + i, s = sOff + i; dst[d] = from[s]; dst[d + 1] = from[s + 1]; dst[d + 2] = from[s + 2]; dst[d + 3] = from[s + 3]; }
    };
    while (pass1 || id < limit) {
      need(pos < end, 'ran past end of video data (pos ' + pos + ', block ' + id + ')');
      if (id >= limit) {
        need(pass1 && id === limit, 'block count overrun');
        dst = colFront; src = colBack; limit = 6000; pass1 = false;
      }
      const ct = frame[pos] & 7;
      if (ct <= 3) {
        const info = u32(pos);
        const modifier = (info << 5) & 0x7F00;
        const srcBlock = (info >>> 9) & 0x7FFE;
        pos += 3;
        let so = block_lookup[srcBlock >> 1]; need(srcBlock >> 1 < 16000, 'source block out of range');
        const od = block_id_to_offset[id];
        for (let i = 0; i < 4; i++) {
          let d = od + 320 * i, s = so;
          if (ct === 0) s += 640 * i; else if (ct === 1) s += 640 * i + 6; else if (ct === 2) s += 640 * (3 - i); else s += 640 * (3 - i) + 6;
          for (let j = 0; j < 4; j++) {
            need(s >= 0 && s < src.length, 'transform source out of range');
            dst[d++] = modify_lookup[modifier + src[s]];
            if (ct === 0 || ct === 2) s += 2; else s -= 2;
          }
        }
        bs[block_status_lookup[id]] = 0; id++;
      } else if (ct === 4) {
        const count = (frame[pos] >> 3) & 0x1F; pos += 1;
        for (let j = 0; j <= count; j++) {
          const o = block_status_lookup[id];
          if (bs[o] & 2) bs[o] = 3;
          else { bs[o] = cs; copy4(block_id_to_offset[id], block_id_to_offset[id], src); }
          id++;
        }
      } else if (ct === 5) {
        const info = frame[pos] >> 4;
        const inputVal = storageH[info];
        let count = 0;
        if ((frame[pos] >> 3) & 1) { pos += 1; count = rd(pos) + 1; pos += 1; } else pos += 1;
        for (let j = 0; j <= count; j++) {
          const od = block_id_to_offset[id];
          copy4(od, od + inputVal, src);
          bs[block_status_lookup[id]] = 0; id++;
        }
      } else if (ct === 6) {
        pos += 1;
        need(pos + 16 <= flen, 'raw block past end of video chunk');
        const od = block_id_to_offset[id];
        for (let i = 0; i < 1280; i += 320) { const d = od + i; dst[d] = frame[pos]; dst[d + 1] = frame[pos + 1]; dst[d + 2] = frame[pos + 2]; dst[d + 3] = frame[pos + 3]; pos += 4; }
        bs[block_status_lookup[id]] = 0; id++;
      } else { // 7
        const subtype = (frame[pos] & 0x78) >> 3;
        if (subtype === 12) {
          let info = (u32(pos) & 0xFFFFFF) >>> 8;
          if (frame[pos] & 0x80) { pos += 3; info += 0x120; } else { info &= 0xFF; pos += 2; info += 0x20; }
          for (let j = 0; j <= info; j++) {
            const o = block_status_lookup[id];
            if (bs[o] & 2) bs[o] = 3;
            else { bs[o] = cs; copy4(block_id_to_offset[id], block_id_to_offset[id], src); }
            id++;
          }
        } else if (subtype <= 7) {
          pos += 1; let modifier = rd(pos); pos += 1;
          need(pos + 2 <= flen, 'truncated 8x8 opcode');
          let sb16 = frame[pos] | (frame[pos + 1] << 8); pos += 2;
          const recursive = (modifier & 0x80) === 0x80;
          modifier = (modifier << 8) & 0x7F00;
          let recOff = 0;
          if (recursive) recOff = (sb16 >> 8) >> 6;
          sb16 &= 0x3FFF;
          need(sb16 < 16000, 'source block out of range');
          const so = block_lookup[sb16];
          const od = block_id_to_offset[id];
          for (let i = 0; i < 8; i++) {
            let s = so;
            switch (subtype) {
              case 0: s += 640 * i; break; case 1: s += 640 * i + 14; break;
              case 2: s += 640 * (7 - i); break; case 3: s += 640 * (7 - i) + 14; break;
              case 4: s += 2 * i; break; case 5: s += 2 * (7 - i); break;
              case 6: s += 640 * 7 + 2 * i; break; default: s += 640 * 7 + 2 * (7 - i);
            }
            let d = od + 320 * i;
            for (let j = 0; j < 8; j++) {
              need(s >= 0 && s < src.length, 'transform source out of range');
              dst[d++] = modify_lookup[modifier + src[s]];
              if (subtype === 0 || subtype === 2) s += 2; else if (subtype === 1 || subtype === 3) s -= 2;
              else if (subtype === 4 || subtype === 5) s += 640; else s -= 640;
            }
          }
          const so2 = block_status_lookup[id];
          bs[so2] = 0; bs[so2 + 1] = 0; bs[so2 + 80] = 0; bs[so2 + 81] = 0;
          if (!recursive) id += 4;
          else {
            let tid = id + recOff;
            const sub = rd(pos) & 7;
            if (sub <= 3) {
              const info = u32(pos); pos += 3;
              const rmod = (info << 5) & 0x7F00, rsrc = (info >>> 9) & 0x7FFE;
              need((rsrc >> 1) < 16000, 'source block out of range');
              const rso = block_lookup[rsrc >> 1], rod = block_id_to_offset[tid];
              for (let i = 0; i < 4; i++) {
                let d = rod + 320 * i, s = rso;
                if (sub === 0) s += 640 * i; else if (sub === 1) s += 640 * i + 6; else if (sub === 2) s += 640 * (3 - i); else s += 640 * (3 - i) + 6;
                for (let j = 0; j < 4; j++) {
                  need(s >= 0 && s < src.length, 'transform source out of range');
                  dst[d++] = modify_lookup[rmod + src[s]];
                  if (sub === 0 || sub === 2) s += 2; else s -= 2;
                }
              }
            } else if (sub === 4) {
              pos += 1;
              bs[block_status_lookup[tid]] |= cs;
              copy4(block_id_to_offset[tid], block_id_to_offset[tid], src);
            } else if (sub === 5) {
              const info = (frame[pos] >> 3) & 0x1F; pos += 1;
              const r2 = (storageB[info] | (storageB[info + 1] << 8)) << 16 >> 16;   // int16 at BYTE offset `info`, as decoder.py does
              const rd_ = block_id_to_offset[tid];
              copy4(rd_, rd_ + r2, src);
            } else if (sub === 6) {
              pos += 1; need(pos + 16 <= flen, 'raw block past end of video chunk');
              const rd_ = block_id_to_offset[tid];
              for (let i = 0; i < 1280; i += 320) { const d = rd_ + i; dst[d] = frame[pos]; dst[d + 1] = frame[pos + 1]; dst[d + 2] = frame[pos + 2]; dst[d + 3] = frame[pos + 3]; pos += 4; }
            } else throw new FVFError('unhandled subsubtype 7');
            if (sub !== 4 && sub !== 5) { bs[block_status_lookup[tid]] = 0; tid++; }
            id += 4;
          }
        } else if (subtype === 15) {
          pos += 1; need(pos + 6 <= flen, 'truncated raw 2x2 opcode');
          const off = frame[pos] | (frame[pos + 1] << 8); pos += 2;
          need(off + 322 <= dst.length, 'raw 2x2 out of range');
          dst[off] = frame[pos]; dst[off + 1] = frame[pos + 1]; pos += 2;
          dst[off + 320] = frame[pos]; dst[off + 321] = frame[pos + 1]; pos += 2;
        } else throw new FVFError('unknown block type ' + subtype);
      }
    }
    // merge colour block status into the luma status
    let inO = 4000, outO = 0;
    for (let i = 0; i < 25; i++) {
      for (let j = 0; j < 40; j++) {
        const x = (bs[inO] & bs[inO + 40]) << 2;
        bs[outO] |= x; bs[outO + 1] |= x; bs[outO + 80] |= x; bs[outO + 81] |= x;
        outO += 2; inO += 1;
      }
      outO += 80; inO += 40;
    }
  }
  _render(source, col) {
    const t = this.rgb, bs = this.blockStatus;
    let srcIdx = 0, colIdx = 0, tgtIdx = 0, blk = 0;
    for (let y = 0; y < 50; y++) {
      for (let x = 0; x < 80; x++) {
        if (bs[blk] !== 0xF) {
          let targ = tgtIdx;
          for (let j = 0; j < 4; j++) {
            let sd = srcIdx + j * 320, cp = colIdx + (j > 1 ? 320 : 0);
            let c1 = col[cp], c2 = col[cp + 160];
            t[targ] = c1; t[targ + 1] = source[sd++]; t[targ + 2] = c2;
            t[targ + 3] = c1; t[targ + 4] = source[sd++]; t[targ + 5] = c2;
            cp++;
            c1 = col[cp]; c2 = col[cp + 160];
            t[targ + 6] = c1; t[targ + 7] = source[sd++]; t[targ + 8] = c2;
            t[targ + 9] = c1; t[targ + 10] = source[sd++]; t[targ + 11] = c2;
            targ += 960;
          }
        }
        blk++; srcIdx += 4; tgtIdx += 12; colIdx += 2;
      }
      srcIdx += 960; tgtIdx += 2880; colIdx += 480;
    }
  }
}

return { WIDTH, HEIGHT, FVFFile, Decoder, FVFError, tables: { block_id_to_offset, block_status_lookup, modify_lookup, block_lookup } };
});
