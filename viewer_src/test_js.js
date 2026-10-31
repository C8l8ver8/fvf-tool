// SPDX-License-Identifier: CC0-1.0
// Frame-by-frame check of fvf_decoder.js against reference hashes made with decoder.py (see mkref.py).
// usage: python mkref.py a.fvf b.fvf ...      (writes ref/<name>.json)
//        node test_js.js a.fvf b.fvf ...      (prints IDENTICAL or MISMATCH for every file)
const fs = require('fs'), path = require('path'), crypto = require('crypto'), FVF = require('./fvf_decoder.js');
const md5 = b => crypto.createHash('md5').update(b).digest('hex');
let bad = 0;
for (const p of process.argv.slice(2)) {
  const name = path.basename(p), refp = path.join(__dirname, 'ref', name + '.json');
  if (!fs.existsSync(refp)) { console.log(name, 'no reference yet (run mkref.py)'); continue; }
  const ref = JSON.parse(fs.readFileSync(refp));
  if (!Array.isArray(ref)) { console.log(name, 'python decoder error:', ref.error); continue; }
  const f = new FVF.FVFFile(new Uint8Array(fs.readFileSync(p))), recs = f.readAll(), dec = new FVF.Decoder();
  let first = -1, n = 0;
  for (let i = 0; i < recs.length; i++) {
    const r = recs[i]; let rgb = null, err = null;
    try { rgb = r.video ? dec.decodeFrame(r.video, r.videoLength) : null; } catch (e) { err = e.message; }
    const fl = r.video ? (r.video[2] | (r.video[3] << 8)) : null;
    const got = [r.videoLength, r.audio ? r.audio.length : null, fl, rgb ? md5(rgb) : null, (r.audio && r.audio.length) ? md5(r.audio) : null];
    if (err !== null || !got.every((v, k) => v === ref[i][k])) { if (first < 0) { first = i; console.log('  first mismatch at frame', i, err || ''); } n++; }
  }
  console.log(name, recs.length, 'frames', n === 0 ? 'IDENTICAL' : 'MISMATCH x' + n);
  if (n) bad++;
}
process.exit(bad ? 1 : 0);
