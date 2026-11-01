# fvf – decode, re-encode and repack the videos of *Star Trek: The Next Generation – A Final Unity*

`fvf.py` is a command-line tool for the **FVF** video format (Total Multimedia / SoftVideo) used by *A Final Unity* (MS-DOS, 1995).

| Command     | What it does |
|-------------|--------------|
| `info`      | Prints the size, frame count, duration and audio format of a video. |
| `decode`    | Converts a game video (`.fvf`) to an `.mp4` movie, a `.wav` sound file and, optionally, a folder of PNG pictures. |
| `encode`    | Builds a new `.fvf` from an original one. The new file has exactly the same size and identical audio. |
| `patch-zip` | Writes new `.fvf` files into a copy of the game's `STTNG.ZIP`, leaving every other byte unchanged. |
| `replace`   | `encode` followed by `patch-zip`. |
| `view`      | Opens the viewer in your web browser: play a video, or compare an original and a re-encode side by side. |
| `check`     | Scans video files and lists anything unusual (structure, codec features used, decoding errors). |
| `compare`   | Prints a report on how a re-encoded file differs from its original. |

Every command accepts `-v` / `--verbose` after the command name (for example `python fvf.py info -v FILE.fvf`) to print decoder header and block details.

**Tested in the game:** the re-encoded `teaser`, `title`, `warpout1`, `warnshot`, `labarriv`, `shiptour` and `tracbloc` videos play in DOSBox-X. `chodesc` was verified with the tool's own decoder and the viewer, but not in the game. The other videos were not re-encoded.

**Quality.** Re-encoding is lossy: the original files are already compressed, and the new file must have exactly the original size. Measured with `compare` (MAD = mean absolute difference per pixel and colour, on a 0–255 scale):

| Video | Mean MAD | Worst frame MAD |
|-------|---------:|----------------:|
| warnshot | 0.005 | 0.020 |
| warpout1 | 0.006 | 0.052 |
| chodesc | 0.018 | 0.134 |
| shiptour | 0.032 | 0.417 |
| labarriv | 0.068 | 0.165 |
| tracbloc | 0.099 | 0.290 |

In the viewer, the differences were not visible at normal size for any of these videos; a few single pixels could be seen when zoomed in on some frames of `labarriv` and `tracbloc`. Results depend on the footage, and the settings in section 5.3 matter for long videos.

---

## Contents

1. [What you need](#1-what-you-need)
2. [Set-up (Windows)](#2-set-up-windows)
3. [Get the tool](#3-get-the-tool)
4. [Get the video files out of the game](#4-get-the-video-files-out-of-the-game)
5. [Commands](#5-commands)
6. [Why `patch-zip` exists](#6-why-patch-zip-exists)
7. [The whole job from start to finish](#7-the-whole-job-from-start-to-finish)
8. [Troubleshooting](#8-troubleshooting)
9. [macOS and Linux](#9-macos-and-linux)
10. [Notes for developers](#10-notes-for-developers)
11. [Acknowledgements and licence](#11-acknowledgements-and-licence)

---

## 1. What you need

| What | Needed for | Notes |
|------|------------|-------|
| **Python 3.9 or newer** | everything | [python.org/downloads](https://www.python.org/downloads/) |
| **numpy** | everything except the viewer | `python -m pip install numpy` |
| **ffmpeg** | `decode` to `.mp4` only | Without it use `--no-mp4`; you still get the `.wav`. |
| **Pillow** | `decode --png` only | `python -m pip install pillow` |
| **A web browser** (Chrome, Edge, Firefox or Safari) | the viewer | The viewer is one HTML file. It works offline and uploads nothing. |
| **The game's `STTNG.ZIP`** | `patch-zip`, `replace` and extracting the videos | About 650 MB (623 MiB). Keep an untouched backup. |

**Encoding needs time and CPU, not extra software.** It searches for the best way to store every picture.

- Measured on a Ryzen 7 7800X3D (16 threads): roughly 5–10 seconds per frame, for example about 40 minutes for a 286-frame video, about 78 minutes for 438 frames and about 3.5 hours for 2,265 frames.
- By default the tool uses half of your CPU threads (at most 8). More workers use more memory; if the PC runs out of memory, use fewer `--jobs`.
- You can stop with **Ctrl+C**. Running the same command again reuses the saved first-pass result (section 5.3).
- Do not let the PC go to sleep while encoding.
- `patch-zip` makes a full copy of `STTNG.ZIP`, so you need that much free disk space as well.

---

## 2. Set-up (Windows)

### 2.1 Open PowerShell

Press the **Windows key**, type `PowerShell` and press **Enter**. A window opens with a line such as `PS C:\Users\YourName>`. Commands in this README are typed after the `>` and confirmed with **Enter**. You can paste with a right-click.

### 2.2 Python

```
python --version
```

Any version from 3.9 up is fine. If you see `Python was not found` or `'python' is not recognized`, install Python from <https://www.python.org/downloads/>. **Tick "Add python.exe to PATH" on the first screen of the installer**, then close PowerShell, open a new one and run `python --version` again. (If `python` still fails, try `py --version` and use `py` in place of `python` everywhere.)

### 2.3 numpy and Pillow

```
python -m pip install numpy pillow
```

Pillow is only needed for `decode --png`.

### 2.4 ffmpeg (only for `.mp4` output)

```
winget install Gyan.FFmpeg
```

Close PowerShell, open a new one and check with `ffmpeg -version`. If `winget` is missing, download a build from <https://www.gyan.dev/ffmpeg/builds/> and add its `bin` folder to your PATH, or always use `--no-mp4`.

---

## 3. Get the tool

Copy the tool to a folder with a short path, for example `C:\FVF\FVF Tool`. The `fvflib` folder must stay next to `fvf.py`:

```
fvf.py
patch_zip.py
check_fvf.py
fvf_viewer.html        (the viewer; see below)
fvflib\                (decoder.py, encoder2.py, encoder3.py, encoder4.py, encoder7.py, recontainer_lite.py)
viewer_src\            (source and tests of the viewer; not needed to run the tool)
```

`fvf_viewer.html` is attached to each release. If you only have the source, create it with `python viewer_src/build_viewer.py`.

Open PowerShell in that folder (in File Explorer click the address bar, type `powershell`, press **Enter**), or type `Set-Location "C:\FVF\FVF Tool"`. Check that it works:

```
python fvf.py --help
```

Every command in this README is typed in that window. Put paths in **double quotes** when they contain spaces. A path starting `.\` is relative to the current folder.

---

## 4. Get the video files out of the game

The game keeps its videos in one archive, **`STTNG.ZIP`**, in the game folder.

1. **Back it up.** Copy `STTNG.ZIP` somewhere safe and never change that copy.
2. **Extract the videos.** Unzipping only reads the archive, so any program works: in Windows, right-click → *Extract All…*; in 7-Zip, open the archive and drag out the `.fvf` files you want.

Never put files back into `STTNG.ZIP` with an archive program; use `patch-zip` or `replace` (section 6).

The examples below use this layout. Replace the paths with your own.

```
C:\FVF\
    FVF Tool\             the tool
    Original\STTNG.ZIP    your untouched backup
    Videos\               the extracted .fvf files
    Output\               everything the tool creates
```

Create the output folder first (`decode` and `patch-zip` do not create it): `New-Item -ItemType Directory -Force "C:\FVF\Output"`.

---

## 5. Commands

### 5.1 `info`

```
python fvf.py info "C:\FVF\Videos\warnshot.fvf"
```

```
C:\FVF\Videos\warnshot.fvf: 608256 bytes, 60 frames, 11 blocks, 4.0s @15fps, audio 1ch 8bit 22050Hz
```

A replacement file must have exactly this size.

### 5.2 `decode`

```
python fvf.py decode "C:\FVF\Videos\warnshot.fvf" "C:\FVF\Output\warnshot"
```

The second argument is a base name without extension. It creates `warnshot.mp4` (320×200, 15 frames/s, with sound) and `warnshot.wav` (8-bit mono, 22050 Hz).

| Option | Meaning |
|--------|---------|
| `--no-mp4` | Do not run ffmpeg; write only the `.wav`. |
| `--png FOLDER` | Also save every picture as `00000.png`, `00001.png`, … (needs Pillow; thousands of files for a long video). |
| `--frames N` | Decode only the first N frames. |

### 5.3 `encode`

```
python fvf.py encode "C:\FVF\Videos\warnshot.fvf" "C:\FVF\Output\warnshot_new.fvf" --work "C:\FVF\Output\work_warnshot"
```

The source must be an **original** game `.fvf`. The new file is written to the second path; use a different name from the original.

What happens:

1. The tool decodes the source, detects scene cuts and runs **pass 1**: it codes every frame at one common quality/size trade-off and records how many bytes each frame gets. The result is saved in the work folder (`need_slope.csv` and `need_slope_mad.csv`).
2. **Pass 2** codes every frame again with a byte budget derived from pass 1, using the real decoder in a closed loop.
3. The frames are packed into blocks so that the file is **exactly** the original size.
4. The tool decodes the new file with `decoder.py` and compares it with what the encoder intended and with the original audio.

Progress is printed and also written to `warnshot_new.fvf.log`. A line such as `file does not fit (short by 1234 B); lowering scale by 0.012 and retrying` is normal: the tool repeats pass 2 with a slightly lower budget until the file fits.

It ends with:

```
wrote C:\FVF\Output\warnshot_new.fvf 608256 bytes (target 608256 )
VERIFIED: decoder matches encoder model, audio identical; mean MAD ...
```

**If you do not see `VERIFIED`, do not use the file.** If the work folder already contains a pass-1 result, it is reused; pass 2 always runs again.

Common options:

| Option | Meaning |
|--------|---------|
| `--work FOLDER` | Folder for the pass-1 result (default: `OUT.work` next to the output). Use a separate folder for every video. |
| `--jobs N` | Worker processes. Default: half of your CPU threads, at most 8. `--jobs 1` runs serially. |
| `--redo` | Ignore a saved pass-1 result. Use it after changing the source file, `--pw` or `--lam0`. |
| `--mode s10` | Default. Two passes, best quality. |
| `--mode fast` | One pass, faster, lower quality; it can fail with "does not fit". |
| `--scale S` | Starting budget scale (default 1.0). It is lowered automatically until the file fits. |

Advanced options (the defaults suit most videos):

| Option | Default | Meaning |
|--------|---------|---------|
| `--need slope\|lossless` | `slope` | Pass-1 method. `slope` codes every frame at one common multiplier; `lossless` measures the bytes each frame needs to be coded without error. |
| `--pw W` | 2.0 | Key-frame persistence weight: static blocks of key frames count 1+W times as much (0 = off). |
| `--lam0 L` | 4.0 | Starting multiplier of the slope pass (it adapts). |
| `--floor F` | 0 (off) | Every frame gets at least this fraction of its original size; other frames pay for it. |
| `--floor-min B` | 0 | With `--floor`: frames planned below B bytes (nearly static frames) are not raised. |
| `--floor-mad M` | 0.03 | With `--floor`: frames whose pass-1 error (MAD) is below M are not raised. |
| `--protect P` | 0.97 | With `--floor` and a pass-1 error file: frames pass 1 codes almost perfectly keep at least this fraction of their planned size. |
| `--mad-tilt T` | 0 (off) | Byte-neutral: weight frames by (pass-1 error / 75th percentile) ** T so that frames pass 1 codes badly get more bytes. |
| `--mad-tilt-min X`, `--mad-tilt-max Y` | 1.0, 1.6 | Smallest and largest weight `--mad-tilt` may apply (1.0 means no frame is reduced by the weighting). |
| `--frame-cap B` | 24000 | Largest video chunk pass 2 may spend on one frame (the largest video chunk in the original files is 18,036 B; 25 of the 27 never exceed 18,008). |
| `--key-boost R` | 0 (off) | Byte-neutral: each key frame takes this fraction of the budgets of the three frames after it. |
| `--no-autoscale` | autoscale on | By default the starting scale is lowered until the packed plan is predicted to fit. This switch disables that. |

For a video with a long stretch that comes out visibly worse than the rest, a good starting point is `--floor 0.95 --floor-min 2500 --mad-tilt 0.2 --mad-tilt-max 1.4`. Compare per-stretch error with `compare` (section 5.8), not only the mean.

### 5.4 `patch-zip`

```
python fvf.py patch-zip "C:\FVF\Original\STTNG.ZIP" "C:\FVF\Output\STTNG.ZIP" "warnshot.fvf=C:\FVF\Output\warnshot_new.fvf"
```

Take the original zip, write a patched copy, and in it replace the entry `warnshot.fvf` with the given file. Give one `name=file` pair per video. The name on the left must match the zip entry exactly; the output must be a different file from the input. The new file must have exactly the size of the entry it replaces. The command takes about a minute.

It prints one line per replaced entry (the CRC values and offset depend on your files) and a final line:

```
patched warnshot.fvf crc XXXXXXXX (was YYYYYYYY) at offset NNNNNNNNN
done -> C:\FVF\Output\STTNG.ZIP 653379061 bytes (source 653379061 )
```

The two sizes at the end must be equal. If the command fails, the partly written output file is deleted.

### 5.5 `replace`

```
python fvf.py replace "C:\FVF\Original\STTNG.ZIP" "C:\FVF\Output\STTNG.ZIP" "warnshot.fvf=C:\FVF\Videos\warnshot.fvf"
```

Encodes every listed video (the right side of each pair is the **original** `.fvf`) and then patches the zip. It accepts the encode options of section 5.3 except `--redo`. Here `--work` is the folder that receives the new files and the per-video work folders (default `OUT.work` next to the output zip). Encoding can take hours, so for a first attempt run `encode` and `patch-zip` separately and inspect the result in between.

### 5.6 `view`

```
python fvf.py view
```

Opens `fvf_viewer.html` in your browser (or double-click the file). Files are read by the browser and never uploaded.

**To watch a video:** click **Open .fvf or .zip…** in box **A** and pick a `.fvf` (or `STTNG.ZIP`, then choose a video inside it; only uncompressed zip entries can be opened). Wait while the page analyses the video, then press **▶ Play** or the space bar.

**To check a re-encode:** put the original in **A** and the new file in **B**.

| Part of the page | What it shows |
|------------------|---------------|
| **View** buttons (keys **1**–**5**) | A or B alone, side by side, difference (the **×** box amplifies small differences) and wipe. **F** flips between A and B at the same spot. |
| Transport | Play/pause, frame step (arrow keys), one second (Shift + arrow keys), first/last frame, frame number, speed, loop, mute. |
| **Timeline** | A graph over the whole video: average error, brightness bias, worst pixel, share of pixels off by more than 8, bytes per frame, or change from frame to frame. Red ticks are scene changes. |
| **flagged ◀ ▶** (keys **[** and **]**) | Jump to the previous/next frame whose graph value is above the dashed line. |
| **This frame** | Bytes, flags, MAD, per-colour bias, worst pixel and share of pixels off by more than 8 for the current frame. |
| **Files** | Size, frames, blocks and decoding errors, plus three checks: same file size, same frame count, identical audio. |
| **Worst frames** | The 40 frames with the highest MAD. |
| **Save PNG** | Saves the picture currently shown. |

### 5.7 `check`

```
python fvf.py check "C:\FVF\Videos"
```

Accepts folders and files. It reads headers, blocks, frame records and every video instruction, decodes the video and prints a table and a list of warnings. It never changes a file.

```
file                 bytes frames blocks vid-max    raw%    8x8%   mv%   pal  warn   dec
----------------------------------------------------------------------------------------
warnshot.fvf       608,256     60     11   18008   53.8%    5.0% 11.4%     3     0    ok
5tharriv.fvf     2,596,864    261     44   18008   73.3%    2.1%  5.0%     8     0    ok
```

`vid-max` is the largest video chunk in bytes; `raw%`, `8x8%` and `mv%` are the shares of raw blocks, 8×8 transforms and motion-vector copies; `pal` is the number of frames with a palette; `warn` counts warnings; `dec` says whether the whole video decoded. `0` warnings and `ok` is a clean result.

| Option | Meaning |
|--------|---------|
| `--report FILE` | Write a detailed text report for every file. |
| `--csv FILE` | Write one table row per file. |
| `--no-decode` | Skip the picture decode (fast; structure and opcode statistics only). |
| `--deep` | Decode a second time with the picture buffers pre-filled with a different value and report differences. (The encoder assumes the buffers start as `0xAB`.) |

### 5.8 `compare`

```
python fvf.py compare "C:\FVF\Videos\warnshot.fvf" "C:\FVF\Output\warnshot_new.fvf" --csv "C:\FVF\Output\warnshot_compare.csv"
```

The first file is the original, the second the new file. Shortened output:

```
ORIGINAL warnshot.fvf: 608,256 bytes, 60 frames, 11 blocks
NEW      warnshot_new.fvf: 608,256 bytes, 60 frames, 11 blocks

Structure
  OK   same file size (required by the game)
  OK   same number of frames
  OK   audio identical in every frame
  OK   file header identical (apart from the last-block offset, bytes 20-23)
  OK   header last-block offset matches the new file
  OK   all blocks are multiples of 2048 and at most 63488

Video bytes by opcode class (share of all video bytes)
  ...

Picture error (new vs original): mean MAD 0.668, median 0.649, 99th percentile 1.313, worst 1.342
Brightness bias (new - original, green): mean -0.214, worst -0.492 at frame 19

Worst 15 frames by MAD:
   frame    time     MAD   R bias   G bias    max    >8 %  orig B   new B
  ...

Result: structure OK
```

(This example is from a `--mode fast` encode, which has a higher error than the default mode.)

- Every **Structure** line must say `OK`; `FAIL` means the game will not accept the file.
- **Video bytes by opcode class** shows what each file is made of.
- **MAD** is the mean absolute difference per pixel and colour; **bias** shows whether the new picture is darker or lighter overall.
- `--thr N` sets the MAD above which frames are listed as flagged (default: the larger of 1.0 and the 99th percentile). `--csv FILE` writes one row per frame.

---

## 6. Why `patch-zip` exists

The game is sensitive to how `STTNG.ZIP` is laid out, and ordinary archive programs change it:

1. **Entry order.** The game finds its videos through the zip's internal order. Archive programs reorder entries when files are added or replaced.
2. **Compression.** In the original zip the `.fvf` entries are stored uncompressed. Most archive programs compress what they add.
3. **Size.** Each `.fvf` must have exactly the size of the entry it replaces; that is why `encode` pads the new file to the original size and `patch-zip` rejects any other size.

`patch-zip` copies the original zip and overwrites the data of the named entries in place, updating only their CRC32 fields. Every other byte, every offset and the order of all entries stay the same. Always start from the untouched original `STTNG.ZIP`.

---

## 7. The whole job from start to finish

1. Set up (sections 2–3), back up `STTNG.ZIP` and extract the videos you want (section 4).
2. *(Optional)* `decode` an original and watch it.
3. `encode` each video and wait for `VERIFIED`.
4. Check each result with `compare` and the viewer (original in A, new file in B, use the flagged-frame buttons).
5. `patch-zip` to build the new `STTNG.ZIP`.
6. In the game folder, rename the old `STTNG.ZIP` to `STTNG.ZIP.bak` and copy the new zip in as `STTNG.ZIP`.
7. Start the game and watch the videos.

To go back, restore the original `STTNG.ZIP`. The tool never modifies your original files.

---

## 8. Troubleshooting

| What you see | What to do |
|--------------|------------|
| `'python' is not recognized` / `Python was not found` | Install Python with "Add python.exe to PATH" ticked, then open a new PowerShell. Or use `py`. |
| `numpy is required` | `python -m pip install numpy` |
| `No module named 'PIL'` | `python -m pip install pillow` (only for `--png`). |
| `FileNotFoundError ... 'ffmpeg'` | Install ffmpeg (section 2.4) or add `--no-mp4`. |
| `can't open file '...fvf.py'` | PowerShell is not in the tool's folder (section 3). |
| `FileNotFoundError: ... No such file or directory` | A path is wrong, or the output folder does not exist. |
| A path with a space breaks the command | Put the whole path in double quotes. |
| `invalid block size ... not a valid FVF file` / `bad magic` | The file is not an FVF video, or it is damaged. |
| `file does not fit (short by ... B); lowering scale ...` | Normal. Wait. |
| `cannot fit` | The tool gave up fitting the file. Do not use any output; keep the `.log` file. |
| The last line is not `VERIFIED` | Do not use that file. |
| Encoding seems stuck | Open the `.log` file next to the output in Notepad; it shows the latest progress line. Long videos take hours. |
| Out of memory | Use fewer workers (`--jobs 2`, or `--jobs 1`) and close other programs. |
| `patch-zip`: `size ... != archive entry ...` | The file does not have the size of the entry it replaces. Use a file produced by `encode`. |
| `patch-zip`: `zip64 archive: not supported` / `... is not stored` | The input is not the game's original `STTNG.ZIP`. |
| `view`: `fvf_viewer.html was not found` | Download the viewer from the release page or run `python viewer_src/build_viewer.py`. |
| The viewer cannot open a video inside a zip | Only uncompressed entries can be opened. |
| The game does not start, or videos are missing | Restore the original `STTNG.ZIP` and repeat with `patch-zip`; do not re-zip with an archive program. |
| The game stops loading a cutscene | Run `python fvf.py check FILE.fvf`; it reports a header byte 20–23 that does not match the last block. |

If a window closes immediately, you started it by double-clicking; always start from PowerShell.

---

## 9. macOS and Linux

The tool is plain Python. Install Python 3, numpy and (optionally) ffmpeg and Pillow with your package manager and pip (for example `brew install python ffmpeg` or `sudo apt install python3 python3-pip ffmpeg`, then `python3 -m pip install numpy pillow`; on newer Linux systems add `--break-system-packages` or use a virtual environment). Use `python3` instead of `python` and forward slashes in paths. The game itself needs DOSBox or DOSBox-X.

---

## 10. Notes for developers

- `fvflib/decoder.py` is the reference decoder. Importing it prints nothing; set `FVF_VERBOSE=1` (or pass `-v` to a command) for header and block details.
- `fvflib/encoder7.py` is the frame encoder used by `fvf.py`. It builds on `encoder4.py` (rate control), `encoder3.py` (block matching and opcode packing) and `encoder2.py` (constants and the temporal-skip builder). `recontainer_lite.py` builds frame records. `check_fvf.py` implements `check` and `compare`, and `patch_zip.py` implements `patch-zip`.
- Encoder behaviour can be changed with environment variables (read at import, so worker processes see them): `FVF_DIST=sad|sse|p15` (distortion measure, default `sad`), `FVF_CTX=0|1` (context-aware byte model, default `1`) and `FVF_COST_C5` (modelled byte cost of a motion-copy block).
- **Frame structure of the originals.** Frame 0 has flags 5 (decoded 16 times; it consists of 6000 affine blocks, about 18 KB). Every frame with index divisible by 4 has flags 4 (the buffers swap, so it becomes the reference); all others have flags 0 and predict from the last key frame. The encoder follows this structure.
- **Container rules the encoder follows.** Blocks are multiples of 2048 bytes and at most 63,488 bytes. The first block holds three frames (at most 61,440 bytes); the encoder puts at most six frames in each later block (the originals have up to seven). Header bytes 20–23 hold the file offset of the last block.
- **Tests.** The viewer's JavaScript decoder is a port of `decoder.py`. `python viewer_src/mkref.py FILE.fvf ...` writes reference hashes, and `node viewer_src/test_js.js FILE.fvf ...` checks the JavaScript output against them (needs Node.js). `python viewer_src/build_viewer.py` builds the single-file viewer from `viewer_src/viewer_src.html` and `viewer_src/fvf_decoder.js`.

---

## 11. Acknowledgements and licence

This work builds on the format research and decoders of others:

- **Kostya Shishkov** and the NihAV project: the Rust decoder for the related TMM-Frac container (`tmmfrac.rs` in `na_eofdec`) and his published work on these formats were used as an independent cross-check of the bitstream syntax.
- **The ScummVM project** and its contributors: the C++ FVF decoder reference was used as a second cross-check.
- **The MultimediaWiki** (<https://wiki.multimedia.cx/>) for its documentation of multimedia formats.
- **Ghidra** (the NSA's open-source reverse-engineering suite): used to decompile and review the functions in the game's `PLAYER.OVL` and `STTNG.OVL` files, and to support the debugging work in DOSBox-X.
- **Claude** (Anthropic's AI assistant) helped build and test the decoder, the encoder and this tool.

The decoder in this repository was written from the bitstream analysis and verified against the game itself running in DOSBox-X, with Ghidra used to review the game's own decode routines; the other projects served as cross-checks.

*Star Trek: The Next Generation – A Final Unity* is a product of its original authors and publishers. This repository contains no game data.

**Licence:** the code in this repository is dedicated to the public domain under [CC0 1.0 Universal](LICENSE).
