# fvf 1.0.0

Command-line tool and viewer for the FVF video format (Total Multimedia / SoftVideo) used by *Star Trek: The Next Generation – A Final Unity* (MS-DOS, 1995).

## What it does

- **Decode** any of the game's videos to `.mp4`, `.wav` and PNG frames.
- **Re-encode** a video into a new `.fvf` of exactly the original size with identical audio.
- **Repack** new videos into a copy of the game's `STTNG.ZIP` without changing any other byte (`patch-zip`, `replace`).
- **Inspect** files (`info`, `check`) and compare a re-encode with its original (`compare`, and the viewer).

## Download

- `fvf_viewer.html` (attached to this release): the viewer. Open it in a browser; it works offline and uploads nothing.
- Source code: this repository. See the README for set-up and usage.

## Tested

- Decoded output matches the game's own decoder behaviour in DOSBox-X.
- Re-encoded `teaser`, `title`, `warpout1`, `warnshot`, `labarriv`, `shiptour` and `tracbloc` play in the game under DOSBox-X. `chodesc` was checked with the tool's decoder and the viewer only.
- Mean error of the re-encodes against the originals (MAD, 0–255 scale): 0.005 to 0.099. Worst single frame: 0.417 (`shiptour`).

## Requirements

Python 3.9 or newer and numpy. ffmpeg is needed only for `.mp4` output, Pillow only for PNG output.

## Known limits

- Re-encoding is lossy and slow: roughly 5–10 seconds per frame on a Ryzen 7 7800X3D (about 3.5 hours for a 2,265-frame video).
- Only the videos listed above have been re-encoded and tested.
- Only the Windows set-up was written up in detail; the tool is plain Python and should run on macOS and Linux.
- The repository contains no game data. Extract the videos from your own copy of the game.

## Licence and credits

CC0 1.0 Universal (public domain dedication). Acknowledgements are in the README: Kostya Shishkov and NihAV, the ScummVM project, the MultimediaWiki, Ghidra and Claude.
