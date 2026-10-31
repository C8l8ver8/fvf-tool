#!/usr/bin/env python3
# SPDX-License-Identifier: CC0-1.0
"""Builds the single-file viewer ../fvf_viewer.html from viewer_src.html and fvf_decoder.js.
usage: python viewer_src/build_viewer.py"""
import os
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, 'viewer_src.html'), encoding='utf-8').read()
dec = open(os.path.join(here, 'fvf_decoder.js'), encoding='utf-8').read()
assert '</script' not in dec and '/*INLINE_DECODER*/' in src
out = os.path.join(here, '..', 'fvf_viewer.html')
open(out, 'w', encoding='utf-8').write(src.replace('/*INLINE_DECODER*/', dec))
print('wrote', os.path.normpath(out))
