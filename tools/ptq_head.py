#!/usr/bin/env python3
"""ptq_head.py — PTQ head L42/49/56: M0/shift = in_sc*w_sc/out_sc (per-channel, M0 uint16
chuẩn hoá về [2^15, 2^16) để tối đa precision). Xuất requant_params_headq.json (bản gộp)."""
import json, os, numpy as np
SD = os.path.join(os.path.dirname(__file__), '..', 'yolov5/runs/quant/extracted_c3')
meta = json.load(open(os.path.join(SD, 'metadata.json')))
req  = json.load(open(os.path.join(SD, 'requant_params.json')))
for lid in (42, 49, 56):
    k = f'layer_{lid:02d}'; m = meta[k]
    scale = m['input_scale'] * np.array(m['weight_scale_per_channel']) / m['output_scale']
    M0, SH = [], []
    for s in scale:
        e = 0
        while s * (1 << (e + 1)) < (1 << 16) and e < 30: e += 1
        v = int(round(s * (1 << e)))
        if v >= (1 << 16): v >>= 1; e -= 1
        assert 0 < v < (1 << 16) and 0 <= e <= 31, (s, v, e)
        M0.append(v); SH.append(e)
    req[k] = {'M0': M0, 'shift': SH}
    print(k, 'M0 range', min(M0), max(M0), 'shift range', min(SH), max(SH))
json.dump(req, open(os.path.join(SD, 'requant_params_headq.json'), 'w'))
print('saved requant_params_headq.json (', len(req), 'layers )')
