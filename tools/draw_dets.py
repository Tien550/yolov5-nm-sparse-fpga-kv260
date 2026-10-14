"""draw_dets.py -- forward INT8 (inference_cpu, khớp board) 1 lần rồi vẽ box cho NHIỀU ngưỡng
NMS/CONF để so trực quan (cân box đỡ chồng). Optional AGN=1: thêm lượt agnostic-NMS xoá chồng
chéo lớp (1 ô không đồng thời empty+occupied). Usage: python draw_dets.py <img> <out_prefix>
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import inference_cpu as C
from PIL import Image, ImageDraw

img_path, out_prefix = sys.argv[1], sys.argv[2]

im = Image.open(img_path).convert("RGB").resize((C.IMG_SIZE, C.IMG_SIZE))
arr = np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0
img = np.round(arr / C.INPUT_SCALE).clip(-128, 127).astype(np.int8)
meta, req, weights, bias = C.load_model()
conv, osc, isc = C.make_runner(meta, req, weights, bias)
outs = C.run_graph(img, conv, osc, isc)


def agnostic_pass(dets, iou):
    """xoá box chồng CHÉO lớp: giữ conf cao nhất trên mỗi cụm (mọi lớp)."""
    if not dets: return dets
    b = np.array([d[:4] for d in dets], float); s = np.array([d[4] for d in dets], float)
    keep = C._nms(b, s, iou)
    return [dets[i] for i in keep]


def draw(dets, path):
    vis = im.copy(); d = ImageDraw.Draw(vis)
    for bx in dets:
        col = (0, 220, 0) if C.CLASS_NAMES[int(bx[5])].endswith('empty') else (230, 0, 0)
        d.rectangle([bx[0], bx[1], bx[2], bx[3]], outline=col, width=2)
    vis.save(path)


configs = [
    ("conf25_93",  0.25, 0.12, 0.0),    # hiện tại
    ("conf20",     0.20, 0.12, 0.0),    # hạ conf → bắt thêm ô sót
    ("conf15",     0.15, 0.12, 0.0),    # hạ nữa
    ("conf10",     0.10, 0.12, 0.0),    # max recall
]
for name, conf, nms, agn in configs:
    dets = C.decode(outs, conf=conf, nms=nms)
    if agn > 0:
        dets = agnostic_pass(dets, agn)
    draw(dets, f"{out_prefix}_{name}.jpg")
    print(f"{name}: {len(dets)} det -> {out_prefix}_{name}.jpg")
