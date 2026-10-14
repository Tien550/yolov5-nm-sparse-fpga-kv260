#!/usr/bin/env python3
"""gen_graphop_tb.py -- Sinh test vector + golden output cho tb_graphop_engine.v.
Dùng ĐÚNG công thức rescale_int8/resid_cat (inference_cpu.py) và find_best_shift
(calibrate_detect_head.py) để đảm bảo golden khớp bit-exact với hệ thống thật.
"""
import numpy as np, os

np.random.seed(42)
OUT = os.path.dirname(os.path.abspath(__file__))

N_POS = 32      # Rev171: 32 = 2 position-tile (16 mỗi tile) → THỬ được reorder tiled position-major
CH_AB = 20      # Rev171: Cout=CH_AB+CH_C=32 (bội 16 → không padding) = 2 channel M-block
CH_C  = 12

def find_best_shift(M_float):
    if M_float <= 0: return 0x8000, 15
    shift = 0; val = M_float
    while val < (1 << 15): val *= 2; shift += 1
    while val >= (1 << 16): val /= 2; shift -= 1
    M0 = int(round(val))
    if M0 >= (1 << 16): M0 = (1 << 16) - 1
    return M0, shift

def rescale_int8(x, fs, ts):
    if abs(fs - ts) < 1e-9: return x
    return np.round(x.astype(np.float32) * (fs / ts)).clip(-128, 127).astype(np.int8)

def resid_cat(cv1, cv1_sc, mc2, mc2_sc, cv2, cv2_sc, u_sc):
    bott = np.clip(rescale_int8(cv1, cv1_sc, u_sc).astype(np.int16)
                   + rescale_int8(mc2, mc2_sc, u_sc).astype(np.int16), -128, 127).astype(np.int8)
    return np.concatenate([bott, rescale_int8(cv2, cv2_sc, u_sc)], axis=0)

def cat(a, a_sc, b, b_sc, u_sc):
    return np.concatenate([rescale_int8(a, a_sc, u_sc), rescale_int8(b, b_sc, u_sc)], axis=0)

# scale ngẫu nhiên nhưng thực tế (giống scale lượng tử thật: 0.001-0.1)
sc_a, sc_b, sc_c, sc_u = 0.021, 0.034, 0.017, 0.025

A = np.random.randint(-128, 128, (CH_AB, N_POS), dtype=np.int8)
B = np.random.randint(-128, 128, (CH_AB, N_POS), dtype=np.int8)
C = np.random.randint(-128, 128, (CH_C,  N_POS), dtype=np.int8)

gold_resid = resid_cat(A, sc_a, B, sc_b, C, sc_c, sc_u)   # (CH_AB+CH_C, N_POS)
gold_cat   = cat(A, sc_a, C, sc_c, sc_u)                   # dùng A,C cho case cat thường (CH_AB+CH_C)

def to_tiled(R):
    """(Cout, N_POS) logical -> TILED byte order khớp graphop store + conv1x1 BI đọc act_uram:
    byte[(pt*N_IC + ic)*256 + ch*16 + sp] = R[ic*16+ch, pt*16+sp].  Cout bội 16 → không padding."""
    Cout, NP = R.shape
    N_IC = (Cout + 15) // 16
    NT   = NP // 16
    out  = np.zeros(NT * N_IC * 256, np.int8)
    for pt in range(NT):
        for ic in range(N_IC):
            for ch in range(16):
                Cc = ic * 16 + ch
                if Cc >= Cout: continue
                for sp in range(16):
                    out[(pt*N_IC + ic)*256 + ch*16 + sp] = R[Cc, pt*16 + sp]
    return out

gold_resid = to_tiled(gold_resid)   # Rev171: xuất theo TILED (graphop ghi tiled)
gold_cat   = to_tiled(gold_cat)

m0_a, sh_a = find_best_shift(sc_a / sc_u)
m0_b, sh_b = find_best_shift(sc_b / sc_u)
m0_c, sh_c = find_best_shift(sc_c / sc_u)

def wr_hex(fn, arr_flat_int8):
    with open(os.path.join(OUT, fn), 'w') as f:
        for v in arr_flat_int8:
            f.write(f'{v & 0xFF:02x}\n')

# Rev200: INPUT cũng TILED (chuẩn hoá full-flow — producer conv1x1 RESOUT scatter tiled).
# Pad kênh RIÊNG mỗi nguồn: A/B ceil(20/16)*16=32 (N_IC=2 → 1024B), C ceil(12/16)*16=16 (512B).
A_t = to_tiled(A); B_t = to_tiled(B); C_t = to_tiled(C)
wr_hex('tb_a.hex', A_t)
wr_hex('tb_b.hex', B_t)
wr_hex('tb_c.hex', C_t)
wr_hex('tb_gold_resid.hex', gold_resid.flatten())
wr_hex('tb_gold_cat.hex',   gold_cat.flatten())

with open(os.path.join(OUT, 'tb_params.vh'), 'w') as f:
    f.write(f"""// AUTO-GENERATED bởi gen_graphop_tb.py -- ĐỪNG sửa tay
`define TB_N_POS   {N_POS}
`define TB_CH_AB   {CH_AB}
`define TB_CH_C    {CH_C}
`define TB_A_BYTES {len(A_t)}
`define TB_B_BYTES {len(B_t)}
`define TB_C_BYTES {len(C_t)}
`define TB_M0_A    16'd{m0_a}
`define TB_SH_A    5'd{sh_a}
`define TB_M0_B    16'd{m0_b}
`define TB_SH_B    5'd{sh_b}
`define TB_M0_C    16'd{m0_c}
`define TB_SH_C    5'd{sh_c}
""")

print(f"M0/shift: A=({m0_a},{sh_a}) B=({m0_b},{sh_b}) C=({m0_c},{sh_c})")
print(f"gold_resid shape={gold_resid.shape}  gold_cat shape={gold_cat.shape}")
print("Đã ghi tb_a.hex/tb_b.hex/tb_c.hex/tb_gold_resid.hex/tb_gold_cat.hex/tb_params.vh")
