// cnn_defines.vh  --  Global constants for N:M Sparse CNN Accelerator
// Include with: `include "cnn_defines.vh"
// Reconciled with CLAUDE.md 2026-06-03
//
// Model : YOLOv5s-PKLot QAT (63 layers, 4 fp32-output layers)
// Target: Kria KV260 (xck26-sfvc784-2LV-c)  fallback ZCU102

`ifndef CNN_DEFINES_VH
`define CNN_DEFINES_VH

// ── Paper / Architecture Parameters (CLAUDE.md §2, FIXED) ──────────────────
`define M             16    // N:M block size
`define G              4    // multiplier groups per SPEC
`define P             16    // multipliers per group  (G*P = 64 MACs/SPEC)
`define N_SPEC_KV      4    // SPEC clusters for KV260
`define N_SPEC_ZCU    16    // SPEC clusters for ZCU102
// B-w wider array: ACTIVE SPEC count in unified_acc_top. 4 = baseline (Rev135). 12 = Option C
// (~801 DSP 64% sau prune R1). 8 = Option A fallback. Đổi 1 dòng này để chọn cấu hình (chờ gate mAP).
`define N_SPEC_ACT    12    // B-w: 12=Option C (R1 pruned, ~801 DSP 64%). 4=baseline Rev135-138. Elaborate PASS cả 4 lẫn 12 (2026-07-02). Rev139: 12 + weights_unified_v4_ns12.
`define TILE_H        28    // spatial tile height (pixels)
`define TILE_W        28    // spatial tile width
`define HALO           1    // overlap pixels for conv3x3
`define LEAKY_SLOPE    3    // LeakyReLU: x >>> 3 = 0.125

// ── Data Widths ─────────────────────────────────────────────────────────────
`define DATA_W         8    // INT8 weight / activation
`define ACC_W         32    // accumulator width (avoids overflow)
`define SCALE_W       16    // M0 requantization multiplier width
`define SHIFT_W        5    // requantization right-shift amount width

// ── AXI Base Addresses (FROZEN -- CLAUDE.md §4) ─────────────────────────────
`define AXI_BASE_CONV1X1    32'hA000_0000
`define AXI_BASE_CONV3X3    32'hA001_0000
`define AXI_BASE_DWCONV3X3  32'hA002_0000

// ── AXI Register Offsets (16-byte stride for 128→32 dwidth_converter) ──────
// AXI Interconnect: 128-bit HPM0 → axi_dwidth_converter → AXI4Lite 32-bit.
// The dwidth_converter only forwards transactions at addresses naturally
// aligned to the MI data width (128-bit = 16 bytes). Board testing confirmed
// that 8-byte-aligned-but-not-16-byte addresses (0x08, 0x18, 0x28...) cause
// AXI timeout and board hang. Use 16-byte stride (0x00, 0x10, 0x20, ...).
// Total: 22 registers × 16 bytes = 352 bytes — need 9-bit awaddr [8:0].
`define AXI_OFF_AP_CTRL     9'h000  // bit0=ap_start(W1SC) bit1=ap_done(RO) bit2=ap_idle(RO)
`define AXI_OFF_AP_GIE      9'h010  // bit0=global interrupt enable
`define AXI_OFF_AP_IER      9'h020  // bit0=ap_done interrupt enable
`define AXI_OFF_AP_ISR      9'h030  // bit0=interrupt status (W1C)
`define AXI_OFF_CIN         9'h040  // input channels
`define AXI_OFF_COUT        9'h050  // output channels (also C for dwconv)
`define AXI_OFF_HIN         9'h060  // input height
`define AXI_OFF_WIN         9'h070  // input width
`define AXI_OFF_N_CONFIG    9'h080  // non-zero count per M-block
`define AXI_OFF_POC         9'h090  // output channel parallelism (1,2,4)
`define AXI_OFF_ACCU_MODE   9'h0A0  // 0=1-in 1=2-in 2=4-in
`define AXI_OFF_APPLY_RELU  9'h0B0  // 0=none 1=leaky relu (>>3)
`define AXI_OFF_WT_ADDR_LO  9'h0C0  // weight DDR base [31:0]
`define AXI_OFF_WT_ADDR_HI  9'h0D0  // weight DDR base [63:32]
`define AXI_OFF_IN_ADDR_LO  9'h0E0  // input feature map base [31:0]
`define AXI_OFF_IN_ADDR_HI  9'h0F0  // input feature map base [63:32]
`define AXI_OFF_OUT_ADDR_LO 9'h100  // output feature map base [31:0]
`define AXI_OFF_OUT_ADDR_HI 9'h110  // output feature map base [63:32]
`define AXI_OFF_STRIDE      9'h120  // conv stride (conv3x3/dwconv only)
`define AXI_OFF_PAD         9'h130  // zero-padding (conv3x3/dwconv only)
`define AXI_OFF_LAYER_TYPE  9'h140  // layer type: 0=conv1x1 1=conv3x3 2=dwconv (cnn_acc_top intercepts)
`define AXI_OFF_OUT_BYTES   9'h150  // output DMA byte count = Cout*Hout*Wout (written by SW before ap_start)
`define AXI_OFF_WT_BYTES    9'h160  // weight blob byte count (written by SW before ap_start)
// B-c3 (Rev139): 0x170/0x180 là 2 slot 16-aligned ≤0x1E0 còn trống (AXI wall: chỉ slot cũ ack).
`define AXI_OFF_C3_CFG      9'h170  // c3 9-pass cfg: [7:0]=W_in [11:8]=ncolg [16:12]=nicc(Cin/16/tap,5b) [24:17]=nrow [25]=s2 [26]=mask_bot [27]=mask_right [31:28]=last_pos
`define AXI_OFF_ACT_TOTAL   9'h180  // LS_ACTL: số chunk l_actc cần DMA vào act_uram (16b). 1x1 B-i: npos*nicb; c3: ceil(plane/l_actc)
// 0x170/0x180 = DBG seq_state / kernel_state (read-only)
// ── B0 layer_sequencer registers (on-chip layer loop; LAYER_MODE=1 enables) ──
`define AXI_OFF_LAYER_MODE  9'h190  // 0=per-tile (host loop), 1=per-layer (layer_sequencer)
`define AXI_OFF_N_OCB       9'h1A0  // number of out-ch-blocks = ceil(Cout/4)
`define AXI_OFF_N_POS       9'h1B0  // positions/oc-block = Hout*ceil(Wout/16)
`define AXI_OFF_N_ICB       9'h1C0  // ic-chunks = ceil(Cin/16)
`define AXI_OFF_WT_BLK_BYTES 9'h1D0 // per-(oc,ic) WT blob bytes (uniform, padded)
`define AXI_OFF_ACT_CHUNK   9'h1E0  // per-tile ACT bytes/stride (1024=conv3x3, 256=conv1x1 R3)
// 0x1F0: RESERVED — ghi vào SLVERR trên IP hiện hành (→ SIGBUS host). Rev169 lag SUY TRONG HW
// (RESOUT & ~INBANK, D=2*ncolg trong LS) nên KHÔNG cần reg này. Đừng dùng 0x1F0.

// ── Model constants ─────────────────────────────────────────────────────────
`define TOTAL_LAYERS  63
`define N_FP32_OUT     4   // fp32-output layers: 42, 49, 56, 62

// ── Conv type encoding (for AXI_OFF_LAYER_TYPE value) ───────────────────────
`define CONV_TYPE_1X1   2'd0  // pointwise (no sliding window)
`define CONV_TYPE_3X3   2'd1  // standard 3x3 (with sliding window)
`define CONV_TYPE_DW    2'd2  // depthwise 3x3 (poc=1 forced, accu_mode=1-in forced)

// ── Shuf mode aliases (used in unified_acc_top MUX) ─────────────────────────
`define SHUF_LOAD       2'b00
`define SHUF_SHIFT      2'b01
`define SHUF_HOLD_MODE  2'b10

`endif // CNN_DEFINES_VH
