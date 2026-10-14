# dont_touch_wt.xdc  (Rev158 — 200MHz timing closure)
#
# ── Multicycle cho config-static regs feed vào tile computation ────────────────
# tile_ctrl.v:56-57:  h_tiles = (Hin + tile_h - 1)/tile_h ; w_tiles = (Win+tile_w-1)/tile_w
#   → phép CHIA (35 logic levels) từ:  Hin=reg_hin, Win=reg_win, tile_h/tile_w ← reg_conv_type.
# tile_ctrl.v:54 comment: "h_tiles/w_tiles settle 1 cyc after config, stable before start".
# Các reg này = AXI config, ghi 1×/layer rồi ĐỨNG YÊN suốt compute (>>4 cyc trước consumer).
# FAIL single-cycle @200MHz. Multicycle 4 = 20ns budget → pass. Cả cloud multicycle nên
# retiming KHÔNG chẻ (không còn nhánh single-cycle ép balance) → giữ nguyên 1 path 4-cycle.
#
# ⚠️ XDC KHÔNG hỗ trợ `if`/`for`/`proc` → áp TRỰC TIẾP (get_cells non-empty, đã verify).
# ⚠️ reg_conv_type QUAN TRỌNG: thiếu nó → retiming chẻ h_tiles thành _bret single-cycle FAIL.
set_multicycle_path 4 -setup -from [get_cells -hierarchical -quiet -filter {NAME =~ *u_axil/reg_hin_reg* || NAME =~ *u_axil/reg_win_reg* || NAME =~ *u_axil/reg_cin_reg* || NAME =~ *u_axil/reg_cout_reg* || NAME =~ *u_axil/reg_conv_type_reg*}]
set_multicycle_path 3 -hold  -from [get_cells -hierarchical -quiet -filter {NAME =~ *u_axil/reg_hin_reg* || NAME =~ *u_axil/reg_win_reg* || NAME =~ *u_axil/reg_cin_reg* || NAME =~ *u_axil/reg_cout_reg* || NAME =~ *u_axil/reg_conv_type_reg*}]
