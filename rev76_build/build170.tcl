# build170.tcl -- Rev168 RESIDENT-ACT: scatter out->act_uram + act_bram_in (BRAM 512KB input bank) + LS skip ACTL (act_total=0) + oc_deint 17b. Gate: LAYER_MODE 0x2000 (RESOUT) + 0x8000 (INBANK).
#   Bit-exact cho dense c3 (idx=slot) → golden 138 giữ nguyên là REGRESSION GATE.
#   Sparse c3 (N=4 lớp 6/8/19/22) drain N kênh kept → c3 kick giảm.
#   Chỉ 2 dòng RTL đổi trong unified_acc_top.v; multiply 4x18b off per-pixel path.
# Chay: /d/Vivado/2025.2/Vivado/bin/vivado.bat -mode batch -source build157.tcl -log build170.log -journal build170.jou
# FREQ 166.67 (giữ như bit 148 đang chạy trên board FPS 0.79). WNS marginal như 148 -> chấp nhận
#   (dense --verify max=0 x3 đã chứng minh silicon đóng timing). Nếu 150 WNS xấu hơn nhiều: fallback 142.

set FREQ 200.0

set script_dir [file dirname [info script]]
cd $script_dir
open_project bd_cnn_acc/bd_cnn_acc.xpr
set_param general.maxThreads 12
config_ip_cache -disable_cache

# ── refresh ipshared tu ip_repo/src (nguon RTL sparse-c3 da sua) + upgrade IP ──
update_ip_catalog -rebuild -scan_changes
open_bd_design [get_files bd.bd]
# Rev156: GLOBAL synth (khong OOC) -> cnn_acc_top synth chung top run co -global_retiming on
set_property synth_checkpoint_mode None [get_files bd.bd]
set locked_ips [get_ips *]
if {[llength $locked_ips] > 0} {
    catch {upgrade_ip $locked_ips} up_msg
    puts "BUILD170: upgrade_ip -> $up_msg"
}

# Copy truc tiep .v/.vh vao ipshared/src (chỗ synthesis doc) — dam bao RTL Rev150 vao build
foreach srcf [concat [glob -nocomplain $script_dir/ip_repo/src/*.v] [glob -nocomplain $script_dir/ip_repo/src/*.vh]] {
    foreach dstdir [concat [glob -nocomplain $script_dir/bd_cnn_acc/bd_cnn_acc.gen/sources_1/bd/bd/ipshared/*/src] \
                           [glob -nocomplain $script_dir/bd_cnn_acc/bd_cnn_acc.ip_user_files/bd/bd/ipshared/*/src]] {
        file copy -force $srcf $dstdir
    }
}
puts "BUILD170: ipshared refreshed (sparse-c3 RTL)"

# Clock -> FREQ (PS PL0)
set ps_cell [get_bd_cells -hierarchical -filter {VLNV =~ "*zynq_ultra_ps_e*"}]
if {[llength $ps_cell] == 0} { puts "BUILD170_ERROR: no PS cell"; exit 1 }
set_property CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $FREQ $ps_cell
puts "BUILD170: PL0 -> [get_property CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $ps_cell] MHz"

# Rev157: ADD multicycle XDC (config-static reg_hin/win/cin/cout -> address) — CHUA co trong BD build.
# Path reg_hin -> DSP(HxW) -> in_bram_rd_addr = worst @200 (config static, on dinh >=2cyc -> MCP2 an toan).
set mcp_xdc "d:/yolov5-fpga/rev76_build/src/constraints/dont_touch_wt.xdc"
if {[file exists $mcp_xdc]} {
    add_files -quiet -fileset constrs_1 -norecurse $mcp_xdc
    set_property USED_IN {synthesis implementation} [get_files -quiet $mcp_xdc]
    puts "BUILD170: added multicycle XDC -> [get_files -quiet dont_touch_wt.xdc]"
} else { puts "BUILD170_WARN: mcp xdc not found" }

validate_bd_design
save_bd_design
generate_target all [get_files bd.bd]

# reset day du (dcp cu khong dung lam incremental)
set_property incremental_checkpoint {} [get_runs impl_1]
foreach r [get_runs -filter {IS_SYNTHESIS == 1}] { catch {reset_run $r} }
reset_run impl_1
set_property STEPS.PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]


# Rev154 200MHz: cnn_acc_top IP synth THIEU -retiming (IP khac deu co). Bat RETIMING + fanout_limit
# TRUC TIEP tren IP OOC synth run + RESET no de re-synth (khong dung cache).
# Rev156: retiming tu global synth (top run -global_retiming on) + RTL max_fanout (use_bm_r) + AggressiveExplore
set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
set_property STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]

launch_runs impl_1 -to_step write_bitstream -jobs 12
wait_on_run impl_1
set prog [get_property PROGRESS [get_runs impl_1]]
puts "BUILD170: impl progress = $prog"
if {$prog != "100%"} { puts "BUILD170_FAILED"; close_project; exit 1 }

# Report WNS
open_run impl_1
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
puts "BUILD170_WNS = $wns"
puts "BUILD170_OK"
close_project
exit 0
