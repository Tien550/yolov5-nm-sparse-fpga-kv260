"""
inspect_coo.py  --  Đọc và hiển thị nội dung COO packed.bin

In từng group 4 tiles, decode thủ công, so sánh với dense gốc.
"""
import argparse, json, os, struct, numpy as np

M = 16


def decode_coo_tile(raw, pos, N):
    idx_bpt = (N + 1) // 2
    idx_raw = raw[pos:pos + idx_bpt]; pos += idx_bpt
    nibbles = []
    for b in idx_raw:
        nibbles += [b & 0xF, (b >> 4) & 0xF]
    indices = nibbles[:N]
    values  = [struct.unpack('b', bytes([raw[pos + i]]))[0] for i in range(N)]
    pos    += N
    tile = np.zeros(M, dtype=np.int8)
    for idx, val in zip(indices, values):
        if 0 <= idx < M:
            tile[idx] = val
    return tile, pos, indices, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',   default='weights')
    parser.add_argument('--extracted', default='yolov5/runs/quant/extracted')
    parser.add_argument('--layers',    nargs='+', default=['01', '07', '25'])
    parser.add_argument('--tiles',     type=int, default=3,
                        help='Số groups (4 tiles/group) đầu tiên cần in')
    args = parser.parse_args()

    with open(os.path.join(args.extracted, 'metadata.json')) as f:
        meta = json.load(f)

    for lstr in args.layers:
        idx  = int(lstr)
        key  = f"layer_{idx:02d}"
        info = meta[key]
        shape = info['weight_shape']
        Cout, Cin, Kh, Kw = shape

        n_path = os.path.join(args.weights, f"L{idx:02d}_N_config.txt")
        p_path = os.path.join(args.weights, f"L{idx:02d}_packed.bin")
        with open(n_path) as f:
            N = int(f.read().strip())

        raw = open(p_path, 'rb').read()

        block_len = Cin * Kh * Kw
        pad_len   = (M - block_len % M) % M
        padded    = block_len + pad_len
        num_blk   = padded // M

        enc = "COO" if N < M else "DENSE"
        idx_bpt    = (N + 1) // 2
        bytes_tile = idx_bpt + N          # bytes per COO tile
        bytes_group = 4 * bytes_tile       # bytes per group-of-4

        print(f"\n{'='*70}")
        print(f"Layer {idx:02d} | {info['name']}")
        print(f"  shape={shape}  N={N}  encoding={enc}")
        print(f"  block_len={block_len}  num_blk/filter={num_blk}")
        print(f"  bytes/tile={bytes_tile} ({idx_bpt}B idx + {N}B vals)  "
              f"bytes/group={bytes_group}")
        print(f"  file size={len(raw)}B  "
              f"expected={(Cout*num_blk+3)//4*4*bytes_tile}B")

        # Load dense original
        w_dense = np.frombuffer(
            open(os.path.join(args.extracted, 'weights_bin', f"layer_{idx:02d}.bin"), 'rb').read(),
            dtype=np.int8).reshape(Cout, block_len)

        print(f"\n  --- {args.tiles} group đầu tiên ({args.tiles*4} tiles) ---")
        pos = 0
        tile_global = 0
        for g in range(args.tiles):
            print(f"\n  [Group {g}]  offset=0x{g*bytes_group:04X}")
            for t in range(4):
                oc  = tile_global // num_blk
                blk = tile_global  % num_blk
                tile_dec, pos_new, raw_idx, raw_val = decode_coo_tile(raw, pos, N)

                # Đọc dense reference — pad đến M trước, rồi argsort zero M-N nhỏ nhất
                # (khớp đúng với gen_nm_sparse.py)
                if oc < Cout:
                    raw_blk = w_dense[oc, blk*M:(blk+1)*M]   # ≤M phần tử
                    dense_padded = np.zeros(M, dtype=np.int8)
                    dense_padded[:len(raw_blk)] = raw_blk
                    abs_d    = np.abs(dense_padded.astype(np.float32))
                    pruned   = dense_padded.copy()
                    zero_idx = np.argsort(abs_d, kind='stable')[:M - N]
                    pruned[zero_idx] = 0
                    match = np.array_equal(tile_dec, pruned)
                else:
                    dense_blk = np.zeros(M, dtype=np.int8)
                    pruned    = np.zeros(M, dtype=np.int8)
                    match     = np.array_equal(tile_dec, np.zeros(M, dtype=np.int8))

                # Hiển thị raw bytes của tile này
                raw_bytes = list(raw[pos:pos_new])
                idx_bytes = raw_bytes[:idx_bpt]
                val_bytes = raw_bytes[idx_bpt:]

                nz_pos = np.where(tile_dec != 0)[0].tolist()

                print(f"    Tile {t} (oc={oc}, blk={blk}) | "
                      f"raw: idx={idx_bytes} val={val_bytes}")
                print(f"           indices={raw_idx[:N]}  values={raw_val[:N]}")
                print(f"           decoded : {tile_dec.tolist()}")
                if oc < Cout:
                    raw_blk_disp = w_dense[oc, blk*M:(blk+1)*M]
                    print(f"           dense   : {raw_blk_disp.tolist()}")
                print(f"           nz_pos  : {nz_pos}  match={'✓' if match else '✗'}")

                pos = pos_new
                tile_global += 1

        # Thống kê toàn layer
        pos = 0
        tile_global = 0
        total_tiles = Cout * num_blk
        total_groups = (total_tiles + 3) // 4
        nz_per_tile = []
        for _ in range(total_groups):
            for _ in range(4):
                t, pos, _, _ = decode_coo_tile(raw, pos, N)
                nz_per_tile.append(int(np.sum(t != 0)))
                tile_global += 1

        nz_arr = np.array(nz_per_tile[:total_tiles])
        sparsity = 100. * (1. - nz_arr.mean() / M)
        print(f"\n  --- Thống kê toàn layer ({total_tiles} tiles hữu ích) ---")
        print(f"  NZ per tile: mean={nz_arr.mean():.2f}  min={nz_arr.min()}  "
              f"max={nz_arr.max()}  (N={N})")
        print(f"  Sparsity thực: {sparsity:.2f}%")
        print(f"  Tiles với NZ<N: {np.sum(nz_arr < N)} "
              f"(padding tiles hoặc zero-weight)")
        print(f"  File size: {len(raw):,} B = {len(raw)/1024:.1f} KB")


if __name__ == '__main__':
    main()
