//! The kernels compute `C = A B^T` of row-major `[m, k]` A and `[n, k]` B on compute capability
//! 12.0 and later. Bulk tensor copies (TMA) fill the tiles and zero-fill the parts outside A and B.

const std = @import("std");
const builtin = @import("builtin");
const nvptx = std.nvptx;
const device = @import("device.zig");
const Chunk = device.Chunk;

pub const Epilogue = enum {
    /// C gets `bf16(product)`, or `bf16(bf16(product) + residual)` with a residual laid out as C.
    bf16,
    /// Column `j` of C gets `bf16(bf16(silu(bf16(gate))) * bf16(up))`, where `up` is the product
    /// with row `j` of B and `gate` is the product with row `gate_row + j` of B.
    swiglu,
    /// The kernel applies the RMS norm and rotary embedding of `qk_norm_rope.QkNormRopeInterleaved`
    /// with the same rounding and writes the heads of C to `QkNormRope.q` and `QkNormRope.kv`.
    qk_norm_rope,
};

/// The activation of a linear layer with a bias. Column `j` of C gets `bf16(product + bias[j])`
/// without one, or `bf16(bf16(product + bias[j]) + residual)` with a residual laid out as C, and
/// `bf16(f(bf16(product + bias[j])))` with an activation `f`, the tanh GELU or the GELU of the
/// error function.
pub const Activation = enum { none, gelu_tanh, gelu };

/// The epilogues and the activations after a bias, which the kernels share.
const Store = enum {
    bf16,
    swiglu,
    qk_norm_rope,
    bias,
    bias_gelu_tanh,
    bias_gelu,

    fn hasBias(store: Store) bool {
        return switch (store) {
            .bias, .bias_gelu_tanh, .bias_gelu => true,
            .bf16, .swiglu, .qk_norm_rope => false,
        };
    }
};

pub const QkNormRope = extern struct {
    q_weight: u64,
    k_weight: u64,
    /// Each row holds 64 FP32 cosines and then 64 FP32 sines.
    rope: u64,
    q: u64,
    /// Each row of the key-value cache holds the key heads and then the value heads.
    kv: u64,
    kv_rows: u64,
    heads: u32,
    epsilon: f32,
};

const head_dim = 128;

pub const TensorMap = extern struct {
    opaque_bytes: [128]u8 align(64),
};

pub const Tiling = struct {
    block_m: comptime_int,
    block_n: comptime_int,
    block_k: comptime_int,
    stages: comptime_int,
    warp_m: comptime_int,
    warp_n: comptime_int,
    group_m: comptime_int,
};

pub const default_tiling: Tiling = .{
    .block_m = 128,
    .block_n = 256,
    .block_k = 64,
    .stages = 2,
    .warp_m = 64,
    .warp_n = 64,
    .group_m = 8,
};

pub const kernels = .{
    Gemm(.bf16, default_tiling),
    Gemm(.swiglu, default_tiling),
    Gemm(.qk_norm_rope, default_tiling),
    GemmBias(.none, default_tiling),
    GemmBias(.gelu_tanh, default_tiling),
    GemmBias(.gelu, default_tiling),
};

comptime {
    if (builtin.cpu.arch.isNvptx() and @import("root") == @This()) {
        for (kernels) |Kernel| @export(&Kernel.kernel, .{ .name = Kernel.name });
    }
}

pub fn Gemm(comptime epilogue: Epilogue, comptime c: Tiling) type {
    return GemmOf(@field(Store, @tagName(epilogue)), c);
}

pub fn GemmBias(comptime activation: Activation, comptime c: Tiling) type {
    return GemmOf(switch (activation) {
        .none => .bias,
        .gelu_tanh => .bias_gelu_tanh,
        .gelu => .bias_gelu,
    }, c);
}

fn GemmOf(comptime epilogue: Store, comptime c: Tiling) type {
    const Element = u16;
    if (c.block_k * @sizeOf(Element) != 128) @compileError("the tensor maps copy rows of 128 bytes with the 128-byte swizzle");
    if (c.block_m % c.warp_m != 0 or c.block_n % c.warp_n != 0)
        @compileError("the warp tiles must divide the work group tile");
    if (c.warp_m % 16 != 0 or c.warp_n % 16 != 0) @compileError("warp tiles hold 16x16 matrices");
    if (epilogue == .swiglu and c.warp_n % 32 != 0)
        @compileError("each half of the columns of a warp tile holds 16x16 matrices");
    if (c.block_n > 2 * 256 or c.block_m > 256) @compileError("a tensor copy moves at most 256 rows");
    if (c.stages < 2) @compileError("the pipeline needs at least two stages");
    if (epilogue == .qk_norm_rope and c.warp_n * 2 != head_dim)
        @compileError("the QK norm epilogue needs two warps per head");
    const warps = c.block_m / c.warp_m * (c.block_n / c.warp_n);
    return struct {
        pub const name = "gemm_" ++ @tagName(epilogue) ++ "_sm120_tma";
        pub const block_m = c.block_m;
        pub const block_columns = if (epilogue == .swiglu) c.block_n / 2 else c.block_n;
        pub const b_box_rows = c.block_n / 2;
        const chunk_elements = 16 / @sizeOf(Element);
        pub const work_items = warps * 32;
        pub const dynamic_shared_bytes = @sizeOf(Tiles);

        const Tiles = extern struct {
            a: [c.stages][c.block_m * c.block_k]Element,
            b: [c.stages][c.block_n * c.block_k]Element,
        };
        const stage_bytes = @sizeOf(Tiles) / c.stages;
        const Accumulators = [c.warp_m / 16][c.warp_n / 8]@Vector(4, f32);
        const mma_options: nvptx.matrix.MmaOptions = .{
            .shape = .m16n8k16,
            .accumulator = .f32,
            .element = .bf16,
        };
        const k_step_elements = 2 * chunk_elements;
        const k_steps = c.block_k / k_step_elements;
        const warps_n = c.block_n / c.warp_n;

        extern var shared_storage: [0]u8 align(1024) addrspace(.shared);
        const tiles: *addrspace(.shared) Tiles = @ptrCast(&shared_storage);
        // A stage is full after its tensor copies and empty after every warp has read it.
        var full: [c.stages]nvptx.Mbarrier addrspace(.shared) = undefined;
        var empty: [c.stages]nvptx.Mbarrier addrspace(.shared) = undefined;

        pub const kernel = if (epilogue == .qk_norm_rope) kernelQkNormRope else if (epilogue.hasBias()) kernelBias else kernelLinear;

        /// Assumes that `n` and `ldc` are multiples of 8.
        fn kernelLinear(
            map_a: TensorMap,
            map_b: TensorMap,
            out: [*]addrspace(.global) Chunk,
            residual: ?[*]addrspace(.global) const Chunk,
            m: u32,
            n: u32,
            k: u32,
            ldc: u32,
            gate_row: u32,
        ) callconv(.{ .nvptx_kernel = .{
            .required_work_group_size = .{ .x = work_items },
            .min_work_groups_per_sm = 1,
        } }) void {
            const t = product(&map_a, &map_b, m, n, k, gate_row);
            store(out, residual, null, m, n, ldc, t);
        }

        /// Assumes that `n` and `ldc` are multiples of 8 and that `bias` holds `n` values.
        fn kernelBias(
            map_a: TensorMap,
            map_b: TensorMap,
            out: [*]addrspace(.global) Chunk,
            bias: [*]addrspace(.global) const u32,
            residual: ?[*]addrspace(.global) const Chunk,
            m: u32,
            n: u32,
            k: u32,
            ldc: u32,
        ) callconv(.{ .nvptx_kernel = .{
            .required_work_group_size = .{ .x = work_items },
            .min_work_groups_per_sm = 1,
        } }) void {
            const t = product(&map_a, &map_b, m, n, k, 0);
            store(out, residual, bias, m, n, ldc, t);
        }

        /// Assumes that C has `n = 3 * heads * 128` columns.
        fn kernelQkNormRope(
            map_a: TensorMap,
            map_b: TensorMap,
            norm: QkNormRope,
            m: u32,
            n: u32,
            k: u32,
        ) callconv(.{ .nvptx_kernel = .{
            .required_work_group_size = .{ .x = work_items },
            .min_work_groups_per_sm = 1,
        } }) void {
            const t = product(&map_a, &map_b, m, n, k, 0);
            normRope(norm, m, t);
        }

        const Product = struct {
            accumulators: Accumulators,
            tile_row: u32,
            tile_column: u32,
            first_row: u32,
            first_column: u32,
        };

        inline fn product(map_a: *const TensorMap, map_b: *const TensorMap, m: u32, n: u32, k: u32, gate_row: u32) Product {
            @setEvalBranchQuota(20_000);
            const tid = @workItemId(0);
            const lane = tid % 32;
            const warp = tid / 32;
            const tiles_m = (m + c.block_m - 1) / c.block_m;
            const tiles_n = (n + block_columns - 1) / block_columns;
            const tile = @workGroupId(0);
            const group_tiles = c.group_m * tiles_n;
            const first_m = tile / group_tiles * c.group_m;
            const rows_in_group = @min(c.group_m, tiles_m - first_m);
            const in_group = tile % group_tiles;
            const m_offset = (first_m + in_group % rows_in_group) * c.block_m;
            const column_tile = in_group / rows_in_group;
            const b_rows: [2]u32 = switch (epilogue) {
                .swiglu => .{ column_tile * b_box_rows, gate_row + column_tile * b_box_rows },
                else => .{ column_tile * c.block_n, column_tile * c.block_n + b_box_rows },
            };
            const k_tiles = (k + c.block_k - 1) / c.block_k;
            if (warp == 0 and nvptx.electSync(.all)) {
                inline for (&full, &empty) |*full_stage, *empty_stage| {
                    full_stage.init(1);
                    empty_stage.init(warps);
                }
                nvptx.fenceProxyAsync(.shared);
                inline for (0..c.stages) |stage| {
                    if (stage < k_tiles) load(stage, map_a, map_b, m_offset, b_rows, stage);
                }
            }
            nvptx.barrierSyncAligned(.work_group, null);
            const warp_m = warp / warps_n * c.warp_m;
            const warp_n = warp % warps_n;
            var accumulators: Accumulators = @splat(@splat(@splat(0)));
            var stage: u32 = 0;
            var phase: u1 = 0;
            while (!full[0].tryWaitParity(0)) {}
            var fragments = loadFragments(0, 0, warp_m, warp_n, lane);
            var k_tile: u32 = 0;
            while (k_tile < k_tiles) : (k_tile += 1) {
                inline for (0..k_steps) |k_step| {
                    const current = fragments;
                    if (k_step + 1 < k_steps) {
                        fragments = loadFragments(stage, k_step + 1, warp_m, warp_n, lane);
                    } else {
                        if (nvptx.electSync(.all)) _ = empty[stage].arrive();
                        if (warp == 0 and k_tile + c.stages < k_tiles and nvptx.electSync(.all)) {
                            while (!empty[stage].tryWaitParity(phase)) {}
                            load(stage, map_a, map_b, m_offset, b_rows, k_tile + c.stages);
                        }
                        stage += 1;
                        if (stage == c.stages) {
                            stage = 0;
                            phase ^= 1;
                        }
                    }
                    multiply(&accumulators, current);
                    if (k_step + 1 == k_steps and k_tile + 1 < k_tiles) {
                        while (!full[stage].tryWaitParity(phase)) {}
                        fragments = loadFragments(stage, 0, warp_m, warp_n, lane);
                    }
                }
            }
            return .{
                .accumulators = accumulators,
                .tile_row = m_offset,
                .tile_column = column_tile * block_columns,
                .first_row = m_offset + warp_m,
                .first_column = column_tile * block_columns + warp_n * (c.warp_n / if (epilogue == .swiglu) 2 else 1),
            };
        }

        inline fn load(
            stage: u32,
            map_a: *const TensorMap,
            map_b: *const TensorMap,
            m_offset: u32,
            b_rows: [2]u32,
            k_tile: u32,
        ) void {
            const k_offset = k_tile * c.block_k;
            _ = full[stage].arriveExpectTx(stage_bytes);
            asm volatile (
                \\cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%[a]], [%[map_a], {%[k], %[m]}], [%[bar]];
                \\cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%[b0]], [%[map_b], {%[k], %[n0]}], [%[bar]];
                \\cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%[b1]], [%[map_b], {%[k], %[n1]}], [%[bar]];
                :
                : [bar] "r" (&full[stage]),
                  [a] "r" (&tiles.a[stage]),
                  [b0] "r" (&tiles.b[stage][0]),
                  [b1] "r" (&tiles.b[stage][b_box_rows * c.block_k]),
                  [map_a] "l" (map_a),
                  [map_b] "l" (map_b),
                  [k] "r" (k_offset),
                  [m] "r" (m_offset),
                  [n0] "r" (b_rows[0]),
                  [n1] "r" (b_rows[1]),
                : .{ .memory = true });
        }

        /// The tensor copy moves 16-byte chunk `x` of each 128-byte line `y` to chunk `x ^ y % 8`
        /// (PTX ISA 9.4, 5.5.7).
        inline fn swizzle(row: u32, column: u32) u32 {
            return row * c.block_k + (column ^ ((row % 8) * chunk_elements));
        }

        inline fn bRow(warp_n: u32, comptime j: u32) u32 {
            return switch (epilogue) {
                .swiglu => blk: {
                    const half = j / (c.warp_n / 32);
                    const within = j % (c.warp_n / 32);
                    break :blk half * b_box_rows + warp_n * (c.warp_n / 2) + within * 16;
                },
                else => warp_n * c.warp_n + j * 16,
            };
        }

        const Fragments = struct {
            a: [c.warp_m / 16]@Vector(4, u32),
            b: [c.warp_n / 8]@Vector(2, u32),
        };

        // PTX ISA 9.4, sections 9.7.16.5.8 and 9.7.16.5.15 give the `mma` and `ldmatrix` layouts.
        // The 8-bit fragments of section 9.7.16.5.10 hold the same bytes as the BF16 fragments.
        inline fn loadFragments(stage: u32, comptime k_step: u32, warp_m: u32, warp_n: u32, lane: u32) Fragments {
            const a_tile = &tiles.a[stage];
            const b_tile = &tiles.b[stage];
            var fragments: Fragments = undefined;
            inline for (&fragments.a, 0..) |*fragment, i| {
                const row = warp_m + @as(u32, i) * 16 + lane % 16;
                const column = k_step * k_step_elements + lane / 16 * chunk_elements;
                fragment.* = nvptx.ldmatrix(.x4, false, &a_tile[swizzle(row, column)]);
            }
            inline for (0..c.warp_n / 16) |j| {
                const row = bRow(warp_n, j) + lane % 8 + lane / 16 * 8;
                const column = k_step * k_step_elements + ((lane / 8) % 2) * chunk_elements;
                const pair = nvptx.ldmatrix(.x4, false, &b_tile[swizzle(row, column)]);
                fragments.b[j * 2] = .{ pair[0], pair[1] };
                fragments.b[j * 2 + 1] = .{ pair[2], pair[3] };
            }
            return fragments;
        }

        inline fn multiply(accumulators: *Accumulators, fragments: Fragments) void {
            inline for (accumulators, fragments.a) |*row, a_fragment| {
                inline for (row, fragments.b) |*fragment, b_fragment| {
                    fragment.* = nvptx.mmaSync(mma_options, a_fragment, b_fragment, fragment.*);
                }
            }
        }

        /// The padding of 8 columns makes the rows of a warp store start in different banks.
        const staged_columns = block_columns + 8;
        const chunk_columns = block_columns / 8;
        const chunk_passes = c.block_m * chunk_columns / work_items;

        inline fn store(
            out: [*]addrspace(.global) Chunk,
            residual: ?[*]addrspace(.global) const Chunk,
            bias: ?[*]addrspace(.global) const u32,
            m: u32,
            n: u32,
            ldc: u32,
            t: Product,
        ) void {
            const tid = @workItemId(0);
            const lane = tid % 32;
            const staged: *addrspace(.shared) [c.block_m][staged_columns]u16 = @ptrCast(&shared_storage);
            const column_fragments = if (epilogue == .swiglu) c.warp_n / 16 else c.warp_n / 8;
            nvptx.barrierSyncAligned(.work_group, null);
            inline for (t.accumulators, 0..) |row_fragments, i| {
                inline for (0..2) |half| {
                    const row = t.first_row - t.tile_row + @as(u32, i) * 16 + lane / 4 + half * 8;
                    inline for (0..column_fragments) |j| {
                        const column = t.first_column - t.tile_column + @as(u32, j) * 8 + (lane % 4) * 2;
                        const v = row_fragments[j];
                        const pair: *addrspace(.shared) u32 = @ptrCast(@alignCast(&staged[row][column]));
                        const b: @Vector(2, f32) = if ((comptime epilogue.hasBias()) and t.tile_column + column < n)
                            device.unpack(bias.?[(t.tile_column + column) / 2])
                        else
                            @splat(0);
                        const sums = @Vector(2, f32){ v[half * 2] + b[0], v[half * 2 + 1] + b[1] };
                        pair.* = switch (epilogue) {
                            .bf16 => device.pack(v[half * 2], v[half * 2 + 1]),
                            .bias, .bias_gelu_tanh, .bias_gelu => device.pack(sums[0], sums[1]),
                            .swiglu => blk: {
                                const g = row_fragments[j + column_fragments];
                                break :blk device.pack(
                                    device.roundBf16(device.silu(device.roundBf16(g[half * 2]))) * device.roundBf16(v[half * 2]),
                                    device.roundBf16(device.silu(device.roundBf16(g[half * 2 + 1]))) * device.roundBf16(v[half * 2 + 1]),
                                );
                            },
                            .qk_norm_rope => unreachable,
                        };
                    }
                }
            }
            nvptx.barrierSyncAligned(.work_group, null);
            // The residual loads come before the stores, which may overwrite the residual.
            var shortcuts: [chunk_passes]Chunk = undefined;
            if (residual) |r| {
                inline for (0..chunk_passes) |pass| {
                    const index = tid + pass * work_items;
                    const row = t.tile_row + index / chunk_columns;
                    const column = t.tile_column + (index % chunk_columns) * 8;
                    if (row < m and column < n) {
                        shortcuts[pass] = r[(@as(u64, row) * ldc + column) / 8];
                    }
                }
            }
            inline for (0..chunk_passes) |pass| {
                const index = tid + pass * work_items;
                const local_row = index / chunk_columns;
                const row = t.tile_row + local_row;
                const column = t.tile_column + (index % chunk_columns) * 8;
                if (row < m and column < n) {
                    const values: *addrspace(.shared) const Chunk = @ptrCast(@alignCast(&staged[local_row][(index % chunk_columns) * 8]));
                    var chunk = values.*;
                    // The GELU runs here, where no accumulator is live, on the rounded sums.
                    switch (epilogue) {
                        .bias_gelu_tanh, .bias_gelu => inline for (0..4) |e| {
                            const v = device.unpack(chunk[e]);
                            chunk[e] = if (epilogue == .bias_gelu_tanh)
                                device.pack(device.geluTanh(v[0]), device.geluTanh(v[1]))
                            else
                                device.pack(device.gelu(v[0]), device.gelu(v[1]));
                        },
                        else => {},
                    }
                    if ((epilogue == .bf16 or epilogue == .bias) and residual != null) {
                        inline for (0..4) |e| {
                            const v = device.unpack(chunk[e]);
                            const shortcut = device.unpack(shortcuts[pass][e]);
                            chunk[e] = device.pack(v[0] + shortcut[0], v[1] + shortcut[1]);
                        }
                    }
                    out[(@as(u64, row) * ldc + column) / 8] = chunk;
                }
            }
        }

        /// `QkNormRopeInterleaved` sums four squares per work item and then adds the 32 sums by a
        /// butterfly. The warps add these sums in the same order, so the norms match bit for bit.
        inline fn normRope(norm: QkNormRope, m: u32, t: Product) void {
            const tid = @workItemId(0);
            const lane = tid % 32;
            const warp = tid / 32;
            const quad = lane % 4;
            const heads = norm.heads;
            const width = heads * head_dim;
            const first_head = t.tile_column / head_dim;
            const head = t.first_column / head_dim;
            const half_offset = t.first_column % head_dim;
            const values = first_head >= 2 * heads;
            // Element `[w][row][l]` holds the sum of the squares of values `4 l` to `4 l + 3`
            // in row `row` of the half of a head that warp `w` holds.
            const Sums = [warps][c.warp_m][c.warp_n / 4]f32;
            const sums: *addrspace(.shared) Sums = @ptrCast(&shared_storage);
            var scales: [c.warp_m / 16][2]f32 = undefined;
            nvptx.barrierSyncAligned(.work_group, null);
            if (!values) {
                inline for (t.accumulators, 0..) |row_fragments, i| {
                    inline for (0..2) |h| {
                        const row = @as(u32, i) * 16 + lane / 4 + h * 8;
                        inline for (row_fragments, 0..) |fragment, j| {
                            const pair = device.pack(fragment[h * 2], fragment[h * 2 + 1]);
                            const v = device.unpack(pair);
                            const next = device.unpack(@bitCast(device.shuffleXor(@bitCast(pair), 1)));
                            if (quad % 2 == 0) sums[warp][row][j * 2 + quad / 2] = v[0] * v[0] + v[1] * v[1] + next[0] * next[0] + next[1] * next[1];
                        }
                    }
                }
                nvptx.barrierSyncAligned(.work_group, null);
                inline for (&scales, 0..) |*row_scales, i| {
                    inline for (row_scales, 0..) |*scale, h| {
                        const row = @as(u32, i) * 16 + lane / 4 + h * 8;
                        // The work items of even `quad` hold the sums of `l = 2 j + quad / 2`. The
                        // butterfly step of mask 16 adds the sums of the two warps of the head, the
                        // steps of masks 8, 4 and 2 add elements of `partial`, and the step of
                        // mask 1 adds the sums of the work items two lanes apart.
                        var partial: [c.warp_n / 8]f32 = undefined;
                        inline for (&partial, 0..) |*sum, j| {
                            const l = @as(u32, j) * 2 + quad / 2;
                            sum.* = sums[warp][row][l] + sums[warp ^ 1][row][l];
                        }
                        comptime var step = partial.len / 2;
                        inline while (step > 0) : (step /= 2) {
                            inline for (0..step) |j| partial[j] += partial[j + step];
                        }
                        var sum = partial[0] + device.shuffleXor(partial[0], 2);
                        const odd = device.shuffleXor(sum, 1);
                        if (quad % 2 == 1) sum = odd;
                        scale.* = nvptx.math.rsqrtApproxF32(false, sum / @as(f32, head_dim) + norm.epsilon);
                    }
                }
                nvptx.barrierSyncAligned(.work_group, null);
            }
            const staged: *addrspace(.shared) [c.block_m][staged_columns]u16 = @ptrCast(&shared_storage);
            const weight: [*]addrspace(.global) const u32 = @ptrFromInt(if (head < heads) norm.q_weight else norm.k_weight);
            var weights: [c.warp_n / 8]u32 = undefined;
            if (!values) {
                inline for (&weights, 0..) |*w, j| w.* = weight[(half_offset + @as(u32, j) * 8 + quad * 2) / 2];
            }
            inline for (t.accumulators, 0..) |row_fragments, i| {
                inline for (0..2) |h| {
                    const local_row = t.first_row - t.tile_row + @as(u32, i) * 16 + lane / 4 + h * 8;
                    inline for (row_fragments, 0..) |fragment, j| {
                        const column = @as(u32, j) * 8 + quad * 2;
                        const pair: *addrspace(.shared) u32 = @ptrCast(@alignCast(&staged[local_row][t.first_column - t.tile_column + column]));
                        if (values) {
                            pair.* = device.pack(fragment[h * 2], fragment[h * 2 + 1]);
                        } else {
                            const scale = scales[i][h];
                            const w = device.unpack(weights[j]);
                            pair.* = device.pack(
                                w[0] * device.roundBf16(device.roundBf16(fragment[h * 2]) * scale),
                                w[1] * device.roundBf16(device.roundBf16(fragment[h * 2 + 1]) * scale),
                            );
                        }
                    }
                }
            }
            nvptx.barrierSyncAligned(.work_group, null);
            const rope: [*]addrspace(.global) const @Vector(4, f32) = @ptrFromInt(norm.rope);
            const queries = first_head < heads;
            const q: [*]addrspace(.global) Chunk = @ptrFromInt(norm.q);
            const kv: [*]addrspace(.global) Chunk = @ptrFromInt(norm.kv);
            const kv_rows: [*]addrspace(.global) const u32 = @ptrFromInt(norm.kv_rows);
            inline for (0..chunk_passes) |pass| {
                const index = tid + pass * work_items;
                const local_row = index / chunk_columns;
                const row = t.tile_row + local_row;
                if (row < m) {
                    const column = first_head * head_dim + (index % chunk_columns) * 8;
                    const chunk: *addrspace(.shared) const Chunk = @ptrCast(@alignCast(&staged[local_row][(index % chunk_columns) * 8]));
                    var rotated = chunk.*;
                    if (!values) {
                        const rotation = (column % head_dim) / 2;
                        const table = rope + @as(u64, row) * (head_dim / 4);
                        const cosines = table[rotation / 4];
                        const sines = table[(head_dim / 2 + rotation) / 4];
                        inline for (0..4) |e| {
                            const v = device.unpack(rotated[e]);
                            const real = v[0] * cosines[e] - v[1] * sines[e];
                            const imaginary = nvptx.math.fmaF32(.nearest_even, .{}, v[1], cosines[e], v[0] * sines[e]);
                            rotated[e] = device.pack(real, imaginary);
                        }
                    }
                    const target = if (queries)
                        @as(u64, row) * width + column
                    else
                        @as(u64, kv_rows[row]) * 2 * width + column - heads * head_dim;
                    (if (queries) q else kv)[target / 8] = rotated;
                }
            }
        }
    };
}
