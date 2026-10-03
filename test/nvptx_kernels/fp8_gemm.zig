//! The product `C = A B^T` of the E4M3 packs of `fp8.zig`, rounded to BF16 once. Work group `i` of
//! `g` computes the tiles `i`, `i + g`, `i + 2 g` and so on of `Raster`, and its last warp group
//! fills a pipeline of four stages that runs across the tiles.

const std = @import("std");
const nvptx = std.nvptx;
const cp_async = nvptx.cp_async;
const device = @import("device.zig");
const fp8 = @import("fp8.zig");
const gemm_tma = @import("gemm_tma.zig");
const Chunk = device.Chunk;

pub const Epilogue = gemm_tma.Epilogue;
pub const QkNormRope = gemm_tma.QkNormRope;
pub const TensorMap = gemm_tma.TensorMap;

/// The scale of K block `j` of row `r` of A lies at device address `a + j * a_stride + r`, and the
/// scale of the 128 x 128 tile `(i, j)` of B lies at `b + i * k / 128 + j`.
pub const Scales = extern struct {
    a: u64,
    b: u64,
    a_stride: u32,
};

pub const Copy = enum { cp_async, tma };

pub const Scaling = enum {
    /// The sum holds the products in units of the scales of the current K block, and the kernel
    /// multiplies the sum by the ratio of the old and the new scales at each K block.
    promote,
    block_scale,
};

/// Consecutive tiles walk down a panel of `panel` tile rows, one tile column after another, or along
/// a panel of `panel` tile columns when `columns` is not zero.
pub const Raster = extern struct {
    panel: u32,
    columns: u32,
};

pub const Tiling = struct {
    warp_m: comptime_int,
    warp_n: comptime_int,
};

pub const linear_tiling: Tiling = .{ .warp_m = 64, .warp_n = 64 };
pub const head_tiling: Tiling = .{ .warp_m = 32, .warp_n = head_dim };

pub const block_m = 128;
pub const block_n = 256;
pub const block_k = 64;
const stages = 4;
const head_dim = 128;
const row_chunks = block_k / 16;
const k_steps = block_k / 32;

pub fn Gemm(comptime copy: Copy, comptime scaling: Scaling, comptime epilogue: Epilogue, comptime c: Tiling) type {
    const warps_n = block_n / c.warp_n;
    const consumers = block_m / c.warp_m * warps_n;
    if (block_m % c.warp_m != 0 or block_n % c.warp_n != 0 or fp8.group % c.warp_n != 0)
        @compileError("the warp tiles must divide the work group tile and a weight tile");
    if (c.warp_m % 32 != 0 or c.warp_n % 32 != 0)
        @compileError("warp tiles hold groups of four 8-column fragments");
    if (epilogue == .qk_norm_rope and c.warp_n != head_dim) @compileError("the QK norm needs one head per warp");
    return struct {
        pub const name = "fp8_gemm_" ++ switch (copy) {
            .cp_async => @tagName(scaling) ++ (if (epilogue == .bf16) "" else "_" ++ @tagName(epilogue)) ++
                (if (scaling == .promote) "_sm89" else "_sm120") ++ "_cp_async",
            .tma => (if (scaling == .promote) "promote_" else "") ++ @tagName(epilogue) ++ "_sm120_tma",
        };
        pub const block_columns = if (epilogue == .swiglu) block_n / 2 else block_n;
        pub const b_box_rows = block_n / 2;
        pub const work_items = consumers * 32 + fillers;
        pub const dynamic_shared_bytes = @sizeOf(Layout);
        const fillers = 128;

        const stage_bytes = block_m * block_k + block_n * block_k + block_m;
        const Layout = struct {
            a: [stages][block_m * row_chunks]Chunk,
            b: [stages][block_n * row_chunks]Chunk,
            a_scales: [stages][block_m]u8,
            b_scales: [stages][2]u32,
            tile_scales: if (copy == .tma) [2][max_k_blocks]u8 else void,
        };
        pub const max_k_blocks = 128;
        comptime {
            if (@offsetOf(Layout, "a") % 1024 != 0 or @offsetOf(Layout, "b") % 1024 != 0)
                @compileError("the swizzle of a tensor copy needs tiles at multiples of 1024 bytes");
        }
        const Accumulators = [c.warp_m / 16][c.warp_n / 8]@Vector(4, f32);
        const halves = if (epilogue == .swiglu) 2 else 1;

        extern var shared_storage: [0]u8 align(1024) addrspace(.shared);
        const layout: *addrspace(.shared) Layout = @ptrCast(&shared_storage);
        var full: [stages]nvptx.Mbarrier addrspace(.shared) = undefined;
        var empty: [stages]nvptx.Mbarrier addrspace(.shared) = undefined;

        const Output = if (epilogue == .qk_norm_rope) QkNormRope else struct {
            out: [*]addrspace(.global) Chunk,
            residual: ?[*]addrspace(.global) const Chunk,
            ldc: u32,
        };

        const Problem = struct {
            scales: Scales,
            m: u32,
            n: u32,
            k: u32,
            gate_row: u32,
            raster: Raster,
            tiles_m: u32,
            tiles_n: u32,

            inline fn init(scales: Scales, m: u32, n: u32, k: u32, gate_row: u32, raster: Raster) Problem {
                return .{
                    .scales = scales,
                    .m = m,
                    .n = n,
                    .k = k,
                    .gate_row = gate_row,
                    .raster = raster,
                    .tiles_m = (m + block_m - 1) / block_m,
                    .tiles_n = (n + block_columns - 1) / block_columns,
                };
            }

            inline fn tileCount(p: Problem) u32 {
                return p.tiles_m * p.tiles_n;
            }

            inline fn tileAt(p: Problem, index: u32) Tile {
                var tile_m: u32 = undefined;
                var tile_n: u32 = undefined;
                if (p.raster.columns == 0) {
                    const first = index / (p.raster.panel * p.tiles_n) * p.raster.panel;
                    const height = @min(p.raster.panel, p.tiles_m - first);
                    const within = index - first * p.tiles_n;
                    tile_m = first + within % height;
                    tile_n = within / height;
                } else {
                    const first = index / (p.raster.panel * p.tiles_m) * p.raster.panel;
                    const width = @min(p.raster.panel, p.tiles_n - first);
                    const within = index - first * p.tiles_m;
                    tile_n = first + within % width;
                    tile_m = within / width;
                }
                return .{
                    .m_offset = tile_m * block_m,
                    .column_offset = tile_n * block_columns,
                    .b_rows = switch (epilogue) {
                        .bf16, .qk_norm_rope => .{ tile_n * block_n, tile_n * block_n + b_box_rows },
                        .swiglu => .{ tile_n * b_box_rows, p.gate_row + tile_n * b_box_rows },
                    },
                };
            }

            inline fn weightRows(p: Problem) u32 {
                return switch (epilogue) {
                    .bf16, .qk_norm_rope => p.n,
                    .swiglu => 2 * p.gate_row,
                };
            }

            /// Rows past the end of B occur only in the zero-filled half of the last tile.
            inline fn weightOffset(p: Problem, row: u32) u32 {
                return @min(row, p.weightRows() - 1) / fp8.group * (p.k / fp8.group);
            }

            inline fn weightScale(p: Problem, row: u32, block: u32) u32 {
                const source: [*]addrspace(.global) const u8 = @ptrFromInt(p.scales.b);
                return source[p.weightOffset(row) + block];
            }
        };

        const Tile = struct {
            m_offset: u32,
            column_offset: u32,
            b_rows: [2]u32,
        };

        pub const kernel = if (epilogue == .qk_norm_rope) kernelNorm else kernelLinear;

        const Operand = if (copy == .tma) TensorMap else [*]addrspace(.global) const u8;
        const Operands = struct {
            a: if (copy == .tma) *const TensorMap else Operand,
            b: if (copy == .tma) *const TensorMap else Operand,

            inline fn init(a: *const Operand, b: *const Operand) Operands {
                return if (copy == .tma) .{ .a = a, .b = b } else .{ .a = a.*, .b = b.* };
            }
        };

        /// Assumes that `k` is a multiple of 128 and that `n` and `ldc` are multiples of 8.
        fn kernelLinear(
            a: Operand,
            b: Operand,
            scales: Scales,
            out: [*]addrspace(.global) Chunk,
            residual: ?[*]addrspace(.global) const Chunk,
            m: u32,
            n: u32,
            k: u32,
            ldc: u32,
            gate_row: u32,
            raster: Raster,
        ) callconv(.{ .nvptx_kernel = .{
            .required_work_group_size = .{ .x = work_items },
            .min_work_groups_per_sm = 1,
        } }) void {
            const problem: Problem = .init(scales, m, n, k, gate_row, raster);
            specialize(.init(&a, &b), problem, .{ .out = out, .residual = residual, .ldc = ldc });
        }

        /// Assumes that C has `n = 3 * heads * 128` columns.
        fn kernelNorm(
            a: Operand,
            b: Operand,
            scales: Scales,
            norm: QkNormRope,
            m: u32,
            n: u32,
            k: u32,
            raster: Raster,
        ) callconv(.{ .nvptx_kernel = .{
            .required_work_group_size = .{ .x = work_items },
            .min_work_groups_per_sm = 1,
        } }) void {
            specialize(.init(&a, &b), .init(scales, m, n, k, 0, raster), norm);
        }

        /// Stage `s` is full after its copies and empty after every multiplying warp has read it.
        inline fn specialize(operands: Operands, problem: Problem, output: Output) void {
            const warp = @workItemId(0) / 32;
            if (@workItemId(0) == 0) {
                inline for (&full, &empty) |*full_stage, *empty_stage| {
                    full_stage.init(if (copy == .tma) 1 else fillers);
                    empty_stage.init(consumers);
                }
                if (copy == .tma) nvptx.fenceProxyAsync(.shared);
            }
            nvptx.barrierSyncAligned(.work_group, null);
            if (warp >= consumers) {
                switch (copy) {
                    .tma => {
                        asm volatile ("setmaxnreg.dec.sync.aligned.u32 40;" ::: .{ .memory = true });
                        if (warp == consumers) fill(operands, problem);
                    },
                    .cp_async => fillCopying(operands, problem),
                }
            } else {
                if (copy == .tma) asm volatile ("setmaxnreg.inc.sync.aligned.u32 232;" ::: .{ .memory = true });
                consume(problem, output);
            }
        }

        inline fn fill(operands: Operands, problem: Problem) void {
            const lane = @workItemId(0) % 32;
            const k_tiles = problem.k / block_k;
            const k_blocks = problem.k / fp8.group;
            var first_count: u32 = 0;
            var index = @workGroupId(0);
            while (index < problem.tileCount()) : ({
                index += nvptx.nctaid(0);
                first_count += k_tiles;
            }) {
                const tile = problem.tileAt(index);
                const a_scales = problem.scales.a + tile.m_offset;
                nvptx.barWarpSync(.all);
                var block = lane;
                while (block < k_blocks) : (block += 32) {
                    inline for (&layout.tile_scales, tile.b_rows) |*row, first| row[block] = @intCast(problem.weightScale(first, block));
                }
                nvptx.barWarpSync(.all);
                var k_tile: u32 = 0;
                while (lane == 0 and k_tile < k_tiles) : (k_tile += 1) {
                    const count = first_count + k_tile;
                    const stage = count % stages;
                    const parity: u1 = @truncate(count / stages);
                    while (!empty[stage].tryWaitParity(parity ^ 1)) {}
                    inline for (&layout.b_scales[stage], &layout.tile_scales) |*word, *row| word.* = row[k_tile / tiles_per_block];
                    _ = full[stage].arriveExpectTx(stage_bytes);
                    const scale_source = a_scales + @as(u64, k_tile * block_k / fp8.group) * problem.scales.a_stride;
                    asm volatile (
                        \\cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%[s]], [%[s_src]], %[size], [%[bar]];
                        \\cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%[a]], [%[map_a], {%[k], %[m]}], [%[bar]];
                        \\cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%[b0]], [%[map_b], {%[k], %[n0]}], [%[bar]];
                        \\cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%[b1]], [%[map_b], {%[k], %[n1]}], [%[bar]];
                        :
                        : [bar] "r" (&full[stage]),
                          [s] "r" (&layout.a_scales[stage]),
                          [s_src] "l" (scale_source),
                          [size] "n" (block_m),
                          [a] "r" (&layout.a[stage]),
                          [b0] "r" (&layout.b[stage][0]),
                          [b1] "r" (&layout.b[stage][b_box_rows * row_chunks]),
                          [map_a] "l" (operands.a),
                          [map_b] "l" (operands.b),
                          [k] "r" (k_tile * block_k),
                          [m] "r" (tile.m_offset),
                          [n0] "r" (tile.b_rows[0]),
                          [n1] "r" (tile.b_rows[1]),
                        : .{ .memory = true });
                }
            }
        }

        inline fn fillCopying(operands: Operands, problem: Problem) void {
            var filler: Filler = .init(problem, operands);
            var count: u32 = 0;
            while (filler.index < problem.tileCount()) : (count += 1) {
                const stage = count % stages;
                const parity: u1 = @truncate(count / stages);
                while (!empty[stage].testWaitParity(parity ^ 1)) {}
                filler.issue(problem, operands, stage);
                cp_async.mbarrierArriveNoinc(&full[stage]);
            }
            cp_async.waitAll();
        }

        const a_passes = block_m * row_chunks / fillers;
        const b_passes = block_n * row_chunks / fillers;
        const copy_rows = fillers / row_chunks;

        /// A copy of a row past the end of A or B reads no bytes. The weight scale copy reads its
        /// word only up to the scale, so that it stays inside the scales of B.
        const Filler = struct {
            index: u32,
            k_tile: u32,
            column: u32,
            row: u32,
            a_sources: [a_passes][*]addrspace(.global) const u8,
            a_bytes: [a_passes]u32,
            b_sources: [b_passes][*]addrspace(.global) const u8,
            b_bytes: [b_passes]u32,
            scale_source: [*]addrspace(.global) const u8,
            weight_offset: u32,

            inline fn init(problem: Problem, operands: Operands) Filler {
                const id = fillerId();
                var filler: Filler = undefined;
                filler.index = @workGroupId(0);
                filler.column = id % row_chunks;
                filler.row = id / row_chunks;
                filler.start(problem, operands);
                return filler;
            }

            inline fn fillerId() u32 {
                return @workItemId(0) - consumers * 32;
            }

            inline fn start(filler: *Filler, problem: Problem, operands: Operands) void {
                filler.k_tile = 0;
                if (filler.index >= problem.tileCount()) return;
                const tile = problem.tileAt(filler.index);
                const k = problem.k;
                inline for (&filler.a_sources, &filler.a_bytes, 0..) |*source, *bytes, pass| {
                    const row = tile.m_offset + filler.row + @as(u32, pass) * copy_rows;
                    bytes.* = if (row < problem.m) 16 else 0;
                    source.* = operands.a + @as(u64, @min(row, problem.m - 1)) * k + filler.column * 16;
                }
                inline for (&filler.b_sources, &filler.b_bytes, 0..) |*source, *bytes, pass| {
                    const half = pass / (b_passes / 2);
                    const row = tile.b_rows[half] + filler.row + @as(u32, pass % (b_passes / 2)) * copy_rows;
                    bytes.* = if (row < problem.weightRows()) 16 else 0;
                    source.* = operands.b + @as(u64, @min(row, problem.weightRows() - 1)) * k + filler.column * 16;
                }
                const id = fillerId();
                filler.scale_source = @ptrFromInt(problem.scales.a + tile.m_offset + @as(u32, @min(id, 7)) * 16);
                filler.weight_offset = problem.weightOffset(if (id % 2 == 0) tile.b_rows[0] else tile.b_rows[1]);
            }

            inline fn issue(filler: *Filler, problem: Problem, operands: Operands, stage: u32) void {
                const id = fillerId();
                if (filler.index >= problem.tileCount()) return;
                const offset = filler.k_tile * block_k;
                inline for (filler.a_sources, filler.a_bytes, 0..) |source, bytes, pass| {
                    const row = filler.row + @as(u32, pass) * copy_rows;
                    const from: *addrspace(.global) const Chunk = @ptrCast(@alignCast(source + offset));
                    cp_async.copyZeroFill(.cache_global_level, &layout.a[stage][swizzle(row, filler.column)], from, bytes);
                }
                inline for (filler.b_sources, filler.b_bytes, 0..) |source, bytes, pass| {
                    const half = pass / (b_passes / 2);
                    const row = @as(u32, half) * b_box_rows + filler.row + @as(u32, pass % (b_passes / 2)) * copy_rows;
                    const from: *addrspace(.global) const Chunk = @ptrCast(@alignCast(source + offset));
                    cp_async.copyZeroFill(.cache_global_level, &layout.b[stage][swizzle(row, filler.column)], from, bytes);
                }
                if (id < block_m / 16) {
                    const scales: *addrspace(.shared) [block_m / 16]Chunk = @ptrCast(@alignCast(&layout.a_scales[stage]));
                    const block = filler.k_tile * block_k / fp8.group;
                    const from: *addrspace(.global) const Chunk = @ptrCast(@alignCast(filler.scale_source + @as(u64, block) * problem.scales.a_stride));
                    cp_async.copy(.cache_global_level, &scales[id], from);
                } else if (id < block_m / 16 + 2) {
                    const index = filler.weight_offset + filler.k_tile / tiles_per_block;
                    const words: [*]addrspace(.global) const u32 = @ptrFromInt(problem.scales.b);
                    cp_async.copyZeroFill(.cache_all_levels, &layout.b_scales[stage][id % 2], &words[index / 4], index % 4 + 1);
                }
                filler.k_tile += 1;
                if (filler.k_tile == problem.k / block_k) {
                    filler.index += nvptx.nctaid(0);
                    filler.start(problem, operands);
                }
            }
        };

        /// The 16-byte chunk `column` of row `row` lies at chunk `column ^ row / 2 % 4` of the
        /// row, as the 64-byte swizzle of a tensor copy places it (PTX ISA 9.4, section 5.5.7).
        inline fn swizzle(row: u32, column: u32) u32 {
            return row * row_chunks + (column ^ ((row / 2) % row_chunks));
        }

        inline fn bRow(warp_n: u32, comptime j: u32) u32 {
            return switch (epilogue) {
                .bf16, .qk_norm_rope => warp_n * c.warp_n + j * 16,
                .swiglu => b_box_rows * (j / (c.warp_n / 32)) + warp_n * (c.warp_n / 2) + (j % (c.warp_n / 32)) * 16,
            };
        }

        /// The swizzle repeats every 8 rows, so fragments 16 rows apart lie 64 chunks apart.
        const Lanes = struct {
            a: [k_steps]u32,
            b: [k_steps]u32,
        };

        inline fn laneOffsets(warp_m: u32, warp_n: u32, lane: u32) Lanes {
            var lanes: Lanes = undefined;
            inline for (&lanes.a, &lanes.b, 0..) |*a, *b, step| {
                const k_step: u32 = step;
                a.* = swizzle(warp_m + lane % 16, k_step * 2 + lane / 16);
                b.* = swizzle(bRow(warp_n, 0) + lane % 8 + lane / 16 * 8, k_step * 2 + (lane / 8) % 2);
            }
            return lanes;
        }

        const Fragments = struct {
            a: [c.warp_m / 16]@Vector(4, u32),
            b: [c.warp_n / 8]@Vector(2, u32),
        };

        // The 8-bit fragments of `mma.m16n8k32` (PTX ISA 9.4, section 9.7.16.5.10) hold the same
        // bytes as the BF16 fragments of `mma.m16n8k16`, so the `ldmatrix` addresses are equal.
        inline fn loadFragments(stage: u32, comptime k_step: u32, lanes: Lanes) Fragments {
            var fragments: Fragments = undefined;
            inline for (&fragments.a, 0..) |*fragment, i| {
                fragment.* = nvptx.ldmatrix(.x4, false, &layout.a[stage][lanes.a[k_step] + i * 16 * row_chunks]);
            }
            inline for (0..c.warp_n / 16) |j| {
                const pair = nvptx.ldmatrix(.x4, false, &layout.b[stage][lanes.b[k_step] + (bRow(0, j) - bRow(0, 0)) * row_chunks]);
                fragments.b[j * 2] = .{ pair[0], pair[1] };
                fragments.b[j * 2 + 1] = .{ pair[2], pair[3] };
            }
            return fragments;
        }

        /// Byte `i` of word `a[h]` holds the scale of row `lane / 4 + 8 h` of A fragment `i`. With
        /// block scaling, work items 4i and 4i + 1 give the scales of rows i and i + 8 of an A
        /// fragment (PTX ISA 9.4, section 9.7.16.5.14), so `a[0]` holds row `lane / 4 + 8 (lane % 2)`.
        const BlockScales = struct {
            a: [if (scaling == .promote) 2 else 1]u32,
            b: [halves]u32,
        };

        inline fn rowScales(stage: u32, warp_m: u32, lane: u32) @FieldType(BlockScales, "a") {
            const scales = &layout.a_scales[stage];
            var words: @FieldType(BlockScales, "a") = @splat(0);
            inline for (&words, 0..) |*word, half| {
                const offset = if (scaling == .block_scale) (lane % 2) * 8 else half * 8;
                inline for (0..c.warp_m / 16) |i| {
                    word.* |= @as(u32, scales[warp_m + i * 16 + lane / 4 + offset]) << (8 * i);
                }
            }
            return words;
        }

        inline fn multiply(
            accumulators: *Accumulators,
            fragments: Fragments,
            scales: BlockScales,
            previous: ?BlockScales,
        ) void {
            inline for (accumulators, fragments.a, 0..) |*row, a_fragment, i| {
                if (previous) |old| {
                    inline for (row, 0..) |*fragment, j| {
                        const half = if (halves == 2) j / (c.warp_n / 16) else 0;
                        var pair: [2]f32 = undefined;
                        inline for (&pair, old.a, scales.a) |*value, word, new_word| {
                            value.* = ratioOf((word >> (8 * i)) & 0xFF, old.b[half], (new_word >> (8 * i)) & 0xFF, scales.b[half]);
                        }
                        fragment.* *= @Vector(4, f32){ pair[0], pair[0], pair[1], pair[1] };
                    }
                }
                inline for (row, fragments.b, 0..) |*fragment, b_fragment, j| {
                    fragment.* = switch (scaling) {
                        .promote => fp8.mmaE4m3(a_fragment, b_fragment, fragment.*),
                        .block_scale => fp8.mmaE4m3BlockScale(
                            i,
                            if (halves == 2) j / (c.warp_n / 16) else 0,
                            a_fragment,
                            b_fragment,
                            fragment.*,
                            scales.a[0],
                            if (halves == 2) scales.b[0] | (scales.b[1] << 8) else scales.b[0],
                        ),
                    };
                }
            }
        }

        /// Assumes that the scale products of two consecutive K blocks of an output differ by a
        /// factor from 2^-126 to 2^127, so that the ratio is exact.
        inline fn ratio(previous: u32, current: u32) f32 {
            const exponent = @as(i32, @intCast(previous)) - @as(i32, @intCast(current)) + 127;
            return @bitCast(@as(u32, @intCast(std.math.clamp(exponent, 1, 254))) << 23);
        }

        const Factors = [c.warp_m / 16][halves]@Vector(4, f32);

        inline fn factorsOf(scales: BlockScales, comptime factor: anytype, other: BlockScales) Factors {
            var factors: Factors = undefined;
            inline for (&factors, 0..) |*row, i| {
                inline for (row, 0..) |*vector, half| {
                    var pair: [2]f32 = undefined;
                    inline for (&pair, scales.a, other.a) |*value, word, other_word| {
                        value.* = factor((word >> (8 * i)) & 0xFF, scales.b[half], (other_word >> (8 * i)) & 0xFF, other.b[half]);
                    }
                    vector.* = .{ pair[0], pair[0], pair[1], pair[1] };
                }
            }
            return factors;
        }

        inline fn ratioOf(a: u32, b: u32, next_a: u32, next_b: u32) f32 {
            return ratio(a + b, next_a + next_b);
        }

        inline fn scaleOf(a: u32, b: u32, _: u32, _: u32) f32 {
            return fp8.scaleOf(a) * fp8.scaleOf(b);
        }

        const tiles_per_block = fp8.group / block_k;

        const Warp = struct {
            lane: u32,
            m: u32,
            n: u32,
            lanes: Lanes,
            box: u32,

            inline fn init() Warp {
                const tid = @workItemId(0);
                const lane = tid % 32;
                const warp_m = tid / 32 / warps_n * c.warp_m;
                const warp_n = (tid / 32) % warps_n;
                return .{
                    .lane = lane,
                    .m = warp_m,
                    .n = warp_n,
                    .lanes = laneOffsets(warp_m, warp_n, lane),
                    .box = if (halves == 2) 0 else warp_n * c.warp_n / b_box_rows,
                };
            }

            /// With `cp.async`, byte `(weight_offsets[half] + block) % 4` holds the scale.
            inline fn blockScales(warp: Warp, stage: u32, weight_offsets: [halves]u32, block: u32) BlockScales {
                var scales: BlockScales = .{ .a = rowScales(stage, warp.m, warp.lane), .b = undefined };
                inline for (&scales.b, weight_offsets, 0..) |*byte, offset, half| {
                    const word = layout.b_scales[stage][warp.box + half];
                    byte.* = if (copy == .tma) word else (word >> @intCast(((offset + block) % 4) * 8)) & 0xFF;
                }
                return scales;
            }

            inline fn weightOffsets(warp: Warp, problem: Problem, tile: Tile) [halves]u32 {
                var offsets: [halves]u32 = undefined;
                if (copy == .cp_async) {
                    inline for (&offsets, 0..) |*offset, half| {
                        const row = if (halves == 2)
                            tile.b_rows[half]
                        else if (warp.box == 0) tile.b_rows[0] + warp.n * c.warp_n else tile.b_rows[1] + warp.n * c.warp_n - b_box_rows;
                        offset.* = problem.weightOffset(row);
                    }
                }
                return offsets;
            }

            inline fn finish(
                warp: Warp,
                problem: Problem,
                output: Output,
                tile: Tile,
                scales: BlockScales,
                accumulators: *Accumulators,
            ) void {
                if (scaling == .promote) {
                    const factors = factorsOf(scales, scaleOf, scales);
                    inline for (accumulators, factors) |*row, row_factors| {
                        inline for (row, 0..) |*fragment, j| fragment.* *= row_factors[if (halves == 2) j / (c.warp_n / 16) else 0];
                    }
                }
                const first_row = tile.m_offset + warp.m;
                const first_column = tile.column_offset + warp.n * (c.warp_n / halves);
                switch (epilogue) {
                    .bf16 => if (output.residual) |residual| {
                        store(true, output.out, residual, output.ldc, problem, first_row, first_column, warp.lane, accumulators);
                    } else {
                        store(false, output.out, undefined, output.ldc, problem, first_row, first_column, warp.lane, accumulators);
                    },
                    .swiglu => store(false, output.out, undefined, output.ldc, problem, first_row, first_column, warp.lane, accumulators),
                    .qk_norm_rope => normRope(output, problem.m, first_row, first_column, warp.lane, accumulators),
                }
            }
        };

        inline fn multiplyStage(
            warp: Warp,
            accumulators: *Accumulators,
            scales: *BlockScales,
            count: u32,
            k_tile: u32,
            weight_offsets: [halves]u32,
        ) void {
            const stage = count % stages;
            if (k_tile % tiles_per_block != 0) {
                inline for (0..k_steps) |k_step| multiply(accumulators, loadFragments(stage, k_step, warp.lanes), scales.*, null);
                return;
            }
            const next = warp.blockScales(stage, weight_offsets, k_tile / tiles_per_block);
            if (scaling == .promote and k_tile > 0) {
                multiply(accumulators, loadFragments(stage, 0, warp.lanes), next, scales.*);
            } else {
                multiply(accumulators, loadFragments(stage, 0, warp.lanes), next, null);
            }
            inline for (1..k_steps) |k_step| multiply(accumulators, loadFragments(stage, k_step, warp.lanes), next, null);
            scales.* = next;
        }

        inline fn consume(problem: Problem, output: Output) void {
            @setEvalBranchQuota(20_000);
            const warp: Warp = .init();
            const k_tiles = problem.k / block_k;
            var scales: BlockScales = undefined;
            var count: u32 = 0;
            var index = @workGroupId(0);
            while (index < problem.tileCount()) : (index += nvptx.nctaid(0)) {
                const tile = problem.tileAt(index);
                const weight_offsets = warp.weightOffsets(problem, tile);
                var accumulators: Accumulators = @splat(@splat(@splat(0)));
                var k_tile: u32 = 0;
                while (k_tile < k_tiles) : (k_tile += 1) {
                    const stage = count % stages;
                    const parity: u1 = @truncate(count / stages);
                    switch (copy) {
                        .tma => while (!full[stage].tryWaitParity(parity)) {},
                        .cp_async => while (!full[stage].testWaitParity(parity)) {},
                    }
                    multiplyStage(warp, &accumulators, &scales, count, k_tile, weight_offsets);
                    nvptx.barWarpSync(.all);
                    if (warp.lane == 0) _ = empty[stage].arrive();
                    count += 1;
                }
                warp.finish(problem, output, tile, scales, &accumulators);
            }
        }

        /// Work item `q` of a quad gets word `q` of the four words of each work item of the quad,
        /// in the order of the work items.
        inline fn transposeQuad(x: [4]u32, quad: u32) Chunk {
            const upper = quad >= 2;
            var y = x;
            inline for (0..2) |e| {
                const got: u32 = @bitCast(device.shuffleXor(@bitCast(if (upper) x[e] else x[e + 2]), 2));
                if (upper) y[e] = got else y[e + 2] = got;
            }
            const odd = quad % 2 == 1;
            var z = y;
            inline for (0..2) |e| {
                const got: u32 = @bitCast(device.shuffleXor(@bitCast(if (odd) y[2 * e] else y[2 * e + 1]), 1));
                if (odd) z[2 * e] = got else z[2 * e + 1] = got;
            }
            return z;
        }

        inline fn storeChunk(out: [*]addrspace(.global) Chunk, index: u64, chunk: Chunk) void {
            asm volatile ("st.global.cs.v4.u32 [%[p]], {%[x], %[y], %[z], %[w]};"
                :
                : [p] "l" (out + index),
                  [x] "r" (chunk[0]),
                  [y] "r" (chunk[1]),
                  [z] "r" (chunk[2]),
                  [w] "r" (chunk[3]),
                : .{ .memory = true });
        }

        /// After a transpose in the quad, work item `q` holds the 8 columns of fragment `4 s + q`
        /// of each group `s` of four fragments.
        inline fn store(
            comptime add_residual: bool,
            out: [*]addrspace(.global) Chunk,
            residual: [*]addrspace(.global) const Chunk,
            ldc: u32,
            problem: Problem,
            first_row: u32,
            first_column: u32,
            lane: u32,
            accumulators: *const Accumulators,
        ) void {
            const column_fragments = c.warp_n / 8 / halves;
            const quad = lane % 4;
            inline for (accumulators, 0..) |row_fragments, i| {
                inline for (0..2) |half| {
                    const row = first_row + @as(u32, i) * 16 + lane / 4 + half * 8;
                    var pairs: [column_fragments]u32 = undefined;
                    inline for (&pairs, 0..) |*pair, j| {
                        const v = row_fragments[j];
                        pair.* = switch (epilogue) {
                            .bf16 => device.pack(v[half * 2], v[half * 2 + 1]),
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
                    inline for (0..column_fragments / 4) |group_index| {
                        var chunk = transposeQuad(pairs[group_index * 4 ..][0..4].*, quad);
                        const column = first_column + (@as(u32, group_index) * 4 + quad) * 8;
                        const index = (@as(u64, row) * ldc + column) / 8;
                        if (row < problem.m and column < problem.n) {
                            if (add_residual) {
                                const shortcuts = residual[index];
                                inline for (0..4) |e| {
                                    const v = device.unpack(chunk[e]);
                                    const shortcut = device.unpack(shortcuts[e]);
                                    chunk[e] = device.pack(v[0] + shortcut[0], v[1] + shortcut[1]);
                                }
                            }
                            storeChunk(out, index, chunk);
                        }
                    }
                }
            }
        }

        /// `qk_norm_rope.QkNormRopeInterleaved` sums four squares per work item and adds the 32 sums
        /// by a butterfly. Work items of even `quad` hold the sums of `l = 2 j + quad / 2`, so the
        /// steps of masks 16 to 2 add elements of `partial` and mask 1 adds lanes two apart.
        inline fn normRope(
            norm: QkNormRope,
            m: u32,
            first_row: u32,
            first_column: u32,
            lane: u32,
            accumulators: *const Accumulators,
        ) void {
            const quad = lane % 4;
            const heads = norm.heads;
            const width = heads * head_dim;
            const head = first_column / head_dim;
            const values = head >= 2 * heads;
            const queries = head < heads;
            const weight: [*]addrspace(.global) const u32 = @ptrFromInt(if (queries) norm.q_weight else norm.k_weight);
            var weights: [c.warp_n / 8]u32 = undefined;
            if (!values) {
                inline for (&weights, 0..) |*w, j| w.* = weight[@as(u32, j) * 4 + quad];
            }
            const rope: [*]addrspace(.global) const f32 = @ptrFromInt(norm.rope);
            const kv_rows: [*]addrspace(.global) const u32 = @ptrFromInt(norm.kv_rows);
            const q: [*]addrspace(.global) Chunk = @ptrFromInt(norm.q);
            const kv: [*]addrspace(.global) Chunk = @ptrFromInt(norm.kv);
            const destination = if (queries) q else kv;
            inline for (accumulators, 0..) |row_fragments, i| {
                inline for (0..2) |h| {
                    const row = first_row + @as(u32, i) * 16 + lane / 4 + h * 8;
                    const safe_row = @min(row, m - 1);
                    var pairs: [c.warp_n / 8]u32 = undefined;
                    inline for (&pairs, row_fragments) |*pair, fragment| pair.* = device.pack(fragment[h * 2], fragment[h * 2 + 1]);
                    if (!values) {
                        var partial: [c.warp_n / 8]f32 = undefined;
                        inline for (&partial, pairs) |*sum, pair| {
                            const v = device.unpack(pair);
                            const next = device.unpack(@bitCast(device.shuffleXor(@bitCast(pair), 1)));
                            sum.* = v[0] * v[0] + v[1] * v[1] + next[0] * next[0] + next[1] * next[1];
                        }
                        comptime var step = c.warp_n / 16;
                        inline while (step > 0) : (step /= 2) {
                            inline for (0..step) |j| partial[j] += partial[j + step];
                        }
                        var sum = partial[0] + device.shuffleXor(partial[0], 2);
                        const even = device.shuffleXor(sum, 1);
                        if (quad % 2 == 1) sum = even;
                        const scale = nvptx.math.rsqrtApproxF32(false, sum / @as(f32, head_dim) + norm.epsilon);
                        const table = rope + @as(u64, safe_row) * head_dim;
                        inline for (&pairs, weights, 0..) |*pair, w_pair, j| {
                            const v = device.unpack(pair.*);
                            const w = device.unpack(w_pair);
                            const a = device.roundBf16(w[0] * device.roundBf16(v[0] * scale));
                            const b = device.roundBf16(w[1] * device.roundBf16(v[1] * scale));
                            const rotation = @as(u32, j) * 4 + quad;
                            const cosine = table[rotation];
                            const sine = table[head_dim / 2 + rotation];
                            const real = a * cosine - b * sine;
                            const imaginary = nvptx.math.fmaF32(.nearest_even, .{}, b, cosine, a * sine);
                            pair.* = device.pack(real, imaginary);
                        }
                    }
                    const line = if (queries)
                        @as(u64, safe_row) * width + first_column
                    else
                        @as(u64, kv_rows[safe_row]) * 2 * width + first_column - heads * head_dim;
                    inline for (0..c.warp_n / 32) |group_index| {
                        const chunk = transposeQuad(pairs[group_index * 4 ..][0..4].*, quad);
                        if (row < m) storeChunk(destination, (line + (@as(u32, group_index) * 4 + quad) * 8) / 8, chunk);
                    }
                }
            }
        }
    };
}
