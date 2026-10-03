//! The kernels compute attention over segments of rows with grouped-query heads by the forward
//! pass of FlashAttention-2 (Dao, 2023), with the softmax computed online in base 2. A segment is
//! a run of query rows that attends to one run of key rows. In a causal segment, query `i` of
//! `q_length` sees keys `0` to `i + k_length - q_length` of the run; otherwise each query sees
//! every key of the run. Query head `h` reads key and value head `h / group`. Q, K, V and O rows
//! have their own strides, so that the kernel reads the values in place from the output of the
//! QKV projection. A head of fewer values than `head_dim` fills the first columns of the tiles,
//! and the kernel skips the products of the other columns.

const std = @import("std");
const builtin = @import("builtin");
const nvptx = std.nvptx;
const cp_async = nvptx.cp_async;

pub const Tiling = struct {
    head_dim: comptime_int,
    block_m: comptime_int,
    block_n: comptime_int,
    stages: comptime_int,
    min_work_groups_per_sm: comptime_int,
    /// Bulk tensor copies (TMA), which need compute capability 12.0, fill the tiles and signal one
    /// mbarrier per stage. Otherwise every work item copies chunks with `cp.async`.
    tensor_copies: bool = false,
    /// The values of one head in Q, K, V and O, a multiple of 8 and at most `head_dim`. The copies
    /// fill each tile row with `head_dim` values from the start of the head, so that they read
    /// the next head or the next row after a shorter head.
    dims: ?comptime_int = null,
};

pub const vision_tiling: Tiling = .{
    .head_dim = 128,
    .block_m = 128,
    .block_n = 64,
    .stages = 2,
    .min_work_groups_per_sm = 1,
    .dims = 72,
};

pub const tensor_copy_tiling: Tiling = .{
    .head_dim = 128,
    .block_m = 128,
    .block_n = 64,
    .stages = 3,
    .min_work_groups_per_sm = 1,
    .tensor_copies = true,
};

/// The 128 bytes of a `CUtensorMap`.
pub const TensorMap = extern struct {
    opaque_bytes: [128]u8 align(64),
};

pub const kernels = .{
    Attention(.{
        .head_dim = 128,
        .block_m = 128,
        .block_n = 64,
        .stages = 2,
        .min_work_groups_per_sm = 1,
    }),
    Attention(vision_tiling),
};

pub const Segment = extern struct {
    q_first_row: u32,
    q_length: u32,
    k_first_row: u32,
    k_length: u32,
    causal: u32,
    /// The index of the first work group of the segment along the grid x dimension.
    first_tile: u32,
    reserved: [2]u32 = .{ 0, 0 },
};

comptime {
    if (builtin.cpu.arch.isNvptx() and @import("root") == @This()) {
        for (kernels) |Kernel| @export(&Kernel.kernel, .{ .name = Kernel.name });
    }
}

/// A chunk holds eight bf16 elements. One `cp.async` copy and one `ldmatrix` row move a chunk.
const Chunk = @Vector(4, u32);
const Accumulator = @Vector(4, f32);
/// A row pair holds values of the rows `lane / 4` and `lane / 4 + 8` of the 16 rows of a warp.
const RowPair = @Vector(2, f32);

const mma_options: nvptx.matrix.MmaOptions = .{
    .shape = .m16n8k16,
    .accumulator = .f32,
    .element = .bf16,
};

pub fn Attention(comptime c: Tiling) type {
    const row_chunks = c.head_dim / 8;
    const dims = c.dims orelse c.head_dim;
    const head_chunks = dims / 8;
    const k_steps = (dims + 15) / 16;
    const warps = c.block_m / 16;
    const copy_rows = warps * 32 / row_chunks;
    if (c.head_dim != 64 and c.head_dim != 128) @compileError("head_dim must be 64 or 128");
    if (dims % 8 != 0 or dims > c.head_dim) @compileError("dims must be a multiple of 8 of at most head_dim");
    if (c.block_m % 16 != 0 or c.block_n % 16 != 0)
        @compileError("block_m and block_n must be multiples of 16");
    if (copy_rows % 8 != 0 or c.block_m % copy_rows != 0 or c.block_n % copy_rows != 0)
        @compileError("each work item must copy chunks of the same swizzled column");
    if (c.stages < 2) @compileError("the copy pipeline needs at least two shared buffers");
    if (c.tensor_copies and (c.block_m > 256 or c.block_n > 256))
        @compileError("a tensor copy moves at most 256 rows");
    return struct {
        pub const name = std.fmt.comptimePrint("attention_segments_d{d}_{s}", .{
            dims,
            if (c.tensor_copies) "sm120_tma" else "sm80_cp_async",
        });
        pub const tensor_copies = c.tensor_copies;
        pub const block_n = c.block_n;
        pub const head_dim = dims;
        pub const block_m = c.block_m;
        pub const work_items = warps * 32;
        const threads = work_items;
        pub const dynamic_shared_bytes = @sizeOf(Tiles);

        /// When the Q tile fits, it occupies the last stage until the first refill of that stage.
        const query_in_stage = c.block_m <= 2 * c.block_n;
        const Stage = struct {
            k: [c.block_n * row_chunks]Chunk,
            v: [c.block_n * row_chunks]Chunk,
        };
        const Tiles = struct {
            stages: [c.stages]Stage,
            q: if (query_in_stage) void else [c.block_m * row_chunks]Chunk,
        };
        const Query = [k_steps]@Vector(4, u32);
        const Scores = [c.block_n / 8]Accumulator;
        const Probabilities = [c.block_n / 16]@Vector(4, u32);
        const Output = [2 * k_steps]Accumulator;
        const Rows = struct {
            output: Output,
            max: RowPair,
            sum: RowPair,
        };

        extern var shared_storage: [0]u8 align(1024) addrspace(.shared);
        const tiles: *addrspace(.shared) Tiles = @ptrCast(&shared_storage);
        var full: [c.stages]nvptx.Mbarrier addrspace(.shared) = undefined;
        var empty: [c.stages]nvptx.Mbarrier addrspace(.shared) = undefined;
        var query_full: nvptx.Mbarrier addrspace(.shared) = undefined;

        pub const kernel = if (c.tensor_copies) kernelTensorCopies else kernelAsyncCopies;
        const query_tile: *addrspace(.shared) [c.block_m * row_chunks]Chunk = if (query_in_stage)
            @ptrCast(&tiles.stages[c.stages - 1])
        else
            &tiles.q;

        /// Work group (x, y) computes one query tile of one segment for query head y. The row
        /// strides count 8-value chunks. `scale_log2` is the softmax scale times log2(e).
        fn kernelAsyncCopies(
            q: [*]addrspace(.global) const Chunk,
            k: [*]addrspace(.global) const Chunk,
            v: [*]addrspace(.global) const Chunk,
            o: [*]addrspace(.global) u32,
            segments: [*]addrspace(.global) const Segment,
            segment_count: u32,
            q_stride: u32,
            k_stride: u32,
            v_stride: u32,
            o_stride: u32,
            group: u32,
            scale_log2: f32,
        ) callconv(.{ .nvptx_kernel = .{
            .required_work_group_size = .{ .x = threads },
            .min_work_groups_per_sm = c.min_work_groups_per_sm,
        } }) void {
            @setEvalBranchQuota(20_000);
            const tid = @workItemId(0);
            const lane = tid % 32;
            const warp = tid / 32;
            std.debug.assert(@workGroupSize(0) == threads);
            const tile_index = @workGroupId(0);
            var index: u32 = 0;
            while (index + 1 < segment_count and tile_index >= segments[index + 1].first_tile) index += 1;
            const segment = segments[index];
            const queries = segment.q_length;
            const keys = segment.k_length;
            const causal = segment.causal != 0;
            const diagonal = keys - queries;
            const query_blocks = (queries + c.block_m - 1) / c.block_m;
            // The last query tiles of a segment have the most keys, so they start first.
            const query_block = query_blocks - 1 - (tile_index - segment.first_tile);
            const first_query = query_block * c.block_m;
            const head = @workGroupId(1);
            const kv_head = head / group;
            const key_end = if (causal) @min(first_query + c.block_m + diagonal, keys) else keys;
            const tile_count = (key_end + c.block_n - 1) / c.block_n;
            const unmasked_tiles = if (causal) (first_query + diagonal + 1) / c.block_n else keys / c.block_n;
            const copy: Copy = .{
                .slot = swizzle(tid / row_chunks, tid % row_chunks),
                .row = tid / row_chunks,
                .q = q + @as(u64, segment.q_first_row) * q_stride + head * head_chunks + tid % row_chunks,
                .k = k + @as(u64, segment.k_first_row) * k_stride + kv_head * head_chunks + tid % row_chunks,
                .v = v + @as(u64, segment.k_first_row) * v_stride + kv_head * head_chunks + tid % row_chunks,
                .q_stride = q_stride,
                .k_stride = k_stride,
                .v_stride = v_stride,
                .seq = queries,
                .keys = keys,
            };
            copy.rows(c.block_m, query_tile, copy.q, copy.q_stride, first_query, true, queries);
            inline for (0..c.stages - 1) |stage| {
                if (stage < tile_count) copy.tile(stage, stage);
                cp_async.commitGroup();
            }
            cp_async.waitGroup(c.stages - 2);
            nvptx.barrierSyncAligned(.work_group, null);
            const query = loadQuery(warp, lane);
            var rows: Rows = .{
                .output = @splat(@splat(0)),
                .max = @splat(-std.math.inf(f32)),
                .sum = @splat(0),
            };
            const first_row = first_query + warp * 16 + lane / 4;
            const all_keys: @Vector(2, u32) = @splat(keys);
            const row_limits: @Vector(2, u32) = if (causal)
                @min(@Vector(2, u32){ first_row + diagonal + 1, first_row + diagonal + 9 }, all_keys)
            else
                all_keys;
            var stage: u32 = 0;
            var tile: u32 = 0;
            while (tile < unmasked_tiles) : (tile += 1) {
                copy.next(stage, tile, tile_count);
                const scores = computeScores(query, stage, lane);
                rows = attend(rows, scores, stage, lane, scale_log2, false, undefined);
                stage = if (stage + 1 == c.stages) 0 else stage + 1;
            }
            while (tile < tile_count) : (tile += 1) {
                copy.next(stage, tile, tile_count);
                const scores = computeScores(query, stage, lane);
                const mask: Mask = .{ .first_key = tile * c.block_n, .row_limits = row_limits };
                rows = attend(rows, scores, stage, lane, scale_log2, true, mask);
                stage = if (stage + 1 == c.stages) 0 else stage + 1;
            }
            store(o + @as(u64, segment.q_first_row) * (o_stride * 4) + head * (head_chunks * 4), o_stride * 4, first_row, queries, lane, rows);
        }

        /// The tensor maps zero-fill the rows after the end of the tensors.
        fn kernelTensorCopies(
            q_map: TensorMap,
            k_map: TensorMap,
            v_map: TensorMap,
            o: [*]addrspace(.global) u32,
            segments: [*]addrspace(.global) const Segment,
            segment_count: u32,
            o_stride: u32,
            group: u32,
            scale_log2: f32,
        ) callconv(.{ .nvptx_kernel = .{
            .required_work_group_size = .{ .x = threads },
            .min_work_groups_per_sm = c.min_work_groups_per_sm,
        } }) void {
            @setEvalBranchQuota(20_000);
            const tid = @workItemId(0);
            const lane = tid % 32;
            const warp = tid / 32;
            const tile_index = @workGroupId(0);
            var index: u32 = 0;
            while (index + 1 < segment_count and tile_index >= segments[index + 1].first_tile) index += 1;
            const segment = segments[index];
            const queries = segment.q_length;
            const keys = segment.k_length;
            const causal = segment.causal != 0;
            const diagonal = keys - queries;
            const query_blocks = (queries + c.block_m - 1) / c.block_m;
            const query_block = query_blocks - 1 - (tile_index - segment.first_tile);
            const first_query = query_block * c.block_m;
            const head = @workGroupId(1);
            const kv_head = head / group;
            const key_end = if (causal) @min(first_query + c.block_m + diagonal, keys) else keys;
            const tile_count = (key_end + c.block_n - 1) / c.block_n;
            const unmasked_tiles = if (causal) (first_query + diagonal + 1) / c.block_n else keys / c.block_n;
            const copies: TensorCopies = .{
                .k_map = &k_map,
                .v_map = &v_map,
                .first_key = segment.k_first_row,
                .column = kv_head * dims,
            };
            if (warp == 0 and nvptx.electSync(.all)) {
                inline for (&full, &empty) |*full_stage, *empty_stage| {
                    full_stage.init(1);
                    empty_stage.init(warps);
                }
                query_full.init(1);
                nvptx.fenceProxyAsync(.shared);
                _ = query_full.arriveExpectTx(c.block_m * c.head_dim * 2);
                inline for (0..c.head_dim / 64) |half| {
                    tensorCopy(&q_map, &query_tile[half * c.block_m * 8], &query_full, head * dims + @as(u32, half) * 64, segment.q_first_row + first_query);
                }
                inline for (0..c.stages - 1) |stage| {
                    if (stage < tile_count) copies.tile(stage, stage);
                }
            }
            nvptx.barrierSyncAligned(.work_group, null);
            while (!query_full.tryWaitParity(0)) {}
            const query = loadQuery(warp, lane);
            // Every warp must read the query tile before the copies refill the last stage.
            nvptx.barrierSyncAligned(.work_group, null);
            if (warp == 0 and c.stages - 1 < tile_count and nvptx.electSync(.all)) copies.tile(c.stages - 1, c.stages - 1);
            var rows: Rows = .{
                .output = @splat(@splat(0)),
                .max = @splat(-std.math.inf(f32)),
                .sum = @splat(0),
            };
            const first_row = first_query + warp * 16 + lane / 4;
            const all_keys: @Vector(2, u32) = @splat(keys);
            const row_limits: @Vector(2, u32) = if (causal)
                @min(@Vector(2, u32){ first_row + diagonal + 1, first_row + diagonal + 9 }, all_keys)
            else
                all_keys;
            var stage: u32 = 0;
            var phase: u1 = 0;
            var tile: u32 = 0;
            while (tile < unmasked_tiles) : (tile += 1) {
                while (!full[stage].tryWaitParity(phase)) {}
                const scores = computeScores(query, stage, lane);
                rows = attend(rows, scores, stage, lane, scale_log2, false, undefined);
                copies.release(stage, phase, tile, tile_count, warp);
                stage += 1;
                if (stage == c.stages) {
                    stage = 0;
                    phase ^= 1;
                }
            }
            while (tile < tile_count) : (tile += 1) {
                while (!full[stage].tryWaitParity(phase)) {}
                const scores = computeScores(query, stage, lane);
                const mask: Mask = .{ .first_key = tile * c.block_n, .row_limits = row_limits };
                rows = attend(rows, scores, stage, lane, scale_log2, true, mask);
                copies.release(stage, phase, tile, tile_count, warp);
                stage += 1;
                if (stage == c.stages) {
                    stage = 0;
                    phase ^= 1;
                }
            }
            store(o + @as(u64, segment.q_first_row) * (o_stride * 4) + head * (head_chunks * 4), o_stride * 4, first_row, queries, lane, rows);
        }

        const TensorCopies = struct {
            k_map: *const TensorMap,
            v_map: *const TensorMap,
            first_key: u32,
            column: u32,

            inline fn tile(copies: TensorCopies, stage: u32, index: u32) void {
                _ = full[stage].arriveExpectTx(@sizeOf(Stage));
                const row = copies.first_key + index * c.block_n;
                inline for (0..c.head_dim / 64) |half| {
                    const column = copies.column + @as(u32, half) * 64;
                    tensorCopy(copies.k_map, &tiles.stages[stage].k[half * c.block_n * 8], &full[stage], column, row);
                    tensorCopy(copies.v_map, &tiles.stages[stage].v[half * c.block_n * 8], &full[stage], column, row);
                }
            }

            inline fn release(copies: TensorCopies, stage: u32, phase: u1, current: u32, tile_count: u32, warp: u32) void {
                if (nvptx.electSync(.all)) _ = empty[stage].arrive();
                const refill = current + c.stages;
                if (warp == 0 and refill < tile_count and nvptx.electSync(.all)) {
                    while (!empty[stage].tryWaitParity(phase)) {}
                    copies.tile(stage, refill);
                }
            }
        };

        inline fn tensorCopy(
            map: *const TensorMap,
            destination: *addrspace(.shared) Chunk,
            barrier: *addrspace(.shared) nvptx.Mbarrier,
            column: u32,
            row: u32,
        ) void {
            asm volatile ("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%[destination]], [%[map], {%[column], %[row]}], [%[barrier]];"
                :
                : [destination] "r" (destination),
                  [map] "l" (map),
                  [column] "r" (column),
                  [row] "r" (row),
                  [barrier] "r" (barrier),
                : .{ .memory = true });
        }

        const Copy = struct {
            slot: u32,
            row: u32,
            q: [*]addrspace(.global) const Chunk,
            k: [*]addrspace(.global) const Chunk,
            v: [*]addrspace(.global) const Chunk,
            q_stride: u32,
            k_stride: u32,
            v_stride: u32,
            seq: u32,
            keys: u32,

            inline fn rows(
                copy: Copy,
                comptime count: u32,
                destination: *addrspace(.shared) [count * row_chunks]Chunk,
                source: [*]addrspace(.global) const Chunk,
                stride: u32,
                first_row: u32,
                comptime bounded: bool,
                limit: u32,
            ) void {
                inline for (0..count / copy_rows) |pass| {
                    const row = first_row + copy.row + @as(u32, pass) * copy_rows;
                    const source_row = if (bounded) @min(row, limit - 1) else row;
                    const slot = copy.slot + @as(u32, pass) * copy_rows * row_chunks;
                    const chunk = @as(u64, source_row) * stride;
                    cp_async.copy(.cache_global_level, &destination[slot], &source[chunk]);
                }
            }

            inline fn tile(copy: Copy, stage: u32, index: u32) void {
                const first_key = index * c.block_n;
                if (first_key + c.block_n <= copy.keys) {
                    copy.rows(c.block_n, &tiles.stages[stage].k, copy.k, copy.k_stride, first_key, false, copy.keys);
                    copy.rows(c.block_n, &tiles.stages[stage].v, copy.v, copy.v_stride, first_key, false, copy.keys);
                } else {
                    copy.rows(c.block_n, &tiles.stages[stage].k, copy.k, copy.k_stride, first_key, true, copy.keys);
                    copy.rows(c.block_n, &tiles.stages[stage].v, copy.v, copy.v_stride, first_key, true, copy.keys);
                }
            }

            /// Waits for tile `current` in `stage` and refills the stage of the previous tile. The
            /// barrier shares the copies and orders the refill after the reads of that stage.
            inline fn next(copy: Copy, stage: u32, current: u32, tile_count: u32) void {
                cp_async.waitGroup(c.stages - 2);
                nvptx.barrierSyncAligned(.work_group, null);
                const refill = current + c.stages - 1;
                const refill_stage = if (stage == 0) c.stages - 1 else stage - 1;
                if (refill < tile_count) copy.tile(refill_stage, refill);
                cp_async.commitGroup();
            }
        };

        /// The XOR with `row % 8` puts the eight rows of an `ldmatrix` matrix in different groups
        /// of four banks.
        inline fn swizzle(row: u32, column: u32) u32 {
            return row * row_chunks + (column ^ (row % 8));
        }

        /// Assumes `row % 8 == lane % 8`. Tensor copies store each 64-value half of the rows as a
        /// block of 128-byte rows and swizzle the chunks as `swizzle` does (PTX ISA 9.4, 5.5.7).
        inline fn matrixChunk(
            tile: anytype,
            row: u32,
            column: u32,
            lane: u32,
        ) *addrspace(.shared) Chunk {
            if (c.tensor_copies) {
                const tile_rows = @typeInfo(@typeInfo(@TypeOf(tile)).pointer.child).array.len / row_chunks;
                return &tile[column / 8 * tile_rows * 8 + row * 8 + ((column % 8) ^ (lane % 8))];
            }
            return &tile[row * row_chunks + (column ^ (lane % 8))];
        }

        // PTX ISA 9.4 sections 9.7.16.5.8 and 9.7.16.5.15 give the `mma` and `ldmatrix` layouts.
        // The last two registers of an A fragment hold columns 8 to 15 of its 16 columns, which
        // lie past a head of `dims % 16 == 8` values and become zeros.
        inline fn loadQuery(warp: u32, lane: u32) Query {
            var query: Query = undefined;
            inline for (&query, 0..) |*fragment, k_step| {
                const row = warp * 16 + lane % 16;
                const column = @as(u32, k_step) * 2 + lane / 16;
                const chunk = matrixChunk(query_tile, row, column, lane);
                fragment.* = nvptx.ldmatrix(.x4, false, chunk);
                if (k_step * 16 + 8 == dims) {
                    fragment.*[2] = 0;
                    fragment.*[3] = 0;
                }
            }
            return query;
        }

        inline fn computeScores(query: Query, stage: u32, lane: u32) Scores {
            const tile = &tiles.stages[stage].k;
            var scores: Scores = @splat(@splat(0));
            inline for (query, 0..) |a, k_step| {
                // One `ldmatrix.x4` loads dimensions 0-15 of keys 0-7, then of keys 8-15.
                inline for (0..c.block_n / 16) |j| {
                    const row = @as(u32, j) * 16 + lane % 8 + lane / 16 * 8;
                    const column = @as(u32, k_step) * 2 + (lane / 8) % 2;
                    const chunk = matrixChunk(tile, row, column, lane);
                    const b = nvptx.ldmatrix(.x4, false, chunk);
                    const low: @Vector(2, u32) = .{ b[0], b[1] };
                    const high: @Vector(2, u32) = .{ b[2], b[3] };
                    scores[2 * j] = nvptx.mmaSync(mma_options, a, low, scores[2 * j]);
                    scores[2 * j + 1] = nvptx.mmaSync(mma_options, a, high, scores[2 * j + 1]);
                }
            }
            return scores;
        }

        const Mask = struct {
            first_key: u32,
            row_limits: @Vector(2, u32),
        };

        inline fn attend(
            rows: Rows,
            scores_in: Scores,
            stage: u32,
            lane: u32,
            scale_log2: f32,
            comptime masked: bool,
            mask: Mask,
        ) Rows {
            var scores = scores_in;
            if (masked) {
                const limits: @Vector(4, u32) = .{
                    mask.row_limits[0],
                    mask.row_limits[0],
                    mask.row_limits[1],
                    mask.row_limits[1],
                };
                inline for (&scores, 0..) |*fragment, n_tile| {
                    const key = mask.first_key + @as(u32, n_tile) * 8 + (lane % 4) * 2;
                    const keys: @Vector(4, u32) = .{ key, key + 1, key, key + 1 };
                    const minus_infinity: Accumulator = @splat(-std.math.inf(f32));
                    fragment.* = @select(f32, keys < limits, fragment.*, minus_infinity);
                }
            }
            var max = rows.max;
            inline for (scores) |fragment| {
                const even: RowPair = .{ fragment[0], fragment[2] };
                const odd: RowPair = .{ fragment[1], fragment[3] };
                max = @max(max, @max(even, odd));
            }
            max = @max(max, shuffleXor(max, 1));
            max = @max(max, shuffleXor(max, 2));
            const scale: RowPair = @splat(scale_log2);
            const scaled_max = max * scale;
            const correction = exp2(rows.max * scale - scaled_max);
            const factors: Accumulator = .{
                correction[0],
                correction[0],
                correction[1],
                correction[1],
            };
            var output = rows.output;
            inline for (&output) |*fragment| fragment.* *= factors;
            var sum = rows.sum * correction;
            const offsets: Accumulator = .{
                scaled_max[0],
                scaled_max[0],
                scaled_max[1],
                scaled_max[1],
            };
            const scale4: Accumulator = @splat(scale_log2);
            var probabilities: Probabilities = undefined;
            inline for (&probabilities, 0..) |*fragment, k_step| {
                const left = exp2(@mulAdd(Accumulator, scores[2 * k_step], scale4, -offsets));
                const right = exp2(@mulAdd(Accumulator, scores[2 * k_step + 1], scale4, -offsets));
                sum += RowPair{ left[0] + left[1], left[2] + left[3] };
                sum += RowPair{ right[0] + right[1], right[2] + right[3] };
                // Two accumulator tiles of 8 keys have the A layout of one tile of 16 keys.
                fragment.* = .{
                    @backingInt(nvptx.Bf16x2.fromF32(left[0], left[1])),
                    @backingInt(nvptx.Bf16x2.fromF32(left[2], left[3])),
                    @backingInt(nvptx.Bf16x2.fromF32(right[0], right[1])),
                    @backingInt(nvptx.Bf16x2.fromF32(right[2], right[3])),
                };
            }
            const tile = &tiles.stages[stage].v;
            inline for (probabilities, 0..) |a, k_step| {
                // One `ldmatrix.x4.trans` loads keys 0-15 of dimensions 0-7, then of 8-15.
                inline for (0..k_steps) |j| {
                    const row = @as(u32, k_step) * 16 + lane % 8 + ((lane / 8) % 2) * 8;
                    const column = @as(u32, j) * 2 + lane / 16;
                    const chunk = matrixChunk(tile, row, column, lane);
                    const b = nvptx.ldmatrix(.x4, true, chunk);
                    const low: @Vector(2, u32) = .{ b[0], b[1] };
                    const high: @Vector(2, u32) = .{ b[2], b[3] };
                    output[2 * j] = nvptx.mmaSync(mma_options, a, low, output[2 * j]);
                    output[2 * j + 1] = nvptx.mmaSync(mma_options, a, high, output[2 * j + 1]);
                }
            }
            return .{ .output = output, .max = max, .sum = sum };
        }

        inline fn store(
            o: [*]addrspace(.global) u32,
            stride: u32,
            first_row: u32,
            seq: u32,
            lane: u32,
            rows: Rows,
        ) void {
            var sum = rows.sum;
            sum += shuffleXor(sum, 1);
            sum += shuffleXor(sum, 2);
            const inverse = @as(RowPair, @splat(1)) / sum;
            inline for (rows.output[0..head_chunks], 0..) |fragment, n_tile| {
                const word = @as(u32, n_tile) * 4 + lane % 4;
                const top = @as(u64, first_row) * stride + word;
                if (first_row < seq) o[top] = @backingInt(nvptx.Bf16x2.fromF32(
                    fragment[0] * inverse[0],
                    fragment[1] * inverse[0],
                ));
                if (first_row + 8 < seq) o[top + 8 * @as(u64, stride)] = @backingInt(nvptx.Bf16x2.fromF32(
                    fragment[2] * inverse[1],
                    fragment[3] * inverse[1],
                ));
            }
        }
    };
}

inline fn shuffleXor(x: RowPair, comptime mask: u32) RowPair {
    const clamp: nvptx.ShuffleClamp = .{ .clamp = 31, .segment_mask = 0 };
    return .{
        nvptx.shflSync(.bfly, f32, x[0], mask, clamp, .all),
        nvptx.shflSync(.bfly, f32, x[1], mask, clamp, .all),
    };
}

inline fn exp2(x: anytype) @TypeOf(x) {
    var result: @TypeOf(x) = undefined;
    inline for (0..@typeInfo(@TypeOf(x)).vector.len) |i| {
        result[i] = nvptx.math.ex2ApproxF32(true, x[i]);
    }
    return result;
}
