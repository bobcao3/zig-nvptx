//! Packs of E4M3 values with one E8M0 scale `2^(e - 127)` per 128 values of an activation row or
//! per 128 x 128 weight tile, and their product, which rounds the FP32 sum to BF16 once.

const std = @import("std");
const builtin = @import("builtin");
const nvptx = std.nvptx;
const device = @import("device.zig");
const fp8_gemm = @import("fp8_gemm.zig");
const Chunk = device.Chunk;

pub const group = 128;
const e4m3_requirement: nvptx.Requirement = .{ .ptx = 81, .sm = .{ .base = 89 }, .threads = .thread };

/// The E8M0 byte of the least power of two at or above `amax / 448`; 127 for a block of zeros.
pub inline fn scaleByte(amax: f32) u32 {
    const bits: u32 = @bitCast(amax * @as(f32, @bitCast(@as(u32, 0x3B124925))));
    const byte = ((bits >> 23) & 0xFF) + @intFromBool(bits & 0x7FFFFF != 0);
    return if (amax == 0) 127 else @min(byte, 254);
}

pub inline fn reciprocal(byte: u32) f32 {
    return @bitCast((254 - byte) << 23);
}

pub inline fn scaleOf(byte: u32) f32 {
    return @bitCast(byte << 23);
}

/// Rounds `x0` and `x1` to E4M3 with ties to even and saturates them to 448 in magnitude, with
/// `x0` in the low byte (`cvt` in PTX ISA section 9.7.10.24 puts operand `a` in the high byte).
inline fn e4m3x2(x0: f32, x1: f32) u32 {
    nvptx.require("cvt.rn.satfinite.e4m3x2.f32", e4m3_requirement);
    const pair: u16 = asm ("cvt.rn.satfinite.e4m3x2.f32 %[d], %[a], %[b];"
        : [d] "=h" (-> u16),
        : [a] "f" (x1),
          [b] "f" (x0),
    );
    return pair;
}

inline fn quantizeChunk(x: [8]f32, factor: f32) @Vector(2, u32) {
    var words: @Vector(2, u32) = undefined;
    inline for (0..2) |w| {
        const low = e4m3x2(x[w * 4] * factor, x[w * 4 + 1] * factor);
        const high = e4m3x2(x[w * 4 + 2] * factor, x[w * 4 + 3] * factor);
        words[w] = low | (high << 16);
    }
    return words;
}

inline fn unpackChunk(chunk: Chunk) [8]f32 {
    var x: [8]f32 = undefined;
    inline for (0..4) |i| {
        const pair = device.unpack(chunk[i]);
        x[i * 2] = pair[0];
        x[i * 2 + 1] = pair[1];
    }
    return x;
}

inline fn chunkAmax(x: [8]f32) f32 {
    var amax: f32 = 0;
    inline for (x) |v| amax = @max(amax, @abs(v));
    return amax;
}

pub const Source = enum { bf16, swiglu };

pub fn QuantizeRows(comptime source: Source) type {
    return struct {
        pub const name = "fp8_quantize_rows_" ++ @tagName(source) ++ "_sm89";
        pub const work_items = 256;
        const group_items = group / 8;

        /// One work item quantizes 8 values, and 16 work items one group. A `swiglu` row holds
        /// `width` up and `width` gate values and yields `bf16(bf16(silu(gate)) * up)`. The scale
        /// of group `g` of row `r` goes to `scales[g * scale_stride + r]`.
        fn kernel(
            input: [*]addrspace(.global) const Chunk,
            input_stride: u32,
            values: [*]addrspace(.global) @Vector(2, u32),
            scales: [*]addrspace(.global) u8,
            scale_stride: u32,
            rows: u32,
            width: u32,
        ) callconv(.{ .nvptx_kernel = .{
            .required_work_group_size = .{ .x = work_items },
        } }) void {
            const chunks = width / 8;
            const i = @workGroupId(0) * work_items + @workItemId(0);
            const valid = i < rows * chunks;
            const row = @min(i / chunks, rows - 1);
            const chunk = i % chunks;
            const line = input + @as(u64, row) * input_stride;
            const x = switch (source) {
                .bf16 => unpackChunk(line[chunk]),
                .swiglu => swiglu(line[chunk], line[chunks + chunk]),
            };
            var amax = chunkAmax(x);
            inline for (.{ 8, 4, 2, 1 }) |mask| amax = @max(amax, device.shuffleXor(amax, mask));
            const byte = scaleByte(amax);
            if (!valid) return;
            values[@as(u64, row) * chunks + chunk] = quantizeChunk(x, reciprocal(byte));
            if (chunk % group_items == 0) {
                scales[@as(u64, chunk / group_items) * scale_stride + row] = @intCast(byte);
            }
        }

        inline fn swiglu(up: Chunk, gate: Chunk) [8]f32 {
            const u = unpackChunk(up);
            const g = unpackChunk(gate);
            var h: [8]f32 = undefined;
            inline for (&h, u, g) |*out, up_value, gate_value| {
                out.* = device.roundBf16(device.roundBf16(device.silu(gate_value)) * up_value);
            }
            return h;
        }
    };
}

pub const QuantizeTiles = struct {
    pub const name = "fp8_quantize_tiles_sm89";
    pub const work_items = 256;
    const warps = work_items / 32;
    const tile_chunks = group * group / 8;
    const passes = tile_chunks / work_items;

    /// One work group quantizes one 128 x 128 tile of the `[n, k]` BF16 matrix `weight`.
    fn kernel(
        weight: [*]addrspace(.global) const Chunk,
        values: [*]addrspace(.global) @Vector(2, u32),
        scales: [*]addrspace(.global) u8,
        k: u32,
    ) callconv(.{ .nvptx_kernel = .{
        .required_work_group_size = .{ .x = work_items },
    } }) void {
        const tid = @workItemId(0);
        const tile = @workGroupId(0);
        const tiles_k = k / group;
        const first_row = tile / tiles_k * group;
        const first_chunk = (tile % tiles_k) * (group / 8);
        var x: [passes][8]f32 = undefined;
        var amax: f32 = 0;
        inline for (&x, 0..) |*chunk, pass| {
            const index = tid + pass * work_items;
            const element = @as(u64, first_row + index / 16) * (k / 8) + first_chunk + index % 16;
            chunk.* = unpackChunk(weight[element]);
            amax = @max(amax, chunkAmax(chunk.*));
        }
        inline for (.{ 16, 8, 4, 2, 1 }) |mask| amax = @max(amax, device.shuffleXor(amax, mask));
        const partials: *addrspace(.shared) [warps]f32 = @ptrCast(&shared_storage);
        if (tid % 32 == 0) partials[tid / 32] = amax;
        nvptx.barrierSyncAligned(.work_group, null);
        inline for (0..warps) |w| amax = @max(amax, partials[w]);
        const byte = scaleByte(amax);
        const factor = reciprocal(byte);
        inline for (x, 0..) |chunk, pass| {
            const index = tid + pass * work_items;
            const element = @as(u64, first_row + index / 16) * (k / 8) + first_chunk + index % 16;
            values[element] = quantizeChunk(chunk, factor);
        }
        if (tid == 0) scales[tile] = @intCast(byte);
    }

    extern var shared_storage: [0]u8 align(16) addrspace(.shared);
    pub const dynamic_shared_bytes = warps * 4;
};

const block_scale_requirement: nvptx.Requirement = .{ .ptx = 88, .sm = .{ .family = &.{ 120, 121 } }, .threads = .warp };

pub const has_block_scale = builtin.cpu.arch.isNvptx() and block_scale_requirement.isMet(nvptx.gpu);

pub inline fn mmaE4m3(a: @Vector(4, u32), b: @Vector(2, u32), c: @Vector(4, f32)) @Vector(4, f32) {
    nvptx.require("mma.sync.aligned.m16n8k32.e4m3", .{ .ptx = 84, .sm = .{ .base = 89 }, .threads = .warp });
    return asm volatile ("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 %[d], %[a], %[b], %[c];"
        : [d] "=f" (-> @Vector(4, f32)),
        : [a] "r" (a),
          [b] "r" (b),
          [c] "f" (c),
    );
}

/// Byte `byte_a` of `scale_a` holds the scale of the row of A that the calling work item names,
/// and byte `byte_b` of `scale_b` the scale of the columns of B.
pub inline fn mmaE4m3BlockScale(
    comptime byte_a: u32,
    comptime byte_b: u32,
    a: @Vector(4, u32),
    b: @Vector(2, u32),
    c: @Vector(4, f32),
    scale_a: u32,
    scale_b: u32,
) @Vector(4, f32) {
    nvptx.require("mma.sync.aligned.m16n8k32.kind::mxf8f6f4.block_scale", block_scale_requirement);
    const text = std.fmt.comptimePrint("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X" ++
        ".f32.e4m3.e4m3.f32.ue8m0 %[d], %[a], %[b], %[c], %[scale_a], {{{d}, 0}}, %[scale_b], {{{d}, 0}};", .{ byte_a, byte_b });
    return asm volatile (text
        : [d] "=f" (-> @Vector(4, f32)),
        : [a] "r" (a),
          [b] "r" (b),
          [c] "f" (c),
          [scale_a] "r" (scale_a),
          [scale_b] "r" (scale_b),
    );
}

pub const kernels = .{
    QuantizeRows(.bf16),
    QuantizeRows(.swiglu),
    QuantizeTiles,
    fp8_gemm.Gemm(.cp_async, .promote, .bf16, fp8_gemm.linear_tiling),
};

pub const block_scale_kernels = .{
    fp8_gemm.Gemm(.cp_async, .block_scale, .bf16, fp8_gemm.linear_tiling),
    fp8_gemm.Gemm(.tma, .block_scale, .bf16, fp8_gemm.linear_tiling),
    fp8_gemm.Gemm(.tma, .block_scale, .swiglu, fp8_gemm.linear_tiling),
    fp8_gemm.Gemm(.tma, .block_scale, .qk_norm_rope, fp8_gemm.head_tiling),
};

comptime {
    if (builtin.cpu.arch.isNvptx() and @import("root") == @This()) {
        for (kernels) |Kernel| @export(&Kernel.kernel, .{ .name = Kernel.name });
        if (has_block_scale) {
            for (block_scale_kernels) |Kernel| @export(&Kernel.kernel, .{ .name = Kernel.name });
        }
    }
}
