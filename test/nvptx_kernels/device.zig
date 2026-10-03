//! The float functions round exactly as the CUDA math library that PyTorch uses.

const std = @import("std");
const nvptx = std.nvptx;
const math = nvptx.math;

/// Eight BF16 values fill the 16 bytes of one vector load, `cp.async` copy or `ldmatrix` row.
pub const Chunk = @Vector(4, u32);

pub inline fn bf16ToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// Rounds to the nearest BF16 value, ties to even (`cvt.rn.bf16.f32`).
pub inline fn f32ToBf16(x: f32) u16 {
    return nvptx.bf16FromF32(x);
}

pub inline fn roundBf16(x: f32) f32 {
    return bf16ToF32(f32ToBf16(x));
}

pub inline fn unpack(pair: u32) @Vector(2, f32) {
    return nvptx.Bf16x2.toF32(@fromBackingInt(pair));
}

pub inline fn pack(x0: f32, x1: f32) u32 {
    return @backingInt(nvptx.Bf16x2.fromF32(x0, x1));
}

pub inline fn expf(x: f32) f32 {
    const t = math.fmaF32(.nearest_even, .{}, x, @bitCast(@as(u32, 0x3BBB989D)), 0.5);
    const clamped = math.cvt(f32, f32, t, .{ .saturate = true });
    const j = math.fmaF32(
        .toward_negative_infinity,
        .{},
        clamped,
        @bitCast(@as(u32, 0x437C0000)),
        @bitCast(@as(u32, 0x4B400001)),
    );
    const integer = j + @as(f32, @bitCast(@as(u32, 0xCB40007F)));
    const high = math.fmaF32(.nearest_even, .{}, x, @bitCast(@as(u32, 0x3FB8AA3B)), -integer);
    const reduced = math.fmaF32(.nearest_even, .{}, x, @bitCast(@as(u32, 0x32A57060)), high);
    const power: f32 = @bitCast(@as(u32, @bitCast(j)) << 23);
    return math.ex2ApproxF32(true, reduced) * power;
}

/// The PyTorch SiLU kernel computes the last multiply of `expf` and the `+ 1` as one `fma`.
pub inline fn silu(x: f32) f32 {
    const minus = -x;
    const t = math.fmaF32(.nearest_even, .{}, minus, @bitCast(@as(u32, 0x3BBB989D)), 0.5);
    const clamped = math.cvt(f32, f32, t, .{ .saturate = true });
    const j = math.fmaF32(
        .toward_negative_infinity,
        .{},
        clamped,
        @bitCast(@as(u32, 0x437C0000)),
        @bitCast(@as(u32, 0x4B400001)),
    );
    const integer = j + @as(f32, @bitCast(@as(u32, 0xCB40007F)));
    const high = math.fmaF32(.nearest_even, .{}, minus, @bitCast(@as(u32, 0x3FB8AA3B)), -integer);
    const reduced = math.fmaF32(.nearest_even, .{}, minus, @bitCast(@as(u32, 0x32A57060)), high);
    const power: f32 = @bitCast(@as(u32, @bitCast(j)) << 23);
    const denominator = math.fmaF32(.nearest_even, .{}, math.ex2ApproxF32(true, reduced), power, 1.0);
    return math.divF32(.nearest_even, false, x, denominator);
}

pub inline fn shuffleXor(x: f32, comptime mask: u32) f32 {
    const clamp: nvptx.ShuffleClamp = .{ .clamp = 31, .segment_mask = 0 };
    return nvptx.shflSync(.bfly, f32, x, mask, clamp, .all);
}

pub inline fn warpSum(x: f32) f32 {
    var sum = x;
    inline for (.{ 16, 8, 4, 2, 1 }) |mask| sum += shuffleXor(sum, mask);
    return sum;
}

pub inline fn tanhf(x: f32) f32 {
    const magnitude = @abs(x);
    const e = math.ex2ApproxF32(true, magnitude * @as(f32, @bitCast(@as(u32, 0x4038AA3B))));
    const r = math.rcpApproxF32(true, e + 1.0);
    const far = if (magnitude >= @as(f32, @bitCast(@as(u32, 0x41102CB4))))
        1.0
    else
        math.fmaF32(.nearest_even, .{}, r, -2.0, 1.0);
    const large = math.copysign(f32, x, far);
    const square = x * x;
    var p = math.fmaF32(.nearest_even, .{}, @bitCast(@as(u32, 0x3C80F082)), square, @bitCast(@as(u32, 0xBD563CAE)));
    p = math.fmaF32(.nearest_even, .{}, p, square, @bitCast(@as(u32, 0x3E085941)));
    p = math.fmaF32(.nearest_even, .{}, p, square, @bitCast(@as(u32, 0xBEAAA9ED)));
    p = math.fmaF32(.nearest_even, .{}, p, square, 0.0);
    const small = math.fmaF32(.nearest_even, .{}, p, x, x);
    return if (magnitude >= @as(f32, @bitCast(@as(u32, 0x3F19999A)))) large else small;
}

/// The function computes `0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))`.
pub inline fn geluTanh(x: f32) f32 {
    const cube = x * (x * x);
    const inner = math.fmaF32(.nearest_even, .{}, cube, @bitCast(@as(u32, 0x3D372713)), x) *
        @as(f32, @bitCast(@as(u32, 0x3F4C422A)));
    return (x * 0.5) * (tanhf(inner) + 1.0);
}

pub inline fn erff(x: f32) f32 {
    const magnitude = @abs(x);
    const far = magnitude >= @as(f32, @bitCast(@as(u32, 0x3F8060FE)));
    const s = if (far) magnitude else x * x;
    const coefficients = if (far)
        [_]u32{ 0x38EB4C3A, 0xBAAE005B, 0x3C09919F, 0xBD24D99A, 0x3E235519, 0x3F69B4F9, 0x3F210A14 }
    else
        [_]u32{ 0x38B1E96A, 0xBA574D20, 0x3BAAD5EA, 0xBCDC1BE7, 0x3DE718AF, 0xBEC093AC, 0x3E0375D3 };
    var p: f32 = @bitCast(coefficients[0]);
    inline for (coefficients[1..]) |c| p = math.fmaF32(.nearest_even, .{}, p, s, @bitCast(c));
    const m = if (far) -magnitude else x;
    const r = math.fmaF32(.nearest_even, .{}, p, m, m);
    return if (far) math.copysign(f32, x, 1.0 - math.ex2ApproxF32(true, r)) else r;
}

/// The function computes `0.5 * x * (1 + erf(x / sqrt(2)))`.
pub inline fn gelu(x: f32) f32 {
    return (x * 0.5) * (erff(x * @as(f32, @bitCast(@as(u32, 0x3F3504F3)))) + 1.0);
}
