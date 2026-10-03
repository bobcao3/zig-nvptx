//! Replacements for the compiler_rt routines of the same names. A GPU runs the replacements many
//! times faster. Each result is within one unit in the last place of the correctly rounded value.

const std = @import("../std.zig");

pub fn expf(x: f32) f32 {
    const clamped = clamp(x, -104, 89);
    const biased = @mulAdd(f32, clamped, std.math.log2e, round_bias);
    const n = biased - round_bias;
    const r = @mulAdd(f32, n, -ln2_low, @mulAdd(f32, n, -ln2_high, clamped));
    return scale(polynomial(&exp_coefficients, r), biased);
}

pub fn exp2f(x: f32) f32 {
    const clamped = clamp(x, -151, 129);
    const biased = clamped + round_bias;
    const r = clamped - (biased - round_bias);
    const head = @mulAdd(f32, ln2_high, r, 1);
    const head_error = @mulAdd(f32, ln2_high, r, 1 - head);
    const tail = @mulAdd(f32, polynomial(&exp2_tail_coefficients, r), r, head_error);
    return scale(head + tail, biased);
}

const round_bias: f32 = 0x1.8p23;
const ln2_high: f32 = std.math.ln2;
const ln2_low: f32 = std.math.ln2 - @as(comptime_float, ln2_high);
const exp_coefficients = [_]f32{
    1,
    1,
    0x1.fffffcp-2,
    0x1.55548cp-3,
    0x1.55585ap-5,
    0x1.123dbcp-7,
    0x1.6ac666p-10,
};
const exp2_tail_coefficients = [_]f32{
    0x1.70c4b4p-25,
    0x1.ebfbe4p-3,
    0x1.c6ae24p-5,
    0x1.3b24e0p-7,
    0x1.5f9006p-10,
    0x1.49094cp-13,
};

/// Limits `x` to [low, high] and returns NaN for a NaN. For the two `@select` calls, ptxas emits
/// fewer instructions than for the equivalent `if` expressions.
inline fn clamp(x: f32, comptime low: f32, comptime high: f32) f32 {
    const Single = @Vector(1, f32);
    const value: Single = @splat(x);
    const lower: Single = @splat(low);
    const upper: Single = @splat(high);
    const at_least = @select(f32, value < lower, lower, value);
    return @select(f32, at_least > upper, upper, at_least)[0];
}

inline fn polynomial(comptime coefficients: []const f32, r: f32) f32 {
    var p = coefficients[coefficients.len - 1];
    inline for (1..coefficients.len) |i| {
        p = @mulAdd(f32, p, r, coefficients[coefficients.len - 1 - i]);
    }
    return p;
}

/// Returns `p * 2^n` rounded to the nearest `f32`, where `p` is in [0.5, 2), `n` is an integer in
/// [-151, 129] and `biased` is `round_bias + n`.
inline fn scale(p: f32, biased: f32) f32 {
    const shift: u32 = if (biased > round_bias) 0 else 250 << 23;
    const first: f32 = @bitCast(@as(u32, @bitCast(@as(f32, 0x1p127))) -% shift);
    const second: f32 = @bitCast((@as(u32, @bitCast(biased)) << 23) +% shift);
    return p * first * second;
}

fn expectFaithful(comptime function: fn (f32) f32, comptime reference: fn (f64) f64) !void {
    var bits: u64 = 0;
    while (bits < 1 << 32) : (bits += 997) {
        try expectFaithfulAt(function, reference, @bitCast(@as(u32, @intCast(bits))));
    }
    const inf = std.math.inf(f32);
    const specials = [_]f32{
        inf,
        -inf,
        0,
        -0.0,
        88.72283,
        88.72284,
        -103.97,
        -149.5,
        127.99999,
        128,
    };
    for (specials) |x| {
        try expectFaithfulAt(function, reference, x);
    }
}

fn expectFaithfulAt(
    comptime function: fn (f32) f32,
    comptime reference: fn (f64) f64,
    x: f32,
) !void {
    const result = function(x);
    const expected: f32 = @floatCast(reference(x));
    if (std.math.isNan(x)) {
        try std.testing.expect(std.math.isNan(result));
    } else if (result != expected) {
        try std.testing.expectEqual(std.math.nextAfter(f32, expected, result), result);
    }
}

fn exp64(x: f64) f64 {
    return @exp(x);
}

fn exp2_64(x: f64) f64 {
    return @exp2(x);
}

test expf {
    try expectFaithful(expf, exp64);
}

test exp2f {
    try expectFaithful(exp2f, exp2_64);
}
