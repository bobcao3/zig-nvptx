//! Number formats that Zig has no type for.

const std = @import("../std.zig");
const nvptx = std.nvptx;

/// A pair of `bf16` values in one 32-bit register. Element 0 is the low half.
pub const Bf16x2 = enum(u32) {
    _,

    const requirement: nvptx.Requirement = .{
        .ptx = 70,
        .sm = .{ .base = 80 },
        .threads = .thread,
    };

    /// Packs two `bf16` bit patterns. Element 0 occupies bits 0 to 15, as in the `a[0:15]`
    /// semantics of `add.bf16x2` (section 9.7.4.1). `cvt.rn.bf16x2.f32 d, a, b` (section
    /// 9.7.10.24) stores the conversion of `a` in the upper half and of `b` in the lower half.
    pub inline fn init(element0: u16, element1: u16) Bf16x2 {
        return @fromBackingInt((@as(u32, element1) << 16) | element0);
    }

    /// Returns the `bf16` bit pattern of element 0.
    pub inline fn lo(x: Bf16x2) u16 {
        return @truncate(@backingInt(x));
    }

    /// Returns the `bf16` bit pattern of element 1.
    pub inline fn hi(x: Bf16x2) u16 {
        return @truncate(@backingInt(x) >> 16);
    }

    /// Emits `cvt.rn.bf16x2.f32` (section 9.7.10.24).
    pub inline fn fromF32(x0: f32, x1: f32) Bf16x2 {
        nvptx.require("cvt.rn.bf16x2.f32", requirement);
        return @fromBackingInt(asm ("cvt.rn.bf16x2.f32 %[d], %[hi], %[lo];"
            : [d] "=r" (-> u32),
            : [lo] "f" (x0),
              [hi] "f" (x1),
        ));
    }

    /// Emits `cvt.rn.relu.bf16x2.f32` (section 9.7.10.24).
    pub inline fn fromF32Relu(x0: f32, x1: f32) Bf16x2 {
        nvptx.require("cvt.rn.relu.bf16x2.f32", requirement);
        return @fromBackingInt(asm ("cvt.rn.relu.bf16x2.f32 %[d], %[hi], %[lo];"
            : [d] "=r" (-> u32),
            : [lo] "f" (x0),
              [hi] "f" (x1),
        ));
    }

    /// A `bf16` has the upper 16 bits of the `f32` with the same value.
    pub inline fn toF32(x: Bf16x2) @Vector(2, f32) {
        const bits = @backingInt(x);
        return .{
            @bitCast(bits << 16),
            @bitCast(bits & 0xffff0000),
        };
    }

    inline fn ternary(comptime text: []const u8, a: Bf16x2, b: Bf16x2, c: Bf16x2) Bf16x2 {
        return @fromBackingInt(asm (text ++ " %[d], %[a], %[b], %[c];"
            : [d] "=r" (-> u32),
            : [a] "r" (@backingInt(a)),
              [b] "r" (@backingInt(b)),
              [c] "r" (@backingInt(c)),
        ));
    }

    inline fn binary(comptime text: []const u8, a: Bf16x2, b: Bf16x2) Bf16x2 {
        return @fromBackingInt(asm (text ++ " %[d], %[a], %[b];"
            : [d] "=r" (-> u32),
            : [a] "r" (@backingInt(a)),
              [b] "r" (@backingInt(b)),
        ));
    }

    inline fn unary(comptime text: []const u8, a: Bf16x2) Bf16x2 {
        return @fromBackingInt(asm (text ++ " %[d], %[a];"
            : [d] "=r" (-> u32),
            : [a] "r" (@backingInt(a)),
        ));
    }

    const one: Bf16x2 = @fromBackingInt(0x3f803f80);
    const negative_zero: Bf16x2 = @fromBackingInt(0x80008000);

    /// Emits `fma.rn.bf16x2` (section 9.7.4.4), which returns `x * y + z`.
    pub inline fn fma(x: Bf16x2, y: Bf16x2, z: Bf16x2) Bf16x2 {
        nvptx.require("fma.rn.bf16x2", requirement);
        return ternary("fma.rn.bf16x2", x, y, z);
    }

    /// Emits `fma.rn.relu.bf16x2` (section 9.7.4.4).
    pub inline fn fmaRelu(x: Bf16x2, y: Bf16x2, z: Bf16x2) Bf16x2 {
        nvptx.require("fma.rn.relu.bf16x2", requirement);
        return ternary("fma.rn.relu.bf16x2", x, y, z);
    }

    /// Computes `x * y` with `fma.rn.bf16x2` and a negative zero addend, which leaves every
    /// product unchanged. The dedicated `mul.rn.bf16x2` needs sm_90.
    pub inline fn mul(x: Bf16x2, y: Bf16x2) Bf16x2 {
        return x.fma(y, negative_zero);
    }

    /// Computes `x + y` with `fma.rn.bf16x2` and a factor of one. The dedicated `add.rn.bf16x2`
    /// needs sm_90.
    pub inline fn add(x: Bf16x2, y: Bf16x2) Bf16x2 {
        return x.fma(one, y);
    }

    /// Emits `min.bf16x2` (section 9.7.4.7), which returns the other operand when one operand is NaN.
    pub inline fn min(x: Bf16x2, y: Bf16x2) Bf16x2 {
        nvptx.require("min.bf16x2", requirement);
        return binary("min.bf16x2", x, y);
    }

    /// Emits `max.bf16x2` (section 9.7.4.8).
    pub inline fn max(x: Bf16x2, y: Bf16x2) Bf16x2 {
        nvptx.require("max.bf16x2", requirement);
        return binary("max.bf16x2", x, y);
    }

    /// Emits `min.NaN.bf16x2` (section 9.7.4.7), which returns NaN when one operand is NaN.
    pub inline fn minNan(x: Bf16x2, y: Bf16x2) Bf16x2 {
        nvptx.require("min.NaN.bf16x2", requirement);
        return binary("min.NaN.bf16x2", x, y);
    }

    /// Emits `max.NaN.bf16x2` (section 9.7.4.8).
    pub inline fn maxNan(x: Bf16x2, y: Bf16x2) Bf16x2 {
        nvptx.require("max.NaN.bf16x2", requirement);
        return binary("max.NaN.bf16x2", x, y);
    }

    /// Emits `neg.bf16x2` (section 9.7.4.5).
    pub inline fn neg(x: Bf16x2) Bf16x2 {
        nvptx.require("neg.bf16x2", requirement);
        return unary("neg.bf16x2", x);
    }

    /// Emits `abs.bf16x2` (section 9.7.4.6).
    pub inline fn abs(x: Bf16x2) Bf16x2 {
        nvptx.require("abs.bf16x2", requirement);
        return unary("abs.bf16x2", x);
    }
};

/// PTX stores two E4M3 values in a `.b16` operand (section 5.2.5.1).
pub const E4m3x2 = Fp8x2(.e4m3);
/// PTX stores two E5M2 values in a `.b16` operand (section 5.2.5.1).
pub const E5m2x2 = Fp8x2(.e5m2);
/// PTX stores four E4M3 values in a `.b32` operand (section 5.2.5.1).
pub const E4m3x4 = Fp8x4(.e4m3);
/// PTX stores four E5M2 values in a `.b32` operand (section 5.2.5.1).
pub const E5m2x4 = Fp8x4(.e5m2);

const Fp8Format = enum { e4m3, e5m2 };

fn Fp8x2(comptime format: Fp8Format) type {
    return enum(u16) {
        _,

        const Pair = @This();

        pub inline fn init(element0: u8, element1: u8) Pair {
            return @fromBackingInt((@as(u16, element1) << 8) | element0);
        }

        pub inline fn lo(x: Pair) u8 {
            return @truncate(@backingInt(x));
        }

        pub inline fn hi(x: Pair) u8 {
            return @truncate(@backingInt(x) >> 8);
        }

        /// Rounds to nearest with ties to even and clamps overflow and infinity to the largest
        /// finite magnitude. The result stores `x0` in the low byte and `x1` in the high byte.
        /// PTX places the first source in the high byte (section 9.7.10.24).
        pub inline fn fromF32(x0: f32, x1: f32) Pair {
            const instruction = "cvt.rn.satfinite." ++ @tagName(format) ++ "x2.f32";
            nvptx.require(instruction, .{
                .ptx = if (nvptx.gpu.sm >= 90) 78 else 81,
                .sm = .{ .base = 89 },
                .threads = .thread,
            });
            return @fromBackingInt(asm (instruction ++ " %[d], %[hi], %[lo];"
                : [d] "=h" (-> u16),
                : [lo] "f" (x0),
                  [hi] "f" (x1),
            ));
        }
    };
}

fn Fp8x4(comptime format: Fp8Format) type {
    return enum(u32) {
        _,

        const Quad = @This();
        const Pair = Fp8x2(format);

        pub inline fn init(element0: u8, element1: u8, element2: u8, element3: u8) Quad {
            return @fromBackingInt((@as(u32, element3) << 24) | (@as(u32, element2) << 16) |
                (@as(u32, element1) << 8) | element0);
        }

        pub inline fn element(x: Quad, index: u2) u8 {
            return @truncate(@backingInt(x) >> (@as(u5, index) * 8));
        }

        /// Rounds to nearest with ties to even and clamps overflow and infinity to the largest
        /// finite magnitude. The result stores `x0` in the lowest byte through `x3` in the highest.
        /// PTX `cvt` (section 9.7.10.24) has no `.rn` FP8x4 form.
        pub inline fn fromF32(x0: f32, x1: f32, x2: f32, x3: f32) Quad {
            const low = @backingInt(Pair.fromF32(x0, x1));
            const high = @backingInt(Pair.fromF32(x2, x3));
            return @fromBackingInt((@as(u32, high) << 16) | low);
        }

        /// Rounds stochastically and clamps overflow and infinity to the largest finite magnitude.
        /// The result stores `x0` in the lowest byte through `x3` in the highest. The caller supplies
        /// `random_bits`; the low 16 bits serve `x0` and `x1`, and the high 16 bits serve `x2` and `x3`.
        /// PTX `cvt` (section 9.7.10.24, Semantics and Target ISA Notes) defines the random-bit
        /// layout and restricts `.rs` to architecture-specific targets.
        /// https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt
        pub inline fn fromF32Stochastic(x0: f32, x1: f32, x2: f32, x3: f32, random_bits: u32) Quad {
            const instruction = "cvt.rs.satfinite." ++ @tagName(format) ++ "x4.f32";
            nvptx.require(instruction, .{
                .ptx = 87,
                .sm = .{ .arch_specific = if (nvptx.gpu.sm == 103) 103 else 100 },
                .threads = .thread,
            });
            return @fromBackingInt(asm (instruction ++ " %[d], {%[x3], %[x2], %[x1], %[x0]}, %[random_bits];"
                : [d] "=r" (-> u32),
                : [x0] "f" (x0),
                  [x1] "f" (x1),
                  [x2] "f" (x2),
                  [x3] "f" (x3),
                  [random_bits] "r" (random_bits),
            ));
        }
    };
}

/// Emits `cvt.rn.bf16.f32` (section 9.7.10.24) and returns the `bf16` bit pattern. The
/// destination format `.bf16` needs PTX ISA 7.0 and sm_80.
pub inline fn bf16FromF32(x: f32) u16 {
    nvptx.require("cvt.rn.bf16.f32", .{ .ptx = 70, .sm = .{ .base = 80 }, .threads = .thread });
    return asm ("cvt.rn.bf16.f32 %[d], %[a];"
        : [d] "=h" (-> u16),
        : [a] "f" (x),
    );
}

/// A `tf32` value in a 32-bit register. The low 13 bits of the `f32` encoding are zero.
pub const Tf32 = enum(u32) {
    _,

    /// Emits `cvt.rna.tf32.f32` (section 9.7.10.24), which rounds to nearest and breaks ties away from zero.
    pub inline fn fromF32(x: f32) Tf32 {
        nvptx.require("cvt.rna.tf32.f32", .{ .ptx = 70, .sm = .{ .base = 80 }, .threads = .thread });
        return @fromBackingInt(asm ("cvt.rna.tf32.f32 %[d], %[a];"
            : [d] "=r" (-> u32),
            : [a] "f" (x),
        ));
    }

    pub inline fn toF32(x: Tf32) f32 {
        return @bitCast(@backingInt(x));
    }
};

test Bf16x2 {
    const x: Bf16x2 = @fromBackingInt(0x4000_3fc0);
    try std.testing.expectEqual(@Vector(2, f32){ 1.5, 2.0 }, x.toF32());
}

test "Bf16x2 elements" {
    const x: Bf16x2 = .init(0x3fc0, 0x4000);
    try std.testing.expectEqual(0x4000_3fc0, @backingInt(x));
    try std.testing.expectEqual(0x3fc0, x.lo());
    try std.testing.expectEqual(0x4000, x.hi());
}

test "FP8 packed elements" {
    inline for (.{ E4m3x2, E5m2x2 }) |Pair| {
        const x: Pair = .init(0x37, 0x82);
        try std.testing.expectEqual(0x8237, @backingInt(x));
        try std.testing.expectEqual(0x37, x.lo());
        try std.testing.expectEqual(0x82, x.hi());
    }
    inline for (.{ E4m3x4, E5m2x4 }) |Quad| {
        const x: Quad = .init(0x37, 0x82, 0x19, 0xfe);
        try std.testing.expectEqual(0xfe198237, @backingInt(x));
        for ([_]u8{ 0x37, 0x82, 0x19, 0xfe }, 0..) |expected, i| {
            try std.testing.expectEqual(expected, x.element(@intCast(i)));
        }
    }
}
