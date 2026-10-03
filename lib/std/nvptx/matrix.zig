//! This file wraps the warp-level matrix instructions. Section numbers refer to PTX ISA 9.4.

const std = @import("../std.zig");
const nvptx = @import("../nvptx.zig");

pub const Shape = struct {
    m: u32,
    n: u32,
    k: u32,
};

/// The shape `mXnYkZ` multiplies an X by Z matrix A by a Z by Y matrix B (section 9.7.16.5.14).
pub const MmaSyncShape = enum {
    m8n8k4,
    m8n8k16,
    m8n8k32,
    m8n8k128,
    m16n8k4,
    m16n8k8,
    m16n8k16,
    m16n8k32,
    m16n8k64,
    m16n8k128,
    m16n8k256,

    /// Returns the shape `m` by `n` by `k`, or gives a compile error when `mma.sync` has none.
    pub inline fn fromMnk(m: comptime_int, n: comptime_int, k: comptime_int) MmaSyncShape {
        const name = std.fmt.comptimePrint("m{d}n{d}k{d}", .{ m, n, k });
        if (!@hasField(MmaSyncShape, name)) @compileError("mma.sync has no shape " ++ name);
        return @field(MmaSyncShape, name);
    }

    pub fn dimensions(shape: MmaSyncShape) Shape {
        return switch (shape) {
            .m8n8k4 => .{ .m = 8, .n = 8, .k = 4 },
            .m8n8k16 => .{ .m = 8, .n = 8, .k = 16 },
            .m8n8k32 => .{ .m = 8, .n = 8, .k = 32 },
            .m8n8k128 => .{ .m = 8, .n = 8, .k = 128 },
            .m16n8k4 => .{ .m = 16, .n = 8, .k = 4 },
            .m16n8k8 => .{ .m = 16, .n = 8, .k = 8 },
            .m16n8k16 => .{ .m = 16, .n = 8, .k = 16 },
            .m16n8k32 => .{ .m = 16, .n = 8, .k = 32 },
            .m16n8k64 => .{ .m = 16, .n = 8, .k = 64 },
            .m16n8k128 => .{ .m = 16, .n = 8, .k = 128 },
            .m16n8k256 => .{ .m = 16, .n = 8, .k = 256 },
        };
    }
};

pub const Layout = enum { row, col };

pub const Element = enum { f16, bf16, tf32, e4m3, e5m2, s8, u8, s4, u4, b1, f64 };

pub const Accumulator = enum { f16, f32, s32, f64 };

pub const BitOperation = enum { xor, @"and" };

pub const Rounding = enum { rn, rz, rm, rp };

const ptx64_sm70: nvptx.Requirement = .{ .ptx = 64, .sm = .{ .base = 70 }, .threads = .warp };
const ptx65_sm75: nvptx.Requirement = .{ .ptx = 65, .sm = .{ .base = 75 }, .threads = .warp };
const ptx70_sm75: nvptx.Requirement = .{ .ptx = 70, .sm = .{ .base = 75 }, .threads = .warp };
const ptx70_sm80: nvptx.Requirement = .{ .ptx = 70, .sm = .{ .base = 80 }, .threads = .warp };
const ptx71_sm80: nvptx.Requirement = .{ .ptx = 71, .sm = .{ .base = 80 }, .threads = .warp };

pub const MmaOptions = struct {
    shape: MmaSyncShape,
    accumulator: Accumulator,
    /// The element type of A, and of B when `b_element` is null.
    element: Element,
    b_element: ?Element = null,
    a_layout: Layout = .row,
    b_layout: Layout = .col,
    satfinite: bool = false,
    bit_operation: ?BitOperation = null,
    rounding: ?Rounding = null,
};

fn mmaError(comptime options: MmaOptions, comptime problem: []const u8) noreturn {
    @compileError(std.fmt.comptimePrint(
        "mma.sync with shape {t}, a {t}, b {t} and accumulator {t} {s}",
        .{
            options.shape,
            options.element,
            options.b_element orelse options.element,
            options.accumulator,
            problem,
        },
    ));
}

fn ElementRegisters(comptime element: Element, comptime count: u32) type {
    const bits: u32 = switch (element) {
        .f64 => return @Vector(count, f64),
        .tf32 => 32,
        .f16, .bf16 => 16,
        .e4m3, .e5m2, .s8, .u8 => 8,
        .s4, .u4 => 4,
        .b1 => 1,
    };
    return @Vector(count * bits / 32, u32);
}

fn validateMma(comptime options: MmaOptions, comptime A: type, comptime B: type, comptime C: type) void {
    const b_element = options.b_element orelse options.element;
    comptime {
        const shapes: []const MmaSyncShape = switch (options.element) {
            .f16 => &.{ .m8n8k4, .m16n8k8, .m16n8k16 },
            .bf16 => &.{ .m16n8k8, .m16n8k16 },
            .tf32 => &.{ .m16n8k4, .m16n8k8 },
            .e4m3, .e5m2 => &.{ .m16n8k16, .m16n8k32 },
            .f64 => &.{.m8n8k4},
            .s8, .u8 => &.{ .m8n8k16, .m16n8k16, .m16n8k32 },
            .s4, .u4 => &.{ .m8n8k32, .m16n8k32, .m16n8k64 },
            .b1 => &.{ .m8n8k128, .m16n8k128, .m16n8k256 },
        };
        const accumulators: []const Accumulator = switch (options.element) {
            .f16, .e4m3, .e5m2 => &.{ .f16, .f32 },
            .bf16, .tf32 => &.{.f32},
            .f64 => &.{.f64},
            .s8, .u8, .s4, .u4, .b1 => &.{.s32},
        };
        const b_matches = switch (options.element) {
            .e4m3, .e5m2 => b_element == .e4m3 or b_element == .e5m2,
            .s8, .u8 => b_element == .s8 or b_element == .u8,
            .s4, .u4 => b_element == .s4 or b_element == .u4,
            else => b_element == options.element,
        };
        if (std.mem.findScalar(MmaSyncShape, shapes, options.shape) == null or
            std.mem.findScalar(Accumulator, accumulators, options.accumulator) == null or
            !b_matches) mmaError(options, "does not exist");
        const quad_pair = options.element == .f16 and options.shape == .m8n8k4;
        if (!quad_pair and (options.a_layout != .row or options.b_layout != .col)) {
            mmaError(options, "needs a_layout .row and b_layout .col");
        }
        const integer = switch (options.element) {
            .s8, .u8, .s4, .u4 => true,
            else => false,
        };
        if (options.satfinite and !integer) mmaError(options, "does not accept satfinite");
        if ((options.element == .b1) != (options.bit_operation != null)) {
            mmaError(options, "needs a bit_operation exactly for .b1 elements");
        }
        if (options.rounding != null and options.element != .f64) {
            mmaError(options, "does not accept a rounding mode");
        }
        const size = options.shape.dimensions();
        // A warp computes four `.f16` `m8n8k4` products, with 8 work items each (9.7.16.5.1).
        const threads = if (quad_pair) 8 else 32;
        const expected = .{
            .{ "a", A, ElementRegisters(options.element, size.m * size.k / threads) },
            .{ "b", B, ElementRegisters(b_element, size.k * size.n / threads) },
            .{ "c", C, switch (options.accumulator) {
                .f16 => @Vector(size.m * size.n / threads / 2, u32),
                .f32 => @Vector(size.m * size.n / threads, f32),
                .s32 => @Vector(size.m * size.n / threads, i32),
                .f64 => @Vector(size.m * size.n / threads, f64),
            } },
        };
        for (expected) |operand| {
            if (operand[1] != operand[2]) mmaError(options, std.fmt.comptimePrint(
                "needs {s} of type '{s}', found '{s}'",
                .{ operand[0], @typeName(operand[2]), @typeName(operand[1]) },
            ));
        }
    }
}

/// Emits `mma.sync.aligned` (section 9.7.16.5.14) and returns D = A * B + C, with the fragment
/// layouts of section 9.7.16.5. Asserts that all work items of the warp call this function.
pub inline fn mmaSync(comptime options: MmaOptions, a: anytype, b: anytype, c: anytype) @TypeOf(c) {
    if (std.debug.runtime_safety) nvptx.assertConverged(.all);
    comptime validateMma(options, @TypeOf(a), @TypeOf(b), @TypeOf(c));
    const b_element = options.b_element orelse options.element;
    nvptx.require("mma.sync.aligned", switch (options.element) {
        .f16 => switch (options.shape) {
            .m8n8k4 => ptx64_sm70,
            .m16n8k8 => ptx65_sm75,
            else => ptx70_sm80,
        },
        .bf16, .tf32, .f64 => ptx70_sm80,
        .e4m3, .e5m2 => .{
            .ptx = if (options.shape == .m16n8k16 or options.accumulator == .f16) 87 else 84,
            .sm = .{ .base = 89 },
            .threads = .warp,
        },
        .s8, .u8, .s4, .u4 => switch (options.shape) {
            .m8n8k16, .m8n8k32 => ptx65_sm75,
            else => ptx70_sm80,
        },
        .b1 => if (options.bit_operation == .@"and")
            ptx71_sm80
        else if (options.shape == .m8n8k128)
            ptx70_sm75
        else
            ptx70_sm80,
    });
    const accumulator = @tagName(options.accumulator);
    // The backend writes braces around a vector of two or more registers, but not around one.
    const a_list = if (@typeInfo(@TypeOf(a)).vector.len == 1) "{%[a]}" else "%[a]";
    const b_list = if (@typeInfo(@TypeOf(b)).vector.len == 1) "{%[b]}" else "%[b]";
    const text = "mma.sync.aligned." ++ @tagName(options.shape) ++ "." ++
        @tagName(options.a_layout) ++ "." ++ @tagName(options.b_layout) ++
        (if (options.satfinite) ".satfinite" else "") ++
        (if (options.rounding) |rounding| "." ++ @tagName(rounding) else "") ++
        "." ++ accumulator ++ "." ++ @tagName(options.element) ++ "." ++ @tagName(b_element) ++
        "." ++ accumulator ++
        (if (options.bit_operation) |operation| "." ++ @tagName(operation) ++ ".popc" else "") ++
        " %[d], " ++ a_list ++ ", " ++ b_list ++ ", %[c];";
    return switch (comptime options.accumulator) {
        .f64 => asm volatile (text
            : [d] "=d" (-> @TypeOf(c)),
            : [a] "d" (a),
              [b] "d" (b),
              [c] "d" (c),
        ),
        .f32 => asm volatile (text
            : [d] "=f" (-> @TypeOf(c)),
            : [a] "r" (a),
              [b] "r" (b),
              [c] "f" (c),
        ),
        .f16, .s32 => asm volatile (text
            : [d] "=r" (-> @TypeOf(c)),
            : [a] "r" (a),
              [b] "r" (b),
              [c] "r" (c),
        ),
    };
}

/// Within each four-lane group, A's thread selector chooses a lane pair and B's chooses one lane
/// (section 9.7.16.3).
pub const BlockScaleOptions = struct {
    a_byte: u2 = 0,
    a_thread: u1 = 0,
    b_byte: u2 = 0,
    b_thread: u2 = 0,
};

/// Emits `kind::mxf8f6f4.block_scale` with `ue8m0` scale bytes (sections 9.7.16.5.14 and 9.7.16.3).
/// Asserts that all work items of the warp call this function.
pub inline fn mmaSyncBlockScale(
    comptime options: MmaOptions,
    comptime scale_options: BlockScaleOptions,
    a: anytype,
    b: anytype,
    c: anytype,
    scale_a: u32,
    scale_b: u32,
) @TypeOf(c) {
    comptime {
        const b_element = options.b_element orelse options.element;
        if ((options.element != .e4m3 and options.element != .e5m2) or
            (b_element != .e4m3 and b_element != .e5m2))
        {
            mmaError(options, "needs .e4m3 or .e5m2 elements for block scaling");
        }
        if (options.shape != .m16n8k32 or options.accumulator != .f32) {
            mmaError(options, "needs .m16n8k32 and an .f32 accumulator for block scaling");
        }
        validateMma(options, @TypeOf(a), @TypeOf(b), @TypeOf(c));
    }
    if (std.debug.runtime_safety) nvptx.assertConverged(.all);
    nvptx.require("mma.sync.aligned.kind::mxf8f6f4.block_scale", if (nvptx.gpu.sm == 120 and
        nvptx.gpu.variant == .arch_specific) .{
        .ptx = 87,
        .sm = .{ .arch_specific = 120 },
        .threads = .warp,
    } else .{
        .ptx = 88,
        .sm = .{ .family = &.{ 120, 121 } },
        .threads = .warp,
    });
    const digits = "0123";
    const text = "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X" ++
        ".f32." ++ @tagName(options.element) ++ "." ++
        @tagName(options.b_element orelse options.element) ++
        ".f32.ue8m0 %[d], %[a], %[b], %[c], %[scale_a], {" ++
        digits[scale_options.a_byte..][0..1] ++ ", " ++
        digits[scale_options.a_thread..][0..1] ++ "}, %[scale_b], {" ++
        digits[scale_options.b_byte..][0..1] ++ ", " ++
        digits[scale_options.b_thread..][0..1] ++ "};";
    return asm volatile (text
        : [d] "=f" (-> @TypeOf(c)),
        : [a] "r" (a),
          [b] "r" (b),
          [c] "f" (c),
          [scale_a] "r" (scale_a),
          [scale_b] "r" (scale_b),
    );
}

pub const LdmatrixCount = enum(u3) { x1 = 1, x2 = 2, x4 = 4 };

/// Emits `ldmatrix.sync.aligned.m8n8` with `.shared.b16`, and with `.trans` when `transposed`
/// (section 9.7.16.5.15). Work items 8 * i to 8 * i + 7 pass the row addresses of matrix i. Asserts
/// that all work items of the warp call this function. Assumes that each row address is 16-byte
/// aligned.
pub inline fn ldmatrix(
    comptime count: LdmatrixCount,
    comptime transposed: bool,
    address: anytype,
) @Vector(@backingInt(count), u32) {
    if (std.debug.runtime_safety) nvptx.assertConverged(.all);
    nvptx.require("ldmatrix", ptx65_sm75);
    comptime {
        const space = switch (@typeInfo(@TypeOf(address))) {
            .pointer => |info| info.attrs.@"addrspace" orelse .generic,
            else => .generic,
        };
        if (space != .shared) {
            @compileError("ldmatrix reads shared memory, found '" ++ @typeName(@TypeOf(address)) ++ "'");
        }
    }
    const text = comptime switch (count) {
        .x1 => if (transposed)
            "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%[d]}, [%[address]];"
        else
            "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%[d]}, [%[address]];",
        .x2 => if (transposed)
            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 %[d], [%[address]];"
        else
            "ldmatrix.sync.aligned.m8n8.x2.shared.b16 %[d], [%[address]];",
        .x4 => if (transposed)
            "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 %[d], [%[address]];"
        else
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 %[d], [%[address]];",
    };
    return asm volatile (text
        : [d] "=r" (-> @Vector(@backingInt(count), u32)),
        : [address] "r" (address),
        : .{ .memory = true });
}

/// Emits `movmatrix.sync.aligned.m8n8.trans.b16` (section 9.7.16.5.17) to transpose an 8 by 8
/// matrix of 16-bit elements. Asserts that all work items of the warp call this function.
pub inline fn movmatrix(fragment: u32) u32 {
    nvptx.require("movmatrix", .{ .ptx = 78, .sm = .{ .base = 75 }, .threads = .warp });
    if (std.debug.runtime_safety) nvptx.assertConverged(.all);
    return asm volatile ("movmatrix.sync.aligned.m8n8.trans.b16 %[d], %[a];"
        : [d] "=r" (-> u32),
        : [a] "r" (fragment),
    );
}

test "mma shape from dimensions" {
    try std.testing.expectEqual(MmaSyncShape.m16n8k16, comptime MmaSyncShape.fromMnk(16, 8, 16));
    try std.testing.expectEqual(MmaSyncShape.m8n8k128, comptime MmaSyncShape.fromMnk(8, 8, 128));
    inline for (std.meta.tags(MmaSyncShape)) |shape| {
        const size = comptime shape.dimensions();
        try std.testing.expectEqual(shape, comptime MmaSyncShape.fromMnk(size.m, size.n, size.k));
    }
}
