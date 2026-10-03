//! Atomic, match and barrier instructions that Zig's atomic builtins cannot express. The section
//! numbers in the doc comments refer to the PTX ISA 9.4.

const std = @import("../std.zig");
const nvptx = @import("../nvptx.zig");

const Scope = nvptx.Scope;
const require = nvptx.require;

pub const AtomicOperation = enum { add, min, max, @"and", @"or", xor, exch, inc, dec };

pub const BarrierReduction = enum { @"and", @"or" };

fn unsupportedOrder(comptime order: std.lang.AtomicOrder) noreturn {
    @compileError("atomics do not take ." ++ @tagName(order));
}

fn unsupported(
    comptime name: []const u8,
    comptime operation: AtomicOperation,
    comptime T: type,
) noreturn {
    @compileError(name ++ " " ++ @tagName(operation) ++ " does not take '" ++ @typeName(T) ++ "'");
}

const atomic_requirement: nvptx.Requirement = .{
    .ptx = 60,
    .sm = .{ .base = 70 },
    .threads = .thread,
};
const cluster_atomic_requirement: nvptx.Requirement = .{
    .ptx = 78,
    .sm = .{ .base = 90 },
    .threads = .thread,
};

inline fn requireHalf(comptime name: []const u8) void {
    require(name, .{ .ptx = 63, .sm = .{ .base = 70 }, .threads = .thread });
}

inline fn requireHalfPair(comptime name: []const u8) void {
    require(name, .{ .ptx = 62, .sm = .{ .base = 60 }, .threads = .thread });
}

fn qualifiers(
    comptime scope: Scope,
    comptime order: std.lang.AtomicOrder,
    comptime space: std.lang.AddressSpace,
) []const u8 {
    return switch (space) {
        .generic => switch (scope) {
            .cta => switch (order) {
                .monotonic => ".relaxed.cta",
                .acquire => ".acquire.cta",
                .release => ".release.cta",
                .acq_rel => ".acq_rel.cta",
                else => unsupportedOrder(order),
            },
            .cluster => switch (order) {
                .monotonic => ".relaxed.cluster",
                .acquire => ".acquire.cluster",
                .release => ".release.cluster",
                .acq_rel => ".acq_rel.cluster",
                else => unsupportedOrder(order),
            },
            .gpu => switch (order) {
                .monotonic => ".relaxed.gpu",
                .acquire => ".acquire.gpu",
                .release => ".release.gpu",
                .acq_rel => ".acq_rel.gpu",
                else => unsupportedOrder(order),
            },
            .sys => switch (order) {
                .monotonic => ".relaxed.sys",
                .acquire => ".acquire.sys",
                .release => ".release.sys",
                .acq_rel => ".acq_rel.sys",
                else => unsupportedOrder(order),
            },
        },
        .global => switch (scope) {
            .cta => switch (order) {
                .monotonic => ".relaxed.cta.global",
                .acquire => ".acquire.cta.global",
                .release => ".release.cta.global",
                .acq_rel => ".acq_rel.cta.global",
                else => unsupportedOrder(order),
            },
            .cluster => switch (order) {
                .monotonic => ".relaxed.cluster.global",
                .acquire => ".acquire.cluster.global",
                .release => ".release.cluster.global",
                .acq_rel => ".acq_rel.cluster.global",
                else => unsupportedOrder(order),
            },
            .gpu => switch (order) {
                .monotonic => ".relaxed.gpu.global",
                .acquire => ".acquire.gpu.global",
                .release => ".release.gpu.global",
                .acq_rel => ".acq_rel.gpu.global",
                else => unsupportedOrder(order),
            },
            .sys => switch (order) {
                .monotonic => ".relaxed.sys.global",
                .acquire => ".acquire.sys.global",
                .release => ".release.sys.global",
                .acq_rel => ".acq_rel.sys.global",
                else => unsupportedOrder(order),
            },
        },
        .shared => switch (scope) {
            .cta => switch (order) {
                .monotonic => ".relaxed.cta.shared",
                .acquire => ".acquire.cta.shared",
                .release => ".release.cta.shared",
                .acq_rel => ".acq_rel.cta.shared",
                else => unsupportedOrder(order),
            },
            .cluster => switch (order) {
                .monotonic => ".relaxed.cluster.shared",
                .acquire => ".acquire.cluster.shared",
                .release => ".release.cluster.shared",
                .acq_rel => ".acq_rel.cluster.shared",
                else => unsupportedOrder(order),
            },
            .gpu => switch (order) {
                .monotonic => ".relaxed.gpu.shared",
                .acquire => ".acquire.gpu.shared",
                .release => ".release.gpu.shared",
                .acq_rel => ".acq_rel.gpu.shared",
                else => unsupportedOrder(order),
            },
            .sys => switch (order) {
                .monotonic => ".relaxed.sys.shared",
                .acquire => ".acquire.sys.shared",
                .release => ".release.sys.shared",
                .acq_rel => ".acq_rel.sys.shared",
                else => unsupportedOrder(order),
            },
        },
        else => @compileError("atomics do not take address space " ++ @tagName(space)),
    };
}

inline fn atomU32(
    comptime q: []const u8,
    comptime operation: AtomicOperation,
    address: usize,
    value: u32,
) u32 {
    return switch (operation) {
        .add => asm volatile ("atom" ++ q ++ ".add.u32 %[d], [%[a]], %[b];"
            : [d] "=r" (-> u32),
            : [a] "l" (address),
              [b] "r" (value),
            : .{ .memory = true }),
        .min => asm volatile ("atom" ++ q ++ ".min.u32 %[d], [%[a]], %[b];"
            : [d] "=r" (-> u32),
            : [a] "l" (address),
              [b] "r" (value),
            : .{ .memory = true }),
        .max => asm volatile ("atom" ++ q ++ ".max.u32 %[d], [%[a]], %[b];"
            : [d] "=r" (-> u32),
            : [a] "l" (address),
              [b] "r" (value),
            : .{ .memory = true }),
        .@"and" => asm volatile ("atom" ++ q ++ ".and.b32 %[d], [%[a]], %[b];"
            : [d] "=r" (-> u32),
            : [a] "l" (address),
              [b] "r" (value),
            : .{ .memory = true }),
        .@"or" => asm volatile ("atom" ++ q ++ ".or.b32 %[d], [%[a]], %[b];"
            : [d] "=r" (-> u32),
            : [a] "l" (address),
              [b] "r" (value),
            : .{ .memory = true }),
        .xor => asm volatile ("atom" ++ q ++ ".xor.b32 %[d], [%[a]], %[b];"
            : [d] "=r" (-> u32),
            : [a] "l" (address),
              [b] "r" (value),
            : .{ .memory = true }),
        .exch => asm volatile ("atom" ++ q ++ ".exch.b32 %[d], [%[a]], %[b];"
            : [d] "=r" (-> u32),
            : [a] "l" (address),
              [b] "r" (value),
            : .{ .memory = true }),
        .inc => asm volatile ("atom" ++ q ++ ".inc.u32 %[d], [%[a]], %[b];"
            : [d] "=r" (-> u32),
            : [a] "l" (address),
              [b] "r" (value),
            : .{ .memory = true }),
        .dec => asm volatile ("atom" ++ q ++ ".dec.u32 %[d], [%[a]], %[b];"
            : [d] "=r" (-> u32),
            : [a] "l" (address),
              [b] "r" (value),
            : .{ .memory = true }),
    };
}

inline fn atomS32(
    comptime q: []const u8,
    comptime operation: AtomicOperation,
    address: usize,
    value: u32,
) u32 {
    return switch (operation) {
        .min => asm volatile ("atom" ++ q ++ ".min.s32 %[d], [%[a]], %[b];"
            : [d] "=r" (-> u32),
            : [a] "l" (address),
              [b] "r" (value),
            : .{ .memory = true }),
        .max => asm volatile ("atom" ++ q ++ ".max.s32 %[d], [%[a]], %[b];"
            : [d] "=r" (-> u32),
            : [a] "l" (address),
              [b] "r" (value),
            : .{ .memory = true }),
        else => unreachable,
    };
}

inline fn atomU64(
    comptime q: []const u8,
    comptime operation: AtomicOperation,
    address: usize,
    value: u64,
) u64 {
    return switch (operation) {
        .add => asm volatile ("atom" ++ q ++ ".add.u64 %[d], [%[a]], %[b];"
            : [d] "=l" (-> u64),
            : [a] "l" (address),
              [b] "l" (value),
            : .{ .memory = true }),
        .min => asm volatile ("atom" ++ q ++ ".min.u64 %[d], [%[a]], %[b];"
            : [d] "=l" (-> u64),
            : [a] "l" (address),
              [b] "l" (value),
            : .{ .memory = true }),
        .max => asm volatile ("atom" ++ q ++ ".max.u64 %[d], [%[a]], %[b];"
            : [d] "=l" (-> u64),
            : [a] "l" (address),
              [b] "l" (value),
            : .{ .memory = true }),
        .@"and" => asm volatile ("atom" ++ q ++ ".and.b64 %[d], [%[a]], %[b];"
            : [d] "=l" (-> u64),
            : [a] "l" (address),
              [b] "l" (value),
            : .{ .memory = true }),
        .@"or" => asm volatile ("atom" ++ q ++ ".or.b64 %[d], [%[a]], %[b];"
            : [d] "=l" (-> u64),
            : [a] "l" (address),
              [b] "l" (value),
            : .{ .memory = true }),
        .xor => asm volatile ("atom" ++ q ++ ".xor.b64 %[d], [%[a]], %[b];"
            : [d] "=l" (-> u64),
            : [a] "l" (address),
              [b] "l" (value),
            : .{ .memory = true }),
        .exch => asm volatile ("atom" ++ q ++ ".exch.b64 %[d], [%[a]], %[b];"
            : [d] "=l" (-> u64),
            : [a] "l" (address),
              [b] "l" (value),
            : .{ .memory = true }),
        else => unreachable,
    };
}

inline fn atomS64(
    comptime q: []const u8,
    comptime operation: AtomicOperation,
    address: usize,
    value: u64,
) u64 {
    return switch (operation) {
        .min => asm volatile ("atom" ++ q ++ ".min.s64 %[d], [%[a]], %[b];"
            : [d] "=l" (-> u64),
            : [a] "l" (address),
              [b] "l" (value),
            : .{ .memory = true }),
        .max => asm volatile ("atom" ++ q ++ ".max.s64 %[d], [%[a]], %[b];"
            : [d] "=l" (-> u64),
            : [a] "l" (address),
              [b] "l" (value),
            : .{ .memory = true }),
        else => unreachable,
    };
}

inline fn atomAddF32(comptime q: []const u8, address: usize, value: u32) u32 {
    return asm volatile ("atom" ++ q ++ ".add.f32 %[d], [%[a]], %[b];"
        : [d] "=r" (-> u32),
        : [a] "l" (address),
          [b] "r" (value),
        : .{ .memory = true });
}

inline fn atomAddF64(comptime q: []const u8, address: usize, value: u64) u64 {
    return asm volatile ("atom" ++ q ++ ".add.f64 %[d], [%[a]], %[b];"
        : [d] "=l" (-> u64),
        : [a] "l" (address),
          [b] "l" (value),
        : .{ .memory = true });
}

inline fn atomAddF16(comptime q: []const u8, address: usize, value: u16) u16 {
    return asm volatile ("atom" ++ q ++ ".add.noftz.f16 %[d], [%[a]], %[b];"
        : [d] "=h" (-> u16),
        : [a] "l" (address),
          [b] "h" (value),
        : .{ .memory = true });
}

inline fn atomAddF16x2(comptime q: []const u8, address: usize, value: u32) u32 {
    return asm volatile ("atom" ++ q ++ ".add.noftz.f16x2 %[d], [%[a]], %[b];"
        : [d] "=r" (-> u32),
        : [a] "l" (address),
          [b] "r" (value),
        : .{ .memory = true });
}

/// Emits `atom` (section 9.7.15.5) and returns the previous value. `add` also takes `f32`, `f64`,
/// `f16` and `@Vector(2, f16)`, `exch` also takes floats, and `inc` and `dec` take `u32`. The f16
/// forms keep subnormals. `order` is `.monotonic`, `.acquire`, `.release` or `.acq_rel`, because
/// the instruction has no `.seq_cst`. Use `nvptx.fence` for that order.
pub inline fn atom(
    comptime scope: Scope,
    comptime order: std.lang.AtomicOrder,
    comptime operation: AtomicOperation,
    ptr: anytype,
    value: @typeInfo(@TypeOf(ptr)).pointer.child,
) @typeInfo(@TypeOf(ptr)).pointer.child {
    const T = @typeInfo(@TypeOf(ptr)).pointer.child;
    require("atom", if (scope == .cluster) cluster_atomic_requirement else atomic_requirement);
    const space = @typeInfo(@TypeOf(ptr)).pointer.attrs.@"addrspace" orelse .generic;
    const q = comptime qualifiers(scope, order, space);
    const address = @intFromPtr(ptr);
    return switch (T) {
        u32 => atomU32(q, operation, address, value),
        i32 => switch (operation) {
            .min, .max => @bitCast(atomS32(q, operation, address, @bitCast(value))),
            .inc, .dec => unsupported("atom", operation, T),
            else => @bitCast(atomU32(q, operation, address, @bitCast(value))),
        },
        u64 => switch (operation) {
            .inc, .dec => unsupported("atom", operation, T),
            else => atomU64(q, operation, address, value),
        },
        i64 => switch (operation) {
            .min, .max => @bitCast(atomS64(q, operation, address, @bitCast(value))),
            .inc, .dec => unsupported("atom", operation, T),
            else => @bitCast(atomU64(q, operation, address, @bitCast(value))),
        },
        f32 => switch (operation) {
            .add => @bitCast(atomAddF32(q, address, @bitCast(value))),
            .exch => @bitCast(atomU32(q, operation, address, @bitCast(value))),
            else => unsupported("atom", operation, T),
        },
        f64 => switch (operation) {
            .add => @bitCast(atomAddF64(q, address, @bitCast(value))),
            .exch => @bitCast(atomU64(q, operation, address, @bitCast(value))),
            else => unsupported("atom", operation, T),
        },
        f16 => switch (operation) {
            .add => add: {
                requireHalf("atom f16");
                break :add @bitCast(atomAddF16(q, address, @bitCast(value)));
            },
            else => unsupported("atom", operation, T),
        },
        @Vector(2, f16) => switch (operation) {
            .add => add: {
                requireHalfPair("atom f16x2");
                break :add @bitCast(atomAddF16x2(q, address, @bitCast(value)));
            },
            else => unsupported("atom", operation, T),
        },
        else => unsupported("atom", operation, T),
    };
}

inline fn atomCas32(
    comptime q: []const u8,
    address: usize,
    expected: u32,
    desired: u32,
) u32 {
    return asm volatile ("atom" ++ q ++ ".cas.b32 %[d], [%[a]], %[b], %[c];"
        : [d] "=r" (-> u32),
        : [a] "l" (address),
          [b] "r" (expected),
          [c] "r" (desired),
        : .{ .memory = true });
}

inline fn atomCas64(
    comptime q: []const u8,
    address: usize,
    expected: u64,
    desired: u64,
) u64 {
    return asm volatile ("atom" ++ q ++ ".cas.b64 %[d], [%[a]], %[b], %[c];"
        : [d] "=l" (-> u64),
        : [a] "l" (address),
          [b] "l" (expected),
          [c] "l" (desired),
        : .{ .memory = true });
}

/// Emits `atom.cas` (section 9.7.15.5) and returns the previous value. `T` is a 32-bit or 64-bit
/// integer or float.
pub inline fn atomCas(
    comptime scope: Scope,
    comptime order: std.lang.AtomicOrder,
    ptr: anytype,
    expected: @typeInfo(@TypeOf(ptr)).pointer.child,
    desired: @typeInfo(@TypeOf(ptr)).pointer.child,
) @typeInfo(@TypeOf(ptr)).pointer.child {
    const T = @typeInfo(@TypeOf(ptr)).pointer.child;
    require("atom", if (scope == .cluster) cluster_atomic_requirement else atomic_requirement);
    const space = @typeInfo(@TypeOf(ptr)).pointer.attrs.@"addrspace" orelse .generic;
    const q = comptime qualifiers(scope, order, space);
    const address = @intFromPtr(ptr);
    return switch (T) {
        u32 => atomCas32(q, address, expected, desired),
        i32, f32 => @bitCast(atomCas32(q, address, @bitCast(expected), @bitCast(desired))),
        u64 => atomCas64(q, address, expected, desired),
        i64, f64 => @bitCast(atomCas64(q, address, @bitCast(expected), @bitCast(desired))),
        else => @compileError("atom.cas needs 32 or 64 bits, found '" ++ @typeName(T) ++ "'"),
    };
}

inline fn redU32(
    comptime q: []const u8,
    comptime operation: AtomicOperation,
    address: usize,
    value: u32,
) void {
    switch (operation) {
        .add => asm volatile ("red" ++ q ++ ".add.u32 [%[a]], %[b];"
            :
            : [a] "l" (address),
              [b] "r" (value),
            : .{ .memory = true }),
        .min => asm volatile ("red" ++ q ++ ".min.u32 [%[a]], %[b];"
            :
            : [a] "l" (address),
              [b] "r" (value),
            : .{ .memory = true }),
        .max => asm volatile ("red" ++ q ++ ".max.u32 [%[a]], %[b];"
            :
            : [a] "l" (address),
              [b] "r" (value),
            : .{ .memory = true }),
        .@"and" => asm volatile ("red" ++ q ++ ".and.b32 [%[a]], %[b];"
            :
            : [a] "l" (address),
              [b] "r" (value),
            : .{ .memory = true }),
        .@"or" => asm volatile ("red" ++ q ++ ".or.b32 [%[a]], %[b];"
            :
            : [a] "l" (address),
              [b] "r" (value),
            : .{ .memory = true }),
        .xor => asm volatile ("red" ++ q ++ ".xor.b32 [%[a]], %[b];"
            :
            : [a] "l" (address),
              [b] "r" (value),
            : .{ .memory = true }),
        .inc => asm volatile ("red" ++ q ++ ".inc.u32 [%[a]], %[b];"
            :
            : [a] "l" (address),
              [b] "r" (value),
            : .{ .memory = true }),
        .dec => asm volatile ("red" ++ q ++ ".dec.u32 [%[a]], %[b];"
            :
            : [a] "l" (address),
              [b] "r" (value),
            : .{ .memory = true }),
        else => unreachable,
    }
}

inline fn redS32(
    comptime q: []const u8,
    comptime operation: AtomicOperation,
    address: usize,
    value: u32,
) void {
    switch (operation) {
        .min => asm volatile ("red" ++ q ++ ".min.s32 [%[a]], %[b];"
            :
            : [a] "l" (address),
              [b] "r" (value),
            : .{ .memory = true }),
        .max => asm volatile ("red" ++ q ++ ".max.s32 [%[a]], %[b];"
            :
            : [a] "l" (address),
              [b] "r" (value),
            : .{ .memory = true }),
        else => unreachable,
    }
}

inline fn redU64(
    comptime q: []const u8,
    comptime operation: AtomicOperation,
    address: usize,
    value: u64,
) void {
    switch (operation) {
        .add => asm volatile ("red" ++ q ++ ".add.u64 [%[a]], %[b];"
            :
            : [a] "l" (address),
              [b] "l" (value),
            : .{ .memory = true }),
        .min => asm volatile ("red" ++ q ++ ".min.u64 [%[a]], %[b];"
            :
            : [a] "l" (address),
              [b] "l" (value),
            : .{ .memory = true }),
        .max => asm volatile ("red" ++ q ++ ".max.u64 [%[a]], %[b];"
            :
            : [a] "l" (address),
              [b] "l" (value),
            : .{ .memory = true }),
        .@"and" => asm volatile ("red" ++ q ++ ".and.b64 [%[a]], %[b];"
            :
            : [a] "l" (address),
              [b] "l" (value),
            : .{ .memory = true }),
        .@"or" => asm volatile ("red" ++ q ++ ".or.b64 [%[a]], %[b];"
            :
            : [a] "l" (address),
              [b] "l" (value),
            : .{ .memory = true }),
        .xor => asm volatile ("red" ++ q ++ ".xor.b64 [%[a]], %[b];"
            :
            : [a] "l" (address),
              [b] "l" (value),
            : .{ .memory = true }),
        else => unreachable,
    }
}

inline fn redS64(
    comptime q: []const u8,
    comptime operation: AtomicOperation,
    address: usize,
    value: u64,
) void {
    switch (operation) {
        .min => asm volatile ("red" ++ q ++ ".min.s64 [%[a]], %[b];"
            :
            : [a] "l" (address),
              [b] "l" (value),
            : .{ .memory = true }),
        .max => asm volatile ("red" ++ q ++ ".max.s64 [%[a]], %[b];"
            :
            : [a] "l" (address),
              [b] "l" (value),
            : .{ .memory = true }),
        else => unreachable,
    }
}

inline fn redAddF32(comptime q: []const u8, address: usize, value: u32) void {
    asm volatile ("red" ++ q ++ ".add.f32 [%[a]], %[b];"
        :
        : [a] "l" (address),
          [b] "r" (value),
        : .{ .memory = true });
}

inline fn redAddF64(comptime q: []const u8, address: usize, value: u64) void {
    asm volatile ("red" ++ q ++ ".add.f64 [%[a]], %[b];"
        :
        : [a] "l" (address),
          [b] "l" (value),
        : .{ .memory = true });
}

inline fn redAddF16(comptime q: []const u8, address: usize, value: u16) void {
    asm volatile ("red" ++ q ++ ".add.noftz.f16 [%[a]], %[b];"
        :
        : [a] "l" (address),
          [b] "h" (value),
        : .{ .memory = true });
}

inline fn redAddF16x2(comptime q: []const u8, address: usize, value: u32) void {
    asm volatile ("red" ++ q ++ ".add.noftz.f16x2 [%[a]], %[b];"
        :
        : [a] "l" (address),
          [b] "r" (value),
        : .{ .memory = true });
}

/// Emits `red` (section 9.7.15.6), which is `atom` without a result or `exch`. `order` is
/// `.monotonic` or `.release`.
pub inline fn red(
    comptime scope: Scope,
    comptime order: std.lang.AtomicOrder,
    comptime operation: AtomicOperation,
    ptr: anytype,
    value: @typeInfo(@TypeOf(ptr)).pointer.child,
) void {
    const T = @typeInfo(@TypeOf(ptr)).pointer.child;
    if (operation == .exch) @compileError("red has no exch");
    if (order == .acquire or order == .acq_rel) {
        @compileError("red takes .monotonic or .release, not ." ++ @tagName(order));
    }
    require("red", if (scope == .cluster) cluster_atomic_requirement else atomic_requirement);
    const space = @typeInfo(@TypeOf(ptr)).pointer.attrs.@"addrspace" orelse .generic;
    const q = comptime qualifiers(scope, order, space);
    const address = @intFromPtr(ptr);
    switch (T) {
        u32 => redU32(q, operation, address, value),
        i32 => switch (operation) {
            .min, .max => redS32(q, operation, address, @bitCast(value)),
            .inc, .dec => unsupported("red", operation, T),
            else => redU32(q, operation, address, @bitCast(value)),
        },
        u64 => switch (operation) {
            .inc, .dec => unsupported("red", operation, T),
            else => redU64(q, operation, address, value),
        },
        i64 => switch (operation) {
            .min, .max => redS64(q, operation, address, @bitCast(value)),
            .inc, .dec => unsupported("red", operation, T),
            else => redU64(q, operation, address, @bitCast(value)),
        },
        f32 => switch (operation) {
            .add => redAddF32(q, address, @bitCast(value)),
            else => unsupported("red", operation, T),
        },
        f64 => switch (operation) {
            .add => redAddF64(q, address, @bitCast(value)),
            else => unsupported("red", operation, T),
        },
        f16 => switch (operation) {
            .add => {
                requireHalf("red f16");
                redAddF16(q, address, @bitCast(value));
            },
            else => unsupported("red", operation, T),
        },
        @Vector(2, f16) => switch (operation) {
            .add => {
                requireHalfPair("red f16x2");
                redAddF16x2(q, address, @bitCast(value));
            },
            else => unsupported("red", operation, T),
        },
        else => unsupported("red", operation, T),
    }
}

inline fn requireMatch(comptime name: []const u8) void {
    require(name, .{ .ptx = 60, .sm = .{ .base = 70 }, .threads = .warp });
}

inline fn matchAny32(value: u32, member_mask: nvptx.MemberMask) u32 {
    return asm volatile ("match.any.sync.b32 %[d], %[a], %[member_mask];"
        : [d] "=r" (-> u32),
        : [a] "r" (value),
          [member_mask] "r" (@backingInt(member_mask)),
    );
}

inline fn matchAny64(value: u64, member_mask: nvptx.MemberMask) u32 {
    return asm volatile ("match.any.sync.b64 %[d], %[a], %[member_mask];"
        : [d] "=r" (-> u32),
        : [a] "l" (value),
          [member_mask] "r" (@backingInt(member_mask)),
    );
}

/// Emits `match.any.sync` (section 9.7.15.11). Returns the mask of the non-exited lanes in
/// `member_mask` that pass a value equal to `value`. `T` is a 32-bit or 64-bit integer or float.
/// Assumes that the calling lane is in `member_mask` and that each non-exited lane in it executes
/// this instruction.
pub inline fn matchAnySync(comptime T: type, value: T, member_mask: nvptx.MemberMask) u32 {
    requireMatch("match.any.sync");
    return switch (T) {
        u32 => matchAny32(value, member_mask),
        i32, f32 => matchAny32(@bitCast(value), member_mask),
        u64 => matchAny64(value, member_mask),
        i64, f64 => matchAny64(@bitCast(value), member_mask),
        else => @compileError("match.any.sync needs 32 or 64 bits, found '" ++ @typeName(T) ++ "'"),
    };
}

pub const MatchAll = struct {
    /// The non-exited lanes of `member_mask` when they all passed the same value, otherwise 0.
    mask: u32,
    all_equal: bool,
};

inline fn matchAll32(value: u32, member_mask: nvptx.MemberMask) MatchAll {
    var all_equal: bool = undefined;
    const mask = asm volatile ("match.all.sync.b32 %[d]|%[p], %[a], %[member_mask];"
        : [d] "=r" (-> u32),
          [p] "=b" (all_equal),
        : [a] "r" (value),
          [member_mask] "r" (@backingInt(member_mask)),
    );
    return .{ .mask = mask, .all_equal = all_equal };
}

inline fn matchAll64(value: u64, member_mask: nvptx.MemberMask) MatchAll {
    var all_equal: bool = undefined;
    const mask = asm volatile ("match.all.sync.b64 %[d]|%[p], %[a], %[member_mask];"
        : [d] "=r" (-> u32),
          [p] "=b" (all_equal),
        : [a] "l" (value),
          [member_mask] "r" (@backingInt(member_mask)),
    );
    return .{ .mask = mask, .all_equal = all_equal };
}

/// Emits `match.all.sync` (section 9.7.15.11).
pub inline fn matchAllSync(comptime T: type, value: T, member_mask: nvptx.MemberMask) MatchAll {
    requireMatch("match.all.sync");
    return switch (T) {
        u32 => matchAll32(value, member_mask),
        i32, f32 => matchAll32(@bitCast(value), member_mask),
        u64 => matchAll64(value, member_mask),
        i64, f64 => matchAll64(@bitCast(value), member_mask),
        else => @compileError("match.all.sync needs 32 or 64 bits, found '" ++ @typeName(T) ++ "'"),
    };
}

inline fn barrierId(id: anytype) u32 {
    const Id = @TypeOf(id);
    return if (Id == @TypeOf(.enum_literal))
        @backingInt(@as(nvptx.NamedBarrier, id))
    else if (@typeInfo(Id) == .@"enum" and @typeInfo(Id).@"enum".tag_type == u4)
        @backingInt(id)
    else
        @compileError("expected an enum(u4) barrier id, found '" ++ @typeName(Id) ++ "'");
}

inline fn requireBarrierCount(comptime name: []const u8, comptime count: u32) void {
    if (count == 0 or count > 1024 or count % nvptx.warp_size != 0) {
        @compileError(name ++ " count is not a multiple of 32 from 32 to 1024");
    }
}

/// ptxas 13.0 can place the SASS `WARPSYNC` of a CTA barrier above earlier stores, right after
/// divergent code. The lanes that hold a spin lock then wait there before they release the lock,
/// and the warp deadlocks. This explicit `bar.warp.sync` orders memory among the threads of the
/// warp (PTX ISA 9.4, section 9.7.15.2). ptxas 13.0 keeps this `bar.warp.sync` below the earlier
/// stores.
pub const barrier_warp_sync = "bar.warp.sync -1;\n";

/// Emits `barrier.arrive.aligned` (section 9.7.15.1). Assumes that all work items of the warp
/// execute the same instruction.
pub inline fn barrierArriveAligned(id: anytype, comptime count: u32) void {
    require("barrier.arrive.aligned", .{ .ptx = 60, .sm = .{ .base = 30 }, .threads = .warp });
    requireBarrierCount("barrier.arrive.aligned", count);
    asm volatile (barrier_warp_sync ++ "barrier.arrive.aligned %[id], %[count];"
        :
        : [id] "r" (barrierId(id)),
          [count] "n" (count),
        : .{ .memory = true });
}

/// Emits `barrier.red.popc.aligned.u32` (section 9.7.15.1) and returns the number of work items
/// that pass true. Assumes that all work items of the warp execute the same instruction.
pub inline fn barrierRedPopcAligned(id: anytype, comptime count: ?u32, predicate: bool) u32 {
    require("barrier.red.popc.aligned", .{
        .ptx = 60,
        .sm = .{ .base = 30 },
        .threads = .work_group,
    });
    if (count) |n| {
        requireBarrierCount("barrier.red.popc.aligned", n);
        return asm volatile (barrier_warp_sync ++ "barrier.red.popc.aligned.u32 %[d], %[id], %[count], %[p];"
            : [d] "=r" (-> u32),
            : [id] "r" (barrierId(id)),
              [count] "n" (n),
              [p] "b" (predicate),
            : .{ .memory = true });
    }
    return asm volatile (barrier_warp_sync ++ "barrier.red.popc.aligned.u32 %[d], %[id], %[p];"
        : [d] "=r" (-> u32),
        : [id] "r" (barrierId(id)),
          [p] "b" (predicate),
        : .{ .memory = true });
}

/// Emits `barrier.red.and.aligned.pred` or `.or` (section 9.7.15.1). Returns whether all, or any,
/// of the work items pass true.
pub inline fn barrierRedAligned(
    comptime reduction: BarrierReduction,
    id: anytype,
    comptime count: ?u32,
    predicate: bool,
) bool {
    require("barrier.red.aligned", .{ .ptx = 60, .sm = .{ .base = 30 }, .threads = .work_group });
    if (count) |n| {
        requireBarrierCount("barrier.red.aligned", n);
        return switch (reduction) {
            .@"and" => asm volatile (barrier_warp_sync ++ "barrier.red.and.aligned.pred %[d], %[id], %[count], %[p];"
                : [d] "=b" (-> bool),
                : [id] "r" (barrierId(id)),
                  [count] "n" (n),
                  [p] "b" (predicate),
                : .{ .memory = true }),
            .@"or" => asm volatile (barrier_warp_sync ++ "barrier.red.or.aligned.pred %[d], %[id], %[count], %[p];"
                : [d] "=b" (-> bool),
                : [id] "r" (barrierId(id)),
                  [count] "n" (n),
                  [p] "b" (predicate),
                : .{ .memory = true }),
        };
    }
    return switch (reduction) {
        .@"and" => asm volatile (barrier_warp_sync ++ "barrier.red.and.aligned.pred %[d], %[id], %[p];"
            : [d] "=b" (-> bool),
            : [id] "r" (barrierId(id)),
              [p] "b" (predicate),
            : .{ .memory = true }),
        .@"or" => asm volatile (barrier_warp_sync ++ "barrier.red.or.aligned.pred %[d], %[id], %[p];"
            : [d] "=b" (-> bool),
            : [id] "r" (barrierId(id)),
              [p] "b" (predicate),
            : .{ .memory = true }),
    };
}

/// Synchronizes all work groups of a kernel launch with a counter in global memory.
pub const GridBarrier = struct {
    counter: *addrspace(.global) u32,

    const flip_bit: u32 = 1 << 31;

    /// Blocks until every work item of every work group has called `sync`, and makes the writes
    /// before the call visible to the reads after it in all work groups. The barrier can be
    /// reused any number of times.
    ///
    /// Assumes that the kernel was launched so that all work groups are resident at the same
    /// time, for example with a cooperative launch. Assumes that the counter holds 0 before the
    /// first use, that no other code writes it, and that all work items of all work groups call
    /// this function from convergent control flow.
    pub inline fn sync(barrier: GridBarrier) void {
        nvptx.barrierSyncAligned(.work_group, null);
        if (@workItemId(0) | @workItemId(1) | @workItemId(2) == 0) {
            const work_groups = nvptx.nctaid(0) * nvptx.nctaid(1) * nvptx.nctaid(2);
            const first = @workGroupId(0) | @workGroupId(1) | @workGroupId(2) == 0;
            const arrival = if (first) flip_bit - (work_groups - 1) else 1;
            const before = atom(.gpu, .acq_rel, .add, barrier.counter, arrival);
            while ((@atomicLoad(u32, barrier.counter, .acquire) ^ before) & flip_bit == 0) {}
        }
        nvptx.barrierSyncAligned(.work_group, null);
    }
};

test qualifiers {
    const relaxed = comptime qualifiers(.cta, .monotonic, .global);
    try std.testing.expectEqualStrings(".relaxed.cta.global", relaxed);
    const acq_rel = comptime qualifiers(.sys, .acq_rel, .generic);
    try std.testing.expectEqualStrings(".acq_rel.sys", acq_rel);
    const release = comptime qualifiers(.gpu, .release, .shared);
    try std.testing.expectEqualStrings(".release.gpu.shared", release);
    const acquire = comptime qualifiers(.cluster, .acquire, .global);
    try std.testing.expectEqualStrings(".acquire.cluster.global", acquire);
}
