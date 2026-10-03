//! Asynchronous copies from global memory to shared memory (section 9.7.10.28.3).

const std = @import("../std.zig");
const nvptx = @import("../nvptx.zig");
const Mbarrier = @import("mbarrier.zig").Mbarrier;
const CachePolicy = @import("cache.zig").CachePolicy;
const require = nvptx.require;

const supported: nvptx.Requirement = .{ .ptx = 70, .sm = .{ .base = 80 }, .threads = .thread };

pub const CacheOperator = enum {
    /// `.ca`: cache at all levels.
    cache_all_levels,
    /// `.cg`: cache in L2 only. This operator needs a copy of 16 bytes.
    cache_global_level,
};

fn copySize(comptime op: CacheOperator, comptime Dst: type, comptime Src: type) comptime_int {
    const dst = @typeInfo(Dst).pointer;
    const src = @typeInfo(Src).pointer;
    if (dst.size != .one or (dst.attrs.@"addrspace" orelse .generic) != .shared) {
        @compileError("cp.async needs a single-item pointer in .shared, found '" ++ @typeName(Dst) ++ "'");
    }
    if (src.size != .one or (src.attrs.@"addrspace" orelse .generic) != .global) {
        @compileError("cp.async needs a single-item pointer in .global, found '" ++ @typeName(Src) ++ "'");
    }
    const size = @sizeOf(dst.child);
    if (@sizeOf(src.child) != size) @compileError("cp.async needs equal sizes for source and destination");
    if (size != 4 and size != 8 and size != 16) {
        @compileError(std.fmt.comptimePrint("cp.async copies 4, 8 or 16 bytes, not {d}", .{size}));
    }
    if (op == .cache_global_level and size != 16) @compileError("cp.async.cg copies 16 bytes");
    return size;
}

/// Emits `cp.async` (section 9.7.10.28.3.1). The copy size is the size of the type that `dst`
/// points to. Asserts that both addresses are aligned to the copy size.
pub inline fn copy(comptime op: CacheOperator, dst: anytype, src: anytype) void {
    require("cp.async", supported);
    const size = comptime copySize(op, @TypeOf(dst), @TypeOf(src));
    nvptx.assertAligned(dst, size);
    nvptx.assertAligned(src, size);
    const text = comptime switch (op) {
        .cache_all_levels => switch (size) {
            4 => "cp.async.ca.shared.global [%[dst]], [%[src]], 4;",
            8 => "cp.async.ca.shared.global [%[dst]], [%[src]], 8;",
            else => "cp.async.ca.shared.global [%[dst]], [%[src]], 16;",
        },
        .cache_global_level => "cp.async.cg.shared.global [%[dst]], [%[src]], 16;",
    };
    asm volatile (text
        :
        : [dst] "r" (dst),
          [src] "l" (src),
        : .{ .memory = true });
}

/// Emits `cp.async` with a source size (section 9.7.10.28.3.1). The instruction copies `src_size`
/// bytes and zero-fills the rest. Asserts that both addresses are aligned to the copy size. Assumes
/// that `src_size` is at most the size of `dst.*`.
pub inline fn copyZeroFill(comptime op: CacheOperator, dst: anytype, src: anytype, src_size: u32) void {
    require("cp.async", supported);
    const size = comptime copySize(op, @TypeOf(dst), @TypeOf(src));
    nvptx.assertAligned(dst, size);
    nvptx.assertAligned(src, size);
    const text = comptime switch (op) {
        .cache_all_levels => switch (size) {
            4 => "cp.async.ca.shared.global [%[dst]], [%[src]], 4, %[src_size];",
            8 => "cp.async.ca.shared.global [%[dst]], [%[src]], 8, %[src_size];",
            else => "cp.async.ca.shared.global [%[dst]], [%[src]], 16, %[src_size];",
        },
        .cache_global_level => "cp.async.cg.shared.global [%[dst]], [%[src]], 16, %[src_size];",
    };
    asm volatile (text
        :
        : [dst] "r" (dst),
          [src] "l" (src),
          [src_size] "r" (src_size),
        : .{ .memory = true });
}

/// Emits `cp.async.L2::cache_hint` (section 9.7.10.28.3.1). Asserts that both addresses are
/// aligned to the copy size.
pub inline fn copyWithPolicy(comptime op: CacheOperator, dst: anytype, src: anytype, policy: CachePolicy) void {
    require("cp.async.L2::cache_hint", .{ .ptx = 74, .sm = .{ .base = 80 }, .threads = .thread });
    const size = comptime copySize(op, @TypeOf(dst), @TypeOf(src));
    nvptx.assertAligned(dst, size);
    nvptx.assertAligned(src, size);
    const text = comptime switch (op) {
        .cache_all_levels => switch (size) {
            4 => "cp.async.ca.shared.global.L2::cache_hint [%[dst]], [%[src]], 4, %[policy];",
            8 => "cp.async.ca.shared.global.L2::cache_hint [%[dst]], [%[src]], 8, %[policy];",
            else => "cp.async.ca.shared.global.L2::cache_hint [%[dst]], [%[src]], 16, %[policy];",
        },
        .cache_global_level => "cp.async.cg.shared.global.L2::cache_hint [%[dst]], [%[src]], 16, %[policy];",
    };
    asm volatile (text
        :
        : [dst] "r" (dst),
          [src] "l" (src),
          [policy] "l" (@backingInt(policy)),
        : .{ .memory = true });
}

/// Emits `cp.async.commit_group` (section 9.7.10.28.3.2).
pub inline fn commitGroup() void {
    require("cp.async.commit_group", supported);
    asm volatile ("cp.async.commit_group;" ::: .{ .memory = true });
}

/// Emits `cp.async.wait_group` (section 9.7.10.28.3.3). Waits until at most `pending` of the most
/// recent groups are incomplete.
pub inline fn waitGroup(comptime pending: u32) void {
    require("cp.async.wait_group", supported);
    asm volatile ("cp.async.wait_group %[pending];"
        :
        : [pending] "n" (pending),
        : .{ .memory = true });
}

/// Emits `cp.async.wait_all` (section 9.7.10.28.3.3).
pub inline fn waitAll() void {
    require("cp.async.wait_all", supported);
    asm volatile ("cp.async.wait_all;" ::: .{ .memory = true });
}

/// Emits `cp.async.mbarrier.arrive` (section 9.7.15.16.18). When all earlier copies of the
/// executing work item are complete, the system performs an arrive operation on `m`. The
/// instruction first increments the pending count of `m`, so the system's arrival changes it by
/// zero in total.
pub inline fn mbarrierArrive(m: *addrspace(.shared) Mbarrier) void {
    require("cp.async.mbarrier.arrive", supported);
    asm volatile ("cp.async.mbarrier.arrive.shared.b64 [%[m]];"
        :
        : [m] "r" (m),
        : .{ .memory = true });
}

/// Emits `cp.async.mbarrier.arrive.noinc` (section 9.7.15.16.18). The pending count of `m` is not
/// incremented, so the count that `Mbarrier.init` received must include this arrival.
pub inline fn mbarrierArriveNoinc(m: *addrspace(.shared) Mbarrier) void {
    require("cp.async.mbarrier.arrive.noinc", supported);
    asm volatile ("cp.async.mbarrier.arrive.noinc.shared.b64 [%[m]];"
        :
        : [m] "r" (m),
        : .{ .memory = true });
}
