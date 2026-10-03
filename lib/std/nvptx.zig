//! Most functions of this namespace emit one PTX instruction. Section numbers refer to PTX ISA 9.4.

const builtin = @import("builtin");
const std = @import("std.zig");

/// PTX calls this constant `WARP_SZ` (section 4.5.1).
pub const warp_size = 32;

pub const Gpu = struct {
    /// The compute capability is major * 10 + minor.
    sm: u16,
    variant: enum { base, family, arch_specific },
    /// The output has this PTX ISA version or a higher one, written as major * 10 + minor.
    ptx: u16,

    /// The `.version` of the output is at least the highest `ptxNN` feature and `lowestPtx(model)`.
    pub fn fromCpu(cpu: std.Target.Cpu) Gpu {
        const model = cpu.model.name;
        var result: Gpu = .{ .sm = 0, .variant = .base, .ptx = lowestPtx(model) };
        if (std.mem.startsWith(u8, model, "sm_")) {
            var end: usize = "sm_".len;
            while (end < model.len and std.ascii.isDigit(model[end])) end += 1;
            result.sm = std.fmt.parseInt(u16, model["sm_".len..end], 10) catch 0;
            if (end < model.len) result.variant = switch (model[end]) {
                'a' => .arch_specific,
                'f' => .family,
                else => .base,
            };
        }
        for (std.meta.tags(std.Target.nvptx.Feature)) |feature| {
            const feature_name = @tagName(feature);
            if (!std.mem.startsWith(u8, feature_name, "ptx") or !cpu.has(.nvptx, feature)) continue;
            const version = std.fmt.parseInt(u16, feature_name["ptx".len..], 10) catch continue;
            result.ptx = @max(result.ptx, version);
        }
        return result;
    }

    fn lowestPtx(model: []const u8) u16 {
        const table = [_]struct { []const u8, u16 }{
            .{ "sm_80", 70 },   .{ "sm_86", 71 },  .{ "sm_87", 74 },  .{ "sm_88", 90 },
            .{ "sm_89", 78 },   .{ "sm_90a", 80 }, .{ "sm_90", 78 },  .{ "sm_100f", 88 },
            .{ "sm_100", 86 },  .{ "sm_101", 86 }, .{ "sm_103", 88 }, .{ "sm_110", 90 },
            .{ "sm_120f", 88 }, .{ "sm_120", 87 }, .{ "sm_121", 88 },
        };
        for (table) |entry| {
            if (std.mem.startsWith(u8, model, entry[0])) return entry[1];
        }
        return 63;
    }

    pub fn name(comptime target: Gpu) []const u8 {
        const suffix = switch (target.variant) {
            .base => "",
            .family => "f",
            .arch_specific => "a",
        };
        return std.fmt.comptimePrint("sm_{d}{s}", .{ target.sm, suffix });
    }
};

pub const gpu: Gpu = .fromCpu(builtin.cpu);

pub const Requirement = struct {
    ptx: u16,
    sm: union(enum) {
        base: u16,
        accelerated: u16,
        family: []const u16,
        arch_specific: u16,
    },
    threads: enum { thread, warp, warpgroup, work_group, cluster },

    pub fn isMet(r: Requirement, target: Gpu) bool {
        if (target.ptx < r.ptx) return false;
        return switch (r.sm) {
            .base => |sm| target.sm >= sm,
            .accelerated => |sm| target.sm >= sm and target.variant != .base,
            .family => |list| target.variant != .base and
                std.mem.findScalar(u16, list, target.sm) != null,
            .arch_specific => |sm| target.sm == sm and target.variant == .arch_specific,
        };
    }

    pub fn describe(comptime r: Requirement) []const u8 {
        const ptx = std.fmt.comptimePrint("PTX {d}.{d}", .{ r.ptx / 10, r.ptx % 10 });
        return ptx ++ " and " ++ switch (r.sm) {
            .base => |sm| std.fmt.comptimePrint("sm_{d} or later", .{sm}),
            .accelerated => |sm| std.fmt.comptimePrint("an a or f target from sm_{d}", .{sm}),
            .arch_specific => |sm| std.fmt.comptimePrint("sm_{d}a", .{sm}),
            .family => |list| family: {
                var text: []const u8 = "";
                for (0..2 * list.len) |i| {
                    const last = i + 1 == 2 * list.len;
                    const separator = if (i == 0) "" else if (last) " or " else ", ";
                    const suffix = if (i % 2 == 0) "a" else "f";
                    const sm = list[i / 2];
                    const entry = std.fmt.comptimePrint("{s}sm_{d}{s}", .{ separator, sm, suffix });
                    text = text ++ entry;
                }
                break :family text;
            },
        };
    }
};

pub inline fn require(comptime name: []const u8, comptime r: Requirement) void {
    const met = comptime switch (r.sm) {
        .base => |sm| gpu.ptx >= r.ptx and gpu.sm >= sm,
        .accelerated => |sm| gpu.ptx >= r.ptx and gpu.sm >= sm and gpu.variant != .base,
        .arch_specific => |sm| gpu.ptx >= r.ptx and gpu.sm == sm and gpu.variant == .arch_specific,
        .family => r.isMet(gpu),
    };
    if (!met) requirementFailed(name, r);
}

fn requirementFailed(comptime name: []const u8, comptime r: Requirement) noreturn {
    const target = comptime std.fmt.comptimePrint("{s} with PTX {d}.{d}", .{
        gpu.name(),
        gpu.ptx / 10,
        gpu.ptx % 10,
    });
    @compileError(name ++ " needs " ++ r.describe() ++ "; the target is " ++ target);
}

/// The `membermask` operand of a `.sync` instruction, one bit for each lane of the warp.
pub const MemberMask = enum(u32) { all = 0xffff_ffff, _ };

pub inline fn assertConverged(member_mask: MemberMask) void {
    const mask = @backingInt(member_mask);
    if (std.debug.runtime_safety) std.debug.assert(activemask() & mask == mask);
}

pub inline fn assertAligned(pointer: anytype, comptime alignment: comptime_int) void {
    if (std.meta.alignment(@TypeOf(pointer)) >= alignment) return;
    std.debug.assert(@intFromPtr(pointer) % alignment == 0);
}

/// PTX has 16 barriers per work group. A kernel can pass its own `enum(u4)` instead.
pub const NamedBarrier = enum(u4) { work_group = 0, _ };

/// Emits `barrier.sync.aligned` (section 9.7.15.1). Without `count`, all work items of the work
/// group take part. Assumes that all work items of the work group execute the same instruction.
pub inline fn barrierSyncAligned(id: anytype, comptime count: ?u32) void {
    require("barrier.sync.aligned", .{ .ptx = 60, .sm = .{ .base = 30 }, .threads = .work_group });
    const Id = @TypeOf(id);
    const barrier: u32 = if (Id == @TypeOf(.enum_literal))
        @backingInt(@as(NamedBarrier, id))
    else if (@typeInfo(Id) == .@"enum" and @typeInfo(Id).@"enum".tag_type == u4)
        @backingInt(id)
    else
        @compileError("expected an enum(u4) barrier id, found '" ++ @typeName(Id) ++ "'");
    if (count) |n| {
        if (n == 0 or n > 1024 or n % warp_size != 0) barrierCountInvalid(n);
        asm volatile (atomic.barrier_warp_sync ++ "barrier.sync.aligned %[id], %[count];"
            :
            : [id] "r" (barrier),
              [count] "n" (n),
            : .{ .memory = true });
    } else {
        asm volatile (atomic.barrier_warp_sync ++ "barrier.sync.aligned %[id];"
            :
            : [id] "r" (barrier),
            : .{ .memory = true });
    }
}

fn barrierCountInvalid(comptime n: u32) noreturn {
    const message = "barrier count {d} is not a multiple of 32 from 32 to 1024";
    @compileError(std.fmt.comptimePrint(message, .{n}));
}

/// Emits `bar.warp.sync` (section 9.7.15.2). Assumes the calling lane is in `member_mask`.
pub inline fn barWarpSync(member_mask: MemberMask) void {
    require("bar.warp.sync", .{ .ptx = 60, .sm = .{ .base = 30 }, .threads = .warp });
    asm volatile ("bar.warp.sync %[member_mask];"
        :
        : [member_mask] "r" (@backingInt(member_mask)),
        : .{ .memory = true });
}

pub const ClusterArriveSemantics = enum { release, relaxed };

/// Emits `barrier.cluster.arrive` (section 9.7.15.3). Without `semantics`, PTX assumes `.release`.
pub inline fn barrierClusterArrive(comptime semantics: ?ClusterArriveSemantics, comptime aligned: bool) void {
    const ptx: u16 = if (semantics == null) 78 else 80;
    require("barrier.cluster.arrive", .{ .ptx = ptx, .sm = .{ .base = 90 }, .threads = .cluster });
    const text = "barrier.cluster.arrive" ++ (if (semantics) |s| "." ++ @tagName(s) else "") ++
        (if (aligned) ".aligned" else "") ++ ";";
    asm volatile (text ::: .{ .memory = true });
}

/// Emits `barrier.cluster.wait` (section 9.7.15.3). PTX assumes `.acquire` without the qualifier, so
/// `acquire` only spells the qualifier out.
pub inline fn barrierClusterWait(comptime acquire: bool, comptime aligned: bool) void {
    const ptx: u16 = if (acquire) 80 else 78;
    require("barrier.cluster.wait", .{ .ptx = ptx, .sm = .{ .base = 90 }, .threads = .cluster });
    const text = "barrier.cluster.wait" ++ (if (acquire) ".acquire" else "") ++
        (if (aligned) ".aligned" else "") ++ ";";
    asm volatile (text ::: .{ .memory = true });
}

pub const GridDependency = enum { wait, launch_dependents };

/// Emits `griddepcontrol` (section 9.7.15.14).
pub inline fn griddepcontrol(comptime action: GridDependency) void {
    require("griddepcontrol", .{ .ptx = 78, .sm = .{ .base = 90 }, .threads = .thread });
    asm volatile ("griddepcontrol." ++ @tagName(action) ++ ";" ::: .{ .memory = true });
}

pub const Scope = enum { cta, cluster, gpu, sys };

/// Emits `fence.sc` for `.seq_cst` and `fence.acq_rel` for `.acq_rel` (section 9.7.15.4).
pub inline fn fence(comptime order: std.lang.AtomicOrder, comptime scope: Scope) void {
    switch (scope) {
        .cluster => require("fence.cluster", .{ .ptx = 78, .sm = .{ .base = 90 }, .threads = .thread }),
        else => require("fence", .{ .ptx = 60, .sm = .{ .base = 70 }, .threads = .thread }),
    }
    switch (order) {
        .seq_cst => switch (scope) {
            .cta => asm volatile ("fence.sc.cta;" ::: .{ .memory = true }),
            .cluster => asm volatile ("fence.sc.cluster;" ::: .{ .memory = true }),
            .gpu => asm volatile ("fence.sc.gpu;" ::: .{ .memory = true }),
            .sys => asm volatile ("fence.sc.sys;" ::: .{ .memory = true }),
        },
        .acq_rel => switch (scope) {
            .cta => asm volatile ("fence.acq_rel.cta;" ::: .{ .memory = true }),
            .cluster => asm volatile ("fence.acq_rel.cluster;" ::: .{ .memory = true }),
            .gpu => asm volatile ("fence.acq_rel.gpu;" ::: .{ .memory = true }),
            .sys => asm volatile ("fence.acq_rel.sys;" ::: .{ .memory = true }),
        },
        else => @compileError("fence supports seq_cst and acq_rel, not " ++ @tagName(order)),
    }
}

/// Emits `fence.proxy.async` (section 9.7.15.4) for `.global`, `.shared::cta` or all state spaces.
pub inline fn fenceProxyAsync(comptime space: ?std.lang.AddressSpace) void {
    require("fence.proxy.async", .{ .ptx = 80, .sm = .{ .base = 90 }, .threads = .thread });
    const text = comptime if (space) |state_space| switch (state_space) {
        .global => "fence.proxy.async.global;",
        .shared => "fence.proxy.async.shared::cta;",
        else => @compileError("fence.proxy.async supports .global and .shared, not " ++ @tagName(state_space)),
    } else "fence.proxy.async;";
    asm volatile (text ::: .{ .memory = true });
}

/// Emits `fence.mbarrier_init.release.cluster` (section 9.7.15.4). The fence orders only the prior
/// `mbarrier.init` operations of the work item on barriers in `.shared::cta`.
pub inline fn fenceMbarrierInit() void {
    require("fence.mbarrier_init", .{ .ptx = 80, .sm = .{ .base = 90 }, .threads = .thread });
    asm volatile ("fence.mbarrier_init.release.cluster;" ::: .{ .memory = true });
}

pub const RestrictedSemantics = enum { acquire, release };

/// Emits `fence.proxy.async::generic` with `.sync_restrict` (section 9.7.15.4). The `.acquire` fence
/// applies to `.shared::cluster` and the `.release` fence to `.shared::cta`, both at cluster scope.
pub inline fn fenceProxyAsyncSyncRestrict(comptime semantics: RestrictedSemantics) void {
    require("fence.proxy.async.sync_restrict", .{ .ptx = 86, .sm = .{ .base = 90 }, .threads = .thread });
    asm volatile (switch (semantics) {
            .acquire => "fence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster;",
            .release => "fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;",
        } ::: .{ .memory = true });
}

/// Emits `mapa.shared::cluster.u32` (section 9.7.10.26). Returns the `.shared::cluster` address in
/// the CTA of cluster rank `rank` that corresponds to the shared address `address` of this CTA.
pub inline fn mapaSharedCluster(address: u32, rank: u32) u32 {
    require("mapa", .{ .ptx = 78, .sm = .{ .base = 90 }, .threads = .thread });
    return asm ("mapa.shared::cluster.u32 %[d], %[a], %[b];"
        : [d] "=r" (-> u32),
        : [a] "r" (address),
          [b] "r" (rank),
    );
}

pub const ShuffleMode = enum { up, down, bfly, idx };

/// This type is the `c` operand of `shfl.sync` (section 9.7.10.6).
pub const ShuffleClamp = packed struct(u32) {
    clamp: u5,
    reserved0: u3 = 0,
    segment_mask: u5,
    reserved1: u19 = 0,

    /// Returns the operand of the CUDA `__shfl_*_sync` functions for segments of `width` lanes.
    /// Asserts that `width` is a power of two from 1 to 32.
    pub inline fn fromWidth(mode: ShuffleMode, width: u32) ShuffleClamp {
        std.debug.assert(std.math.isPowerOfTwo(width) and width <= warp_size);
        return .{
            .clamp = if (mode == .up) 0 else warp_size - 1,
            .segment_mask = @intCast(warp_size - width),
        };
    }
};

/// Emits `shfl.sync` (section 9.7.10.6). Assumes the calling and source lanes are in `member_mask`.
pub inline fn shflSync(
    comptime mode: ShuffleMode,
    comptime T: type,
    value: T,
    lane_or_offset: u32,
    clamp: ShuffleClamp,
    member_mask: MemberMask,
) T {
    require("shfl.sync", .{ .ptx = 60, .sm = .{ .base = 30 }, .threads = .warp });
    if (@bitSizeOf(T) != 32) @compileError("shfl.sync needs 32 bits, found " ++ @typeName(T));
    const text = "shfl.sync." ++ @tagName(mode) ++ ".b32 %[d], %[a], %[b], %[c], %[member_mask];";
    const c: u32 = @bitCast(clamp);
    // A `.b32` instruction accepts `.f32` registers (section 6.1).
    if (T == f32) return asm volatile (text
        : [d] "=f" (-> f32),
        : [a] "f" (value),
          [b] "rn" (lane_or_offset),
          [c] "rn" (c),
          [member_mask] "rn" (@backingInt(member_mask)),
    );
    return @bitCast(asm volatile (text
        : [d] "=r" (-> u32),
        : [a] "r" (@as(u32, @bitCast(value))),
          [b] "rn" (lane_or_offset),
          [c] "rn" (c),
          [member_mask] "rn" (@backingInt(member_mask)),
    ));
}

pub const VoteMode = enum { all, any, uni };

/// Emits `vote.sync` (section 9.7.15.10). Assumes the calling lane is in `member_mask`.
pub inline fn voteSync(comptime mode: VoteMode, predicate: bool, member_mask: MemberMask) bool {
    require("vote.sync", .{ .ptx = 60, .sm = .{ .base = 30 }, .threads = .warp });
    return switch (mode) {
        .all => asm volatile ("vote.sync.all.pred %[d], %[a], %[member_mask];"
            : [d] "=b" (-> bool),
            : [a] "b" (predicate),
              [member_mask] "r" (@backingInt(member_mask)),
        ),
        .any => asm volatile ("vote.sync.any.pred %[d], %[a], %[member_mask];"
            : [d] "=b" (-> bool),
            : [a] "b" (predicate),
              [member_mask] "r" (@backingInt(member_mask)),
        ),
        .uni => asm volatile ("vote.sync.uni.pred %[d], %[a], %[member_mask];"
            : [d] "=b" (-> bool),
            : [a] "b" (predicate),
              [member_mask] "r" (@backingInt(member_mask)),
        ),
    };
}

/// Emits `vote.sync.ballot` (section 9.7.15.10). Assumes the calling lane is in `member_mask`.
pub inline fn voteBallotSync(predicate: bool, member_mask: MemberMask) u32 {
    require("vote.sync.ballot", .{ .ptx = 60, .sm = .{ .base = 30 }, .threads = .warp });
    return asm volatile ("vote.sync.ballot.b32 %[d], %[a], %[member_mask];"
        : [d] "=r" (-> u32),
        : [a] "b" (predicate),
          [member_mask] "r" (@backingInt(member_mask)),
    );
}

/// Returns true in the lane that `elect.sync` elects. Assumes the calling lane is in `member_mask`.
pub inline fn electSync(member_mask: MemberMask) bool {
    require("elect.sync", .{ .ptx = 80, .sm = .{ .base = 90 }, .threads = .warp });
    return asm volatile ("elect.sync _|%[p], %[member_mask];"
        : [p] "=b" (-> bool),
        : [member_mask] "r" (@backingInt(member_mask)),
    );
}

pub const ReduxOperation = enum { add, min, max, @"and", @"or", xor };

/// Emits `redux.sync` (section 9.7.15.13). Assumes the calling lane is in `member_mask`.
pub inline fn reduxSync(
    comptime operation: ReduxOperation,
    comptime T: type,
    value: T,
    member_mask: MemberMask,
) T {
    require("redux.sync", .{ .ptx = 70, .sm = .{ .base = 80 }, .threads = .warp });
    if (@bitSizeOf(T) != 32) @compileError("redux.sync needs 32 bits, found " ++ @typeName(T));
    const bits: u32 = @bitCast(value);
    const result: u32 = switch (operation) {
        .@"and" => asm volatile ("redux.sync.and.b32 %[d], %[a], %[member_mask];"
            : [d] "=r" (-> u32),
            : [a] "r" (bits),
              [member_mask] "r" (@backingInt(member_mask)),
        ),
        .@"or" => asm volatile ("redux.sync.or.b32 %[d], %[a], %[member_mask];"
            : [d] "=r" (-> u32),
            : [a] "r" (bits),
              [member_mask] "r" (@backingInt(member_mask)),
        ),
        .xor => asm volatile ("redux.sync.xor.b32 %[d], %[a], %[member_mask];"
            : [d] "=r" (-> u32),
            : [a] "r" (bits),
              [member_mask] "r" (@backingInt(member_mask)),
        ),
        .add => switch (T) {
            u32 => asm volatile ("redux.sync.add.u32 %[d], %[a], %[member_mask];"
                : [d] "=r" (-> u32),
                : [a] "r" (bits),
                  [member_mask] "r" (@backingInt(member_mask)),
            ),
            i32 => asm volatile ("redux.sync.add.s32 %[d], %[a], %[member_mask];"
                : [d] "=r" (-> u32),
                : [a] "r" (bits),
                  [member_mask] "r" (@backingInt(member_mask)),
            ),
            else => @compileError("redux.sync needs u32 or i32, found '" ++ @typeName(T) ++ "'"),
        },
        .min => switch (T) {
            u32 => asm volatile ("redux.sync.min.u32 %[d], %[a], %[member_mask];"
                : [d] "=r" (-> u32),
                : [a] "r" (bits),
                  [member_mask] "r" (@backingInt(member_mask)),
            ),
            i32 => asm volatile ("redux.sync.min.s32 %[d], %[a], %[member_mask];"
                : [d] "=r" (-> u32),
                : [a] "r" (bits),
                  [member_mask] "r" (@backingInt(member_mask)),
            ),
            else => @compileError("redux.sync needs u32 or i32, found '" ++ @typeName(T) ++ "'"),
        },
        .max => switch (T) {
            u32 => asm volatile ("redux.sync.max.u32 %[d], %[a], %[member_mask];"
                : [d] "=r" (-> u32),
                : [a] "r" (bits),
                  [member_mask] "r" (@backingInt(member_mask)),
            ),
            i32 => asm volatile ("redux.sync.max.s32 %[d], %[a], %[member_mask];"
                : [d] "=r" (-> u32),
                : [a] "r" (bits),
                  [member_mask] "r" (@backingInt(member_mask)),
            ),
            else => @compileError("redux.sync needs u32 or i32, found '" ++ @typeName(T) ++ "'"),
        },
    };
    return @bitCast(result);
}

pub inline fn activemask() u32 {
    require("activemask", .{ .ptx = 62, .sm = .{ .base = 30 }, .threads = .thread });
    return asm volatile ("activemask.b32 %[d];"
        : [d] "=r" (-> u32),
    );
}

pub inline fn laneid() u32 {
    return asm ("mov.u32 %[d], %laneid;"
        : [d] "=r" (-> u32),
    );
}

/// Returns `%nctaid`, the number of work groups of the launch in `dimension` (section 10.7).
pub inline fn nctaid(comptime dimension: u32) u32 {
    return switch (dimension) {
        0 => asm ("mov.u32 %[d], %nctaid.x;"
            : [d] "=r" (-> u32),
        ),
        1 => asm ("mov.u32 %[d], %nctaid.y;"
            : [d] "=r" (-> u32),
        ),
        2 => asm ("mov.u32 %[d], %nctaid.z;"
            : [d] "=r" (-> u32),
        ),
        else => @compileError("nctaid needs a dimension from 0 to 2"),
    };
}

pub inline fn clock() u32 {
    return asm volatile ("mov.u32 %[d], %clock;"
        : [d] "=r" (-> u32),
    );
}

pub inline fn clock64() u64 {
    return asm volatile ("mov.u64 %[d], %clock64;"
        : [d] "=l" (-> u64),
    );
}

/// Returns an approximation of 2 to the power `x` with `ex2.approx.f32` (section 9.7.3.21).
pub inline fn ex2Approx(x: f32) f32 {
    return asm ("ex2.approx.f32 %[d], %[a];"
        : [d] "=f" (-> f32),
        : [a] "f" (x),
    );
}

/// Returns an approximation of log2(`x`) with `lg2.approx.f32` (section 9.7.3.20).
pub inline fn lg2Approx(x: f32) f32 {
    return asm ("lg2.approx.f32 %[d], %[a];"
        : [d] "=f" (-> f32),
        : [a] "f" (x),
    );
}

/// Returns an approximation of the sine of `x` radians with `sin.approx.f32` (section 9.7.3.18).
pub inline fn sinApprox(x: f32) f32 {
    return asm ("sin.approx.f32 %[d], %[a];"
        : [d] "=f" (-> f32),
        : [a] "f" (x),
    );
}

/// Returns an approximation of the cosine of `x` radians with `cos.approx.f32` (section 9.7.3.19).
pub inline fn cosApprox(x: f32) f32 {
    return asm ("cos.approx.f32 %[d], %[a];"
        : [d] "=f" (-> f32),
        : [a] "f" (x),
    );
}

/// Returns an approximation of 1 / `x` with `rcp.approx.f32` (section 9.7.3.13).
pub inline fn rcpApprox(x: f32) f32 {
    return asm ("rcp.approx.f32 %[d], %[a];"
        : [d] "=f" (-> f32),
        : [a] "f" (x),
    );
}

/// Returns an approximation of 1 / sqrt(`x`) with `rsqrt.approx.f32` (section 9.7.3.16).
pub inline fn rsqrtApprox(x: f32) f32 {
    return asm ("rsqrt.approx.f32 %[d], %[a];"
        : [d] "=f" (-> f32),
        : [a] "f" (x),
    );
}

/// Returns an approximation of `a` / `b` with `div.approx.f32` (section 9.7.3.8).
pub inline fn divApprox(a: f32, b: f32) f32 {
    return asm ("div.approx.f32 %[d], %[a], %[b];"
        : [d] "=f" (-> f32),
        : [a] "f" (a),
          [b] "f" (b),
    );
}

test Requirement {
    const models = std.Target.nvptx.cpu;
    const sm_121a = comptime Gpu.fromCpu(models.sm_121a.toCpu(.nvptx64));
    try std.testing.expectEqual(Gpu{ .sm = 121, .variant = .arch_specific, .ptx = 88 }, sm_121a);
    var sm_80_cpu = models.sm_80.toCpu(.nvptx64);
    try std.testing.expectEqual(70, Gpu.fromCpu(sm_80_cpu).ptx);
    sm_80_cpu.features.addFeature(@backingInt(std.Target.nvptx.Feature.ptx85));
    try std.testing.expectEqual(85, Gpu.fromCpu(sm_80_cpu).ptx);

    const gather4: Requirement = .{
        .ptx = 86,
        .sm = .{ .family = &.{ 100, 103, 110 } },
        .threads = .thread,
    };
    try std.testing.expect(!gather4.isMet(sm_121a));
    try std.testing.expect(gather4.isMet(.fromCpu(models.sm_103f.toCpu(.nvptx64))));
    try std.testing.expect(!gather4.isMet(.fromCpu(models.sm_100.toCpu(.nvptx64))));
    try std.testing.expectEqualStrings(
        "PTX 8.6 and sm_100a, sm_100f, sm_103a, sm_103f, sm_110a or sm_110f",
        comptime gather4.describe(),
    );
    const fence_cluster: Requirement = .{ .ptx = 78, .sm = .{ .base = 90 }, .threads = .thread };
    try std.testing.expect(fence_cluster.isMet(sm_121a));
    try std.testing.expect(!fence_cluster.isMet(.fromCpu(models.sm_89.toCpu(.nvptx64))));
    try std.testing.expectEqualStrings("sm_121a", comptime sm_121a.name());
}

test ShuffleClamp {
    try std.testing.expectEqual(0x101f, @as(u32, @bitCast(ShuffleClamp.fromWidth(.down, 16))));
    try std.testing.expectEqual(0x0000, @as(u32, @bitCast(ShuffleClamp.fromWidth(.up, 32))));
    try std.testing.expectEqual(0x1f1f, @as(u32, @bitCast(ShuffleClamp.fromWidth(.idx, 1))));
}

pub const atomic = @import("nvptx/atomic.zig");
pub const cache = @import("nvptx/cache.zig");
pub const cp_async = @import("nvptx/cp_async.zig");
pub const cp_async_bulk = @import("nvptx/cp_async_bulk.zig");
pub const math = @import("nvptx/math.zig");
pub const matrix = @import("nvptx/matrix.zig");
pub const panic = @import("nvptx/panic.zig");
pub const ptx_device_rt = @import("nvptx/ptx_device_rt.zig");
pub const sreg = @import("nvptx/sreg.zig");
pub const tcgen05 = @import("nvptx/tcgen05.zig");
pub const texture = @import("nvptx/texture.zig");
pub const wgmma = @import("nvptx/wgmma.zig");
pub const mbarrier = @import("nvptx/mbarrier.zig");
const packed_math = @import("nvptx/packed.zig");

pub const AtomicOperation = atomic.AtomicOperation;
pub const atom = atomic.atom;
pub const atomCas = atomic.atomCas;
pub const red = atomic.red;
pub const MatchAll = atomic.MatchAll;
pub const matchAnySync = atomic.matchAnySync;
pub const matchAllSync = atomic.matchAllSync;
pub const BarrierReduction = atomic.BarrierReduction;
pub const barrierArriveAligned = atomic.barrierArriveAligned;
pub const barrierRedPopcAligned = atomic.barrierRedPopcAligned;
pub const barrierRedAligned = atomic.barrierRedAligned;
pub const GridBarrier = atomic.GridBarrier;

pub const Mbarrier = mbarrier.Mbarrier;

pub const dp4a = math.dp4a;
pub const dp2a = math.dp2a;
pub const Bf16x2 = packed_math.Bf16x2;
pub const E4m3x2 = packed_math.E4m3x2;
pub const E5m2x2 = packed_math.E5m2x2;
pub const E4m3x4 = packed_math.E4m3x4;
pub const E5m2x4 = packed_math.E5m2x4;
pub const Tf32 = packed_math.Tf32;
pub const bf16FromF32 = packed_math.bf16FromF32;

pub const mmaSync = matrix.mmaSync;
pub const mmaSyncBlockScale = matrix.mmaSyncBlockScale;
pub const ldmatrix = matrix.ldmatrix;
pub const movmatrix = matrix.movmatrix;

pub const smid = sreg.smid;
pub const nsmid = sreg.nsmid;
pub const warpid = sreg.warpid;
pub const nwarpid = sreg.nwarpid;
pub const clusterCtarank = sreg.clusterCtarank;
pub const clusterNctarank = sreg.clusterNctarank;
pub const gridid = sreg.gridid;
pub const lanemaskEq = sreg.lanemaskEq;
pub const lanemaskLe = sreg.lanemaskLe;
pub const lanemaskLt = sreg.lanemaskLt;
pub const lanemaskGe = sreg.lanemaskGe;
pub const lanemaskGt = sreg.lanemaskGt;
pub const clockHi = sreg.clockHi;
pub const globaltimer = sreg.globaltimer;
pub const globaltimerLo = sreg.globaltimerLo;
pub const globaltimerHi = sreg.globaltimerHi;
pub const totalSmemSize = sreg.totalSmemSize;
pub const dynamicSmemSize = sreg.dynamicSmemSize;
pub const reservedSmemOffsetBegin = sreg.reservedSmemOffsetBegin;
pub const reservedSmemOffsetEnd = sreg.reservedSmemOffsetEnd;
pub const reservedSmemOffsetCap = sreg.reservedSmemOffsetCap;
pub const reservedSmemOffset = sreg.reservedSmemOffset;
pub const envreg = sreg.envreg;
pub const nanosleep = sreg.nanosleep;
pub const pmevent = sreg.pmevent;
pub const pmeventMask = sreg.pmeventMask;
pub const vprintf = sreg.vprintf;
pub const malloc = sreg.malloc;
pub const free = sreg.free;
pub const assertFail = sreg.assertFail;
pub const printf = sreg.printf;

pub const TextureObject = texture.TextureObject;
pub const SurfaceObject = texture.SurfaceObject;

test {
    _ = atomic;
    _ = cache;
    _ = cp_async;
    _ = math;
    _ = matrix;
    _ = panic;
    _ = mbarrier;
    _ = packed_math;
    _ = ptx_device_rt;
    _ = sreg;
    _ = texture;
}
