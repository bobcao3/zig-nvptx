//! Warpgroup-level asynchronous matrix multiply-accumulate instructions (section 9.7.17).

const nvptx = @import("../nvptx.zig");
const require = nvptx.require;

const supported: nvptx.Requirement = .{
    .ptx = 80,
    .sm = .{ .arch_specific = 90 },
    .threads = .warpgroup,
};

/// Emits `wgmma.fence.sync.aligned` (section 9.7.17.7.1), which orders the prior accesses to the
/// registers of the warpgroup before the later `wgmma.mma_async` accesses to them.
pub inline fn fenceSyncAligned() void {
    require("wgmma.fence", supported);
    asm volatile ("wgmma.fence.sync.aligned;" ::: .{ .memory = true });
}

/// Emits `wgmma.commit_group.sync.aligned` (section 9.7.17.7.2), which batches the uncommitted
/// `wgmma.mma_async` operations of the warp into a new wgmma-group.
pub inline fn commitGroupSyncAligned() void {
    require("wgmma.commit_group", supported);
    asm volatile ("wgmma.commit_group.sync.aligned;" ::: .{ .memory = true });
}

/// Emits `wgmma.wait_group.sync.aligned` (section 9.7.17.7.3), which waits until at most `pending`
/// of the most recent wgmma-groups are pending.
pub inline fn waitGroupSyncAligned(comptime pending: u32) void {
    require("wgmma.wait_group", supported);
    asm volatile ("wgmma.wait_group.sync.aligned %[n];"
        :
        : [n] "n" (pending),
        : .{ .memory = true });
}
