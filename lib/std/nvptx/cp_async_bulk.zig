//! Bulk asynchronous copies between global and shared memory and their completion groups (section
//! 9.7.10.28).

const nvptx = @import("../nvptx.zig");
const require = nvptx.require;

const groups: nvptx.Requirement = .{ .ptx = 80, .sm = .{ .base = 90 }, .threads = .thread };

/// Emits `cp.async.bulk.commit_group` (section 9.7.10.28.6.1).
pub inline fn commitGroup() void {
    require("cp.async.bulk.commit_group", groups);
    asm volatile ("cp.async.bulk.commit_group;" ::: .{ .memory = true });
}

/// Emits `cp.async.bulk.wait_group` (section 9.7.10.28.6.2), which waits until at most `pending`
/// of the most recent bulk async-groups are pending. With `reads_only`, the instruction emits
/// `.read` and waits only until the operations of the groups have read their tensor maps and
/// sources.
pub inline fn waitGroup(comptime pending: u32, comptime reads_only: bool) void {
    require("cp.async.bulk.wait_group", groups);
    const text = if (reads_only)
        "cp.async.bulk.wait_group.read %[n];"
    else
        "cp.async.bulk.wait_group %[n];";
    asm volatile (text
        :
        : [n] "n" (pending),
        : .{ .memory = true });
}
