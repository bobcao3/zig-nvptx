//! Fifth-generation tensor core instructions (section 9.7.18).

const nvptx = @import("../nvptx.zig");
const require = nvptx.require;

const supported: nvptx.Requirement = .{
    .ptx = 86,
    .sm = .{ .family = &.{ 100, 103, 110 } },
    .threads = .thread,
};

pub const ThreadSync = enum { before_thread_sync, after_thread_sync };

/// Emits `tcgen05.fence` (section 9.7.18.11.1). The `.before_thread_sync` fence orders the prior
/// asynchronous `tcgen05` operations before the later `tcgen05` and execution ordering operations.
/// The `.after_thread_sync` fence orders the later asynchronous `tcgen05` operations after the
/// prior `tcgen05` and execution ordering operations.
pub inline fn fence(comptime position: ThreadSync) void {
    require("tcgen05.fence", supported);
    asm volatile ("tcgen05.fence::" ++ @tagName(position) ++ ";" ::: .{ .memory = true });
}

pub const WaitOperation = enum { ld, st };

/// Emits `tcgen05.wait::ld.sync.aligned` or `tcgen05.wait::st.sync.aligned` (section 9.7.18.8.5),
/// which waits for the prior `tcgen05.ld` or `tcgen05.st` operations of the work item.
pub inline fn waitSyncAligned(comptime operation: WaitOperation) void {
    require("tcgen05.wait", supported);
    const text = "tcgen05.wait::" ++ @tagName(operation) ++ ".sync.aligned;";
    asm volatile (text ::: .{ .memory = true });
}
