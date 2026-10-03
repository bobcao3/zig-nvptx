//! Defines the test runner functions that lib/fuzzer.zig calls, for a `-ffuzz` program that has no
//! test runner.

const Slice = @import("std").Build.abi.fuzz.Slice;

export fn runner_test_run(i: u32) void {
    _ = i;
    @panic("not a fuzz test");
}

export fn runner_test_name(i: u32) Slice {
    _ = i;
    @panic("not a fuzz test");
}

export fn runner_start_input_poller() void {
    @panic("not a fuzz test");
}

export fn runner_stop_input_poller() void {
    @panic("not a fuzz test");
}

export fn runner_futex_wait(ptr: *const u32, expected: u32) bool {
    _ = ptr;
    _ = expected;
    @panic("not a fuzz test");
}

export fn runner_futex_wake(ptr: *const u32, waiters: u32) void {
    _ = ptr;
    _ = waiters;
    @panic("not a fuzz test");
}

export fn runner_broadcast_input(test_i: u32, bytes: Slice) void {
    _ = test_i;
    _ = bytes;
    @panic("not a fuzz test");
}
