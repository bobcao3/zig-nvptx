//! Runs the tests of a test binary that `-ffuzz` instruments. The default test runner needs the
//! build server in that mode, and the coverage tracer starts the binary itself. The runner ignores
//! its command line arguments.

const std = @import("std");
const builtin = @import("builtin");
const testing = std.testing;

pub const std_options: std.Options = .{ .logFn = log };

fn log(
    comptime message_level: std.log.Level,
    comptime scope: @EnumLiteral(),
    comptime format: []const u8,
    args: anytype,
) void {
    if (@backingInt(message_level) <= @backingInt(testing.log_level)) {
        std.debug.print("[" ++ @tagName(scope) ++ "] (" ++ @tagName(message_level) ++ "): " ++ format ++ "\n", args);
    }
}

pub fn main(init: std.process.Init.Minimal) !void {
    var failed: usize = 0;
    for (builtin.test_functions) |test_fn| {
        testing.environ = init.environ;
        testing.allocator_instance = .init(std.heap.page_allocator, .{
            .canary = 0xc3a701ba,
            .check_write_after_free = true,
        });
        testing.io_instance = .init(testing.allocator, .{
            .argv0 = .init(init.args),
            .environ = init.environ,
        });
        const result = test_fn.func();
        testing.io_instance.deinit();
        const leaks = testing.allocator_instance.deinit();
        if (result) |_| {} else |err| switch (err) {
            error.SkipZigTest => continue,
            else => {
                if (@errorReturnTrace()) |trace| std.debug.dumpErrorReturnTrace(trace);
                std.debug.print("{s}: {t}\n", .{ test_fn.name, err });
                failed += 1;
                continue;
            },
        }
        if (leaks != 0) {
            std.debug.print("{s}: leaked memory\n", .{test_fn.name});
            failed += 1;
        }
    }
    if (failed != 0) std.process.exit(1);
}

/// Runs a fuzz test on its corpus and on an empty input, as the default test runner does in a
/// build without `-ffuzz`.
pub fn fuzz(
    context: anytype,
    comptime testOne: fn (context: @TypeOf(context), *testing.Smith) anyerror!void,
    options: testing.FuzzInputOptions,
) anyerror!void {
    for (options.corpus) |input| {
        var smith: testing.Smith = .{ .in = input };
        try testOne(context, &smith);
    }
    var smith: testing.Smith = .{ .in = "" };
    try testOne(context, &smith);
}
