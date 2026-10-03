//! Exits with an error when the file in the first argument contains any of the other arguments.

const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 2) std.process.fatal("usage: {s} <file> <text>...", .{args[0]});

    const contents = try std.Io.Dir.cwd().readFileAlloc(io, args[1], arena, .unlimited);
    var found = false;
    for (args[2..]) |text| {
        if (std.mem.find(u8, contents, text) == null) continue;
        std.debug.print("{s} contains '{s}'\n", .{ args[1], text });
        found = true;
    }
    if (found) {
        std.debug.print("========= file contents: =========\n{s}\n", .{contents});
        std.process.exit(1);
    }
}
