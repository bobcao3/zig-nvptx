//! Exits with an error unless the file in the first argument contains the other arguments as
//! non-overlapping substrings in the given order.

const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 2) std.process.fatal("usage: {s} <file> <text>...", .{args[0]});

    const contents = try std.Io.Dir.cwd().readFileAlloc(io, args[1], arena, .unlimited);
    var position: usize = 0;
    for (args[2..]) |text| {
        const index = std.mem.find(u8, contents[position..], text) orelse {
            std.debug.print("{s} has no '{s}' after offset {d}\n", .{ args[1], text, position });
            std.debug.print("========= file contents: =========\n{s}\n", .{contents});
            std.process.exit(1);
        };
        position += index + text.len;
    }
}
