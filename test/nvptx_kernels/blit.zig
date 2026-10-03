//! Copies tensors from the bytes of a checkpoint file to the weights of a model. An entry copies
//! `counts[0] * counts[1] * counts[2] * counts[3]` runs of `run` elements. Dimension `d` moves a
//! run by `source_strides[d]` bytes at the source and `destination_strides[d]` bytes at the
//! destination.

const std = @import("std");
const builtin = @import("builtin");
const device = @import("device.zig");

pub const Kind = enum(u32) {
    words,
    halves,
    /// FP32 elements to BF16, rounded to nearest with ties to even.
    bf16_from_f32,

    pub fn sourceSize(kind: Kind) u32 {
        return switch (kind) {
            .words => 8,
            .halves => 2,
            .bf16_from_f32 => 4,
        };
    }

    pub fn destinationSize(kind: Kind) u32 {
        return switch (kind) {
            .words => 8,
            .halves, .bf16_from_f32 => 2,
        };
    }
};

pub const Entry = extern struct {
    destination: u64,
    source: u64,
    run: u32,
    kind: Kind,
    counts: [4]u32,
    source_strides: [4]u64,
    destination_strides: [4]u64,

    pub fn elements(entry: Entry) u64 {
        var total: u64 = entry.run;
        for (entry.counts) |count| total *= count;
        return total;
    }
};

pub const Blit = struct {
    pub const name = "blit_sm80";
    pub const work_items = 256;

    fn kernel(entries: [*]addrspace(.global) const Entry, slices: u32) callconv(.kernel) void {
        const entry = entries[@workGroupId(1)];
        const total = entry.elements();
        const step = @as(u64, slices) * work_items;
        var i = @as(u64, @workGroupId(0)) * work_items + @workItemId(0);
        while (i < total) : (i += step) {
            var rest = i / entry.run;
            var source = entry.source + (i % entry.run) * entry.kind.sourceSize();
            var destination = entry.destination + (i % entry.run) * entry.kind.destinationSize();
            inline for (0..4) |d| {
                const index = rest % entry.counts[d];
                rest /= entry.counts[d];
                source += index * entry.source_strides[d];
                destination += index * entry.destination_strides[d];
            }
            switch (entry.kind) {
                .words => @as(*addrspace(.global) u64, @ptrFromInt(destination)).* = @as(*addrspace(.global) const u64, @ptrFromInt(source)).*,
                .halves => @as(*addrspace(.global) u16, @ptrFromInt(destination)).* = @as(*addrspace(.global) const u16, @ptrFromInt(source)).*,
                .bf16_from_f32 => @as(*addrspace(.global) u16, @ptrFromInt(destination)).* = device.f32ToBf16(@as(*addrspace(.global) const f32, @ptrFromInt(source)).*),
            }
        }
    }
};

pub const kernels = .{Blit};

comptime {
    if (builtin.cpu.arch.isNvptx()) {
        for (kernels) |Kernel| @export(&Kernel.kernel, .{ .name = Kernel.name });
    }
}
