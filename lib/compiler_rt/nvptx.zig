//! Replaces the compiler_rt routines of the same names on nvptx64 with the routines of
//! `std.nvptx.ptx_device_rt`. These exports are strong, and compiler_rt exports its own routines as
//! weak symbols, so a strong export takes the name.

const builtin = @import("builtin");
const std = @import("std");

comptime {
    if (!builtin.is_test) {
        @export(&expf, .{ .name = "expf", .visibility = .hidden });
        @export(&exp2f, .{ .name = "exp2f", .visibility = .hidden });
    }
}

fn expf(x: f32) callconv(.c) f32 {
    return std.nvptx.ptx_device_rt.expf(x);
}

fn exp2f(x: f32) callconv(.c) f32 {
    return std.nvptx.ptx_device_rt.exp2f(x);
}
