//! A scalar of a Zig value has a register form and a memory form. A kind describes both forms.

const std = @import("std");
const assert = std.debug.assert;
const Mir = @import("../Mir.zig");
const Kind = @This();

class: Mir.Class,
/// The scalar takes this many bytes in memory.
size: u8,
repr: Repr,
/// An integer has this many significant bits.
bits: u16,

pub const Repr = enum { unsigned, signed, float, bool };

pub const pointer: Kind = .{ .class = .b64, .size = 8, .repr = .unsigned, .bits = 64 };
pub const pointer32: Kind = .{ .class = .b32, .size = 4, .repr = .unsigned, .bits = 32 };
pub const @"bool": Kind = .{ .class = .pred, .size = 1, .repr = .bool, .bits = 1 };
pub const @"f32": Kind = .{ .class = .f32, .size = 4, .repr = .float, .bits = 32 };

pub fn eql(a: Kind, b: Kind) bool {
    return a.class == b.class and a.size == b.size and a.bits == b.bits;
}

pub fn arithmeticType(kind: Kind) Mir.Type {
    return switch (kind.repr) {
        .float => switch (kind.class) {
            .f32 => .f32,
            .f64 => .f64,
            else => .f16,
        },
        .signed => switch (kind.class) {
            .b16 => .s16,
            .b32 => .s32,
            else => .s64,
        },
        .unsigned, .bool => switch (kind.class) {
            .pred => .pred,
            .b16 => .u16,
            .b32 => .u32,
            else => .u64,
        },
    };
}

pub fn memoryType(kind: Kind) Mir.Type {
    if (kind.repr == .float) return switch (kind.class) {
        .f32 => .f32,
        .f64 => .f64,
        else => .b16,
    };
    return switch (kind.size) {
        1 => if (kind.repr == .signed) .s8 else .u8,
        2 => .b16,
        4 => .b32,
        else => .b64,
    };
}

pub fn bitType(kind: Kind) Mir.Type {
    return kind.class.bitType();
}

pub fn isNarrowInt(kind: Kind) bool {
    return kind.repr != .float and kind.class != .pred and kind.bits != kind.class.bits();
}

pub fn int(repr: Repr, bits: u16) Kind {
    assert(bits <= 64);
    return .{
        .class = if (bits <= 16) .b16 else if (bits <= 32) .b32 else .b64,
        .size = if (bits <= 8) 1 else if (bits <= 16) 2 else if (bits <= 32) 4 else 8,
        .repr = repr,
        .bits = @max(bits, 1),
    };
}
