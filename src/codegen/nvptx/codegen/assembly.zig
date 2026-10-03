//! This file encodes inline assembly. The backend copies the template text unchanged except for
//! `%%`, `%=`, `%[name]` and a tab after each newline. `%%` becomes `%`, `%=` becomes a label name
//! unique to the `asm` instruction, and `%[name]` becomes the operand bound to `name`.

const std = @import("std");

const Mir = @import("../Mir.zig");
const Layout = @import("Layout.zig");
const Builder = @import("Builder.zig");

pub const Binding = struct {
    name: []const u8,
    value: union(enum) {
        regs: []const Mir.Reg,
        imm: u64,
    },
};

pub const Flags = struct {
    is_volatile: bool,
    clobbers: bool,
    pointer_inputs: bool,
};

/// Appends an `asm` instruction that defines `outputs`. Input `tied[i]` shares a register with
/// output `i`. `tied[i]` is null when output `i` has no tied input.
pub fn emit(builder: *Builder, layout: *const Layout, source: []const u8, outputs: []const Mir.Reg, tied: []const ?Mir.Reg, bindings: []const Binding, flags: Flags) Layout.Error!void {
    const unique = builder.newLabel();
    const start = builder.here();
    try builder.extra.append(builder.arena, 0);
    for (outputs) |reg| try builder.appendArg(reg);
    for (tied) |input| if (input) |reg| try builder.appendArg(reg) else try builder.appendImm(0);
    var count: u32 = @intCast(2 * outputs.len);
    var chunk: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < source.len) : (i += 1) {
        const c = source[i];
        if (c != '%' or i + 1 >= source.len) {
            try chunk.append(builder.arena, c);
            if (c == '\n') try chunk.append(builder.arena, '\t');
            continue;
        }
        switch (source[i + 1]) {
            '%' => {
                try chunk.append(builder.arena, '%');
                i += 1;
            },
            '=' => {
                count += try flushChunk(builder, &chunk);
                try builder.appendArg(Mir.Label{ .index = unique });
                count += 1;
                i += 1;
            },
            '[' => {
                const end = std.mem.findScalarPos(u8, source, i, ']') orelse return layout.fail("unterminated '%[' in asm", .{});
                const name = source[i + 2 .. end];
                const binding = for (bindings) |binding| {
                    if (std.mem.eql(u8, binding.name, name)) break binding;
                } else return layout.fail("asm has no operand named '{s}'", .{name});
                count += try flushChunk(builder, &chunk);
                switch (binding.value) {
                    .regs => |regs| try builder.appendRegList(regs),
                    .imm => |imm| try builder.appendImm(imm),
                }
                count += 1;
                i = end;
            },
            else => try chunk.append(builder.arena, '%'),
        }
    }
    count += try flushChunk(builder, &chunk);
    builder.extra.items[start] = count;
    const output_count = std.math.cast(u11, outputs.len) orelse return layout.fail("asm has more than {d} output registers", .{std.math.maxInt(u11)});
    try builder.insts.append(builder.arena, .{ .tag = .@"asm", .modifiers = .{ .@"asm" = .{
        .outputs = output_count,
        .is_volatile = flags.is_volatile,
        .clobbers = flags.clobbers,
        .pointer_inputs = flags.pointer_inputs,
    } }, .data = start });
}

fn flushChunk(builder: *Builder, chunk: *std.ArrayList(u8)) !u32 {
    if (chunk.items.len == 0) return 0;
    try builder.appendString(chunk.items);
    chunk.clearRetainingCapacity();
    return 1;
}

pub fn checkConstraint(layout: *const Layout, constraint: []const u8, scalars: []const Layout.Scalar) Layout.Error!void {
    if (constraint.len != 1) return layout.fail("unsupported asm constraint '{s}'", .{constraint});
    const class: Mir.Class = switch (constraint[0]) {
        'b' => .pred,
        'h' => .b16,
        'r' => .b32,
        'l' => .b64,
        'f' => .f32,
        'd' => .f64,
        else => return layout.fail("unsupported asm constraint '{s}'", .{constraint}),
    };
    for (scalars) |scalar| {
        if (scalar.kind.class != class) return layout.fail("asm constraint '{s}' needs {t} registers, but the operand has {t} registers", .{
            constraint, class, scalar.kind.class,
        });
    }
}
