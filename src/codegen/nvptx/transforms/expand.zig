//! This file expands each `copy` into loads and stores. A `copy` of a known size below
//! `Layout.max_register_bytes` bytes becomes one access for each unit, and a `move` of that size
//! loads every unit before it stores one. LLVM draws the same line (`MaxAggrCopySize` in
//! `NVPTXLowerAggrCopies.cpp`). Every other `copy` becomes a loop. ptxas can unroll a loop of a
//! known count and move all its loads to the start, so the loop header holds `nounroll`.

const std = @import("std");
const Allocator = std.mem.Allocator;

const Mir = @import("../Mir.zig");
const Layout = @import("../codegen/Layout.zig");

pub fn run(gpa: Allocator, mir: *Mir) Error!void {
    const tags = mir.instructions.items(.tag);
    if (std.mem.findScalar(Mir.Inst.Tag, tags, .copy) == null) return;
    var b: Mir.Builder = .{ .gpa = gpa, .reg_counts = mir.reg_counts, .label_count = mir.label_count };
    defer b.deinit();
    for (0..mir.instructions.len) |i| {
        const inst = mir.instructions.get(i);
        if (inst.tag == .copy) {
            try expand(&b, mir, inst);
        } else {
            try b.addTyped(inst.tag, inst.ty, inst.modifiers, mir.span(inst));
        }
    }
    var instructions = b.instructions;
    b.instructions = .empty;
    errdefer instructions.deinit(gpa);
    const extra = try b.extra.toOwnedSlice(gpa);
    mir.instructions.deinit(gpa);
    gpa.free(mir.extra);
    mir.instructions = instructions.toOwnedSlice();
    mir.extra = extra;
    mir.reg_counts = b.reg_counts;
    mir.label_count = b.label_count;
}

pub const Error = Allocator.Error || error{TooManyRegisters};

fn imm(value: u64) [3]u32 {
    return .{ @bitCast(Mir.Operand{ .kind = .imm64, .payload = @backingInt(Mir.ImmFormat.unsigned) }), @truncate(value), @truncate(value >> 32) };
}

const Unit = struct { ty: Mir.Type, class: Mir.Class, vec: Mir.Vec };

fn unitOf(log2_unit: u3) Unit {
    return switch (log2_unit) {
        0 => .{ .ty = .u8, .class = .b16, .vec = .scalar },
        1 => .{ .ty = .b16, .class = .b16, .vec = .scalar },
        2 => .{ .ty = .b32, .class = .b32, .vec = .scalar },
        3 => .{ .ty = .b64, .class = .b64, .vec = .scalar },
        else => .{ .ty = .b64, .class = .b64, .vec = .v2 },
    };
}

/// `value` is the source address, or the stored value of a `set`.
const Copy = struct {
    op: Mir.CopyOp,
    dest: u32,
    value: u32,
    /// The number of units, or null when a register holds it.
    count: ?u64,
    count_words: [3]u32,
    count_len: u32,
    dest_mem: Mir.MemOperand,
    src_mem: Mir.MemOperand,

    fn init(mir: *const Mir, inst: Mir.Inst) Copy {
        const range = mir.operands(inst).?;
        const count_op: Mir.Operand = @bitCast(mir.extra[range.first + 2]);
        const op = inst.modifiers.copy.op;
        var count_words: [3]u32 = undefined;
        @memcpy(count_words[0..count_op.len()], mir.extra[range.first + 2 ..][0..count_op.len()]);
        return .{
            .op = op,
            .dest = mir.extra[range.first],
            .value = mir.extra[range.first + 1],
            .count = switch (count_op.kind) {
                .imm => count_op.payload,
                .imm64 => @as(u64, mir.extra[range.first + 3]) | (@as(u64, mir.extra[range.first + 4]) << 32),
                else => null,
            },
            .count_words = count_words,
            .count_len = count_op.len(),
            .dest_mem = mir.memOperand(inst, 0),
            .src_mem = if (op == .set) undefined else mir.memOperand(inst, 1),
        };
    }

    fn unitSize(copy: Copy) u64 {
        return copy.dest_mem.size();
    }

    fn countWords(copy: *const Copy) []const u32 {
        return copy.count_words[0..copy.count_len];
    }
};

fn expand(b: *Mir.Builder, mir: *const Mir, inst: Mir.Inst) Error!void {
    const copy: Copy = .init(mir, inst);
    const unit_size = copy.unitSize();
    if (copy.count) |count| if (count * unit_size < Layout.max_register_bytes) {
        if (copy.op != .move) {
            for (0..count) |k| try moveUnit(b, copy, copy.dest, copy.value, k * unit_size);
            return;
        }
        const values = try b.gpa.alloc([2]u32, count);
        defer b.gpa.free(values);
        for (values, 0..) |*value, k| value.* = try loadUnit(b, copy.src_mem, copy.value, k * unit_size);
        for (values, 0..) |value, k| try storeUnit(b, copy.dest_mem, copy.dest, k * unit_size, value);
        return;
    };
    const exit = b.label_count;
    b.label_count += 1;
    if (copy.op == .move) if (try backwardCheck(b, copy)) |backward| {
        const back = b.label_count;
        b.label_count += 1;
        try b.addTyped(.bra, .none, .{ .none = .{ .guard = .positive } }, &.{ backward, label(back) });
        try loop(b, copy, exit, false);
        try b.addLabel(back, &.{});
        try loop(b, copy, exit, true);
        try b.addLabel(exit, &.{});
        return;
    };
    try loop(b, copy, exit, false);
    try b.addLabel(exit, &.{});
}

fn label(index: u32) u32 {
    return @bitCast(Mir.Operand{ .kind = .label, .payload = @intCast(index) });
}

/// Returns a predicate that is true when the destination of a `move` starts above its source, or
/// null when the ranges cannot overlap. A writable destination never overlaps `.const` or `.param`
/// memory, and two different specific state spaces name different memory.
fn backwardCheck(b: *Mir.Builder, copy: Copy) Error!?u32 {
    const dest_space = copy.dest_mem.space;
    const src_space = copy.src_mem.space;
    var dest = copy.dest;
    var src = copy.value;
    if (dest_space != src_space) {
        if (dest_space != .generic and src_space != .generic) return null;
        dest = try genericAddress(b, dest, dest_space) orelse return null;
        src = try genericAddress(b, src, src_space) orelse return null;
    } else if (@as(Mir.Operand, @bitCast(dest)).regClass() != @as(Mir.Operand, @bitCast(src)).regClass()) {
        // Inference can leave one side of a `.shared` or `.local` copy in a 64-bit register.
        dest = try widened(b, dest);
        src = try widened(b, src);
    }
    const class = @as(Mir.Operand, @bitCast(dest)).regClass();
    const backward = try b.newReg(.pred);
    try b.addTyped(.setp, class.unsignedType(), .{ .setp = .{ .cmp = .hi } }, &.{ backward, dest, src });
    return backward;
}

fn genericAddress(b: *Mir.Builder, address: u32, space: Mir.Space) Error!?u32 {
    switch (space) {
        .generic => return address,
        .@"const", .param => return null,
        .global, .shared, .local => {},
    }
    const generic = try b.newReg(.b64);
    try b.addTyped(.cvta, .u64, .{ .cvta = .{ .space = space } }, &.{ generic, try widened(b, address) });
    return generic;
}

fn widened(b: *Mir.Builder, address: u32) Error!u32 {
    if (@as(Mir.Operand, @bitCast(address)).regClass() == .b64) return address;
    const wide = try b.newReg(.b64);
    try b.addTyped(.cvt, .u64, .{ .cvt = .{ .from = .u32 } }, &.{ wide, address });
    return wide;
}

/// Moves one unit in each iteration, from the last unit down when `backward` is set.
fn loop(b: *Mir.Builder, copy: Copy, exit: u32, backward: bool) Error!void {
    const head = b.label_count;
    b.label_count += 1;
    const start = try b.newReg(.b64);
    var words: [5]u32 = undefined;
    words[0] = start;
    if (backward) {
        @memcpy(words[1..][0..copy.count_len], copy.countWords());
        try b.addTyped(.mov, .b64, .{ .none = .{} }, words[0 .. 1 + copy.count_len]);
    } else {
        try b.addTyped(.mov, .b64, .{ .none = .{} }, &(.{start} ++ imm(0)));
    }
    try b.addBranch(head, &.{start});
    const index = try b.newReg(.b64);
    try b.addLabel(head, &.{index});
    if (copy.count != null) try b.addTyped(.nounroll, .none, .{ .none = .{} }, &.{});
    const done = try b.newReg(.pred);
    if (backward) {
        try b.addTyped(.setp, .u64, .{ .setp = .{ .cmp = .eq } }, &(.{ done, index } ++ imm(0)));
    } else {
        words[0] = done;
        words[1] = index;
        @memcpy(words[2..][0..copy.count_len], copy.countWords());
        try b.addTyped(.setp, .u64, .{ .setp = .{ .cmp = .ge } }, words[0 .. 2 + copy.count_len]);
    }
    try b.addTyped(.bra, .none, .{ .none = .{ .guard = .positive } }, &.{ done, label(exit) });
    const unit = if (backward) unit: {
        const previous = try b.newReg(.b64);
        try b.addTyped(.sub, .u64, .{ .arith = .{} }, &(.{ previous, index } ++ imm(1)));
        break :unit previous;
    } else index;
    const unit_size = copy.unitSize();
    const dest_at = try scaled(b, copy.dest, unit, unit_size);
    const value = if (copy.op == .set) copy.value else try scaled(b, copy.value, unit, unit_size);
    try moveUnit(b, copy, dest_at, value, 0);
    const next = if (backward) unit else next: {
        const following = try b.newReg(.b64);
        try b.addTyped(.add, .u64, .{ .arith = .{} }, &(.{ following, index } ++ imm(1)));
        break :next following;
    };
    try b.addBranch(head, &.{next});
}

fn scaled(b: *Mir.Builder, base: u32, index: u32, scale: u64) Error!u32 {
    const class = @as(Mir.Operand, @bitCast(base)).regClass();
    if (class == .b64) {
        const result = try b.newReg(.b64);
        try b.addTyped(.mad, .s64, .{ .arith = .{ .mode = .lo, .address = true } }, &(.{ result, index } ++ imm(scale) ++ .{base}));
        return result;
    }
    const low = try b.newReg(.b32);
    try b.addTyped(.cvt, .u32, .{ .cvt = .{ .from = .u64 } }, &.{ low, index });
    const result = try b.newReg(.b32);
    try b.addTyped(.mad, .s32, .{ .arith = .{ .mode = .lo, .address = true } }, &(.{ result, low } ++ imm(scale) ++ .{base}));
    return result;
}

fn moveUnit(b: *Mir.Builder, copy: Copy, dest: u32, value: u32, offset: u64) Error!void {
    if (copy.op == .set) return storeUnit(b, copy.dest_mem, dest, offset, .{ value, undefined });
    try storeUnit(b, copy.dest_mem, dest, offset, try loadUnit(b, copy.src_mem, value, offset));
}

/// Each access of an expanded copy keeps the facts of its side. The unit divides its address.
fn unitOperand(mem: Mir.MemOperand) Mir.MemOperand {
    var unit = mem;
    unit.log2_align = mem.log2_size;
    return unit;
}

/// Returns the registers of one unit. Only a unit of 16 bytes uses the second register.
fn loadUnit(b: *Mir.Builder, mem: Mir.MemOperand, src: u32, offset: u64) Error![2]u32 {
    const unit = unitOf(mem.log2_size);
    const modifiers: Mir.Inst.Modifiers = .{ .mem = .{ .vec = unit.vec, .offset = true } };
    const mem_word: u32 = @bitCast(unitOperand(mem));
    if (unit.vec == .scalar) {
        const value = try b.newReg(unit.class);
        try b.addTyped(.ld, unit.ty, modifiers, &(.{ value, src } ++ imm(offset) ++ .{mem_word}));
        return .{ value, undefined };
    }
    const values: [2]u32 = .{ try b.newReg(unit.class), try b.newReg(unit.class) };
    const list: u32 = @bitCast(Mir.Operand{ .kind = .list, .payload = 2 });
    try b.addTyped(.ld, unit.ty, modifiers, &(.{ list, values[0], values[1], src } ++ imm(offset) ++ .{mem_word}));
    return values;
}

fn storeUnit(b: *Mir.Builder, mem: Mir.MemOperand, dest: u32, offset: u64, values: [2]u32) Error!void {
    const unit = unitOf(mem.log2_size);
    const modifiers: Mir.Inst.Modifiers = .{ .mem = .{ .vec = unit.vec, .offset = true } };
    const mem_word: u32 = @bitCast(unitOperand(mem));
    if (unit.vec == .scalar) {
        const ty: Mir.Type = switch (@as(Mir.Operand, @bitCast(values[0])).regClass()) {
            .f32 => .f32,
            .f64 => .f64,
            else => unit.ty,
        };
        try b.addTyped(.st, ty, modifiers, &(.{dest} ++ imm(offset) ++ .{ values[0], mem_word }));
        return;
    }
    const list: u32 = @bitCast(Mir.Operand{ .kind = .list, .payload = 2 });
    try b.addTyped(.st, unit.ty, modifiers, &(.{dest} ++ imm(offset) ++ .{ list, values[0], values[1], mem_word }));
}
