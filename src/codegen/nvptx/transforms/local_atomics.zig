//! This file replaces each atomic or ordered access to `.local` memory, which PTX does not allow
//! (PTX ISA 9.4, sections 9.7.10.8 and 9.7.15.5), with plain accesses. No other thread can access
//! `.local` memory and observe the order. An `atom` becomes a load, the operation and a store, as
//! `NVPTXAtomicLower.cpp` of LLVM does.

const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const Mir = @import("../Mir.zig");
const expand = @import("expand.zig");

pub fn run(gpa: Allocator, mir: *Mir) expand.Error!void {
    for (0..mir.instructions.len) |i| {
        if (lowers(mir, mir.instructions.get(i))) break;
    } else return;
    var b: Mir.Builder = .{ .gpa = gpa, .reg_counts = mir.reg_counts, .label_count = mir.label_count };
    defer b.deinit();
    var words: std.ArrayList(u32) = .empty;
    defer words.deinit(gpa);
    for (0..mir.instructions.len) |i| {
        const inst = mir.instructions.get(i);
        if (!lowers(mir, inst)) {
            try b.addTyped(inst.tag, inst.ty, inst.modifiers, mir.span(inst));
        } else if (inst.tag == .atom) {
            try lowerAtom(&b, mir, inst);
        } else {
            words.clearRetainingCapacity();
            try words.appendSlice(gpa, mir.span(inst));
            const mem_at = mir.memOperandAt(inst) - inst.data;
            var mem: Mir.MemOperand = @bitCast(words.items[mem_at]);
            mem.sem = .none;
            mem.scope = .none;
            words.items[mem_at] = @bitCast(mem);
            try b.addTyped(inst.tag, inst.ty, inst.modifiers, words.items);
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
}

fn lowers(mir: *const Mir, inst: Mir.Inst) bool {
    switch (inst.tag) {
        .ld, .st, .atom => {},
        else => return false,
    }
    const mem = mir.memOperand(inst, 0);
    if (mem.space != .local) return false;
    return inst.tag == .atom or (mem.sem != .none and mem.sem != .@"volatile");
}

fn imm(value: u28) u32 {
    return @bitCast(Mir.Operand{ .kind = .imm, .payload = value });
}

/// `ld` and `st` have no half-precision types (PTX ISA 9.4, section 9.7.10.8).
fn memoryType(ty: Mir.Type) Mir.Type {
    return switch (ty) {
        .f16, .bf16 => .b16,
        .f16x2, .bf16x2 => .b32,
        else => ty,
    };
}

/// `atom` returns the old value, and `.inc` and `.dec` wrap at `b` (PTX ISA 9.4, section 9.7.15.5).
fn lowerAtom(b: *Mir.Builder, mir: *const Mir, inst: Mir.Inst) expand.Error!void {
    assert(inst.modifiers.guard() == .none);
    const range = mir.operands(inst).?;
    const old = mir.extra[range.first];
    const address = mir.extra[range.first + 1];
    const operand = mir.extra[range.first + 2];
    const class = @as(Mir.Operand, @bitCast(old)).regClass();
    const ty = inst.ty;
    const word = class.bitType();
    var mem = mir.memOperand(inst, 0);
    mem.sem = .none;
    mem.scope = .none;
    const mem_word: u32 = @bitCast(mem);
    try b.addTyped(.ld, memoryType(ty), .{ .mem = .{} }, &.{ old, address, mem_word });
    const new = switch (inst.modifiers.atom.op) {
        .exch => operand,
        .add => add: {
            const sum = try b.newReg(class);
            const rounding: Mir.Rounding = switch (ty) {
                .f16, .bf16, .f32, .f64 => .rn,
                else => .none,
            };
            try b.addTyped(.add, ty, .{ .arith = .{ .rounding = rounding } }, &.{ sum, old, operand });
            break :add sum;
        },
        .min, .max, .@"and", .@"or", .xor => |op| result: {
            const result = try b.newReg(class);
            const tag: Mir.Inst.Tag = switch (op) {
                .min => .min,
                .max => .max,
                .@"and" => .@"and",
                .@"or" => .@"or",
                .xor => .xor,
                else => unreachable,
            };
            try b.addTyped(tag, ty, .{ .arith = .{} }, &.{ result, old, operand });
            break :result result;
        },
        .inc => inc: {
            const wraps = try b.newReg(.pred);
            try b.addTyped(.setp, ty, .{ .setp = .{ .cmp = .ge } }, &.{ wraps, old, operand });
            const next = try b.newReg(class);
            try b.addTyped(.add, ty, .{ .arith = .{} }, &.{ next, old, imm(1) });
            const result = try b.newReg(class);
            try b.addTyped(.selp, word, .{ .none = .{} }, &.{ result, imm(0), next, wraps });
            break :inc result;
        },
        .dec => dec: {
            const zero = try b.newReg(.pred);
            try b.addTyped(.setp, ty, .{ .setp = .{ .cmp = .eq } }, &.{ zero, old, imm(0) });
            const above = try b.newReg(.pred);
            try b.addTyped(.setp, ty, .{ .setp = .{ .cmp = .gt } }, &.{ above, old, operand });
            const wraps = try b.newReg(.pred);
            try b.addTyped(.@"or", .pred, .{ .none = .{} }, &.{ wraps, zero, above });
            const previous = try b.newReg(class);
            try b.addTyped(.sub, ty, .{ .arith = .{} }, &.{ previous, old, imm(1) });
            const result = try b.newReg(class);
            try b.addTyped(.selp, word, .{ .none = .{} }, &.{ result, operand, previous, wraps });
            break :dec result;
        },
        .cas => cas: {
            const equal = try b.newReg(.pred);
            try b.addTyped(.setp, word, .{ .setp = .{ .cmp = .eq } }, &.{ equal, old, operand });
            const result = try b.newReg(class);
            try b.addTyped(.selp, word, .{ .none = .{} }, &.{ result, mir.extra[range.first + 3], old, equal });
            break :cas result;
        },
    };
    try b.addTyped(.st, memoryType(ty), .{ .mem = .{} }, &.{ address, new, mem_word });
}
