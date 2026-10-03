//! This file does common subexpression elimination (CSE). The pass removes an instruction when a
//! dominating instruction has the same operation and inputs. It also removes copies and integer
//! identities such as `x + 0`. A block parameter is a copy when all branches pass it one value.
//! When a branch is the only way to the next label, the pass removes the branch and the label. It
//! keeps each move of an immediate, because merged moves keep a register live and can force copies.
//! It merges `asm` instructions only within one block, where one set of threads runs both.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Mir = @import("../Mir.zig");
const Cfg = @import("../Cfg.zig");

const none = Cfg.none;

pub fn run(gpa: Allocator, mir: *Mir) Allocator.Error!void {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var work = mir.*;
    work.extra = try arena.dupe(u32, mir.extra);
    const cfg = try Cfg.init(arena, &work);
    const branches = try arena.alloc(std.ArrayList(u32), work.label_count);
    @memset(branches, .empty);
    const guarded_branch_counts = try arena.alloc(u32, work.label_count);
    @memset(guarded_branch_counts, 0);
    for (0..work.instructions.len) |i| {
        const inst = work.instructions.get(i);
        if (inst.tag != .bra) continue;
        if (inst.modifiers.guard() == .none) {
            try branches[work.branchTarget(inst)].append(arena, @intCast(i));
        } else {
            guarded_branch_counts[work.branchTarget(inst)] += 1;
        }
    }

    var c: Cse = .{
        .arena = arena,
        .work = &work,
        .cfg = &cfg,
        .branches = branches,
        .guarded_branch_counts = guarded_branch_counts,
        .replace = try arena.alloc(u32, cfg.reg_count),
        .removed = try .initEmpty(arena, work.instructions.len),
    };
    @memset(c.replace, none);
    try c.walk();

    var b: Mir.Builder = .{ .gpa = gpa, .reg_counts = mir.reg_counts, .label_count = mir.label_count };
    defer b.deinit();
    for (0..work.instructions.len) |i| {
        if (c.removed.isSet(i)) continue;
        const inst = work.instructions.get(i);
        const at = b.here();
        try b.extra.appendSlice(gpa, work.span(inst));
        try b.instructions.append(gpa, .{ .tag = inst.tag, .ty = inst.ty, .modifiers = inst.modifiers, .data = at });
    }
    var instructions = b.instructions;
    b.instructions = .empty;
    errdefer instructions.deinit(gpa);
    const extra = try b.extra.toOwnedSlice(gpa);
    mir.instructions.deinit(gpa);
    gpa.free(mir.extra);
    mir.instructions = instructions.toOwnedSlice();
    mir.extra = extra;
}

const Cse = struct {
    arena: Allocator,
    work: *Mir,
    cfg: *const Cfg,
    /// `branches[l]` lists the unguarded branches to label `l`.
    branches: []const std.ArrayList(u32),
    guarded_branch_counts: []const u32,
    /// `replace[v]` is the value that replaces the removed result `v`, or `none`.
    replace: []u32,
    removed: std.DynamicBitSetUnmanaged,
    /// `table` maps each key to its first instruction in the dominating blocks.
    table: std.StringHashMapUnmanaged(u32) = .empty,
    added: std.ArrayList([]const u8) = .empty,
    refs: std.ArrayList(Ref) = .empty,

    const Ref = struct { at: u32, word: u32, role: Mir.RegRole };

    pub fn reg(c: *Cse, at: u32, role: Mir.RegRole) void {
        c.refs.appendAssumeCapacity(.{ .at = at, .word = c.work.extra[at], .role = role });
    }

    fn collect(c: *Cse, inst: Mir.Inst) Allocator.Error!void {
        c.refs.clearRetainingCapacity();
        try c.refs.ensureTotalCapacity(c.arena, c.work.span(inst).len);
        c.work.visitRegs(inst, c);
    }

    fn valueOf(c: *Cse, word: u32) u32 {
        var w = word;
        while (true) {
            const next = c.replace[c.cfg.regId(@bitCast(w))];
            if (next == none) return w;
            w = next;
        }
    }

    fn walk(c: *Cse) Allocator.Error!void {
        const cfg = c.cfg;
        const children = try c.arena.alloc(std.ArrayList(u32), cfg.blockCount());
        @memset(children, .empty);
        for (cfg.order) |b| {
            if (b != 0) try children[cfg.idom[b]].append(c.arena, b);
        }
        const Frame = struct { block: u32, mark: usize, next: u32 };
        var stack: std.ArrayList(Frame) = .empty;
        try stack.append(c.arena, .{ .block = 0, .mark = 0, .next = 0 });
        try c.enter(0);
        while (stack.items.len > 0) {
            const top = &stack.items[stack.items.len - 1];
            if (top.next < children[top.block].items.len) {
                const child = children[top.block].items[top.next];
                top.next += 1;
                try stack.append(c.arena, .{ .block = child, .mark = c.added.items.len, .next = 0 });
                try c.enter(child);
                continue;
            }
            const mark = top.mark;
            _ = stack.pop();
            while (c.added.items.len > mark) _ = c.table.remove(c.added.pop().?);
        }
    }

    fn enter(c: *Cse, block: u32) Allocator.Error!void {
        const work = c.work;
        for (c.cfg.starts[block]..c.cfg.starts[block + 1]) |i| {
            const inst = work.instructions.get(i);
            try c.collect(inst);
            for (c.refs.items) |ref| {
                if (ref.role == .use) work.extra[ref.at] = c.valueOf(ref.word);
            }
            if (inst.tag == .label) {
                const label = work.labelOf(inst);
                const forwarded = c.forwardParams(inst);
                const sole = c.branches[label].items.len == 1 and c.guarded_branch_counts[label] == 0;
                if (forwarded and sole and c.branches[label].items[0] + 1 == i) {
                    c.removed.set(i - 1);
                    c.removed.set(i);
                }
                continue;
            }
            if (c.identity(inst)) |source| {
                c.replace[c.cfg.regId(@bitCast(work.extra[inst.data]))] = source;
                c.removed.set(i);
                continue;
            }
            if (!isCandidate(work, inst)) continue;
            const inst_key = try c.key(inst, block);
            const gop = try c.table.getOrPut(c.arena, inst_key);
            if (!gop.found_existing) {
                gop.value_ptr.* = @intCast(i);
                try c.added.append(c.arena, inst_key);
                continue;
            }
            const earlier = work.instructions.get(gop.value_ptr.*);
            var earlier_defs: std.ArrayList(u32) = .empty;
            try c.collect(earlier);
            for (c.refs.items) |ref| if (ref.role == .def) try earlier_defs.append(c.arena, ref.word);
            try c.collect(inst);
            var k: usize = 0;
            for (c.refs.items) |ref| {
                if (ref.role != .def) continue;
                c.replace[c.cfg.regId(@bitCast(ref.word))] = earlier_defs.items[k];
                k += 1;
            }
            c.removed.set(i);
        }
    }

    /// Replaces each block parameter with the one value that the branches pass to it, ignoring a
    /// branch that passes the parameter itself. Returns whether the pass replaced every parameter.
    fn forwardParams(c: *Cse, label: Mir.Inst) bool {
        const work = c.work;
        const branches = c.branches[work.labelOf(label)].items;
        var all = true;
        for (work.blockParams(label), 0..) |param, k| {
            var value: u32 = none;
            for (branches) |bra| {
                const arg = c.valueOf(work.branchArgs(work.instructions.get(bra))[k]);
                if (arg == param or arg == value) continue;
                if (value != none) break;
                value = arg;
            } else if (value != none) {
                c.replace[c.cfg.regId(@bitCast(param))] = value;
                continue;
            }
            all = false;
        }
        return all;
    }

    /// Returns the unchanged input of an unguarded copy or integer identity.
    fn identity(c: *const Cse, inst: Mir.Inst) ?u32 {
        if (inst.modifiers.guard() != .none) return null;
        const neutral: u32 = switch (inst.tag) {
            .mov => undefined,
            .add, .sub => if (inst.modifiers.arith.cc or inst.modifiers.arith.sat) return null else 0,
            .@"or", .xor, .shl, .shr => 0,
            .mul => if (inst.modifiers.arith.mode == .lo) 1 else return null,
            else => return null,
        };
        const work = c.work;
        const range = work.operands(inst).?;
        const dst: Mir.Operand = @bitCast(work.extra[range.first]);
        const src: Mir.Operand = @bitCast(work.extra[range.first + 1]);
        if (dst.kind != .reg or src.kind != .reg or src.regClass() != dst.regClass()) return null;
        if (inst.tag == .mov) return if (range.count == 2) @bitCast(src) else null;
        if (range.count != 3 or !isInteger(inst.ty)) return null;
        const rhs: Mir.Operand = @bitCast(work.extra[range.first + 2]);
        return if (rhs.kind == .imm and rhs.payload == neutral) @bitCast(src) else null;
    }

    /// Returns the tag, type, modifiers and inputs of the instruction as bytes.
    fn key(c: *Cse, inst: Mir.Inst, block: u32) Allocator.Error![]const u8 {
        const work = c.work;
        const words = work.span(inst);
        var buffer: std.ArrayList(u32) = .empty;
        try buffer.ensureTotalCapacity(c.arena, words.len + 3);
        buffer.appendAssumeCapacity(@backingInt(inst.tag) | (@as(u32, @backingInt(inst.ty)) << 8) | (@as(u32, @as(u16, @bitCast(inst.modifiers))) << 16));
        buffer.appendAssumeCapacity(if (inst.tag == .@"asm") block else none);
        const start = buffer.items.len;
        buffer.appendSliceAssumeCapacity(words);
        try c.collect(inst);
        for (c.refs.items) |ref| {
            if (ref.role != .use) buffer.items[start + ref.at - inst.data] = 0;
        }
        if (isCommutative(inst) and words.len == 3) {
            const ops = buffer.items[start + 1 ..][0..2];
            if (ops[0] > ops[1]) std.mem.swap(u32, &ops[0], &ops[1]);
        }
        return std.mem.sliceAsBytes(buffer.items);
    }
};

fn isInteger(ty: Mir.Type) bool {
    return switch (ty) {
        .b16, .b32, .b64, .u16, .u32, .u64, .s16, .s32, .s64 => true,
        else => false,
    };
}

fn isCommutative(inst: Mir.Inst) bool {
    if (inst.modifiers.guard() != .none) return false;
    return switch (inst.tag) {
        .@"and", .@"or", .xor, .min, .max => true,
        .add, .mul => !inst.modifiers.arith.cc,
        else => false,
    };
}

fn isCandidate(mir: *const Mir, inst: Mir.Inst) bool {
    if (!mir.isRemovable(inst)) return false;
    return switch (inst.tag) {
        .label, .bra => false,
        .mov => switch (@as(Mir.Operand, @bitCast(mir.extra[mir.operands(inst).?.first + @intFromBool(inst.modifiers.guard() != .none) + 1])).kind) {
            .imm, .imm64 => false,
            else => true,
        },
        .param_load, .ld => mir.isDereferenceableInvariantLoad(inst),
        .@"asm" => !inst.modifiers.@"asm".pointer_inputs,
        else => Mir.hasResult(inst.tag),
    };
}
