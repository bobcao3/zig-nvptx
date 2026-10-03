//! This file lowers the SSA form of a function to PTX registers. The pass tries to give one
//! register to a copy and its source, to a block parameter and its branch argument, to a guarded
//! result and its else operand, and to a tied `asm` output and its input. When the two values of a
//! pair cannot share a register, the pass emits a `mov`.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Mir = @import("Mir.zig");
const Cfg = @import("Cfg.zig");
const Lower = @This();

const none = Cfg.none;

arena: Allocator,
mir: *const Mir,
cfg: *const Cfg,
live_out: []const std.DynamicBitSetUnmanaged,
def_inst: []u32,
inst_block: []u32,
uses: []std.ArrayList(u32),
/// `group[v]` is the parent of value `v` in a union-find forest. The values of one tree share a
/// register.
group: []u32,
members: []std.ArrayList(u32),
/// `v` copies `root[v]`, directly or through a chain of copies. Values with the same root are
/// equal, so they can share a register even when their live ranges overlap.
root: []u32,
/// `remat[v]` is true when an unguarded read of kernel parameter memory or of constant memory
/// defines `root[v]`. ptxas reloads such a value from the constant bank in the cases that the
/// tests observe, but only while one instruction defines the register of the value.
remat: []bool,
/// `inputs[v]` lists the inputs of the instruction that defines a guarded result or an `asm`
/// output `v`. Only `tie[v]` among these inputs can share a register with `v`.
inputs: [][]const u32,
/// `tie[v]` holds the else operand of a guarded result `v` or the tied input of an `asm` output
/// `v`, and `none` for other values. The instruction reads `tie[v]` from the register of `v`.
tie: []u32,
names: []u32 = &.{},
counts: [Mir.class_count]u32 = @splat(0),
refs: std.ArrayList(Ref) = .empty,

/// Returns `mir` with PTX registers in place of the SSA values. The result shares
/// `text`, `relocs`, `strings`, `symbols` and `locals` with `mir`. The caller frees only the
/// `instructions` and `extra` of the result.
pub fn run(gpa: Allocator, mir: *const Mir) Allocator.Error!Mir {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const cfg = try Cfg.init(arena, mir);
    const live = try cfg.liveness(arena, mir);

    var lower: Lower = .{
        .arena = arena,
        .mir = mir,
        .cfg = &cfg,
        .live_out = live.live_out,
        .def_inst = try arena.alloc(u32, cfg.reg_count),
        .inst_block = try arena.alloc(u32, mir.instructions.len),
        .uses = try arena.alloc(std.ArrayList(u32), cfg.reg_count),
        .group = try arena.alloc(u32, cfg.reg_count),
        .members = try arena.alloc(std.ArrayList(u32), cfg.reg_count),
        .root = try arena.alloc(u32, cfg.reg_count),
        .remat = try arena.alloc(bool, cfg.reg_count),
        .inputs = try arena.alloc([]const u32, cfg.reg_count),
        .tie = try arena.alloc(u32, cfg.reg_count),
    };
    try lower.scan();
    try lower.coalesce();

    var b: Mir.Builder = .{ .gpa = gpa, .reg_counts = @splat(0), .label_count = mir.label_count };
    defer b.deinit();
    try lower.write(&b);
    try tidy(arena, &b);
    var instructions = b.instructions;
    b.instructions = .empty;
    errdefer instructions.deinit(gpa);
    var result = mir.*;
    result.extra = try b.extra.toOwnedSlice(gpa);
    result.instructions = instructions.toOwnedSlice();
    result.reg_counts = b.reg_counts;
    return result;
}

const Ref = struct { at: u32, word: u32, role: Mir.RegRole };

pub fn reg(lower: *Lower, at: u32, role: Mir.RegRole) void {
    lower.refs.appendAssumeCapacity(.{ .at = at, .word = lower.mir.extra[at], .role = role });
}

fn collect(lower: *Lower, inst: Mir.Inst) Allocator.Error!void {
    lower.refs.clearRetainingCapacity();
    try lower.refs.ensureTotalCapacity(lower.arena, lower.mir.span(inst).len);
    lower.mir.visitRegs(inst, lower);
}

fn id(lower: *const Lower, word: u32) u32 {
    return lower.cfg.regId(@bitCast(word));
}

fn scan(lower: *Lower) Allocator.Error!void {
    const mir = lower.mir;
    @memset(lower.def_inst, none);
    @memset(lower.inst_block, none);
    @memset(lower.uses, .empty);
    @memset(lower.inputs, &.{});
    @memset(lower.tie, none);
    for (lower.group, lower.root, 0..) |*g, *r, v| {
        g.* = @intCast(v);
        r.* = @intCast(v);
    }
    @memset(lower.members, .empty);
    for (lower.cfg.order) |b| {
        for (lower.cfg.starts[b]..lower.cfg.starts[b + 1]) |i| lower.inst_block[i] = b;
    }
    for (0..mir.instructions.len) |i| {
        if (lower.inst_block[i] == none) continue;
        const inst = mir.instructions.get(i);
        try lower.collect(inst);
        var inputs: std.ArrayList(u32) = .empty;
        for (lower.refs.items) |ref| switch (ref.role) {
            .def => lower.def_inst[lower.id(ref.word)] = @intCast(i),
            .use => {
                const v = lower.id(ref.word);
                const list = &lower.uses[v];
                if (list.items.len == 0 or list.items[list.items.len - 1] != i) try list.append(lower.arena, @intCast(i));
                try inputs.append(lower.arena, v);
            },
            .def_ref => {},
        };
        const constrained = inst.tag == .@"asm" or Mir.elseCount(inst) != 0;
        if (constrained) for (lower.refs.items) |ref| {
            if (ref.role == .def) lower.inputs[lower.id(ref.word)] = inputs.items;
        };
        if (Mir.elseCount(inst) != 0) {
            const range = mir.operands(inst).?;
            var at = range.first;
            for (0..range.count - 1) |_| at += @as(Mir.Operand, @bitCast(mir.extra[at])).len();
            lower.tieResult(range.first + 1, at);
        }
        if (inst.tag == .@"asm") {
            const outputs = inst.modifiers.@"asm".outputs;
            for (0..outputs) |k| {
                const tied: Mir.Operand = @bitCast(mir.extra[inst.data + 1 + outputs + k]);
                if (tied.kind != .reg) continue;
                lower.tie[lower.id(mir.extra[inst.data + 1 + k])] = lower.id(@bitCast(tied));
            }
        }
        if (inst.tag == .mov) {
            const range = mir.operands(inst).?;
            const dst: Mir.Operand = @bitCast(mir.extra[range.first]);
            const src: Mir.Operand = @bitCast(mir.extra[range.first + 1]);
            if (range.count == 2 and dst.kind == .reg and src.kind == .reg and dst.regClass() == src.regClass()) {
                lower.root[lower.id(@bitCast(dst))] = lower.root[lower.id(@bitCast(src))];
            }
        }
    }
    for (lower.remat, lower.root) |*r, root| {
        const def = lower.def_inst[root];
        r.* = def != none and lower.rematerializable(mir.instructions.get(def));
    }
}

fn rematerializable(lower: *const Lower, inst: Mir.Inst) bool {
    const mir = lower.mir;
    const entry = mir.kind == .entry;
    switch (inst.tag) {
        .param_load, .ld => return inst.modifiers.guard() == .none and mir.isDereferenceableInvariantLoad(inst),
        .mov, .cvta => {
            if (!entry or inst.modifiers.guard() != .none) return false;
            const range = mir.operands(inst).?;
            const src: Mir.Operand = @bitCast(mir.extra[range.first + 1]);
            return src.kind == .param and src.paramRef() == .own;
        },
        else => return false,
    }
}

fn tieResult(lower: *Lower, result_at: u32, else_at: u32) void {
    const extra = lower.mir.extra;
    const result: Mir.Operand = @bitCast(extra[result_at]);
    if (result.kind == .reg) {
        lower.tie[lower.id(extra[result_at])] = lower.id(extra[else_at]);
        return;
    }
    for (0..result.payload) |k| lower.tie[lower.id(extra[result_at + 1 + k])] = lower.id(extra[else_at + 1 + k]);
}

fn liveAfter(lower: *const Lower, v: u32, i: u32) bool {
    const def = lower.def_inst[v];
    if (def == none) return false;
    const b = lower.inst_block[i];
    if (lower.inst_block[def] == b and def > i) return false;
    if (lower.live_out[b].isSet(v)) return true;
    const end = lower.cfg.starts[b + 1];
    for (lower.uses[v].items) |use| {
        if (use > i and use < end) return true;
    }
    return false;
}

fn interferes(lower: *const Lower, x: u32, y: u32) bool {
    if (lower.root[x] == lower.root[y]) return false;
    const dx = lower.def_inst[x];
    const dy = lower.def_inst[y];
    if (dx == none or dy == none) return false;
    if (dx == dy) return true;
    return lower.liveAfter(x, dy) or lower.liveAfter(y, dx);
}

fn find(lower: *Lower, v: u32) u32 {
    var r = v;
    while (lower.group[r] != r) r = lower.group[r];
    var x = v;
    while (lower.group[x] != r) {
        const next = lower.group[x];
        lower.group[x] = r;
        x = next;
    }
    return r;
}

/// Merges the groups of `a` and `b` when `interferes` is false for every pair of values from the
/// two groups, no guarded result or `asm` output gets the register of another input, and no value
/// with `remat` set meets a value that is not a copy of it.
fn merge(lower: *Lower, a: u32, b: u32) Allocator.Error!void {
    if (lower.def_inst[a] == none or lower.def_inst[b] == none) return;
    const root_a = lower.find(a);
    const root_b = lower.find(b);
    if (root_a == root_b) return;
    const left = try lower.groupList(root_a);
    const right = try lower.groupList(root_b);
    for (left) |x| for (right) |y| {
        if (lower.interferes(x, y)) return;
        if (lower.root[x] != lower.root[y] and (lower.remat[x] or lower.remat[y])) return;
    };
    for ([2][]const u32{ left, right }) |side| for (side) |m| {
        for (lower.inputs[m]) |input| {
            if (input == lower.tie[m]) continue;
            const root_input = lower.find(input);
            if (root_input == root_a or root_input == root_b) return;
        }
    };
    lower.group[root_b] = root_a;
    try lower.members[root_a].appendSlice(lower.arena, right);
}

fn groupList(lower: *Lower, rep: u32) Allocator.Error![]const u32 {
    if (lower.members[rep].items.len == 0) try lower.members[rep].append(lower.arena, rep);
    return lower.members[rep].items;
}

fn coalesce(lower: *Lower) Allocator.Error!void {
    const mir = lower.mir;
    for (lower.tie, 0..) |tied, v| {
        if (tied != none) try lower.merge(@intCast(v), tied);
    }
    for (0..mir.instructions.len) |i| {
        if (lower.inst_block[i] == none) continue;
        const inst = mir.instructions.get(i);
        if (inst.tag != .bra or inst.modifiers.guard() != .none) continue;
        const target = lower.cfg.label_blocks[mir.branchTarget(inst)];
        const params = mir.blockParams(mir.instructions.get(lower.cfg.starts[target]));
        for (params, mir.branchArgs(inst)) |param, arg| try lower.merge(lower.id(param), lower.id(arg));
    }
    for (0..mir.instructions.len) |i| {
        if (lower.inst_block[i] == none) continue;
        const inst = mir.instructions.get(i);
        if (inst.tag != .mov) continue;
        const range = mir.operands(inst).?;
        if (range.count != 2) continue;
        const dst: Mir.Operand = @bitCast(mir.extra[range.first]);
        const src: Mir.Operand = @bitCast(mir.extra[range.first + 1]);
        if (dst.kind != .reg or src.kind != .reg or dst.regClass() != src.regClass()) continue;
        try lower.merge(lower.id(@bitCast(dst)), lower.id(@bitCast(src)));
    }
}

/// Returns the register name for the group of `word`. The first request names the group.
fn nameOf(lower: *Lower, word: u32) u32 {
    const rep = lower.find(lower.id(word));
    if (lower.names[rep] == none) {
        const class = @as(Mir.Operand, @bitCast(word)).regClass();
        const count = &lower.counts[@backingInt(class)];
        lower.names[rep] = @bitCast(Mir.Operand.reg(class, @intCast(count.*)));
        count.* += 1;
    }
    return lower.names[rep];
}

fn temporary(lower: *Lower, class: Mir.Class) u32 {
    const count = &lower.counts[@backingInt(class)];
    defer count.* += 1;
    return @bitCast(Mir.Operand.reg(class, @intCast(count.*)));
}

fn write(lower: *Lower, b: *Mir.Builder) Allocator.Error!void {
    const mir = lower.mir;
    const gpa = b.gpa;
    lower.names = try lower.arena.alloc(u32, lower.cfg.reg_count);
    @memset(lower.names, none);
    const count = mir.instructions.len;
    for (0..count) |i| {
        if (lower.inst_block[i] == none) continue;
        const inst = mir.instructions.get(i);
        switch (inst.tag) {
            .label => {
                try b.addLabel(mir.labelOf(inst), &.{});
                continue;
            },
            .bra => if (inst.modifiers.guard() == .none) {
                const target = mir.branchTarget(inst);
                const target_block = lower.cfg.label_blocks[target];
                const params = mir.blockParams(mir.instructions.get(lower.cfg.starts[target_block]));
                var copies: std.ArrayList(Copy) = .empty;
                for (params, mir.branchArgs(inst)) |param, arg| {
                    if (lower.def_inst[lower.id(arg)] == none) continue;
                    const dst = lower.nameOf(param);
                    const src = lower.nameOf(arg);
                    if (dst != src) try copies.append(lower.arena, .{ .dst = dst, .src = src });
                }
                try lower.parallelCopy(b, copies.items);
                const next_is_target = i + 1 < count and lower.inst_block[i + 1] != none and
                    mir.instructions.items(.tag)[i + 1] == .label and mir.labelOf(mir.instructions.get(i + 1)) == target;
                if (!next_is_target) try b.addBranch(target, &.{});
                continue;
            },
            .mov => {
                const range = mir.operands(inst).?;
                const dst: Mir.Operand = @bitCast(mir.extra[range.first]);
                const src: Mir.Operand = @bitCast(mir.extra[range.first + 1]);
                if (range.count == 2 and dst.kind == .reg and src.kind == .reg and
                    lower.nameOf(@bitCast(dst)) == lower.nameOf(@bitCast(src))) continue;
            },
            else => {},
        }
        try lower.collect(inst);
        for (lower.refs.items) |ref| {
            if (ref.role != .def) continue;
            const v = lower.id(ref.word);
            const tied = lower.tie[v];
            if (tied == none or lower.def_inst[tied] == none) continue;
            const dst = lower.nameOf(ref.word);
            const src = lower.nameOf(mir.extra[lower.tieAt(inst, ref.at)]);
            if (dst != src) try addMov(b, dst, src);
        }
        const at = b.here();
        try b.extra.appendSlice(gpa, mir.span(inst));
        for (lower.refs.items) |ref| b.extra.items[at + (ref.at - inst.data)] = lower.nameOf(ref.word);
        try b.instructions.append(gpa, .{ .tag = inst.tag, .ty = inst.ty, .modifiers = inst.modifiers, .data = at });
    }
    b.reg_counts = lower.counts;
}

fn tieAt(lower: *const Lower, inst: Mir.Inst, result_at: u32) u32 {
    const mir = lower.mir;
    if (inst.tag == .@"asm") return result_at + inst.modifiers.@"asm".outputs;
    const range = mir.operands(inst).?;
    var at = range.first;
    for (0..range.count - 1) |_| at += @as(Mir.Operand, @bitCast(mir.extra[at])).len();
    const result_start = range.first + 1;
    const first: Mir.Operand = @bitCast(mir.extra[result_start]);
    if (first.kind == .list) return at + 1 + (result_at - (result_start + 1));
    return at;
}

const Copy = struct { dst: u32, src: u32 };

/// Orders the copies. Each copy runs after every read of its destination.
fn parallelCopy(lower: *Lower, b: *Mir.Builder, copies: []Copy) Allocator.Error!void {
    var pending = copies;
    while (pending.len > 0) {
        const ready = for (pending, 0..) |copy, k| {
            const blocked = for (pending) |other| {
                if (other.src == copy.dst) break true;
            } else false;
            if (!blocked) break k;
        } else null;
        if (ready) |k| {
            try addMov(b, pending[k].dst, pending[k].src);
            pending[k] = pending[pending.len - 1];
            pending.len -= 1;
            continue;
        }
        const src = pending[0].src;
        const local = lower.temporary(@as(Mir.Operand, @bitCast(src)).regClass());
        try addMov(b, local, src);
        for (pending) |*copy| {
            if (copy.src == src) copy.src = local;
        }
    }
}

fn addMov(b: *Mir.Builder, dst: u32, src: u32) Allocator.Error!void {
    const ty: Mir.Type = switch (@as(Mir.Operand, @bitCast(dst)).regClass()) {
        .pred => .pred,
        .b16 => .b16,
        .b32 => .b32,
        .b64 => .b64,
        .f32 => .f32,
        .f64 => .f64,
    };
    try b.addTyped(.mov, ty, .{ .none = .{} }, &.{ dst, src });
}

/// Retargets each branch past the blocks that hold only an unguarded branch, and then removes
/// branches to the next instruction, unused labels and unreachable instructions.
fn tidy(arena: Allocator, b: *Mir.Builder) Allocator.Error!void {
    const tags = b.instructions.items(.tag);
    const datas = b.instructions.items(.data);
    const modifiers = b.instructions.items(.modifiers);
    const extra = b.extra.items;
    const count = tags.len;
    const label_pos = try arena.alloc(u32, b.label_count);
    @memset(label_pos, none);
    for (tags, datas, 0..) |tag, data, i| {
        if (tag == .label) label_pos[@as(Mir.Operand, @bitCast(extra[data])).payload] = @intCast(i);
    }
    const targetAt = struct {
        fn at(tag: Mir.Inst.Tag, inst_modifiers: Mir.Inst.Modifiers, data: u32) u32 {
            _ = tag;
            return data + @intFromBool(inst_modifiers.guard() != .none);
        }
    }.at;
    // `forward[l]` is the final target of a branch to label `l`.
    const forward = try arena.alloc(u32, b.label_count);
    for (forward, 0..) |*f, label| {
        var current: u32 = @intCast(label);
        var steps: u32 = 0;
        while (steps < b.label_count) : (steps += 1) {
            const pos = label_pos[current];
            if (pos == none or pos + 1 >= count) break;
            if (tags[pos + 1] != .bra or modifiers[pos + 1].guard() != .none) break;
            const next = @as(Mir.Operand, @bitCast(extra[datas[pos + 1]])).payload;
            if (next == current) break;
            current = next;
        }
        f.* = current;
    }
    for (tags, datas, modifiers) |tag, data, inst_modifiers| {
        if (tag != .bra) continue;
        const at = targetAt(tag, inst_modifiers, data);
        const op: Mir.Operand = @bitCast(extra[at]);
        extra[at] = @bitCast(Mir.Operand{ .kind = .label, .payload = @intCast(forward[op.payload]) });
    }

    var keep: std.DynamicBitSetUnmanaged = try .initFull(arena, count);
    const refs = try arena.alloc(u32, b.label_count);
    var changed = true;
    while (changed) {
        changed = false;
        @memset(refs, 0);
        for (tags, datas, modifiers, 0..) |tag, data, inst_modifiers, i| {
            if (tag == .bra and keep.isSet(i)) refs[@as(Mir.Operand, @bitCast(extra[targetAt(tag, inst_modifiers, data)])).payload] += 1;
        }
        var reachable = true;
        var i: usize = 0;
        while (i < count) : (i += 1) {
            if (!keep.isSet(i)) continue;
            const tag = tags[i];
            if (tag == .label) {
                if (refs[@as(Mir.Operand, @bitCast(extra[datas[i]])).payload] > 0) {
                    reachable = true;
                } else {
                    keep.unset(i);
                    changed = true;
                }
                continue;
            }
            if (!reachable) {
                keep.unset(i);
                changed = true;
                continue;
            }
            if (tag == .ret or tag == .trap or (tag == .bra and modifiers[i].guard() == .none)) reachable = false;
            if (tag != .bra) continue;
            // A branch to the next kept instruction has no effect.
            var next = i + 1;
            while (next < count and !keep.isSet(next)) next += 1;
            const target = @as(Mir.Operand, @bitCast(extra[targetAt(tag, modifiers[i], datas[i])])).payload;
            if (next < count and tags[next] == .label and @as(Mir.Operand, @bitCast(extra[datas[next]])).payload == target) {
                keep.unset(i);
                changed = true;
                reachable = true;
            }
        }
    }
    var out: usize = 0;
    for (0..count) |i| {
        if (!keep.isSet(i)) continue;
        b.instructions.set(out, b.instructions.get(i));
        out += 1;
    }
    b.instructions.shrinkRetainingCapacity(out);
}
