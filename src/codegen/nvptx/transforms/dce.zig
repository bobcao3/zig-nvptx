//! This file removes dead code. An instruction is dead when `Mir.isRemovable` allows its removal
//! and no kept instruction uses its results. A block parameter is dead when no kept instruction
//! uses the parameter.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Mir = @import("../Mir.zig");
const Cfg = @import("../Cfg.zig");

const none = Cfg.none;

pub fn run(gpa: Allocator, mir: *Mir) Allocator.Error!void {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const cfg = try Cfg.init(arena, mir);
    const count = mir.instructions.len;

    // `def_inst[r]` is the instruction that defines register `r`. `param_index[r]` is the position
    // of the block parameter `r` in its label.
    const def_inst = try arena.alloc(u32, cfg.reg_count);
    const param_index = try arena.alloc(u32, cfg.reg_count);
    @memset(def_inst, none);
    @memset(param_index, none);
    const branches = try arena.alloc(std.ArrayList(u32), mir.label_count);
    @memset(branches, .empty);
    var refs: Refs = .{ .mir = mir };
    for (0..count) |i| {
        const inst = mir.instructions.get(i);
        if (inst.tag == .label) {
            for (mir.blockParams(inst), 0..) |param, k| {
                def_inst[cfg.regId(@bitCast(param))] = @intCast(i);
                param_index[cfg.regId(@bitCast(param))] = @intCast(k);
            }
            continue;
        }
        if (inst.tag == .bra and inst.modifiers.guard() == .none) try branches[mir.branchTarget(inst)].append(arena, @intCast(i));
        try refs.collect(arena, inst);
        for (refs.list.items) |ref| {
            if (ref.role == .def) def_inst[cfg.regId(@bitCast(ref.word))] = @intCast(i);
        }
    }

    var live_inst: std.DynamicBitSetUnmanaged = try .initEmpty(arena, count);
    var live_reg: std.DynamicBitSetUnmanaged = try .initEmpty(arena, cfg.reg_count);
    var work: std.ArrayList(u32) = .empty;
    for (0..count) |i| {
        const inst = mir.instructions.get(i);
        if (mir.isRemovable(inst) or inst.tag == .label) continue;
        live_inst.set(i);
        try pushUses(arena, &cfg, &refs, inst, &work);
    }
    while (work.pop()) |reg| {
        if (live_reg.isSet(reg)) continue;
        live_reg.set(reg);
        const i = def_inst[reg];
        if (i == none) continue;
        const inst = mir.instructions.get(i);
        if (inst.tag == .label) {
            const k = param_index[reg];
            for (branches[mir.labelOf(inst)].items) |bra| {
                try work.append(arena, cfg.regId(@bitCast(mir.branchArgs(mir.instructions.get(bra))[k])));
            }
            continue;
        }
        if (live_inst.isSet(i)) continue;
        live_inst.set(i);
        try pushUses(arena, &cfg, &refs, inst, &work);
    }

    var b: Mir.Builder = .{ .gpa = gpa, .reg_counts = mir.reg_counts, .label_count = mir.label_count };
    defer b.deinit();
    var kept: std.ArrayList(u32) = .empty;
    for (0..count) |i| {
        const inst = mir.instructions.get(i);
        switch (inst.tag) {
            .label => {
                kept.clearRetainingCapacity();
                for (mir.blockParams(inst)) |param| {
                    if (live_reg.isSet(cfg.regId(@bitCast(param)))) try kept.append(arena, param);
                }
                try b.addLabel(mir.labelOf(inst), kept.items);
            },
            .bra => if (inst.modifiers.guard() == .none) {
                const target = mir.branchTarget(inst);
                const target_inst = mir.instructions.get(cfg.starts[cfg.label_blocks[target]]);
                kept.clearRetainingCapacity();
                for (mir.blockParams(target_inst), mir.branchArgs(inst)) |param, arg| {
                    if (live_reg.isSet(cfg.regId(@bitCast(param)))) try kept.append(arena, arg);
                }
                try b.addBranch(target, kept.items);
            } else try copyInst(&b, mir, inst),
            else => if (live_inst.isSet(i)) try copyInst(&b, mir, inst),
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
}

/// Pushes each input register of `inst`. A branch argument waits until its parameter is live.
fn pushUses(arena: Allocator, cfg: *const Cfg, refs: *Refs, inst: Mir.Inst, work: *std.ArrayList(u32)) Allocator.Error!void {
    if (inst.tag == .bra and inst.modifiers.guard() == .none) return;
    try refs.collect(arena, inst);
    for (refs.list.items) |ref| {
        if (ref.role == .use) try work.append(arena, cfg.regId(@bitCast(ref.word)));
    }
}

fn copyInst(b: *Mir.Builder, mir: *const Mir, inst: Mir.Inst) Allocator.Error!void {
    const at = b.here();
    try b.extra.appendSlice(b.gpa, mir.span(inst));
    try b.instructions.append(b.gpa, .{ .tag = inst.tag, .ty = inst.ty, .modifiers = inst.modifiers, .data = at });
}

const Refs = struct {
    mir: *const Mir,
    list: std.ArrayList(Ref) = .empty,

    const Ref = struct { word: u32, role: Mir.RegRole };

    fn collect(refs: *Refs, arena: Allocator, inst: Mir.Inst) Allocator.Error!void {
        refs.list.clearRetainingCapacity();
        try refs.list.ensureTotalCapacity(arena, refs.mir.span(inst).len);
        refs.mir.visitRegs(inst, refs);
    }

    pub fn reg(refs: *Refs, at: u32, role: Mir.RegRole) void {
        refs.list.appendAssumeCapacity(.{ .word = refs.mir.extra[at], .role = role });
    }
};
