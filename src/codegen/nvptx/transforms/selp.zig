//! This file turns a guarded `bra` whose arms only choose the arguments of one block into `selp`,
//! as `SimplifyCFG.cpp` of LLVM folds a two-entry PHI node. Each arm has no other predecessor, and
//! the arms hold at most `max_speculated` instructions that `canSpeculate` allows. When the arms
//! were the only predecessors of the joined block, the block moves into the block of the branch.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Mir = @import("../Mir.zig");
const Cfg = @import("../Cfg.zig");

const none = Cfg.none;

/// The default of `two-entry-phi-node-folding-threshold` in LLVM.
const max_speculated = 4;

pub fn run(gpa: Allocator, mir: *Mir) Allocator.Error!void {
    while (try foldOnce(gpa, mir)) {}
}

const Fold = struct {
    arms: [2]u32,
    join: u32,
    merges: bool,
    values: []u32,
    selects: []Select,
};

const Select = struct {
    result: u32,
    pred: u32,
    when_set: u32,
    when_clear: u32,
    temps: [3]u32,
};

const Renames = std.AutoHashMapUnmanaged(u32, u32);

fn foldOnce(gpa: Allocator, mir: *Mir) Allocator.Error!bool {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const cfg = try Cfg.init(arena, mir);
    const block_count = cfg.blockCount();
    const folds = try arena.alloc(?Fold, block_count);
    @memset(folds, null);
    var skipped: std.DynamicBitSetUnmanaged = try .initEmpty(arena, block_count);
    var reg_counts = mir.reg_counts;
    var renames: Renames = .empty;
    var changed = false;
    for (folds, 0..) |*fold, head| {
        const arms = foldArms(mir, &cfg, @intCast(head)) orelse continue;
        const guard = mir.instructions.get(cfg.starts[head + 1] - 1);
        const exits = [2]Mir.Inst{
            mir.instructions.get(cfg.starts[arms[0] + 1] - 1),
            mir.instructions.get(cfg.starts[arms[1] + 1] - 1),
        };
        const fall_args = mir.branchArgs(exits[0]);
        const values = try arena.alloc(u32, fall_args.len);
        var selects: std.ArrayList(Select) = .empty;
        for (values, fall_args, mir.branchArgs(exits[1])) |*value, fall_arg, taken_arg| {
            if (fall_arg == taken_arg) {
                value.* = fall_arg;
                continue;
            }
            const class = @as(Mir.Operand, @bitCast(fall_arg)).regClass();
            const when_set, const when_clear = switch (guard.modifiers.guard()) {
                .positive => .{ taken_arg, fall_arg },
                .negative => .{ fall_arg, taken_arg },
                .none => unreachable,
            };
            var temps: [3]u32 = @splat(none);
            if (class == .pred) {
                for (&temps) |*temp| temp.* = newReg(&reg_counts, .pred);
            }
            value.* = newReg(&reg_counts, class);
            try selects.append(arena, .{
                .result = value.*,
                .pred = mir.extra[guard.data],
                .when_set = when_set,
                .when_clear = when_clear,
                .temps = temps,
            });
        }
        const join = cfg.label_blocks[mir.branchTarget(exits[0])];
        const merges = join != head and cfg.predecessors[join].len == 2 and endsControl(mir, &cfg, join);
        if (merges) {
            const params = mir.blockParams(mir.instructions.get(cfg.starts[join]));
            for (params, values) |param, value| try renames.put(arena, param, value);
            skipped.set(join);
        }
        for (arms) |arm| skipped.set(arm);
        fold.* = .{ .arms = arms, .join = join, .merges = merges, .values = values, .selects = selects.items };
        changed = true;
    }
    if (!changed) return false;

    var b: Mir.Builder = .{ .gpa = gpa, .reg_counts = reg_counts, .label_count = mir.label_count };
    defer b.deinit();
    var args: std.ArrayList(u32) = .empty;
    for (folds, 0..) |maybe_fold, block| {
        if (skipped.isSet(block)) continue;
        const start = cfg.starts[block];
        const end = cfg.starts[block + 1];
        const fold = maybe_fold orelse {
            for (start..end) |i| try copyInst(&b, mir, &renames, mir.instructions.get(i));
            continue;
        };
        for (start..end - 1) |i| try copyInst(&b, mir, &renames, mir.instructions.get(i));
        for (fold.arms) |arm| {
            for (cfg.starts[arm]..cfg.starts[arm + 1] - 1) |i| {
                const inst = mir.instructions.get(i);
                if (inst.tag != .label) try copyInst(&b, mir, &renames, inst);
            }
        }
        for (fold.selects) |select| try emitSelect(&b, &renames, select);
        if (fold.merges) {
            for (cfg.starts[fold.join] + 1..cfg.starts[fold.join + 1]) |i| {
                try copyInst(&b, mir, &renames, mir.instructions.get(i));
            }
            continue;
        }
        args.clearRetainingCapacity();
        for (fold.values) |value| try args.append(arena, resolve(&renames, value));
        try b.addBranch(mir.labelOf(mir.instructions.get(cfg.starts[fold.join])), args.items);
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
    return true;
}

fn foldArms(mir: *const Mir, cfg: *const Cfg, head: u32) ?[2]u32 {
    const end = cfg.starts[head + 1];
    if (end == cfg.starts[head]) return null;
    const last = mir.instructions.get(end - 1);
    if (last.tag != .bra or last.modifiers.guard() == .none) return null;
    const arms: [2]u32 = .{ cfg.successors[head][1], cfg.successors[head][0] };
    if (arms[0] == none or arms[1] == none or arms[0] == arms[1]) return null;
    var join: u32 = none;
    var speculated: u32 = 0;
    for (arms) |arm| {
        if (arm == head or cfg.predecessors[arm].len != 1) return null;
        const arm_start = cfg.starts[arm];
        const arm_end = cfg.starts[arm + 1];
        if (arm_end == arm_start) return null;
        const exit = mir.instructions.get(arm_end - 1);
        if (exit.tag != .bra or exit.modifiers.guard() != .none) return null;
        const target = cfg.label_blocks[mir.branchTarget(exit)];
        if (join != none and join != target) return null;
        join = target;
        for (arm_start..arm_end - 1) |i| {
            const inst = mir.instructions.get(i);
            if (inst.tag == .label) continue;
            if (!canSpeculate(mir, inst)) return null;
            speculated += 1;
        }
    }
    if (join == arms[0] or join == arms[1] or speculated > max_speculated) return null;
    if (mir.blockParams(mir.instructions.get(cfg.starts[join])).len == 0) return null;
    return arms;
}

/// Integer `div` and `rem` by zero give an unspecified value and do not trap (PTX ISA 9.4, sections
/// 9.7.1.9 and 9.7.1.10). The pass does not see what inline assembly does.
fn canSpeculate(mir: *const Mir, inst: Mir.Inst) bool {
    return switch (inst.tag) {
        .@"asm" => false,
        else => !Mir.mayLoad(inst) and mir.isRemovable(inst),
    };
}

fn endsControl(mir: *const Mir, cfg: *const Cfg, block: u32) bool {
    const last = mir.instructions.get(cfg.starts[block + 1] - 1);
    return switch (last.tag) {
        .ret, .trap => true,
        .bra => last.modifiers.guard() == .none,
        else => false,
    };
}

/// PTX has no `selp.pred` (PTX ISA 9.4, section 9.7.7.3), so a predicate takes logic instructions.
fn emitSelect(b: *Mir.Builder, renames: *const Renames, select: Select) Allocator.Error!void {
    const pred = resolve(renames, select.pred);
    const when_set = resolve(renames, select.when_set);
    const when_clear = resolve(renames, select.when_clear);
    const class = @as(Mir.Operand, @bitCast(select.result)).regClass();
    if (class != .pred) {
        try b.addTyped(.selp, class.bitType(), .{ .none = .{} }, &.{ select.result, when_set, when_clear, pred });
        return;
    }
    const set_part, const not_pred, const clear_part = select.temps;
    try b.addTyped(.@"and", .pred, .{ .none = .{} }, &.{ set_part, pred, when_set });
    try b.addTyped(.not, .pred, .{ .none = .{} }, &.{ not_pred, pred });
    try b.addTyped(.@"and", .pred, .{ .none = .{} }, &.{ clear_part, not_pred, when_clear });
    try b.addTyped(.@"or", .pred, .{ .none = .{} }, &.{ select.result, set_part, clear_part });
}

fn newReg(reg_counts: *[Mir.class_count]u32, class: Mir.Class) u32 {
    const index = reg_counts[@backingInt(class)];
    reg_counts[@backingInt(class)] += 1;
    return @bitCast(Mir.Operand.reg(class, @intCast(index)));
}

fn resolve(renames: *const Renames, word: u32) u32 {
    var result = word;
    while (renames.get(result)) |next| result = next;
    return result;
}

fn copyInst(b: *Mir.Builder, mir: *const Mir, renames: *const Renames, inst: Mir.Inst) Allocator.Error!void {
    const at = b.here();
    try b.extra.appendSlice(b.gpa, mir.span(inst));
    if (renames.count() != 0) {
        var patch: Patch = .{ .extra = b.extra.items, .shift = at -% inst.data, .renames = renames };
        mir.visitRegs(inst, &patch);
    }
    try b.instructions.append(b.gpa, .{ .tag = inst.tag, .ty = inst.ty, .modifiers = inst.modifiers, .data = at });
}

const Patch = struct {
    extra: []u32,
    shift: u32,
    renames: *const Renames,

    pub fn reg(patch: *Patch, at: u32, role: Mir.RegRole) void {
        if (role != .use) return;
        const word = &patch.extra[at +% patch.shift];
        word.* = resolve(patch.renames, word.*);
    }
};
