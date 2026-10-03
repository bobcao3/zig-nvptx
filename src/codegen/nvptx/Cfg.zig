//! This file builds the control flow graph of a MIR function, its dominator tree and its live
//! registers. A block starts at each `label` and after each `bra`, `ret` or `trap`. When the first
//! instruction is a label, the entry block is empty. No branch reaches the entry block.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Mir = @import("Mir.zig");
const Cfg = @This();

pub const none = std.math.maxInt(u32);

/// `starts[b]` is the first instruction of block `b`. The last entry is the instruction count.
starts: []u32,
/// `[0]` is the branch target or the next block. `[1]` is the next block after a guarded `bra`.
successors: [][2]u32,
predecessors: []const []const u32,
label_blocks: []u32,
/// `order` lists the reachable blocks in reverse postorder.
order: []u32,
/// The entry block and the unreachable blocks have no immediate dominator (`none`).
idom: []u32,
/// `rank[b]` is the position of block `b` in `order`. Unreachable blocks have `none`.
rank: []u32,
reg_bases: [Mir.class_count]u32,
reg_count: u32,

pub fn init(arena: Allocator, mir: *const Mir) Allocator.Error!Cfg {
    const tags = mir.instructions.items(.tag);
    var starts: std.ArrayList(u32) = .empty;
    try starts.append(arena, 0);
    for (tags, 0..) |tag, i| {
        const index: u32 = @intCast(i);
        switch (tag) {
            .label => if (index == 0 or starts.items[starts.items.len - 1] != index) try starts.append(arena, index),
            .bra, .ret, .trap => if (index + 1 < tags.len) try starts.append(arena, index + 1),
            else => {},
        }
    }
    const block_count = starts.items.len;
    try starts.append(arena, @intCast(tags.len));

    const label_blocks = try arena.alloc(u32, mir.label_count);
    @memset(label_blocks, none);
    for (starts.items[0..block_count], 0..) |start, b| {
        if (start < starts.items[b + 1] and tags[start] == .label) {
            label_blocks[mir.labelOf(mir.instructions.get(start))] = @intCast(b);
        }
    }

    const successors = try arena.alloc([2]u32, block_count);
    const pred_counts = try arena.alloc(u32, block_count);
    @memset(pred_counts, 0);
    for (successors, 0..) |*succ, b| {
        succ.* = .{ none, none };
        const next: u32 = if (b + 1 < block_count) @intCast(b + 1) else none;
        const end = starts.items[b + 1];
        if (end == starts.items[b]) {
            succ[0] = next;
        } else {
            const last = mir.instructions.get(end - 1);
            switch (last.tag) {
                .ret, .trap => {},
                .bra => {
                    succ[0] = label_blocks[mir.branchTarget(last)];
                    if (last.modifiers.guard() != .none) succ[1] = next;
                },
                else => succ[0] = next,
            }
        }
        for (succ) |s| if (s != none) {
            pred_counts[s] += 1;
        };
    }
    const predecessors = try arena.alloc([]u32, block_count);
    for (predecessors, pred_counts) |*preds, count| {
        preds.* = try arena.alloc(u32, count);
        preds.len = 0;
    }
    for (successors, 0..) |succ, b| for (succ) |s| if (s != none) {
        predecessors[s].len += 1;
        predecessors[s][predecessors[s].len - 1] = @intCast(b);
    };

    var cfg: Cfg = .{
        .starts = starts.items,
        .successors = successors,
        .predecessors = predecessors,
        .label_blocks = label_blocks,
        .order = &.{},
        .idom = try arena.alloc(u32, block_count),
        .rank = try arena.alloc(u32, block_count),
        .reg_bases = undefined,
        .reg_count = 0,
    };
    try cfg.computeOrder(arena);
    cfg.computeDominators();
    for (&cfg.reg_bases, mir.reg_counts) |*base, count| {
        base.* = cfg.reg_count;
        cfg.reg_count += count;
    }
    return cfg;
}

pub fn blockCount(cfg: *const Cfg) u32 {
    return @intCast(cfg.successors.len);
}

fn isReachable(cfg: *const Cfg, block: u32) bool {
    return cfg.rank[block] != none;
}

/// Numbers the registers of all classes in one range.
pub fn regId(cfg: *const Cfg, op: Mir.Operand) u32 {
    return cfg.reg_bases[@backingInt(op.regClass())] + op.regIndex();
}

fn computeOrder(cfg: *Cfg, arena: Allocator) Allocator.Error!void {
    const count = cfg.blockCount();
    var postorder: std.ArrayList(u32) = try .initCapacity(arena, count);
    const visited = try arena.alloc(bool, count);
    @memset(visited, false);
    const Frame = struct { block: u32, next: u2 };
    var stack: std.ArrayList(Frame) = .empty;
    try stack.append(arena, .{ .block = 0, .next = 0 });
    visited[0] = true;
    while (stack.items.len > 0) {
        const top = &stack.items[stack.items.len - 1];
        if (top.next == 2) {
            postorder.appendAssumeCapacity(top.block);
            _ = stack.pop();
            continue;
        }
        const s = cfg.successors[top.block][top.next];
        top.next += 1;
        if (s == none or visited[s]) continue;
        visited[s] = true;
        try stack.append(arena, .{ .block = s, .next = 0 });
    }
    std.mem.reverse(u32, postorder.items);
    cfg.order = postorder.items;
    @memset(cfg.rank, none);
    for (cfg.order, 0..) |block, i| cfg.rank[block] = @intCast(i);
}

/// Uses the dominance algorithm of Cooper, Harvey and Kennedy (2001).
fn computeDominators(cfg: *Cfg) void {
    @memset(cfg.idom, none);
    cfg.idom[0] = 0;
    var changed = true;
    while (changed) {
        changed = false;
        for (cfg.order[1..]) |block| {
            var new: u32 = none;
            for (cfg.predecessors[block]) |pred| {
                if (cfg.idom[pred] == none) continue;
                new = if (new == none) pred else cfg.intersect(pred, new);
            }
            if (new != cfg.idom[block]) {
                cfg.idom[block] = new;
                changed = true;
            }
        }
    }
    cfg.idom[0] = none;
}

fn intersect(cfg: *const Cfg, a_start: u32, b_start: u32) u32 {
    var a = a_start;
    var b = b_start;
    while (a != b) {
        while (cfg.rank[a] > cfg.rank[b]) a = cfg.idom[a];
        while (cfg.rank[b] > cfg.rank[a]) b = cfg.idom[b];
    }
    return a;
}

/// Returns whether block `a` dominates block `b`. Assumes that both blocks are reachable.
pub fn dominates(cfg: *const Cfg, a: u32, b: u32) bool {
    var block = b;
    while (true) {
        if (block == a) return true;
        if (block == 0) return false;
        block = cfg.idom[block];
    }
}

/// A register without a definition is never live.
pub const Liveness = struct {
    live_in: []std.DynamicBitSetUnmanaged,
    live_out: []std.DynamicBitSetUnmanaged,
};

pub fn liveness(cfg: *const Cfg, arena: Allocator, mir: *const Mir) Allocator.Error!Liveness {
    const count = cfg.blockCount();
    var defined: std.DynamicBitSetUnmanaged = try .initEmpty(arena, cfg.reg_count);
    const uses = try arena.alloc(std.DynamicBitSetUnmanaged, count);
    const defs = try arena.alloc(std.DynamicBitSetUnmanaged, count);
    const Visitor = struct {
        cfg: *const Cfg,
        mir: *const Mir,
        uses: *std.DynamicBitSetUnmanaged,
        defs: *std.DynamicBitSetUnmanaged,
        defined: *std.DynamicBitSetUnmanaged,

        pub fn reg(v: @This(), at: u32, role: Mir.RegRole) void {
            const id = v.cfg.regId(@bitCast(v.mir.extra[at]));
            switch (role) {
                .use => if (!v.defs.isSet(id)) v.uses.set(id),
                .def => {
                    v.defs.set(id);
                    v.defined.set(id);
                },
                .def_ref => {},
            }
        }
    };
    for (uses, defs, 0..) |*u, *d, b| {
        u.* = try .initEmpty(arena, cfg.reg_count);
        d.* = try .initEmpty(arena, cfg.reg_count);
        if (!cfg.isReachable(@intCast(b))) continue;
        for (cfg.starts[b]..cfg.starts[b + 1]) |i| mir.visitRegs(mir.instructions.get(i), Visitor{
            .cfg = cfg,
            .mir = mir,
            .uses = u,
            .defs = d,
            .defined = &defined,
        });
    }
    const live_in = try arena.alloc(std.DynamicBitSetUnmanaged, count);
    const live_out = try arena.alloc(std.DynamicBitSetUnmanaged, count);
    for (live_in, live_out) |*in, *out| {
        in.* = try .initEmpty(arena, cfg.reg_count);
        out.* = try .initEmpty(arena, cfg.reg_count);
    }
    const words = (cfg.reg_count + @bitSizeOf(usize) - 1) / @bitSizeOf(usize);
    var changed = true;
    while (changed) {
        changed = false;
        var i = cfg.order.len;
        while (i > 0) {
            i -= 1;
            const b = cfg.order[i];
            for (cfg.successors[b]) |s| if (s != none) live_out[b].setUnion(live_in[s]);
            for (0..words) |w| {
                const in = ((live_out[b].masks[w] & ~defs[b].masks[w]) | uses[b].masks[w]) & defined.masks[w];
                if (in == live_in[b].masks[w]) continue;
                live_in[b].masks[w] = in;
                changed = true;
            }
        }
    }
    return .{ .live_in = live_in, .live_out = live_out };
}
