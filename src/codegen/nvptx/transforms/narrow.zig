//! A 64-bit register fits when its values stay below 2^32. The counter of a loop over a `usize`
//! range is 64-bit, so each iteration costs a carry add and a second compare instruction. This pass
//! gives a counter that fits a 32-bit register. `Solver` puts all 64-bit registers on a worklist,
//! like the SCCP solver of LLVM, and removes each register whose definition does not show that it
//! fits. `visit` lists those definitions. `Rewriter` gives each block parameter that fits a reduced
//! register, a 32-bit register with the same value. It gives one to an immediate move, or to an
//! increment of `v` when `v` gets one, only when a compare, a truncation, a branch, a copy or
//! another increment reads the result. It compares registers that fit in 32 bits and folds an
//! unsigned compare with an immediate of at least 2^32 - 1 when all values below 2^32 give one
//! result. It removes a truncation when a reduced register holds its value, and a zero extension
//! from 32 bits that undoes a truncation of a register that fits. The reduced registers follow
//! `TruncInstCombine` of LLVM, the compares follow `foldICmpWithZextOrSext`, and the removals
//! follow `isEliminableCastPair`.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Mir = @import("../Mir.zig");
const Cfg = @import("../Cfg.zig");

const none = Cfg.none;
const max_u32: u64 = std.math.maxInt(u32);

pub fn run(gpa: Allocator, mir: *Mir) Allocator.Error!void {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const cfg = try Cfg.init(arena, mir);
    var solver: Solver = try .init(arena, mir, &cfg);
    try solver.solve();
    var rewriter: Rewriter = try .init(&solver);
    try rewriter.planReplacements();
    if (!rewriter.changes) return;
    if (rewriter.next_b32 + rewriter.truncationBound() > std.math.maxInt(u16)) return;
    try rewriter.rewrite(gpa, mir);
}

const Input = struct {
    word: u32,
    imm: ?u64,

    fn reg(input: Input) ?Mir.Operand {
        if (input.imm != null) return null;
        const decoded: Mir.Operand = @bitCast(input.word);
        return if (decoded.kind == .reg) decoded else null;
    }
};

const Solver = struct {
    arena: Allocator,
    mir: *const Mir,
    cfg: *const Cfg,
    def: []u32,
    param_index: []u32,
    block_of: []u32,
    branches: []std.ArrayList(u32),
    /// `users[r]` lists the registers whose rules read whether register `r` fits.
    users: []std.ArrayList(u32),
    fits: std.DynamicBitSetUnmanaged,

    fn init(arena: Allocator, mir: *const Mir, cfg: *const Cfg) Allocator.Error!Solver {
        const count = mir.instructions.len;
        var s: Solver = .{
            .arena = arena,
            .mir = mir,
            .cfg = cfg,
            .def = try arena.alloc(u32, cfg.reg_count),
            .param_index = try arena.alloc(u32, cfg.reg_count),
            .block_of = try arena.alloc(u32, count),
            .branches = try arena.alloc(std.ArrayList(u32), mir.label_count),
            .users = try arena.alloc(std.ArrayList(u32), cfg.reg_count),
            .fits = try .initEmpty(arena, cfg.reg_count),
        };
        @memset(s.def, none);
        @memset(s.param_index, none);
        @memset(s.branches, .empty);
        @memset(s.users, .empty);
        for (0..cfg.blockCount()) |b| {
            for (cfg.starts[b]..cfg.starts[b + 1]) |i| s.block_of[i] = @intCast(b);
        }
        var defs: Defs = .{ .s = &s, .index = undefined };
        for (0..count) |i| {
            const instruction = mir.instructions.get(i);
            if (instruction.tag == .label) {
                for (mir.blockParams(instruction), 0..) |param, k| {
                    s.def[s.id(param)] = @intCast(i);
                    s.param_index[s.id(param)] = @intCast(k);
                }
            } else if (instruction.tag == .bra and instruction.modifiers.guard() == .none) {
                try s.branches[mir.branchTarget(instruction)].append(arena, @intCast(i));
            } else {
                defs.index = @intCast(i);
                mir.visitRegs(instruction, &defs);
            }
        }
        const base = cfg.reg_bases[@backingInt(Mir.Class.b64)];
        for (base..base + mir.reg_counts[@backingInt(Mir.Class.b64)]) |r| try s.addUsers(@intCast(r));
        return s;
    }

    const Defs = struct {
        s: *Solver,
        index: u32,

        pub fn reg(defs: *Defs, at: u32, role: Mir.RegRole) void {
            if (role == .def) defs.s.def[defs.s.id(defs.s.mir.extra[at])] = defs.index;
        }
    };

    fn id(s: *const Solver, word: u32) u32 {
        return s.cfg.regId(@bitCast(word));
    }

    fn instAt(s: *const Solver, i: u32) Mir.Inst {
        return s.mir.instructions.get(i);
    }

    fn operand(s: *const Solver, instruction: Mir.Inst, k: u32) Input {
        const range = s.mir.operands(instruction).?;
        var at = range.first + @intFromBool(instruction.modifiers.guard() != .none);
        for (0..k) |_| at += @as(Mir.Operand, @bitCast(s.mir.extra[at])).len();
        const op: Mir.Operand = @bitCast(s.mir.extra[at]);
        return .{ .word = s.mir.extra[at], .imm = switch (op.kind) {
            .imm => op.payload,
            .imm64 => switch (@as(Mir.ImmFormat, @fromBackingInt(@intCast(op.payload)))) {
                .unsigned, .signed => s.mir.extra[at + 1] | (@as(u64, s.mir.extra[at + 2]) << 32),
                else => null,
            },
            else => null,
        } };
    }

    fn inputFits(s: *const Solver, input: Input) bool {
        if (input.imm) |imm| return imm <= max_u32;
        const reg = input.reg() orelse return false;
        return reg.regClass() == .b64 and s.fits.isSet(s.cfg.regId(reg));
    }

    fn solve(s: *Solver) Allocator.Error!void {
        const base = s.cfg.reg_bases[@backingInt(Mir.Class.b64)];
        const count = s.mir.reg_counts[@backingInt(Mir.Class.b64)];
        var worklist: std.ArrayList(u32) = .empty;
        for (base..base + count) |r| {
            s.fits.set(r);
            try worklist.append(s.arena, @intCast(r));
        }
        while (worklist.pop()) |r| {
            if (!s.fits.isSet(r) or s.visit(r)) continue;
            s.fits.unset(r);
            try worklist.appendSlice(s.arena, s.users[r].items);
        }
    }

    fn addUsers(s: *Solver, r: u32) Allocator.Error!void {
        const i = s.def[r];
        if (i == none) return;
        const instruction = s.instAt(i);
        if (instruction.tag == .label) {
            for (s.branches[s.mir.labelOf(instruction)].items) |bra| {
                try s.addUser(s.mir.branchArgs(s.instAt(bra))[s.param_index[r]], r);
            }
            return;
        }
        if (instruction.modifiers.guard() != .none) return;
        switch (instruction.tag) {
            .mov => try s.addUser(s.operand(instruction, 1).word, r),
            .add => if (s.incrementSource(i)) |v| {
                var t = s.block_of[i];
                while (t != none) : (t = s.cfg.idom[t]) {
                    if (s.cfg.predecessors[t].len == 1) {
                        if (s.edgeTest(s.cfg.predecessors[t][0], t, v)) |bound| try s.addUser(bound.word, r);
                    }
                    if (t == 0) break;
                }
            },
            else => {},
        }
    }

    fn addUser(s: *Solver, word: u32, user: u32) Allocator.Error!void {
        const op: Mir.Operand = @bitCast(word);
        if (op.kind != .reg or op.regClass() != .b64) return;
        try s.users[s.id(word)].append(s.arena, user);
    }

    fn visit(s: *const Solver, r: u32) bool {
        const i = s.def[r];
        if (i == none) return false;
        const instruction = s.instAt(i);
        if (instruction.tag == .label) return s.visitParam(instruction, s.param_index[r]);
        if (instruction.modifiers.guard() != .none) return false;
        return switch (instruction.tag) {
            .cvt => isZeroExtension(instruction),
            .mov => s.inputFits(s.operand(instruction, 1)),
            .@"and" => s.visitMask(instruction),
            .add => s.visitIncrement(i),
            else => false,
        };
    }

    fn visitParam(s: *const Solver, label: Mir.Inst, k: u32) bool {
        for (s.branches[s.mir.labelOf(label)].items) |bra| {
            if (!s.inputFits(.{ .word = s.mir.branchArgs(s.instAt(bra))[k], .imm = null })) return false;
        }
        return true;
    }

    fn visitMask(s: *const Solver, instruction: Mir.Inst) bool {
        for ([_]u32{ 1, 2 }) |k| {
            if (s.operand(instruction, k).imm) |imm| return imm <= max_u32;
        }
        return false;
    }

    fn visitIncrement(s: *const Solver, i: u32) bool {
        const v = s.incrementSource(i) orelse return false;
        return s.dominatingBound(i, v) != null;
    }

    /// Assumes a `cvt`. Returns whether it zero-extends to 64 bits (PTX ISA 9.4, section 6.5.1).
    fn isZeroExtension(instruction: Mir.Inst) bool {
        const cvt = instruction.modifiers.cvt;
        if (cvt.rounding != .none or cvt.sat) return false;
        return (instruction.ty == .u64 or instruction.ty == .s64) and switch (cvt.from) {
            .u8, .u16, .u32 => true,
            else => false,
        };
    }

    fn isTruncation(instruction: Mir.Inst) bool {
        const cvt = instruction.modifiers.cvt;
        if (instruction.tag != .cvt or instruction.modifiers.guard() != .none) return false;
        if (cvt.rounding != .none or cvt.sat) return false;
        const to_32 = instruction.ty == .u32 or instruction.ty == .s32;
        return to_32 and (cvt.from == .u64 or cvt.from == .s64);
    }

    fn incrementSource(s: *const Solver, i: u32) ?u32 {
        const instruction = s.instAt(i);
        const arith = instruction.modifiers.arith;
        if (instruction.ty != .u64 and instruction.ty != .s64) return null;
        if (arith.mode != .none or arith.cc or arith.sat) return null;
        const a = s.operand(instruction, 1);
        const b = s.operand(instruction, 2);
        if (b.imm == 1 and a.reg() != null) return a.word;
        if (a.imm == 1 and b.reg() != null) return b.word;
        return null;
    }

    /// Returns a bound `b` that fits, with `v < b` at `i`, so that `v + 1` stays below 2^32.
    fn dominatingBound(s: *const Solver, i: u32, v: u32) ?Input {
        var t = s.block_of[i];
        while (t != none) : (t = s.cfg.idom[t]) {
            if (s.cfg.predecessors[t].len == 1) {
                if (s.edgeTest(s.cfg.predecessors[t][0], t, v)) |bound| {
                    if (s.inputFits(bound)) return bound;
                }
            }
            if (t == 0) break;
        }
        return null;
    }

    /// Returns `b` if the branch at the end of `g` goes to `t` only when `v < b`, or `null`.
    fn edgeTest(s: *const Solver, g: u32, t: u32, v: u32) ?Input {
        const end = s.cfg.starts[g + 1];
        if (end == s.cfg.starts[g]) return null;
        const bra = s.instAt(end - 1);
        if (bra.tag != .bra) return null;
        const succ = s.cfg.successors[g];
        if (succ[0] == succ[1]) return null;
        const holds_edge = switch (bra.modifiers.guard()) {
            .none => return null,
            .positive => succ[0],
            .negative => succ[1],
        };
        if (holds_edge != t) return null;
        const predicate = s.mir.extra[s.mir.operands(bra).?.first];
        const setp_index = s.def[s.id(predicate)];
        if (setp_index == none or s.block_of[setp_index] != g) return null;
        const setp = s.instAt(setp_index);
        if (setp.tag != .setp or setp.modifiers.guard() != .none) return null;
        if (setp.modifiers.setp.bool_op != .none or setp.ty != .u64) return null;
        const a = s.operand(setp, 1);
        const b = s.operand(setp, 2);
        return switch (setp.modifiers.setp.cmp) {
            .lt, .lo => if (a.imm == null and a.word == v) b else null,
            .gt, .hi => if (b.imm == null and b.word == v) a else null,
            else => null,
        };
    }
};

const Insertion = struct {
    tag: Mir.Inst.Tag,
    result: u32,
    source: u32,
    imm: u64,
};

const Rewriter = struct {
    s: *const Solver,
    /// `reduced[r]` is a 32-bit register that holds the value of `r`, or `none`.
    reduced: []u32,
    reduced_done: std.DynamicBitSetUnmanaged,
    /// `replace[r]` is the register that takes the place of the removed register `r`, or `none`.
    replace: []u32,
    removed: std.DynamicBitSetUnmanaged,
    after: []?Insertion,
    next_b32: u32,
    changes: bool = false,

    fn init(s: *const Solver) Allocator.Error!Rewriter {
        const count = s.mir.instructions.len;
        const rewriter: Rewriter = .{
            .s = s,
            .reduced = try s.arena.alloc(u32, s.cfg.reg_count),
            .reduced_done = try .initEmpty(s.arena, s.cfg.reg_count),
            .replace = try s.arena.alloc(u32, s.cfg.reg_count),
            .removed = try .initEmpty(s.arena, count),
            .after = try s.arena.alloc(?Insertion, count),
            .next_b32 = s.mir.reg_counts[@backingInt(Mir.Class.b32)],
        };
        @memset(rewriter.reduced, none);
        @memset(rewriter.replace, none);
        @memset(rewriter.after, null);
        return rewriter;
    }

    /// Returns `none` when no register index is left, because an operand holds 16 bits of index.
    fn newReg(w: *Rewriter) u32 {
        if (w.next_b32 > std.math.maxInt(u16)) return none;
        defer w.next_b32 += 1;
        return @bitCast(Mir.Operand.reg(.b32, @intCast(w.next_b32)));
    }

    /// Returns a 32-bit register with the value of `word`, or `none` when `word` does not fit or
    /// only a truncation gives the value. Plans the copy of an increment or an immediate move.
    fn reducedOperand(w: *Rewriter, word: u32) u32 {
        const s = w.s;
        const r = s.id(word);
        if (w.reduced_done.isSet(r)) return w.reduced[r];
        w.reduced_done.set(r);
        if (!s.fits.isSet(r)) return none;
        const i = s.def[r];
        const instruction = s.instAt(i);
        const result: u32 = switch (instruction.tag) {
            .label => w.newReg(),
            .cvt => cvt: {
                const source = s.operand(instruction, 1);
                const reg = source.reg() orelse break :cvt none;
                break :cvt if (reg.regClass() == .b32) source.word else none;
            },
            .mov => mov: {
                const source = s.operand(instruction, 1);
                if (source.imm) |imm| {
                    const result = w.newReg();
                    if (result != none) {
                        w.after[i] = .{ .tag = .mov, .result = result, .source = none, .imm = imm };
                    }
                    break :mov result;
                }
                break :mov w.reducedOperand(source.word);
            },
            .add => add: {
                const v32 = w.reducedOperand(s.incrementSource(i).?);
                if (v32 == none) break :add none;
                const result = w.newReg();
                if (result != none) {
                    w.after[i] = .{ .tag = .add, .result = result, .source = v32, .imm = 1 };
                }
                break :add result;
            },
            else => none,
        };
        w.reduced[r] = result;
        return result;
    }

    /// Decides the reduced operands, the removals and the replacements before the rewrite, because
    /// a branch argument can come before its definition. The rewrite adds the truncations.
    fn planReplacements(w: *Rewriter) Allocator.Error!void {
        const s = w.s;
        const count: u32 = @intCast(s.mir.instructions.len);
        for (0..count) |i| {
            const instruction = s.instAt(@intCast(i));
            if (instruction.tag != .label) continue;
            for (s.mir.blockParams(instruction)) |param| {
                const param_operand: Mir.Operand = @bitCast(param);
                if (param_operand.regClass() != .b64 or !s.fits.isSet(s.id(param))) continue;
                if (w.reducedOperand(param) != none) w.changes = true;
            }
        }
        for (0..count) |i| {
            const instruction = s.instAt(@intCast(i));
            if (instruction.tag != .cvt or instruction.modifiers.guard() != .none) continue;
            const replacement = w.eliminableCastPair(instruction);
            if (replacement == none) continue;
            w.replace[s.id(s.mir.extra[s.mir.operands(instruction).?.first])] = replacement;
            w.removed.set(i);
            w.changes = true;
        }
        for (0..count) |i| {
            const instruction = s.instAt(@intCast(i));
            if (instruction.tag != .setp or !w.isNarrowCompare(instruction)) continue;
            w.changes = true;
            for ([_]u32{ 1, 2 }) |k| {
                const input = s.operand(instruction, k);
                if (input.reg() != null) _ = w.reducedOperand(input.word);
            }
        }
        for (0..count) |i| {
            const instruction = s.instAt(@intCast(i));
            if (instruction.tag != .bra or instruction.modifiers.guard() != .none) continue;
            for (s.mir.branchArgs(instruction)) |arg| if (s.fits.isSet(s.id(arg))) {
                _ = w.reducedOperand(arg);
            };
        }
    }

    /// Assumes an unguarded `cvt`. Returns the reduced register of the register that it truncates,
    /// or the register that its source truncates when it zero-extends from 32 bits, or `none`.
    fn eliminableCastPair(w: *Rewriter, instruction: Mir.Inst) u32 {
        const s = w.s;
        const source = s.operand(instruction, 1);
        const reg = source.reg() orelse return none;
        if (Solver.isTruncation(instruction) and s.inputFits(source)) return w.reducedOperand(source.word);
        if (!Solver.isZeroExtension(instruction) or instruction.modifiers.cvt.from != .u32) return none;
        const truncation = s.def[s.cfg.regId(reg)];
        if (truncation == none or !Solver.isTruncation(s.instAt(truncation))) return none;
        const wide = s.operand(s.instAt(truncation), 1);
        if (wide.reg() == null or !s.inputFits(wide)) return none;
        return wide.word;
    }

    /// Returns an upper bound of the truncations that `rewrite` adds, for the check in `run`.
    fn truncationBound(w: *const Rewriter) u32 {
        const s = w.s;
        var count: u32 = 0;
        for (0..s.mir.instructions.len) |i| {
            const instruction = s.instAt(@intCast(i));
            switch (instruction.tag) {
                .bra => if (instruction.modifiers.guard() == .none) {
                    count += @intCast(s.mir.branchArgs(instruction).len);
                },
                .setp => if (w.isNarrowCompare(instruction)) {
                    count += 2;
                },
                else => {},
            }
        }
        return count;
    }

    fn isNarrowParam(w: *const Rewriter, word: u32) bool {
        const operand_word: Mir.Operand = @bitCast(word);
        const r = w.s.id(word);
        return operand_word.regClass() == .b64 and w.s.fits.isSet(r) and w.reduced[r] != none;
    }

    fn isNarrowCompare(w: *const Rewriter, setp: Mir.Inst) bool {
        const s = w.s;
        if (setp.modifiers.guard() != .none or setp.modifiers.setp.bool_op != .none) return false;
        if (setp.ty != .u64 and setp.ty != .s64) return false;
        switch (setp.modifiers.setp.cmp) {
            .eq, .ne, .lt, .le, .gt, .ge, .lo, .ls, .hi, .hs => {},
            else => return false,
        }
        const a = s.operand(setp, 1);
        const b = s.operand(setp, 2);
        if (!s.inputFits(a)) return false;
        if (s.inputFits(b)) return true;
        return b.imm != null and setp.ty == .u64 and foldCompare(setp, b.imm.?) != null;
    }

    /// Assumes that the first operand fits. Returns the result of `setp`, or `null` if unknown.
    fn foldCompare(setp: Mir.Inst, imm: u64) ?bool {
        return switch (setp.modifiers.setp.cmp) {
            .gt, .hi => if (imm >= max_u32) false else null,
            .le, .ls => if (imm >= max_u32) true else null,
            .lt, .lo => if (imm > max_u32) true else null,
            .ge, .hs => if (imm > max_u32) false else null,
            .eq => if (imm > max_u32) false else null,
            .ne => if (imm > max_u32) true else null,
            else => null,
        };
    }

    /// Returns the register that replaces `word`. A new register has no replacement.
    fn resolve(w: *const Rewriter, word: u32) u32 {
        var current = word;
        while (true) {
            const op: Mir.Operand = @bitCast(current);
            if (op.kind != .reg or op.regIndex() >= w.s.mir.reg_counts[@backingInt(op.regClass())]) return current;
            const next = w.replace[w.s.id(current)];
            if (next == none) return current;
            current = next;
        }
    }

    fn rewrite(w: *Rewriter, gpa: Allocator, mir: *Mir) Allocator.Error!void {
        const s = w.s;
        var b: Mir.Builder = .{ .gpa = gpa, .reg_counts = mir.reg_counts, .label_count = mir.label_count };
        defer b.deinit();
        var words: std.ArrayList(u32) = .empty;
        for (0..mir.instructions.len) |index| {
            const i: u32 = @intCast(index);
            if (w.removed.isSet(i)) continue;
            const instruction = s.instAt(i);
            switch (instruction.tag) {
                .label => try w.narrowParams(&b, &words, instruction),
                .bra => if (instruction.modifiers.guard() == .none) {
                    const target = mir.branchTarget(instruction);
                    const label = s.instAt(s.cfg.starts[s.cfg.label_blocks[target]]);
                    words.clearRetainingCapacity();
                    for (mir.blockParams(label), mir.branchArgs(instruction)) |param, arg| {
                        try words.append(s.arena, if (w.isNarrowParam(param)) try w.reducedOrTruncated(&b, arg) else w.resolve(arg));
                    }
                    try b.addBranch(target, words.items);
                } else try w.copyInst(&b, &words, instruction),
                .setp => if (w.isNarrowCompare(instruction)) {
                    try w.narrowCompare(&b, &words, instruction);
                } else try w.copyInst(&b, &words, instruction),
                else => try w.copyInst(&b, &words, instruction),
            }
            if (w.after[i]) |extra| {
                words.clearRetainingCapacity();
                try words.append(s.arena, extra.result);
                if (extra.tag == .add) try words.append(s.arena, w.resolve(extra.source));
                try appendImm(s.arena, &words, extra.imm);
                switch (extra.tag) {
                    .mov => try b.addTyped(.mov, .b32, .{ .none = .{} }, words.items),
                    .add => try b.addTyped(.add, .u32, .{ .arith = .{} }, words.items),
                    else => unreachable,
                }
            }
        }
        b.reg_counts[@backingInt(Mir.Class.b32)] = w.next_b32;
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

    fn narrowParams(w: *Rewriter, b: *Mir.Builder, words: *std.ArrayList(u32), label: Mir.Inst) Allocator.Error!void {
        const s = w.s;
        words.clearRetainingCapacity();
        const params = s.mir.blockParams(label);
        for (params) |param| {
            try words.append(s.arena, if (w.isNarrowParam(param)) w.reduced[s.id(param)] else param);
        }
        try b.addLabel(s.mir.labelOf(label), words.items);
        for (params) |param| if (w.isNarrowParam(param)) {
            const extend: Mir.Inst.Modifiers = .{ .cvt = .{ .from = .u32 } };
            try b.addTyped(.cvt, .u64, extend, &.{ param, w.reduced[s.id(param)] });
        };
    }

    /// Assumes that `word` fits. Returns its reduced register, or a new truncation of `word`.
    fn reducedOrTruncated(w: *Rewriter, b: *Mir.Builder, word: u32) Allocator.Error!u32 {
        const r = w.s.id(word);
        if (w.reduced_done.isSet(r) and w.reduced[r] != none) return w.resolve(w.reduced[r]);
        const result: u32 = @bitCast(Mir.Operand.reg(.b32, @intCast(w.next_b32)));
        w.next_b32 += 1;
        try b.addTyped(.cvt, .u32, .{ .cvt = .{ .from = .u64 } }, &.{ result, w.resolve(word) });
        return result;
    }

    fn narrowCompare(w: *Rewriter, b: *Mir.Builder, words: *std.ArrayList(u32), setp: Mir.Inst) Allocator.Error!void {
        const s = w.s;
        const result = s.mir.extra[s.mir.operands(setp).?.first];
        const a = s.operand(setp, 1);
        const c = s.operand(setp, 2);
        words.clearRetainingCapacity();
        try words.append(s.arena, result);
        if (c.imm) |imm| if (setp.ty == .u64) if (foldCompare(setp, imm)) |value| {
            try appendImm(s.arena, words, @intFromBool(value));
            try appendImm(s.arena, words, 0);
            return b.addTyped(.setp, .u32, .{ .setp = .{ .cmp = .ne } }, words.items);
        };
        for ([_]Input{ a, c }) |input| {
            if (input.imm) |imm| try appendImm(s.arena, words, imm) else try words.append(s.arena, try w.reducedOrTruncated(b, input.word));
        }
        try b.addTyped(.setp, .u32, setp.modifiers, words.items);
    }

    fn copyInst(w: *const Rewriter, b: *Mir.Builder, words: *std.ArrayList(u32), instruction: Mir.Inst) Allocator.Error!void {
        words.clearRetainingCapacity();
        try words.appendSlice(w.s.arena, w.s.mir.span(instruction));
        var uses: Uses = .{ .w = w, .words = words.items, .base = instruction.data };
        w.s.mir.visitRegs(instruction, &uses);
        try b.addTyped(instruction.tag, instruction.ty, instruction.modifiers, words.items);
    }

    const Uses = struct {
        w: *const Rewriter,
        words: []u32,
        base: u32,

        pub fn reg(uses: *Uses, at: u32, role: Mir.RegRole) void {
            if (role != .use) return;
            uses.words[at - uses.base] = uses.w.resolve(uses.words[at - uses.base]);
        }
    };
};

fn appendImm(arena: Allocator, words: *std.ArrayList(u32), value: u64) Allocator.Error!void {
    if (value < 1 << 28) return words.append(arena, @bitCast(Mir.Operand{ .kind = .imm, .payload = @intCast(value) }));
    try words.appendSlice(arena, &.{
        @bitCast(Mir.Operand{ .kind = .imm64, .payload = @backingInt(Mir.ImmFormat.unsigned) }),
        @truncate(value),
        @truncate(value >> 32),
    });
}
