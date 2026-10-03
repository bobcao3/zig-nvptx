//! This file gives generic accesses the state space of their address, as `InferAddressSpaces.cpp`
//! of LLVM does. Input: MIR in SSA form. Code generation marks each `cvta.S.u64 d, a` with S in
//! `.local`, `.shared` or `.global` as uncommitted. The tree of S holds the `d` of each uncommitted
//! `cvta.S`, and each register that one of these instructions defines from tree registers of S
//! only: a `mov`, a 64-bit `cvt`, a `selp`, a block parameter, and the pointer arithmetic `a + k`,
//! `a - k` or `mad.lo` with a tree register as the addend. The operand `k` is not a tree register.
//! A part is a set of tree registers that these instructions connect. Rule: when an `ld`, `st`,
//! `atom` or a side of a `copy` without a state space takes its address from a part, each root of
//! the part becomes `mov.u64 d, a` and each such access names S. Each other use of a part register
//! reads a new `cvta.S`, and a `cvta.to.S` of a part register becomes a `mov`. A volatile `.local`
//! access stays generic, because it needs PTX 9.1. Every other uncommitted `cvta` loses its flag.
//! Output: no `cvta` is uncommitted. Proof: let w be the window base of S. `cvta.S` computes
//! g = a + w for an address a of S, and g names the location g - w of S (PTX ISA 9.4, sections
//! 6.4.1.1 and 9.7.10.23). By induction, each part register holds g - w where the old code held g.
//! Pointer arithmetic adds an offset in the object, which keeps the difference and keeps an address
//! of S, so a new `cvta.S` is valid. `selp` and parameters copy one value of the part. An access to
//! g is an access to the location g - w of S (section 8), so each access and each unit of a `copy`
//! side keeps its location, order and scope. A new `cvta.S` gives each other use g.

const std = @import("std");
const Allocator = std.mem.Allocator;

const Mir = @import("../Mir.zig");
const Cfg = @import("../Cfg.zig");

const none = Cfg.none;

/// Without `infer`, the pass only commits each conversion where it stands.
pub fn run(gpa: Allocator, mir: *Mir, infer: bool) Allocator.Error!void {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    if (!hasUncommitted(mir)) return;
    if (!infer) {
        for (mir.instructions.items(.tag), mir.instructions.items(.modifiers)) |tag, *modifiers| {
            if (tag == .cvta) modifiers.cvta.uncommitted = false;
        }
        return;
    }
    var inference = try Inference.init(arena_state.allocator(), mir);
    inference.propagate();
    while (inference.confirm()) {}
    inference.group();
    inference.markUseful();
    if (!try inference.rewrite(gpa, mir)) {
        inference.enabled = false;
        _ = try inference.rewrite(gpa, mir);
    }
}

fn hasUncommitted(mir: *const Mir) bool {
    for (mir.instructions.items(.tag), mir.instructions.items(.modifiers)) |tag, modifiers| {
        if (tag == .cvta and modifiers.cvta.uncommitted) return true;
    }
    return false;
}

const Label = enum(u8) {
    none,
    local,
    shared,
    global,
    conflict,

    fn isTree(label: Label) bool {
        return switch (label) {
            .local, .shared, .global => true,
            .none, .conflict => false,
        };
    }

    fn space(label: Label) Mir.Space {
        return switch (label) {
            .local => .local,
            .shared => .shared,
            .global => .global,
            .none, .conflict => unreachable,
        };
    }

    fn fromSpace(space_: Mir.Space) ?Label {
        return switch (space_) {
            .local => .local,
            .shared => .shared,
            .global => .global,
            else => null,
        };
    }
};

const Kind = enum { root, single, sum, merge };

/// An instruction that can define a register of a tree, with positions in `Mir.extra`.
const Derived = struct {
    inst: u32,
    kind: Kind,
    def: u32,
    root: Label = .none,
    inputs: [2]u32 = undefined,
    input_ats: [2]u32 = undefined,
    input_count: u2 = 0,
    others: [2]u32 = undefined,
    other_count: u2 = 0,

    fn addInput(derived: *Derived, inference: *const Inference, at: u32) void {
        derived.input_ats[derived.input_count] = at;
        derived.inputs[derived.input_count] = inference.regId(inference.mir.extra[at]);
        derived.input_count += 1;
    }

    fn addOther(derived: *Derived, inference: *const Inference, at: u32) void {
        if (!inference.isWideReg(at)) return;
        derived.others[derived.other_count] = inference.regId(inference.mir.extra[at]);
        derived.other_count += 1;
    }
};

const Edge = struct {
    param: u32,
    arg: u32,
    arg_at: u32,
};

const Need = struct { at: u32, word: u32 };

const Base = struct { at: u32, side: u1 };

const Inference = struct {
    arena: Allocator,
    mir: *const Mir,
    cfg: Cfg,
    label: []Label,
    def_inst: []u32,
    derived: std.ArrayList(Derived) = .empty,
    derived_of: []u32,
    edges: std.ArrayList(Edge) = .empty,
    parent: []u32,
    useful: []bool,
    /// False when the pass only clears the flags of the conversions.
    enabled: bool = true,

    fn init(arena: Allocator, mir: *const Mir) Allocator.Error!Inference {
        const cfg: Cfg = try .init(arena, mir);
        const label = try arena.alloc(Label, cfg.reg_count);
        @memset(label, .none);
        const def_inst = try arena.alloc(u32, cfg.reg_count);
        @memset(def_inst, none);
        const derived_of = try arena.alloc(u32, mir.instructions.len);
        @memset(derived_of, none);
        const parent = try arena.alloc(u32, cfg.reg_count);
        for (parent, 0..) |*p, i| p.* = @intCast(i);
        const useful = try arena.alloc(bool, cfg.reg_count);
        @memset(useful, false);
        var inference: Inference = .{
            .arena = arena,
            .mir = mir,
            .cfg = cfg,
            .label = label,
            .def_inst = def_inst,
            .derived_of = derived_of,
            .parent = parent,
            .useful = useful,
        };
        var defs: Defs = .{ .inference = &inference, .inst = 0 };
        for (0..mir.instructions.len) |i| {
            defs.inst = @intCast(i);
            mir.visitRegs(mir.instructions.get(i), &defs);
        }
        for (0..mir.instructions.len) |i| {
            const inst = mir.instructions.get(i);
            if (try inference.derive(@intCast(i), inst)) |derived| {
                derived_of[i] = @intCast(inference.derived.items.len);
                try inference.derived.append(arena, derived);
            }
            if (inst.tag == .bra and inst.modifiers.guard() == .none) try inference.collectEdges(inst);
        }
        for (inference.derived.items) |derived| {
            if (derived.kind == .root) label[derived.def] = derived.root;
        }
        return inference;
    }

    const Defs = struct {
        inference: *Inference,
        inst: u32,

        pub fn reg(defs: *Defs, at: u32, role: Mir.RegRole) void {
            if (role != .def) return;
            defs.inference.def_inst[defs.inference.regId(defs.inference.mir.extra[at])] = defs.inst;
        }
    };

    fn regId(inference: *const Inference, word: u32) u32 {
        return inference.cfg.regId(@bitCast(word));
    }

    fn isWideReg(inference: *const Inference, at: u32) bool {
        const op: Mir.Operand = @bitCast(inference.mir.extra[at]);
        return op.kind == .reg and op.regClass() == .b64;
    }

    fn operandAts(inference: *const Inference, inst: Mir.Inst, buf: *[6]u32) []const u32 {
        const range = inference.mir.operands(inst).?;
        var at = range.first;
        var count: usize = 0;
        for (0..range.count) |_| {
            if (count < buf.len) {
                buf[count] = at;
                count += 1;
            }
            at += @as(Mir.Operand, @bitCast(inference.mir.extra[at])).len();
        }
        return buf[@intFromBool(inst.modifiers.guard() != .none)..count];
    }

    fn derive(inference: *const Inference, index: u32, inst: Mir.Inst) Allocator.Error!?Derived {
        switch (inst.tag) {
            .cvta, .mov, .cvt, .add, .sub, .mad, .selp => {},
            else => return null,
        }
        if (inst.modifiers.guard() != .none) return null;
        var buf: [6]u32 = undefined;
        const ops = inference.operandAts(inst, &buf);
        if (!inference.isWideReg(ops[0])) return null;
        var derived: Derived = .{ .inst = index, .kind = .single, .def = inference.regId(inference.mir.extra[ops[0]]) };
        switch (inst.tag) {
            .cvta => {
                if (inst.modifiers.cvta.to or !inst.modifiers.cvta.uncommitted) return null;
                derived.kind = .root;
                derived.root = Label.fromSpace(inst.modifiers.cvta.space) orelse return null;
            },
            .mov => {
                if (!inference.isWideReg(ops[1])) return null;
                derived.addInput(inference, ops[1]);
            },
            .cvt => {
                const cvt = inst.modifiers.cvt;
                if (!isWideInt(inst.ty) or !isWideInt(cvt.from) or cvt.rounding != .none or cvt.sat) return null;
                if (!inference.isWideReg(ops[1])) return null;
                derived.addInput(inference, ops[1]);
            },
            .add => {
                if (!isWideInt(inst.ty) or !isPointerArithmetic(inst)) return null;
                derived.kind = .sum;
                for (ops[1..3]) |at| if (inference.isWideReg(at)) derived.addInput(inference, at);
                if (derived.input_count == 0) return null;
            },
            .sub => {
                if (!isWideInt(inst.ty) or !isPointerArithmetic(inst)) return null;
                if (!inference.isWideReg(ops[1])) return null;
                derived.addInput(inference, ops[1]);
                derived.addOther(inference, ops[2]);
            },
            .mad => {
                if (!isWideInt(inst.ty) or inst.modifiers.arith.mode != .lo or !isPointerArithmetic(inst)) return null;
                if (!inference.isWideReg(ops[3])) return null;
                derived.addInput(inference, ops[3]);
                derived.addOther(inference, ops[1]);
                derived.addOther(inference, ops[2]);
            },
            .selp => {
                if (inst.ty != .b64 or !inference.isWideReg(ops[1]) or !inference.isWideReg(ops[2])) return null;
                derived.kind = .merge;
                derived.addInput(inference, ops[1]);
                derived.addInput(inference, ops[2]);
            },
            else => unreachable,
        }
        return derived;
    }

    fn collectEdges(inference: *Inference, inst: Mir.Inst) Allocator.Error!void {
        const mir = inference.mir;
        const target = inference.cfg.label_blocks[mir.branchTarget(inst)];
        if (target == none) return;
        const params = mir.blockParams(mir.instructions.get(inference.cfg.starts[target]));
        for (mir.branchArgs(inst), params, 0..) |arg, param, j| {
            const op: Mir.Operand = @bitCast(arg);
            if (op.kind != .reg or op.regClass() != .b64) continue;
            try inference.edges.append(inference.arena, .{
                .param = inference.regId(param),
                .arg = inference.regId(arg),
                .arg_at = inst.data + 2 + @as(u32, @intCast(j)),
            });
        }
    }

    /// Unlike integer arithmetic on addresses, pointer arithmetic adds an offset, never an address.
    fn isPointerArithmetic(inst: Mir.Inst) bool {
        const arith = inst.modifiers.arith;
        return arith.address and !arith.cc and !arith.sat;
    }

    fn isWideInt(ty: Mir.Type) bool {
        return ty == .b64 or ty == .u64 or ty == .s64;
    }

    fn join(inference: *Inference, id: u32, label: Label) bool {
        if (label == .none) return false;
        const current = inference.label[id];
        const joined: Label = if (current == .none) label else if (current == label) current else .conflict;
        if (joined == current) return false;
        inference.label[id] = joined;
        return true;
    }

    fn propagate(inference: *Inference) void {
        var changed = true;
        while (changed) {
            changed = false;
            for (inference.derived.items) |derived| {
                for (derived.inputs[0..derived.input_count]) |input| {
                    if (inference.join(derived.def, inference.label[input])) changed = true;
                }
            }
            for (inference.edges.items) |edge| {
                if (inference.join(edge.param, inference.label[edge.arg])) changed = true;
            }
        }
    }

    fn confirm(inference: *Inference) bool {
        var changed = false;
        for (inference.derived.items) |derived| {
            const label = inference.label[derived.def];
            if (!label.isTree() or inference.holds(derived, label)) continue;
            inference.label[derived.def] = .conflict;
            changed = true;
        }
        for (inference.edges.items) |edge| {
            const label = inference.label[edge.param];
            if (!label.isTree() or inference.label[edge.arg] == label or inference.def_inst[edge.arg] == none) continue;
            inference.label[edge.param] = .conflict;
            changed = true;
        }
        return changed;
    }

    fn holds(inference: *const Inference, derived: Derived, label: Label) bool {
        for (derived.others[0..derived.other_count]) |other| if (inference.label[other].isTree()) return false;
        const inputs = derived.inputs[0..derived.input_count];
        switch (derived.kind) {
            .root => return true,
            .single => return inference.label[inputs[0]] == label,
            .merge => return inference.label[inputs[0]] == label and inference.label[inputs[1]] == label,
            .sum => {
                var trees: u32 = 0;
                for (inputs) |input| {
                    if (!inference.label[input].isTree()) continue;
                    if (inference.label[input] != label) return false;
                    trees += 1;
                }
                return trees == 1;
            },
        }
    }

    fn find(inference: *Inference, id: u32) u32 {
        var root = id;
        while (inference.parent[root] != root) root = inference.parent[root];
        var at = id;
        while (inference.parent[at] != root) {
            const next = inference.parent[at];
            inference.parent[at] = root;
            at = next;
        }
        return root;
    }

    fn unite(inference: *Inference, a: u32, b: u32) void {
        inference.parent[inference.find(a)] = inference.find(b);
    }

    fn group(inference: *Inference) void {
        for (inference.derived.items) |derived| {
            if (derived.kind == .root or !inference.label[derived.def].isTree()) continue;
            for (derived.inputs[0..derived.input_count]) |input| {
                if (inference.label[input].isTree()) inference.unite(derived.def, input);
            }
        }
        for (inference.edges.items) |edge| {
            if (inference.label[edge.param].isTree() and inference.label[edge.arg].isTree()) inference.unite(edge.param, edge.arg);
        }
    }

    /// Returns the address operands whose state space the pass gives, with their `MemOperand` side.
    fn accessBases(inference: *const Inference, inst: Mir.Inst, buf: *[2]Base) []const Base {
        var ops_buf: [6]u32 = undefined;
        const ops = switch (inst.tag) {
            .ld, .st, .atom, .copy => inference.operandAts(inst, &ops_buf),
            else => return buf[0..0],
        };
        var count: usize = 0;
        const candidates: [2]Base = switch (inst.tag) {
            .ld, .atom => .{ .{ .at = ops[1], .side = 0 }, undefined },
            .st => .{ .{ .at = ops[0], .side = 0 }, undefined },
            .copy => .{ .{ .at = ops[0], .side = 0 }, .{ .at = ops[1], .side = 1 } },
            else => unreachable,
        };
        for (candidates[0..Mir.memOperandCount(inst)]) |base| {
            const mem = inference.mir.memOperand(inst, base.side);
            if (mem.space != .generic or !inference.isWideReg(base.at)) continue;
            const label = inference.label[inference.regId(inference.mir.extra[base.at])];
            if (!label.isTree() or (label == .local and mem.sem == .@"volatile")) continue;
            buf[count] = base;
            count += 1;
        }
        return buf[0..count];
    }

    fn markUseful(inference: *Inference) void {
        var buf: [2]Base = undefined;
        for (0..inference.mir.instructions.len) |i| {
            for (inference.accessBases(inference.mir.instructions.get(i), &buf)) |base| {
                inference.useful[inference.find(inference.regId(inference.mir.extra[base.at]))] = true;
            }
        }
    }

    fn converted(inference: *Inference, id: u32) bool {
        return inference.enabled and inference.label[id].isTree() and inference.useful[inference.find(id)];
    }

    /// Returns the operand position of a `cvta.to` whose operand belongs to a converted part of its
    /// state space. That operand already holds the address in the state space.
    fn undoneConversion(inference: *Inference, inst: Mir.Inst) ?u32 {
        if (inst.tag != .cvta or !inst.modifiers.cvta.to or inst.modifiers.guard() != .none) return null;
        var buf: [6]u32 = undefined;
        const at = inference.operandAts(inst, &buf)[1];
        if (!inference.isWideReg(at)) return null;
        const id = inference.regId(inference.mir.extra[at]);
        if (!inference.converted(id) or inference.label[id].space() != inst.modifiers.cvta.space) return null;
        return at;
    }

    /// Collects the operand positions where an instruction reads a register of a converted part as
    /// an address in the state space of the part.
    fn keptPositions(inference: *Inference, index: u32, inst: Mir.Inst, kept: *std.ArrayList(u32)) Allocator.Error!void {
        const mir = inference.mir;
        const arena = inference.arena;
        kept.clearRetainingCapacity();
        if (inference.undoneConversion(inst)) |at| try kept.append(arena, at);
        var bases: [2]Base = undefined;
        for (inference.accessBases(inst, &bases)) |base| try kept.append(arena, base.at);
        if (inference.derived_of[index] != none) {
            const derived = inference.derived.items[inference.derived_of[index]];
            if (derived.kind != .root and inference.converted(derived.def)) {
                for (derived.input_ats[0..derived.input_count], derived.inputs[0..derived.input_count]) |at, input| {
                    if (inference.label[input].isTree()) try kept.append(arena, at);
                }
            }
        }
        if (inst.tag != .bra or inst.modifiers.guard() != .none) return;
        const target = inference.cfg.label_blocks[mir.branchTarget(inst)];
        if (target == none) return;
        const params = mir.blockParams(mir.instructions.get(inference.cfg.starts[target]));
        for (mir.branchArgs(inst), params, 0..) |arg, param, j| {
            const op: Mir.Operand = @bitCast(arg);
            if (op.kind == .reg and inference.converted(inference.regId(param)) and inference.converted(inference.regId(arg))) {
                try kept.append(arena, inst.data + 2 + @as(u32, @intCast(j)));
            }
        }
    }

    const Scan = struct {
        inference: *Inference,
        kept: []const u32,
        needs: *std.ArrayList(Need),
        failed: bool = false,

        pub fn reg(scan: *Scan, at: u32, role: Mir.RegRole) void {
            if (role != .use) return;
            const word = scan.inference.mir.extra[at];
            const op: Mir.Operand = @bitCast(word);
            if (op.regClass() != .b64 or !scan.inference.converted(scan.inference.regId(word))) return;
            if (std.mem.findScalar(u32, scan.kept, at) != null) return;
            scan.needs.append(scan.inference.arena, .{ .at = at, .word = word }) catch {
                scan.failed = true;
            };
        }
    };

    /// Returns false and keeps the MIR when a new register needs an index above `maxInt(u16)`.
    fn rewrite(inference: *Inference, gpa: Allocator, mir: *Mir) Allocator.Error!bool {
        const arena = inference.arena;
        var b: Mir.Builder = .{ .gpa = gpa, .reg_counts = mir.reg_counts, .label_count = mir.label_count };
        defer b.deinit();
        var overflow = false;
        var kept: std.ArrayList(u32) = .empty;
        var needs: std.ArrayList(Need) = .empty;
        var words: std.ArrayList(u32) = .empty;
        var fresh: std.ArrayList(Need) = .empty;
        for (0..mir.instructions.len) |i| {
            const inst = mir.instructions.get(i);
            try inference.keptPositions(@intCast(i), inst, &kept);
            needs.clearRetainingCapacity();
            fresh.clearRetainingCapacity();
            var scan: Scan = .{ .inference = inference, .kept = kept.items, .needs = &needs };
            mir.visitRegs(inst, &scan);
            if (scan.failed) return error.OutOfMemory;
            words.clearRetainingCapacity();
            try words.appendSlice(arena, mir.span(inst));
            for (needs.items) |need| {
                const known = for (fresh.items) |entry| {
                    if (entry.at == need.word) break entry.word;
                } else null;
                const reg_word = known orelse reg_word: {
                    const index = b.reg_counts[@backingInt(Mir.Class.b64)];
                    if (index > std.math.maxInt(u16)) {
                        overflow = true;
                        break :reg_word need.word;
                    }
                    b.reg_counts[@backingInt(Mir.Class.b64)] += 1;
                    const new: u32 = @bitCast(Mir.Operand.reg(.b64, @intCast(index)));
                    const label = inference.label[inference.regId(need.word)];
                    try b.addTyped(.cvta, .u64, .{ .cvta = .{ .space = label.space() } }, &.{ new, need.word });
                    try fresh.append(arena, .{ .at = need.word, .word = new });
                    break :reg_word new;
                };
                words.items[need.at - inst.data] = reg_word;
            }
            var tag = inst.tag;
            var ty = inst.ty;
            var modifiers = inst.modifiers;
            if (inference.derived_of[i] != none) {
                const derived = inference.derived.items[inference.derived_of[i]];
                if (derived.kind == .root and inference.converted(derived.def)) {
                    tag = .mov;
                    ty = .u64;
                    modifiers = .{ .none = .{} };
                }
            }
            if (inference.undoneConversion(inst) != null) {
                tag = .mov;
                ty = .u64;
                modifiers = .{ .none = .{} };
            }
            if (tag == .cvta) modifiers.cvta.uncommitted = false;
            var bases: [2]Base = undefined;
            for (inference.accessBases(inst, &bases)) |base| {
                if (!inference.converted(inference.regId(mir.extra[base.at]))) continue;
                const mem_at = mir.memOperandAt(inst) - inst.data + base.side;
                var mem: Mir.MemOperand = @bitCast(words.items[mem_at]);
                mem.space = inference.label[inference.regId(mir.extra[base.at])].space();
                words.items[mem_at] = @bitCast(mem);
            }
            try b.addTyped(tag, ty, modifiers, words.items);
        }
        if (overflow) return false;
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
};
