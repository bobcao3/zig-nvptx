//! This file checks that the MIR of a function is in SSA form. Each register has at most one
//! definition. The definition dominates each use. A branch passes one argument for each parameter
//! of its target, and each argument has the class of its parameter. A register without a
//! definition is undefined. The file also checks register ranges, else operands, branch targets,
//! guarded branches, the fall-through into blocks with parameters and the sizes of the memory
//! operands.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Mir = @import("Mir.zig");
const Cfg = @import("Cfg.zig");

const none = Cfg.none;

/// Returns a description of the first violation, or null.
pub fn check(arena: Allocator, mir: *const Mir) Allocator.Error!?[]const u8 {
    for (0..mir.instructions.len) |i| {
        const inst = mir.instructions.get(i);
        var bounds: Bounds = .{ .mir = mir };
        mir.visitRegs(inst, &bounds);
        if (bounds.bad) |word| return try std.fmt.allocPrint(arena, "instruction {d} names register {f}, which is out of range", .{ i, fmtReg(word) });
        if (!memOperandsFit(mir, inst)) return try std.fmt.allocPrint(arena, "the memory operand of instruction {d} does not give its size", .{i});
    }
    const cfg = try Cfg.init(arena, mir);
    const def_inst = try arena.alloc(u32, cfg.reg_count);
    const def_block = try arena.alloc(u32, cfg.reg_count);
    @memset(def_inst, none);
    @memset(def_block, none);
    var refs: Refs = .{ .mir = mir };
    for (cfg.order) |b| {
        for (cfg.starts[b]..cfg.starts[b + 1]) |i| {
            try refs.collect(arena, i);
            for (refs.list.items) |ref| {
                if (ref.role != .def) continue;
                const id = cfg.regId(@bitCast(ref.word));
                if (def_inst[id] != none) return try std.fmt.allocPrint(arena, "register {f} has definitions in instructions {d} and {d}", .{ fmtReg(ref.word), def_inst[id], i });
                def_inst[id] = @intCast(i);
                def_block[id] = b;
            }
        }
    }
    for (cfg.order) |b| {
        for (cfg.starts[b]..cfg.starts[b + 1]) |i| {
            const inst = mir.instructions.get(i);
            try refs.collect(arena, i);
            for (refs.list.items) |ref| {
                if (ref.role != .use) continue;
                const id = cfg.regId(@bitCast(ref.word));
                if (def_inst[id] == none) continue;
                const dominated = if (def_block[id] == b) def_inst[id] < i else cfg.dominates(def_block[id], b);
                if (!dominated) return try std.fmt.allocPrint(arena, "the definition of {f} in instruction {d} does not dominate its use in instruction {d}", .{ fmtReg(ref.word), def_inst[id], i });
            }
            if (Mir.elseCount(inst) != 0) {
                const range = mir.operands(inst).?;
                var at = range.first;
                for (0..range.count - 1) |_| at += @as(Mir.Operand, @bitCast(mir.extra[at])).len();
                const result: Mir.Operand = @bitCast(mir.extra[range.first + 1]);
                const otherwise: Mir.Operand = @bitCast(mir.extra[at]);
                if (result.kind != otherwise.kind or (result.kind == .reg and result.regClass() != otherwise.regClass())) {
                    return try std.fmt.allocPrint(arena, "the else operand of instruction {d} does not match its result", .{i});
                }
            }
            if (inst.tag == .bra) {
                const target = cfg.label_blocks[mir.branchTarget(inst)];
                if (target == none) return try std.fmt.allocPrint(arena, "instruction {d} jumps to a label that no instruction places", .{i});
                const params = mir.blockParams(mir.instructions.get(cfg.starts[target]));
                if (inst.modifiers.guard() != .none) {
                    if (params.len != 0) return try std.fmt.allocPrint(arena, "the guarded branch in instruction {d} jumps to a block with parameters", .{i});
                    continue;
                }
                const args = mir.branchArgs(inst);
                if (args.len != params.len) return try std.fmt.allocPrint(arena, "the branch in instruction {d} passes {d} arguments to {d} parameters", .{ i, args.len, params.len });
                for (args, params) |arg, param| {
                    const a: Mir.Operand = @bitCast(arg);
                    const p: Mir.Operand = @bitCast(param);
                    if (a.regClass() != p.regClass()) return try std.fmt.allocPrint(arena, "the branch in instruction {d} passes {f} to the parameter {f}", .{ i, fmtReg(arg), fmtReg(param) });
                }
            }
        }
        const end = cfg.starts[b + 1];
        const falls = end == cfg.starts[b] or switch (mir.instructions.items(.tag)[end - 1]) {
            .ret, .trap => false,
            .bra => mir.instructions.items(.modifiers)[end - 1].guard() != .none,
            else => true,
        };
        if (!falls) continue;
        const next = cfg.successors[b][if (end > cfg.starts[b] and mir.instructions.items(.tag)[end - 1] == .bra) 1 else 0];
        if (next == none or cfg.starts[next] == mir.instructions.len) continue;
        if (mir.blockParams(mir.instructions.get(cfg.starts[next])).len != 0) {
            return try std.fmt.allocPrint(arena, "control falls into the block with parameters at instruction {d}", .{cfg.starts[next]});
        }
    }
    return null;
}

/// Returns null when PTX can hold each instruction after the link-time passes, and else describes
/// the first other instruction. `.local` takes no `atom`, no order and, before PTX 9.1, no
/// `.volatile` (PTX ISA 9.4, sections 9.7.10.8 and 9.7.15.5).
pub fn checkLowered(mir: *const Mir) ?[]const u8 {
    for (0..mir.instructions.len) |i| {
        const inst = mir.instructions.get(i);
        switch (inst.tag) {
            .cvta => if (inst.modifiers.cvta.uncommitted) return "a conversion to a generic address is still uncommitted",
            .copy => return "a copy is still unexpanded",
            .ld, .st, .atom => {
                const mem = mir.memOperand(inst, 0);
                if (mem.space == .local and (inst.tag == .atom or mem.sem != .none)) return "an atomic, volatile or ordered access names local memory";
            },
            else => {},
        }
    }
    return null;
}

fn memOperandsFit(mir: *const Mir, inst: Mir.Inst) bool {
    switch (inst.tag) {
        .atom => return mir.memOperand(inst, 0).size() == typeSize(inst.ty),
        .ld, .st => {
            const elems: u64 = switch (inst.modifiers.mem.vec) {
                .scalar => 1,
                .v2 => 2,
                .v4 => 4,
            };
            return mir.memOperand(inst, 0).size() == typeSize(inst.ty) * elems;
        },
        .copy => return inst.modifiers.copy.op == .set or mir.memOperand(inst, 0).log2_size == mir.memOperand(inst, 1).log2_size,
        else => return true,
    }
}

fn typeSize(ty: Mir.Type) u64 {
    return switch (ty) {
        .b8, .u8, .s8 => 1,
        .b16, .u16, .s16, .f16, .bf16 => 2,
        .b32, .u32, .s32, .f32, .f16x2, .bf16x2 => 4,
        .b64, .u64, .s64, .f64 => 8,
        .none, .pred => 0,
    };
}

const Bounds = struct {
    mir: *const Mir,
    bad: ?u32 = null,

    pub fn reg(bounds: *Bounds, at: u32, role: Mir.RegRole) void {
        _ = role;
        const op: Mir.Operand = @bitCast(bounds.mir.extra[at]);
        if (op.kind != .reg or op.regIndex() >= bounds.mir.reg_counts[@backingInt(op.regClass())]) bounds.bad = bounds.mir.extra[at];
    }
};

const Refs = struct {
    mir: *const Mir,
    list: std.ArrayList(Ref) = .empty,

    const Ref = struct { word: u32, role: Mir.RegRole };

    fn collect(refs: *Refs, arena: Allocator, i: usize) Allocator.Error!void {
        const inst = refs.mir.instructions.get(i);
        refs.list.clearRetainingCapacity();
        try refs.list.ensureTotalCapacity(arena, refs.mir.span(inst).len);
        refs.mir.visitRegs(inst, refs);
    }

    pub fn reg(refs: *Refs, at: u32, role: Mir.RegRole) void {
        refs.list.appendAssumeCapacity(.{ .word = refs.mir.extra[at], .role = role });
    }
};

fn fmtReg(word: u32) std.fmt.Alt(u32, formatReg) {
    return .{ .data = word };
}

fn formatReg(word: u32, w: *std.Io.Writer) std.Io.Writer.Error!void {
    const op: Mir.Operand = @bitCast(word);
    if (op.kind != .reg) return w.print("(operand {x})", .{word});
    try w.print("{s}{d}", .{ op.regClass().prefix(), op.regIndex() });
}
