//! This file writes PTX text and the declarations with their relocations. The file assumes that
//! `Lower.run` has already given each function PTX registers.

const std = @import("std");
const Mir = @import("Mir.zig");
const Emit = @This();

mir: *const Mir,
writer: *std.Io.Writer,

/// Writes `mir.text[start..end]`. Each relocation in the range inserts its name at its offset.
pub fn text(emit: Emit, names: anytype, start: usize, end: usize) !void {
    const mir = emit.mir;
    const w = emit.writer;
    var pos = start;
    for (mir.relocs) |reloc| {
        if (reloc.offset < start or reloc.offset >= end) continue;
        try w.writeAll(mir.text[pos..reloc.offset]);
        try names.writeName(w, reloc.target);
        pos = reloc.offset;
    }
    try w.writeAll(mir.text[pos..end]);
}

pub fn body(emit: Emit, names: anytype) !void {
    const mir = emit.mir;
    const w = emit.writer;
    try w.writeAll("\n{\n");
    for (std.enums.values(Mir.Class)) |class| {
        const count = mir.reg_counts[@backingInt(class)];
        if (count == 0) continue;
        try w.print("\t.reg .{s} {s}<{d}>;\n", .{
            if (class == .pred) "pred" else @tagName(class), class.prefix(), count,
        });
    }
    for (mir.locals, 0..) |local, i| {
        try w.print("\t.local .align {d} .b8 local_{d}[{d}];\n", .{ local.alignment, i, local.size });
    }
    for (0..mir.instructions.len) |i| try emit.instruction(names, mir.instructions.get(i), @intCast(i));
    try w.writeAll("}\n");
}

/// Returns whether the block that starts at `label_index` holds a `nounroll`.
fn holdsNounroll(mir: *const Mir, label_index: u32) bool {
    const tags = mir.instructions.items(.tag);
    for (tags[label_index + 1 ..]) |tag| switch (tag) {
        .nounroll => return true,
        .label, .bra, .ret, .trap => return false,
        else => {},
    };
    return false;
}

fn instruction(emit: Emit, names: anytype, inst: Mir.Inst, index: u32) !void {
    const mir = emit.mir;
    const w = emit.writer;
    const extra = mir.extra;
    const tag = inst.tag;
    const data = inst.data;
    switch (tag) {
        .label => {
            try w.print("$L__{d}:\n", .{mir.labelOf(inst)});
            if (holdsNounroll(mir, index)) try w.writeAll("\t.pragma \"nounroll\";\n");
        },
        .nounroll => {},
        .param_load, .param_store => {
            const access: Mir.ParamAccess = @bitCast(extra[data]);
            const param: Mir.Operand = @bitCast(extra[data + 1]);
            const regs = extra[data + 2 ..][0..access.len];
            try w.writeAll(if (tag == .param_load) "\tld.param" else "\tst.param");
            if (access.len > 1) try w.print(".v{d}", .{access.len});
            try w.print(".{t} ", .{access.mem});
            if (tag == .param_load) {
                try writeRegs(w, regs);
                try w.writeAll(", [");
                try writeParam(w, param);
                try w.print("+{d}]", .{access.offset});
            } else {
                try w.writeAll("[");
                try writeParam(w, param);
                try w.print("+{d}], ", .{access.offset});
                try writeRegs(w, regs);
            }
            try w.writeAll(";\n");
        },
        .call_begin => try w.writeAll("\t{\n"),
        .param_decl => {
            try w.writeByte('\t');
            try emit.operand(names, data);
            try w.writeAll(";\n");
        },
        .call => {
            const call: Mir.Call = @bitCast(extra[data]);
            try w.writeAll(if (call.has_ret) "\tcall (retval0), " else "\tcall ");
            try emit.operand(names, data + 1);
            if (!call.indirect or call.arg_count != 0) {
                try w.writeAll(", (");
                for (0..call.arg_count) |i| try w.print("{s}param{d}", .{ if (i == 0) "" else ", ", i });
                try w.writeAll(")");
            }
            if (call.indirect) try w.print(", $P__{d}", .{extra[data + 2]});
            try w.writeAll(";\n");
        },
        .call_end => try w.writeAll("\t}\n"),
        .ret => try w.writeAll("\tret;\n"),
        .@"asm" => {
            try w.writeAll("\t// begin inline asm\n\t");
            var at = data + 1;
            for (0..extra[data]) |k| {
                const op: Mir.Operand = @bitCast(extra[at]);
                if (k >= 2 * @as(u32, inst.modifiers.@"asm".outputs)) try emit.operand(names, at);
                at += op.len();
            }
            try w.writeAll("\n\t// end inline asm\n");
        },
        .callprototype => {
            try w.print("\t$P__{d}: .callprototype ", .{@as(Mir.Operand, @bitCast(extra[data])).payload});
            try emit.operand(names, data + 1);
            try w.writeByte('_');
            try emit.operand(names, data + 3);
            try w.writeAll(";\n");
        },
        else => try emit.typedInstruction(names, inst),
    }
}

fn typedInstruction(emit: Emit, names: anytype, inst: Mir.Inst) !void {
    const mir = emit.mir;
    const w = emit.writer;
    const range = mir.operands(inst).?;
    var at = range.first;
    try w.writeByte('\t');
    switch (inst.modifiers.guard()) {
        .none => {},
        .positive, .negative => |guard| {
            try w.writeAll(if (guard == .positive) "@" else "@!");
            try writeReg(w, @bitCast(mir.extra[at]));
            try w.writeByte(' ');
            at += 1;
        },
    }
    try w.writeAll(@tagName(inst.tag));
    const modifiers = inst.modifiers;
    switch (inst.tag) {
        .setp => {
            try w.print(".{t}", .{modifiers.setp.cmp});
            if (modifiers.setp.ftz) try w.writeAll(".ftz");
            if (modifiers.setp.bool_op != .none) try w.print(".{t}", .{modifiers.setp.bool_op});
        },
        .cvt => {
            try writeRounding(w, modifiers.cvt.rounding);
            if (modifiers.cvt.ftz) try w.writeAll(".ftz");
            if (modifiers.cvt.sat) try w.writeAll(".sat");
            try w.print(".{t}.{t}", .{ inst.ty, modifiers.cvt.from });
        },
        .cvta => {
            std.debug.assert(!modifiers.cvta.uncommitted);
            if (modifiers.cvta.to) try w.writeAll(".to");
            try writeSpace(w, modifiers.cvta.space);
        },
        .ld, .st => {
            const mem = mir.memOperand(inst, 0);
            try writeSemScope(w, mem.sem, mem.scope);
            try writeSpace(w, mem.space);
            if (modifiers.mem.vec != .scalar) try w.print(".{t}", .{modifiers.mem.vec});
        },
        .atom => {
            const mem = mir.memOperand(inst, 0);
            try writeSemScope(w, mem.sem, mem.scope);
            try writeSpace(w, mem.space);
            try w.print(".{t}", .{modifiers.atom.op});
            if (modifiers.atom.noftz) try w.writeAll(".noftz");
        },
        .fence => try writeSemScope(w, modifiers.fence.sem, modifiers.fence.scope),
        .bra, .brkpt, .trap => {},
        else => {
            if (modifiers.arith.mode != .none) try w.print(".{t}", .{modifiers.arith.mode});
            if (modifiers.arith.cc) try w.writeAll(".cc");
            try writeRounding(w, modifiers.arith.rounding);
            if (modifiers.arith.ftz) try w.writeAll(".ftz");
            if (modifiers.arith.sat) try w.writeAll(".sat");
        },
    }
    if (inst.tag != .cvt and inst.ty != .none) try w.print(".{t}", .{inst.ty});
    const guard_count: u32 = @intFromBool(modifiers.guard() != .none);
    // The emitter writes neither the block arguments of a `bra` nor the else operand.
    const hidden: u32 = Mir.elseCount(inst) + @intFromBool(inst.tag == .bra and guard_count == 0);
    var operand_index: u32 = 0;
    while (operand_index + guard_count + hidden < range.count) : (operand_index += 1) {
        try w.writeAll(if (operand_index == 0) " " else ", ");
        if (operand_index == addressOperand(inst)) {
            try w.writeByte('[');
            try emit.typedOperand(names, at);
            at += @as(Mir.Operand, @bitCast(mir.extra[at])).len();
            if ((inst.tag == .ld or inst.tag == .st) and modifiers.mem.offset) {
                try w.writeByte('+');
                try emit.typedOperand(names, at);
                at += @as(Mir.Operand, @bitCast(mir.extra[at])).len();
                operand_index += 1;
            }
            try w.writeByte(']');
            continue;
        }
        try emit.typedOperand(names, at);
        at += @as(Mir.Operand, @bitCast(mir.extra[at])).len();
    }
    try w.writeAll(";\n");
}

/// Returns the index of the bracketed operand.
fn addressOperand(inst: Mir.Inst) ?u32 {
    return switch (inst.tag) {
        .st => 0,
        .ld, .atom => 1,
        else => null,
    };
}

fn writeRounding(w: *std.Io.Writer, rounding: Mir.Rounding) !void {
    if (rounding != .none) try w.print(".{t}", .{rounding});
}

fn writeSpace(w: *std.Io.Writer, space: Mir.Space) !void {
    if (space != .generic) try w.print(".{t}", .{space});
}

fn writeSemScope(w: *std.Io.Writer, sem: Mir.Sem, scope: Mir.Scope) !void {
    if (sem != .none) try w.print(".{t}", .{sem});
    if (scope != .none) try w.print(".{t}", .{scope});
}

fn typedOperand(emit: Emit, names: anytype, at: u32) !void {
    const mir = emit.mir;
    const w = emit.writer;
    const op: Mir.Operand = @bitCast(mir.extra[at]);
    switch (op.kind) {
        .local => try w.print("local_{d}", .{op.payload}),
        else => try emit.operand(names, at),
    }
}

fn operand(emit: Emit, names: anytype, at: u32) !void {
    const mir = emit.mir;
    const w = emit.writer;
    const extra = mir.extra;
    const op: Mir.Operand = @bitCast(extra[at]);
    switch (op.kind) {
        .reg => try writeReg(w, op),
        .label => try w.print("$L__{d}", .{op.payload}),
        .local, .imm => try w.print("{d}", .{op.payload}),
        .param => try writeParam(w, op),
        .symbol => {
            const sym = mir.symbols[op.payload];
            try names.writeName(w, sym.target);
            if (sym.offset != 0) try w.print("+{d}", .{sym.offset});
        },
        .imm64 => {
            const value = @as(u64, extra[at + 1]) | (@as(u64, extra[at + 2]) << 32);
            switch (@as(Mir.ImmFormat, @fromBackingInt(@intCast(op.payload)))) {
                .unsigned => try w.print("{d}", .{value}),
                .signed => try w.print("{d}", .{@as(i64, @bitCast(value))}),
                .f32 => try w.print("0f{X:0>8}", .{value}),
                .f64 => try w.print("0d{X:0>16}", .{value}),
                .prmt => try w.print("0x{X:0>4}", .{value}),
            }
        },
        .string => try w.writeAll(mir.strings[op.payload..][0..extra[at + 1]]),
        .list => try writeRegs(w, extra[at + 1 ..][0..op.payload]),
        .special_reg => try w.print("%{t}.{c}", .{
            @as(Mir.SpecialReg, @fromBackingInt(@intCast(op.payload >> 2))),
            "xyz"[@as(u2, @truncate(op.payload))],
        }),
    }
}

fn writeReg(w: *std.Io.Writer, op: Mir.Operand) !void {
    try w.print("{s}{d}", .{ op.regClass().prefix(), op.regIndex() });
}

fn writeRegs(w: *std.Io.Writer, words: []const u32) !void {
    if (words.len == 1) return writeReg(w, @bitCast(words[0]));
    try w.writeByte('{');
    for (words, 0..) |word, i| {
        if (i != 0) try w.writeAll(", ");
        try writeReg(w, @bitCast(word));
    }
    try w.writeByte('}');
}

fn writeParam(w: *std.Io.Writer, op: Mir.Operand) !void {
    const index = op.paramIndex();
    switch (op.paramRef()) {
        .own => try w.print("param_{d}", .{index}),
        .own_ret => try w.writeAll("func_retval0"),
        .arg => try w.print("param{d}", .{index}),
        .arg_ret => try w.writeAll("retval0"),
    }
}
