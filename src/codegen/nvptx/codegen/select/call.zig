//! This file lowers calls and inline assembly.

const std = @import("std");

const Air = @import("../../../../Air.zig");
const InternPool = @import("../../../../InternPool.zig");
const Type = @import("../../../../Type.zig");
const Value = @import("../../../../Value.zig");
const Mir = @import("../../Mir.zig");
const Layout = @import("../Layout.zig");
const Function = @import("../Function.zig");
const call = @import("../call.zig");
const assembly = @import("../assembly.zig");

const Val = Function.Val;

/// Returns null unless `callee` is comptime-known to point at a function.
fn directCallee(func: *Function, callee: Air.Inst.Ref) ?InternPool.Nav.Index {
    const ip = &func.zcu.intern_pool;
    return switch (ip.indexToKey(callee.toInterned() orelse return null)) {
        .func => |func_key| func_key.owner_nav,
        .@"extern" => |@"extern"| @"extern".owner_nav,
        .ptr => |ptr| switch (ptr.base_addr) {
            .nav => |nav| if (ptr.byte_offset == 0 and ip.isFunctionType(ip.getNav(nav).resolved.?.type)) nav else null,
            else => null,
        },
        else => null,
    };
}

pub fn airCall(func: *Function, inst: Air.Inst.Index) Layout.Error!?Val {
    const zcu = func.zcu;
    const unwrapped = func.air.unwrapCall(inst);
    const callee_type = func.typeOf(unwrapped.callee);
    const fn_type = if (callee_type.zigTypeTag(zcu) == .pointer) callee_type.childType(zcu) else callee_type;
    const callee: call.Callee = if (directCallee(func, unwrapped.callee)) |nav|
        .{ .target = .{ .nav = nav } }
    else
        .{ .reg = try func.builder.toReg((try func.partsOf(unwrapped.callee))[0], .pointer) };
    const never_inline = func.air.instructions.items(.tag)[@backingInt(inst)] == .call_never_inline;
    const fn_info = zcu.typeToFunc(fn_type).?;
    const ret_type: Type = .fromInterned(fn_info.return_type);
    return .{ .parts = try call.genCall(&func.builder, &func.layout, callee, ret_type, try func.callArgs(unwrapped.args, fn_info.cc), never_inline) };
}

pub fn airCompilerRtCall(func: *Function, inst: Air.Inst.Index) Layout.Error!Val {
    const unwrapped = func.air.unwrapCompilerRtCall(inst);
    return .{ .parts = try call.callCompilerRt(&func.builder, &func.layout, unwrapped.func, try func.callArgs(unwrapped.args, unwrapped.func.@"callconv"(func.layout.target))) };
}

pub fn airAsm(func: *Function, inst: Air.Inst.Index) Layout.Error!?Val {
    const zcu = func.zcu;
    const unwrapped = func.air.unwrapAsm(inst);
    var bindings: std.ArrayList(assembly.Binding) = .empty;
    var outputs: std.ArrayList(Mir.Reg) = .empty;
    var tied: std.ArrayList(?Mir.Reg) = .empty;
    const Writeback = struct { ref: Air.Inst.Ref, ty: Type, regs: []const Mir.Reg };
    var writebacks: std.ArrayList(Writeback) = .empty;
    var result: ?Val = null;

    var it = unwrapped.iterateOutputs();
    while (it.next()) |output| {
        const constraint = output.constraint;
        if (constraint.len < 2 or (constraint[0] != '=' and constraint[0] != '+')) {
            return func.layout.fail("invalid asm output constraint '{s}'", .{constraint});
        }
        const ty = if (output.operand == .none) func.typeOfIndex(inst) else func.typeOf(output.operand).childType(zcu);
        const scalars = try func.layout.scalarsOf(ty);
        try assembly.checkConstraint(&func.layout, constraint[1..], scalars);
        const classes = try func.arena.alloc(Mir.Class, scalars.len);
        for (classes, scalars) |*class, scalar| class.* = scalar.kind.class;
        const regs = try func.builder.newRegs(classes);
        if (output.operand == .none) {
            result = .{ .parts = try func.builder.regParts(regs) };
        } else {
            if (constraint[0] == '+') {
                const current = try func.loadThrough(output.operand, scalars);
                for (current, scalars) |part, scalar| try tied.append(func.arena, try func.builder.toReg(part, scalar.kind));
            }
            try writebacks.append(func.arena, .{ .ref = output.operand, .ty = ty, .regs = regs });
        }
        try outputs.appendSlice(func.arena, regs);
        while (tied.items.len < outputs.items.len) try tied.append(func.arena, null);
        try bindings.append(func.arena, .{ .name = output.name, .value = .{ .regs = regs } });
    }
    it = unwrapped.iterateInputs();
    while (it.next()) |input| {
        const ty = func.typeOf(input.operand);
        const scalars = try func.layout.scalarsOf(ty);
        const src = try func.partsOf(input.operand);
        // A constraint such as `rn` takes an immediate for a comptime-known input, else a register.
        const or_immediate = input.constraint.len == 2 and input.constraint[1] == 'n';
        const constraint = if (or_immediate) input.constraint[0..1] else input.constraint;
        const is_immediate = src.len == 1 and src[0] == .imm;
        if (std.mem.eql(u8, constraint, "n") or (or_immediate and is_immediate)) {
            if (!is_immediate) return func.layout.fail("asm input '{s}' with constraint 'n' must be comptime-known", .{input.name});
            try bindings.append(func.arena, .{ .name = input.name, .value = .{ .imm = src[0].imm } });
        } else {
            try assembly.checkConstraint(&func.layout, constraint, scalars);
            try bindings.append(func.arena, .{ .name = input.name, .value = .{ .regs = try func.builder.toRegs(src, scalars) } });
        }
    }
    var clobbers: Value.BigIntSpace = undefined;
    const pointer_inputs = for (unwrapped.inputs) |input| {
        if (func.typeOf(input).isPtrAtRuntime(zcu)) break true;
    } else false;
    try assembly.emit(&func.builder, &func.layout, unwrapped.source, outputs.items, tied.items, bindings.items, .{
        .is_volatile = unwrapped.is_volatile,
        .clobbers = !Value.fromInterned(unwrapped.clobbers).toBigInt(&clobbers, zcu).eqlZero(),
        .pointer_inputs = pointer_inputs,
    });
    for (writebacks.items) |writeback| try func.store(writeback.ref, try func.builder.regParts(writeback.regs), writeback.ty);
    return result orelse .{ .parts = &.{} };
}
