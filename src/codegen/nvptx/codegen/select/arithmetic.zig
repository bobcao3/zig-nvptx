//! This file lowers the AIR instructions for arithmetic, comparisons and conversions.

const std = @import("std");
const CompareOperator = std.math.CompareOperator;
const assert = std.debug.assert;

const Air = @import("../../../../Air.zig");
const Type = @import("../../../../Type.zig");
const Mir = @import("../../Mir.zig");
const Kind = @import("../Kind.zig");
const Layout = @import("../Layout.zig");
const Builder = @import("../Builder.zig");
const Function = @import("../Function.zig");
const int = @import("../int.zig");
const float = @import("../float.zig");
const wide = @import("../wide.zig");
const call = @import("../call.zig");

const Val = Function.Val;

pub fn airBinaryOp(func: *Function, inst: Air.Inst.Index, op: int.BinaryOp) Layout.Error!Val {
    if (func.layout.wideIntOf(func.typeOfIndex(inst))) |wide_int| return wideBinaryOp(func, inst, op, wide_int);
    const bin_op = func.dataOf(inst).bin_op;
    const lhs = try func.partsOf(bin_op.lhs);
    const rhs = try func.partsOf(bin_op.rhs);
    const lhs_scalars = try func.layout.scalarsOf(func.typeOf(bin_op.lhs));
    const rhs_scalars = try func.layout.scalarsOf(func.typeOf(bin_op.rhs));
    const out = try func.arena.alloc(Builder.Part, lhs.len);
    const is_shift = op == .shl or op == .shr or op == .shl_sat;
    const amount_limbs = if (is_shift and rhs.len > lhs.len) rhs.len / lhs.len else 1;
    for (out, lhs, lhs_scalars, 0..) |*result, lhs_operand, scalar, i| {
        const rhs_index = if (rhs.len == 1) 0 else i * amount_limbs;
        if (int.powerOfTwoDivisor(op, rhs[rhs_index], scalar.kind)) |exponent| {
            result.* = .{ .reg = try int.divPowerOfTwo(&func.builder, op, try func.builder.toReg(lhs_operand, scalar.kind), exponent, scalar.kind) };
            continue;
        }
        var left = lhs_operand;
        var right = rhs[rhs_index];
        if (int.isCommutative(op) and left == .imm and right != .imm) std.mem.swap(Builder.Part, &left, &right);
        const amount_kind = rhs_scalars[rhs_index].kind;
        const right_source: Mir.Source = if (amount_limbs > 1)
            .{ .reg = try int.shiftAmount(&func.builder, try wide.limitedShiftAmount(&func.builder, rhs[rhs_index..][0..amount_limbs]), amount_kind) }
        else if (is_shift and right != .imm)
            .{ .reg = try int.shiftAmount(&func.builder, try func.builder.toReg(right, amount_kind), amount_kind) }
        else if (is_shift)
            .{ .imm = .{ .value = right.imm, .format = .unsigned } }
        else
            try func.builder.toSource(right, amount_kind);
        result.* = .{ .reg = try func.binary(op, try func.builder.toReg(left, scalar.kind), right_source, scalar.kind, !func.isOptimized(inst)) };
    }
    return .{ .parts = out };
}

pub fn airOverflow(func: *Function, inst: Air.Inst.Index, op: int.OverflowOp) Layout.Error!Val {
    const bin = func.binFromPayload(inst);
    if (func.layout.wideIntOf(func.typeOf(bin.lhs))) |wide_int| return wideOverflow(func, inst, op, wide_int);
    const lhs = try func.partsOf(bin.lhs);
    const rhs = try func.partsOf(bin.rhs);
    const lhs_scalars = try func.layout.scalarsOf(func.typeOf(bin.lhs));
    const rhs_scalars = try func.layout.scalarsOf(func.typeOf(bin.rhs));
    const out = try func.arena.alloc(Builder.Part, 2 * lhs.len);
    for (lhs, rhs, lhs_scalars, rhs_scalars, 0..) |lhs_operand, rhs_operand, lhs_scalar, rhs_scalar, i| {
        var amount = try func.builder.toReg(rhs_operand, rhs_scalar.kind);
        if (op == .shl) amount = try int.shiftAmount(&func.builder, amount, rhs_scalar.kind);
        const wrapped, const overflowed = try int.overflow(&func.builder, op, try func.builder.toReg(lhs_operand, lhs_scalar.kind), amount, lhs_scalar.kind);
        out[i] = .{ .reg = wrapped };
        out[lhs.len + i] = .{ .reg = try func.builder.emit(.b16, .selp, .u16, .{ .none = .{} }, .{ 1, 0, overflowed }) };
    }
    return .{ .parts = out };
}

pub fn airUnaryOp(func: *Function, inst: Air.Inst.Index, op: int.UnaryOp) Layout.Error!Val {
    const data = func.dataOf(inst);
    const operand = switch (op) {
        .not, .abs, .popcount, .clz, .ctz, .bit_reverse, .byte_swap => data.ty_op.operand,
        else => data.un_op,
    };
    const ty = func.typeOf(operand);
    if (func.layout.wideIntOf(ty)) |wide_int| {
        return wideUnaryOp(func, op, try wide.limbRegs(&func.builder, try func.partsOf(operand), wide_int), wide_int, func.typeOfIndex(inst));
    }
    const result_scalars = try func.layout.scalarsOf(func.typeOfIndex(inst));
    const src = try func.partsOf(operand);
    const out = try func.arena.alloc(Builder.Part, src.len);
    for (out, src, try func.layout.scalarsOf(ty), result_scalars) |*result, operand_part, scalar, result_scalar| {
        result.* = .{ .reg = try func.unary(op, try func.builder.toReg(operand_part, scalar.kind), scalar.kind, result_scalar.kind) };
    }
    return .{ .parts = out };
}

pub fn airCompare(func: *Function, inst: Air.Inst.Index, op: CompareOperator) Layout.Error!Val {
    const bin_op = func.dataOf(inst).bin_op;
    return func.single(try compare(func, op, bin_op.lhs, bin_op.rhs, 0));
}

fn compare(func: *Function, op: CompareOperator, lhs: Air.Inst.Ref, rhs: Air.Inst.Ref, index: u32) Layout.Error!Mir.Reg {
    const zcu = func.zcu;
    const ty = func.typeOf(lhs);
    if (func.layout.wideIntOf(ty)) |wide_int| {
        return wide.compareLimbs(&func.builder, op, try wide.limbRegs(&func.builder, try func.partsOf(lhs), wide_int), try wide.limbRegs(&func.builder, try func.partsOf(rhs), wide_int), wide_int.signedness);
    }
    if (ty.zigTypeTag(zcu) == .optional) {
        const equal = try equalParts(func, ty, try func.partsOf(lhs), try func.partsOf(rhs));
        if (op == .eq) return equal;
        return func.builder.emit(.pred, .not, .pred, .{ .none = .{} }, .{equal});
    }
    const kind = (try func.layout.scalarsOf(ty))[index].kind;
    const lhs_reg = try func.builder.toReg((try func.partsOf(lhs))[index], kind);
    const rhs_type = func.typeOf(rhs);
    if (ty.zigTypeTag(zcu) == .pointer and rhs_type.zigTypeTag(zcu) == .pointer) {
        const rhs_kind = func.layout.ptrTypeKind(rhs_type);
        // Pointers into address spaces of different widths, as in the alias check of `@memcpy`,
        // compare as 64-bit addresses.
        if (rhs_kind.class != kind.class) {
            const rhs_reg = try func.builder.toReg((try func.partsOf(rhs))[index], rhs_kind);
            const a = try int.widen(&func.builder, lhs_reg, .pointer32);
            const b = try int.widen(&func.builder, rhs_reg, .pointer32);
            return func.compareSource(op, a, .{ .reg = b }, .pointer);
        }
    }
    return func.compareSource(op, lhs_reg, try func.builder.toSource((try func.partsOf(rhs))[index], kind), kind);
}

/// Two optionals are equal when both are null, or when both hold equal payloads. A null
/// optional has undefined payload scalars, and the comparison ignores them.
fn equalParts(func: *Function, ty: Type, lhs: []const Builder.Part, rhs: []const Builder.Part) Layout.Error!Mir.Reg {
    const zcu = func.zcu;
    const scalars = try func.layout.scalarsOf(ty);
    if (func.layout.wideIntOf(ty)) |wide_int| {
        return wide.compareLimbs(&func.builder, .eq, try wide.limbRegs(&func.builder, lhs, wide_int), try wide.limbRegs(&func.builder, rhs, wide_int), wide_int.signedness);
    }
    if (ty.zigTypeTag(zcu) != .optional or ty.optionalReprIsPayload(zcu)) {
        if (scalars.len != 1) return func.layout.todo("comparison of '{f}'", .{ty.fmt(zcu)});
        return func.compareSource(.eq, try func.builder.toReg(lhs[0], scalars[0].kind), .{ .reg = try func.builder.toReg(rhs[0], scalars[0].kind) }, scalars[0].kind);
    }
    const last = scalars.len - 1;
    const flag_kind = scalars[last].kind;
    const lhs_flag = try func.builder.toReg(lhs[last], flag_kind);
    const flags_equal = try func.compareSource(.eq, lhs_flag, .{ .reg = try func.builder.toReg(rhs[last], flag_kind) }, flag_kind);
    if (scalars.len == 1) return flags_equal;
    const payloads_equal = try equalParts(func, ty.optionalChild(zcu), lhs[0..last], rhs[0..last]);
    const lhs_null = try func.builder.emit(.pred, .not, .pred, .{ .none = .{} }, .{lhs_flag});
    const null_or_equal = try func.builder.emit(.pred, .@"or", .pred, .{ .none = .{} }, .{ lhs_null, payloads_equal });
    return func.builder.emit(.pred, .@"and", .pred, .{ .none = .{} }, .{ flags_equal, null_or_equal });
}

pub fn airCompareVector(func: *Function, inst: Air.Inst.Index) Layout.Error!Val {
    const extra = func.air.extraData(Air.VectorCmp, func.dataOf(inst).ty_pl.payload).data;
    const len = func.typeOf(extra.lhs).vectorLen(func.zcu);
    const out = try func.arena.alloc(Builder.Part, len);
    for (out, 0..) |*result, i| result.* = .{ .reg = try compare(func, extra.compareOperator(), extra.lhs, extra.rhs, @intCast(i)) };
    return .{ .parts = out };
}

pub fn airSelect(func: *Function, inst: Air.Inst.Index) Layout.Error!Val {
    const pl_op = func.dataOf(inst).pl_op;
    const bin = func.air.extraData(Air.Bin, pl_op.payload).data;
    const preds = try func.partsOf(pl_op.operand);
    const lhs = try func.partsOf(bin.lhs);
    const rhs = try func.partsOf(bin.rhs);
    const scalars = try func.layout.scalarsOf(func.typeOfIndex(inst));
    const out = try func.arena.alloc(Builder.Part, scalars.len);
    for (out, lhs, rhs, scalars, 0..) |*result, lhs_part, rhs_part, scalar, i| {
        const pred = try func.builder.toReg(preds[if (preds.len == 1) 0 else i], .bool);
        result.* = .{ .reg = try int.select(&func.builder, pred, try func.builder.toReg(lhs_part, scalar.kind), try func.builder.toReg(rhs_part, scalar.kind), scalar.kind) };
    }
    return .{ .parts = out };
}

/// The language reference defines `@reduce` as a sequential reduction that starts with the first
/// element. `Legalize` would start with an identity value, which gives +inf for a `.Min` of NaN
/// elements, so the backend lowers `@reduce` itself. Only the vectors of `f80` and `f128` reach
/// `Legalize`, which calls compiler_rt for each of their elements. Integer operations, float `.Min`
/// and `.Max` (PTX ISA 9.4, section 9.7.3.11) and the float operations of `reduce_optimized` are
/// associative, so they combine pairs of elements as a tree.
pub fn airReduce(func: *Function, inst: Air.Inst.Index, optimized: bool) Layout.Error!Val {
    const zcu = func.zcu;
    const reduce = func.dataOf(inst).reduce;
    const vector_type = func.typeOf(reduce.operand);
    const elem_type = vector_type.childType(zcu);
    const src = try func.partsOf(reduce.operand);
    assert(src.len != 0);
    const op: int.BinaryOp = switch (reduce.operation) {
        .And => .@"and",
        .Or => .@"or",
        .Xor => .xor,
        .Min => .min,
        .Max => .max,
        .Add => .add,
        .Mul => .mul,
    };
    const len = vector_type.vectorLen(zcu);
    var elems = try func.arena.alloc([]const Builder.Part, len);
    for (elems, 0..) |*elem, i| elem.* = src[i * (src.len / len) ..][0 .. src.len / len];
    const sequential = elem_type.isRuntimeFloat() and !optimized and (op == .add or op == .mul);
    if (sequential) {
        var accumulator = elems[0];
        for (elems[1..]) |elem| accumulator = try reduceStep(func, op, elem_type, accumulator, elem, optimized);
        return .{ .parts = accumulator };
    }
    while (elems.len > 1) {
        const pairs = elems.len / 2;
        for (0..pairs) |i| elems[i] = try reduceStep(func, op, elem_type, elems[2 * i], elems[2 * i + 1], optimized);
        if (elems.len % 2 == 1) elems[pairs] = elems[elems.len - 1];
        elems = elems[0 .. elems.len - pairs];
    }
    return .{ .parts = elems[0] };
}

fn reduceStep(func: *Function, op: int.BinaryOp, elem_type: Type, a: []const Builder.Part, b: []const Builder.Part, optimized: bool) Layout.Error![]const Builder.Part {
    if (func.layout.wideIntOf(elem_type)) |wide_int| {
        const x = try wide.limbRegs(&func.builder, a, wide_int);
        const y = try wide.limbRegs(&func.builder, b, wide_int);
        return func.builder.regParts(try wideLimbOp(func, op, x, y, wide_int));
    }
    const kind = try func.layout.kindOf(elem_type);
    const result = try func.binary(op, try func.builder.toReg(a[0], kind), .{ .reg = try func.builder.toReg(b[0], kind) }, kind, !optimized);
    return func.builder.regParts(&.{result});
}

pub fn airIntCast(func: *Function, inst: Air.Inst.Index) Layout.Error!Val {
    const ty_op = func.dataOf(inst).ty_op;
    const src_type = func.typeOf(ty_op.operand);
    if (func.layout.wideIntOf(src_type) != null or func.layout.wideIntOf(ty_op.ty) != null) return wideIntCast(func, src_type, ty_op.ty, try func.partsOf(ty_op.operand));
    const src_scalars = try func.layout.scalarsOf(src_type);
    const dest_scalars = try func.layout.scalarsOf(ty_op.ty);
    const src = try func.partsOf(ty_op.operand);
    const out = try func.arena.alloc(Builder.Part, dest_scalars.len);
    for (out, dest_scalars, 0..) |*result, dest_scalar, i| {
        const source = if (src.len == dest_scalars.len) i else 0;
        result.* = .{ .reg = try int.intConvert(&func.builder, try func.builder.toReg(src[source], src_scalars[source].kind), src_scalars[source].kind, dest_scalar.kind) };
    }
    return .{ .parts = out };
}

pub fn airMulAdd(func: *Function, inst: Air.Inst.Index) Layout.Error!Val {
    const pl_op = func.dataOf(inst).pl_op;
    const bin = func.air.extraData(Air.Bin, pl_op.payload).data;
    const scalars = try func.layout.scalarsOf(func.typeOfIndex(inst));
    const lhs = try func.partsOf(bin.lhs);
    const rhs = try func.partsOf(bin.rhs);
    const addend = try func.partsOf(pl_op.operand);
    const out = try func.arena.alloc(Builder.Part, scalars.len);
    for (out, lhs, rhs, addend, scalars) |*result, lhs_part, rhs_part, addend_part, scalar| {
        const kind = scalar.kind;
        const args = .{ try func.builder.toReg(lhs_part, kind), try func.builder.toReg(rhs_part, kind), try func.builder.toReg(addend_part, kind) };
        result.* = .{ .reg = try func.builder.emit(kind.class, .fma, kind.arithmeticType(), .{ .arith = .{ .rounding = .rn } }, args) };
    }
    return .{ .parts = out };
}

const Conversion = enum { float, from_int, to_int };

pub fn airConvert(func: *Function, inst: Air.Inst.Index, op: Conversion) Layout.Error!Val {
    const zcu = func.zcu;
    const ty_op = func.dataOf(inst).ty_op;
    const src_type = func.typeOf(ty_op.operand);
    if (func.layout.wideIntOf(ty_op.ty.scalarType(zcu)) != null or func.layout.wideIntOf(src_type.scalarType(zcu)) != null) {
        return wideConvert(func, op, src_type, ty_op.ty, try func.partsOf(ty_op.operand));
    }
    const src_scalars = try func.layout.scalarsOf(src_type);
    const dest_scalars = try func.layout.scalarsOf(ty_op.ty);
    const src = try func.partsOf(ty_op.operand);
    const out = try func.arena.alloc(Builder.Part, dest_scalars.len);
    for (out, src, src_scalars, dest_scalars) |*result, operand, from, to| {
        const rounding: Mir.Rounding = switch (op) {
            .to_int => .rzi,
            .from_int => .rn,
            .float => if (to.kind.bits < from.kind.bits) .rn else .none,
        };
        const modifiers: Mir.Inst.Modifiers = .{ .cvt = .{ .rounding = rounding, .from = from.kind.arithmeticType() } };
        const converted = try func.builder.emit(to.kind.class, .cvt, to.kind.arithmeticType(), modifiers, .{try func.builder.toReg(operand, from.kind)});
        result.* = .{ .reg = if (op == .to_int) try int.normalized(&func.builder, converted, to.kind) else converted };
    }
    return .{ .parts = out };
}

/// Converts between a float and an integer wider than 64 bits with a 128-bit compiler_rt function.
/// `Legalize` turns f80 and f128 conversions into calls, so the float has at most 64 bits.
fn wideConvert(func: *Function, op: Conversion, src_type: Type, dest_type: Type, src: []const Builder.Part) Layout.Error!Val {
    const int_type, const float_type = if (op == .from_int) .{ src_type, dest_type } else .{ dest_type, src_type };
    const info = int_type.intInfo(func.zcu);
    if (info.bits > 128) return func.layout.todo("conversion between floats and integers wider than 128 bits", .{});
    const float_index: u2 = switch (float_type.floatBits(func.layout.target)) {
        16 => 0,
        32 => 1,
        64 => 2,
        else => unreachable,
    };
    const wide_type: Type = if (info.signedness == .signed) .i128 else .u128;
    switch (op) {
        .from_int => {
            const functions: [3]Air.CompilerRtFunc = switch (info.signedness) {
                .signed => .{ .__floattihf, .__floattisf, .__floattidf },
                .unsigned => .{ .__floatuntihf, .__floatuntisf, .__floatuntidf },
            };
            const widened = try wideIntCast(func, int_type, wide_type, src);
            return .{ .parts = try call.callCompilerRt(&func.builder, &func.layout, functions[float_index], &.{.{ .ty = wide_type, .parts = widened.parts }}) };
        },
        .to_int => {
            const functions: [3]Air.CompilerRtFunc = switch (info.signedness) {
                .signed => .{ .__fixhfti, .__fixsfti, .__fixdfti },
                .unsigned => .{ .__fixunshfti, .__fixunssfti, .__fixunsdfti },
            };
            const result = try call.callCompilerRt(&func.builder, &func.layout, functions[float_index], &.{.{ .ty = float_type, .parts = src }});
            return wideIntCast(func, wide_type, dest_type, result);
        },
        .float => unreachable,
    }
}

/// Calls the compiler_rt function of a math builtin, once for each element of a vector. `Legalize`
/// scalarizes the vectors wider than 16 bytes and replaces the `f80` and `f128` forms with
/// compiler_rt calls.
pub fn airMathCall(func: *Function, inst: Air.Inst.Index, tag: Air.Inst.Tag) Layout.Error!Val {
    const zcu = func.zcu;
    const operand = func.dataOf(inst).un_op;
    const operand_type = func.typeOf(operand);
    const float_type = operand_type.scalarType(zcu);
    const float_index: u2 = switch (float_type.floatBits(func.layout.target)) {
        16 => 0,
        32 => 1,
        64 => 2,
        else => unreachable,
    };
    const functions: [3]Air.CompilerRtFunc = switch (tag) {
        .sin => .{ .__sinh, .sinf, .sin },
        .cos => .{ .__cosh, .cosf, .cos },
        .tan => .{ .__tanh, .tanf, .tan },
        .exp => .{ .__exph, .expf, .exp },
        .exp2 => .{ .__exp2h, .exp2f, .exp2 },
        .log => .{ .__logh, .logf, .log },
        .log2 => .{ .__log2h, .log2f, .log2 },
        .log10 => .{ .__log10h, .log10f, .log10 },
        else => unreachable,
    };
    const function = functions[float_index];
    if (!operand_type.isVector(zcu)) {
        return .{ .parts = try call.callCompilerRt(&func.builder, &func.layout, function, try func.callArgs(&.{operand}, function.@"callconv"(func.layout.target))) };
    }
    const src = try func.partsOf(operand);
    assert(src.len == operand_type.vectorLen(zcu));
    const parts = try func.arena.alloc(Builder.Part, src.len);
    for (parts, src) |*part, element| {
        part.* = (try call.callCompilerRt(&func.builder, &func.layout, function, &.{.{ .ty = float_type, .parts = &.{element} }}))[0];
    }
    return .{ .parts = parts };
}

/// An amount of `limbs * 64` or more moves all bits out, so the amount saturates there.
fn shiftAmountOf(func: *Function, ref: Air.Inst.Ref, limbs: u32) Layout.Error!wide.ShiftAmount {
    const limit = @as(u64, limbs) * 64;
    const operands = try func.partsOf(ref);
    known: {
        var value: u64 = 0;
        for (operands, 0..) |operand, i| switch (operand) {
            .imm => |imm| if (i == 0) {
                value = imm;
            } else if (imm != 0) {
                value = limit;
            },
            else => break :known,
        };
        return .{ .known = @min(value, limit) };
    }
    const kind = (try func.layout.scalarsOf(func.typeOf(ref)))[0].kind;
    const amount = if (operands.len > 1) try wide.limitedShiftAmount(&func.builder, operands) else try func.builder.toReg(operands[0], kind);
    const type_max = if (operands.len > 1 or kind.bits == 64) std.math.maxInt(u64) else (@as(u64, 1) << @intCast(kind.bits)) - 1;
    const reg = if (amount.class == .b64) reg: {
        const limited = try func.builder.emit(.b64, .min, .u64, .{ .none = .{} }, .{ amount, limit });
        break :reg try func.builder.emit(.b32, .cvt, .u32, .{ .cvt = .{ .from = .u64 } }, .{limited});
    } else reg: {
        const converted = try func.builder.emit(.b32, .cvt, .u32, .{ .cvt = .{ .from = kind.arithmeticType() } }, .{amount});
        if (type_max <= limit) break :reg converted;
        break :reg try func.builder.emit(.b32, .min, .u32, .{ .none = .{} }, .{ converted, limit });
    };
    return .{ .reg = .{ .reg = reg, .max = @min(type_max, limit) } };
}

fn wideBinaryOp(func: *Function, inst: Air.Inst.Index, op: int.BinaryOp, wide_int: Layout.WideInt) Layout.Error!Val {
    const bin_op = func.dataOf(inst).bin_op;
    const a = try wide.limbRegs(&func.builder, try func.partsOf(bin_op.lhs), wide_int);
    const result = switch (op) {
        .shl => try wide.normalizedLimbs(&func.builder, try wide.shiftLimbs(&func.builder, .left, a, try shiftAmountOf(func, bin_op.rhs, wide_int.limbs), wide_int.signedness), wide_int),
        .shr => try wide.shiftLimbs(&func.builder, .right, a, try shiftAmountOf(func, bin_op.rhs, wide_int.limbs), wide_int.signedness),
        .add_sat => try saturateLimbs(func, .add, a, bin_op.rhs, wide_int),
        .sub_sat => try saturateLimbs(func, .sub, a, bin_op.rhs, wide_int),
        .mul_sat => try saturateLimbs(func, .mul, a, bin_op.rhs, wide_int),
        .shl_sat => try saturateLimbs(func, .shl, a, bin_op.rhs, wide_int),
        .div, .div_trunc, .div_floor, .rem, .mod => result: {
            const divisor = try func.partsOf(bin_op.rhs);
            const divided = if (wide.powerOfTwoLimbs(divisor, wide_int)) |exponent|
                try wide.divPowerOfTwoLimbs(&func.builder, op, a, exponent, wide_int)
            else
                try wide.divideLimbs(&func.builder, op, a, try wide.limbRegs(&func.builder, divisor, wide_int), wide_int);
            break :result try wide.normalizedLimbs(&func.builder, divided, wide_int);
        },
        else => try wideLimbOp(func, op, a, try wide.limbRegs(&func.builder, try func.partsOf(bin_op.rhs), wide_int), wide_int),
    };
    return .{ .parts = try func.builder.regParts(result) };
}

fn wideLimbOp(func: *Function, op: int.BinaryOp, a: []const Mir.Reg, b: []const Mir.Reg, wide_int: Layout.WideInt) Layout.Error![]const Mir.Reg {
    return switch (op) {
        .add => try wide.normalizedLimbs(&func.builder, try wide.carryChain(&func.builder, .add, a, b), wide_int),
        .sub => try wide.normalizedLimbs(&func.builder, try wide.carryChain(&func.builder, .sub, a, b), wide_int),
        .mul => try wide.normalizedLimbs(&func.builder, try wide.multiplyLimbs(&func.builder, a, b, a.len), wide_int),
        .@"and", .@"or", .xor => bitwise: {
            const out = try func.arena.alloc(Mir.Reg, a.len);
            for (out, a, b) |*limb, a_limb, b_limb| limb.* = try func.builder.emit(.b64, int.instructionTag(op), .b64, .{ .none = .{} }, .{ a_limb, b_limb });
            break :bitwise out;
        },
        .min => try wide.selectLimbs(&func.builder, try wide.compareLimbs(&func.builder, .lt, a, b, wide_int.signedness), a, b),
        .max => try wide.selectLimbs(&func.builder, try wide.compareLimbs(&func.builder, .lt, a, b, wide_int.signedness), b, a),
        else => unreachable,
    };
}

fn overflowLimbs(func: *Function, op: int.OverflowOp, a: []const Mir.Reg, rhs: Air.Inst.Ref, wide_int: Layout.WideInt) Layout.Error!struct { []const Mir.Reg, Mir.Reg } {
    if (op == .shl) {
        // The shift overflows when shifting the wrapped result back does not give `a`.
        const amount = try shiftAmountOf(func, rhs, wide_int.limbs);
        const wrapped = try wide.normalizedLimbs(&func.builder, try wide.shiftLimbs(&func.builder, .left, a, amount, wide_int.signedness), wide_int);
        const back = try wide.shiftLimbs(&func.builder, .right, wrapped, amount, wide_int.signedness);
        return .{ wrapped, try wide.compareLimbs(&func.builder, .neq, back, a, .unsigned) };
    }
    // The exact result fits in the limbs of `exact`. The operation overflows exactly when the
    // exact result differs from the wrapped result.
    const b = try wide.limbRegs(&func.builder, try func.partsOf(rhs), wide_int);
    const exact = switch (op) {
        .add, .sub => try wide.carryChain(
            &func.builder,
            if (op == .add) .add else .sub,
            try wide.extendLimbs(&func.builder, a, wide_int.signedness, a.len + 1),
            try wide.extendLimbs(&func.builder, b, wide_int.signedness, b.len + 1),
        ),
        .mul => try wide.productLimbs(&func.builder, a, b, wide_int.signedness),
        .shl => unreachable,
    };
    const wrapped = try wide.normalizedLimbs(&func.builder, exact[0..a.len], wide_int);
    const extended = try wide.extendLimbs(&func.builder, wrapped, wide_int.signedness, exact.len);
    const top = wide_int.topLimb();
    return .{ wrapped, try wide.compareLimbs(&func.builder, .neq, exact[top..], extended[top..], .unsigned) };
}

fn wideOverflow(func: *Function, inst: Air.Inst.Index, op: int.OverflowOp, wide_int: Layout.WideInt) Layout.Error!Val {
    const bin = func.binFromPayload(inst);
    const wrapped, const overflowed = try overflowLimbs(func, op, try wide.limbRegs(&func.builder, try func.partsOf(bin.lhs), wide_int), bin.rhs, wide_int);
    const out = try func.arena.alloc(Builder.Part, wrapped.len + 1);
    for (out[0..wrapped.len], wrapped) |*result, reg| result.* = .{ .reg = reg };
    out[wrapped.len] = .{ .reg = try func.builder.emit(.b16, .selp, .u16, .{ .none = .{} }, .{ 1, 0, overflowed }) };
    return .{ .parts = out };
}

fn saturateLimbs(func: *Function, op: int.OverflowOp, a: []const Mir.Reg, rhs: Air.Inst.Ref, wide_int: Layout.WideInt) Layout.Error![]const Mir.Reg {
    const wrapped, const overflowed = try overflowLimbs(func, op, a, rhs, wide_int);
    const out = try func.arena.alloc(Mir.Reg, a.len);
    if (wide_int.signedness == .unsigned) {
        for (out, wrapped, 0..) |*result, limb, i| {
            const limit = if (op == .sub) 0 else wide_int.limitLimb(@intCast(i), .max);
            result.* = try func.builder.emit(.b64, .selp, .b64, .{ .none = .{} }, .{ limit, limb, overflowed });
        }
        return out;
    }
    const sign = if (op == .mul) sign: {
        const b = try wide.limbRegs(&func.builder, try func.partsOf(rhs), wide_int);
        break :sign try func.builder.emit(.b64, .xor, .b64, .{ .none = .{} }, .{ a[a.len - 1], b[b.len - 1] });
    } else a[a.len - 1];
    const negative = try func.builder.emit(.pred, .setp, .s64, .{ .setp = .{ .cmp = .lt } }, .{ sign, 0 });
    for (out, wrapped, 0..) |*result, limb, i| {
        const limits = .{ wide_int.limitLimb(@intCast(i), .min), wide_int.limitLimb(@intCast(i), .max), negative };
        const limit = try func.builder.emit(.b64, .selp, .b64, .{ .none = .{} }, limits);
        result.* = try func.builder.emit(.b64, .selp, .b64, .{ .none = .{} }, .{ limit, limb, overflowed });
    }
    return out;
}

fn wideUnaryOp(func: *Function, op: int.UnaryOp, a: []const Mir.Reg, wide_int: Layout.WideInt, result_type: Type) Layout.Error!Val {
    const result = switch (op) {
        .not => not: {
            const out = try func.arena.alloc(Mir.Reg, a.len);
            for (out, a) |*limb, a_limb| limb.* = try func.builder.emit(.b64, .not, .b64, .{ .none = .{} }, .{a_limb});
            break :not try wide.normalizedLimbs(&func.builder, out, wide_int);
        },
        .neg => neg: {
            const zeros = try func.arena.alloc(Mir.Reg, a.len);
            @memset(zeros, try func.builder.movImm(.b64, 0));
            break :neg try wide.normalizedLimbs(&func.builder, try wide.carryChain(&func.builder, .sub, zeros, a), wide_int);
        },
        .abs => if (wide_int.signedness == .unsigned)
            a
        else
            try wide.normalizedLimbs(&func.builder, try wide.negateIf(&func.builder, a, try wide.fillLimb(&func.builder, a[a.len - 1], .signed)), func.layout.wideIntOf(result_type).?),
        .popcount, .clz, .ctz => return func.single(try wide.countBits(&func.builder, op, a, wide_int, (try func.layout.scalarsOf(result_type))[0].kind)),
        .byte_swap, .bit_reverse => try wide.reverseLimbs(&func.builder, op, a, wide_int),
        else => unreachable,
    };
    return .{ .parts = try func.builder.regParts(result) };
}

fn wideIntCast(func: *Function, src_type: Type, dest_type: Type, src: []const Builder.Part) Layout.Error!Val {
    const dest_scalars = try func.layout.scalarsOf(dest_type);
    assert(dest_scalars.len != 0);
    var low: []const Mir.Reg = undefined;
    var signedness: std.lang.Signedness = .unsigned;
    if (func.layout.wideIntOf(src_type)) |src_int| {
        low = try wide.limbRegs(&func.builder, src, src_int);
        signedness = src_int.signedness;
    } else if (src.len == 0) {
        low = try func.arena.dupe(Mir.Reg, &.{try func.builder.movImm(.b64, 0)});
    } else {
        const kind = (try func.layout.scalarsOf(src_type))[0].kind;
        if (kind.repr == .signed) signedness = .signed;
        low = try func.arena.dupe(Mir.Reg, &.{try int.intConvert(&func.builder, try func.builder.toReg(src[0], kind), kind, Kind.int(kind.repr, 64))});
    }
    const dst_int = func.layout.wideIntOf(dest_type) orelse return func.single(try int.intConvert(&func.builder, low[0], Kind.int(.unsigned, 64), dest_scalars[0].kind));
    const kept = low[0..@min(low.len, dst_int.limbs)];
    return .{ .parts = try func.builder.regParts(try wide.normalizedLimbs(&func.builder, try wide.extendLimbs(&func.builder, kept, signedness, dst_int.limbs), dst_int)) };
}
