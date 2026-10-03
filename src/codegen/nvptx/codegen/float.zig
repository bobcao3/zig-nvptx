//! This file lowers float operations on registers. Arithmetic on `f16` computes in `f32`.

const std = @import("std");
const CompareOperator = std.math.CompareOperator;

const Mir = @import("../Mir.zig");
const Kind = @import("Kind.zig");
const Builder = @import("Builder.zig");
const Layout = @import("Layout.zig");
const int = @import("int.zig");

pub fn toF32(builder: *Builder, reg: Mir.Reg) Layout.Error!Mir.Reg {
    return builder.emit(.f32, .cvt, .f32, .{ .cvt = .{ .from = .f16 } }, .{reg});
}

fn toF16(builder: *Builder, reg: Mir.Reg) Layout.Error!Mir.Reg {
    return builder.emit(.b16, .cvt, .f16, .{ .cvt = .{ .rounding = .rn, .from = .f32 } }, .{reg});
}

fn floatImm(class: Mir.Class, bits_f32: u32, bits_f64: u64) Mir.Imm {
    return if (class == .f32) .{ .value = bits_f32, .format = .f32 } else .{ .value = bits_f64, .format = .f64 };
}

pub fn binary(builder: *Builder, op: int.BinaryOp, lhs: Mir.Reg, rhs: Mir.Source, kind: Kind, strict: bool) Layout.Error!Mir.Reg {
    if (op == .rem or op == .mod) return remainder(builder, op, lhs, try builder.sourceReg(rhs, kind), kind);
    if (kind.class == .b16) {
        if (op == .div_trunc or op == .div_floor) {
            const quotient = try binary(builder, .div, lhs, rhs, kind, strict);
            return unary(builder, if (op == .div_trunc) .trunc else .floor, quotient, kind);
        }
        const lhs_wide = try toF32(builder, lhs);
        const rhs_wide = try toF32(builder, try builder.sourceReg(rhs, kind));
        return toF16(builder, try binary(builder, op, lhs_wide, .{ .reg = rhs_wide }, .f32, strict));
    }
    const ptx_type = kind.arithmeticType();
    // In the optimized float mode, `f32` arithmetic flushes subnormal values to zero, and division
    // uses `div.approx` (PTX ISA 9.4, sections 9.7.3.3 and 9.7.3.8).
    const fast = !strict and kind.class == .f32;
    const rounding: Mir.Rounding = if (strict) .rn else .none;
    switch (op) {
        .add, .sub, .mul => return builder.emit(kind.class, int.instructionTag(op), ptx_type, .{ .arith = .{ .rounding = rounding, .ftz = fast } }, .{ lhs, rhs }),
        .div => {
            const modifiers: Mir.Inst.Modifiers = .{ .arith = if (fast) .{ .rounding = .approx, .ftz = true } else .{ .rounding = .rn } };
            return builder.emit(kind.class, .div, ptx_type, modifiers, .{ lhs, rhs });
        },
        .min, .max => return builder.emit(kind.class, int.instructionTag(op), ptx_type, .{ .none = .{} }, .{ lhs, rhs }),
        .div_trunc, .div_floor => {
            const quotient = try builder.emit(kind.class, .div, ptx_type, .{ .arith = .{ .rounding = .rn } }, .{ lhs, rhs });
            const integer_rounding: Mir.Rounding = if (op == .div_trunc) .rzi else .rmi;
            return builder.emit(kind.class, .cvt, ptx_type, .{ .cvt = .{ .rounding = integer_rounding, .from = ptx_type } }, .{quotient});
        },
        else => unreachable,
    }
}

pub fn unary(builder: *Builder, op: int.UnaryOp, src: Mir.Reg, kind: Kind) Layout.Error!Mir.Reg {
    if (op == .neg or op == .abs) {
        // `abs.f32` and `neg.f32` can change a NaN (PTX ISA 9.4, sections 9.7.3.9 and 9.7.3.10).
        const width = kind.class.bits();
        const sign = @as(u64, 1) << @intCast(width - 1);
        const integer_class = Kind.int(.unsigned, width).class;
        const bit_type = integer_class.bitType();
        const bits = if (integer_class == kind.class) src else try builder.emit(integer_class, .mov, bit_type, .{ .none = .{} }, .{src});
        const changed = if (op == .neg)
            try builder.emit(integer_class, .xor, bit_type, .{ .none = .{} }, .{ bits, sign })
        else
            try builder.emit(integer_class, .@"and", bit_type, .{ .none = .{} }, .{ bits, sign - 1 });
        if (integer_class == kind.class) return changed;
        return builder.emit(kind.class, .mov, bit_type, .{ .none = .{} }, .{changed});
    }
    if (kind.class == .b16) return toF16(builder, try unary(builder, op, try toF32(builder, src), .f32));
    const ptx_type = kind.arithmeticType();
    const integer_rounding: Mir.Rounding = switch (op) {
        .sqrt => return builder.emit(kind.class, .sqrt, ptx_type, .{ .arith = .{ .rounding = .rn } }, .{src}),
        .round => return round(builder, src, kind),
        .floor => .rmi,
        .ceil => .rpi,
        .trunc => .rzi,
        else => unreachable,
    };
    return builder.emit(kind.class, .cvt, ptx_type, .{ .cvt = .{ .rounding = integer_rounding, .from = ptx_type } }, .{src});
}

/// No rounding mode of `cvt` rounds half away from zero. The function therefore steps from
/// `trunc(src)` away from zero when the dropped fraction is at least one half.
fn round(builder: *Builder, src: Mir.Reg, kind: Kind) Layout.Error!Mir.Reg {
    const class = kind.class;
    const ptx_type = kind.arithmeticType();
    const truncated = try builder.emit(class, .cvt, ptx_type, .{ .cvt = .{ .rounding = .rzi, .from = ptx_type } }, .{src});
    const fraction = try builder.emit(class, .sub, ptx_type, .{ .arith = .{ .rounding = .rn } }, .{ src, truncated });
    const magnitude = try builder.emit(class, .abs, ptx_type, .{ .none = .{} }, .{fraction});
    const at_least_half = try builder.emit(.pred, .setp, ptx_type, .{ .setp = .{ .cmp = .ge } }, .{ magnitude, floatImm(class, 0x3F000000, 0x3FE0000000000000) });
    const negative = try builder.emit(.pred, .setp, ptx_type, .{ .setp = .{ .cmp = .lt } }, .{ src, floatImm(class, 0, 0) });
    const step = try builder.emit(class, .selp, ptx_type, .{ .none = .{} }, .{ floatImm(class, 0xBF800000, 0xBFF0000000000000), floatImm(class, 0x3F800000, 0x3FF0000000000000), negative });
    const stepped = try builder.emit(class, .add, ptx_type, .{ .arith = .{ .rounding = .rn } }, .{ truncated, step });
    return builder.emit(class, .selp, ptx_type, .{ .none = .{} }, .{ stepped, truncated, at_least_half });
}

/// PTX `rem` takes only integer types, so `fmod` computes a remainder with the sign of `lhs`. For
/// a negative `lhs`, `@mod` runs `fmod` again on the remainder plus `rhs`.
fn remainder(builder: *Builder, op: int.BinaryOp, lhs: Mir.Reg, rhs: Mir.Reg, kind: Kind) Layout.Error!Mir.Reg {
    const is_half = kind.class == .b16;
    const compute_kind: Kind = if (is_half) .f32 else kind;
    const dividend = if (is_half) try toF32(builder, lhs) else lhs;
    const divisor = if (is_half) try toF32(builder, rhs) else rhs;
    if (op == .rem) {
        const result = try fmod(builder, dividend, divisor, compute_kind);
        return if (is_half) toF16(builder, result) else result;
    }
    const ptx_type = compute_kind.arithmeticType();
    const negative = try builder.emit(.pred, .setp, ptx_type, .{ .setp = .{ .cmp = .lt } }, .{ dividend, floatImm(compute_kind.class, 0, 0) });
    const head = try builder.newBlock(&.{ compute_kind.class, .pred });
    const done = builder.labelBlock();
    try builder.branch(head, &.{ dividend, negative });
    const params = try builder.startBlock(head);
    const result = try fmod(builder, params[0], divisor, compute_kind);
    try builder.branchIf(params[1], true, done, &.{});
    const sum = try builder.emit(compute_kind.class, .add, ptx_type, .{ .arith = .{ .rounding = .rn } }, .{ result, divisor });
    // `@mod` adds in the operand type, so an `f16` sum rounds to `f16`.
    const next = if (is_half) try toF32(builder, try toF16(builder, sum)) else sum;
    try builder.branch(head, &.{ next, try builder.movImm(.pred, 0) });
    _ = try builder.startBlock(done);
    return if (is_half) toF16(builder, result) else result;
}

/// Returns C `fmod(dividend, divisor)`.
fn fmod(builder: *Builder, dividend: Mir.Reg, divisor: Mir.Reg, kind: Kind) Layout.Error!Mir.Reg {
    const width = kind.class.bits();
    const mantissa_bits: u32 = if (width == 32) 23 else 52;
    const hidden: u64 = @as(u64, 1) << @intCast(mantissa_bits);
    const sign: u64 = @as(u64, 1) << @intCast(width - 1);
    const magnitude_mask = sign - 1;
    const infinity: u64 = magnitude_mask & ~(hidden - 1);
    const quiet_nan: u64 = infinity | (hidden >> 1);
    const class: Mir.Class = if (width == 32) .b32 else .b64;
    const bit_type = class.bitType();
    const unsigned_type = class.unsignedType();
    const plain: Mir.Inst.Modifiers = .{ .none = .{} };
    const eq: Mir.Inst.Modifiers = .{ .setp = .{ .cmp = .eq } };
    const ge: Mir.Inst.Modifiers = .{ .setp = .{ .cmp = .ge } };
    const end = try builder.newBlock(&.{class});

    const dividend_bits = try builder.emit(class, .mov, bit_type, plain, .{dividend});
    const divisor_bits = try builder.emit(class, .mov, bit_type, plain, .{divisor});
    const dividend_sign = try builder.emit(class, .@"and", bit_type, plain, .{ dividend_bits, sign });
    const dividend_magnitude = try builder.emit(class, .@"and", bit_type, plain, .{ dividend_bits, magnitude_mask });
    const divisor_magnitude = try builder.emit(class, .@"and", bit_type, plain, .{ divisor_bits, magnitude_mask });
    const divisor_zero = try builder.emit(.pred, .setp, unsigned_type, eq, .{ divisor_magnitude, 0 });
    const divisor_nan = try builder.emit(.pred, .setp, unsigned_type, .{ .setp = .{ .cmp = .gt, .bool_op = .@"or" } }, .{ divisor_magnitude, infinity, divisor_zero });
    const invalid = try builder.emit(.pred, .setp, unsigned_type, .{ .setp = .{ .cmp = .ge, .bool_op = .@"or" } }, .{ dividend_magnitude, infinity, divisor_nan });
    try builder.branchIf(invalid, false, end, &.{try builder.movImm(class, quiet_nan)});
    const smaller = try builder.emit(.pred, .setp, unsigned_type, .{ .setp = .{ .cmp = .lt } }, .{ dividend_magnitude, divisor_magnitude });
    try builder.branchIf(smaller, false, end, &.{dividend_bits});
    const same = try builder.emit(.pred, .setp, unsigned_type, eq, .{ dividend_magnitude, divisor_magnitude });
    try builder.branchIf(same, false, end, &.{dividend_sign});

    const normal_dividend = try normalizeSignificand(builder, dividend_magnitude, class, mantissa_bits);
    const normal_divisor = try normalizeSignificand(builder, divisor_magnitude, class, mantissa_bits);
    const head = try builder.newBlock(&.{ class, .b32 });
    const last_step = builder.labelBlock();
    try builder.branch(head, &.{ normal_dividend.significand, normal_dividend.exponent });
    const loop = try builder.startBlock(head);
    const more = try builder.emit(.pred, .setp, .s32, .{ .setp = .{ .cmp = .gt } }, .{ loop[1], normal_divisor.exponent });
    try builder.branchIf(more, true, last_step, &.{});
    const reduced = try subtractIfGreater(builder, loop[0], normal_divisor.significand, class);
    const doubled = try builder.emit(class, .shl, bit_type, plain, .{ reduced, 1 });
    const exponent = try builder.emit(.b32, .sub, .s32, plain, .{ loop[1], 1 });
    try builder.branch(head, &.{ doubled, exponent });
    _ = try builder.startBlock(last_step);
    const final = try subtractIfGreater(builder, loop[0], normal_divisor.significand, class);
    const zero = try builder.emit(.pred, .setp, unsigned_type, eq, .{ final, 0 });
    try builder.branchIf(zero, false, end, &.{dividend_sign});

    const leading = try builder.emit(.b32, .clz, bit_type, plain, .{final});
    const shift = try builder.emit(.b32, .sub, .u32, plain, .{ leading, width - 1 - mantissa_bits });
    const significand = try builder.emit(class, .shl, bit_type, plain, .{ final, shift });
    const result_exponent = try builder.emit(.b32, .sub, .s32, plain, .{ loop[1], shift });
    const is_normal = try builder.emit(.pred, .setp, .s32, .{ .setp = .{ .cmp = .gt } }, .{ result_exponent, 0 });
    const exponent_field = if (width == 32)
        result_exponent
    else
        try builder.emit(.b64, .cvt, .u64, .{ .cvt = .{ .from = .u32 } }, .{result_exponent});
    const exponent_bits = try builder.emit(class, .shl, bit_type, plain, .{ exponent_field, mantissa_bits });
    const fraction = try builder.emit(class, .sub, unsigned_type, plain, .{ significand, hidden });
    const normal_bits = try builder.emit(class, .@"or", bit_type, plain, .{ fraction, exponent_bits });
    const subnormal_shift = try builder.emit(.b32, .sub, .s32, plain, .{ 1, result_exponent });
    const subnormal_bits = try builder.emit(class, .shr, unsigned_type, plain, .{ significand, subnormal_shift });
    const magnitude = try builder.emit(class, .selp, bit_type, plain, .{ normal_bits, subnormal_bits, is_normal });
    try builder.branch(end, &.{try builder.emit(class, .@"or", bit_type, plain, .{ magnitude, dividend_sign })});
    const result_bits = (try builder.startBlock(end))[0];
    _ = ge;
    return builder.emit(kind.class, .mov, bit_type, plain, .{result_bits});
}

const Significand = struct { significand: Mir.Reg, exponent: Mir.Reg };

fn normalizeSignificand(builder: *Builder, magnitude: Mir.Reg, class: Mir.Class, mantissa_bits: u32) Layout.Error!Significand {
    const width = class.bits();
    const hidden: u64 = @as(u64, 1) << @intCast(mantissa_bits);
    const bit_type = class.bitType();
    const plain: Mir.Inst.Modifiers = .{ .none = .{} };
    const raw_exponent = try builder.emit(class, .shr, class.unsignedType(), plain, .{ magnitude, mantissa_bits });
    const biased = if (width == 32) raw_exponent else try builder.emit(.b32, .cvt, .u32, .{ .cvt = .{ .from = .u64 } }, .{raw_exponent});
    const fraction = try builder.emit(class, .@"and", bit_type, plain, .{ magnitude, hidden - 1 });
    const is_normal = try builder.emit(.pred, .setp, .u32, .{ .setp = .{ .cmp = .ne } }, .{ biased, 0 });
    const implicit = try builder.emit(class, .selp, bit_type, plain, .{ hidden, 0, is_normal });
    const unscaled = try builder.emit(class, .@"or", bit_type, plain, .{ fraction, implicit });
    const exponent = try builder.emit(.b32, .max, .s32, plain, .{ biased, 1 });
    const leading = try builder.emit(.b32, .clz, bit_type, plain, .{unscaled});
    const shift = try builder.emit(.b32, .sub, .u32, plain, .{ leading, width - 1 - mantissa_bits });
    return .{
        .significand = try builder.emit(class, .shl, bit_type, plain, .{ unscaled, shift }),
        .exponent = try builder.emit(.b32, .sub, .s32, plain, .{ exponent, shift }),
    };
}

fn subtractIfGreater(builder: *Builder, minuend: Mir.Reg, subtrahend: Mir.Reg, class: Mir.Class) Layout.Error!Mir.Reg {
    const difference = try builder.emit(class, .sub, class.unsignedType(), .{ .none = .{} }, .{ minuend, subtrahend });
    const fits = try builder.emit(.pred, .setp, class.unsignedType(), .{ .setp = .{ .cmp = .ge } }, .{ minuend, subtrahend });
    return builder.emit(class, .selp, class.bitType(), .{ .none = .{} }, .{ difference, minuend, fits });
}

pub fn compare(builder: *Builder, op: CompareOperator, lhs: Mir.Reg, rhs: Mir.Source, kind: Kind) Layout.Error!Mir.Reg {
    if (kind.class == .b16) {
        const lhs_wide = try toF32(builder, lhs);
        return compare(builder, op, lhs_wide, .{ .reg = try toF32(builder, try builder.sourceReg(rhs, kind)) }, .f32);
    }
    const cmp: Mir.Cmp = switch (op) {
        .lt => .lt,
        .lte => .le,
        .eq => .eq,
        .gte => .ge,
        .gt => .gt,
        // Zig `!=` is true when an operand is NaN.
        .neq => .neu,
    };
    return builder.emit(.pred, .setp, kind.arithmeticType(), .{ .setp = .{ .cmp = cmp } }, .{ lhs, rhs });
}
