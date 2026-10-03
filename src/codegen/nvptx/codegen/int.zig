//! This file lowers integer and bool operations on registers.

const std = @import("std");
const assert = std.debug.assert;
const CompareOperator = std.math.CompareOperator;

const Mir = @import("../Mir.zig");
const Kind = @import("Kind.zig");
const Builder = @import("Builder.zig");
const Layout = @import("Layout.zig");

pub fn indexIs(builder: *Builder, index: Mir.Reg, value: usize) Layout.Error!Mir.Reg {
    return builder.emit(.pred, .setp, index.class.unsignedType(), .{ .setp = .{ .cmp = .eq } }, .{ index, value });
}

pub fn anyOf(builder: *Builder, matches: []const Mir.Reg) Layout.Error!Mir.Reg {
    if (matches.len == 0) return builder.movImm(.pred, 0);
    var any = matches[0];
    for (matches[1..]) |match| any = try builder.emit(.pred, .@"or", .pred, .{ .none = .{} }, .{ any, match });
    return any;
}

pub fn packScalar(builder: *Builder, kind: Kind, src: Builder.Part, limbs: []Builder.Part, bit: u64) Layout.Error!void {
    const field_kind: Kind = .{ .class = .b64, .size = 8, .repr = .unsigned, .bits = kind.bits };
    const value = try builder.toReg(src, kind);
    const field = switch (kind.repr) {
        .bool => try builder.emit(.b64, .selp, .b64, .{ .none = .{} }, .{ 1, 0, value }),
        .float => field: {
            const raw_kind = Kind.int(.unsigned, kind.class.bits());
            const raw = if (raw_kind.class == kind.class) value else try builder.emit(raw_kind.class, .mov, raw_kind.bitType(), .{ .none = .{} }, .{value});
            break :field try intConvert(builder, raw, raw_kind, field_kind);
        },
        .unsigned, .signed => try intConvert(builder, value, kind, field_kind),
    };
    const shift = bit % 64;
    try orInto(builder, &limbs[bit / 64], if (shift == 0) field else try builder.emit(.b64, .shl, .b64, .{ .none = .{} }, .{ field, shift }));
    if (shift + kind.bits <= 64) return;
    try orInto(builder, &limbs[bit / 64 + 1], try builder.emit(.b64, .shr, .u64, .{ .none = .{} }, .{ field, 64 - shift }));
}

fn orInto(builder: *Builder, limb: *Builder.Part, bits: Mir.Reg) Layout.Error!void {
    if (limb.* == .imm and limb.imm == 0) {
        limb.* = .{ .reg = bits };
        return;
    }
    const old = try builder.toReg(limb.*, Kind.int(.unsigned, 64));
    limb.* = .{ .reg = try builder.emit(.b64, .@"or", .b64, .{ .none = .{} }, .{ old, bits }) };
}

pub fn unpackScalar(builder: *Builder, kind: Kind, limbs: []const Builder.Part, bit: u64) Layout.Error!Mir.Reg {
    const limb_kind = Kind.int(.unsigned, 64);
    const shift = bit % 64;
    const low = try builder.toReg(limbs[bit / 64], limb_kind);
    var field = if (shift == 0) low else try builder.emit(.b64, .shr, .u64, .{ .none = .{} }, .{ low, shift });
    if (shift + kind.bits > 64) {
        const high = try builder.emit(.b64, .shl, .b64, .{ .none = .{} }, .{ try builder.toReg(limbs[bit / 64 + 1], limb_kind), 64 - shift });
        field = try builder.emit(.b64, .@"or", .b64, .{ .none = .{} }, .{ field, high });
    }
    switch (kind.repr) {
        .bool => {
            const masked = try builder.emit(.b64, .@"and", .b64, .{ .none = .{} }, .{ field, 1 });
            return builder.emit(.pred, .setp, .u64, .{ .setp = .{ .cmp = .ne } }, .{ masked, 0 });
        },
        .float => {
            const raw_kind = Kind.int(.unsigned, kind.class.bits());
            const raw = try intConvert(builder, field, limb_kind, raw_kind);
            if (raw_kind.class == kind.class) return raw;
            return builder.emit(kind.class, .mov, raw_kind.bitType(), .{ .none = .{} }, .{raw});
        },
        .unsigned, .signed => return intConvert(builder, field, limb_kind, kind),
    }
}

pub const BinaryOp = enum { add, sub, mul, div, div_trunc, div_floor, rem, mod, min, max, @"and", @"or", xor, shl, shr, add_sat, sub_sat, mul_sat, shl_sat };

pub const UnaryOp = enum { not, neg, abs, sqrt, floor, ceil, trunc, round, popcount, clz, ctz, bit_reverse, byte_swap };

pub const OverflowOp = enum { add, sub, mul, shl };

pub fn powerOfTwoDivisor(op: BinaryOp, divisor: Builder.Part, kind: Kind) ?u6 {
    switch (op) {
        .div, .div_trunc, .div_floor, .rem, .mod => {},
        else => return null,
    }
    if (kind.repr != .unsigned and kind.repr != .signed) return null;
    const imm = switch (divisor) {
        .imm => |imm| imm,
        else => return null,
    };
    const value = imm & (@as(u64, std.math.maxInt(u64)) >> @intCast(64 - kind.bits));
    if (kind.repr == .signed and value >> @intCast(kind.bits - 1) != 0) return null;
    if (!std.math.isPowerOfTwo(value)) return null;
    return @intCast(@ctz(value));
}

/// Divides `lhs` by `1 << exponent`. `lhs` holds a narrow integer extended to the register width.
pub fn divPowerOfTwo(builder: *Builder, op: BinaryOp, lhs: Mir.Reg, exponent: u6, kind: Kind) Layout.Error!Mir.Reg {
    const ptx_type = kind.arithmeticType();
    const bit_type = kind.bitType();
    const amount: u64 = exponent;
    const mask = (@as(u64, 1) << exponent) - 1;
    if (kind.repr == .unsigned or op == .div_floor or op == .mod or op == .div) return switch (op) {
        .div, .div_trunc, .div_floor => builder.emit(kind.class, .shr, ptx_type, .{ .none = .{} }, .{ lhs, amount }),
        .rem, .mod => builder.emit(kind.class, .@"and", bit_type, .{ .none = .{} }, .{ lhs, mask }),
        else => unreachable,
    };
    // For a negative dividend, truncating division adds `divisor - 1` before the arithmetic shift.
    const sign = try builder.emit(kind.class, .shr, ptx_type, .{ .none = .{} }, .{ lhs, @as(u64, kind.class.bits() - 1) });
    const bias = try builder.emit(kind.class, .@"and", bit_type, .{ .none = .{} }, .{ sign, mask });
    const biased = try builder.emit(kind.class, .add, ptx_type, .{ .none = .{} }, .{ lhs, bias });
    const quotient = try builder.emit(kind.class, .shr, ptx_type, .{ .none = .{} }, .{ biased, amount });
    if (op == .div_trunc) return quotient;
    const multiple = try builder.emit(kind.class, .shl, bit_type, .{ .none = .{} }, .{ quotient, amount });
    return builder.emit(kind.class, .sub, ptx_type, .{ .none = .{} }, .{ lhs, multiple });
}

/// `shl` and `shr` take a `.u32` amount (PTX ISA 9.4, sections 9.7.9.8 and 9.7.9.9). A 64-bit
/// amount saturates at 64 first, so that the conversion to 32 bits keeps a large amount large.
pub fn shiftAmount(builder: *Builder, amount: Mir.Reg, kind: Kind) Layout.Error!Mir.Reg {
    switch (amount.class) {
        .b32 => return amount,
        .b64 => {
            const limited = try builder.emit(.b64, .min, .u64, .{ .none = .{} }, .{ amount, 64 });
            return builder.emit(.b32, .cvt, .u32, .{ .cvt = .{ .from = .u64 } }, .{limited});
        },
        else => return builder.emit(.b32, .cvt, .u32, .{ .cvt = .{ .from = kind.arithmeticType() } }, .{amount}),
    }
}

pub fn instructionTag(op: BinaryOp) Mir.Inst.Tag {
    return switch (op) {
        .add => .add,
        .sub => .sub,
        .mul => .mul,
        .min => .min,
        .max => .max,
        .@"and" => .@"and",
        .@"or" => .@"or",
        .xor => .xor,
        .shl => .shl,
        .shr => .shr,
        else => unreachable,
    };
}

pub fn isCommutative(op: BinaryOp) bool {
    return switch (op) {
        .add, .mul, .@"and", .@"or", .xor, .min, .max => true,
        else => false,
    };
}

pub fn binaryReg(builder: *Builder, op: BinaryOp, lhs: Mir.Reg, rhs: Mir.Reg, kind: Kind) Layout.Error!Mir.Reg {
    return binary(builder, op, lhs, .{ .reg = rhs }, kind);
}

pub fn binary(builder: *Builder, op: BinaryOp, lhs: Mir.Reg, rhs: Mir.Source, kind: Kind) Layout.Error!Mir.Reg {
    assert(kind.repr != .float);
    switch (op) {
        .add_sat, .sub_sat, .mul_sat => return saturate(builder, op, lhs, try builder.sourceReg(rhs, kind), kind),
        .shl_sat => return saturate(builder, op, lhs, try builder.sourceReg(rhs, .int(.unsigned, 32)), kind),
        .div_floor, .mod => return divideFloor(builder, op, lhs, rhs, kind),
        else => {},
    }
    if (kind.class == .pred) {
        const tag: Mir.Inst.Tag = switch (op) {
            .@"and", .min, .mul => .@"and",
            .@"or", .max => .@"or",
            .xor, .add, .sub => .xor,
            else => unreachable,
        };
        return builder.emit(.pred, tag, .pred, .{ .none = .{} }, .{ lhs, try builder.sourceReg(rhs, kind) });
    }
    const ptx_type = kind.arithmeticType();
    // ptxas 13.4 drops the 16-bit wrap of `sub.s16` from zero when a later compare uses the result.
    const sub_type: Mir.Type = if (kind.class == .b16) .u16 else ptx_type;
    const result = switch (op) {
        .sub => try builder.emit(kind.class, .sub, sub_type, .{ .none = .{} }, .{ lhs, rhs }),
        .add, .min, .max, .shr => try builder.emit(kind.class, instructionTag(op), ptx_type, .{ .none = .{} }, .{ lhs, rhs }),
        .mul => try builder.emit(kind.class, .mul, ptx_type, .{ .arith = .{ .mode = .lo } }, .{ lhs, rhs }),
        .div, .div_trunc => try builder.emit(kind.class, .div, ptx_type, .{ .none = .{} }, .{ lhs, rhs }),
        .rem => try builder.emit(kind.class, .rem, ptx_type, .{ .none = .{} }, .{ lhs, rhs }),
        .@"and", .@"or", .xor, .shl => try builder.emit(kind.class, instructionTag(op), kind.bitType(), .{ .none = .{} }, .{ lhs, rhs }),
        .div_floor, .mod, .add_sat, .sub_sat, .mul_sat, .shl_sat => unreachable,
    };
    return switch (op) {
        .add, .sub, .mul, .shl => normalized(builder, result, kind),
        else => result,
    };
}

/// Truncating division rounds toward zero. Floor division therefore subtracts one when the
/// remainder is nonzero and differs in sign from the divisor. `@mod` adds `rhs` in that case.
fn divideFloor(builder: *Builder, op: BinaryOp, lhs: Mir.Reg, rhs: Mir.Source, kind: Kind) Layout.Error!Mir.Reg {
    const ptx_type = kind.arithmeticType();
    if (kind.repr == .unsigned) return builder.emit(kind.class, if (op == .mod) .rem else .div, ptx_type, .{ .none = .{} }, .{ lhs, rhs });
    const remainder = try builder.emit(kind.class, .rem, ptx_type, .{ .none = .{} }, .{ lhs, rhs });
    const signs = try builder.emit(kind.class, .xor, kind.bitType(), .{ .none = .{} }, .{ remainder, rhs });
    const signs_differ = try builder.emit(.pred, .setp, ptx_type, .{ .setp = .{ .cmp = .lt } }, .{ signs, 0 });
    const adjust = try builder.emit(.pred, .setp, ptx_type, .{ .setp = .{ .cmp = .ne, .bool_op = .@"and" } }, .{ remainder, 0, signs_differ });
    if (op == .mod) {
        const sum = try builder.emit(kind.class, .add, ptx_type, .{ .none = .{} }, .{ remainder, rhs });
        return builder.emit(kind.class, .selp, kind.bitType(), .{ .none = .{} }, .{ sum, remainder, adjust });
    }
    const quotient = try builder.emit(kind.class, .div, ptx_type, .{ .none = .{} }, .{ lhs, rhs });
    return builder.emitGuarded(adjust, quotient, .sub, ptx_type, .{ .arith = .{ .guard = .positive } }, .{ quotient, 1 });
}

pub fn overflow(builder: *Builder, op: OverflowOp, lhs: Mir.Reg, rhs: Mir.Reg, kind: Kind) Layout.Error!struct { Mir.Reg, Mir.Reg } {
    const ptx_type = kind.arithmeticType();
    if (op == .shl) {
        // The shift overflows when shifting the wrapped result back does not give `lhs`. An amount
        // above the register width also changes the value (PTX ISA 9.4, section 9.7.9.8).
        const shifted = try builder.emit(kind.class, .shl, kind.bitType(), .{ .none = .{} }, .{ lhs, rhs });
        const wrapped = try normalized(builder, shifted, kind);
        const back = try builder.emit(kind.class, .shr, ptx_type, .{ .none = .{} }, .{ wrapped, rhs });
        return .{ wrapped, try builder.emit(.pred, .setp, ptx_type, .{ .setp = .{ .cmp = .ne } }, .{ back, lhs }) };
    }
    if (kind.bits == 64 and op != .mul) {
        const wrapped = try builder.emit(.b64, if (op == .add) .add else .sub, ptx_type, .{ .none = .{} }, .{ lhs, rhs });
        if (kind.repr == .unsigned) {
            const overflowed = if (op == .add)
                try builder.emit(.pred, .setp, .u64, .{ .setp = .{ .cmp = .lt } }, .{ wrapped, lhs })
            else
                try builder.emit(.pred, .setp, .u64, .{ .setp = .{ .cmp = .lt } }, .{ lhs, rhs });
            return .{ wrapped, overflowed };
        }
        // Overflow occurs exactly when `rhs < 0` differs from `wrapped < lhs` for addition, or
        // from `wrapped > lhs` for subtraction.
        const rhs_negative = try builder.emit(.pred, .setp, .s64, .{ .setp = .{ .cmp = .lt } }, .{ rhs, 0 });
        const cmp: Mir.Cmp = if (op == .add) .lt else .gt;
        return .{ wrapped, try builder.emit(.pred, .setp, .s64, .{ .setp = .{ .cmp = cmp, .bool_op = .xor } }, .{ wrapped, lhs, rhs_negative }) };
    }
    if (op == .mul and kind.bits > 32) {
        const low = try builder.emit(.b64, .mul, ptx_type, .{ .arith = .{ .mode = .lo } }, .{ lhs, rhs });
        const high = try builder.emit(.b64, .mul, ptx_type, .{ .arith = .{ .mode = .hi } }, .{ lhs, rhs });
        const high_overflowed = if (kind.repr == .signed) signed: {
            const fill = try builder.emit(.b64, .shr, .s64, .{ .none = .{} }, .{ low, 63 });
            break :signed try builder.emit(.pred, .setp, .u64, .{ .setp = .{ .cmp = .ne } }, .{ high, fill });
        } else try builder.emit(.pred, .setp, .u64, .{ .setp = .{ .cmp = .ne } }, .{ high, 0 });
        if (kind.bits == 64) return .{ low, high_overflowed };
        const wrapped = try normalized(builder, low, kind);
        return .{ wrapped, try builder.emit(.pred, .setp, .u64, .{ .setp = .{ .cmp = .ne, .bool_op = .@"or" } }, .{ low, wrapped, high_overflowed }) };
    }
    // The exact result fits in `exact_kind`, and the operation overflows when narrowing changes it.
    const exact_kind = Kind.int(kind.repr, if (kind.bits <= 16) 32 else 64);
    var reduced_kind = exact_kind;
    reduced_kind.bits = kind.bits;
    const lhs_exact = if (kind.class == exact_kind.class) lhs else try intConvert(builder, lhs, kind, exact_kind);
    const rhs_exact = if (kind.class == exact_kind.class) rhs else try intConvert(builder, rhs, kind, exact_kind);
    const binary_op: BinaryOp = switch (op) {
        .add => .add,
        .sub => .sub,
        .mul => .mul,
        .shl => unreachable,
    };
    const exact = try binaryReg(builder, binary_op, lhs_exact, rhs_exact, exact_kind);
    const wrapped = try normalized(builder, exact, reduced_kind);
    const overflowed = try builder.emit(.pred, .setp, exact_kind.arithmeticType(), .{ .setp = .{ .cmp = .ne } }, .{ exact, wrapped });
    if (kind.class == exact_kind.class) return .{ wrapped, overflowed };
    return .{ try intConvert(builder, wrapped, reduced_kind, kind), overflowed };
}

fn saturate(builder: *Builder, op: BinaryOp, lhs: Mir.Reg, rhs: Mir.Reg, kind: Kind) Layout.Error!Mir.Reg {
    const signed = kind.repr == .signed;
    if (signed and kind.bits == 32 and (op == .add_sat or op == .sub_sat)) {
        return builder.emit(.b32, if (op == .add_sat) .add else .sub, .s32, .{ .arith = .{ .sat = true } }, .{ lhs, rhs });
    }
    const wrapped, const overflowed = try overflow(builder, switch (op) {
        .add_sat => .add,
        .sub_sat => .sub,
        .mul_sat => .mul,
        .shl_sat => .shl,
        else => unreachable,
    }, lhs, rhs, kind);
    const bit_type = kind.bitType();
    const value_mask = if (kind.bits == 64) std.math.maxInt(u64) else (@as(u64, 1) << @intCast(kind.bits)) - 1;
    const register_mask = if (kind.class.bits() == 64) std.math.maxInt(u64) else (@as(u64, 1) << @intCast(kind.class.bits())) - 1;
    const max = if (signed) value_mask >> 1 else value_mask;
    if (!signed) {
        const limit: u64 = if (op == .sub_sat) 0 else max;
        return builder.emit(kind.class, .selp, bit_type, .{ .none = .{} }, .{ limit, wrapped, overflowed });
    }
    // A signed result saturates toward the sign of `lhs`, or of `lhs ^ rhs` for a product.
    const sign_source = if (op == .mul_sat) try builder.emit(kind.class, .xor, bit_type, .{ .none = .{} }, .{ lhs, rhs }) else lhs;
    const negative = try builder.emit(.pred, .setp, kind.arithmeticType(), .{ .setp = .{ .cmp = .lt } }, .{ sign_source, 0 });
    const limit = try builder.emit(kind.class, .selp, bit_type, .{ .none = .{} }, .{ ~max & register_mask, max, negative });
    return builder.emit(kind.class, .selp, bit_type, .{ .none = .{} }, .{ limit, wrapped, overflowed });
}

pub fn normalized(builder: *Builder, reg: Mir.Reg, kind: Kind) Layout.Error!Mir.Reg {
    if (!kind.isNarrowInt()) return reg;
    const bit_type = kind.bitType();
    if (kind.repr == .signed) {
        const shift = kind.class.bits() - kind.bits;
        const high = try builder.emit(kind.class, .shl, bit_type, .{ .none = .{} }, .{ reg, shift });
        return builder.emit(kind.class, .shr, kind.arithmeticType(), .{ .none = .{} }, .{ high, shift });
    }
    return builder.emit(kind.class, .@"and", bit_type, .{ .none = .{} }, .{ reg, (@as(u64, 1) << @intCast(kind.bits)) - 1 });
}

pub fn unary(builder: *Builder, op: UnaryOp, src: Mir.Reg, kind: Kind, result: Kind) Layout.Error!Mir.Reg {
    assert(kind.repr != .float);
    if (kind.class == .pred) {
        assert(op == .not);
        return builder.emit(.pred, .not, .pred, .{ .none = .{} }, .{src});
    }
    return switch (op) {
        .not => normalized(builder, try builder.emit(kind.class, .not, kind.bitType(), .{ .none = .{} }, .{src}), result),
        .neg => normalized(builder, try builder.emit(kind.class, .neg, signedType(kind), .{ .none = .{} }, .{src}), result),
        .abs => normalized(builder, try builder.emit(kind.class, .abs, signedType(kind), .{ .none = .{} }, .{src}), result),
        .byte_swap => byteSwap(builder, src, kind),
        .popcount, .clz, .ctz, .bit_reverse => countBits(builder, op, src, kind, result),
        .sqrt, .floor, .ceil, .trunc, .round => unreachable,
    };
}

fn countBits(builder: *Builder, op: UnaryOp, src: Mir.Reg, kind: Kind, result: Kind) Layout.Error!Mir.Reg {
    var value = src;
    if (kind.class == .b16) value = try builder.emit(.b32, .cvt, .u32, .{ .cvt = .{ .from = .u16 } }, .{src});
    if (kind.repr == .signed and kind.bits != kind.class.bits() and op != .bit_reverse) {
        value = try builder.emit(value.class, .@"and", value.class.bitType(), .{ .none = .{} }, .{ value, (@as(u64, 1) << @intCast(kind.bits)) - 1 });
    }
    const width = value.class.bits();
    const bit_type = value.class.bitType();
    const count = switch (op) {
        .popcount => try builder.emit(.b32, .popc, bit_type, .{ .none = .{} }, .{value}),
        .clz => count: {
            const leading = try builder.emit(.b32, .clz, bit_type, .{ .none = .{} }, .{value});
            if (kind.bits == width) break :count leading;
            break :count try builder.emit(.b32, .sub, .u32, .{ .none = .{} }, .{ leading, width - kind.bits });
        },
        .ctz => count: {
            const reversed = try builder.emit(value.class, .brev, bit_type, .{ .none = .{} }, .{value});
            const trailing = try builder.emit(.b32, .clz, bit_type, .{ .none = .{} }, .{reversed});
            break :count try builder.emit(.b32, .min, .u32, .{ .none = .{} }, .{ trailing, kind.bits });
        },
        .bit_reverse => {
            const reversed = try builder.emit(value.class, .brev, bit_type, .{ .none = .{} }, .{value});
            const wide_kind = Kind.int(kind.repr, width);
            const aligned = if (kind.bits == width)
                reversed
            else
                try builder.emit(value.class, .shr, wide_kind.arithmeticType(), .{ .none = .{} }, .{ reversed, width - kind.bits });
            if (aligned.class == kind.class) return aligned;
            var narrow = wide_kind;
            narrow.bits = kind.bits;
            return intConvert(builder, aligned, narrow, kind);
        },
        else => unreachable,
    };
    return intConvert(builder, count, Kind.int(.unsigned, 32), result);
}

/// Assumes `kind.bits` is a multiple of 8.
fn byteSwap(builder: *Builder, src: Mir.Reg, kind: Kind) Layout.Error!Mir.Reg {
    if (kind.bits == 8) return src;
    const reverse_bytes: Mir.Imm = .{ .value = 0x0123, .format = .prmt };
    switch (kind.class) {
        .b16 => {
            // `prmt` exists only for `.b32` (PTX ISA 9.4, section 9.7.10.7). Selector 0x4401 swaps
            // bytes 1 and 0 and fills the high bytes with zero bytes of `b`.
            const widened = try builder.emit(.b32, .cvt, .u32, .{ .cvt = .{ .from = .u16 } }, .{src});
            const swapped = try builder.emit(.b32, .prmt, .b32, .{ .none = .{} }, .{ widened, 0, Mir.Imm{ .value = 0x4401, .format = .prmt } });
            return builder.emit(.b16, .cvt, .u16, .{ .cvt = .{ .from = .u32 } }, .{swapped});
        },
        .b32 => {
            const swapped = try builder.emit(.b32, .prmt, .b32, .{ .none = .{} }, .{ src, 0, reverse_bytes });
            // `prmt` leaves a narrower integer in the high bytes, so a shift moves it back down.
            if (kind.bits == 32) return swapped;
            return builder.emit(.b32, .shr, kind.arithmeticType(), .{ .none = .{} }, .{ swapped, 32 - kind.bits });
        },
        else => {
            const halves = try builder.emitList(&.{ .b32, .b32 }, .mov, .b64, .{ .none = .{} }, .{src});
            const low = try builder.emit(.b32, .prmt, .b32, .{ .none = .{} }, .{ halves[1], 0, reverse_bytes });
            const high = try builder.emit(.b32, .prmt, .b32, .{ .none = .{} }, .{ halves[0], 0, reverse_bytes });
            const swapped = try builder.emit(.b64, .mov, .b64, .{ .none = .{} }, .{Mir.RegList{ .regs = &.{ low, high } }});
            if (kind.bits == 64) return swapped;
            return builder.emit(.b64, .shr, kind.arithmeticType(), .{ .none = .{} }, .{ swapped, 64 - kind.bits });
        },
    }
}

fn signedType(kind: Kind) Mir.Type {
    if (kind.repr == .float) return kind.arithmeticType();
    return switch (kind.class) {
        .b16 => .s16,
        .b32 => .s32,
        else => .s64,
    };
}

pub fn compareReg(builder: *Builder, op: CompareOperator, lhs: Mir.Reg, rhs: Mir.Reg, kind: Kind) Layout.Error!Mir.Reg {
    return compareSource(builder, op, lhs, .{ .reg = rhs }, kind);
}

pub fn compareSource(builder: *Builder, op: CompareOperator, lhs: Mir.Reg, rhs: Mir.Source, kind: Kind) Layout.Error!Mir.Reg {
    if (kind.class == .pred) {
        assert(op == .eq or op == .neq);
        const different = try builder.emit(.pred, .xor, .pred, .{ .none = .{} }, .{ lhs, try builder.sourceReg(rhs, kind) });
        if (op == .neq) return different;
        return builder.emit(.pred, .not, .pred, .{ .none = .{} }, .{different});
    }
    const cmp: Mir.Cmp = switch (op) {
        .lt => .lt,
        .lte => .le,
        .eq => .eq,
        .gte => .ge,
        .gt => .gt,
        .neq => .ne,
    };
    return builder.emit(.pred, .setp, kind.arithmeticType(), .{ .setp = .{ .cmp = cmp } }, .{ lhs, rhs });
}

pub fn select(builder: *Builder, pred: Mir.Reg, lhs: Mir.Reg, rhs: Mir.Reg, kind: Kind) Layout.Error!Mir.Reg {
    if (kind.class != .pred) return builder.emit(kind.class, .selp, kind.bitType(), .{ .none = .{} }, .{ lhs, rhs, pred });
    const when_true = try builder.emit(.pred, .@"and", .pred, .{ .none = .{} }, .{ pred, lhs });
    const not_pred = try builder.emit(.pred, .not, .pred, .{ .none = .{} }, .{pred});
    const when_false = try builder.emit(.pred, .@"and", .pred, .{ .none = .{} }, .{ not_pred, rhs });
    return builder.emit(.pred, .@"or", .pred, .{ .none = .{} }, .{ when_true, when_false });
}

pub fn intConvert(builder: *Builder, src: Mir.Reg, from: Kind, to: Kind) Layout.Error!Mir.Reg {
    const renormalize = to.bits < from.bits or to.repr != from.repr;
    if (from.class == to.class) return if (renormalize) normalized(builder, src, to) else src;
    // `cvt.s64.u32` zero-extends (PTX ISA 9.4, section 6.5.1), but ptxas -O3 sign-extends it before
    // a `shl` and `shr.s64` pair. The conversion therefore keeps the signedness of the source.
    var extended = to;
    extended.repr = from.repr;
    // ptxas 13.0 can crash at -O1 and above for sm_90 to sm_103. The crash needs a `min` or `max`
    // of `.f16x2` or `.bf16x2` with a constant, a `cvt.u16.u32` of that result and a `setp` that
    // compares the `cvt` result with the low half of the constant. An `and.b32` avoids the crash.
    const source = if (from.class == .b32 and to.class == .b16)
        try builder.emit(.b32, .@"and", .b32, .{ .none = .{} }, .{ src, @as(u64, 0xffff) })
    else
        src;
    const converted = try builder.emit(to.class, .cvt, extended.arithmeticType(), .{ .cvt = .{ .from = from.arithmeticType() } }, .{source});
    return if (renormalize) normalized(builder, converted, to) else converted;
}

pub fn widen(builder: *Builder, reg: Mir.Reg, kind: Kind) Layout.Error!Mir.Reg {
    if (reg.class == .b64) return reg;
    return builder.emit(.b64, .cvt, if (kind.repr == .signed) .s64 else .u64, .{ .cvt = .{ .from = kind.arithmeticType() } }, .{reg});
}
