//! This file lowers the operations on integers wider than 64 bits. Such an integer has one 64-bit
//! limb for each 8 bytes of its ABI size, and the least significant limb comes first. The limbs
//! above the value hold its zero or sign extension. The names `a` and `b` follow `std.math.big`.

const std = @import("std");
const CompareOperator = std.math.CompareOperator;

const Mir = @import("../Mir.zig");
const Kind = @import("Kind.zig");
const Builder = @import("Builder.zig");
const Layout = @import("Layout.zig");
const int = @import("int.zig");

pub fn limbRegs(builder: *Builder, operands: []const Builder.Part, wide_int: Layout.WideInt) Layout.Error![]const Mir.Reg {
    const regs = try builder.arena.alloc(Mir.Reg, operands.len);
    for (regs, operands, 0..) |*reg, operand, i| reg.* = try builder.toReg(operand, wide_int.limbKind(@intCast(i)));
    return regs;
}

pub fn fillLimb(builder: *Builder, top: Mir.Reg, signedness: std.lang.Signedness) Layout.Error!Mir.Reg {
    if (signedness == .unsigned) return builder.movImm(.b64, 0);
    return builder.emit(.b64, .shr, .s64, .{ .none = .{} }, .{ top, 63 });
}

pub fn extendLimbs(builder: *Builder, src: []const Mir.Reg, signedness: std.lang.Signedness, len: usize) Layout.Error![]const Mir.Reg {
    const out = try builder.arena.alloc(Mir.Reg, len);
    @memcpy(out[0..src.len], src);
    if (len > src.len) @memset(out[src.len..], try fillLimb(builder, src[src.len - 1], signedness));
    return out;
}

pub fn normalizedLimbs(builder: *Builder, regs: []const Mir.Reg, wide_int: Layout.WideInt) Layout.Error![]const Mir.Reg {
    const top = wide_int.topLimb();
    const out = try builder.arena.dupe(Mir.Reg, regs);
    out[top] = try int.normalized(builder, regs[top], wide_int.limbKind(top));
    if (out.len > top + 1) @memset(out[top + 1 ..], try fillLimb(builder, out[top], wide_int.signedness));
    return out;
}

const Carry = enum { add, sub };

/// Adds or subtracts limb by limb. The condition code register carries between the limbs (PTX ISA
/// 9.4, section 9.7.2).
pub fn carryChain(builder: *Builder, op: Carry, lhs: []const Mir.Reg, rhs: []const Mir.Reg) Layout.Error![]const Mir.Reg {
    const out = try builder.arena.alloc(Mir.Reg, lhs.len);
    for (out, lhs, rhs, 0..) |*result, lhs_limb, rhs_limb, i| {
        const tag: Mir.Inst.Tag = switch (op) {
            .add => if (i == 0) .add else .addc,
            .sub => if (i == 0) .sub else .subc,
        };
        result.* = try builder.emit(.b64, tag, .u64, .{ .arith = .{ .cc = i + 1 < out.len } }, .{ lhs_limb, rhs_limb });
    }
    return out;
}

pub fn negateIf(builder: *Builder, limbs: []const Mir.Reg, sign: Mir.Reg) Layout.Error![]const Mir.Reg {
    const flipped = try builder.arena.alloc(Mir.Reg, limbs.len);
    for (flipped, limbs) |*result, limb| result.* = try builder.emit(.b64, .xor, .b64, .{ .none = .{} }, .{ limb, sign });
    const signs = try builder.arena.alloc(Mir.Reg, limbs.len);
    @memset(signs, sign);
    return carryChain(builder, .sub, flipped, signs);
}

pub fn selectLimbs(builder: *Builder, pred: Mir.Reg, lhs: []const Mir.Reg, rhs: []const Mir.Reg) Layout.Error![]const Mir.Reg {
    const out = try builder.arena.alloc(Mir.Reg, lhs.len);
    for (out, lhs, rhs) |*result, lhs_limb, rhs_limb| result.* = try builder.emit(.b64, .selp, .b64, .{ .none = .{} }, .{ lhs_limb, rhs_limb, pred });
    return out;
}

/// The most significant limb that differs decides an ordered comparison.
pub fn compareLimbs(builder: *Builder, op: CompareOperator, a: []const Mir.Reg, b: []const Mir.Reg, signedness: std.lang.Signedness) Layout.Error!Mir.Reg {
    if (op == .eq or op == .neq) {
        const cmp: Mir.Cmp = if (op == .eq) .eq else .ne;
        const join: Mir.BoolOp = if (op == .eq) .@"and" else .@"or";
        var result = try builder.emit(.pred, .setp, .u64, .{ .setp = .{ .cmp = cmp } }, .{ a[0], b[0] });
        for (a[1..], b[1..]) |lhs_limb, rhs_limb| {
            result = try builder.emit(.pred, .setp, .u64, .{ .setp = .{ .cmp = cmp, .bool_op = join } }, .{ lhs_limb, rhs_limb, result });
        }
        return result;
    }
    const lhs, const rhs = if (op == .gt or op == .gte) .{ b, a } else .{ a, b };
    const strict = op == .lt or op == .gt;
    var result = try builder.emit(.pred, .setp, .u64, .{ .setp = .{ .cmp = if (strict) .lt else .le } }, .{ lhs[0], rhs[0] });
    for (lhs[1..], rhs[1..], 1..) |lhs_limb, rhs_limb, i| {
        const ptx_type: Mir.Type = if (i + 1 == lhs.len and signedness == .signed) .s64 else .u64;
        const equal = try builder.emit(.pred, .setp, ptx_type, .{ .setp = .{ .cmp = .eq, .bool_op = .@"and" } }, .{ lhs_limb, rhs_limb, result });
        result = try builder.emit(.pred, .setp, ptx_type, .{ .setp = .{ .cmp = .lt, .bool_op = .@"or" } }, .{ lhs_limb, rhs_limb, equal });
    }
    return result;
}

pub const ShiftAmount = union(enum) {
    known: u64,
    reg: struct { reg: Mir.Reg, max: u64 },
};

pub fn limitedShiftAmount(builder: *Builder, limbs: []const Builder.Part) Layout.Error!Mir.Reg {
    const limb_kind = Kind.int(.unsigned, 64);
    const low = try builder.toReg(limbs[0], limb_kind);
    var high = try builder.toReg(limbs[1], limb_kind);
    for (limbs[2..]) |limb| high = try builder.emit(.b64, .@"or", .b64, .{ .none = .{} }, .{ high, try builder.toReg(limb, limb_kind) });
    const is_large = try builder.emit(.pred, .setp, .u64, .{ .setp = .{ .cmp = .ne } }, .{ high, 0 });
    return builder.emit(.b64, .selp, .b64, .{ .none = .{} }, .{ std.math.maxInt(u64), low, is_large });
}

const Direction = enum { left, right };

/// Moves whole limbs first and then shifts the bits within the limbs. For a register amount, the
/// backend emits one `selp` per limb for each bit of `amount / 64`.
pub fn shiftLimbs(builder: *Builder, direction: Direction, src: []const Mir.Reg, amount: ShiftAmount, signedness: std.lang.Signedness) Layout.Error![]const Mir.Reg {
    const len = src.len;
    const fill = if (direction == .right) try fillLimb(builder, src[len - 1], signedness) else try builder.movImm(.b64, 0);
    var moved = try builder.arena.dupe(Mir.Reg, src);
    const bit: Builder.Part, const inverse: Builder.Part = switch (amount) {
        .known => |known| known: {
            const step = @min(known / 64, len);
            for (moved, 0..) |*limb, i| limb.* = limbAfterMove(src, direction, i, step, fill);
            const bit = if (step == len) 0 else known % 64;
            break :known .{ .{ .imm = bit }, .{ .imm = 64 - bit } };
        },
        .reg => |register| register: {
            const step = try builder.emit(.b32, .shr, .u32, .{ .none = .{} }, .{ register.reg, 6 });
            var stride: u64 = 1;
            while (stride < len) : (stride *= 2) {
                const flag = try builder.emit(.b32, .@"and", .b32, .{ .none = .{} }, .{ step, stride });
                const takes_stride = try builder.emit(.pred, .setp, .u32, .{ .setp = .{ .cmp = .ne } }, .{ flag, 0 });
                const previous = moved;
                moved = try builder.arena.alloc(Mir.Reg, len);
                for (moved, 0..) |*limb, i| {
                    limb.* = try builder.emit(.b64, .selp, .b64, .{ .none = .{} }, .{ limbAfterMove(previous, direction, i, stride, fill), previous[i], takes_stride });
                }
            }
            if (register.max / 64 >= stride) {
                const too_far = try builder.emit(.pred, .setp, .u32, .{ .setp = .{ .cmp = .ge } }, .{ step, stride });
                for (moved) |*limb| limb.* = try builder.emit(.b64, .selp, .b64, .{ .none = .{} }, .{ fill, limb.*, too_far });
            }
            const bit = try builder.emit(.b32, .@"and", .b32, .{ .none = .{} }, .{ register.reg, 63 });
            const inverse = try builder.emit(.b32, .sub, .u32, .{ .none = .{} }, .{ 64, bit });
            break :register .{ .{ .reg = bit }, .{ .reg = inverse } };
        },
    };
    if (bit == .imm and bit.imm == 0) return moved;
    // `shl` and `shr` shift out every bit for an amount of 64 (PTX ISA 9.4, sections 9.7.9.8 and
    // 9.7.9.9), so a zero `bit` takes nothing from the neighbor limb.
    const out = try builder.arena.alloc(Mir.Reg, len);
    for (out, moved, 0..) |*result, limb, i| {
        const neighbor = switch (direction) {
            .left => if (i == 0) null else moved[i - 1],
            .right => if (i + 1 == len) fill else moved[i + 1],
        };
        const shifted = switch (direction) {
            .left => try shiftLimb(builder, .shl, .b64, limb, bit),
            .right => try shiftLimb(builder, .shr, .u64, limb, bit),
        };
        const other = neighbor orelse {
            result.* = shifted;
            continue;
        };
        const carried = switch (direction) {
            .left => try shiftLimb(builder, .shr, .u64, other, inverse),
            .right => try shiftLimb(builder, .shl, .b64, other, inverse),
        };
        result.* = try builder.emit(.b64, .@"or", .b64, .{ .none = .{} }, .{ shifted, carried });
    }
    return out;
}

fn limbAfterMove(src: []const Mir.Reg, direction: Direction, i: usize, step: u64, fill: Mir.Reg) Mir.Reg {
    return switch (direction) {
        .left => if (i >= step) src[i - @as(usize, @intCast(step))] else fill,
        .right => if (i + step < src.len) src[i + @as(usize, @intCast(step))] else fill,
    };
}

fn shiftLimb(builder: *Builder, tag: Mir.Inst.Tag, ptx_type: Mir.Type, src: Mir.Reg, amount: Builder.Part) Layout.Error!Mir.Reg {
    return switch (amount) {
        .imm => |imm| builder.emit(.b64, tag, ptx_type, .{ .none = .{} }, .{ src, imm }),
        .reg => |reg| builder.emit(.b64, tag, ptx_type, .{ .none = .{} }, .{ src, reg }),
        .symbol, .undef => unreachable,
    };
}

/// Row `i` adds the low halves of `a[i] * b[j]` at limb `i + j` in one carry chain and the high
/// halves at limb `i + j + 1` in another (PTX ISA 9.4, sections 9.7.2.5 and 9.7.2.6). Limb
/// `i + b.len` is still zero and receives the carry out of the first chain.
pub fn multiplyLimbs(builder: *Builder, a: []const Mir.Reg, b: []const Mir.Reg, len: usize) Layout.Error![]const Mir.Reg {
    const n = b.len;
    const sums = try builder.arena.alloc(Mir.Reg, len);
    @memset(sums, try builder.movImm(.b64, 0));
    for (a[0..@min(a.len, len)], 0..) |multiplier, i| {
        const has_carry = i + n < len;
        try multiplyAddChain(builder, .lo, sums[i..@min(i + n, len)], multiplier, b, has_carry);
        if (has_carry) sums[i + n] = try builder.emit(.b64, .addc, .u64, .{ .arith = .{} }, .{ 0, 0 });
        if (i + 1 < len) try multiplyAddChain(builder, .hi, sums[i + 1 .. @min(i + n + 1, len)], multiplier, b, false);
    }
    return sums;
}

fn multiplyAddChain(builder: *Builder, mode: Mir.MulMode, sums: []Mir.Reg, multiplier: Mir.Reg, b: []const Mir.Reg, carry_out: bool) Layout.Error!void {
    for (sums, b[0..sums.len], 0..) |*sum, multiplicand, j| {
        const cc = j + 1 < sums.len or carry_out;
        sum.* = try builder.emit(.b64, if (j == 0) .mad else .madc, .u64, .{ .arith = .{ .mode = mode, .cc = cc } }, .{ multiplier, multiplicand, sum.* });
    }
}

pub fn productLimbs(builder: *Builder, a: []const Mir.Reg, b: []const Mir.Reg, signedness: std.lang.Signedness) Layout.Error![]const Mir.Reg {
    const n = a.len;
    const product = try builder.arena.dupe(Mir.Reg, try multiplyLimbs(builder, a, b, 2 * n));
    if (signedness == .unsigned) return product;
    for ([2][]const Mir.Reg{ a, b }, [2][]const Mir.Reg{ b, a }) |signed_operand, other| {
        const mask = try fillLimb(builder, signed_operand[n - 1], .signed);
        const term = try builder.arena.alloc(Mir.Reg, n);
        for (term, other) |*limb, other_limb| limb.* = try builder.emit(.b64, .@"and", .b64, .{ .none = .{} }, .{ other_limb, mask });
        @memcpy(product[n..], try carryChain(builder, .sub, product[n..], term));
    }
    return product;
}

/// Restoring division finds one quotient bit per iteration. Each iteration doubles the remainder
/// and the dividend as one number and subtracts the denominator when the difference is not
/// negative. The remainder holds only bits of the numerator, so doubling it cannot overflow.
fn divideUnsigned(builder: *Builder, numerator: []const Mir.Reg, denominator: []const Mir.Reg) Layout.Error!struct { []const Mir.Reg, []const Mir.Reg } {
    const len = numerator.len;
    const zero = try builder.movImm(.b64, 0);
    const divisor = try builder.arena.alloc(Mir.Reg, len + 1);
    @memcpy(divisor[0..len], denominator);
    divisor[len] = zero;
    const classes = try builder.arena.alloc(Mir.Class, 2 * len + 1);
    @memset(classes[0 .. 2 * len], .b64);
    classes[2 * len] = .b32;
    const head = try builder.newBlock(classes);
    const exit = builder.labelBlock();
    const start = try builder.arena.alloc(Mir.Reg, 2 * len + 1);
    @memcpy(start[0..len], numerator);
    @memset(start[len .. 2 * len], zero);
    start[2 * len] = try builder.movImm(.b32, len * 64);
    try builder.branch(head, start);
    const params = try builder.startBlock(head);
    const doubled = try carryChain(builder, .add, params[0 .. 2 * len], params[0 .. 2 * len]);
    const remainder_extended = try builder.arena.alloc(Mir.Reg, len + 1);
    @memcpy(remainder_extended[0..len], doubled[len..]);
    remainder_extended[len] = zero;
    const difference = try carryChain(builder, .sub, remainder_extended, divisor);
    const take = try builder.emit(.pred, .setp, .s64, .{ .setp = .{ .cmp = .ge } }, .{ difference[len], 0 });
    const when_taken: Mir.Inst.Modifiers = .{ .none = .{ .guard = .positive } };
    const next = try builder.arena.alloc(Mir.Reg, 2 * len + 1);
    @memcpy(next[0 .. 2 * len], doubled);
    for (next[len .. 2 * len], difference[0..len]) |*limb, reduced| limb.* = try builder.emitGuarded(take, limb.*, .mov, .b64, when_taken, .{reduced});
    next[0] = try builder.emitGuarded(take, next[0], .@"or", .b64, when_taken, .{ next[0], 1 });
    next[2 * len] = try builder.emit(.b32, .sub, .u32, .{ .none = .{} }, .{ params[2 * len], 1 });
    const more = try builder.emit(.pred, .setp, .u32, .{ .setp = .{ .cmp = .ne } }, .{ next[2 * len], 0 });
    try builder.branchIf(more, true, exit, &.{});
    try builder.branch(head, next);
    _ = try builder.startBlock(exit);
    return .{ next[0..len], next[len .. 2 * len] };
}

pub fn powerOfTwoLimbs(divisor: []const Builder.Part, wide_int: Layout.WideInt) ?u32 {
    var exponent: ?u32 = null;
    for (divisor, 0..) |operand, i| {
        const imm = switch (operand) {
            .imm => |imm| imm,
            else => return null,
        };
        if (imm == 0) continue;
        if (exponent != null or !std.math.isPowerOfTwo(imm)) return null;
        exponent = @intCast(i * 64 + @ctz(imm));
    }
    const result = exponent orelse return null;
    if (result >= wide_int.bits - @intFromBool(wide_int.signedness == .signed)) return null;
    return result;
}

pub fn divPowerOfTwoLimbs(builder: *Builder, op: int.BinaryOp, a: []const Mir.Reg, exponent: u32, wide_int: Layout.WideInt) Layout.Error![]const Mir.Reg {
    const amount: ShiftAmount = .{ .known = exponent };
    const mask = try builder.arena.alloc(u64, a.len);
    for (mask, 0..) |*limb, i| limb.* = Layout.lowOnes(exponent, @intCast(i));
    if (wide_int.signedness == .unsigned or op == .div_floor or op == .mod or op == .div) switch (op) {
        .rem, .mod => {
            const out = try builder.arena.alloc(Mir.Reg, a.len);
            for (out, a, mask) |*result, limb, limb_mask| result.* = try builder.emit(.b64, .@"and", .b64, .{ .none = .{} }, .{ limb, limb_mask });
            return out;
        },
        else => return shiftLimbs(builder, .right, a, amount, wide_int.signedness),
    };
    // For a negative dividend, truncating division adds `divisor - 1` before the arithmetic shift.
    const sign = try fillLimb(builder, a[a.len - 1], .signed);
    const bias = try builder.arena.alloc(Mir.Reg, a.len);
    for (bias, mask) |*limb, limb_mask| limb.* = try builder.emit(.b64, .@"and", .b64, .{ .none = .{} }, .{ sign, limb_mask });
    const quotient = try shiftLimbs(builder, .right, try carryChain(builder, .add, a, bias), amount, .signed);
    if (op == .div_trunc) return quotient;
    return carryChain(builder, .sub, a, try shiftLimbs(builder, .left, quotient, amount, .signed));
}

pub fn divideLimbs(builder: *Builder, op: int.BinaryOp, a: []const Mir.Reg, b: []const Mir.Reg, wide_int: Layout.WideInt) Layout.Error![]const Mir.Reg {
    if (wide_int.signedness == .unsigned) {
        const quotient, const remainder = try divideUnsigned(builder, a, b);
        return if (op == .rem or op == .mod) remainder else quotient;
    }
    const a_sign = try fillLimb(builder, a[a.len - 1], .signed);
    const b_sign = try fillLimb(builder, b[b.len - 1], .signed);
    const magnitude_quotient, const magnitude_remainder = try divideUnsigned(builder, try negateIf(builder, a, a_sign), try negateIf(builder, b, b_sign));
    const signs = try builder.emit(.b64, .xor, .b64, .{ .none = .{} }, .{ a_sign, b_sign });
    const quotient = try negateIf(builder, magnitude_quotient, signs);
    const remainder = try negateIf(builder, magnitude_remainder, a_sign);
    switch (op) {
        .div, .div_trunc => return quotient,
        .rem => return remainder,
        .div_floor, .mod => {},
        else => unreachable,
    }
    // When the remainder is nonzero and the operand signs differ, floor division subtracts one from
    // the truncated quotient. In that case, `@mod` adds `b` to the remainder.
    const zeros = try builder.arena.alloc(Mir.Reg, a.len);
    @memset(zeros, try builder.movImm(.b64, 0));
    const nonzero = try compareLimbs(builder, .neq, magnitude_remainder, zeros, .unsigned);
    const adjust = try builder.emit(.pred, .setp, .u64, .{ .setp = .{ .cmp = .ne, .bool_op = .@"and" } }, .{ signs, 0, nonzero });
    const mask = try builder.emit(.b64, .selp, .b64, .{ .none = .{} }, .{ std.math.maxInt(u64), 0, adjust });
    const term = try builder.arena.alloc(Mir.Reg, a.len);
    if (op == .div_floor) {
        @memset(term, mask);
        return carryChain(builder, .add, quotient, term);
    }
    for (term, b) |*limb, b_limb| limb.* = try builder.emit(.b64, .@"and", .b64, .{ .none = .{} }, .{ b_limb, mask });
    return carryChain(builder, .add, remainder, term);
}

pub fn countBits(builder: *Builder, op: int.UnaryOp, a: []const Mir.Reg, wide_int: Layout.WideInt, result: Kind) Layout.Error!Mir.Reg {
    const top = wide_int.topLimb();
    const top_bits = wide_int.limbKind(top).bits;
    const limbs = try builder.arena.dupe(Mir.Reg, a[0 .. top + 1]);
    if (wide_int.signedness == .signed and top_bits < 64) {
        limbs[top] = try builder.emit(.b64, .@"and", .b64, .{ .none = .{} }, .{ limbs[top], Layout.lowOnes(top_bits, 0) });
    }
    const unless_found: Mir.Inst.Modifiers = .{ .none = .{ .guard = .negative } };
    const total = switch (op) {
        .popcount => total: {
            var total = try builder.emit(.b32, .popc, .b64, .{ .none = .{} }, .{limbs[0]});
            for (limbs[1..]) |limb| {
                const count = try builder.emit(.b32, .popc, .b64, .{ .none = .{} }, .{limb});
                total = try builder.emit(.b32, .add, .u32, .{ .none = .{} }, .{ total, count });
            }
            break :total total;
        },
        .clz => total: {
            var total = try builder.emit(.b32, .clz, .b64, .{ .none = .{} }, .{limbs[top]});
            if (top_bits < 64) total = try builder.emit(.b32, .sub, .u32, .{ .none = .{} }, .{ total, 64 - top_bits });
            var found = try builder.emit(.pred, .setp, .u64, .{ .setp = .{ .cmp = .ne } }, .{ limbs[top], 0 });
            var i = top;
            while (i > 0) {
                i -= 1;
                const count = try builder.emit(.b32, .clz, .b64, .{ .none = .{} }, .{limbs[i]});
                total = try builder.emitGuarded(found, total, .add, .u32, unless_found, .{ total, count });
                found = try builder.emit(.pred, .setp, .u64, .{ .setp = .{ .cmp = .ne, .bool_op = .@"or" } }, .{ limbs[i], 0, found });
            }
            break :total total;
        },
        .ctz => total: {
            var total: Mir.Reg = undefined;
            var found: Mir.Reg = undefined;
            for (limbs, 0..) |limb, i| {
                const reversed = try builder.emit(.b64, .brev, .b64, .{ .none = .{} }, .{limb});
                const count = try builder.emit(.b32, .clz, .b64, .{ .none = .{} }, .{reversed});
                if (i == 0) {
                    total = count;
                } else {
                    total = try builder.emitGuarded(found, total, .add, .u32, unless_found, .{ total, count });
                }
                found = if (i == 0)
                    try builder.emit(.pred, .setp, .u64, .{ .setp = .{ .cmp = .ne } }, .{ limb, 0 })
                else
                    try builder.emit(.pred, .setp, .u64, .{ .setp = .{ .cmp = .ne, .bool_op = .@"or" } }, .{ limb, 0, found });
            }
            break :total try builder.emit(.b32, .min, .u32, .{ .none = .{} }, .{ total, wide_int.bits });
        },
        else => unreachable,
    };
    return int.intConvert(builder, total, Kind.int(.unsigned, 32), result);
}

pub fn reverseLimbs(builder: *Builder, op: int.UnaryOp, a: []const Mir.Reg, wide_int: Layout.WideInt) Layout.Error![]const Mir.Reg {
    const reversed = try builder.arena.alloc(Mir.Reg, a.len);
    for (reversed, 0..) |*result, i| {
        const limb = a[a.len - 1 - i];
        result.* = if (op == .byte_swap)
            try int.unary(builder, .byte_swap, limb, Kind.int(.unsigned, 64), Kind.int(.unsigned, 64))
        else
            try builder.emit(.b64, .brev, .b64, .{ .none = .{} }, .{limb});
    }
    const extension_bits = a.len * 64 - wide_int.bits;
    if (extension_bits == 0) return reversed;
    return normalizedLimbs(builder, try shiftLimbs(builder, .right, reversed, .{ .known = extension_bits }, .unsigned), wide_int);
}
