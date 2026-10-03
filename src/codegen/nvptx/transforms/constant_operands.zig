//! This file rewrites two forms with a constant operand that ptxas 13.0 and 13.4 compile wrongly
//! for sm_120 and sm_121, also when the driver compiles the PTX of an older target for such a GPU.
//! Input: SSA MIR after the other link-time passes. A constant is an immediate or a register that
//! one unguarded `mov` of an immediate defines. Rules:
//! - An unguarded 64-bit `add` or `sub` of a constant outside the signed 32-bit range, with `.cc`
//!   or outside an open carry chain, runs on the 32-bit halves with `add.cc` and `addc`, or with
//!   `sub.cc` and `subc`. ptxas otherwise can give the sum for a call result the value of the
//!   sum for an earlier call result.
//! - An unguarded 32-bit `div` or `rem` of a register by a nonzero constant becomes a multiply-high
//!   sequence (Granlund and Montgomery, as LLVM computes it). ptxas otherwise can keep the divisor,
//!   converted to float, in a uniform register that the other side of a divergent branch also uses.
//! Output: SSA MIR with new registers for the partial results. Proof: the halves with the carry
//! give the 64-bit sum (PTX ISA 9.4, section 9.7.2). Granlund and Montgomery prove that each
//! sequence gives the truncated quotient q, and `x - q * d` is the remainder that `@rem` defines.

const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const Mir = @import("../Mir.zig");

const none = std.math.maxInt(u32);

const Error = Allocator.Error || error{TooManyRegisters};

pub fn run(gpa: Allocator, mir: *Mir) Error!void {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var scan: Scan = .{ .mir = mir, .defs = .{
        try arena.alloc(u32, mir.reg_counts[@backingInt(Mir.Class.b32)]),
        try arena.alloc(u32, mir.reg_counts[@backingInt(Mir.Class.b64)]),
    } };
    for (scan.defs) |defs| @memset(defs, none);
    for (0..mir.instructions.len) |i| {
        scan.inst = @intCast(i);
        mir.visitRegs(mir.instructions.get(i), &scan);
    }
    const forms = try arena.alloc(Form, mir.instructions.len);
    var carry_open = false;
    for (forms, 0..) |*f, i| {
        const inst = mir.instructions.get(i);
        f.* = scan.form(inst, carry_open);
        carry_open = switch (inst.tag) {
            .add, .sub, .mad => carry_open or inst.modifiers.arith.cc,
            .addc, .subc, .madc => inst.modifiers.arith.cc,
            else => carry_open,
        };
    }
    for (forms) |f| {
        if (f != .none) break;
    } else return;
    var b: Mir.Builder = .{ .gpa = gpa, .reg_counts = mir.reg_counts, .label_count = mir.label_count };
    defer b.deinit();
    for (forms, 0..) |f, i| {
        const inst = mir.instructions.get(i);
        switch (f) {
            .none => try b.addTyped(inst.tag, inst.ty, inst.modifiers, mir.span(inst)),
            .split => try split(&b, &scan, inst),
            .divide => |divisor| try divide(&b, mir, inst, divisor),
        }
    }
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

const Form = union(enum) { none, split, divide: u32 };

const Scan = struct {
    mir: *const Mir,
    /// The instruction that defines each `.b32` register and each `.b64` register.
    defs: [2][]u32,
    inst: u32 = 0,

    pub fn reg(scan: *Scan, at: u32, role: Mir.RegRole) void {
        if (role == .use) return;
        const op: Mir.Operand = @bitCast(scan.mir.extra[at]);
        const defs = scan.classDefs(op.regClass()) orelse return;
        defs[op.regIndex()] = scan.inst;
    }

    fn classDefs(scan: *const Scan, class: Mir.Class) ?[]u32 {
        return switch (class) {
            .b32 => scan.defs[0],
            .b64 => scan.defs[1],
            else => null,
        };
    }

    fn constant(scan: *const Scan, at: u32) ?u64 {
        const mir = scan.mir;
        const op: Mir.Operand = @bitCast(mir.extra[at]);
        switch (op.kind) {
            .imm => return op.payload,
            .imm64 => {
                const format: Mir.ImmFormat = @fromBackingInt(@intCast(op.payload));
                if (format != .unsigned and format != .signed) return null;
                return @as(u64, mir.extra[at + 1]) | (@as(u64, mir.extra[at + 2]) << 32);
            },
            .reg => {
                const def = (scan.classDefs(op.regClass()) orelse return null)[op.regIndex()];
                if (def == none) return null;
                const inst = mir.instructions.get(def);
                if (inst.tag != .mov or inst.modifiers.guard() != .none) return null;
                const source = inst.data + @as(Mir.Operand, @bitCast(mir.extra[inst.data])).len();
                if (@as(Mir.Operand, @bitCast(mir.extra[source])).kind == .reg) return null;
                return scan.constant(source);
            },
            else => return null,
        }
    }

    fn form(scan: *const Scan, inst: Mir.Inst, carry_open: bool) Form {
        if (inst.modifiers.guard() != .none) return .none;
        const lhs = inst.data + 1;
        switch (inst.tag) {
            .add, .sub => {
                if ((inst.ty != .u64 and inst.ty != .s64) or inst.modifiers.arith.mode != .none) return .none;
                if (carry_open and !inst.modifiers.arith.cc) return .none;
                const rhs = lhs + operandLen(scan.mir, lhs);
                var wide = false;
                for ([_]u32{ lhs, rhs }) |at| {
                    if (scan.constant(at)) |value| {
                        const signed: i64 = @bitCast(value);
                        wide = wide or signed < std.math.minInt(i32) or signed > std.math.maxInt(i32);
                    } else if (operandKind(scan.mir, at) != .reg) return .none;
                }
                return if (wide) .split else .none;
            },
            .div, .rem => {
                if ((inst.ty != .u32 and inst.ty != .s32) or operandKind(scan.mir, lhs) != .reg) return .none;
                const divisor: u32 = @truncate(scan.constant(lhs + operandLen(scan.mir, lhs)) orelse return .none);
                return if (divisor == 0) .none else .{ .divide = divisor };
            },
            else => return .none,
        }
    }

    fn halves(scan: *const Scan, b: *Mir.Builder, at: u32) Error![2]Arg {
        if (scan.constant(at)) |value| return .{ .imm(.unsigned, @as(u32, @truncate(value))), .imm(.unsigned, value >> 32) };
        const low = try b.newReg(.b32);
        const high = try b.newReg(.b32);
        try b.addTyped(.mov, .b64, .{ .none = .{} }, &.{ list_of_two, low, high, scan.mir.extra[at] });
        return .{ .reg(low), .reg(high) };
    }
};

const list_of_two: u32 = @bitCast(Mir.Operand{ .kind = .list, .payload = 2 });

fn operandKind(mir: *const Mir, at: u32) Mir.OperandKind {
    return @as(Mir.Operand, @bitCast(mir.extra[at])).kind;
}

fn operandLen(mir: *const Mir, at: u32) u32 {
    return @as(Mir.Operand, @bitCast(mir.extra[at])).len();
}

/// A register or an immediate operand of a new instruction.
const Arg = struct {
    words: [3]u32,
    len: u32,

    fn reg(word: u32) Arg {
        return .{ .words = .{ word, 0, 0 }, .len = 1 };
    }

    fn imm(format: Mir.ImmFormat, value: u64) Arg {
        const op: Mir.Operand = .{ .kind = .imm64, .payload = @backingInt(format) };
        return .{ .words = .{ @bitCast(op), @truncate(value), @truncate(value >> 32) }, .len = 3 };
    }

    fn of(mir: *const Mir, at: u32) Arg {
        var arg: Arg = .{ .words = undefined, .len = operandLen(mir, at) };
        @memcpy(arg.words[0..arg.len], mir.extra[at..][0..arg.len]);
        return arg;
    }
};

fn emitTo(b: *Mir.Builder, result: u32, tag: Mir.Inst.Tag, ty: Mir.Type, modifiers: Mir.Inst.Modifiers, args: []const Arg) Allocator.Error!void {
    var words: [7]u32 = undefined;
    words[0] = result;
    var len: usize = 1;
    for (args) |arg| {
        @memcpy(words[len..][0..arg.len], arg.words[0..arg.len]);
        len += arg.len;
    }
    try b.addTyped(tag, ty, modifiers, words[0..len]);
}

fn emit(b: *Mir.Builder, tag: Mir.Inst.Tag, ty: Mir.Type, modifiers: Mir.Inst.Modifiers, args: []const Arg) Error!Arg {
    const result = try b.newReg(.b32);
    try emitTo(b, result, tag, ty, modifiers, args);
    return .reg(result);
}

fn split(b: *Mir.Builder, scan: *const Scan, inst: Mir.Inst) Error!void {
    const mir = scan.mir;
    const lhs_at = inst.data + 1;
    const lhs = try scan.halves(b, lhs_at);
    const rhs = try scan.halves(b, lhs_at + operandLen(mir, lhs_at));
    const low_tag: Mir.Inst.Tag, const high_tag: Mir.Inst.Tag = if (inst.tag == .add) .{ .add, .addc } else .{ .sub, .subc };
    const low = try emit(b, low_tag, .u32, .{ .arith = .{ .cc = true } }, &.{ lhs[0], rhs[0] });
    const high = try emit(b, high_tag, .u32, .{ .arith = .{ .cc = inst.modifiers.arith.cc } }, &.{ lhs[1], rhs[1] });
    try b.addTyped(.mov, .b64, .{ .none = .{} }, &.{ mir.extra[inst.data], list_of_two, low.words[0], high.words[0] });
}

fn divide(b: *Mir.Builder, mir: *const Mir, inst: Mir.Inst, divisor: u32) Error!void {
    const signed = inst.ty == .s32;
    const dividend: Arg = .of(mir, inst.data + 1);
    const quotient = try emitQuotient(b, dividend, plan(divisor, signed));
    const result = mir.extra[inst.data];
    if (inst.tag == .div) return emitTo(b, result, .mov, .b32, .{ .none = .{} }, &.{quotient});
    const d: Arg = if (signed) .imm(.signed, @bitCast(@as(i64, @as(i32, @bitCast(divisor))))) else .imm(.unsigned, divisor);
    const product = try emit(b, .mul, inst.ty, .{ .arith = .{ .mode = .lo } }, &.{ quotient, d });
    try emitTo(b, result, .sub, inst.ty, .{ .arith = .{} }, &.{ dividend, product });
}

fn shiftRight(b: *Mir.Builder, ty: Mir.Type, value: Arg, amount: u5) Error!Arg {
    if (amount == 0) return value;
    return emit(b, .shr, ty, .{ .none = .{} }, &.{ value, .imm(.unsigned, amount) });
}

fn emitQuotient(b: *Mir.Builder, x: Arg, p: Plan) Error!Arg {
    const arith: Mir.Inst.Modifiers = .{ .arith = .{} };
    const high: Mir.Inst.Modifiers = .{ .arith = .{ .mode = .hi } };
    switch (p) {
        .shift_unsigned => |shift| return shiftRight(b, .u32, x, shift),
        .unsigned => |u| {
            var q = try emit(b, .mul, .u32, high, &.{ try shiftRight(b, .u32, x, u.pre_shift), .imm(.unsigned, u.magic) });
            if (u.add) {
                const difference = try emit(b, .sub, .u32, arith, &.{ x, q });
                q = try emit(b, .add, .u32, arith, &.{ try shiftRight(b, .u32, difference, 1), q });
            }
            return shiftRight(b, .u32, q, u.post_shift);
        },
        .shift_signed => |s| {
            var q = x;
            if (s.shift != 0) {
                const sign = try shiftRight(b, .s32, x, 31);
                const bias = try shiftRight(b, .u32, sign, @intCast(32 - @as(u6, s.shift)));
                q = try shiftRight(b, .s32, try emit(b, .add, .s32, arith, &.{ x, bias }), s.shift);
            }
            return if (s.negate) emit(b, .neg, .s32, arith, &.{q}) else q;
        },
        .signed => |s| {
            var q = try emit(b, .mul, .s32, high, &.{ x, .imm(.signed, @bitCast(@as(i64, s.magic))) });
            switch (s.correction) {
                .none => {},
                .add => q = try emit(b, .add, .s32, arith, &.{ q, x }),
                .sub => q = try emit(b, .sub, .s32, arith, &.{ q, x }),
            }
            q = try shiftRight(b, .s32, q, s.shift);
            return emit(b, .add, .s32, arith, &.{ q, try shiftRight(b, .u32, q, 31) });
        },
    }
}

const Plan = union(enum) {
    shift_unsigned: u5,
    unsigned: struct { pre_shift: u5, magic: u32, add: bool, post_shift: u5 },
    shift_signed: struct { shift: u5, negate: bool },
    signed: struct { magic: i32, correction: enum { none, add, sub }, shift: u5 },
};

fn plan(divisor: u32, signed: bool) Plan {
    assert(divisor != 0);
    if (!signed) {
        if (std.math.isPowerOfTwo(divisor)) return .{ .shift_unsigned = @intCast(@ctz(divisor)) };
        return .{ .unsigned = unsignedMagic(divisor, 0) };
    }
    const d: i32 = @bitCast(divisor);
    const magnitude = @abs(d);
    if (std.math.isPowerOfTwo(magnitude)) return .{ .shift_signed = .{ .shift = @intCast(@ctz(magnitude)), .negate = d < 0 } };
    return signedMagic(d);
}

fn unsignedMagic(d: u32, leading_zeros: u5) @FieldType(Plan, "unsigned") {
    const all_ones = @as(u32, std.math.maxInt(u32)) >> leading_zeros;
    const signed_min: u32 = 1 << 31;
    const signed_max: u32 = signed_min - 1;
    const nc = all_ones - (all_ones +% 1 -% d) % d;
    var p: u32 = 31;
    var q1 = signed_min / nc;
    var r1 = signed_min % nc;
    var q2 = signed_max / d;
    var r2 = signed_max % d;
    var add = false;
    while (true) {
        p += 1;
        if (r1 >= nc - r1) {
            q1 = (q1 << 1) +% 1;
            r1 = (r1 << 1) -% nc;
        } else {
            q1 <<= 1;
            r1 <<= 1;
        }
        if (r2 +% 1 >= d - r2) {
            if (q2 >= signed_max) add = true;
            q2 = (q2 << 1) +% 1;
            r2 = (r2 << 1) +% 1 -% d;
        } else {
            if (q2 >= signed_min) add = true;
            q2 <<= 1;
            r2 = (r2 << 1) +% 1;
        }
        const delta = d - 1 - r2;
        if (p >= 64 or q1 > delta or (q1 == delta and r1 != 0)) break;
    }
    if (add and d & 1 == 0) {
        const pre_shift: u5 = @intCast(@ctz(d));
        var result = unsignedMagic(d >> pre_shift, leading_zeros + pre_shift);
        assert(!result.add and result.pre_shift == 0);
        result.pre_shift = pre_shift;
        return result;
    }
    return .{ .pre_shift = 0, .magic = q2 +% 1, .add = add, .post_shift = @intCast(p - 32 - @intFromBool(add)) };
}

fn signedMagic(d: i32) Plan {
    const signed_min: u32 = 1 << 31;
    const ad: u32 = @abs(d);
    const t = signed_min + @intFromBool(d < 0);
    const anc = t - 1 - t % ad;
    var p: u32 = 31;
    var q1 = signed_min / anc;
    var r1 = signed_min % anc;
    var q2 = signed_min / ad;
    var r2 = signed_min % ad;
    while (true) {
        p += 1;
        q1 <<= 1;
        r1 <<= 1;
        if (r1 >= anc) {
            q1 +%= 1;
            r1 -%= anc;
        }
        q2 <<= 1;
        r2 <<= 1;
        if (r2 >= ad) {
            q2 +%= 1;
            r2 -%= ad;
        }
        const delta = ad - r2;
        if (q1 > delta or (q1 == delta and r1 != 0)) break;
    }
    var magic: u32 = q2 +% 1;
    if (d < 0) magic = 0 -% magic;
    const signed_magic: i32 = @bitCast(magic);
    return .{ .signed = .{
        .magic = signed_magic,
        .correction = if (d > 0 and signed_magic < 0) .add else if (d < 0 and signed_magic > 0) .sub else .none,
        .shift = @intCast(p - 32),
    } };
}

/// Runs the instructions that `emitQuotient` writes for `divisor` on `dividend`.
fn emittedQuotient(divisor: u32, signed: bool, dividend: u32) !u32 {
    var b: Mir.Builder = .{ .gpa = std.testing.allocator, .reg_counts = @splat(0), .label_count = 0 };
    defer b.deinit();
    const x = try b.newReg(.b32);
    const quotient = try emitQuotient(&b, .reg(x), plan(divisor, signed));
    var regs: [32]u32 = undefined;
    regs[0] = dividend;
    for (0..b.instructions.len) |i| {
        const inst = b.instructions.get(i);
        const words = b.extra.items[inst.data..];
        var sources: [2]u32 = undefined;
        var at: usize = 1;
        for (0..if (inst.tag == .neg) 1 else 2) |k| {
            const op: Mir.Operand = @bitCast(words[at]);
            sources[k] = if (op.kind == .reg) regs[op.regIndex()] else words[at + 1];
            at += op.len();
        }
        const a, const c = sources;
        const signed_a: i32 = @bitCast(a);
        regs[@as(Mir.Operand, @bitCast(words[0])).regIndex()] = switch (inst.tag) {
            .mul => if (inst.ty == .s32) @bitCast(mulHigh(i32, signed_a, @bitCast(c))) else mulHigh(u32, a, c),
            .shr => if (inst.ty == .s32) @bitCast(signed_a >> @intCast(c)) else a >> @intCast(c),
            .add => a +% c,
            .sub => a -% c,
            .neg => 0 -% a,
            else => unreachable,
        };
    }
    return regs[@as(Mir.Operand, @bitCast(quotient.words[0])).regIndex()];
}

fn mulHigh(comptime T: type, a: T, b: T) T {
    const Wide = @Int(@typeInfo(T).int.signedness, 64);
    return @intCast((@as(Wide, a) * b) >> 32);
}

fn expectQuotients(divisor: u32, dividend: u32) !void {
    try std.testing.expectEqual(dividend / divisor, try emittedQuotient(divisor, false, dividend));
    const d: i32 = @bitCast(divisor);
    const x: i32 = @bitCast(dividend);
    if (d == -1 and x == std.math.minInt(i32)) return;
    try std.testing.expectEqual(@divTrunc(x, d), @as(i32, @bitCast(try emittedQuotient(divisor, true, dividend))));
}

test "the emitted sequences give the truncated quotient of div" {
    var prng: std.Random.DefaultPrng = .init(0x5eed);
    const random = prng.random();
    var divisors: [3 * 32 + 2000]u32 = undefined;
    for (0..32) |k| for (0..3) |i| {
        divisors[3 * k + i] = (@as(u32, 1) << @intCast(k)) +% @as(u32, @intCast(i)) -% 1;
    };
    for (divisors[3 * 32 ..][0..1000], 1..) |*d, i| d.* = @intCast(i);
    for (divisors[3 * 32 + 1000 ..]) |*d| d.* = random.intRangeAtMost(u32, 1, std.math.maxInt(u32));
    for (divisors) |d| {
        if (d == 0) continue;
        for ([_]u32{ 0, 1, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff, d -% 1, d, d +% 1, d *% 7 -% 1 }) |x| try expectQuotients(d, x);
        for (0..64) |_| try expectQuotients(d, random.int(u32));
    }
}

test "fuzz the emitted sequences against div" {
    try std.testing.fuzz({}, fuzzQuotients, .{});
}

fn fuzzQuotients(_: void, smith: *std.testing.Smith) anyerror!void {
    const divisor = smith.value(u32);
    if (divisor != 0) try expectQuotients(divisor, smith.value(u32));
}
