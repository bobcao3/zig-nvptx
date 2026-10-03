//! This file makes a loop exit branch positive, because ptxas unrolls a loop only when the exit
//! branch has a positive guard. Input: a `setp` without a guard or a boolean operator whose only
//! user is a `bra` with a negative guard. Rule: the `setp` computes the complement comparison and
//! the `bra` takes a positive guard. Proof: the complement is true exactly when the comparison is
//! false (PTX ISA 9.4, section 9.7.7.2). A float complement exchanges ordered and unordered
//! comparisons, because a NaN operand makes an ordered comparison false.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Mir = @import("../Mir.zig");
const Cfg = @import("../Cfg.zig");

const none = Cfg.none;

pub fn run(gpa: Allocator, mir: *Mir) Allocator.Error!void {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const pred_count = mir.reg_counts[@backingInt(Mir.Class.pred)];
    var scan: Scan = .{
        .mir = mir,
        .uses = try arena.alloc(u32, pred_count),
        .defs = try arena.alloc(u32, pred_count),
    };
    @memset(scan.uses, 0);
    @memset(scan.defs, none);
    for (0..mir.instructions.len) |i| {
        scan.inst = @intCast(i);
        mir.visitRegs(mir.instructions.get(i), &scan);
    }
    for (0..mir.instructions.len) |i| {
        var branch = mir.instructions.get(i);
        if (branch.tag != .bra or branch.modifiers.guard() != .negative) continue;
        const pred: Mir.Operand = @bitCast(mir.extra[branch.data]);
        if (scan.uses[pred.regIndex()] != 1) continue;
        const def = scan.defs[pred.regIndex()];
        if (def == none) continue;
        var compare = mir.instructions.get(def);
        if (compare.tag != .setp or compare.modifiers.setp.guard != .none or compare.modifiers.setp.bool_op != .none) continue;
        compare.modifiers.setp.cmp = complement(compare.modifiers.setp.cmp, compare.ty) orelse continue;
        branch.modifiers = .{ .none = .{ .guard = .positive } };
        mir.instructions.set(def, compare);
        mir.instructions.set(i, branch);
    }
}

const Scan = struct {
    mir: *const Mir,
    uses: []u32,
    defs: []u32,
    inst: u32 = 0,

    pub fn reg(scan: *Scan, at: u32, role: Mir.RegRole) void {
        const operand: Mir.Operand = @bitCast(scan.mir.extra[at]);
        if (operand.regClass() != .pred) return;
        switch (role) {
            .def => scan.defs[operand.regIndex()] = scan.inst,
            else => scan.uses[operand.regIndex()] += 1,
        }
    }
};

fn complement(cmp: Mir.Cmp, ty: Mir.Type) ?Mir.Cmp {
    return switch (ty) {
        .f16, .bf16, .f32, .f64 => switch (cmp) {
            .eq => .neu,
            .ne => .equ,
            .lt => .geu,
            .le => .gtu,
            .gt => .leu,
            .ge => .ltu,
            .equ => .ne,
            .neu => .eq,
            .ltu => .ge,
            .leu => .gt,
            .gtu => .le,
            .geu => .lt,
            .num => .nan,
            .nan => .num,
            .lo, .ls, .hi, .hs => null,
        },
        .b16, .b32, .b64, .u8, .u16, .u32, .u64, .s8, .s16, .s32, .s64 => switch (cmp) {
            .eq => .ne,
            .ne => .eq,
            .lt => .ge,
            .ge => .lt,
            .le => .gt,
            .gt => .le,
            .lo => .hs,
            .hs => .lo,
            .ls => .hi,
            .hi => .ls,
            .equ, .neu, .ltu, .leu, .gtu, .geu, .num, .nan => null,
        },
        else => null,
    };
}

fn holds(cmp: Mir.Cmp, a: f64, b: f64) bool {
    const unordered = std.math.isNan(a) or std.math.isNan(b);
    return switch (cmp) {
        .eq => !unordered and a == b,
        .ne => !unordered and a != b,
        .lt, .lo => !unordered and a < b,
        .le, .ls => !unordered and a <= b,
        .gt, .hi => !unordered and a > b,
        .ge, .hs => !unordered and a >= b,
        .equ => unordered or a == b,
        .neu => unordered or a != b,
        .ltu => unordered or a < b,
        .leu => unordered or a <= b,
        .gtu => unordered or a > b,
        .geu => unordered or a >= b,
        .num => !unordered,
        .nan => unordered,
    };
}

test "the complement of a comparison holds exactly when the comparison fails" {
    const values = [_]f64{ -std.math.inf(f64), -1, 0, 1, std.math.inf(f64), std.math.nan(f64) };
    const types = [_]Mir.Type{ .f16, .bf16, .f32, .f64, .u32, .s32, .u64, .s64, .u16, .b32 };
    for (types) |ty| {
        for (std.enums.values(Mir.Cmp)) |cmp| {
            const other = complement(cmp, ty) orelse continue;
            try std.testing.expectEqual(cmp, complement(other, ty).?);
            for (values) |a| for (values) |b| {
                if (ty != .f16 and ty != .bf16 and ty != .f32 and ty != .f64 and (std.math.isNan(a) or std.math.isNan(b))) continue;
                try std.testing.expect(holds(cmp, a, b) != holds(other, a, b));
            };
        }
    }
}
