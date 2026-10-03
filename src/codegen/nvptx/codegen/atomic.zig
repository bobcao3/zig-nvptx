//! This file lowers atomic loads, stores, read-modify-write and compare-and-swap operations.

const std = @import("std");
const AtomicOrder = std.lang.AtomicOrder;
const AtomicRmwOp = std.lang.AtomicRmwOp;

const Mir = @import("../Mir.zig");
const Kind = @import("Kind.zig");
const Layout = @import("Layout.zig");
const Builder = @import("Builder.zig");
const int = @import("int.zig");
const float = @import("float.zig");

/// `Sync` records which PTX memory-order qualifiers a Zig atomic order needs at `.gpu` scope. The
/// backend emits a `fence.sc.gpu` before a `seq_cst` access (PTX ISA 9.4, sections 8.8 and 8.9.3).
pub const Sync = struct {
    acquire: bool,
    release: bool,
    sequential: bool,

    pub fn init(order: AtomicOrder) Sync {
        return .{
            .acquire = order == .acquire or order == .acq_rel or order == .seq_cst,
            .release = order == .release or order == .acq_rel or order == .seq_cst,
            .sequential = order == .seq_cst,
        };
    }

    pub fn merge(a: Sync, b: Sync) Sync {
        return .{
            .acquire = a.acquire or b.acquire,
            .release = a.release or b.release,
            .sequential = a.sequential or b.sequential,
        };
    }

    fn forLoad(sync: Sync) Mir.Sem {
        return if (sync.acquire) .acquire else .relaxed;
    }

    fn forStore(sync: Sync) Mir.Sem {
        return if (sync.release) .release else .relaxed;
    }

    fn forRmw(sync: Sync) Mir.Sem {
        if (sync.acquire and sync.release) return .acq_rel;
        if (sync.acquire) return .acquire;
        if (sync.release) return .release;
        return .relaxed;
    }
};

/// `atom.cas.b16` and `atom.add.noftz.f16` need PTX 6.3 (PTX ISA 9.4, section 9.7.15.5).
const atom_16_bit_ptx_version = 63;

/// An atomic address is a multiple of the access size (PTX ISA 9.4, section 6.4.1), so
/// `log2_align` stays 0.
fn memOperand(space: Mir.Space, sem: Mir.Sem, size: u64) Mir.MemOperand {
    return .{ .space = space, .sem = sem, .scope = .gpu, .log2_size = @intCast(std.math.log2_int(u64, size)) };
}

pub const CompareExchange = struct {
    old: Mir.Reg,
    equal: Mir.Reg,
};

/// Rejects targets below sm_70, because the `.sem` qualifiers need PTX 6.0 and sm_70 (PTX ISA 9.4,
/// sections 9.7.10.8, 9.7.10.11, 9.7.15.4 and 9.7.15.5). `begin` then emits the `seq_cst` fence.
fn begin(builder: *Builder, layout: *const Layout, sync: Sync) Layout.Error!void {
    if (layout.smVersion() < 70) return layout.fail("atomic operations need sm_70 or later", .{});
    builder.needPtx(60);
    if (sync.sequential) try builder.emitEffect(.fence, .none, .{ .fence = .{ .sem = .sc, .scope = .gpu } }, .{});
}

pub fn load(builder: *Builder, layout: *const Layout, ptr: Mir.Reg, space: Mir.Space, order: AtomicOrder, kind: Kind) Layout.Error!Mir.Reg {
    const sync: Sync = .init(order);
    try begin(builder, layout, sync);
    const modifiers: Mir.Inst.Modifiers = .{ .mem = .{} };
    if (kind.class == .pred) {
        const byte = try builder.emit(.b16, .ld, .u8, modifiers, .{ ptr, memOperand(space, sync.forLoad(), 1) });
        return builder.emit(.pred, .setp, .u16, .{ .setp = .{ .cmp = .ne } }, .{ byte, 0 });
    }
    const value = try builder.emit(kind.class, .ld, kind.memoryType(), modifiers, .{ ptr, memOperand(space, sync.forLoad(), kind.size) });
    return if (kind.bits % 8 != 0) int.normalized(builder, value, kind) else value;
}

pub fn store(builder: *Builder, layout: *const Layout, ptr: Mir.Reg, space: Mir.Space, order: AtomicOrder, kind: Kind, value: Mir.Reg) Layout.Error!void {
    const sync: Sync = .init(order);
    try begin(builder, layout, sync);
    const stored = if (kind.class == .pred) try toRaw(builder, value, kind) else value;
    const ptx_type: Mir.Type = if (kind.class == .pred) .u8 else kind.memoryType();
    const size: u64 = if (kind.class == .pred) 1 else kind.size;
    try builder.emitEffect(.st, ptx_type, .{ .mem = .{} }, .{ ptr, stored, memOperand(space, sync.forStore(), size) });
}

pub fn rmw(builder: *Builder, layout: *const Layout, ptr: Mir.Reg, space: Mir.Space, order: AtomicOrder, op: AtomicRmwOp, kind: Kind, operand: Mir.Reg) Layout.Error!Mir.Reg {
    const sync: Sync = .init(order);
    try begin(builder, layout, sync);
    if (try nativeRmw(builder, ptr, space, sync, op, kind, operand)) |old| return old;
    return (try casLoop(builder, ptr, space, sync, kind, .{ .rmw = op }, operand, operand)).old;
}

pub fn compareExchange(builder: *Builder, layout: *const Layout, ptr: Mir.Reg, space: Mir.Space, sync: Sync, kind: Kind, expected: Mir.Reg, new: Mir.Reg) Layout.Error!CompareExchange {
    try begin(builder, layout, sync);
    if (kind.class == .pred or kind.bits != @as(u16, kind.size) * 8 or kind.size == 1) {
        const result = try casLoop(builder, ptr, space, sync, kind, .compare, expected, new);
        return .{ .old = result.old, .equal = result.equal.? };
    }
    if (kind.size == 2) builder.needPtx(atom_16_bit_ptx_version);
    const expected_raw = try toRaw(builder, expected, kind);
    const new_raw = try toRaw(builder, new, kind);
    const modifiers: Mir.Inst.Modifiers = .{ .atom = .{ .op = .cas } };
    const previous = try builder.emit(expected_raw.class, .atom, expected_raw.class.bitType(), modifiers, .{ ptr, expected_raw, new_raw, memOperand(space, sync.forRmw(), kind.size) });
    const equal = try builder.emit(.pred, .setp, expected_raw.class.unsignedType(), .{ .setp = .{ .cmp = .eq } }, .{ previous, expected_raw });
    return .{ .old = try fromRaw(builder, previous, kind), .equal = equal };
}

fn rawClass(size: u8) Mir.Class {
    return switch (size) {
        1, 2 => .b16,
        4 => .b32,
        else => .b64,
    };
}

fn fromRaw(builder: *Builder, raw: Mir.Reg, kind: Kind) Layout.Error!Mir.Reg {
    if (kind.class == .pred) return builder.emit(.pred, .setp, .u16, .{ .setp = .{ .cmp = .ne } }, .{ raw, 0 });
    if (kind.class != raw.class) return builder.emit(kind.class, .mov, kind.bitType(), .{ .none = .{} }, .{raw});
    if (kind.repr == .float) return raw;
    return int.normalized(builder, raw, kind);
}

pub fn toRaw(builder: *Builder, value: Mir.Reg, kind: Kind) Layout.Error!Mir.Reg {
    if (kind.class == .pred) return builder.emit(.b16, .selp, .u16, .{ .none = .{} }, .{ 1, 0, value });
    const class = rawClass(kind.size);
    if (class == value.class) return value;
    return builder.emit(class, .mov, kind.bitType(), .{ .none = .{} }, .{value});
}

/// Compares floats by bit pattern to match `atom.cas`, so a NaN equals itself and -0.0 != 0.0.
pub fn bitEqual(builder: *Builder, a: Mir.Reg, b: Mir.Reg, kind: Kind) Layout.Error!Mir.Reg {
    if (kind.repr != .float) return int.compareReg(builder, .eq, a, b, kind);
    const raw_a = try toRaw(builder, a, kind);
    const raw_b = try toRaw(builder, b, kind);
    return builder.emit(.pred, .setp, raw_a.class.unsignedType(), .{ .setp = .{ .cmp = .eq } }, .{ raw_a, raw_b });
}

pub fn rmwValue(builder: *Builder, op: AtomicRmwOp, old: Mir.Reg, operand: Mir.Reg, kind: Kind) Layout.Error!Mir.Reg {
    const binary_op: int.BinaryOp = switch (op) {
        .Xchg => return operand,
        .Add => .add,
        .Sub => .sub,
        .And, .Nand => .@"and",
        .Or => .@"or",
        .Xor => .xor,
        .Max => .max,
        .Min => .min,
    };
    const result = if (kind.repr == .float)
        try float.binary(builder, binary_op, old, .{ .reg = operand }, kind, true)
    else
        try int.binaryReg(builder, binary_op, old, operand, kind);
    if (op != .Nand) return result;
    return int.normalized(builder, try builder.emit(kind.class, .not, kind.bitType(), .{ .none = .{} }, .{result}), kind);
}

/// Returns null when `atom` has no form for the operation (PTX ISA 9.4, section 9.7.15.5) or when
/// `casLoop` must extend an odd-width integer. Plain `atom.add.f32` flushes subnormals on global
/// memory, and `.noftz` needs PTX 9.4 for `.f32`, so only `.shared` memory gets `atom.add.f32`.
fn nativeRmw(builder: *Builder, ptr: Mir.Reg, space: Mir.Space, sync: Sync, op: AtomicRmwOp, kind: Kind, operand: Mir.Reg) Layout.Error!?Mir.Reg {
    if (kind.class == .pred or kind.bits != @as(u16, kind.size) * 8) return null;
    const is_float = kind.repr == .float;
    const word: Mir.Type = if (kind.size == 4) .b32 else .b64;
    var negate: ?Mir.Type = null;
    const atom_op: Mir.AtomOp, const ptx_type: Mir.Type = if (is_float) switch (op) {
        .Xchg => if (kind.size == 2) return null else .{ .exch, word },
        .Add, .Sub => add: {
            if (op == .Sub) negate = kind.arithmeticType();
            break :add .{ .add, switch (kind.size) {
                2 => .f16,
                4 => if (space == .shared) .f32 else return null,
                else => .f64,
            } };
        },
        else => return null,
    } else switch (op) {
        .Xchg => .{ .exch, word },
        .Add, .Sub => add: {
            if (op == .Sub) negate = if (kind.size == 4) .s32 else .s64;
            break :add .{ .add, if (kind.size == 4) .u32 else .u64 };
        },
        .And => .{ .@"and", word },
        .Or => .{ .@"or", word },
        .Xor => .{ .xor, word },
        .Max => .{ .max, kind.arithmeticType() },
        .Min => .{ .min, kind.arithmeticType() },
        .Nand => return null,
    };
    if (!is_float and kind.size < 4) return null;
    if (kind.size == 2) builder.needPtx(atom_16_bit_ptx_version);
    var src = operand;
    if (negate) |negate_type| src = try builder.emit(kind.class, .neg, negate_type, .{ .none = .{} }, .{operand});
    if (is_float and op == .Xchg) src = try toRaw(builder, src, kind);
    const modifiers: Mir.Inst.Modifiers = .{ .atom = .{
        .op = atom_op,
        .noftz = is_float and kind.size == 2 and atom_op == .add,
    } };
    const old = try builder.emit(src.class, .atom, ptx_type, modifiers, .{ ptr, src, memOperand(space, sync.forRmw(), kind.size) });
    return if (is_float and op == .Xchg) try fromRaw(builder, old, kind) else old;
}

const CasAction = union(enum) {
    rmw: AtomicRmwOp,
    compare,
};

const CasResult = struct {
    old: Mir.Reg,
    equal: ?Mir.Reg,
};

/// The loop handles the values and operations that `atom` cannot update directly. `atom.cas` has no
/// 8-bit form (PTX ISA 9.4, section 9.7.15.5), so the loop updates a 1-byte value inside its
/// aligned 16-bit word. The byte sits at bit `8 * (ptr & 1)` of the word.
fn casLoop(builder: *Builder, ptr: Mir.Reg, space: Mir.Space, sync: Sync, kind: Kind, action: CasAction, operand: Mir.Reg, new: Mir.Reg) Layout.Error!CasResult {
    const part_word = kind.size == 1;
    if (kind.size <= 2) builder.needPtx(atom_16_bit_ptx_version);
    const raw_class = rawClass(kind.size);
    const raw_type = raw_class.bitType();
    var address = ptr;
    var shift: Mir.Reg = undefined;
    var keep: Mir.Reg = undefined;
    if (part_word) {
        address = try builder.emit(ptr.class, .@"and", ptr.class.bitType(), .{ .none = .{} }, .{ ptr, Mir.Imm{ .value = @bitCast(@as(i64, -2)), .format = .signed } });
        const low = try builder.emit(ptr.class, .@"and", ptr.class.bitType(), .{ .none = .{} }, .{ ptr, 1 });
        const low32 = if (low.class == .b32) low else try builder.emit(.b32, .cvt, .u32, .{ .cvt = .{ .from = .u64 } }, .{low});
        shift = try builder.emit(.b32, .shl, .b32, .{ .none = .{} }, .{ low32, 3 });
        const mask = try builder.emit(.b16, .shl, .b16, .{ .none = .{} }, .{ 255, shift });
        keep = try builder.emit(.b16, .not, .b16, .{ .none = .{} }, .{mask});
    }
    const raw_size: u64 = raw_class.bits() / 8;
    const initial = try builder.emit(raw_class, .ld, raw_type, .{ .mem = .{} }, .{ address, memOperand(space, .relaxed, raw_size) });
    const head = try builder.newBlock(&.{raw_class});
    const exit = builder.labelBlock();
    try builder.branch(head, &.{initial});
    const current = (try builder.startBlock(head))[0];
    const field = if (part_word) field: {
        const shifted = try builder.emit(.b16, .shr, .u16, .{ .none = .{} }, .{ current, shift });
        break :field try builder.emit(.b16, .@"and", .b16, .{ .none = .{} }, .{ shifted, 255 });
    } else current;
    const old = try fromRaw(builder, field, kind);
    var equal: ?Mir.Reg = null;
    const next = switch (action) {
        .rmw => |op| try rmwValue(builder, op, old, operand, kind),
        .compare => compare: {
            equal = try bitEqual(builder, old, operand, kind);
            try builder.branchIf(equal.?, true, exit, &.{});
            break :compare new;
        },
    };
    var next_raw = try toRaw(builder, next, kind);
    if (part_word) {
        const byte = try builder.emit(.b16, .@"and", .b16, .{ .none = .{} }, .{ next_raw, 255 });
        const positioned = try builder.emit(.b16, .shl, .b16, .{ .none = .{} }, .{ byte, shift });
        const kept = try builder.emit(.b16, .@"and", .b16, .{ .none = .{} }, .{ current, keep });
        next_raw = try builder.emit(.b16, .@"or", .b16, .{ .none = .{} }, .{ positioned, kept });
    }
    const previous = try builder.emit(raw_class, .atom, raw_type, .{ .atom = .{ .op = .cas } }, .{ address, current, next_raw, memOperand(space, sync.forRmw(), raw_size) });
    const changed = try builder.emit(.pred, .setp, raw_class.unsignedType(), .{ .setp = .{ .cmp = .ne } }, .{ previous, current });
    try builder.branchIf(changed, true, exit, &.{});
    try builder.branch(head, &.{previous});
    _ = try builder.startBlock(exit);
    return .{ .old = old, .equal = equal };
}
