//! This file lowers loads, stores and copy loops of scalars.

const std = @import("std");
const assert = std.debug.assert;

const Mir = @import("../Mir.zig");
const Type = @import("../../../Type.zig");
const Kind = @import("Kind.zig");
const Builder = @import("Builder.zig");
const Layout = @import("Layout.zig");
const int = @import("int.zig");

pub fn memSpace(address_space: std.lang.AddressSpace) Mir.Space {
    return switch (address_space) {
        .global => .global,
        .shared => .shared,
        .constant => .@"const",
        .local => .local,
        .param => .param,
        else => .generic,
    };
}

pub fn volatileSem(address_space: std.lang.AddressSpace, is_volatile: bool) Mir.Sem {
    // No other thread can access `.local` memory, so the backend omits `.volatile` there. With
    // `.local`, `.volatile` would also need PTX 9.1 (PTX ISA 9.4, sections 9.7.10.8 and 9.7.10.11).
    return if (is_volatile and address_space != .local) .@"volatile" else .none;
}

/// Returns the `MemOperand` of an access of `size` bytes at an address that `alignment` divides.
/// A load without an order from `.const` memory or from a kernel parameter is invariant.
pub fn memOperand(builder: *const Builder, address_space: std.lang.AddressSpace, sem: Mir.Sem, size: u64, alignment: u64, is_load: bool) Mir.MemOperand {
    return .{
        .space = memSpace(address_space),
        .sem = sem,
        .log2_size = @intCast(std.math.log2_int(u64, size)),
        .log2_align = Mir.log2Align(alignment),
        .invariant = is_load and sem == .none and (address_space == .constant or (address_space == .param and builder.is_kernel)),
    };
}

fn mirVec(len: u32) Mir.Vec {
    return switch (len) {
        1 => .scalar,
        2 => .v2,
        4 => .v4,
        else => unreachable,
    };
}

fn memType(kind: Kind) Mir.MemType {
    return std.meta.stringToEnum(Mir.MemType, @tagName(kind.memoryType())).?;
}

pub fn offsetAlignment(alignment: u64, offset: u64) u64 {
    if (offset == 0) return alignment;
    return @min(alignment, offset & -%offset);
}

/// Returns 4, 2 or 1, the number of scalars that one access moves. A vector access holds at most
/// 128 bits, and its address is a multiple of its size (PTX ISA 9.4, sections 5.4.2 and 6.4.1).
/// Assumes that `alignment` divides the base address.
fn vectorLen(scalars: []const Layout.Scalar, alignment: u64) u32 {
    const first = scalars[0];
    if (first.kind.class == .pred) return 1;
    if (first.kind.repr != .float and first.kind.bits != @as(u16, first.kind.size) * 8) return 1;
    var len: u32 = 4;
    while (len > 1) : (len /= 2) {
        const size = @as(u64, first.kind.size) * len;
        if (len > scalars.len or size > 16 or offsetAlignment(alignment, first.offset) % size != 0) continue;
        const contiguous = for (scalars[1..len], 1..) |scalar, i| {
            if (scalar.kind.class != first.kind.class or scalar.kind.memoryType() != first.kind.memoryType()) break false;
            if (scalar.offset != first.offset + i * first.kind.size) break false;
        } else true;
        if (contiguous) return len;
    }
    return 1;
}

/// A narrow integer has fewer bits than its memory size. Unless `normalize_narrow` is set, the
/// register of a narrow integer keeps the bits from memory.
pub fn loadRegs(builder: *Builder, base: Builder.Base, address_space: std.lang.AddressSpace, alignment: u64, scalars: []const Layout.Scalar, is_volatile: bool, normalize_narrow: bool) Layout.Error![]const Mir.Reg {
    if (hasBitPackedVector(scalars)) {
        return unpackVectors(builder, scalars, try loadRegs(builder, base, address_space, alignment, try memoryForm(builder, scalars), is_volatile, normalize_narrow));
    }
    const regs = try builder.arena.alloc(Mir.Reg, scalars.len);
    const sem = volatileSem(address_space, is_volatile);
    var i: usize = 0;
    while (i < scalars.len) {
        const scalar = scalars[i];
        const len = if (is_volatile) 1 else vectorLen(scalars[i..], alignment);
        const normalize = normalize_narrow and scalar.kind.isNarrowInt() and scalar.kind.bits % 8 != 0;
        const part_alignment = offsetAlignment(alignment, scalar.offset);
        if (part_alignment < scalar.kind.size) {
            regs[i] = try loadPieces(builder, base, address_space, sem, scalar, part_alignment);
        } else if (scalar.kind.class == .pred) {
            // `ld` and `st` have no `.pred` type (PTX ISA 9.4, sections 9.7.10.8 and 9.7.10.11),
            // so a bool occupies one byte.
            const byte = if (base == .name)
                (try builder.paramLoad(base.name, scalar.offset, .u8, &.{.b16}))[0]
            else
                try builder.emit(.b16, .ld, .u8, .{ .mem = .{ .offset = true } }, .{ base, scalar.offset, memOperand(builder, address_space, sem, 1, 1, true) });
            regs[i] = try builder.emit(.pred, .setp, .u16, .{ .setp = .{ .cmp = .ne } }, .{ byte, 0 });
        } else if (base == .name) {
            assert(!is_volatile);
            @memcpy(regs[i..][0..len], try builder.paramLoad(base.name, scalar.offset, memType(scalar.kind), try classesOf(builder, scalars[i..][0..len])));
        } else if (len > 1) {
            const modifiers: Mir.Inst.Modifiers = .{ .mem = .{ .vec = mirVec(len), .offset = true } };
            const mem = memOperand(builder, address_space, .none, @as(u64, scalar.kind.size) * len, part_alignment, true);
            @memcpy(regs[i..][0..len], try builder.emitList(try classesOf(builder, scalars[i..][0..len]), .ld, scalar.kind.memoryType(), modifiers, .{ base, scalar.offset, mem }));
        } else {
            const mem = memOperand(builder, address_space, sem, scalar.kind.size, part_alignment, true);
            regs[i] = try builder.emit(scalar.kind.class, .ld, scalar.kind.memoryType(), .{ .mem = .{ .offset = true } }, .{ base, scalar.offset, mem });
        }
        if (len == 1 and normalize) regs[i] = try int.normalized(builder, regs[i], scalar.kind);
        i += len;
    }
    return regs;
}

pub fn loadMem(builder: *Builder, base: Builder.Base, address_space: std.lang.AddressSpace, alignment: u64, scalars: []const Layout.Scalar, is_volatile: bool) Layout.Error![]const Builder.Part {
    return builder.regParts(try loadRegs(builder, base, address_space, alignment, scalars, is_volatile, true));
}

pub fn storeMem(builder: *Builder, base: Builder.Base, address_space: std.lang.AddressSpace, alignment: u64, scalars: []const Layout.Scalar, src: []const Builder.Part, is_volatile: bool) Layout.Error!void {
    if (hasBitPackedVector(scalars)) {
        return storeMem(builder, base, address_space, alignment, try memoryForm(builder, scalars), try packVectors(builder, scalars, src), is_volatile);
    }
    const sem = volatileSem(address_space, is_volatile);
    var i: usize = 0;
    while (i < scalars.len) {
        const scalar = scalars[i];
        if (src[i] == .undef) {
            i += 1;
            continue;
        }
        const defined = for (src[i..], 0..) |part, count| {
            if (part == .undef) break count;
        } else src.len - i;
        // Stores through a register stay scalar for `transforms/store_vectorize.zig` to merge.
        const len = if (is_volatile or base != .name) 1 else vectorLen(scalars[i..][0..defined], alignment);
        const regs = try builder.toRegs(src[i..][0..len], scalars[i..][0..len]);
        const part_alignment = offsetAlignment(alignment, scalar.offset);
        if (part_alignment < scalar.kind.size) {
            try storePieces(builder, base, address_space, sem, scalar, part_alignment, regs[0]);
        } else if (scalar.kind.class == .pred) {
            const byte = try builder.emit(.b16, .selp, .u16, .{ .none = .{} }, .{ 1, 0, regs[0] });
            if (base == .name) {
                try builder.paramStore(base.name, scalar.offset, .u8, &.{byte});
            } else {
                try builder.emitEffect(.st, .u8, .{ .mem = .{ .offset = true } }, .{ base, scalar.offset, byte, memOperand(builder, address_space, sem, 1, 1, false) });
            }
        } else if (base == .name) {
            assert(!is_volatile);
            try builder.paramStore(base.name, scalar.offset, memType(scalar.kind), regs);
        } else {
            const mem = memOperand(builder, address_space, sem, scalar.kind.size, part_alignment, false);
            try builder.emitEffect(.st, scalar.kind.memoryType(), .{ .mem = .{ .offset = true } }, .{ base, scalar.offset, regs[0], mem });
        }
        i += len;
    }
}

/// `ld` and `st` need an address aligned to the access size (PTX ISA 9.4, section 6.4.1).
fn loadPieces(builder: *Builder, base: Builder.Base, address_space: std.lang.AddressSpace, sem: Mir.Sem, scalar: Layout.Scalar, piece: u64) Layout.Error!Mir.Reg {
    const whole = Kind.int(.unsigned, @as(u16, scalar.kind.size) * 8);
    const piece_kind = Kind.int(.unsigned, @intCast(piece * 8));
    var bits: ?Mir.Reg = null;
    var pos: u64 = 0;
    while (pos < scalar.kind.size) : (pos += piece) {
        const loaded = if (base == .name)
            (try builder.paramLoad(base.name, scalar.offset + pos, memType(piece_kind), &.{piece_kind.class}))[0]
        else
            try builder.emit(piece_kind.class, .ld, piece_kind.memoryType(), .{ .mem = .{ .offset = true } }, .{ base, scalar.offset + pos, memOperand(builder, address_space, sem, piece, piece, true) });
        const widened = try int.intConvert(builder, loaded, piece_kind, whole);
        const moved = if (pos == 0) widened else try builder.emit(whole.class, .shl, whole.bitType(), .{ .none = .{} }, .{ widened, pos * 8 });
        bits = if (bits) |low| try builder.emit(whole.class, .@"or", whole.bitType(), .{ .none = .{} }, .{ low, moved }) else moved;
    }
    if (scalar.kind.class == whole.class) return bits.?;
    return builder.emit(scalar.kind.class, .mov, whole.bitType(), .{ .none = .{} }, .{bits.?});
}

fn storePieces(builder: *Builder, base: Builder.Base, address_space: std.lang.AddressSpace, sem: Mir.Sem, scalar: Layout.Scalar, piece: u64, src: Mir.Reg) Layout.Error!void {
    const whole = Kind.int(.unsigned, @as(u16, scalar.kind.size) * 8);
    const piece_kind = Kind.int(.unsigned, @intCast(piece * 8));
    const bits = if (src.class == whole.class) src else try builder.emit(whole.class, .mov, whole.bitType(), .{ .none = .{} }, .{src});
    var pos: u64 = 0;
    while (pos < scalar.kind.size) : (pos += piece) {
        const high = if (pos == 0) bits else try builder.emit(whole.class, .shr, whole.arithmeticType(), .{ .none = .{} }, .{ bits, pos * 8 });
        const narrow = try int.intConvert(builder, high, whole, piece_kind);
        if (base == .name) {
            try builder.paramStore(base.name, scalar.offset + pos, memType(piece_kind), &.{narrow});
        } else {
            try builder.emitEffect(.st, piece_kind.memoryType(), .{ .mem = .{ .offset = true } }, .{ base, scalar.offset + pos, narrow, memOperand(builder, address_space, sem, piece, piece, false) });
        }
    }
}

fn hasBitPackedVector(scalars: []const Layout.Scalar) bool {
    for (scalars) |scalar| if (scalar.bit != null) return true;
    return false;
}

fn vectorEnd(scalars: []const Layout.Scalar, start: usize) usize {
    var end = start + 1;
    while (end < scalars.len and scalars[end].bit != null and scalars[end].bit.? != 0) end += 1;
    return end;
}

fn vectorChunks(builder: *Builder, elems: []const Layout.Scalar) ![]const Layout.Scalar {
    const last = elems[elems.len - 1];
    const bits = last.bit.? + last.kind.bits;
    const offset = elems[0].offset;
    if (bits <= 64) return builder.arena.dupe(Layout.Scalar, &.{.{ .offset = offset, .kind = Kind.int(.unsigned, @intCast(bits)) }});
    const limbs = try builder.arena.alloc(Layout.Scalar, std.math.divCeil(u32, bits, 64) catch unreachable);
    for (limbs, 0..) |*limb, i| limb.* = .{ .offset = offset + @as(u32, @intCast(i * 8)), .kind = Kind.int(.unsigned, 64) };
    return limbs;
}

/// Replaces each bit-packed vector in `scalars` with the integers that hold it in memory.
pub fn memoryForm(builder: *Builder, scalars: []const Layout.Scalar) ![]const Layout.Scalar {
    if (!hasBitPackedVector(scalars)) return scalars;
    var list: std.ArrayList(Layout.Scalar) = .empty;
    var i: usize = 0;
    while (i < scalars.len) {
        if (scalars[i].bit == null) {
            try list.append(builder.arena, scalars[i]);
            i += 1;
            continue;
        }
        const end = vectorEnd(scalars, i);
        try list.appendSlice(builder.arena, try vectorChunks(builder, scalars[i..end]));
        i = end;
    }
    return list.items;
}

pub fn packVectors(builder: *Builder, scalars: []const Layout.Scalar, parts: []const Builder.Part) Layout.Error![]const Builder.Part {
    if (!hasBitPackedVector(scalars)) return parts;
    var list: std.ArrayList(Builder.Part) = .empty;
    var i: usize = 0;
    while (i < scalars.len) {
        if (scalars[i].bit == null) {
            try list.append(builder.arena, parts[i]);
            i += 1;
            continue;
        }
        const end = vectorEnd(scalars, i);
        try packVector(builder, scalars[i..end], parts[i..end], &list);
        i = end;
    }
    return list.items;
}

fn packVector(builder: *Builder, elems: []const Layout.Scalar, parts: []const Builder.Part, out: *std.ArrayList(Builder.Part)) Layout.Error!void {
    const chunks = try vectorChunks(builder, elems);
    const all_undef = for (parts) |part| {
        if (part != .undef) break false;
    } else true;
    if (all_undef) {
        for (chunks) |_| try out.append(builder.arena, .undef);
        return;
    }
    const limbs = try builder.arena.alloc(Builder.Part, chunks.len);
    @memset(limbs, .{ .imm = 0 });
    const constant = for (parts) |part| {
        if (part != .imm and part != .undef) break false;
    } else true;
    if (constant) {
        for (elems, parts) |scalar, part| {
            if (part != .imm) continue;
            const value = part.imm & (@as(u64, std.math.maxInt(u64)) >> @intCast(64 - scalar.kind.bits));
            const bit = scalar.bit.?;
            limbs[bit / 64].imm |= value << @intCast(bit % 64);
            if (bit % 64 + scalar.kind.bits > 64) limbs[bit / 64 + 1].imm |= value >> @intCast(64 - bit % 64);
        }
        return out.appendSlice(builder.arena, limbs);
    }
    for (elems, parts) |scalar, part| try int.packScalar(builder, scalar.kind, part, limbs, scalar.bit.?);
    for (chunks, limbs) |chunk, limb| {
        const reg = try builder.toReg(limb, Kind.int(.unsigned, 64));
        try out.append(builder.arena, .{ .reg = if (chunk.kind.class == .b64) reg else try int.intConvert(builder, reg, Kind.int(.unsigned, 64), chunk.kind) });
    }
}

fn unpackVectors(builder: *Builder, scalars: []const Layout.Scalar, mem: []const Mir.Reg) Layout.Error![]const Mir.Reg {
    const out = try builder.arena.alloc(Mir.Reg, scalars.len);
    var i: usize = 0;
    var m: usize = 0;
    while (i < scalars.len) {
        if (scalars[i].bit == null) {
            out[i] = mem[m];
            i += 1;
            m += 1;
            continue;
        }
        const end = vectorEnd(scalars, i);
        const chunks = try vectorChunks(builder, scalars[i..end]);
        const limbs = try builder.arena.alloc(Builder.Part, chunks.len);
        for (limbs, chunks, mem[m..][0..chunks.len]) |*limb, chunk, reg| {
            limb.* = .{ .reg = if (chunk.kind.class == .b64) reg else try int.intConvert(builder, reg, chunk.kind, Kind.int(.unsigned, 64)) };
        }
        for (out[i..end], scalars[i..end]) |*o, scalar| o.* = try int.unpackScalar(builder, scalar.kind, limbs, scalar.bit.?);
        i = end;
        m += chunks.len;
    }
    return out;
}

/// Pointers in two different specific spaces name different memory. A `.shared` or `.local`
/// pointer therefore needs a generic address only when `other` is generic. `cvta` needs a source
/// and a destination of one size (PTX ISA 9.4, section 9.7.10.23), so a 32-bit pointer is
/// zero-extended first.
pub fn genericAddress(builder: *Builder, ptr: Mir.Reg, address_space: std.lang.AddressSpace, other: std.lang.AddressSpace) Layout.Error!Mir.Reg {
    if (address_space == other or address_space == .generic or other != .generic) return ptr;
    return switch (address_space) {
        .shared => builder.emit(.b64, .cvta, .u64, .{ .cvta = .{ .space = .shared, .uncommitted = true } }, .{try int.widen(builder, ptr, .pointer32)}),
        .local => builder.emit(.b64, .cvta, .u64, .{ .cvta = .{ .space = .local, .uncommitted = true } }, .{try int.widen(builder, ptr, .pointer32)}),
        else => ptr,
    };
}

/// Returns `ptr + index * scale` in the register class of `ptr`. `index` is a `usize`.
pub fn scaledAddress(builder: *Builder, ptr: Mir.Reg, index: Mir.Reg, scale: u64) Layout.Error!Mir.Reg {
    if (ptr.class == .b64) return builder.emit(.b64, .mad, .s64, .{ .arith = .{ .mode = .lo, .address = true } }, .{ index, scale, ptr });
    const low = try int.intConvert(builder, index, Kind.int(.unsigned, 64), .pointer32);
    return builder.emit(.b32, .mad, .s32, .{ .arith = .{ .mode = .lo, .address = true } }, .{ low, @as(u32, @intCast(scale)), ptr });
}

/// Assumes that `dest_alignment` divides `dest` and that `src_alignment` divides `src`.
pub fn copyLoop(builder: *Builder, dest: Mir.Reg, count: Mir.Reg, src: ?Mir.Reg, dest_address_space: std.lang.AddressSpace, src_address_space: std.lang.AddressSpace, dest_alignment: u64, src_alignment: u64, scalars: []const Layout.Scalar, size: u64, value: []const Builder.Part, backward: bool, dest_volatile: bool, src_volatile: bool) Layout.Error!void {
    const head = try builder.newBlock(&.{.b64});
    const end = builder.labelBlock();
    try builder.branch(head, &.{if (backward) count else try builder.movImm(.b64, 0)});
    const i = (try builder.startBlock(head))[0];
    const done = if (backward)
        try builder.emit(.pred, .setp, .u64, .{ .setp = .{ .cmp = .eq } }, .{ i, 0 })
    else
        try builder.emit(.pred, .setp, .u64, .{ .setp = .{ .cmp = .ge } }, .{ i, count });
    try builder.branchIf(done, false, end, &.{});
    const index = if (backward) try builder.emit(.b64, .sub, .u64, .{ .arith = .{} }, .{ i, 1 }) else i;
    const dest_address = try scaledAddress(builder, dest, index, size);
    const elems = if (src) |src_base| elems: {
        const src_address = try scaledAddress(builder, src_base, index, size);
        break :elems try loadMem(builder, .{ .reg = src_address }, src_address_space, offsetAlignment(src_alignment, size), scalars, src_volatile);
    } else value;
    try storeMem(builder, .{ .reg = dest_address }, dest_address_space, offsetAlignment(dest_alignment, size), scalars, elems, dest_volatile);
    const next = if (backward) index else try builder.emit(.b64, .add, .u64, .{ .arith = .{} }, .{ index, 1 });
    try builder.branch(head, &.{next});
    _ = try builder.startBlock(end);
}

fn classesOf(builder: *Builder, scalars: []const Layout.Scalar) ![]const Mir.Class {
    const classes = try builder.arena.alloc(Mir.Class, scalars.len);
    for (classes, scalars) |*class, scalar| class.* = scalar.kind.class;
    return classes;
}
