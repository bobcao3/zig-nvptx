//! This file splits Zig types into the scalars that the backend keeps in registers. It also
//! reports the code generation errors of a declaration.

const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const InternPool = @import("../../../InternPool.zig");
const Type = @import("../../../Type.zig");
const Value = @import("../../../Value.zig");
const Zcu = @import("../../../Zcu.zig");
const codegen = @import("../../../codegen.zig");
const Mir = @import("../Mir.zig");
const Kind = @import("Kind.zig");
const Layout = @This();

/// A fill of more scalars than this is a loop. Each store of a fill writes the same value, so the
/// stores keep no more values in registers than a loop does, and the limit bounds only the size
/// of the code.
pub const max_unrolled_scalars = 1024;

/// A load of a larger array or struct keeps the bytes in memory, so that each use loads only the
/// scalars that it reads. nvcc 13.0 also keeps each kernel parameter of more than 128 bytes in
/// `.param`.
pub const max_register_bytes = 128;

pt: Zcu.PerThread,
zcu: *Zcu,
gpa: Allocator,
arena: Allocator,
target: *const std.Target,
owner_nav: InternPool.Nav.Index,

/// `CodeGen` reports the two errors as errors of the declaration.
pub const Error = codegen.Error || error{ CannotSplit, TooManyRegisters };

pub fn init(pt: Zcu.PerThread, owner_nav: InternPool.Nav.Index, arena: Allocator) Layout {
    const zcu = pt.zcu;
    return .{
        .pt = pt,
        .zcu = zcu,
        .gpa = zcu.gpa,
        .arena = arena,
        .target = &zcu.navFileScope(owner_nav).mod.?.resolved_target.result,
        .owner_nav = owner_nav,
    };
}

pub const Scalar = struct {
    offset: u32,
    kind: Kind,
    /// An element of a bit-packed vector starts at this bit and has the offset of the vector.
    bit: ?u32 = null,
};

pub const Space = enum {
    global,
    shared,
    @"const",
    local,
    param,
    /// A function symbol is in no PTX state space.
    func,

    pub fn fromAddressSpace(address_space: std.lang.AddressSpace) ?Space {
        return switch (address_space) {
            .generic, .global => .global,
            .shared => .shared,
            .constant => .@"const",
            .local => .local,
            .param => .param,
            else => null,
        };
    }
};

pub fn fail(layout: *const Layout, comptime format: []const u8, args: anytype) codegen.Error {
    @branchHint(.cold);
    return layout.zcu.codegenFail(layout.owner_nav, format, args);
}

pub fn todo(layout: *const Layout, comptime format: []const u8, args: anytype) codegen.Error {
    return layout.fail("TODO (PTX): " ++ format, args);
}

pub fn smVersion(layout: *const Layout) u16 {
    const name = layout.target.cpu.model.name;
    if (!std.mem.startsWith(u8, name, "sm_")) return 0;
    var version: u16 = 0;
    for (name["sm_".len..]) |digit| {
        if (!std.ascii.isDigit(digit)) break;
        version = version * 10 + (digit - '0');
    }
    return version;
}

pub fn ptrKind(layout: *const Layout, address_space: std.lang.AddressSpace) Kind {
    return if (layout.target.addressSpacePtrBitWidth(address_space) == 32) .pointer32 else .pointer;
}

/// Asserts that `ty` is a pointer, a slice or an optional of either.
pub fn ptrTypeKind(layout: *const Layout, ty: Type) Kind {
    return layout.ptrKind(ty.ptrAddressSpace(layout.zcu));
}

pub fn kindOf(layout: *const Layout, ty: Type) Error!Kind {
    const zcu = layout.zcu;
    switch (ty.zigTypeTag(zcu)) {
        .bool => return .bool,
        .pointer => return layout.ptrTypeKind(ty),
        .optional => return if (ty.isPtrLikeOptional(zcu)) layout.ptrTypeKind(ty) else .pointer,
        .float => return switch (ty.floatBits(layout.target)) {
            16 => .{ .class = .b16, .size = 2, .repr = .float, .bits = 16 },
            32 => .f32,
            64 => .{ .class = .f64, .size = 8, .repr = .float, .bits = 64 },
            else => error.CannotSplit,
        },
        .error_set => return Kind.int(.unsigned, zcu.errorSetBits()),
        else => {
            const info = ty.intInfo(zcu);
            if (info.bits > 64) return error.CannotSplit;
            return Kind.int(switch (info.signedness) {
                .signed => .signed,
                .unsigned => .unsigned,
            }, info.bits);
        },
    }
}

pub fn scalarsOf(layout: *const Layout, ty: Type) Error![]const Scalar {
    return layout.split(ty) catch |err| switch (err) {
        error.CannotSplit => layout.todo("values of type '{f}'", .{ty.fmt(layout.zcu)}),
        else => |e| e,
    };
}

pub fn split(layout: *const Layout, ty: Type) Error![]const Scalar {
    var list: std.ArrayList(Scalar) = .empty;
    try layout.appendScalars(&list, ty, 0);
    return list.items;
}

pub fn scalarCount(layout: *const Layout, ty: Type) Error!u32 {
    return @intCast((try layout.scalarsOf(ty)).len);
}

fn appendScalars(layout: *const Layout, list: *std.ArrayList(Scalar), ty: Type, offset: u64) Error!void {
    const zcu = layout.zcu;
    if (!ty.hasRuntimeBits(zcu)) return;
    if (list.items.len > max_unrolled_scalars * 64) return error.CannotSplit;
    const off: u32 = @intCast(offset);
    switch (ty.zigTypeTag(zcu)) {
        .bool, .error_set => try list.append(layout.arena, .{ .offset = off, .kind = try layout.kindOf(ty) }),
        .float => switch (ty.floatBits(layout.target)) {
            // On nvptx targets, compiler_rt passes an f80 or an f128 as two integers.
            80, 128 => |bits| try list.appendSlice(layout.arena, &.{
                .{ .offset = off, .kind = Kind.int(.unsigned, 64) },
                .{ .offset = off + 8, .kind = Kind.int(.unsigned, bits - 64) },
            }),
            else => try list.append(layout.arena, .{ .offset = off, .kind = try layout.kindOf(ty) }),
        },
        .int, .@"enum" => {
            const info = ty.intInfo(zcu);
            if (info.bits <= 64) return list.append(layout.arena, .{ .offset = off, .kind = try layout.kindOf(ty) });
            const wide_int = WideInt.init(info, ty.abiSize(zcu));
            for (0..wide_int.limbs) |limb| {
                try list.append(layout.arena, .{ .offset = off + @as(u32, @intCast(limb)) * 8, .kind = wide_int.limbKind(@intCast(limb)) });
            }
        },
        .pointer => {
            try list.append(layout.arena, .{ .offset = off, .kind = layout.ptrTypeKind(ty) });
            if (ty.isSlice(zcu)) try list.append(layout.arena, .{ .offset = off + 8, .kind = Kind.int(.unsigned, 64) });
        },
        .optional => {
            const payload_type = ty.optionalChild(zcu);
            if (ty.optionalReprIsPayload(zcu)) return layout.appendScalars(list, payload_type, offset);
            try layout.appendScalars(list, payload_type, offset);
            try list.append(layout.arena, .{ .offset = off + @as(u32, @intCast(payload_type.abiSize(zcu))), .kind = .bool });
        },
        .error_union => {
            const payload_type = ty.errorUnionPayload(zcu);
            try layout.appendScalars(list, ty.errorUnionSet(zcu), offset + codegen.errUnionErrorOffset(payload_type, zcu));
            try layout.appendScalars(list, payload_type, offset + codegen.errUnionPayloadOffset(payload_type, zcu));
        },
        .@"struct" => switch (ty.containerLayout(zcu)) {
            .@"packed" => try layout.appendPackedScalars(list, ty, offset),
            .auto, .@"extern" => for (0..ty.structFieldCount(zcu)) |i| {
                if (ty.structFieldIsComptime(i, zcu)) continue;
                try layout.appendScalars(list, ty.fieldType(i, zcu), offset + ty.structFieldOffset(i, zcu));
            },
        },
        .array, .vector => {
            const elem_type = ty.childType(zcu);
            const elem_size = elem_type.abiSize(zcu);
            const len = if (ty.zigTypeTag(zcu) == .array) ty.arrayLenIncludingSentinel(zcu) else ty.vectorLen(zcu);
            if (layout.isBitPackedVector(ty)) {
                const elem_bits = elem_type.bitSize(zcu);
                const elem_scalars = try layout.scalarsOf(elem_type);
                // The packed form has no place for a limb without bits of the element.
                if ((elem_scalars.len - 1) * 64 >= elem_bits) return error.CannotSplit;
                for (0..len) |i| for (elem_scalars, 0..) |scalar, limb| {
                    try list.append(layout.arena, .{ .offset = off, .kind = scalar.kind, .bit = @intCast(i * elem_bits + limb * 64) });
                };
                return;
            }
            for (0..len) |i| try layout.appendScalars(list, elem_type, offset + i * elem_size);
        },
        .@"union" => switch (ty.containerLayout(zcu)) {
            .@"packed" => try layout.appendPackedScalars(list, ty, offset),
            .auto, .@"extern" => {
                const size = ty.abiSize(zcu);
                const chunk: u64 = @min(8, ty.abiAlignment(zcu).toByteUnits().?);
                var pos: u64 = 0;
                while (pos < size) : (pos += chunk) {
                    try list.append(layout.arena, .{ .offset = @intCast(offset + pos), .kind = Kind.int(.unsigned, @intCast(chunk * 8)) });
                }
            },
        },
        else => return error.CannotSplit,
    }
}

fn appendPackedScalars(layout: *const Layout, list: *std.ArrayList(Scalar), ty: Type, offset: u64) Error!void {
    const bits = ty.bitSize(layout.zcu);
    if (bits > 64) return layout.appendScalars(list, ty.backingIntType(layout.zcu), offset);
    try list.append(layout.arena, .{ .offset = @intCast(offset), .kind = Kind.int(.unsigned, @intCast(bits)) });
}

/// Returns true if element `i` of the vector `ty` starts at bit `i * @bitSizeOf(elem)` in memory,
/// in place of byte `i * @sizeOf(elem)`. `Type.abiSize` lays out vectors of integers and bools
/// this way.
pub fn isBitPackedVector(layout: *const Layout, ty: Type) bool {
    const zcu = layout.zcu;
    if (ty.zigTypeTag(zcu) != .vector) return false;
    const elem_type = ty.childType(zcu);
    return !elem_type.isRuntimeFloat() and elem_type.bitSize(zcu) != 8 * elem_type.abiSize(zcu);
}

/// Returns the vector when `ptr_type` points to an element of a bit-packed vector.
pub fn bitPackedHost(layout: *const Layout, ptr_type: Type) !?Type {
    const zcu = layout.zcu;
    if (ptr_type.zigTypeTag(zcu) != .pointer) return null;
    const info = ptr_type.ptrInfo(zcu);
    if (info.flags.vector_index == .none) return null;
    const vector_type = try layout.pt.vectorType(.{ .len = info.packed_offset.host_size, .child = info.child });
    return if (layout.isBitPackedVector(vector_type)) vector_type else null;
}

pub fn vectorElemOffset(layout: *const Layout, src_ptr_type: Type, dest_ptr_type: Type) !u64 {
    const zcu = layout.zcu;
    const index = dest_ptr_type.ptrInfo(zcu).flags.vector_index;
    if (index == .none or src_ptr_type.ptrInfo(zcu).flags.vector_index != .none) return 0;
    if (try layout.bitPackedHost(dest_ptr_type) != null) return 0;
    return @backingInt(index) * dest_ptr_type.childType(zcu).abiSize(zcu);
}

pub fn fieldScalars(layout: *const Layout, ty: Type, index: u32) Error!struct { u32, u32 } {
    const zcu = layout.zcu;
    switch (ty.zigTypeTag(zcu)) {
        .array, .vector => {
            const count = try layout.scalarCount(ty.childType(zcu));
            return .{ index * count, count };
        },
        .pointer => return .{ index, 1 },
        .@"struct" => {
            var first: u32 = 0;
            for (0..index) |i| {
                if (ty.structFieldIsComptime(i, zcu)) continue;
                first += try layout.scalarCount(ty.fieldType(i, zcu));
            }
            return .{ first, try layout.scalarCount(ty.fieldType(index, zcu)) };
        },
        else => return layout.todo("field access of '{f}'", .{ty.fmt(zcu)}),
    }
}

pub fn fieldOffset(layout: *const Layout, ty: Type, index: u32) Error!u32 {
    const zcu = layout.zcu;
    return switch (ty.zigTypeTag(zcu)) {
        .array, .vector => @intCast(index * ty.childType(zcu).abiSize(zcu)),
        .pointer => if (index == 0) 0 else 8,
        .@"struct" => switch (ty.containerLayout(zcu)) {
            .@"packed" => 0,
            else => @intCast(ty.structFieldOffset(index, zcu)),
        },
        .@"union" => switch (ty.containerLayout(zcu)) {
            .@"packed" => 0,
            else => layout.unionPayloadOffset(ty),
        },
        else => layout.todo("field offset of '{f}'", .{ty.fmt(zcu)}),
    };
}

pub fn offsetScalars(layout: *const Layout, scalars: []const Scalar, offset: u32) ![]const Scalar {
    const out = try layout.arena.alloc(Scalar, scalars.len);
    for (out, scalars) |*result, scalar| {
        result.* = scalar;
        result.offset += offset;
    }
    return out;
}

pub fn ptrAlignment(layout: *const Layout, ptr_type: Type) u64 {
    return ptr_type.ptrAlignment(layout.zcu).toByteUnits().?;
}

/// Assumes `ty` is not a packed union.
pub fn unionPayloadOffset(layout: *const Layout, ty: Type) u32 {
    const union_layout = ty.unionGetLayout(layout.zcu);
    return if (union_layout.tag_size == 0) 0 else @intCast(union_layout.payloadOffset());
}

/// Returns null when the tag has no runtime bits. Assumes `ty` is not a packed union.
pub fn unionTagScalar(layout: *const Layout, ty: Type) Error!?Scalar {
    const zcu = layout.zcu;
    const tag_type = ty.unionTagTypeRuntime(zcu) orelse return null;
    if (!tag_type.hasRuntimeBits(zcu)) return null;
    return .{ .offset = @intCast(ty.unionGetLayout(zcu).tagOffset()), .kind = try layout.kindOf(tag_type) };
}

/// An enum with no stored field values numbers its fields from zero.
pub fn enumFieldBits(layout: *const Layout, ty: Type, index: usize) u64 {
    const ip = &layout.zcu.intern_pool;
    const values = ip.loadEnumType(ty.toIntern()).field_values;
    if (values.len == 0) return index;
    var space: Value.BigIntSpace = undefined;
    const big = Value.fromInterned(values.get(ip)[index]).toBigInt(&space, layout.zcu);
    var bytes: [8]u8 = @splat(0);
    big.writeTwosComplement(&bytes, .little);
    return std.mem.readInt(u64, &bytes, .little);
}

pub fn enumIsDense(layout: *const Layout, ty: Type) bool {
    for (0..ty.enumFieldCount(layout.zcu)) |i| if (layout.enumFieldBits(ty, i) != i) return false;
    return true;
}

/// Returns true for an array and for a struct that is not packed. A load of such a value can stay
/// in memory, and each use of a field or an element loads only that field or element.
pub fn isAggregate(layout: *const Layout, ty: Type) bool {
    const zcu = layout.zcu;
    return switch (ty.zigTypeTag(zcu)) {
        .array => true,
        .@"struct" => ty.containerLayout(zcu) != .@"packed",
        else => false,
    };
}

/// Returns true if an element of the array or vector `ty` has padding bits in memory, directly
/// or in a nested array. `@bitCast` packs such elements without the padding.
pub fn isBitPacked(layout: *const Layout, ty: Type) bool {
    const zcu = layout.zcu;
    if (!ty.isArrayOrVector(zcu)) return false;
    const elem_type = ty.childType(zcu);
    return elem_type.bitSize(zcu) != elem_type.abiSize(zcu) * 8 or layout.isBitPacked(elem_type);
}

pub const WideInt = struct {
    bits: u16,
    signedness: std.lang.Signedness,
    limbs: u32,

    pub fn init(info: InternPool.Key.IntType, abi_size: u64) WideInt {
        return .{ .bits = info.bits, .signedness = info.signedness, .limbs = @intCast(abi_size / 8) };
    }

    pub fn topLimb(wide_int: WideInt) u32 {
        return (wide_int.bits - 1) / 64;
    }

    pub fn limbKind(wide_int: WideInt, limb: u32) Kind {
        const top = wide_int.topLimb();
        if (limb < top) return Kind.int(.unsigned, 64);
        const repr: Kind.Repr = switch (wide_int.signedness) {
            .signed => .signed,
            .unsigned => .unsigned,
        };
        const bits: u16 = if (limb == top) @intCast(wide_int.bits - 64 * top) else 64;
        return .{ .class = .b64, .size = 8, .repr = repr, .bits = bits };
    }

    pub fn limitLimb(wide_int: WideInt, limb: u32, limit: enum { min, max }) u64 {
        const max = lowOnes(wide_int.bits - @intFromBool(wide_int.signedness == .signed), limb);
        return switch (limit) {
            .max => max,
            .min => if (wide_int.signedness == .signed) ~max else 0,
        };
    }
};

pub fn lowOnes(count: u32, limb: u32) u64 {
    const low = limb * 64;
    if (count >= low + 64) return std.math.maxInt(u64);
    if (count <= low) return 0;
    return (@as(u64, 1) << @intCast(count - low)) - 1;
}

pub fn wideIntOf(layout: *const Layout, ty: Type) ?WideInt {
    const zcu = layout.zcu;
    const int_type = switch (ty.zigTypeTag(zcu)) {
        .int, .@"enum" => ty,
        .@"struct", .@"union" => if (ty.containerLayout(zcu) == .@"packed") ty.backingIntType(zcu) else return null,
        else => return null,
    };
    if (!int_type.hasRuntimeBits(zcu)) return null;
    const info = int_type.intInfo(zcu);
    if (info.bits <= 64) return null;
    return .init(info, int_type.abiSize(zcu));
}
