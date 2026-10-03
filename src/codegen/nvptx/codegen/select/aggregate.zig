//! This file lowers the AIR instructions on arrays, vectors, structs, unions, optionals and error
//! unions, the names of enum tags and errors, and bit casts.

const std = @import("std");

const assert = std.debug.assert;
const Air = @import("../../../../Air.zig");
const Type = @import("../../../../Type.zig");
const Value = @import("../../../../Value.zig");
const Zcu = @import("../../../../Zcu.zig");
const codegen = @import("../../../../codegen.zig");
const Mir = @import("../../Mir.zig");
const Kind = @import("../Kind.zig");
const Layout = @import("../Layout.zig");
const Builder = @import("../Builder.zig");
const Function = @import("../Function.zig");
const int = @import("../int.zig");
const wide = @import("../wide.zig");
const memory = @import("../memory.zig");
const constant = @import("../constant.zig");

const Val = Function.Val;

/// An array of more scalars than this stays in `.local` memory when a run-time index reads it.
const max_selected_scalars = 32;

pub fn airArrayElemVal(func: *Function, inst: Air.Inst.Index) Layout.Error!Val {
    const zcu = func.zcu;
    const bin_op = func.dataOf(inst).bin_op;
    const array_type = func.typeOf(bin_op.lhs);
    if (bin_op.rhs.toInterned()) |ip_index| {
        return fieldVal(func, bin_op.lhs, @intCast(Value.fromInterned(ip_index).toUnsignedInt(zcu)));
    }
    const elem_type = array_type.childType(zcu);
    const elem_size = elem_type.abiSize(zcu);
    const place = func.placeOf(bin_op.lhs) orelse try constantTable(func, bin_op.lhs);
    if (place == null and array_type.arrayLenIncludingSentinel(zcu) > 0 and try func.layout.scalarCount(array_type) <= max_selected_scalars) {
        return pickElem(func, try func.partsOf(bin_op.lhs), elem_type, array_type.arrayLenIncludingSentinel(zcu), try func.scalarReg(bin_op.rhs));
    }
    const base_alignment = if (place) |p| func.placeAlignment(p) else array_type.abiAlignment(zcu).toByteUnits().?;
    const base = if (place) |p| try func.placeAddress(p) else base: {
        const local = try func.builder.newLocal(array_type.abiSize(zcu), base_alignment, .local, func.layout.ptrKind(.local));
        try memory.storeMem(&func.builder, .{ .reg = local }, .local, base_alignment, try func.layout.scalarsOf(array_type), try func.partsOf(bin_op.lhs), false);
        break :base local;
    };
    const index = try int.widen(&func.builder, try func.scalarReg(bin_op.rhs), try func.layout.kindOf(func.typeOf(bin_op.rhs)));
    const elem: Function.Place = .{ .base = .{ .address = .{
        .reg = try memory.scaledAddress(&func.builder, base, index, elem_size),
        .address_space = if (place) |p| p.addressSpace() else .local,
        .alignment = memory.offsetAlignment(base_alignment, elem_size),
    } }, .offset = 0 };
    if (place != null) return func.placeVal(elem, elem_type);
    return .{ .parts = try func.loadPlace(elem, elem_type) };
}

/// A table of more than 4,096 bytes goes to `.global` memory. `.const` memory holds 64 KB (PTX ISA
/// 9.4, section 5.1.3), so 16 tables of the largest size fill it.
const max_const_table_bytes = 4096;

/// Returns a place at a constant in memory with the comptime-known value of `array`, or null for
/// other arrays. The linker writes the constant, and a failure there fails the link, so the
/// elements must be plain data.
fn constantTable(func: *Function, array: Air.Inst.Ref) Layout.Error!?Function.Place {
    const zcu = func.zcu;
    const ip_index = array.toInterned() orelse return null;
    const array_type = func.typeOf(array);
    if (!isPlainData(zcu, array_type.childType(zcu))) return null;
    const symbol: Builder.SymbolPointer = if (array_type.abiSize(zcu) <= max_const_table_bytes)
        .{ .target = .{ .const_uav = ip_index }, .offset = 0, .space = .@"const", .address_space = .constant }
    else
        .{ .target = .{ .uav = ip_index }, .offset = 0, .space = .global, .address_space = .global };
    return .{ .base = .{ .address = .{
        .reg = try func.builder.toReg(.{ .symbol = symbol }, func.layout.ptrKind(symbol.address_space)),
        .address_space = symbol.address_space,
        .alignment = array_type.abiAlignment(zcu).toByteUnits().?,
    } }, .offset = 0 };
}

/// Returns true when the language defines the memory layout of `ty` and `ty` holds no pointer.
fn isPlainData(zcu: *const Zcu, ty: Type) bool {
    return switch (ty.zigTypeTag(zcu)) {
        .int, .float, .bool => true,
        .@"enum" => ty.hasWellDefinedLayout(zcu),
        .array => isPlainData(zcu, ty.childType(zcu)),
        .@"struct" => ty.containerLayout(zcu) != .auto and for (0..ty.structFieldCount(zcu)) |i| {
            if (!isPlainData(zcu, ty.fieldType(i, zcu))) break false;
        } else true,
        else => false,
    };
}

pub fn fieldVal(func: *Function, aggregate: Air.Inst.Ref, index: u32) Layout.Error!Val {
    const zcu = func.zcu;
    const ty = func.typeOf(aggregate);
    if (ty.zigTypeTag(zcu) == .@"union") return unionFieldVal(func, aggregate, ty, index);
    const place = if (func.layout.isBitPackedVector(ty)) null else func.placeOf(aggregate);
    if (place) |p| {
        const field_type = if (ty.isArrayOrVector(zcu)) ty.childType(zcu) else ty.fieldType(index, zcu);
        return func.placeVal(.{ .base = p.base, .offset = p.offset + try func.layout.fieldOffset(ty, index) }, field_type);
    }
    const first, const count = try func.layout.fieldScalars(ty, index);
    return .{ .parts = (try func.partsOf(aggregate))[first..][0..count] };
}

/// `Legalize` turns an operation on vectors into a loop over the element index.
pub fn airVecElemVal(func: *Function, inst: Air.Inst.Index) Layout.Error!Val {
    const bin_op = func.dataOf(inst).bin_op;
    if (bin_op.rhs.toInterned()) |ip_index| {
        return fieldVal(func, bin_op.lhs, @intCast(Value.fromInterned(ip_index).toUnsignedInt(func.zcu)));
    }
    const vector_type = func.typeOf(bin_op.lhs);
    return pickElem(func, try func.partsOf(bin_op.lhs), vector_type.childType(func.zcu), vector_type.vectorLen(func.zcu), try func.scalarReg(bin_op.rhs));
}

/// A register has no address, so `selp` picks the element. A chain of `len - 1` `setp` and `selp`
/// pairs for each scalar of the element costs less than the `len` stores and one load that an
/// array in `.local` memory takes, while the array has at most `max_selected_scalars` scalars.
fn pickElem(func: *Function, elems: []const Builder.Part, elem_type: Type, len: u64, index: Mir.Reg) Layout.Error!Val {
    const scalars = try func.layout.scalarsOf(elem_type);
    const out = try func.arena.dupe(Mir.Reg, try func.builder.toRegs(elems[0..scalars.len], scalars));
    for (1..len) |elem| {
        const is_elem = try int.indexIs(&func.builder, index, elem);
        for (out, scalars, elems[elem * scalars.len ..][0..scalars.len]) |*result, scalar, src| {
            result.* = try int.select(&func.builder, is_elem, try func.builder.toReg(src, scalar.kind), result.*, scalar.kind);
        }
    }
    return .{ .parts = try func.builder.regParts(out) };
}

pub fn airVecStoreElem(func: *Function, inst: Air.Inst.Index) Layout.Error!void {
    const pl_op = func.dataOf(inst).pl_op;
    const bin = func.air.extraData(Air.Bin, pl_op.payload).data;
    const vector_type = func.typeOf(pl_op.operand).childType(func.zcu);
    const scalars = try func.layout.scalarsOf(vector_type.childType(func.zcu));
    const elems = try func.arena.dupe(Builder.Part, try func.loadVector(pl_op.operand, vector_type));
    const value = try func.partsOf(bin.rhs);
    const index = try func.scalarReg(bin.lhs);
    for (0..vector_type.vectorLen(func.zcu)) |elem| {
        const is_elem = try int.indexIs(&func.builder, index, elem);
        for (elems[elem * scalars.len ..][0..scalars.len], scalars, value) |*dst, scalar, src| {
            dst.* = .{ .reg = try int.select(&func.builder, is_elem, try func.builder.toReg(src, scalar.kind), try func.builder.toReg(dst.*, scalar.kind), scalar.kind) };
        }
    }
    try func.store(pl_op.operand, elems, vector_type);
}

pub fn airConcat(func: *Function, refs: []const Air.Inst.Ref) Layout.Error!Val {
    var list: std.ArrayList(Builder.Part) = .empty;
    for (refs) |ref| try list.appendSlice(func.arena, try func.partsOf(ref));
    return .{ .parts = list.items };
}

pub fn airAggregateInit(func: *Function, inst: Air.Inst.Index) Layout.Error!Val {
    const zcu = func.zcu;
    const ty = func.typeOfIndex(inst);
    const len: usize = switch (ty.zigTypeTag(zcu)) {
        .array, .vector => ty.arrayLen(zcu),
        .@"struct" => ty.structFieldCount(zcu),
        else => return func.layout.todo("aggregate_init of '{f}'", .{ty.fmt(zcu)}),
    };
    const elems: []const Air.Inst.Ref = @ptrCast(func.air.extra.items[func.dataOf(inst).ty_pl.payload..][0..len]);
    var list: std.ArrayList(Builder.Part) = .empty;
    for (elems, 0..) |elem, i| {
        if (ty.zigTypeTag(zcu) == .@"struct" and ty.structFieldIsComptime(i, zcu)) continue;
        try list.appendSlice(func.arena, try func.partsOf(elem));
    }
    if (ty.zigTypeTag(zcu) == .array) if (ty.sentinel(zcu)) |sentinel| try constant.constParts(&func.builder, &func.layout, &list, ty.childType(zcu), sentinel);
    return .{ .parts = list.items };
}

pub fn airSplat(func: *Function, inst: Air.Inst.Index) Layout.Error!Val {
    const ty_op = func.dataOf(inst).ty_op;
    const elem = try func.partsOf(ty_op.operand);
    const len = ty_op.ty.vectorLen(func.zcu);
    const out = try func.arena.alloc(Builder.Part, elem.len * len);
    for (0..len) |i| @memcpy(out[i * elem.len ..][0..elem.len], elem);
    return .{ .parts = out };
}

pub fn airShuffleOne(func: *Function, inst: Air.Inst.Index) Layout.Error!Val {
    const unwrapped = func.air.unwrapShuffleOne(func.zcu, inst);
    const src = try func.partsOf(unwrapped.operand);
    var list: std.ArrayList(Builder.Part) = .empty;
    for (unwrapped.mask) |element| switch (element.unwrap()) {
        .elem => |i| try list.append(func.arena, src[i]),
        .value => |value| try constant.constParts(&func.builder, &func.layout, &list, unwrapped.result_ty.childType(func.zcu), .fromInterned(value)),
    };
    return .{ .parts = list.items };
}

pub fn airShuffleTwo(func: *Function, inst: Air.Inst.Index) Layout.Error!Val {
    const unwrapped = func.air.unwrapShuffleTwo(func.zcu, inst);
    const operand_a = try func.partsOf(unwrapped.operand_a);
    const operand_b = try func.partsOf(unwrapped.operand_b);
    var list: std.ArrayList(Builder.Part) = .empty;
    for (unwrapped.mask) |element| try list.append(func.arena, switch (element.unwrap()) {
        .a_elem => |i| operand_a[i],
        .b_elem => |i| operand_b[i],
        .undef => .undef,
    });
    return .{ .parts = list.items };
}

/// The fields of a union overlap, so a union goes through a local variable to change its scalars.
fn spill(func: *Function, ty: Type, src: []const Builder.Part) Layout.Error!Mir.Reg {
    const alignment = ty.abiAlignment(func.zcu).toByteUnits().?;
    const local = try func.builder.newLocal(ty.abiSize(func.zcu), alignment, .local, func.layout.ptrKind(.local));
    try memory.storeMem(&func.builder, .{ .reg = local }, .local, alignment, try func.layout.scalarsOf(ty), src, false);
    return local;
}

fn buildUnion(func: *Function, ty: Type, tag: ?Builder.Part, payload_scalars: []const Layout.Scalar, payload: []const Builder.Part) Layout.Error![]const Builder.Part {
    const scalars = try func.layout.scalarsOf(ty);
    assert(scalars.len != 0);
    const alignment = ty.abiAlignment(func.zcu).toByteUnits().?;
    const local = try func.builder.newLocal(ty.abiSize(func.zcu), alignment, .local, func.layout.ptrKind(.local));
    if (tag) |part| if (try func.layout.unionTagScalar(ty)) |scalar| try memory.storeMem(&func.builder, .{ .reg = local }, .local, alignment, &.{scalar}, &.{part}, false);
    try memory.storeMem(&func.builder, .{ .reg = local }, .local, alignment, try func.layout.offsetScalars(payload_scalars, func.layout.unionPayloadOffset(ty)), payload, false);
    return memory.loadMem(&func.builder, .{ .reg = local }, .local, alignment, scalars, false);
}

fn unionFieldVal(func: *Function, aggregate: Air.Inst.Ref, ty: Type, index: u32) Layout.Error!Val {
    const scalars = try func.layout.scalarsOf(ty.unionFieldTypeByIndex(index, func.zcu));
    assert(scalars.len != 0);
    const local = try spill(func, ty, try func.partsOf(aggregate));
    const alignment = ty.abiAlignment(func.zcu).toByteUnits().?;
    return .{ .parts = try memory.loadMem(&func.builder, .{ .reg = local }, .local, alignment, try func.layout.offsetScalars(scalars, func.layout.unionPayloadOffset(ty)), false) };
}

pub fn airUnionInit(func: *Function, inst: Air.Inst.Index) Layout.Error!Val {
    const zcu = func.zcu;
    const ty = func.typeOfIndex(inst);
    const extra = func.air.extraData(Air.UnionInit, func.dataOf(inst).ty_pl.payload).data;
    if (ty.containerLayout(zcu) == .@"packed") return func.layout.todo("union_init of the packed union '{f}'", .{ty.fmt(zcu)});
    const field_type = ty.unionFieldTypeByIndex(extra.field_index, zcu);
    const tag: ?Builder.Part = if (ty.unionTagTypeRuntime(zcu)) |tag_type|
        .{ .imm = func.layout.enumFieldBits(tag_type, extra.field_index) }
    else
        null;
    const payload = if (field_type.hasRuntimeBits(zcu)) try func.partsOf(extra.init) else &.{};
    return .{ .parts = try buildUnion(func, ty, tag, try func.layout.scalarsOf(field_type), payload) };
}

pub fn airUnionFromEnum(func: *Function, inst: Air.Inst.Index) Layout.Error!Val {
    const ty_op = func.dataOf(inst).ty_op;
    const tag = try func.partsOf(ty_op.operand);
    return .{ .parts = try buildUnion(func, ty_op.ty, if (tag.len == 0) null else tag[0], &.{}, &.{}) };
}

pub fn airGetUnionTag(func: *Function, inst: Air.Inst.Index) Layout.Error!Val {
    const union_type = func.typeOf(func.dataOf(inst).ty_op.operand);
    const scalar = (try func.layout.unionTagScalar(union_type)) orelse return .{ .parts = &.{} };
    const local = try spill(func, union_type, try func.partsOf(func.dataOf(inst).ty_op.operand));
    const alignment = union_type.abiAlignment(func.zcu).toByteUnits().?;
    return .{ .parts = try memory.loadMem(&func.builder, .{ .reg = local }, .local, alignment, &.{scalar}, false) };
}

pub fn airSetUnionTag(func: *Function, inst: Air.Inst.Index) Layout.Error!void {
    const bin_op = func.dataOf(inst).bin_op;
    const ptr_type = func.typeOf(bin_op.lhs);
    const scalar = (try func.layout.unionTagScalar(ptr_type.childType(func.zcu))) orelse return;
    try func.storeVal(try func.resolve(bin_op.lhs), ptr_type, scalar.offset, try func.partsOf(bin_op.rhs), func.typeOf(bin_op.rhs));
}

pub fn airIsNull(func: *Function, inst: Air.Inst.Index, is_null: bool, is_ptr: bool) Layout.Error!Val {
    const zcu = func.zcu;
    const un_op = func.dataOf(inst).un_op;
    const optional_type = if (is_ptr) func.typeOf(un_op).childType(zcu) else func.typeOf(un_op);
    const flag = try optionalFlag(func, un_op, optional_type, is_ptr);
    if (optional_type.optionalReprIsPayload(zcu)) {
        const cmp: Mir.Cmp = if (is_null) .eq else .ne;
        const null_value = if (optional_type.optionalChild(zcu).zigTypeTag(zcu) == .pointer) optional_type.nullPtrAddress(zcu) else 0;
        return func.single(try func.builder.emit(.pred, .setp, flag.class.unsignedType(), .{ .setp = .{ .cmp = cmp } }, .{ flag, null_value }));
    }
    if (!is_null) return func.single(flag);
    return func.single(try func.builder.emit(.pred, .not, .pred, .{ .none = .{} }, .{flag}));
}

/// An optional of a pointer or slice stores null as `Type.nullPtrAddress` in its first scalar, and
/// an optional of an error set stores null as 0. Any other optional has a `.pred` flag as its last
/// scalar.
fn optionalFlag(func: *Function, ref: Air.Inst.Ref, optional_type: Type, is_ptr: bool) Layout.Error!Mir.Reg {
    const scalars = try func.layout.scalarsOf(optional_type);
    assert(scalars.len != 0);
    const index = if (optional_type.optionalReprIsPayload(func.zcu)) 0 else scalars.len - 1;
    const flag = scalars[index];
    if (!is_ptr) return func.builder.toReg((try func.partsOf(ref))[index], flag.kind);
    return func.builder.toReg((try func.loadThrough(ref, scalars[index..][0..1]))[0], flag.kind);
}

pub fn airPayload(func: *Function, inst: Air.Inst.Index, container: enum { optional, error_union }) Layout.Error!Val {
    const ty_op = func.dataOf(inst).ty_op;
    const src = try func.partsOf(ty_op.operand);
    const count = try func.layout.scalarCount(ty_op.ty);
    return .{ .parts = switch (container) {
        .optional => src[0..count],
        .error_union => src[src.len - count ..],
    } };
}

pub fn airPayloadPtr(func: *Function, inst: Air.Inst.Index, container: enum { optional, error_union }, set: bool) Layout.Error!Val {
    const zcu = func.zcu;
    const ty_op = func.dataOf(inst).ty_op;
    const ptr_type = func.typeOf(ty_op.operand);
    const ty = ptr_type.childType(zcu);
    const base = try func.resolve(ty_op.operand);
    switch (container) {
        .optional => {
            if (set and !ty.optionalReprIsPayload(zcu)) {
                const flag_offset: u32 = @intCast(ty.optionalChild(zcu).abiSize(zcu));
                try func.storeVal(base, ptr_type, flag_offset, &.{.{ .imm = 1 }}, .bool);
            }
            return base;
        },
        .error_union => {
            const payload_type = ty.errorUnionPayload(zcu);
            if (set and ty.errorUnionSet(zcu).hasRuntimeBits(zcu)) {
                const err_offset: u32 = @intCast(codegen.errUnionErrorOffset(payload_type, zcu));
                try func.storeVal(base, ptr_type, err_offset, &.{.{ .imm = 0 }}, ty.errorUnionSet(zcu));
            }
            return func.offsetPtr(base, ptr_type, @intCast(codegen.errUnionPayloadOffset(payload_type, zcu)));
        },
    }
}

pub fn airWrapOptional(func: *Function, inst: Air.Inst.Index) Layout.Error!Val {
    const ty_op = func.dataOf(inst).ty_op;
    const payload = try func.partsOf(ty_op.operand);
    if (ty_op.ty.optionalReprIsPayload(func.zcu)) return .{ .parts = payload };
    const out = try func.arena.alloc(Builder.Part, payload.len + 1);
    @memcpy(out[0..payload.len], payload);
    out[payload.len] = .{ .imm = 1 };
    return .{ .parts = out };
}

pub fn airIsErr(func: *Function, inst: Air.Inst.Index, is_err: bool, is_ptr: bool) Layout.Error!Val {
    const zcu = func.zcu;
    const un_op = func.dataOf(inst).un_op;
    const error_union_type = if (is_ptr) func.typeOf(un_op).childType(zcu) else func.typeOf(un_op);
    if (!error_union_type.errorUnionSet(zcu).hasRuntimeBits(zcu)) return func.single(try func.builder.movImm(.pred, @intFromBool(!is_err)));
    const cmp: Mir.Cmp = if (is_err) .ne else .eq;
    return func.single(try func.builder.emit(.pred, .setp, .u16, .{ .setp = .{ .cmp = cmp } }, .{ try errorCode(func, un_op, error_union_type, is_ptr), 0 }));
}

pub fn errorCode(func: *Function, ref: Air.Inst.Ref, error_union_type: Type, is_ptr: bool) Layout.Error!Mir.Reg {
    const zcu = func.zcu;
    const kind = try func.layout.kindOf(error_union_type.errorUnionSet(zcu));
    if (!is_ptr) return func.builder.toReg((try func.partsOf(ref))[0], kind);
    const scalar: Layout.Scalar = .{ .offset = @intCast(codegen.errUnionErrorOffset(error_union_type.errorUnionPayload(zcu), zcu)), .kind = kind };
    return func.builder.toReg((try func.loadThrough(ref, &.{scalar}))[0], kind);
}

pub fn airUnwrapErr(func: *Function, inst: Air.Inst.Index, is_ptr: bool) Layout.Error!Val {
    const zcu = func.zcu;
    const ty_op = func.dataOf(inst).ty_op;
    const error_union_type = if (is_ptr) func.typeOf(ty_op.operand).childType(zcu) else func.typeOf(ty_op.operand);
    assert(error_union_type.errorUnionSet(zcu).hasRuntimeBits(zcu));
    return func.single(try errorCode(func, ty_op.operand, error_union_type, is_ptr));
}

pub fn airWrapErrUnion(func: *Function, inst: Air.Inst.Index, is_payload: bool) Layout.Error!Val {
    const zcu = func.zcu;
    const ty_op = func.dataOf(inst).ty_op;
    const error_union_type = ty_op.ty;
    const operand = try func.partsOf(ty_op.operand);
    var list: std.ArrayList(Builder.Part) = .empty;
    if (is_payload) {
        if (error_union_type.errorUnionSet(zcu).hasRuntimeBits(zcu)) try list.append(func.arena, .{ .imm = 0 });
        try list.appendSlice(func.arena, operand);
    } else {
        try list.appendSlice(func.arena, operand);
        for (try func.layout.scalarsOf(error_union_type.errorUnionPayload(zcu))) |_| try list.append(func.arena, .undef);
    }
    return .{ .parts = list.items };
}

/// A tag value that equals its field index needs no lookup. Otherwise a chain of `selp` maps
/// each value to its index. A value without a field yields 0.
fn enumFieldIndex(func: *Function, enum_type: Type, value: Mir.Reg) Layout.Error!Mir.Reg {
    if (func.layout.enumIsDense(enum_type)) return int.widen(&func.builder, value, try func.layout.kindOf(enum_type));
    const kind = try func.layout.kindOf(enum_type);
    var index = try func.builder.movImm(.b64, 0);
    for (1..enum_type.enumFieldCount(func.zcu)) |i| {
        const bits = try func.builder.toSource(.{ .imm = func.layout.enumFieldBits(enum_type, i) }, kind);
        const match = try func.compareSource(.eq, value, bits, kind);
        index = try func.builder.emit(.b64, .selp, .b64, .{ .none = .{} }, .{ i, index, match });
    }
    return index;
}

/// Each entry of a name table is a 16-byte pair of pointer and length.
fn loadName(func: *Function, table: Mir.Target, index: Mir.Reg) Layout.Error!Val {
    const base = try func.builder.toReg(.{ .symbol = .{ .target = table, .offset = 0, .space = .global, .address_space = .global } }, .pointer);
    const entry = try func.builder.emit(.b64, .mad, .u64, .{ .arith = .{ .mode = .lo, .address = true } }, .{ index, 16, base });
    return .{ .parts = try memory.loadMem(&func.builder, .{ .reg = entry }, .global, 8, &.{
        .{ .offset = 0, .kind = .pointer },
        .{ .offset = 8, .kind = Kind.int(.unsigned, 64) },
    }, false) };
}

pub fn airTagName(func: *Function, inst: Air.Inst.Index) Layout.Error!Val {
    const un_op = func.dataOf(inst).un_op;
    const enum_type = func.typeOf(un_op);
    const index = if (func.layout.wideIntOf(enum_type)) |wide_int|
        try wideEnumFieldIndex(func, enum_type, try wide.limbRegs(&func.builder, try func.partsOf(un_op), wide_int), wide_int)
    else if (enum_type.hasRuntimeBits(func.zcu))
        try enumFieldIndex(func, enum_type, try func.scalarReg(un_op))
    else
        try func.builder.movImm(.b64, 0);
    return loadName(func, .{ .tag_names = enum_type.toIntern() }, index);
}

pub fn airErrorName(func: *Function, inst: Air.Inst.Index) Layout.Error!Val {
    const un_op = func.dataOf(inst).un_op;
    return loadName(func, .error_names, try int.widen(&func.builder, try func.scalarReg(un_op), try func.layout.kindOf(func.typeOf(un_op))));
}

pub fn airIsNamedEnumValue(func: *Function, inst: Air.Inst.Index) Layout.Error!Val {
    const un_op = func.dataOf(inst).un_op;
    const enum_type = func.typeOf(un_op);
    if (!enum_type.hasRuntimeBits(func.zcu)) return func.single(try func.builder.movImm(.pred, 1));
    const count = enum_type.enumFieldCount(func.zcu);
    const matches = try func.arena.alloc(Mir.Reg, count);
    if (func.layout.wideIntOf(enum_type)) |wide_int| {
        const value = try wide.limbRegs(&func.builder, try func.partsOf(un_op), wide_int);
        for (matches, 0..) |*match, i| match.* = try wide.compareLimbs(&func.builder, .eq, value, try enumFieldLimbs(func, enum_type, i, wide_int), wide_int.signedness);
        return func.single(try int.anyOf(&func.builder, matches));
    }
    const value = try func.scalarReg(un_op);
    if (func.layout.enumIsDense(enum_type)) {
        return func.single(try func.builder.emit(.pred, .setp, .u64, .{ .setp = .{ .cmp = .lt } }, .{ try int.widen(&func.builder, value, try func.layout.kindOf(enum_type)), count }));
    }
    const kind = try func.layout.kindOf(enum_type);
    for (matches, 0..) |*match, i| match.* = try func.compareSource(.eq, value, try func.builder.toSource(.{ .imm = func.layout.enumFieldBits(enum_type, i) }, kind), kind);
    return func.single(try int.anyOf(&func.builder, matches));
}

pub fn airErrorSetHasValue(func: *Function, inst: Air.Inst.Index) Layout.Error!Val {
    const zcu = func.zcu;
    const ty_op = func.dataOf(inst).ty_op;
    if (ty_op.ty.toIntern() == .anyerror_type) return func.single(try func.builder.movImm(.pred, 1));
    const kind = try func.layout.kindOf(func.typeOf(ty_op.operand));
    const value = try func.scalarReg(ty_op.operand);
    const names = ty_op.ty.errorSetNames(zcu).get(&zcu.intern_pool);
    const matches = try func.arena.alloc(Mir.Reg, names.len);
    for (matches, names) |*match, name| {
        match.* = try func.compareSource(.eq, value, try func.builder.toSource(.{ .imm = try func.pt.getErrorValue(name) }, kind), kind);
    }
    return func.single(try int.anyOf(&func.builder, matches));
}

/// Error value 0 has no name. Entry 0 of the table therefore holds the number of errors in its
/// length word.
pub fn airCmpLteErrorsLen(func: *Function, inst: Air.Inst.Index) Layout.Error!Val {
    const un_op = func.dataOf(inst).un_op;
    const len = (try loadName(func, .error_names, try func.builder.movImm(.b64, 0))).parts[1].reg;
    const value = try int.widen(&func.builder, try func.scalarReg(un_op), try func.layout.kindOf(func.typeOf(un_op)));
    return func.single(try func.builder.emit(.pred, .setp, .u64, .{ .setp = .{ .cmp = .le } }, .{ value, len }));
}

pub fn airBitCast(func: *Function, inst: Air.Inst.Index) Layout.Error!Val {
    const zcu = func.zcu;
    const ty_op = func.dataOf(inst).ty_op;
    const dest_type = ty_op.ty;
    const src_type = func.typeOf(ty_op.operand);
    if (func.layout.isBitPacked(src_type) or func.layout.isBitPacked(dest_type)) {
        return bitCastPacked(func, src_type, dest_type, try func.partsOf(ty_op.operand));
    }
    const src = try func.resolve(ty_op.operand);
    if (dest_type.zigTypeTag(zcu) == .pointer and src_type.zigTypeTag(zcu) == .pointer and dest_type.ptrInfo(zcu).flags.vector_index != .none) {
        return func.offsetPtr(src, src_type, @intCast(try func.layout.vectorElemOffset(src_type, dest_type)));
    }
    const src_scalars = try func.layout.scalarsOf(src_type);
    const dest_scalars = try func.layout.scalarsOf(dest_type);
    const same = src_scalars.len == dest_scalars.len and for (src_scalars, dest_scalars) |from, to| {
        if (from.offset != to.offset or from.kind.size != to.kind.size or from.kind.class.bits() != to.kind.class.bits()) break false;
    } else true;
    if (same) {
        const out = try func.arena.alloc(Builder.Part, dest_scalars.len);
        for (out, src.parts, src_scalars, dest_scalars) |*result, part, from, to| {
            // The upper bits of a narrow integer register hold its zero or sign extension. A change
            // of signedness therefore rewrites them.
            const reextend = from.kind.repr != to.kind.repr and from.kind.repr != .float and to.kind.repr != .float and
                to.kind.bits != to.kind.class.bits();
            if ((from.kind.class == to.kind.class or part != .reg) and !reextend) {
                result.* = part;
                continue;
            }
            const moved = try func.builder.emit(to.kind.class, .mov, to.kind.bitType(), .{ .none = .{} }, .{try func.builder.toReg(part, from.kind)});
            result.* = .{ .reg = if (reextend) try int.normalized(&func.builder, moved, to.kind) else moved };
        }
        if (func.layout.wideIntOf(dest_type)) |wide_int| {
            // The limbs above the value hold the extension, which depends on the signedness.
            if (wide_int.limbs > wide_int.topLimb() + 1) return .{ .parts = try func.builder.regParts(try wide.normalizedLimbs(&func.builder, try wide.limbRegs(&func.builder, out, wide_int), wide_int)) };
        }
        return .{ .parts = out };
    }
    // `mov` cannot move between `.pred` and integer registers, so `selp` and `setp` convert.
    if (src_scalars.len == 1 and dest_scalars.len == 1) {
        const from = src_scalars[0].kind;
        const to = dest_scalars[0].kind;
        if (from.class == .pred) return func.single(try func.builder.emit(to.class, .selp, to.bitType(), .{ .none = .{} }, .{ 1, 0, try func.builder.toReg(src.parts[0], from) }));
        if (to.class == .pred) return func.single(try func.builder.emit(.pred, .setp, from.arithmeticType(), .{ .setp = .{ .cmp = .ne } }, .{ try func.builder.toReg(src.parts[0], from), 0 }));
        // A cast between unsigned scalars of different sizes converts the register. Therefore
        // `@intFromPtr` zero-extends a 32-bit pointer, and `@ptrFromInt` keeps its low 32 bits.
        if (from.repr == .unsigned and to.repr == .unsigned) return func.single(try int.intConvert(&func.builder, try func.builder.toReg(src.parts[0], from), from, to));
    }
    // The scalars differ in offset or size. The backend therefore stores the value with the source
    // layout. Then it loads the value with the destination layout.
    const size = @max(src_type.abiSize(zcu), dest_type.abiSize(zcu));
    const local = try func.builder.newLocal(size, 8, .generic, .pointer);
    try memory.storeMem(&func.builder, .{ .reg = local }, .generic, 8, src_scalars, src.parts, false);
    const dst = try memory.loadRegs(&func.builder, .{ .reg = local }, .generic, 8, dest_scalars, false, true);
    // The source can leave the bytes above its value undefined, such as the six bytes after the
    // exponent of an f80. `normalizedLimbs` writes the extension of a wide integer into them.
    if (func.layout.wideIntOf(dest_type)) |wide_int| return .{ .parts = try func.builder.regParts(try wide.normalizedLimbs(&func.builder, dst, wide_int)) };
    return .{ .parts = try func.builder.regParts(dst) };
}

/// Moves the bits of the cast through 64-bit limbs, where bit `n` of the value is bit `n % 64` of
/// limb `n / 64`.
fn bitCastPacked(func: *Function, src_type: Type, dest_type: Type, src: []const Builder.Part) Layout.Error!Val {
    const zcu = func.zcu;
    const limb_count = std.math.divCeil(u64, src_type.bitSize(zcu), 64) catch unreachable;
    const limb_scalars = try func.arena.alloc(Layout.Scalar, limb_count);
    for (limb_scalars, 0..) |*limb, i| limb.* = .{ .offset = @intCast(i * 8), .kind = Kind.int(.unsigned, 64) };
    const size = @max(limb_count * 8, src_type.abiSize(zcu), dest_type.abiSize(zcu));
    var memory_ptr: ?Mir.Reg = null;
    const limbs = if (func.layout.isBitPacked(src_type)) limbs: {
        const limbs = try func.arena.alloc(Builder.Part, limb_count);
        @memset(limbs, .{ .imm = 0 });
        try packBits(func, src_type, src, limbs, 0);
        break :limbs limbs;
    } else limbs: {
        memory_ptr = try func.builder.newLocal(size, 8, .generic, .pointer);
        try memory.storeMem(&func.builder, .{ .reg = memory_ptr.? }, .generic, 8, try func.layout.scalarsOf(src_type), src, false);
        break :limbs try memory.loadMem(&func.builder, .{ .reg = memory_ptr.? }, .generic, 8, limb_scalars, false);
    };
    if (func.layout.isBitPacked(dest_type)) {
        var list: std.ArrayList(Builder.Part) = .empty;
        try unpackBits(func, dest_type, limbs, 0, &list);
        return .{ .parts = list.items };
    }
    const ptr = memory_ptr orelse try func.builder.newLocal(size, 8, .generic, .pointer);
    try memory.storeMem(&func.builder, .{ .reg = ptr }, .generic, 8, limb_scalars, limbs, false);
    const dest_scalars = try func.layout.scalarsOf(dest_type);
    const dst = try memory.loadRegs(&func.builder, .{ .reg = ptr }, .generic, 8, dest_scalars, false, true);
    // The limbs hold zeros above the bits of the value, and an integer register needs its sign
    // extension there.
    const out = try func.arena.alloc(Builder.Part, dst.len);
    for (out, dst, dest_scalars) |*result, reg, scalar| result.* = .{ .reg = try int.normalized(&func.builder, reg, scalar.kind) };
    return .{ .parts = out };
}

fn packBits(func: *Function, ty: Type, src: []const Builder.Part, limbs: []Builder.Part, bit: u64) Layout.Error!void {
    const zcu = func.zcu;
    if (ty.isArrayOrVector(zcu)) {
        const elem_type = ty.childType(zcu);
        const count = try func.layout.scalarCount(elem_type);
        for (0..ty.arrayLenIncludingSentinel(zcu)) |i| {
            try packBits(func, elem_type, src[i * count ..][0..count], limbs, bit + i * elem_type.bitSize(zcu));
        }
        return;
    }
    const bit_size = ty.bitSize(zcu);
    // Each scalar of a wide integer, a wide packed type, an f80 or an f128 holds the bits of the
    // value from bit `8 * offset` on.
    for (try func.layout.scalarsOf(ty), src) |scalar, part| {
        if (8 * scalar.offset >= bit_size) return func.layout.todo("@bitCast of '{f}' in a bit-packed aggregate", .{ty.fmt(zcu)});
        try int.packScalar(&func.builder, scalar.kind, part, limbs, bit + 8 * scalar.offset);
    }
}

fn unpackBits(func: *Function, ty: Type, limbs: []const Builder.Part, bit: u64, out: *std.ArrayList(Builder.Part)) Layout.Error!void {
    const zcu = func.zcu;
    if (ty.isArrayOrVector(zcu)) {
        const elem_type = ty.childType(zcu);
        for (0..ty.arrayLenIncludingSentinel(zcu)) |i| {
            try unpackBits(func, elem_type, limbs, bit + i * elem_type.bitSize(zcu), out);
        }
        return;
    }
    const bit_size = ty.bitSize(zcu);
    for (try func.layout.scalarsOf(ty)) |scalar| {
        if (8 * scalar.offset >= bit_size) return func.layout.todo("@bitCast to '{f}' in a bit-packed aggregate", .{ty.fmt(zcu)});
        try out.append(func.arena, .{ .reg = try int.unpackScalar(&func.builder, scalar.kind, limbs, bit + 8 * scalar.offset) });
    }
}

fn enumFieldLimbs(func: *Function, ty: Type, index: usize, wide_int: Layout.WideInt) Layout.Error![]const Mir.Reg {
    const ip = &func.zcu.intern_pool;
    const values = ip.loadEnumType(ty.toIntern()).field_values;
    var list: std.ArrayList(Builder.Part) = .empty;
    if (values.len == 0) {
        try list.append(func.arena, .{ .imm = index });
        while (list.items.len < wide_int.limbs) try list.append(func.arena, .{ .imm = 0 });
    } else {
        const val: Value = .fromInterned(values.get(ip)[index]);
        try constant.constParts(&func.builder, &func.layout, &list, val.typeOf(func.zcu), val);
    }
    return wide.limbRegs(&func.builder, list.items, wide_int);
}

fn wideEnumFieldIndex(func: *Function, ty: Type, value: []const Mir.Reg, wide_int: Layout.WideInt) Layout.Error!Mir.Reg {
    var index = try func.builder.movImm(.b64, 0);
    for (1..ty.enumFieldCount(func.zcu)) |i| {
        const match = try wide.compareLimbs(&func.builder, .eq, value, try enumFieldLimbs(func, ty, i, wide_int), wide_int.signedness);
        index = try func.builder.emit(.b64, .selp, .b64, .{ .none = .{} }, .{ i, index, match });
    }
    return index;
}
