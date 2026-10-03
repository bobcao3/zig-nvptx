//! This file holds the state of one function and resolves its AIR operands, memory and parameters.
//! `binary`, `unary` and `compareSource` choose between the integer and float lowerings.

const std = @import("std");
const Allocator = std.mem.Allocator;
const CompareOperator = std.math.CompareOperator;

const Air = @import("../../../Air.zig");
const Type = @import("../../../Type.zig");
const Value = @import("../../../Value.zig");
const Zcu = @import("../../../Zcu.zig");
const Mir = @import("../Mir.zig");
const Kind = @import("Kind.zig");
const Layout = @import("Layout.zig");
const Builder = @import("Builder.zig");
const int = @import("int.zig");
const float = @import("float.zig");
const memory = @import("memory.zig");
const constant = @import("constant.zig");
const assert = std.debug.assert;
const call = @import("call.zig");
const elision = @import("elision.zig");

const Function = @This();

pt: Zcu.PerThread,
zcu: *Zcu,
gpa: Allocator,
arena: Allocator,
air: Air,
liveness: Air.Liveness,
builder: Builder,
layout: Layout,
/// `elision.keepsSource` computes the facts about the AIR at the first large load, and the field is
/// null until then.
elision_facts: ?elision.Facts = null,
ret_type: Type = .void,
is_kernel: bool = false,
/// The calling convention of the function.
cc: std.lang.CallingConvention = .auto,
values: std.AutoHashMapUnmanaged(Air.Inst.Index, Val) = .empty,
param_allocs: std.AutoHashMapUnmanaged(Air.Inst.Index, u32) = .empty,
/// The local variables of `param_allocs` in a kernel.
param_copies: std.ArrayList(Mir.ParamCopy) = .empty,

pub const Val = union(enum) {
    /// Each scalar of the type has one operand.
    parts: []const Builder.Part,
    /// No instruction writes the memory of the value while the value lives. `resolve` loads the
    /// value at each use.
    place: Place,
};

pub const Place = struct {
    base: union(enum) {
        /// The index of a kernel parameter in `.param` memory.
        param: u32,
        address: struct {
            reg: Mir.Reg,
            address_space: std.lang.AddressSpace,
            alignment: u64,
        },
    },
    offset: u32,

    pub fn addressSpace(place: Place) std.lang.AddressSpace {
        return switch (place.base) {
            .param => .param,
            .address => |address| address.address_space,
        };
    }
};

pub fn typeOf(func: *Function, ref: Air.Inst.Ref) Type {
    return func.air.typeOf(ref, &func.zcu.intern_pool);
}

pub fn typeOfIndex(func: *Function, inst: Air.Inst.Index) Type {
    return func.air.typeOfIndex(inst, &func.zcu.intern_pool);
}

pub fn dataOf(func: *Function, inst: Air.Inst.Index) Air.Inst.Data {
    return func.air.instructions.items(.data)[@backingInt(inst)];
}

pub fn binFromPayload(func: *Function, inst: Air.Inst.Index) Air.Bin {
    return func.air.extraData(Air.Bin, func.dataOf(inst).ty_pl.payload).data;
}

pub fn resolve(func: *Function, ref: Air.Inst.Ref) Layout.Error!Val {
    if (ref.toInterned()) |ip_index| {
        const val: Value = .fromInterned(ip_index);
        var list: std.ArrayList(Builder.Part) = .empty;
        try constant.constParts(&func.builder, &func.layout, &list, val.typeOf(func.zcu), val);
        return .{ .parts = list.items };
    }
    return switch (func.values.get(ref.toIndex().?).?) {
        .place => |place| .{ .parts = try func.loadPlace(place, func.typeOf(ref)) },
        else => |val| val,
    };
}

pub fn partsOf(func: *Function, ref: Air.Inst.Ref) Layout.Error![]const Builder.Part {
    return switch (try func.resolve(ref)) {
        .place => unreachable,
        .parts => |parts| parts,
    };
}

pub fn scalarReg(func: *Function, ref: Air.Inst.Ref) Layout.Error!Mir.Reg {
    const parts = try func.partsOf(ref);
    if (parts.len != 1) return func.layout.todo("an operand of type '{f}' wider than 64 bits", .{func.typeOf(ref).fmt(func.zcu)});
    const scalar = (try func.layout.scalarsOf(func.typeOf(ref)))[0];
    return func.builder.toReg(parts[0], scalar.kind);
}

pub fn single(func: *Function, reg: Mir.Reg) !Val {
    const parts = try func.arena.alloc(Builder.Part, 1);
    parts[0] = .{ .reg = reg };
    return .{ .parts = parts };
}

/// Returns true when `alloc` is a copy of a parameter in a pointer into `.param`. Such a pointer
/// cannot point to a local variable, so the code uses the `.param` address of the parameter.
pub fn isParamSpaceCopy(func: *const Function, alloc: Air.Inst.Index) bool {
    return func.param_allocs.contains(alloc) and func.air.typeOfIndex(alloc, &func.zcu.intern_pool).ptrAddressSpace(func.zcu) == .param;
}

/// Returns the value that `ptr_ref` points to. A large aggregate gets a place at the source when
/// the AIR `load` shows that the source keeps its bytes, and else a copy. `load` can be null.
pub fn loadPtr(func: *Function, ptr_ref: Air.Inst.Ref, ty: Type, load: ?Air.Inst.Index) Layout.Error!Val {
    const zcu = func.zcu;
    const ptr_type = func.typeOf(ptr_ref);
    const scalars = try func.layout.scalarsOf(ty);
    switch (try func.resolve(ptr_ref)) {
        .place => unreachable,
        .parts => |parts| {
            const ptr = try func.ptrReg(parts[0], ptr_type);
            const address_space = ptr_type.ptrAddressSpace(zcu);
            const alignment = try func.knownAlignment(ptr_ref);
            if (!ptr_type.isVolatilePtr(zcu) and func.layout.isAggregate(ty) and try func.isLarge(ty)) {
                const src: Place = .{ .base = .{ .address = .{ .reg = ptr, .address_space = address_space, .alignment = alignment } }, .offset = 0 };
                if (load) |inst| if (try elision.keepsSource(func, inst, ptr_ref)) return .{ .place = src };
                const copy_alignment = ty.abiAlignment(zcu).toByteUnits().?;
                const copy = try func.builder.newLocal(ty.abiSize(zcu), copy_alignment, .local, func.layout.ptrKind(.local));
                try func.copyMemory(copy, .local, copy_alignment, src, ty);
                return .{ .place = .{ .base = .{ .address = .{ .reg = copy, .address_space = .local, .alignment = copy_alignment } }, .offset = 0 } };
            }
            return .{ .parts = try memory.loadMem(&func.builder, .{ .reg = ptr }, address_space, alignment, scalars, ptr_type.isVolatilePtr(zcu)) };
        },
    }
}

/// Returns the vector type and the element index when `ptr` points to an element of a bit-packed
/// vector. The pointer type gives the index, except for the `ptr_elem_ptr` instructions with which
/// Sema fills a vector literal in memory. Their index operand gives the index.
pub fn bitPackedElem(func: *Function, ptr: Air.Inst.Ref) Layout.Error!?struct { Type, u32 } {
    const zcu = func.zcu;
    const ptr_type = func.typeOf(ptr);
    if (try func.layout.bitPackedHost(ptr_type)) |vector_type| return .{ vector_type, @backingInt(ptr_type.ptrInfo(zcu).flags.vector_index) };
    const inst = ptr.toIndex() orelse return null;
    if (func.air.instructions.items(.tag)[@backingInt(inst)] != .ptr_elem_ptr) return null;
    const bin = func.binFromPayload(inst);
    const base_type = func.typeOf(bin.lhs);
    if (base_type.ptrSize(zcu) != .one) return null;
    const vector_type = base_type.childType(zcu);
    if (!func.layout.isBitPackedVector(vector_type)) return null;
    return .{ vector_type, @intCast(Value.fromInterned(bin.rhs.toInterned().?).toUnsignedInt(zcu)) };
}

pub fn loadVector(func: *Function, elem_ptr: Air.Inst.Ref, vector_type: Type) Layout.Error![]const Builder.Part {
    return switch (try func.loadPtr(elem_ptr, vector_type, null)) {
        .parts => |parts| parts,
        .place => |place| func.loadPlace(place, vector_type),
    };
}

/// In Debug and ReleaseSafe mode, Zig writes 0xaa bytes to undefined memory. The `_safe` forms of
/// AIR instructions request that pattern.
pub fn safeParts(func: *Function, ref: Air.Inst.Ref, safety: bool) Layout.Error![]const Builder.Part {
    const ip_index = ref.toInterned() orelse return func.partsOf(ref);
    if (!safety or !Value.fromInterned(ip_index).isUndef(func.zcu)) return func.partsOf(ref);
    const scalars = try func.layout.scalarsOf(func.typeOf(ref));
    const parts = try func.arena.alloc(Builder.Part, scalars.len);
    for (parts, scalars) |*part, scalar| part.* = .{ .imm = @as(u64, 0xaaaa_aaaa_aaaa_aaaa) >> @intCast(64 - @as(u64, scalar.kind.size) * 8) };
    return parts;
}

/// `st` cannot write to `.constant` memory (PTX ISA 9.4, section 9.7.10.11).
pub fn checkWritable(func: *const Function, address_space: std.lang.AddressSpace) Layout.Error!void {
    if (address_space == .constant) return func.layout.fail("store to address space '{t}'", .{address_space});
}

pub fn store(func: *Function, ptr_ref: Air.Inst.Ref, src: []const Builder.Part, ty: Type) Layout.Error!void {
    const zcu = func.zcu;
    const ptr_type = func.typeOf(ptr_ref);
    try func.checkWritable(ptr_type.ptrAddressSpace(zcu));
    const scalars = try func.layout.scalarsOf(ty);
    switch (try func.resolve(ptr_ref)) {
        .place => unreachable,
        .parts => |parts| try memory.storeMem(&func.builder, .{ .reg = try func.ptrReg(parts[0], ptr_type) }, ptr_type.ptrAddressSpace(zcu), try func.knownAlignment(ptr_ref), scalars, src, ptr_type.isVolatilePtr(zcu)),
    }
}

pub fn storePlace(func: *Function, ptr_ref: Air.Inst.Ref, place: Place, ty: Type) Layout.Error!void {
    const zcu = func.zcu;
    const ptr_type = func.typeOf(ptr_ref);
    if (ptr_type.isVolatilePtr(zcu)) return func.store(ptr_ref, try func.loadPlace(place, ty), ty);
    const dest = try func.ptrReg((try func.partsOf(ptr_ref))[0], ptr_type);
    try func.copyMemory(dest, ptr_type.ptrAddressSpace(zcu), try func.knownAlignment(ptr_ref), place, ty);
}

/// Copies the bytes of `ty` from `src` to `dest` with one `copy` instruction of the widest unit
/// that the size and the alignments allow. Promotion moves the variables at both ends to registers
/// where it can, and `transforms/expand.zig` lowers the other copies.
pub fn copyMemory(func: *Function, dest: Mir.Reg, dest_address_space: std.lang.AddressSpace, dest_alignment: u64, src: Place, ty: Type) Layout.Error!void {
    const zcu = func.zcu;
    const size = ty.abiSize(zcu);
    assert(size != 0);
    try func.checkWritable(dest_address_space);
    const max_unit = @min(dest_alignment, func.placeAlignment(src), ty.abiAlignment(zcu).toByteUnits().?);
    var unit: u64 = 16;
    while (unit > max_unit or size % unit != 0) unit /= 2;
    try func.builder.emitEffect(.copy, .none, .{ .copy = .{} }, .{
        dest,
        try func.placeAddress(src),
        size / unit,
        memory.memOperand(&func.builder, dest_address_space, .none, unit, unit, false),
        memory.memOperand(&func.builder, src.addressSpace(), .none, unit, unit, true),
    });
}

/// Returns true when `ptr` points into an alloc, through casts, field pointers, element pointers
/// and payload pointers.
pub fn allocRoot(func: *const Function, ptr: Air.Inst.Ref) bool {
    const air = func.air;
    var inst = ptr.toIndex() orelse return false;
    while (true) {
        const data = air.instructions.items(.data)[@backingInt(inst)];
        const next: Air.Inst.Ref = switch (air.instructions.items(.tag)[@backingInt(inst)]) {
            .alloc, .ret_ptr => return !func.isParamSpaceCopy(inst),
            .ptr_cast,
            .bit_cast,
            .addrspace_cast,
            .optional_payload_ptr,
            .optional_payload_ptr_set,
            .unwrap_errunion_payload_ptr,
            .errunion_payload_ptr_set,
            .struct_field_ptr_index_0,
            .struct_field_ptr_index_1,
            .struct_field_ptr_index_2,
            .struct_field_ptr_index_3,
            => data.ty_op.operand,
            .struct_field_ptr => air.extraData(Air.StructField, data.ty_pl.payload).data.struct_operand,
            .ptr_elem_ptr, .ptr_add, .ptr_sub => air.extraData(Air.Bin, data.ty_pl.payload).data.lhs,
            .field_parent_ptr => air.extraData(Air.FieldParentPtr, data.ty_pl.payload).data.field_ptr,
            .try_ptr, .try_ptr_cold => air.extraData(Air.TryPtr, data.ty_pl.payload).data.ptr,
            else => return false,
        };
        inst = next.toIndex() orelse return false;
    }
}

/// Returns the larger of the type alignment of `ptr` and the alignment that `basePointer` implies.
pub fn knownAlignment(func: *Function, ptr: Air.Inst.Ref) Layout.Error!u64 {
    const own = func.layout.ptrAlignment(func.typeOf(ptr));
    const inst = ptr.toIndex() orelse return own;
    const base_ptr, const offset = try func.basePointer(inst) orelse return own;
    return @max(own, memory.offsetAlignment(try func.knownAlignment(base_ptr), offset));
}

/// Returns the base pointer and the constant byte offset of `inst`, or null without a base pointer.
fn basePointer(func: *Function, inst: Air.Inst.Index) Layout.Error!?struct { Air.Inst.Ref, u64 } {
    const zcu = func.zcu;
    return switch (func.air.instructions.items(.tag)[@backingInt(inst)]) {
        .ptr_cast => .{ func.dataOf(inst).ty_op.operand, try func.layout.vectorElemOffset(func.typeOf(func.dataOf(inst).ty_op.operand), func.dataOf(inst).ty_op.ty) },
        .struct_field_ptr => lowered: {
            const extra = func.air.extraData(Air.StructField, func.dataOf(inst).ty_pl.payload).data;
            break :lowered try func.fieldPtrOffset(extra.struct_operand, extra.field_index);
        },
        .struct_field_ptr_index_0 => try func.fieldPtrOffset(func.dataOf(inst).ty_op.operand, 0),
        .struct_field_ptr_index_1 => try func.fieldPtrOffset(func.dataOf(inst).ty_op.operand, 1),
        .struct_field_ptr_index_2 => try func.fieldPtrOffset(func.dataOf(inst).ty_op.operand, 2),
        .struct_field_ptr_index_3 => try func.fieldPtrOffset(func.dataOf(inst).ty_op.operand, 3),
        .ptr_elem_ptr, .ptr_add => lowered: {
            const bin = func.binFromPayload(inst);
            if (try func.bitPackedElem(inst.toRef()) != null) break :lowered .{ bin.lhs, 0 };
            const index = Value.fromInterned(bin.rhs.toInterned() orelse return null).toUnsignedInt(zcu);
            break :lowered .{ bin.lhs, index * func.typeOfIndex(inst).childType(zcu).abiSize(zcu) };
        },
        else => null,
    };
}

fn fieldPtrOffset(func: *Function, ptr: Air.Inst.Ref, index: u32) Layout.Error!struct { Air.Inst.Ref, u64 } {
    return .{ ptr, try func.layout.fieldOffset(func.typeOf(ptr).childType(func.zcu), index) };
}

pub fn loadPlace(func: *Function, place: Place, ty: Type) Layout.Error![]const Builder.Part {
    const scalars = try func.layout.offsetScalars(try func.layout.scalarsOf(ty), place.offset);
    return switch (place.base) {
        .param => |index| func.builder.regParts(try memory.loadRegs(&func.builder, .{ .name = try call.paramName(&func.builder, index) }, .param, func.paramAlignment(index), scalars, false, false)),
        .address => |address| memory.loadMem(&func.builder, .{ .reg = address.reg }, address.address_space, address.alignment, scalars, false),
    };
}

/// Returns the alignment that `call.declare` gives the `.param` variable `param_{index}`.
pub fn paramAlignment(func: *Function, index: u32) u64 {
    const zcu = func.zcu;
    const ip = &zcu.intern_pool;
    const fn_info = zcu.typeToFunc(.fromInterned(ip.getNav(func.layout.owner_nav).resolved.?.type)).?;
    var runtime_index: u32 = 0;
    for (fn_info.param_types.get(ip)) |param_type_index| {
        const param_type: Type = .fromInterned(param_type_index);
        if (!param_type.hasRuntimeBits(zcu)) continue;
        if (runtime_index == index) {
            if (call.passesByReference(&func.layout, fn_info.cc, param_type)) return func.layout.ptrKind(.generic).size;
            return @min(call.max_param_alignment, param_type.abiAlignment(zcu).toByteUnits().?);
        }
        runtime_index += 1;
    }
    unreachable;
}

pub fn placeOf(func: *Function, ref: Air.Inst.Ref) ?Place {
    return switch (func.values.get(ref.toIndex() orelse return null) orelse return null) {
        .place => |place| place,
        else => null,
    };
}

/// Returns true when `ty` has more than `Layout.max_register_bytes` bytes or no form in registers.
fn isLarge(func: *Function, ty: Type) Layout.Error!bool {
    if (ty.abiSize(func.zcu) > Layout.max_register_bytes) return true;
    _ = func.layout.split(ty) catch |err| switch (err) {
        error.CannotSplit => return true,
        else => |e| return e,
    };
    return false;
}

pub fn staysInParam(func: *Function, ty: Type) Layout.Error!bool {
    return func.is_kernel and try func.isLarge(ty);
}

pub fn placeVal(func: *Function, place: Place, ty: Type) Layout.Error!Val {
    if (func.layout.isAggregate(ty) or try func.isLarge(ty)) return .{ .place = place };
    return .{ .parts = try func.loadPlace(place, ty) };
}

/// Only `ld.param` can read the `.param` address of a kernel parameter (PTX ISA 9.4, section
/// 5.1.6.1).
pub fn placeAddress(func: *Function, place: Place) !Mir.Reg {
    const base = switch (place.base) {
        .param => |index| try func.builder.emit(.b64, .mov, .u64, .{ .none = .{} }, .{Builder.Base{ .name = try call.paramName(&func.builder, index) }}),
        .address => |address| address.reg,
    };
    if (place.offset == 0) return base;
    return func.builder.emit(base.class, .add, base.class.unsignedType(), .{ .arith = .{ .address = true } }, .{ base, place.offset });
}

pub fn placeAlignment(func: *Function, place: Place) u64 {
    const alignment = switch (place.base) {
        .param => |index| func.paramAlignment(index),
        .address => |address| address.alignment,
    };
    return memory.offsetAlignment(alignment, place.offset);
}

/// `ptr_type` is the type of `base`.
pub fn offsetPtr(func: *Function, base: Val, ptr_type: Type, offset: u32) Layout.Error!Val {
    switch (base) {
        .place => unreachable,
        .parts => |parts| {
            if (offset == 0) return base;
            const ptr = try func.ptrReg(parts[0], ptr_type);
            return func.single(try func.builder.emit(ptr.class, .add, ptr.class.unsignedType(), .{ .arith = .{ .address = true } }, .{ ptr, offset }));
        },
    }
}

pub fn ptrReg(func: *Function, part: Builder.Part, ptr_type: Type) Layout.Error!Mir.Reg {
    return func.builder.toReg(part, func.layout.ptrTypeKind(ptr_type));
}

pub fn fieldPtr(func: *Function, ptr: Air.Inst.Ref, index: u32) Layout.Error!Val {
    const ptr_type = func.typeOf(ptr);
    const aggregate_type = ptr_type.childType(func.zcu);
    return func.offsetPtr(try func.resolve(ptr), ptr_type, try func.layout.fieldOffset(aggregate_type, index));
}

pub fn loadThrough(func: *Function, ptr_ref: Air.Inst.Ref, scalars: []const Layout.Scalar) Layout.Error![]const Builder.Part {
    const ptr_type = func.typeOf(ptr_ref);
    return switch (try func.resolve(ptr_ref)) {
        .place => unreachable,
        .parts => |parts| memory.loadMem(&func.builder, .{ .reg = try func.ptrReg(parts[0], ptr_type) }, ptr_type.ptrAddressSpace(func.zcu), func.layout.ptrAlignment(ptr_type), scalars, ptr_type.isVolatilePtr(func.zcu)),
    };
}

pub fn storeVal(func: *Function, base: Val, ptr_type: Type, offset: u32, src: []const Builder.Part, ty: Type) Layout.Error!void {
    const scalars = try func.layout.scalarsOf(ty);
    switch (try func.offsetPtr(base, ptr_type, offset)) {
        .place => unreachable,
        .parts => |parts| {
            const alignment = memory.offsetAlignment(func.layout.ptrAlignment(ptr_type), offset);
            try memory.storeMem(&func.builder, .{ .reg = try func.ptrReg(parts[0], ptr_type) }, ptr_type.ptrAddressSpace(func.zcu), alignment, scalars, src, false);
        },
    }
}

pub fn isOptimized(func: *Function, inst: Air.Inst.Index) bool {
    return switch (func.air.instructions.items(.tag)[@backingInt(inst)]) {
        .add_optimized, .sub_optimized, .mul_optimized, .div_float_optimized, .div_trunc_optimized, .div_exact_optimized => true,
        else => false,
    };
}

/// `cc` is the calling convention of the callee.
pub fn callArgs(func: *Function, refs: []const Air.Inst.Ref, cc: std.lang.CallingConvention) Layout.Error![]const call.Arg {
    const args = try func.arena.alloc(call.Arg, refs.len);
    for (args, refs) |*arg, ref| {
        const ty = func.typeOf(ref);
        arg.* = if (call.passesByReference(&func.layout, cc, ty))
            .{ .ty = try func.pt.singleConstPtrType(ty), .parts = try func.builder.regParts(&.{try func.argumentAddress(ref, ty)}) }
        else
            .{ .ty = ty, .parts = try func.partsOf(ref) };
    }
    return args;
}

/// Returns the generic address of memory that holds the value of `ref` and that no instruction
/// writes during the call. A value in registers goes to a new local variable.
fn argumentAddress(func: *Function, ref: Air.Inst.Ref, ty: Type) Layout.Error!Mir.Reg {
    const zcu = func.zcu;
    if (func.placeOf(ref)) |place| {
        const address = try func.placeAddress(place);
        const address_space = place.addressSpace();
        if (address_space != .param) return memory.genericAddress(&func.builder, address, address_space, .generic);
        if (func.layout.smVersion() >= 70) {
            // PTX ISA 9.4, section 9.7.10.23: `cvta.param` needs PTX 7.7 and sm_70.
            func.builder.needPtx(77);
            return func.builder.emit(.b64, .cvta, .u64, .{ .cvta = .{ .space = .param } }, .{address});
        }
    }
    const alignment = ty.abiAlignment(zcu).toByteUnits().?;
    const copy = try func.builder.newLocal(ty.abiSize(zcu), alignment, .local, func.layout.ptrKind(.local));
    try memory.storeMem(&func.builder, .{ .reg = copy }, .local, alignment, try func.layout.scalarsOf(ty), try func.partsOf(ref), false);
    return memory.genericAddress(&func.builder, copy, .local, .generic);
}

/// The caller passes a false `strict` in the optimized float mode.
pub fn binary(func: *Function, op: int.BinaryOp, lhs: Mir.Reg, rhs: Mir.Source, kind: Kind, strict: bool) Layout.Error!Mir.Reg {
    if (kind.repr == .float) return float.binary(&func.builder, op, lhs, rhs, kind, strict);
    return int.binary(&func.builder, op, lhs, rhs, kind);
}

pub fn unary(func: *Function, op: int.UnaryOp, src: Mir.Reg, kind: Kind, result: Kind) Layout.Error!Mir.Reg {
    if (kind.repr == .float) return float.unary(&func.builder, op, src, kind);
    return int.unary(&func.builder, op, src, kind, result);
}

pub fn compareSource(func: *Function, op: CompareOperator, lhs: Mir.Reg, rhs: Mir.Source, kind: Kind) Layout.Error!Mir.Reg {
    if (kind.repr == .float) return float.compare(&func.builder, op, lhs, rhs, kind);
    return int.compareSource(&func.builder, op, lhs, rhs, kind);
}
