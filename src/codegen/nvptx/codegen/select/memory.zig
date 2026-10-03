//! This file lowers the AIR instructions that address, load and store memory, including the
//! atomic operations.

const std = @import("std");
const assert = std.debug.assert;

const Air = @import("../../../../Air.zig");
const Value = @import("../../../../Value.zig");
const Type = @import("../../../../Type.zig");
const Mir = @import("../../Mir.zig");
const Layout = @import("../Layout.zig");
const Builder = @import("../Builder.zig");
const Function = @import("../Function.zig");
const int = @import("../int.zig");
const memory = @import("../memory.zig");
const atomic = @import("../atomic.zig");

const Val = Function.Val;

pub fn airLoad(func: *Function, inst: Air.Inst.Index) Layout.Error!Val {
    const ty_op = func.dataOf(inst).ty_op;
    if (try func.bitPackedElem(ty_op.operand)) |elem| {
        const vector_type, const index = elem;
        const count = try func.layout.scalarCount(ty_op.ty);
        return .{ .parts = (try func.loadVector(ty_op.operand, vector_type))[index * count ..][0..count] };
    }
    return func.loadPtr(ty_op.operand, ty_op.ty, inst);
}

pub fn airStore(func: *Function, inst: Air.Inst.Index, safety: bool) Layout.Error!void {
    const bin_op = func.dataOf(inst).bin_op;
    if (bin_op.lhs.toIndex()) |alloc| if (func.isParamSpaceCopy(alloc)) return;
    if (try func.bitPackedElem(bin_op.lhs)) |elem| {
        const vector_type, const index = elem;
        const elems = try func.arena.dupe(Builder.Part, try func.loadVector(bin_op.lhs, vector_type));
        const value = try func.safeParts(bin_op.rhs, safety);
        @memcpy(elems[index * value.len ..][0..value.len], value);
        return func.store(bin_op.lhs, elems, vector_type);
    }
    if (func.placeOf(bin_op.rhs)) |place| return func.storePlace(bin_op.lhs, place, func.typeOf(bin_op.rhs));
    try func.store(bin_op.lhs, try func.safeParts(bin_op.rhs, safety), func.typeOf(bin_op.rhs));
}

pub fn airPtrAdd(func: *Function, inst: Air.Inst.Index, part: enum { add, sub, elem }) Layout.Error!Val {
    const zcu = func.zcu;
    const bin = func.binFromPayload(inst);
    const elem_size = func.typeOfIndex(inst).childType(zcu).abiSize(zcu);
    const ptr_type = func.typeOf(bin.lhs);
    const base = try func.resolve(bin.lhs);
    if (part == .elem and try func.bitPackedElem(inst.toRef()) != null) return base;
    if (part != .sub) if (bin.rhs.toInterned()) |ip_index| {
        const index = Value.fromInterned(ip_index).toUnsignedInt(zcu);
        const offset = std.math.mul(u64, index, elem_size) catch null;
        if (std.math.cast(u32, offset orelse std.math.maxInt(u64))) |small| return func.offsetPtr(base, ptr_type, small);
    };
    const ptr = switch (base) {
        .place => unreachable,
        .parts => |parts| try func.ptrReg(parts[0], ptr_type),
    };
    assert(!ptr_type.isSlice(zcu));
    var index = try func.scalarReg(bin.rhs);
    if (part == .sub) index = try func.builder.emit(.b64, .neg, .s64, .{ .arith = .{} }, .{index});
    return func.single(try memory.scaledAddress(&func.builder, ptr, index, elem_size));
}

pub fn airSliceElem(func: *Function, inst: Air.Inst.Index, load: bool) Layout.Error!Val {
    const zcu = func.zcu;
    const bin: Air.Bin = if (load) .{ .lhs = func.dataOf(inst).bin_op.lhs, .rhs = func.dataOf(inst).bin_op.rhs } else func.binFromPayload(inst);
    const slice_type = func.typeOf(bin.lhs);
    const elem_type = slice_type.childType(zcu);
    const ptr = try func.ptrReg((try func.partsOf(bin.lhs))[0], slice_type);
    const address = try memory.scaledAddress(&func.builder, ptr, try func.scalarReg(bin.rhs), elem_type.abiSize(zcu));
    if (!load) return func.single(address);
    const alignment = memory.offsetAlignment(func.layout.ptrAlignment(slice_type), elem_type.abiSize(zcu));
    return .{ .parts = try memory.loadMem(&func.builder, .{ .reg = address }, slice_type.ptrAddressSpace(zcu), alignment, try func.layout.scalarsOf(elem_type), slice_type.isVolatilePtr(zcu)) };
}

pub fn airMemset(func: *Function, inst: Air.Inst.Index, safety: bool) Layout.Error!?Val {
    const zcu = func.zcu;
    const bin_op = func.dataOf(inst).bin_op;
    const dest_type = func.typeOf(bin_op.lhs);
    const elem_type = dest_type.indexableElem(zcu);
    try func.checkWritable(dest_type.ptrAddressSpace(zcu));
    const value = try func.safeParts(bin_op.rhs, safety);
    const dest_volatile = dest_type.isVolatilePtr(zcu);
    const dest = try func.partsOf(bin_op.lhs);
    const ptr = try func.ptrReg(dest[0], dest_type);
    const scalars = try func.layout.scalarsOf(elem_type);
    const elem_size = elem_type.abiSize(zcu);
    if (try setValue(func, scalars, elem_size, func.layout.ptrAlignment(dest_type), value)) |set| {
        if (set == .undef) return null;
        if (dest_type.isSlice(zcu)) {
            try emitCopy(func, .set, ptr, dest_type, set.reg, null, elem_size, try func.builder.toReg(dest[1], .pointer));
        } else {
            try emitCopy(func, .set, ptr, dest_type, set.reg, null, elem_size, dest_type.childType(zcu).arrayLen(zcu));
        }
        return null;
    }
    if (dest_type.isSlice(zcu)) {
        try memory.copyLoop(&func.builder, ptr, try func.builder.toReg(dest[1], .pointer), null, dest_type.ptrAddressSpace(zcu), .generic, func.layout.ptrAlignment(dest_type), 1, scalars, elem_size, value, false, dest_volatile, false);
        return null;
    }
    const len = dest_type.childType(zcu).arrayLen(zcu);
    if (len * scalars.len <= Layout.max_unrolled_scalars) {
        for (0..len) |i| {
            const element = try func.layout.offsetScalars(scalars, @intCast(i * elem_size));
            try memory.storeMem(&func.builder, .{ .reg = ptr }, dest_type.ptrAddressSpace(zcu), func.layout.ptrAlignment(dest_type), element, value, dest_volatile);
        }
        return null;
    }
    const count = try func.builder.movImm(.b64, len);
    try memory.copyLoop(&func.builder, ptr, count, null, dest_type.ptrAddressSpace(zcu), .generic, func.layout.ptrAlignment(dest_type), 1, scalars, elem_size, value, false, dest_volatile, false);
    return null;
}

const SetValue = union(enum) { reg: Mir.Reg, undef };

/// Returns the scalar that a `set` stores into each element. Returns null when one aligned store
/// cannot write the whole element. The Zig backend for LLVM also lowers such a `@memset` with a
/// loop of its own. A bool takes one byte in memory, so the `set` stores 0 or 1.
fn setValue(func: *Function, scalars: []const Layout.Scalar, elem_size: u64, alignment: u64, value: []const Builder.Part) Layout.Error!?SetValue {
    if (scalars.len != 1 or elem_size > 8 or alignment < elem_size) return null;
    const scalar = scalars[0];
    if (scalar.bit != null or scalar.offset != 0 or scalar.kind.size != elem_size) return null;
    if (value[0] == .undef) return .undef;
    const reg = try func.builder.toReg(value[0], scalar.kind);
    if (scalar.kind.class != .pred) return .{ .reg = reg };
    return .{ .reg = try func.builder.emit(.b16, .selp, .u16, .{ .none = .{} }, .{ 1, 0, reg }) };
}

/// Returns the largest unit of at most 16 bytes that divides `size` and `max_unit`.
fn copyUnit(size: u64, max_unit: u64) u64 {
    var unit: u64 = 16;
    while (unit > max_unit or size % unit != 0) unit /= 2;
    return unit;
}

/// Emits the memory intrinsic `copy`. `count` counts units of `unit` bytes, and `value` is the
/// source address, or the stored value of a `set`.
fn emitCopy(func: *Function, op: Mir.CopyOp, dest: Mir.Reg, dest_type: Type, value: Mir.Reg, src_type: ?Type, unit: u64, count: anytype) Layout.Error!void {
    const zcu = func.zcu;
    const builder = &func.builder;
    const dest_space = dest_type.ptrAddressSpace(zcu);
    const dest_mem = memory.memOperand(builder, dest_space, memory.volatileSem(dest_space, dest_type.isVolatilePtr(zcu)), unit, unit, false);
    const modifiers: Mir.Inst.Modifiers = .{ .copy = .{ .op = op } };
    const ty = src_type orelse return builder.emitEffect(.copy, .none, modifiers, .{ dest, value, count, dest_mem });
    const src_space = ty.ptrAddressSpace(zcu);
    const src_mem = memory.memOperand(builder, src_space, memory.volatileSem(src_space, ty.isVolatilePtr(zcu)), unit, unit, true);
    try builder.emitEffect(.copy, .none, modifiers, .{ dest, value, count, dest_mem, src_mem });
}

/// Returns the number of units of `unit` bytes in the elements of the slice or array `dest`.
fn unitCount(func: *Function, dest: []const Builder.Part, dest_type: Type, elem_size: u64, unit: u64) Layout.Error!union(enum) { imm: u64, reg: Mir.Reg } {
    const zcu = func.zcu;
    const per_elem = elem_size / unit;
    if (!dest_type.isSlice(zcu)) return .{ .imm = dest_type.childType(zcu).arrayLen(zcu) * per_elem };
    const len = try func.builder.toReg(dest[1], .pointer);
    if (per_elem == 1) return .{ .reg = len };
    return .{ .reg = try func.builder.emit(.b64, .mul, .u64, .{ .arith = .{ .mode = .lo } }, .{ len, per_elem }) };
}

pub fn airMemcpy(func: *Function, inst: Air.Inst.Index) Layout.Error!?Val {
    const zcu = func.zcu;
    const bin_op = func.dataOf(inst).bin_op;
    const dest_type = func.typeOf(bin_op.lhs);
    const src_type = func.typeOf(bin_op.rhs);
    const dest = try func.partsOf(bin_op.lhs);
    const src = try func.partsOf(bin_op.rhs);
    if (!dest_type.isSlice(zcu) and !dest_type.isVolatilePtr(zcu) and !src_type.isVolatilePtr(zcu)) {
        const src_place: Function.Place = .{ .base = .{ .address = .{
            .reg = try func.ptrReg(src[0], src_type),
            .address_space = src_type.ptrAddressSpace(zcu),
            .alignment = func.layout.ptrAlignment(src_type),
        } }, .offset = 0 };
        try func.copyMemory(try func.ptrReg(dest[0], dest_type), dest_type.ptrAddressSpace(zcu), func.layout.ptrAlignment(dest_type), src_place, dest_type.childType(zcu));
        return null;
    }
    try func.checkWritable(dest_type.ptrAddressSpace(zcu));
    const elem_type = dest_type.indexableElem(zcu);
    const elem_size = elem_type.abiSize(zcu);
    assert(elem_size != 0);
    const max_unit = @min(func.layout.ptrAlignment(dest_type), func.layout.ptrAlignment(src_type), elem_type.abiAlignment(zcu).toByteUnits().?);
    const unit = copyUnit(elem_size, max_unit);
    const dest_ptr = try func.ptrReg(dest[0], dest_type);
    const src_ptr = try func.ptrReg(src[0], src_type);
    switch (try unitCount(func, dest, dest_type, elem_size, unit)) {
        inline else => |count| try emitCopy(func, .copy, dest_ptr, dest_type, src_ptr, src_type, unit, count),
    }
    return null;
}

/// `ld` and `st` need aligned addresses (PTX ISA 9.4, section 6.4.1), so the unit size follows the
/// alignments of the pointers.
pub fn airMemmove(func: *Function, inst: Air.Inst.Index) Layout.Error!?Val {
    const zcu = func.zcu;
    const bin_op = func.dataOf(inst).bin_op;
    const dest_type = func.typeOf(bin_op.lhs);
    const src_type = func.typeOf(bin_op.rhs);
    const elem_size = dest_type.indexableElem(zcu).abiSize(zcu);
    assert(elem_size != 0);
    const dest = try func.partsOf(bin_op.lhs);
    const src = try func.partsOf(bin_op.rhs);
    try func.checkWritable(dest_type.ptrAddressSpace(zcu));
    const unit = copyUnit(elem_size, @min(func.layout.ptrAlignment(dest_type), func.layout.ptrAlignment(src_type)));
    const dest_ptr = try func.ptrReg(dest[0], dest_type);
    const src_ptr = try func.ptrReg(src[0], src_type);
    switch (try unitCount(func, dest, dest_type, elem_size, unit)) {
        inline else => |count| try emitCopy(func, .move, dest_ptr, dest_type, src_ptr, src_type, unit, count),
    }
    return null;
}

pub fn airNavPtr(func: *Function, inst: Air.Inst.Index) Layout.Error!Val {
    const ty_nav = func.air.instructions.items(.data)[@backingInt(inst)].ty_nav;
    const nav_address_space = func.zcu.intern_pool.getNav(ty_nav.nav).resolved.?.@"addrspace";
    return .{ .parts = try func.arena.dupe(Builder.Part, &.{.{ .symbol = .{
        .target = .{ .nav = ty_nav.nav },
        .offset = 0,
        .space = Layout.Space.fromAddressSpace(nav_address_space).?,
        .address_space = ty_nav.ty.ptrAddressSpace(func.zcu),
    } }}) };
}

pub fn airAddrSpaceCast(func: *Function, inst: Air.Inst.Index) Layout.Error!Val {
    const zcu = func.zcu;
    const ty_op = func.dataOf(inst).ty_op;
    const operand_type = func.typeOf(ty_op.operand);
    const from = operand_type.ptrAddressSpace(zcu);
    const to = ty_op.ty.ptrAddressSpace(zcu);
    const src = try func.partsOf(ty_op.operand);
    if (from == to) return .{ .parts = src };
    const parts = try func.arena.dupe(Builder.Part, src);
    if (src[0] == .symbol) {
        parts[0].symbol.address_space = to;
        return .{ .parts = parts };
    }
    const reg = try func.ptrReg(src[0], operand_type);
    if (from == .param or to == .param) {
        // PTX ISA 9.4, section 9.7.10.23: `cvta.param` needs PTX 7.7 and sm_70.
        if (func.layout.smVersion() < 70) return func.layout.fail("a cast from or to address space 'param' needs sm_70 or later", .{});
        func.builder.needPtx(77);
    }
    const cast = if (to == .generic)
        try func.builder.emit(.b64, .cvta, .u64, .{ .cvta = .{ .space = memory.memSpace(from), .uncommitted = Builder.isInferable(memory.memSpace(from)) } }, .{try int.widen(&func.builder, reg, .pointer32)})
    else if (from == .generic)
        try int.intConvert(&func.builder, try func.builder.emit(.b64, .cvta, .u64, .{ .cvta = .{ .space = memory.memSpace(to), .to = true } }, .{reg}), .pointer, func.layout.ptrTypeKind(ty_op.ty))
    else
        return func.layout.fail("cannot cast a pointer from '{t}' to '{t}'", .{ from, to });
    const from_null = operand_type.nullPtrAddress(zcu);
    const to_null = ty_op.ty.nullPtrAddress(zcu);
    if (operand_type.zigTypeTag(zcu) != .optional or (from_null == 0 and to_null == 0)) {
        parts[0] = .{ .reg = cast };
        return .{ .parts = parts };
    }
    const is_null = try func.builder.emit(.pred, .setp, reg.class.unsignedType(), .{ .setp = .{ .cmp = .eq } }, .{ reg, from_null });
    parts[0] = .{ .reg = try func.builder.emit(cast.class, .selp, cast.class.unsignedType(), .{ .none = .{} }, .{ to_null, cast, is_null }) };
    return .{ .parts = parts };
}

pub fn airFieldParentPtr(func: *Function, inst: Air.Inst.Index) Layout.Error!Val {
    const zcu = func.zcu;
    const ty_pl = func.dataOf(inst).ty_pl;
    const extra = func.air.extraData(Air.FieldParentPtr, ty_pl.payload).data;
    const parent_type = ty_pl.ty.childType(zcu);
    if (parent_type.containerLayout(zcu) == .@"packed") return func.layout.todo("@fieldParentPtr of the packed type '{f}'", .{parent_type.fmt(zcu)});
    const offset = try func.layout.fieldOffset(parent_type, extra.field_index);
    switch (try func.resolve(extra.field_ptr)) {
        .place => unreachable,
        .parts => |parts| {
            if (offset == 0) return .{ .parts = parts };
            const ptr = try func.ptrReg(parts[0], func.typeOf(extra.field_ptr));
            return func.single(try func.builder.emit(ptr.class, .sub, ptr.class.unsignedType(), .{ .arith = .{ .address = true } }, .{ ptr, offset }));
        },
    }
}

/// `AtomicTarget` names the memory of an atomic access. `atom` cannot address `.local` memory
/// (PTX ISA 9.4, section 9.7.15.5). No other thread can access `.local` memory, so plain loads and
/// stores replace `atom` there. Each alloc is in `.local` memory.
const AtomicTarget = union(enum) {
    local: struct { ptr: Mir.Reg, address_space: std.lang.AddressSpace },
    memory: struct { ptr: Mir.Reg, space: Mir.Space },
};

fn atomicTarget(func: *Function, ptr_ref: Air.Inst.Ref) Layout.Error!AtomicTarget {
    const ptr = try func.ptrReg((try func.partsOf(ptr_ref))[0], func.typeOf(ptr_ref));
    const address_space = func.typeOf(ptr_ref).ptrAddressSpace(func.zcu);
    if (address_space == .local or func.allocRoot(ptr_ref)) return .{ .local = .{ .ptr = ptr, .address_space = address_space } };
    return switch (address_space) {
        .constant, .param => func.layout.fail("atomic access to address space '{t}'", .{address_space}),
        else => .{ .memory = .{ .ptr = ptr, .space = memory.memSpace(address_space) } },
    };
}

fn plainLoad(func: *Function, target: AtomicTarget, scalar: Layout.Scalar) Layout.Error!Mir.Reg {
    const local = target.local;
    const part = (try memory.loadMem(&func.builder, .{ .reg = local.ptr }, local.address_space, scalar.kind.size, &.{scalar}, false))[0];
    return func.builder.toReg(part, scalar.kind);
}

fn plainStore(func: *Function, target: AtomicTarget, scalar: Layout.Scalar, value: Mir.Reg) Layout.Error!void {
    const local = target.local;
    try memory.storeMem(&func.builder, .{ .reg = local.ptr }, local.address_space, scalar.kind.size, &.{scalar}, &.{.{ .reg = value }}, false);
}

pub fn airAtomicLoad(func: *Function, inst: Air.Inst.Index) Layout.Error!Val {
    const atomic_load = func.dataOf(inst).atomic_load;
    const scalars = try func.layout.scalarsOf(func.typeOfIndex(inst));
    assert(scalars.len != 0);
    return func.single(switch (try atomicTarget(func, atomic_load.ptr)) {
        .memory => |target| try atomic.load(&func.builder, &func.layout, target.ptr, target.space, atomic_load.order, scalars[0].kind),
        else => |target| try plainLoad(func, target, scalars[0]),
    });
}

pub fn airAtomicStore(func: *Function, inst: Air.Inst.Index, order: std.lang.AtomicOrder) Layout.Error!?Val {
    const bin_op = func.dataOf(inst).bin_op;
    const scalars = try func.layout.scalarsOf(func.typeOf(bin_op.rhs));
    assert(scalars.len != 0);
    const src = (try func.partsOf(bin_op.rhs))[0];
    if (src == .undef) return null;
    const value = try func.builder.toReg(src, scalars[0].kind);
    switch (try atomicTarget(func, bin_op.lhs)) {
        .memory => |target| try atomic.store(&func.builder, &func.layout, target.ptr, target.space, order, scalars[0].kind, value),
        else => |target| try plainStore(func, target, scalars[0], value),
    }
    return null;
}

pub fn airAtomicRmw(func: *Function, inst: Air.Inst.Index) Layout.Error!Val {
    const pl_op = func.dataOf(inst).pl_op;
    const extra = func.air.extraData(Air.AtomicRmw, pl_op.payload).data;
    const scalars = try func.layout.scalarsOf(func.typeOfIndex(inst));
    assert(scalars.len != 0);
    const scalar = scalars[0];
    const operand = try func.builder.toReg((try func.partsOf(extra.operand))[0], scalar.kind);
    switch (try atomicTarget(func, pl_op.operand)) {
        .memory => |target| return func.single(try atomic.rmw(&func.builder, &func.layout, target.ptr, target.space, extra.ordering(), extra.op(), scalar.kind, operand)),
        else => |target| {
            const old = try plainLoad(func, target, scalar);
            try plainStore(func, target, scalar, try atomic.rmwValue(&func.builder, extra.op(), old, operand, scalar.kind));
            return func.single(old);
        },
    }
}

pub fn airCmpxchg(func: *Function, inst: Air.Inst.Index) Layout.Error!Val {
    const zcu = func.zcu;
    const extra = func.air.extraData(Air.Cmpxchg, func.dataOf(inst).ty_pl.payload).data;
    const optional_type = func.typeOfIndex(inst);
    const scalar = (try func.layout.scalarsOf(optional_type.optionalChild(zcu)))[0];
    const kind = scalar.kind;
    const expected = try func.builder.toReg((try func.partsOf(extra.expected_value))[0], kind);
    const new = try func.builder.toReg((try func.partsOf(extra.new_value))[0], kind);
    const result: atomic.CompareExchange = switch (try atomicTarget(func, extra.ptr)) {
        // `atom` takes one `.sem`. Zig allows an acquire failure order with a release success
        // order. The qualifier therefore merges both orders.
        .memory => |target| try atomic.compareExchange(&func.builder, &func.layout, target.ptr, target.space, atomic.Sync.init(extra.successOrder()).merge(.init(extra.failureOrder())), kind, expected, new),
        else => |target| local: {
            const old = try plainLoad(func, target, scalar);
            const equal = try atomic.bitEqual(&func.builder, old, expected, kind);
            try plainStore(func, target, scalar, try int.select(&func.builder, equal, new, old, kind));
            break :local .{ .old = old, .equal = equal };
        },
    };
    // `cmpxchg` returns null on success. An optional that stores null as zero is zero on success.
    if (optional_type.optionalReprIsPayload(zcu)) {
        const zero = try func.builder.movImm(kind.class, 0);
        return func.single(try int.select(&func.builder, result.equal, zero, result.old, kind));
    }
    const has_value = try func.builder.emit(.pred, .not, .pred, .{ .none = .{} }, .{result.equal});
    return .{ .parts = try func.builder.regParts(&.{ result.old, has_value }) };
}
