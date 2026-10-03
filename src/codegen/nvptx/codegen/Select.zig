//! This file selects the instructions of one function. It walks the AIR of the function and lowers
//! the control flow. The files in `select/` lower the other AIR instructions.

const std = @import("std");
const assert = std.debug.assert;

const Air = @import("../../../Air.zig");
const Type = @import("../../../Type.zig");
const codegen = @import("../../../codegen.zig");
const Mir = @import("../Mir.zig");
const verify = @import("../verify.zig");
const Layout = @import("Layout.zig");
const Builder = @import("Builder.zig");
const Function = @import("Function.zig");
const wide = @import("wide.zig");
const memory = @import("memory.zig");
const call = @import("call.zig");
const Regions = @import("../regions.zig").Regions;
const select = struct {
    const aggregate = @import("select/aggregate.zig");
    const arithmetic = @import("select/arithmetic.zig");
    const call = @import("select/call.zig");
    const memory = @import("select/memory.zig");
};
const Val = Function.Val;
const Place = Function.Place;
const Select = @This();

func: Function,
regions: Regions,
is_noinline: bool,
next_arg: u32 = 0,
exits: std.AutoHashMapUnmanaged(Air.Inst.Index, Exit) = .empty,
loops: std.AutoHashMapUnmanaged(Air.Inst.Index, Loop) = .empty,

/// A `br` to an AIR block continues at `target`. The parameters of `target` take the result.
const Exit = struct {
    target: Builder.Block,
    /// After this last instruction of the body, control reaches the target without a branch.
    last: ?Air.Inst.Index,
    /// At most one `br` leaves the block. The target then has no parameters, and the code after
    /// the block uses the operands of that `br`.
    single: bool,
    result_len: u32 = 0,
    result: []const Builder.Part = &.{},
};

/// The parameters of `head` hold the `loop_switch_br` operand.
const Loop = struct {
    head: Builder.Block,
};

pub fn genFunction(isel: *Select) Layout.Error!Mir {
    const func = &isel.func;
    const zcu = func.zcu;
    const ip = &zcu.intern_pool;
    const nav = ip.getNav(func.layout.owner_nav);
    const fn_info = zcu.typeToFunc(.fromInterned(nav.resolved.?.type)).?;
    const is_kernel = fn_info.cc == .nvptx_kernel;
    func.is_kernel = is_kernel;
    func.builder.is_kernel = is_kernel;
    func.cc = fn_info.cc;
    func.ret_type = .fromInterned(fn_info.return_type);
    try isel.findParamAllocs();

    var prototype: std.ArrayList(u8) = .empty;
    const name_offset = try call.declare(&func.builder, &func.layout, fn_info, "", &prototype);

    try isel.genBody(func.air.getMainBody());
    assert(func.builder.trampolines.items.len == 0);

    var prefix: std.ArrayList(u8) = .empty;
    try prefix.appendSlice(func.arena, prototype.items);
    if (is_kernel) try call.kernelDirectives(&func.builder, &func.layout, fn_info.cc.nvptx_kernel, &prefix);
    var mir = try func.builder.finish(func.gpa, if (is_kernel) .entry else .func, @intCast(prototype.items.len), prefix.items, .{
        .offset = name_offset,
        .target = .{ .nav = func.layout.owner_nav },
    });
    errdefer mir.deinit(func.gpa);
    mir.is_noinline = isel.is_noinline;
    if (std.debug.runtime_safety) {
        if (try verify.check(func.arena, &mir)) |message| return func.layout.fail("internal error: invalid SSA form: {s}", .{message});
    }
    return mir;
}

/// When the code takes the address of a parameter, Sema copies the parameter into an `alloc`.
/// Sema then casts the pointer of the copy to a const pointer. Zig parameters are immutable, so
/// promotion can use the `.param` address of a kernel parameter in place of the copy.
fn findParamAllocs(isel: *Select) Layout.Error!void {
    const func = &isel.func;
    const tags = func.air.instructions.items(.tag);
    const body = func.air.getMainBody();
    var args: std.AutoHashMapUnmanaged(Air.Inst.Index, u32) = .empty;
    for (body[0 .. body.len - 1], body[1..]) |inst, next| switch (tags[@backingInt(inst)]) {
        .arg => try args.put(func.arena, inst, args.count()),
        .store, .store_safe => {
            const bin_op = func.dataOf(inst).bin_op;
            const alloc = bin_op.lhs.toIndex() orelse continue;
            const arg = bin_op.rhs.toIndex() orelse continue;
            if (tags[@backingInt(alloc)] != .alloc) continue;
            if (tags[@backingInt(next)] != .ptr_cast or func.dataOf(next).ty_op.operand != bin_op.lhs) continue;
            if (!func.typeOfIndex(next).isConstPtr(func.zcu) or !func.liveness.operandDies(next, 0)) continue;
            try func.param_allocs.put(func.arena, alloc, args.get(arg) orelse continue);
        },
        else => {},
    };
}

/// The backend loads a parameter at its `arg` instruction, which dominates every use. A kernel
/// parameter stays in `.param` memory when `Function.staysInParam` holds for its type. A parameter
/// that `call.passesByReference` passes stays at the address that the caller passes.
fn airArg(isel: *Select, inst: Air.Inst.Index) Layout.Error!Val {
    const func = &isel.func;
    defer isel.next_arg += 1;
    const ty = func.typeOfIndex(inst);
    const place: Place = .{ .base = .{ .param = isel.next_arg }, .offset = 0 };
    if (try func.staysInParam(ty)) return .{ .place = place };
    if (call.passesByReference(&func.layout, func.cc, ty)) {
        const address = (try func.loadPlace(place, try func.pt.singleConstPtrType(ty)))[0];
        return .{ .place = .{ .base = .{ .address = .{
            .reg = try func.builder.toReg(address, .pointer),
            .address_space = .generic,
            .alignment = ty.abiAlignment(func.zcu).toByteUnits().?,
        } }, .offset = 0 } };
    }
    return .{ .parts = try func.loadPlace(place, ty) };
}

fn airAlloc(isel: *Select, inst: Air.Inst.Index) Layout.Error!Val {
    const func = &isel.func;
    const zcu = func.zcu;
    const ptr_type = func.typeOfIndex(inst);
    const child_type = ptr_type.childType(zcu);
    if (func.isParamSpaceCopy(inst)) {
        const name = try call.paramName(&func.builder, func.param_allocs.get(inst).?);
        return func.single(try func.builder.emit(.b64, .mov, .u64, .{ .none = .{} }, .{Builder.Base{ .name = name }}));
    }
    const local = try func.builder.newLocal(child_type.abiSize(zcu), ptr_type.ptrAlignment(zcu).toByteUnits().?, ptr_type.ptrAddressSpace(zcu), func.layout.ptrTypeKind(ptr_type));
    // PTX ISA 9.4, section 9.7.10.23: `cvta.param` needs PTX 7.7 and sm_70.
    if (func.param_allocs.get(inst)) |param| if (func.is_kernel and func.layout.smVersion() >= 70) {
        try func.param_copies.append(func.arena, .{ .local = @intCast(func.builder.locals.items.len - 1), .param = param });
    };
    return func.single(local);
}

fn genBody(isel: *Select, body: []const Air.Inst.Index) Layout.Error!void {
    const func = &isel.func;
    const ip = &func.zcu.intern_pool;
    for (body) |inst| {
        if (func.liveness.isUnused(inst) and !func.air.mustLower(inst, ip)) continue;
        if (try isel.genInst(inst)) |val| try func.values.put(func.arena, inst, val);
    }
}

fn genInst(isel: *Select, inst: Air.Inst.Index) Layout.Error!?Val {
    const func = &isel.func;
    const tag = func.air.instructions.items(.tag)[@backingInt(inst)];
    return switch (tag) {
        .arg => try isel.airArg(inst),
        .alloc, .ret_ptr => try isel.airAlloc(inst),
        .load => try select.memory.airLoad(func, inst),
        .store, .store_safe => lowered: {
            try select.memory.airStore(func, inst, tag == .store_safe);
            break :lowered null;
        },

        .add, .add_optimized, .add_wrap => try select.arithmetic.airBinaryOp(func, inst, .add),
        .sub, .sub_optimized, .sub_wrap => try select.arithmetic.airBinaryOp(func, inst, .sub),
        .mul, .mul_optimized, .mul_wrap => try select.arithmetic.airBinaryOp(func, inst, .mul),
        .div_float, .div_float_optimized, .div_exact, .div_exact_optimized => try select.arithmetic.airBinaryOp(func, inst, .div),
        .div_trunc, .div_trunc_optimized => try select.arithmetic.airBinaryOp(func, inst, .div_trunc),
        .div_floor, .div_floor_optimized => try select.arithmetic.airBinaryOp(func, inst, .div_floor),
        .rem, .rem_optimized => try select.arithmetic.airBinaryOp(func, inst, .rem),
        .mod, .mod_optimized => try select.arithmetic.airBinaryOp(func, inst, .mod),
        .min => try select.arithmetic.airBinaryOp(func, inst, .min),
        .max => try select.arithmetic.airBinaryOp(func, inst, .max),
        .bit_and => try select.arithmetic.airBinaryOp(func, inst, .@"and"),
        .bit_or => try select.arithmetic.airBinaryOp(func, inst, .@"or"),
        .xor => try select.arithmetic.airBinaryOp(func, inst, .xor),
        .shl, .shl_exact => try select.arithmetic.airBinaryOp(func, inst, .shl),
        .shr, .shr_exact => try select.arithmetic.airBinaryOp(func, inst, .shr),
        .add_sat => try select.arithmetic.airBinaryOp(func, inst, .add_sat),
        .sub_sat => try select.arithmetic.airBinaryOp(func, inst, .sub_sat),
        .mul_sat => try select.arithmetic.airBinaryOp(func, inst, .mul_sat),
        .shl_sat => try select.arithmetic.airBinaryOp(func, inst, .shl_sat),
        .add_with_overflow => try select.arithmetic.airOverflow(func, inst, .add),
        .sub_with_overflow => try select.arithmetic.airOverflow(func, inst, .sub),
        .mul_with_overflow => try select.arithmetic.airOverflow(func, inst, .mul),
        .shl_with_overflow => try select.arithmetic.airOverflow(func, inst, .shl),

        .not => try select.arithmetic.airUnaryOp(func, inst, .not),
        .neg, .neg_optimized => try select.arithmetic.airUnaryOp(func, inst, .neg),
        .abs => try select.arithmetic.airUnaryOp(func, inst, .abs),
        .sqrt => try select.arithmetic.airUnaryOp(func, inst, .sqrt),
        .floor => try select.arithmetic.airUnaryOp(func, inst, .floor),
        .ceil => try select.arithmetic.airUnaryOp(func, inst, .ceil),
        .trunc_float => try select.arithmetic.airUnaryOp(func, inst, .trunc),
        .round => try select.arithmetic.airUnaryOp(func, inst, .round),
        .byte_swap => try select.arithmetic.airUnaryOp(func, inst, .byte_swap),
        .popcount => try select.arithmetic.airUnaryOp(func, inst, .popcount),
        .clz => try select.arithmetic.airUnaryOp(func, inst, .clz),
        .ctz => try select.arithmetic.airUnaryOp(func, inst, .ctz),
        .bit_reverse => try select.arithmetic.airUnaryOp(func, inst, .bit_reverse),
        .mul_add => try select.arithmetic.airMulAdd(func, inst),
        .sin, .cos, .tan, .exp, .exp2, .log, .log2, .log10 => try select.arithmetic.airMathCall(func, inst, tag),

        .cmp_lt, .cmp_lt_optimized => try select.arithmetic.airCompare(func, inst, .lt),
        .cmp_lte, .cmp_lte_optimized => try select.arithmetic.airCompare(func, inst, .lte),
        .cmp_eq, .cmp_eq_optimized => try select.arithmetic.airCompare(func, inst, .eq),
        .cmp_gte, .cmp_gte_optimized => try select.arithmetic.airCompare(func, inst, .gte),
        .cmp_gt, .cmp_gt_optimized => try select.arithmetic.airCompare(func, inst, .gt),
        .cmp_neq, .cmp_neq_optimized => try select.arithmetic.airCompare(func, inst, .neq),
        .cmp_vector, .cmp_vector_optimized => try select.arithmetic.airCompareVector(func, inst),
        .select => try select.arithmetic.airSelect(func, inst),

        .bit_cast, .ptr_cast, .error_cast, .int_from_ptr, .ptr_from_int, .int_from_error, .error_from_int => try select.aggregate.airBitCast(func, inst),
        .int_cast, .trunc => try select.arithmetic.airIntCast(func, inst),
        .fpext, .fptrunc => try select.arithmetic.airConvert(func, inst, .float),
        .float_from_int => try select.arithmetic.airConvert(func, inst, .from_int),
        .int_from_float, .int_from_float_optimized => try select.arithmetic.airConvert(func, inst, .to_int),
        .addrspace_cast => try select.memory.airAddrSpaceCast(func, inst),

        .ptr_add => try select.memory.airPtrAdd(func, inst, .add),
        .ptr_sub => try select.memory.airPtrAdd(func, inst, .sub),
        .ptr_elem_ptr => try select.memory.airPtrAdd(func, inst, .elem),
        .slice_elem_ptr => try select.memory.airSliceElem(func, inst, false),
        .slice_elem_val => try select.memory.airSliceElem(func, inst, true),
        .struct_field_ptr => lowered: {
            const extra = func.air.extraData(Air.StructField, func.dataOf(inst).ty_pl.payload).data;
            break :lowered try func.fieldPtr(extra.struct_operand, extra.field_index);
        },
        .struct_field_ptr_index_0 => try func.fieldPtr(func.dataOf(inst).ty_op.operand, 0),
        .struct_field_ptr_index_1 => try func.fieldPtr(func.dataOf(inst).ty_op.operand, 1),
        .struct_field_ptr_index_2 => try func.fieldPtr(func.dataOf(inst).ty_op.operand, 2),
        .struct_field_ptr_index_3 => try func.fieldPtr(func.dataOf(inst).ty_op.operand, 3),
        .ptr_slice_ptr_ptr => try func.fieldPtr(func.dataOf(inst).ty_op.operand, 0),
        .ptr_slice_len_ptr => try func.fieldPtr(func.dataOf(inst).ty_op.operand, 1),
        .agg_field_val => lowered: {
            const extra = func.air.extraData(Air.StructField, func.dataOf(inst).ty_pl.payload).data;
            break :lowered try select.aggregate.fieldVal(func, extra.struct_operand, extra.field_index);
        },
        .slice_ptr => try select.aggregate.fieldVal(func, func.dataOf(inst).ty_op.operand, 0),
        .slice_len => try select.aggregate.fieldVal(func, func.dataOf(inst).ty_op.operand, 1),
        .array_elem_val => try select.aggregate.airArrayElemVal(func, inst),
        .legalize_vec_elem_val => try select.aggregate.airVecElemVal(func, inst),
        .legalize_vec_store_elem => lowered: {
            try select.aggregate.airVecStoreElem(func, inst);
            break :lowered null;
        },
        .slice => try select.aggregate.airConcat(func, &.{ func.binFromPayload(inst).lhs, func.binFromPayload(inst).rhs }),
        .aggregate_init => try select.aggregate.airAggregateInit(func, inst),
        .splat => try select.aggregate.airSplat(func, inst),
        .shuffle_one => try select.aggregate.airShuffleOne(func, inst),
        .shuffle_two => try select.aggregate.airShuffleTwo(func, inst),
        .reduce => try select.arithmetic.airReduce(func, inst, false),
        .reduce_optimized => try select.arithmetic.airReduce(func, inst, true),

        .is_null, .is_non_null => try select.aggregate.airIsNull(func, inst, tag == .is_null, false),
        .is_null_ptr, .is_non_null_ptr => try select.aggregate.airIsNull(func, inst, tag == .is_null_ptr, true),
        .optional_payload => try select.aggregate.airPayload(func, inst, .optional),
        .optional_payload_ptr => try select.aggregate.airPayloadPtr(func, inst, .optional, false),
        .optional_payload_ptr_set => try select.aggregate.airPayloadPtr(func, inst, .optional, true),
        .wrap_optional => try select.aggregate.airWrapOptional(func, inst),
        .is_err, .is_non_err => try select.aggregate.airIsErr(func, inst, tag == .is_err, false),
        .is_err_ptr, .is_non_err_ptr => try select.aggregate.airIsErr(func, inst, tag == .is_err_ptr, true),
        .unwrap_errunion_payload => try select.aggregate.airPayload(func, inst, .error_union),
        .unwrap_errunion_err => try select.aggregate.airUnwrapErr(func, inst, false),
        .unwrap_errunion_err_ptr => try select.aggregate.airUnwrapErr(func, inst, true),
        .unwrap_errunion_payload_ptr => try select.aggregate.airPayloadPtr(func, inst, .error_union, false),
        .errunion_payload_ptr_set => try select.aggregate.airPayloadPtr(func, inst, .error_union, true),
        .wrap_errunion_payload => try select.aggregate.airWrapErrUnion(func, inst, true),
        .wrap_errunion_err => try select.aggregate.airWrapErrUnion(func, inst, false),
        .@"try", .try_cold => try isel.airTry(inst, false),
        .try_ptr, .try_ptr_cold => try isel.airTry(inst, true),

        .block => try isel.airBlock(inst, func.air.unwrapBlock(inst).body, func.air.unwrapBlock(inst).ty),
        .dbg_inline_block => try isel.airBlock(inst, func.air.unwrapDbgBlock(inst).body, func.air.unwrapDbgBlock(inst).ty),
        .loop => try isel.airLoop(inst),
        .repeat => try isel.airRepeat(inst),
        .br => try isel.airBr(inst),
        .cond_br => try isel.airCondBr(inst),
        .switch_br, .loop_switch_br => try isel.airSwitchBr(inst),
        .switch_dispatch => try isel.airSwitchDispatch(inst),
        .ret, .ret_safe => try isel.airRet(func.dataOf(inst).un_op, false, tag == .ret_safe),
        .ret_load => try isel.airRet(func.dataOf(inst).un_op, true, false),
        .unreach, .trap => lowered: {
            try func.builder.trap();
            break :lowered null;
        },
        .breakpoint => lowered: {
            try func.builder.emitEffect(.brkpt, .none, .{ .none = .{} }, .{});
            break :lowered null;
        },
        .dbg_stmt, .dbg_empty_stmt, .dbg_var_ptr, .dbg_var_val, .dbg_arg_inline, .prefetch => null,

        .call, .call_always_tail, .call_never_tail, .call_never_inline => try select.call.airCall(func, inst),
        .legalize_compiler_rt_call => try select.call.airCompilerRtCall(func, inst),
        .assembly => try select.call.airAsm(func, inst),
        .memset, .memset_safe => try select.memory.airMemset(func, inst, tag == .memset_safe),
        .memcpy => try select.memory.airMemcpy(func, inst),
        .memmove => try select.memory.airMemmove(func, inst),
        .atomic_load => try select.memory.airAtomicLoad(func, inst),
        .atomic_store_unordered => try select.memory.airAtomicStore(func, inst, .unordered),
        .atomic_store_monotonic => try select.memory.airAtomicStore(func, inst, .monotonic),
        .atomic_store_release => try select.memory.airAtomicStore(func, inst, .release),
        .atomic_store_seq_cst => try select.memory.airAtomicStore(func, inst, .seq_cst),
        .atomic_rmw => try select.memory.airAtomicRmw(func, inst),
        .cmpxchg_weak, .cmpxchg_strong => try select.memory.airCmpxchg(func, inst),
        .runtime_nav_ptr => try select.memory.airNavPtr(func, inst),

        .union_init => try select.aggregate.airUnionInit(func, inst),
        .union_from_enum => try select.aggregate.airUnionFromEnum(func, inst),
        .get_union_tag => try select.aggregate.airGetUnionTag(func, inst),
        .set_union_tag => lowered: {
            try select.aggregate.airSetUnionTag(func, inst);
            break :lowered null;
        },
        .tag_name => try select.aggregate.airTagName(func, inst),
        .error_name => try select.aggregate.airErrorName(func, inst),
        .is_named_enum_value => try select.aggregate.airIsNamedEnumValue(func, inst),
        .error_set_has_value => try select.aggregate.airErrorSetHasValue(func, inst),
        .cmp_lte_errors_len => try select.aggregate.airCmpLteErrorsLen(func, inst),
        .field_parent_ptr => try select.memory.airFieldParentPtr(func, inst),
        // PTX has no instruction that reads a return address or a frame address. The language
        // reference calls both results target-specific. The backend returns 0 for both.
        .ret_addr, .frame_addr => .{ .parts = try func.arena.dupe(Builder.Part, &.{.{ .imm = 0 }}) },
        .c_va_arg, .c_va_copy, .c_va_end, .c_va_start => return func.layout.fail("C variadic arguments are not supported on PTX", .{}),

        .work_item_id => try isel.airSpecialReg(inst, .tid),
        .work_group_size => try isel.airSpecialReg(inst, .ntid),
        .work_group_id => try isel.airSpecialReg(inst, .ctaid),

        else => |unhandled| return func.layout.todo("implement AIR tag '{t}'", .{unhandled}),
    };
}

fn airSpecialReg(isel: *Select, inst: Air.Inst.Index, reg: Mir.SpecialReg) Layout.Error!Val {
    const func = &isel.func;
    const dimension: u2 = @intCast(func.dataOf(inst).pl_op.payload);
    return func.single(try func.builder.emit(.b32, .mov, .u32, .{ .none = .{} }, .{Mir.SpecialRegRef{ .reg = reg, .dimension = dimension }}));
}

/// A block with at most one `br` continues with the result of that `br`. Every other block gets a
/// MIR block, whose parameters take the result. A block needs no label when its last instruction
/// is its only `br`.
fn airBlock(isel: *Select, inst: Air.Inst.Index, body: []const Air.Inst.Index, ty: Type) Layout.Error!?Val {
    const func = &isel.func;
    const zcu = func.zcu;
    const region = isel.regions.get(inst).?;
    const last = if (body.len > 0) body[body.len - 1] else null;
    if (region.brs <= 1) {
        const needs_label = region.brs == 1 and !region.final_br;
        const target = if (needs_label) func.builder.labelBlock() else undefined;
        try isel.exits.put(func.arena, inst, .{ .target = target, .last = last, .single = true });
        try isel.genBody(body);
        if (needs_label) _ = try func.builder.startBlock(target);
        const exit = isel.exits.get(inst).?;
        if (ty.isNoReturn(zcu)) return null;
        return .{ .parts = exit.result };
    }
    const result_scalars = if (ty.hasRuntimeBits(zcu)) try func.layout.scalarsOf(ty) else &.{};
    var classes: std.ArrayList(Mir.Class) = .empty;
    for (result_scalars) |scalar| try classes.append(func.arena, scalar.kind.class);
    const target = try func.builder.newBlock(classes.items);
    try isel.exits.put(func.arena, inst, .{
        .target = target,
        .last = last,
        .single = false,
        .result_len = @intCast(result_scalars.len),
    });
    try isel.genBody(body);
    const params = try func.builder.startBlock(target);
    return .{ .parts = try func.builder.regParts(params) };
}

fn airBr(isel: *Select, inst: Air.Inst.Index) Layout.Error!?Val {
    const func = &isel.func;
    const br = func.air.instructions.items(.data)[@backingInt(inst)].br;
    const exit = isel.exits.getPtr(br.block_inst).?;
    const has_result = func.typeOf(br.operand).hasRuntimeBits(func.zcu);
    if (exit.single) {
        if (has_result) exit.result = try func.partsOf(br.operand);
        if (exit.last == inst) return null;
        try func.builder.branch(exit.target, &.{});
        return null;
    }
    var args: std.ArrayList(Mir.Reg) = .empty;
    if (exit.result_len > 0) {
        const scalars = try func.layout.scalarsOf(func.typeOf(br.operand));
        for (try func.partsOf(br.operand), scalars) |part, scalar| try args.append(func.arena, try func.builder.toArg(part, scalar.kind));
    }
    try func.builder.branch(exit.target, args.items);
    return null;
}

fn airLoop(isel: *Select, inst: Air.Inst.Index) Layout.Error!?Val {
    const func = &isel.func;
    _ = try isel.enterLoop(inst, &.{});
    try isel.genBody(func.air.unwrapBlock(inst).body);
    return null;
}

/// Returns the parameters of the new loop header block for `operand`.
fn enterLoop(isel: *Select, inst: Air.Inst.Index, operand: []const Mir.Reg) Layout.Error![]const Mir.Reg {
    const func = &isel.func;
    var classes: std.ArrayList(Mir.Class) = .empty;
    for (operand) |reg| try classes.append(func.arena, reg.class);
    const head = try func.builder.newBlock(classes.items);
    try func.builder.branch(head, operand);
    try isel.loops.put(func.arena, inst, .{ .head = head });
    return func.builder.startBlock(head);
}

fn airRepeat(isel: *Select, inst: Air.Inst.Index) Layout.Error!?Val {
    const func = &isel.func;
    const loop = isel.loops.get(func.air.instructions.items(.data)[@backingInt(inst)].repeat.loop_inst).?;
    try isel.nextIteration(loop, &.{});
    return null;
}

fn nextIteration(isel: *Select, loop: Loop, operand: []const Mir.Reg) Layout.Error!void {
    try isel.func.builder.branch(loop.head, operand);
}

fn airCondBr(isel: *Select, inst: Air.Inst.Index) Layout.Error!?Val {
    const func = &isel.func;
    const cond_br = func.air.unwrapCondBr(inst);
    const cond = try func.scalarReg(cond_br.condition);
    const otherwise = func.builder.labelBlock();
    try func.builder.branchIf(cond, true, otherwise, &.{});
    try isel.genBody(cond_br.then_body);
    _ = try func.builder.startBlock(otherwise);
    try isel.genBody(cond_br.else_body);
    return null;
}

fn airSwitchBr(isel: *Select, inst: Air.Inst.Index) Layout.Error!?Val {
    const func = &isel.func;
    const switch_br = func.air.unwrapSwitch(inst);
    const ty = func.typeOf(switch_br.operand);
    const is_loop = func.air.instructions.items(.tag)[@backingInt(inst)] == .loop_switch_br;
    const wide_int = func.layout.wideIntOf(ty);
    var operand = if (wide_int) |info| try wide.limbRegs(&func.builder, try func.partsOf(switch_br.operand), info) else try func.arena.dupe(Mir.Reg, &.{try func.scalarReg(switch_br.operand)});
    if (is_loop) operand = try isel.enterLoop(inst, operand);
    const cases = try func.arena.alloc(Builder.Block, switch_br.cases_len);
    for (cases) |*case| case.* = func.builder.labelBlock();
    var it = switch_br.iterateCases();
    while (it.next()) |case| {
        for (case.items) |item| {
            const is_case = if (wide_int) |info|
                try wide.compareLimbs(&func.builder, .eq, operand, try wide.limbRegs(&func.builder, try func.partsOf(item), info), info.signedness)
            else
                try func.compareSource(.eq, operand[0], try func.builder.toSource((try func.partsOf(item))[0], (try func.layout.scalarsOf(ty))[0].kind), (try func.layout.scalarsOf(ty))[0].kind);
            try func.builder.branchIf(is_case, false, cases[case.idx], &.{});
        }
        for (case.ranges) |range| {
            const in_range = if (wide_int) |info| in_range: {
                const lo = try wide.compareLimbs(&func.builder, .gte, operand, try wide.limbRegs(&func.builder, try func.partsOf(range[0]), info), info.signedness);
                const hi = try wide.compareLimbs(&func.builder, .lte, operand, try wide.limbRegs(&func.builder, try func.partsOf(range[1]), info), info.signedness);
                break :in_range try func.builder.emit(.pred, .@"and", .pred, .{ .none = .{} }, .{ lo, hi });
            } else in_range: {
                const kind = (try func.layout.scalarsOf(ty))[0].kind;
                const lo = try func.builder.emit(.pred, .setp, kind.arithmeticType(), .{ .setp = .{ .cmp = .ge } }, .{ operand[0], try func.builder.toSource((try func.partsOf(range[0]))[0], kind) });
                break :in_range try func.builder.emit(.pred, .setp, kind.arithmeticType(), .{ .setp = .{ .cmp = .le, .bool_op = .@"and" } }, .{ operand[0], try func.builder.toSource((try func.partsOf(range[1]))[0], kind), lo });
            };
            try func.builder.branchIf(in_range, false, cases[case.idx], &.{});
        }
    }
    try isel.genBody(it.elseBody());
    it = switch_br.iterateCases();
    while (it.next()) |case| {
        _ = try func.builder.startBlock(cases[case.idx]);
        try isel.genBody(case.body);
    }
    return null;
}

fn airSwitchDispatch(isel: *Select, inst: Air.Inst.Index) Layout.Error!?Val {
    const func = &isel.func;
    const br = func.air.instructions.items(.data)[@backingInt(inst)].br;
    const loop = isel.loops.get(br.block_inst).?;
    const scalars = try func.layout.scalarsOf(func.typeOf(br.operand));
    try isel.nextIteration(loop, try func.builder.toRegs(try func.partsOf(br.operand), scalars));
    return null;
}

fn airTry(isel: *Select, inst: Air.Inst.Index, is_ptr: bool) Layout.Error!Val {
    const func = &isel.func;
    const zcu = func.zcu;
    const operand, const else_body = if (is_ptr) lowered: {
        const unwrapped = func.air.unwrapTryPtr(inst);
        break :lowered .{ unwrapped.error_union_ptr, unwrapped.else_body };
    } else lowered: {
        const unwrapped = func.air.unwrapTry(inst);
        break :lowered .{ unwrapped.error_union, unwrapped.else_body };
    };
    const error_union_type = if (is_ptr) func.typeOf(operand).childType(zcu) else func.typeOf(operand);
    if (error_union_type.errorUnionSet(zcu).hasRuntimeBits(zcu)) {
        const ok = func.builder.labelBlock();
        const is_ok = try func.builder.emit(.pred, .setp, .u16, .{ .setp = .{ .cmp = .eq } }, .{ try select.aggregate.errorCode(func, operand, error_union_type, is_ptr), 0 });
        try func.builder.branchIf(is_ok, false, ok, &.{});
        try isel.genBody(else_body);
        _ = try func.builder.startBlock(ok);
    }
    const payload_type = error_union_type.errorUnionPayload(zcu);
    if (is_ptr) return func.offsetPtr(try func.resolve(operand), func.typeOf(operand), @intCast(codegen.errUnionPayloadOffset(payload_type, zcu)));
    const src = try func.partsOf(operand);
    return .{ .parts = src[src.len - try func.layout.scalarCount(payload_type) ..] };
}

fn airRet(isel: *Select, operand: Air.Inst.Ref, is_load: bool, safety: bool) Layout.Error!?Val {
    const func = &isel.func;
    const zcu = func.zcu;
    if (func.ret_type.hasRuntimeBits(zcu)) {
        const scalars = try func.layout.scalarsOf(func.ret_type);
        const src = if (is_load) try func.loadThrough(operand, scalars) else try func.safeParts(operand, safety);
        try memory.storeMem(&func.builder, .{ .name = "func_retval0" }, .param, func.ret_type.abiAlignment(zcu).toByteUnits().?, scalars, src, false);
    }
    try func.builder.ret();
    return null;
}
