//! This file decides whether `Function.loadPtr` can give the value of a large AIR `load` a place at
//! the memory that the load reads, with no copy. Each use of a place reads the memory, so the
//! memory must keep its bytes until the last use of the value.

const std = @import("std");
const Allocator = std.mem.Allocator;

const Air = @import("../../../Air.zig");
const Value = @import("../../../Value.zig");
const Zcu = @import("../../../Zcu.zig");
const Function = @import("Function.zig");

/// `init` finds the facts about the AIR of one function in one walk over its bodies.
pub const Facts = struct {
    bodies: std.ArrayList([]const Air.Inst.Index) = .empty,
    /// The number of bodies that contain each body of `bodies`.
    depths: std.ArrayList(u32) = .empty,
    positions: []Position,
    /// `last_uses[x]` is the largest index in the body of `x` of an instruction that uses `x`,
    /// directly or in a nested body.
    last_uses: []u32,
    /// The alloc that each pointer points into, through the instructions of `pointerBase`.
    roots: []?Air.Inst.Index,
    /// An alloc escapes when an instruction uses a pointer into the alloc as a value, for example
    /// as a call argument, as the stored value of a `store` or as an operand of `int_from_ptr`.
    escaped: std.DynamicBitSetUnmanaged,

    const Position = struct {
        body: u32,
        index: u32,
    };

    pub fn init(func: *const Function) Allocator.Error!Facts {
        const count = func.air.instructions.len;
        var walk: Walk = .{ .func = func, .facts = .{
            .positions = try func.arena.alloc(Position, count),
            .last_uses = try func.arena.alloc(u32, count),
            .roots = try func.arena.alloc(?Air.Inst.Index, count),
            .escaped = try .initEmpty(func.arena, count),
        } };
        @memset(walk.facts.roots, null);
        try walk.body(func.air.getMainBody());
        return walk.facts;
    }

    fn rootOf(facts: *const Facts, ptr: Air.Inst.Ref) ?Air.Inst.Index {
        return facts.roots[@backingInt(ptr.toIndex() orelse return null)];
    }

    fn isEscaped(facts: *const Facts, alloc: Air.Inst.Index) bool {
        return facts.escaped.isSet(@backingInt(alloc));
    }

    fn keeps(facts: *const Facts, func: *const Function, body: []const Air.Inst.Index, source: Source) bool {
        for (body) |inst| {
            if (!isLowered(func, inst)) continue;
            if (!facts.preserves(func, inst, source)) return false;
            var bodies: Bodies = .init(&func.air, inst);
            while (bodies.next()) |nested| if (!facts.keeps(func, nested, source)) return false;
        }
        return true;
    }

    /// Returns true when `inst` cannot change the bytes at `source`. Without an alloc, a called
    /// function can write `source`, and an atomic operation can order a write of another thread
    /// before a use. So `inst` is no call, no atomic operation and no `asm` that accesses memory.
    fn preserves(facts: *const Facts, func: *const Function, inst: Air.Inst.Index, source: Source) bool {
        const air = &func.air;
        const data = air.instructions.items(.data)[@backingInt(inst)];
        const dest: Air.Inst.Ref = switch (air.instructions.items(.tag)[@backingInt(inst)]) {
            .store, .store_safe, .memset, .memset_safe, .memcpy, .memmove, .set_union_tag => data.bin_op.lhs,
            .optional_payload_ptr_set, .errunion_payload_ptr_set => data.ty_op.operand,
            .legalize_vec_store_elem => data.pl_op.operand,
            .atomic_store_unordered, .atomic_store_monotonic, .atomic_store_release, .atomic_store_seq_cst => if (source.alloc == null) return false else data.bin_op.lhs,
            .atomic_rmw => if (source.alloc == null) return false else data.pl_op.operand,
            .cmpxchg_weak, .cmpxchg_strong => if (source.alloc == null) return false else air.extraData(Air.Cmpxchg, data.ty_pl.payload).data.ptr,
            .atomic_load, .call, .call_always_tail, .call_never_tail, .call_never_inline, .legalize_compiler_rt_call => return source.alloc != null,
            .assembly => return source.alloc != null or !accessesMemory(func, inst),
            else => return true,
        };
        if (source.alloc) |alloc| return facts.rootOf(dest) != alloc;
        if (facts.rootOf(dest)) |alloc| {
            // An alloc lies in `.local` memory, or in `.param` memory for a parameter copy. Only a
            // `.generic` or a `.local` address can reach `.local` memory (PTX ISA 9.4, section
            // 5.1), and no instruction writes `.param` memory (section 5.1.6.1).
            return !facts.isEscaped(alloc) or (source.address_space != .generic and source.address_space != .local);
        }
        const dest_address_space = air.typeOf(dest, &func.zcu.intern_pool).ptrAddressSpace(func.zcu);
        return source.address_space != .generic and dest_address_space != .generic and source.address_space != dest_address_space;
    }
};

const Source = struct {
    alloc: ?Air.Inst.Index,
    address_space: std.lang.AddressSpace,
};

/// Returns true when the source of `load` keeps its bytes until the last use of its value, which
/// needs an unescaped alloc or no alloc. A use of a field or an element of the value counts.
pub fn keepsSource(func: *Function, load: Air.Inst.Index, ptr: Air.Inst.Ref) Allocator.Error!bool {
    if (func.elision_facts == null) func.elision_facts = try .init(func);
    const facts = &func.elision_facts.?;
    const position = facts.positions[@backingInt(load)];
    const body = facts.bodies.items[position.body];
    var values: std.ArrayList(Air.Inst.Index) = .empty;
    try values.append(func.arena, load);
    var last = @max(position.index, facts.last_uses[@backingInt(load)]);
    var index = position.index + 1;
    while (index <= last) : (index += 1) {
        const inst = body[index];
        if (!isLowered(func, inst)) continue;
        const aggregate = (placeOperand(&func.air, inst) orelse continue).toIndex() orelse continue;
        if (std.mem.findScalar(Air.Inst.Index, values.items, aggregate) == null) continue;
        try values.append(func.arena, inst);
        last = @max(last, facts.last_uses[@backingInt(inst)]);
    }
    const source: Source = .{ .alloc = facts.rootOf(ptr), .address_space = func.typeOf(ptr).ptrAddressSpace(func.zcu) };
    if (source.alloc) |alloc| if (facts.isEscaped(alloc)) return false;
    return facts.keeps(func, body[position.index + 1 .. last + 1], source);
}

/// Returns false for the instructions that `Select.genBody` skips.
fn isLowered(func: *const Function, inst: Air.Inst.Index) bool {
    return !func.liveness.isUnused(inst) or func.air.mustLower(inst, &func.zcu.intern_pool);
}

const Walk = struct {
    func: *const Function,
    facts: Facts,
    /// The index of the current instruction in each body that contains it.
    indices: std.ArrayList(u32) = .empty,
    operands: std.ArrayList(Air.Inst.Ref) = .empty,

    fn body(walk: *Walk, insts: []const Air.Inst.Index) Allocator.Error!void {
        const func = walk.func;
        const facts = &walk.facts;
        const body_index: u32 = @intCast(facts.bodies.items.len);
        try facts.bodies.append(func.arena, insts);
        try facts.depths.append(func.arena, @intCast(walk.indices.items.len));
        try walk.indices.append(func.arena, 0);
        defer _ = walk.indices.pop();
        for (insts, 0..) |inst, index| {
            if (!isLowered(func, inst)) continue;
            walk.indices.items[walk.indices.items.len - 1] = @intCast(index);
            facts.positions[@backingInt(inst)] = .{ .body = body_index, .index = @intCast(index) };
            facts.last_uses[@backingInt(inst)] = @intCast(index);
            walk.operands.clearRetainingCapacity();
            try appendOperands(&func.air, func.zcu, inst, &walk.operands, func.arena);
            const addresses = addressOperands(func.air.instructions.items(.tag)[@backingInt(inst)]);
            for (walk.operands.items, 0..) |operand, i| {
                const used = operand.toIndex() orelse continue;
                const defined_in = facts.positions[@backingInt(used)].body;
                const at = walk.indices.items[facts.depths.items[defined_in]];
                facts.last_uses[@backingInt(used)] = @max(facts.last_uses[@backingInt(used)], at);
                if (i >= addresses) if (facts.roots[@backingInt(used)]) |alloc| facts.escaped.set(@backingInt(alloc));
            }
            facts.roots[@backingInt(inst)] = switch (func.air.instructions.items(.tag)[@backingInt(inst)]) {
                .alloc, .ret_ptr => inst,
                else => if (pointerBase(&func.air, inst)) |base| facts.rootOf(base) else null,
            };
            var bodies: Bodies = .init(&func.air, inst);
            while (bodies.next()) |nested| try walk.body(nested);
        }
    }
};

const Bodies = struct {
    fixed: [2][]const Air.Inst.Index = undefined,
    fixed_len: u2 = 0,
    next_fixed: u2 = 0,
    cases: ?Air.UnwrappedSwitch.CaseIterator = null,

    fn init(air: *const Air, inst: Air.Inst.Index) Bodies {
        return switch (air.instructions.items(.tag)[@backingInt(inst)]) {
            .block, .loop => .{ .fixed = .{ air.unwrapBlock(inst).body, &.{} }, .fixed_len = 1 },
            .dbg_inline_block => .{ .fixed = .{ air.unwrapDbgBlock(inst).body, &.{} }, .fixed_len = 1 },
            .@"try", .try_cold => .{ .fixed = .{ air.unwrapTry(inst).else_body, &.{} }, .fixed_len = 1 },
            .try_ptr, .try_ptr_cold => .{ .fixed = .{ air.unwrapTryPtr(inst).else_body, &.{} }, .fixed_len = 1 },
            .cond_br => cond_br: {
                const cond_br = air.unwrapCondBr(inst);
                break :cond_br .{ .fixed = .{ cond_br.then_body, cond_br.else_body }, .fixed_len = 2 };
            },
            .switch_br, .loop_switch_br => .{ .cases = air.unwrapSwitch(inst).iterateCases() },
            else => .{},
        };
    }

    fn next(bodies: *Bodies) ?[]const Air.Inst.Index {
        if (bodies.cases) |*cases| {
            if (cases.next()) |case| return case.body;
            const else_body = cases.elseBody();
            bodies.cases = null;
            return else_body;
        }
        if (bodies.next_fixed == bodies.fixed_len) return null;
        defer bodies.next_fixed += 1;
        return bodies.fixed[bodies.next_fixed];
    }
};

/// Returns the base pointer of `inst` when both point into the same object. `Function.allocRoot`
/// follows the same instructions.
fn pointerBase(air: *const Air, inst: Air.Inst.Index) ?Air.Inst.Ref {
    const data = air.instructions.items(.data)[@backingInt(inst)];
    return switch (air.instructions.items(.tag)[@backingInt(inst)]) {
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
        else => null,
    };
}

/// Returns the number of leading operands of `appendOperands` that an instruction with `tag` uses
/// only to access memory or to derive a pointer into the same object.
fn addressOperands(tag: Air.Inst.Tag) u2 {
    return switch (tag) {
        .memcpy, .memmove => 2,
        .load,
        .store,
        .store_safe,
        .memset,
        .memset_safe,
        .set_union_tag,
        .atomic_load,
        .atomic_store_unordered,
        .atomic_store_monotonic,
        .atomic_store_release,
        .atomic_store_seq_cst,
        .atomic_rmw,
        .cmpxchg_weak,
        .cmpxchg_strong,
        .legalize_vec_store_elem,
        .is_null_ptr,
        .is_non_null_ptr,
        .is_err_ptr,
        .is_non_err_ptr,
        .unwrap_errunion_err_ptr,
        .ret_load,
        .dbg_var_ptr,
        .prefetch,
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
        .struct_field_ptr,
        .ptr_elem_ptr,
        .ptr_add,
        .ptr_sub,
        .field_parent_ptr,
        .try_ptr,
        .try_ptr_cold,
        => 1,
        else => 0,
    };
}

/// Returns the aggregate operand of an instruction that can read a field or an element of a place.
fn placeOperand(air: *const Air, inst: Air.Inst.Index) ?Air.Inst.Ref {
    const data = air.instructions.items(.data)[@backingInt(inst)];
    return switch (air.instructions.items(.tag)[@backingInt(inst)]) {
        .array_elem_val, .legalize_vec_elem_val => data.bin_op.lhs,
        .agg_field_val => air.extraData(Air.StructField, data.ty_pl.payload).data.struct_operand,
        .slice_ptr, .slice_len => data.ty_op.operand,
        else => null,
    };
}

/// Returns true for a volatile `asm`, an `asm` with clobbers and an `asm` with a pointer operand.
fn accessesMemory(func: *const Function, inst: Air.Inst.Index) bool {
    const zcu = func.zcu;
    const unwrapped = func.air.unwrapAsm(inst);
    if (unwrapped.is_volatile) return true;
    var clobbers: Value.BigIntSpace = undefined;
    if (!Value.fromInterned(unwrapped.clobbers).toBigInt(&clobbers, zcu).eqlZero()) return true;
    for (unwrapped.outputs) |output| if (output != .none) return true;
    for (unwrapped.inputs) |input| if (func.air.typeOf(input, &zcu.intern_pool).isPtrAtRuntime(zcu)) return true;
    return false;
}

/// Appends the operands of `inst`, without its bodies, in the order of `Air.Liveness`.
fn appendOperands(air: *const Air, zcu: *const Zcu, inst: Air.Inst.Index, list: *std.ArrayList(Air.Inst.Ref), arena: Allocator) Allocator.Error!void {
    const data = air.instructions.items(.data)[@backingInt(inst)];
    switch (air.instructions.items(.tag)[@backingInt(inst)]) {
        .add,
        .add_safe,
        .add_optimized,
        .add_wrap,
        .add_sat,
        .sub,
        .sub_safe,
        .sub_optimized,
        .sub_wrap,
        .sub_sat,
        .mul,
        .mul_safe,
        .mul_optimized,
        .mul_wrap,
        .mul_sat,
        .div_float,
        .div_float_optimized,
        .div_trunc,
        .div_trunc_optimized,
        .div_floor,
        .div_floor_optimized,
        .div_exact,
        .div_exact_optimized,
        .div_ceil,
        .div_ceil_optimized,
        .rem,
        .rem_optimized,
        .mod,
        .mod_optimized,
        .bit_and,
        .bit_or,
        .xor,
        .cmp_lt,
        .cmp_lt_optimized,
        .cmp_lte,
        .cmp_lte_optimized,
        .cmp_eq,
        .cmp_eq_optimized,
        .cmp_gte,
        .cmp_gte_optimized,
        .cmp_gt,
        .cmp_gt_optimized,
        .cmp_neq,
        .cmp_neq_optimized,
        .store,
        .store_safe,
        .array_elem_val,
        .slice_elem_val,
        .ptr_elem_val,
        .shl,
        .shl_exact,
        .shl_sat,
        .shr,
        .shr_exact,
        .atomic_store_unordered,
        .atomic_store_monotonic,
        .atomic_store_release,
        .atomic_store_seq_cst,
        .set_union_tag,
        .min,
        .max,
        .memset,
        .memset_safe,
        .memcpy,
        .memmove,
        .legalize_vec_elem_val,
        => try list.appendSlice(arena, &.{ data.bin_op.lhs, data.bin_op.rhs }),

        .arg,
        .alloc,
        .ret_ptr,
        .breakpoint,
        .dbg_stmt,
        .dbg_empty_stmt,
        .ret_addr,
        .frame_addr,
        .wasm_memory_size,
        .err_return_trace,
        .save_err_return_trace_index,
        .runtime_nav_ptr,
        .c_va_start,
        .work_item_id,
        .work_group_size,
        .work_group_id,
        .trap,
        .unreach,
        .repeat,
        .block,
        .loop,
        .dbg_inline_block,
        => {},

        .inferred_alloc, .inferred_alloc_comptime => unreachable,

        .not,
        .bit_cast,
        .bit_cast_safe,
        .ptr_cast,
        .ptr_from_int,
        .int_from_ptr,
        .error_cast,
        .error_from_int,
        .int_from_error,
        .union_from_enum,
        .load,
        .fpext,
        .fptrunc,
        .int_cast,
        .int_cast_safe,
        .trunc,
        .optional_payload,
        .optional_payload_ptr,
        .optional_payload_ptr_set,
        .errunion_payload_ptr_set,
        .wrap_optional,
        .unwrap_errunion_payload,
        .unwrap_errunion_err,
        .unwrap_errunion_payload_ptr,
        .unwrap_errunion_err_ptr,
        .wrap_errunion_payload,
        .wrap_errunion_err,
        .slice_ptr,
        .slice_len,
        .ptr_slice_len_ptr,
        .ptr_slice_ptr_ptr,
        .struct_field_ptr_index_0,
        .struct_field_ptr_index_1,
        .struct_field_ptr_index_2,
        .struct_field_ptr_index_3,
        .array_to_slice,
        .array_to_vector,
        .int_from_float,
        .int_from_float_optimized,
        .int_from_float_safe,
        .int_from_float_optimized_safe,
        .float_from_int,
        .get_union_tag,
        .clz,
        .ctz,
        .popcount,
        .byte_swap,
        .bit_reverse,
        .splat,
        .error_set_has_value,
        .addrspace_cast,
        .c_va_arg,
        .c_va_copy,
        .abs,
        => try list.append(arena, data.ty_op.operand),

        .is_null,
        .is_non_null,
        .is_null_ptr,
        .is_non_null_ptr,
        .is_err,
        .is_non_err,
        .is_err_ptr,
        .is_non_err_ptr,
        .is_named_enum_value,
        .tag_name,
        .error_name,
        .sqrt,
        .sin,
        .cos,
        .tan,
        .exp,
        .exp2,
        .log,
        .log2,
        .log10,
        .floor,
        .ceil,
        .round,
        .trunc_float,
        .neg,
        .neg_optimized,
        .cmp_lte_errors_len,
        .set_err_return_trace,
        .c_va_end,
        .ret,
        .ret_safe,
        .ret_load,
        => try list.append(arena, data.un_op),

        .add_with_overflow,
        .sub_with_overflow,
        .mul_with_overflow,
        .shl_with_overflow,
        .ptr_add,
        .ptr_sub,
        .ptr_elem_ptr,
        .slice_elem_ptr,
        .slice,
        => {
            const bin = air.extraData(Air.Bin, data.ty_pl.payload).data;
            try list.appendSlice(arena, &.{ bin.lhs, bin.rhs });
        },

        .dbg_var_ptr,
        .dbg_var_val,
        .dbg_arg_inline,
        .wasm_memory_grow,
        .cond_br,
        .@"try",
        .try_cold,
        .switch_br,
        .loop_switch_br,
        => try list.append(arena, data.pl_op.operand),

        .prefetch => try list.append(arena, data.prefetch.ptr),
        .call, .call_always_tail, .call_never_tail, .call_never_inline => {
            const unwrapped = air.unwrapCall(inst);
            try list.append(arena, unwrapped.callee);
            try list.appendSlice(arena, unwrapped.args);
        },
        .legalize_compiler_rt_call => try list.appendSlice(arena, air.unwrapCompilerRtCall(inst).args),
        .select, .mul_add, .legalize_vec_store_elem => {
            const bin = air.extraData(Air.Bin, data.pl_op.payload).data;
            try list.appendSlice(arena, if (air.instructions.items(.tag)[@backingInt(inst)] == .mul_add)
                &.{ bin.lhs, bin.rhs, data.pl_op.operand }
            else
                &.{ data.pl_op.operand, bin.lhs, bin.rhs });
        },
        .shuffle_one => try list.append(arena, air.unwrapShuffleOne(zcu, inst).operand),
        .shuffle_two => {
            const unwrapped = air.unwrapShuffleTwo(zcu, inst);
            try list.appendSlice(arena, &.{ unwrapped.operand_a, unwrapped.operand_b });
        },
        .reduce, .reduce_optimized => try list.append(arena, data.reduce.operand),
        .cmp_vector, .cmp_vector_optimized => {
            const extra = air.extraData(Air.VectorCmp, data.ty_pl.payload).data;
            try list.appendSlice(arena, &.{ extra.lhs, extra.rhs });
        },
        .aggregate_init => {
            const len: usize = @intCast(data.ty_pl.ty.arrayLenIp(&zcu.intern_pool));
            try list.appendSlice(arena, @ptrCast(air.extra.items[data.ty_pl.payload..][0..len]));
        },
        .union_init => try list.append(arena, air.extraData(Air.UnionInit, data.ty_pl.payload).data.init),
        .struct_field_ptr, .agg_field_val, .spirv_runtime_array_len => try list.append(arena, air.extraData(Air.StructField, data.ty_pl.payload).data.struct_operand),
        .field_parent_ptr => try list.append(arena, air.extraData(Air.FieldParentPtr, data.ty_pl.payload).data.field_ptr),
        .cmpxchg_weak, .cmpxchg_strong => {
            const extra = air.extraData(Air.Cmpxchg, data.ty_pl.payload).data;
            try list.appendSlice(arena, &.{ extra.ptr, extra.expected_value, extra.new_value });
        },
        .atomic_load => try list.append(arena, data.atomic_load.ptr),
        .atomic_rmw => try list.appendSlice(arena, &.{ data.pl_op.operand, air.extraData(Air.AtomicRmw, data.pl_op.payload).data.operand }),
        .br, .switch_dispatch => try list.append(arena, data.br.operand),
        .assembly => {
            const unwrapped = air.unwrapAsm(inst);
            for (unwrapped.outputs) |output| if (output != .none) try list.append(arena, output);
            try list.appendSlice(arena, unwrapped.inputs);
        },
        .try_ptr, .try_ptr_cold => try list.append(arena, air.unwrapTryPtr(inst).error_union_ptr),
    }
}
