//! This file generates the MIR of functions and the PTX declarations of data. It calls
//! `codegen/Select.zig` for functions and `codegen/constant.zig` for data. CSE and DCE simplify
//! each function that `Select.zig` generates, before the linker inlines the function.

const std = @import("std");

const Air = @import("../../Air.zig");
const InternPool = @import("../../InternPool.zig");
const Type = @import("../../Type.zig");
const Value = @import("../../Value.zig");
const Zcu = @import("../../Zcu.zig");
const codegen = @import("../../codegen.zig");
const link = @import("../../link.zig");
const tracy = @import("../../tracy.zig");
const Mir = @import("Mir.zig");
const findRegions = @import("regions.zig").find;
const verify = @import("verify.zig");
const cse = @import("transforms/cse.zig");
const dce = @import("transforms/dce.zig");
const promote = @import("transforms/promote.zig");
const Builder = @import("codegen/Builder.zig");
const Layout = @import("codegen/Layout.zig");
const Select = @import("codegen/Select.zig");
const call = @import("codegen/call.zig");
const constant = @import("codegen/constant.zig");

/// The backend uses registers of at most 64 bits and moves at most 16 bytes with one `ld` or `st`.
/// Only sm_100 and later have 32-byte forms (PTX ISA 9.4, sections 9.7.10.8 and 9.7.10.11).
/// `Legalize` therefore scalarizes the operations on vectors wider than 16 bytes or with elements
/// wider than 64 bits. The backend lowers the operations on smaller vectors one element at a time.
pub fn legalizeFeatures(_: *const std.Target) *const Air.Legalize.Features {
    return comptime &.initMany(&.{
        .scalarize_only_large_vectors,
        .scalarize_add,
        .scalarize_add_optimized,
        .scalarize_add_wrap,
        .scalarize_add_sat,
        .scalarize_sub,
        .scalarize_sub_optimized,
        .scalarize_sub_wrap,
        .scalarize_sub_sat,
        .scalarize_mul,
        .scalarize_mul_optimized,
        .scalarize_mul_wrap,
        .scalarize_mul_sat,
        .scalarize_div_float,
        .scalarize_div_float_optimized,
        .scalarize_div_trunc,
        .scalarize_div_trunc_optimized,
        .scalarize_div_floor,
        .scalarize_div_floor_optimized,
        .scalarize_div_exact,
        .scalarize_div_exact_optimized,
        .scalarize_rem,
        .scalarize_rem_optimized,
        .scalarize_mod,
        .scalarize_mod_optimized,
        .scalarize_max,
        .scalarize_min,
        .scalarize_add_with_overflow,
        .scalarize_sub_with_overflow,
        .scalarize_mul_with_overflow,
        .scalarize_shl_with_overflow,
        .scalarize_bit_and,
        .scalarize_bit_or,
        .scalarize_shr,
        .scalarize_shr_exact,
        .scalarize_shl,
        .scalarize_shl_exact,
        .scalarize_shl_sat,
        .scalarize_xor,
        .scalarize_not,
        .scalarize_ptr_cast,
        .scalarize_ptr_from_int,
        .scalarize_int_from_ptr,
        .scalarize_clz,
        .scalarize_ctz,
        .scalarize_popcount,
        .scalarize_byte_swap,
        .scalarize_bit_reverse,
        .scalarize_sqrt,
        .scalarize_sin,
        .scalarize_cos,
        .scalarize_tan,
        .scalarize_exp,
        .scalarize_exp2,
        .scalarize_log,
        .scalarize_log2,
        .scalarize_log10,
        .scalarize_abs,
        .scalarize_floor,
        .scalarize_ceil,
        .scalarize_round,
        .scalarize_trunc_float,
        .scalarize_neg,
        .scalarize_neg_optimized,
        .scalarize_cmp_vector,
        .scalarize_cmp_vector_optimized,
        .scalarize_fptrunc,
        .scalarize_fpext,
        .scalarize_int_cast,
        .scalarize_trunc,
        .scalarize_int_from_float,
        .scalarize_int_from_float_optimized,
        .scalarize_float_from_int,
        .scalarize_select,
        .scalarize_mul_add,
        .expand_bit_cast_safe,
        .expand_int_cast_safe,
        .expand_int_from_float_safe,
        .expand_int_from_float_optimized_safe,
        .expand_add_safe,
        .expand_sub_safe,
        .expand_mul_safe,
        .expand_div_ceil,
        .expand_div_ceil_optimized,
        .expand_packed_load,
        .expand_packed_store,
        .expand_packed_agg_field_val,
        .expand_packed_aggregate_init,
        .expand_array_splat,
        .expand_array_to_vector,
        .expand_ptr_elem_val,
        .expand_array_to_slice,
        .soft_f80,
        .soft_f128,
    });
}

pub fn generate(
    _: *link.File,
    pt: Zcu.PerThread,
    func_index: InternPool.Index,
    air: *const Air,
    liveness: *const ?Air.Liveness,
) codegen.Error!Mir {
    const zcu = pt.zcu;
    const gpa = zcu.gpa;
    const owner_nav = zcu.funcInfo(func_index).owner_nav;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var isel: Select = .{
        .func = .{
            .pt = pt,
            .zcu = zcu,
            .gpa = gpa,
            .arena = arena.allocator(),
            .air = air.*,
            .liveness = liveness.*.?,
            .builder = .{ .arena = arena.allocator() },
            .layout = .init(pt, owner_nav, arena.allocator()),
        },
        .regions = try findRegions(arena.allocator(), air),
        .is_noinline = zcu.funcInfo(func_index).analysisUnordered(&zcu.intern_pool).is_noinline,
    };
    var mir = generated: {
        const tracy_trace = tracy.traceNamed(@src(), "NVPTX code generation");
        defer tracy_trace.end();
        break :generated isel.genFunction() catch |err| return failure(&isel.func.layout, err);
    };
    errdefer mir.deinit(gpa);
    try optimize(gpa, &mir, zcu.navFileScope(owner_nav).mod.?.optimize_mode, isel.func.param_copies.items);
    if (std.debug.runtime_safety) {
        if (try verify.check(arena.allocator(), &mir)) |message| return isel.func.layout.fail("internal error: invalid SSA form after the transforms: {s}", .{message});
    }
    return mir;
}

/// Runs promotion on the thread that generates the function, and in release modes CSE and DCE. The
/// linker inlines the function later and expands its copies. Debug mode runs no other transform,
/// so that a Debug build is the baseline of those transforms.
fn optimize(gpa: std.mem.Allocator, mir: *Mir, optimize_mode: std.lang.Optimize, param_copies: []const Mir.ParamCopy) std.mem.Allocator.Error!void {
    {
        const tracy_trace = tracy.traceNamed(@src(), "NVPTX promotion before inlining");
        defer tracy_trace.end();
        tracy_trace.setValue(mir.instructions.len);
        try promote.run(gpa, mir, param_copies);
    }
    if (optimize_mode == .debug) return;
    {
        const tracy_trace = tracy.traceNamed(@src(), "NVPTX CSE before inlining");
        defer tracy_trace.end();
        tracy_trace.setValue(mir.instructions.len);
        try cse.run(gpa, mir);
    }
    const tracy_trace = tracy.traceNamed(@src(), "NVPTX DCE before inlining");
    defer tracy_trace.end();
    tracy_trace.setValue(mir.instructions.len);
    try dce.run(gpa, mir);
}

pub fn generateNav(pt: Zcu.PerThread, nav_index: InternPool.Nav.Index) codegen.Error!Mir {
    const zcu = pt.zcu;
    const ip = &zcu.intern_pool;
    const nav = ip.getNav(nav_index);
    const resolved = nav.resolved.?;
    const ty: Type = .fromInterned(resolved.type);
    var arena: std.heap.ArenaAllocator = .init(zcu.gpa);
    defer arena.deinit();
    var layout: Layout = .init(pt, nav_index, arena.allocator());
    var b: Builder = .{ .arena = arena.allocator() };
    const space = Layout.Space.fromAddressSpace(resolved.@"addrspace") orelse
        return layout.fail("address space '{t}' is not valid for a global", .{resolved.@"addrspace"});
    if (resolved.@"threadlocal") return layout.fail("threadlocal variables are not supported", .{});
    const val: ?Value = if (nav.getExtern(ip) != null) null else zcu.navValue(nav_index);
    const alignment = resolved.@"align".toByteUnits() orelse ty.abiAlignment(zcu).toByteUnits().?;
    constant.writeData(&b, &layout, .{ .nav = nav_index }, ty, val, space, alignment) catch |err| return failure(&layout, err);
    return b.finish(zcu.gpa, .data, 0, &.{}, null);
}

/// Returns the data of an unnamed constant in `.global` memory for a `.uav` and in `.const` memory
/// for a `.const_uav`. Assumes that `target` is one of these.
pub fn generateUav(pt: Zcu.PerThread, owner_nav: InternPool.Nav.Index, target: Mir.Target) codegen.Error!Mir {
    const zcu = pt.zcu;
    var arena: std.heap.ArenaAllocator = .init(zcu.gpa);
    defer arena.deinit();
    var layout: Layout = .init(pt, owner_nav, arena.allocator());
    var b: Builder = .{ .arena = arena.allocator() };
    const uav, const space: Layout.Space = switch (target) {
        .uav => |uav| .{ uav, .global },
        .const_uav => |uav| .{ uav, .@"const" },
        else => unreachable,
    };
    const val: Value = .fromInterned(uav);
    const ty = val.typeOf(zcu);
    // A pointer in the constant needs 8-byte alignment to become a `.u64` symbol.
    const alignment = @max(8, ty.abiAlignment(zcu).toByteUnits().?);
    constant.writeData(&b, &layout, target, ty, val, space, alignment) catch |err| return failure(&layout, err);
    return b.finish(zcu.gpa, .data, 0, &.{}, null);
}

pub fn generateExternFunc(
    pt: Zcu.PerThread,
    owner_nav: InternPool.Nav.Index,
    fn_type: InternPool.Index,
    target: Mir.Target,
) codegen.Error!Mir {
    const zcu = pt.zcu;
    var arena: std.heap.ArenaAllocator = .init(zcu.gpa);
    defer arena.deinit();
    var layout: Layout = .init(pt, owner_nav, arena.allocator());
    var b: Builder = .{ .arena = arena.allocator() };
    var prototype: std.ArrayList(u8) = .empty;
    const fn_info = zcu.typeToFunc(.fromInterned(fn_type)).?;
    const name_offset = call.declare(&b, &layout, fn_info, ".extern ", &prototype) catch |err| return failure(&layout, err);
    return b.finish(zcu.gpa, .func, @intCast(prototype.items.len), prototype.items, .{ .offset = name_offset, .target = target });
}

/// Returns the name table of the enum type in `target.tag_names`, or the error name table for
/// `.error_names`.
pub fn generateNames(pt: Zcu.PerThread, owner_nav: InternPool.Nav.Index, target: Mir.Target) codegen.Error!Mir {
    const zcu = pt.zcu;
    var arena: std.heap.ArenaAllocator = .init(zcu.gpa);
    defer arena.deinit();
    var layout: Layout = .init(pt, owner_nav, arena.allocator());
    var b: Builder = .{ .arena = arena.allocator() };
    constant.writeNames(&b, &layout, target) catch |err| return failure(&layout, err);
    return b.finish(zcu.gpa, .data, 0, &.{}, null);
}

fn failure(layout: *const Layout, err: Layout.Error) codegen.Error {
    return switch (err) {
        error.CannotSplit => layout.todo("a value without a register form", .{}),
        error.TooManyRegisters => layout.fail("a function needs more than 65535 registers of one class", .{}),
        else => |e| e,
    };
}
