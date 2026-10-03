//! This file declares functions and their parameters, and lowers call sequences.

const std = @import("std");

const Air = @import("../../../Air.zig");
const InternPool = @import("../../../InternPool.zig");
const Type = @import("../../../Type.zig");
const Mir = @import("../Mir.zig");
const Kind = @import("Kind.zig");
const Builder = @import("Builder.zig");
const Layout = @import("Layout.zig");
const int = @import("int.zig");
const memory = @import("memory.zig");

pub fn kernelDirectives(builder: *Builder, layout: *const Layout, options: std.lang.CallingConvention.NvptxKernelOptions, out: *std.ArrayList(u8)) !void {
    if (options.max_work_group_size) |size| try out.print(builder.arena, "\n.maxntid {d}, {d}, {d}", .{ size.x, size.y, size.z });
    if (options.required_work_group_size) |size| try out.print(builder.arena, "\n.reqntid {d}, {d}, {d}", .{ size.x, size.y, size.z });
    if (options.min_work_groups_per_sm) |count| try out.print(builder.arena, "\n.minnctapersm {d}", .{count});
    if (options.cluster_size) |size| {
        // `.reqnctapercluster` needs PTX 7.8 and sm_90 (PTX ISA 9.4, section 11.7.1).
        if (layout.smVersion() < 90) return layout.fail("'cluster_size' needs sm_90 or later", .{});
        builder.needPtx(78);
        try out.print(builder.arena, "\n.reqnctapercluster {d}, {d}, {d}", .{ size.x, size.y, size.z });
    }
    if (options.max_registers_per_work_item) |count| try out.print(builder.arena, "\n.maxnreg {d}", .{count});
}

/// A function with the Zig calling convention receives an array or a struct of more than
/// `Layout.max_register_bytes` bytes as a pointer, so that the call needs no copy into registers.
/// The language reference allows a pass by reference, because parameters are immutable. The caller
/// passes memory that no instruction writes during the call.
pub fn passesByReference(layout: *const Layout, cc: std.lang.CallingConvention, ty: Type) bool {
    return cc == .auto and layout.isAggregate(ty) and ty.abiSize(layout.zcu) > Layout.max_register_bytes;
}

/// Returns the type of the `.param` variable that holds a parameter of type `ty`.
pub fn abiParamType(layout: *const Layout, cc: std.lang.CallingConvention, ty: Type) !Type {
    if (!passesByReference(layout, cc, ty)) return ty;
    return layout.pt.singleConstPtrType(ty);
}

pub fn paramName(builder: *Builder, index: u32) ![]const u8 {
    return std.fmt.allocPrint(builder.arena, "param_{d}", .{index});
}

/// ptxas 13.0 rejects a `.param` alignment above this many bytes.
pub const max_param_alignment = 128;

/// ptxas 13.0 rejects `.func` parameters of type predicate, 8 bits or 16 bits. `test/nvptx.zig`
/// checks the byte arrays that the backend declares instead.
fn paramDecl(builder: *Builder, layout: *const Layout, ty: Type, name: []const u8, out: *std.ArrayList(u8)) Layout.Error!void {
    const zcu = layout.zcu;
    const scalars = try layout.scalarsOf(ty);
    const first = if (scalars.len == 1) scalars[0] else null;
    if (first != null and first.?.offset == 0 and first.?.kind.size == ty.abiSize(zcu) and first.?.kind.size >= 4 and first.?.kind.class != .pred) {
        try out.print(builder.arena, ".param .{t} {s}", .{ first.?.kind.memoryType(), name });
    } else {
        const alignment = @min(max_param_alignment, ty.abiAlignment(zcu).toByteUnits().?);
        try out.print(builder.arena, ".param .align {d} .b8 {s}[{d}]", .{ alignment, name, ty.abiSize(zcu) });
    }
}

fn paramDeclInst(builder: *Builder, layout: *const Layout, ty: Type, name: []const u8) Layout.Error!void {
    var decl: std.ArrayList(u8) = .empty;
    try paramDecl(builder, layout, ty, name, &decl);
    const start = builder.here();
    try builder.appendString(decl.items);
    try builder.addInst(.param_decl, start);
}

pub fn declare(builder: *Builder, layout: *const Layout, fn_info: InternPool.Key.FuncType, linkage: []const u8, out: *std.ArrayList(u8)) Layout.Error!u32 {
    const zcu = layout.zcu;
    const ip = &zcu.intern_pool;
    const is_kernel = fn_info.cc == .nvptx_kernel;
    const ret_type: Type = .fromInterned(fn_info.return_type);
    try out.appendSlice(builder.arena, linkage);
    try out.appendSlice(builder.arena, if (is_kernel) ".entry " else ".func ");
    if (!is_kernel and ret_type.hasRuntimeBits(zcu)) {
        try out.append(builder.arena, '(');
        try paramDecl(builder, layout, ret_type, "func_retval0", out);
        try out.appendSlice(builder.arena, ") ");
    }
    const name_offset: u32 = @intCast(out.items.len);
    try out.append(builder.arena, '(');
    var param_index: u32 = 0;
    var param_bytes: u64 = 0;
    for (fn_info.param_types.get(ip)) |param_type_index| {
        const param_type: Type = .fromInterned(param_type_index);
        if (!param_type.hasRuntimeBits(zcu)) continue;
        try out.appendSlice(builder.arena, if (param_index == 0) "\n\t" else ",\n\t");
        try paramDecl(builder, layout, try abiParamType(layout, fn_info.cc, param_type), try paramName(builder, param_index), out);
        param_index += 1;
        param_bytes = param_type.abiAlignment(zcu).forward(param_bytes) + param_type.abiSize(zcu);
    }
    try out.append(builder.arena, ')');
    // Kernel parameters over 4352 bytes in total need PTX 8.1 (PTX ISA 9.4, section 11.2.1).
    if (is_kernel and param_bytes > 4352) builder.needPtx(81);
    return name_offset;
}

pub const Callee = union(enum) {
    target: Mir.Target,
    reg: Mir.Reg,
};

pub const Arg = struct {
    ty: Type,
    parts: []const Builder.Part,
};

pub fn callCompilerRt(builder: *Builder, layout: *const Layout, function: Air.CompilerRtFunc, args: []const Arg) Layout.Error![]const Builder.Part {
    const zcu = layout.zcu;
    const param_types = try builder.arena.alloc(InternPool.Index, args.len);
    for (param_types, args) |*param_type, arg| param_type.* = arg.ty.toIntern();
    const ret_type = function.returnType();
    const abi_ret_type: Type = if (returnsOrder(function)) .fromInterned(.c_long_type) else ret_type;
    const fn_type = try layout.pt.funcType(.{
        .param_types = param_types,
        .return_type = abi_ret_type.toIntern(),
        .cc = function.@"callconv"(layout.target),
    });
    const name = try zcu.intern_pool.getOrPutString(layout.gpa, zcu.comp.io, layout.pt.tid, function.name(layout.target), .no_embedded_nulls);
    const callee: Callee = .{ .target = .{ .extern_func = .{ .name = name, .ty = fn_type.toIntern() } } };
    const result = try genCall(builder, layout, callee, abi_ret_type, args, false);
    if (abi_ret_type.toIntern() == ret_type.toIntern()) return result;
    const from = try layout.kindOf(abi_ret_type);
    const converted = try int.intConvert(builder, try builder.toReg(result[0], from), from, try layout.kindOf(ret_type));
    return builder.regParts(&.{converted});
}

/// On nvptx targets, compiler_rt returns these float comparisons as a C `long`. ptxas rejects a
/// call when its return type differs from the return type of the callee.
fn returnsOrder(function: Air.CompilerRtFunc) bool {
    return switch (function) {
        .__eqhf2, .__eqsf2, .__eqdf2, .__eqxf2, .__eqtf2 => true,
        .__nehf2, .__nesf2, .__nedf2, .__nexf2, .__netf2 => true,
        .__lthf2, .__ltsf2, .__ltdf2, .__ltxf2, .__lttf2 => true,
        .__lehf2, .__lesf2, .__ledf2, .__lexf2, .__letf2 => true,
        .__gthf2, .__gtsf2, .__gtdf2, .__gtxf2, .__gttf2 => true,
        .__gehf2, .__gesf2, .__gedf2, .__gexf2, .__getf2 => true,
        else => false,
    };
}

pub fn genCall(builder: *Builder, layout: *const Layout, callee: Callee, ret_type: Type, args: []const Arg, never_inline: bool) Layout.Error![]const Builder.Part {
    const zcu = layout.zcu;
    const has_ret = ret_type.hasRuntimeBits(zcu);
    try builder.addInst(.call_begin, 0);
    var arg_count: u16 = 0;
    var prototype_params: std.ArrayList(u8) = .empty;
    for (args) |arg| {
        const arg_type = arg.ty;
        if (!arg_type.hasRuntimeBits(zcu)) continue;
        const name = try std.fmt.allocPrint(builder.arena, "param{d}", .{arg_count});
        try paramDeclInst(builder, layout, arg_type, name);
        if (arg_count != 0) try prototype_params.appendSlice(builder.arena, ", ");
        try paramDecl(builder, layout, arg_type, "_", &prototype_params);
        try memory.storeMem(builder, .{ .name = name }, .param, arg_type.abiAlignment(zcu).toByteUnits().?, try layout.scalarsOf(arg_type), arg.parts, false);
        arg_count += 1;
    }
    if (has_ret) try paramDeclInst(builder, layout, ret_type, "retval0");
    const prototype_label: u32 = if (callee == .reg) builder.newLabel() else 0;
    if (callee == .reg) {
        // A call through a pointer names a `.callprototype` (PTX ISA 9.4, section 9.7.14.5).
        var prototype_ret: std.ArrayList(u8) = .empty;
        if (has_ret) {
            try prototype_ret.append(builder.arena, '(');
            try paramDecl(builder, layout, ret_type, "_", &prototype_ret);
            try prototype_ret.appendSlice(builder.arena, ") ");
        }
        const params_text = if (arg_count == 0) "" else try std.fmt.allocPrint(builder.arena, " ({s})", .{prototype_params.items});
        const start = builder.here();
        try builder.appendArg(Mir.Label{ .index = prototype_label });
        try builder.appendString(prototype_ret.items);
        try builder.appendString(params_text);
        try builder.addInst(.callprototype, start);
    }
    const start = builder.here();
    try builder.extra.append(builder.arena, @bitCast(Mir.Call{
        .arg_count = arg_count,
        .has_ret = has_ret,
        .indirect = callee == .reg,
        .never_inline = never_inline,
    }));
    switch (callee) {
        .target => |target| try builder.appendArg(Mir.Symbol{ .target = target, .offset = 0 }),
        .reg => |reg| {
            try builder.appendArg(reg);
            try builder.extra.append(builder.arena, prototype_label);
        },
    }
    try builder.addInst(.call, start);
    var result: []const Builder.Part = &.{};
    if (has_ret) {
        const ret_alignment = ret_type.abiAlignment(zcu).toByteUnits().?;
        result = try builder.regParts(try memory.loadRegs(builder, .{ .name = "retval0" }, .param, ret_alignment, try layout.scalarsOf(ret_type), false, false));
    }
    try builder.addInst(.call_end, 0);
    if (ret_type.isNoReturn(zcu)) try builder.trap();
    return result;
}
