//! This file lowers comptime-known values to the operands of their scalars. It also writes the
//! PTX declarations of global variables, constants and name tables.

const std = @import("std");

const InternPool = @import("../../../InternPool.zig");
const Type = @import("../../../Type.zig");
const Value = @import("../../../Value.zig");
const codegen = @import("../../../codegen.zig");
const Mir = @import("../Mir.zig");
const Kind = @import("Kind.zig");
const Builder = @import("Builder.zig");
const Layout = @import("Layout.zig");
const memory = @import("memory.zig");

pub fn constParts(builder: *Builder, layout: *const Layout, list: *std.ArrayList(Builder.Part), ty: Type, val: Value) Layout.Error!void {
    const zcu = layout.zcu;
    const ip = &zcu.intern_pool;
    if (!ty.hasRuntimeBits(zcu)) return;
    if (val.isUndef(zcu)) {
        for (try layout.scalarsOf(ty)) |_| try list.append(builder.arena, .undef);
        return;
    }
    switch (ty.zigTypeTag(zcu)) {
        .bool => try list.append(builder.arena, .{ .imm = @intFromBool(val.toBool()) }),
        .float => {
            const float_bits = ty.floatBits(layout.target);
            const repr: u128 = switch (float_bits) {
                16 => @as(u16, @bitCast(val.toFloat(f16, zcu))),
                32 => @as(u32, @bitCast(val.toFloat(f32, zcu))),
                64 => @as(u64, @bitCast(val.toFloat(f64, zcu))),
                80 => @as(u80, @bitCast(val.toFloat(f80, zcu))),
                128 => @bitCast(val.toFloat(f128, zcu)),
                else => unreachable,
            };
            try list.append(builder.arena, .{ .imm = @truncate(repr) });
            if (float_bits > 64) try list.append(builder.arena, .{ .imm = @intCast(repr >> 64) });
        },
        .int, .@"enum" => {
            var space: Value.BigIntSpace = undefined;
            const big = val.toBigInt(&space, zcu);
            const bytes = try builder.arena.alloc(u8, @intCast(@max(8, ty.abiSize(zcu))));
            big.writeTwosComplement(bytes, .little);
            for (0..try layout.scalarCount(ty)) |i| {
                try list.append(builder.arena, .{ .imm = std.mem.readInt(u64, bytes[i * 8 ..][0..8], .little) });
            }
        },
        .error_set => {
            const name = switch (ip.indexToKey(val.toIntern())) {
                .err => |err| err.name,
                else => unreachable,
            };
            try list.append(builder.arena, .{ .imm = try layout.pt.getErrorValue(name) });
        },
        .pointer => switch (ip.indexToKey(val.toIntern())) {
            .slice => |slice| {
                try constParts(builder, layout, list, ty.slicePtrFieldType(zcu), .fromInterned(slice.ptr));
                try list.append(builder.arena, .{ .imm = Value.fromInterned(slice.len).toUnsignedInt(zcu) });
            },
            else => try list.append(builder.arena, try constPtr(builder, layout, val.toIntern(), 0)),
        },
        .optional => {
            const payload_type = ty.optionalChild(zcu);
            const payload = val.optionalValue(zcu);
            if (ty.optionalReprIsPayload(zcu)) {
                if (payload) |payload_value| return constParts(builder, layout, list, payload_type, payload_value);
                // The length of a null slice is undefined.
                const null_value = if (payload_type.zigTypeTag(zcu) == .pointer) ty.nullPtrAddress(zcu) else 0;
                try list.append(builder.arena, .{ .imm = null_value });
                for ((try layout.scalarsOf(payload_type))[1..]) |_| try list.append(builder.arena, .undef);
                return;
            }
            if (payload) |payload_value| {
                try constParts(builder, layout, list, payload_type, payload_value);
            } else {
                for (try layout.scalarsOf(payload_type)) |_| try list.append(builder.arena, .undef);
            }
            try list.append(builder.arena, .{ .imm = @intFromBool(payload != null) });
        },
        .error_union => {
            const payload_type = ty.errorUnionPayload(zcu);
            switch (ip.indexToKey(val.toIntern()).error_union.val) {
                .err_name => |name| {
                    try list.append(builder.arena, .{ .imm = try layout.pt.getErrorValue(name) });
                    for (try layout.scalarsOf(payload_type)) |_| try list.append(builder.arena, .undef);
                },
                .payload => |payload| {
                    if (ty.errorUnionSet(zcu).hasRuntimeBits(zcu)) try list.append(builder.arena, .{ .imm = 0 });
                    try constParts(builder, layout, list, payload_type, .fromInterned(payload));
                },
            }
        },
        .@"struct" => switch (ty.containerLayout(zcu)) {
            .@"packed" => try constBytes(builder, layout, list, ty, val),
            .auto, .@"extern" => for (0..ty.structFieldCount(zcu)) |i| {
                if (ty.structFieldIsComptime(i, zcu)) continue;
                try constParts(builder, layout, list, ty.fieldType(i, zcu), try val.fieldValue(layout.pt, i));
            },
        },
        .array, .vector => {
            const elem_type = ty.childType(zcu);
            const len = if (ty.zigTypeTag(zcu) == .array) ty.arrayLen(zcu) else ty.vectorLen(zcu);
            for (0..len) |i| try constParts(builder, layout, list, elem_type, try val.elemValue(layout.pt, i));
            if (ty.zigTypeTag(zcu) == .array) if (ty.sentinel(zcu)) |sentinel| try constParts(builder, layout, list, elem_type, sentinel);
        },
        .@"union" => switch (ty.containerLayout(zcu)) {
            .@"packed" => try constBytes(builder, layout, list, ty, val),
            .auto, .@"extern" => try constUnion(builder, layout, list, ty, val),
        },
        else => return layout.todo("constants of type '{f}'", .{ty.fmt(zcu)}),
    }
}

/// Packed fields share bits, so the constant goes through its bytes in memory to the scalars of
/// `ty`.
fn constBytes(builder: *Builder, layout: *const Layout, list: *std.ArrayList(Builder.Part), ty: Type, val: Value) Layout.Error!void {
    const zcu = layout.zcu;
    const size: usize = @intCast(ty.abiSize(zcu));
    // Eight spare bytes keep the `u64` read of the last scalar in bounds.
    const bytes = try builder.arena.alloc(u8, size + 8);
    @memset(bytes, 0);
    val.writeToMemory(zcu, bytes) catch |err| switch (err) {
        error.OutOfMemory => |e| return e,
        error.ReinterpretDeclRef, error.IllDefinedMemoryLayout => return layout.todo("constant '{f}' with pointers in its memory form", .{ty.fmt(zcu)}),
    };
    for (try layout.scalarsOf(ty)) |scalar| {
        var word: u64 = std.mem.readInt(u64, bytes[scalar.offset..][0..8], .little);
        if (scalar.kind.size < 8) word &= (@as(u64, 1) << @intCast(scalar.kind.size * 8)) - 1;
        try list.append(builder.arena, .{ .imm = word });
    }
}

/// A union constant goes through its bytes. Bytes cannot hold an address, so a pointer stays a
/// symbol only when it fills one whole scalar of the union.
fn constUnion(builder: *Builder, layout: *const Layout, list: *std.ArrayList(Builder.Part), ty: Type, val: Value) Layout.Error!void {
    const zcu = layout.zcu;
    const bytes = try builder.arena.alloc(u8, @intCast(ty.abiSize(zcu)));
    @memset(bytes, 0);
    var symbols: std.AutoHashMapUnmanaged(u32, Builder.SymbolPointer) = .empty;
    if (try layout.unionTagScalar(ty)) |scalar| {
        const tag = val.unionTag(zcu) orelse return layout.todo("a constant '{f}' without a tag", .{ty.fmt(zcu)});
        var tag_parts: std.ArrayList(Builder.Part) = .empty;
        try constParts(builder, layout, &tag_parts, tag.typeOf(zcu), tag);
        try placeConst(builder, bytes, &symbols, &.{scalar}, tag_parts.items, 0);
    }
    const payload = val.unionPayload(zcu);
    const payload_type = payload.typeOf(zcu);
    if (payload_type.hasRuntimeBits(zcu)) {
        var payload_parts: std.ArrayList(Builder.Part) = .empty;
        try constParts(builder, layout, &payload_parts, payload_type, payload);
        try placeConst(builder, bytes, &symbols, try layout.scalarsOf(payload_type), payload_parts.items, layout.unionPayloadOffset(ty));
    }
    var placed: u32 = 0;
    for (try layout.scalarsOf(ty)) |chunk| {
        if (symbols.get(chunk.offset)) |symbol| {
            if (chunk.kind.size != 8) return layout.todo("a pointer in a constant '{f}'", .{ty.fmt(zcu)});
            try list.append(builder.arena, .{ .symbol = symbol });
            placed += 1;
        } else {
            try list.append(builder.arena, .{ .imm = std.mem.readVarInt(u64, bytes[chunk.offset..][0..chunk.kind.size], .little) });
        }
    }
    if (placed != symbols.count()) return layout.todo("an unaligned pointer in a constant '{f}'", .{ty.fmt(zcu)});
}

/// `symbols` records each symbol by its byte offset, because a symbol cannot become bytes.
fn placeConst(builder: *Builder, bytes: []u8, symbols: *std.AutoHashMapUnmanaged(u32, Builder.SymbolPointer), scalars: []const Layout.Scalar, parts: []const Builder.Part, base: u32) Layout.Error!void {
    for (try memory.memoryForm(builder, scalars), try memory.packVectors(builder, scalars, parts)) |scalar, part| switch (part) {
        .imm => |imm| {
            var word: [8]u8 = undefined;
            std.mem.writeInt(u64, &word, imm, .little);
            @memcpy(bytes[base + scalar.offset ..][0..scalar.kind.size], word[0..scalar.kind.size]);
        },
        .symbol => |symbol| try symbols.put(builder.arena, base + scalar.offset, symbol),
        .undef, .reg => {},
    };
}

fn constPtr(builder: *Builder, layout: *const Layout, ptr_val: InternPool.Index, prev_offset: u64) Layout.Error!Builder.Part {
    const zcu = layout.zcu;
    const ip = &zcu.intern_pool;
    const ptr = ip.indexToKey(ptr_val).ptr;
    const ptr_type: Type = .fromInterned(ptr.ty);
    // A pointer to a vector element holds the address of the element.
    const elem_offset = switch (ptr_type.ptrInfo(zcu).flags.vector_index) {
        .none => 0,
        _ => |index| if (try layout.bitPackedHost(ptr_type) != null) 0 else @backingInt(index) * ptr_type.childType(zcu).abiSize(zcu),
    };
    const offset = prev_offset + ptr.byte_offset + elem_offset;
    const address_space = ptr_type.ptrAddressSpace(zcu);
    switch (ptr.base_addr) {
        .nav => |nav| {
            const resolved = ip.getNav(nav).resolved.?;
            const space: Layout.Space = if (ip.isFunctionType(resolved.type)) .func else Layout.Space.fromAddressSpace(resolved.@"addrspace") orelse
                return layout.todo("pointer to address space '{t}'", .{resolved.@"addrspace"});
            return .{ .symbol = .{ .target = .{ .nav = nav }, .offset = offset, .space = space, .address_space = address_space } };
        },
        .uav => |uav| return .{ .symbol = .{ .target = .{ .uav = uav.val }, .offset = offset, .space = .global, .address_space = address_space } },
        .int => return .{ .imm = offset },
        .eu_payload => |error_union_ptr| {
            const payload_type = Value.fromInterned(error_union_ptr).typeOf(zcu).childType(zcu).errorUnionPayload(zcu);
            return constPtr(builder, layout, error_union_ptr, offset + codegen.errUnionPayloadOffset(payload_type, zcu));
        },
        .opt_payload => |optional_ptr| return constPtr(builder, layout, optional_ptr, offset),
        .field => |field| {
            const aggregate_type = Value.fromInterned(field.base).typeOf(zcu).childType(zcu);
            return constPtr(builder, layout, field.base, offset + try layout.fieldOffset(aggregate_type, @intCast(field.index)));
        },
        .arr_elem => |arr_elem| {
            const elem_size = Value.fromInterned(arr_elem.base).typeOf(zcu).childType(zcu).abiSize(zcu);
            return constPtr(builder, layout, arr_elem.base, offset + elem_size * arr_elem.index);
        },
        .comptime_field, .comptime_alloc => unreachable,
    }
}

/// A null `maybe_val` declares an extern variable.
pub fn writeData(builder: *Builder, layout: *const Layout, target: Mir.Target, ty: Type, maybe_val: ?Value, space: Layout.Space, alignment: u64) Layout.Error!void {
    const zcu = layout.zcu;
    try builder.print("{s}.{t} .align {d} ", .{ if (maybe_val == null) ".extern " else "", space, alignment });
    const size = if (ty.hasRuntimeBits(zcu)) ty.abiSize(zcu) else 0;
    const val = maybe_val orelse return declareBytes(builder, target, size);
    // A definition without an initializer needs an array length (PTX ISA 9.4, section 5.4.3).
    if (size == 0) return declareBytes(builder, target, 1);
    if (val.isUndef(zcu)) return declareBytes(builder, target, size);
    if (space == .shared or space == .local) return layout.fail("a variable in '.{t}' cannot have an initial value", .{space});
    var list: std.ArrayList(Builder.Part) = .empty;
    try constParts(builder, layout, &list, ty, val);
    const parts = try memory.packVectors(builder, try layout.scalarsOf(ty), list.items);
    const scalars = try memory.memoryForm(builder, try layout.scalarsOf(ty));
    const bytes = try builder.arena.alloc(u8, std.mem.alignForward(u64, size, 8));
    @memset(bytes, 0);
    var symbols: std.AutoHashMapUnmanaged(u32, Builder.SymbolPointer) = .empty;
    for (parts, scalars) |part, scalar| switch (part) {
        .imm => |imm| {
            var word: [8]u8 = undefined;
            std.mem.writeInt(u64, &word, imm, .little);
            @memcpy(bytes[scalar.offset..][0..scalar.kind.size], word[0..scalar.kind.size]);
        },
        .symbol => |symbol| {
            if (scalar.offset % 8 != 0 or alignment < 8) return layout.todo("unaligned pointer in a global", .{});
            try symbols.put(builder.arena, scalar.offset, symbol);
        },
        .undef, .reg => {},
    };
    if (symbols.count() == 0) {
        try builder.print(".b8 ", .{});
        try builder.emitSymbol(target);
        try builder.print("[{d}] = {{", .{size});
        for (bytes[0..@intCast(size)], 0..) |byte, i| try builder.print("{s}{d}", .{ if (i == 0) "" else ",", byte });
        try builder.print("}};\n", .{});
        return;
    }
    try builder.print(".u64 ", .{});
    try builder.emitSymbol(target);
    try builder.print("[{d}] = {{", .{bytes.len / 8});
    var offset: u32 = 0;
    while (offset < bytes.len) : (offset += 8) {
        if (offset != 0) try builder.print(",", .{});
        const symbol = symbols.get(offset) orelse {
            try builder.print("{d}", .{std.mem.readInt(u64, bytes[offset..][0..8], .little)});
            continue;
        };
        // A function name in an initializer stands for the address of the function. The
        // `generic()` operator applies only to variables (PTX ISA 9.4, section 5.4.4).
        const generic = symbol.address_space == .generic and symbol.space != .func;
        if (generic) try builder.print("generic(", .{});
        try builder.emitSymbol(symbol.target);
        if (generic) try builder.print(")", .{});
        if (symbol.offset != 0) try builder.print("+{d}", .{symbol.offset});
    }
    try builder.print("}};\n", .{});
}

fn declareBytes(builder: *Builder, target: Mir.Target, size: u64) !void {
    try builder.print(".b8 ", .{});
    try builder.emitSymbol(target);
    if (size == 0) try builder.print("[];\n", .{}) else try builder.print("[{d}];\n", .{size});
}

/// Each entry of a name table holds a pointer to the name and the length of the name.
pub fn writeNames(builder: *Builder, layout: *const Layout, target: Mir.Target) Layout.Error!void {
    const ip = &layout.zcu.intern_pool;
    const names, const prefix = switch (target) {
        .tag_names => |enum_type| .{
            ip.loadEnumType(enum_type).field_names.get(ip),
            try std.fmt.allocPrint(builder.arena, "__zig_tag_name_{d}_", .{@backingInt(enum_type)}),
        },
        .error_names => .{ ip.global_error_set.getNamesFromMainThread(), "__zig_error_name_" },
        .nav, .uav, .const_uav, .name, .extern_func => unreachable,
    };
    for (names, 0..) |name, i| {
        try builder.print(".global .align 1 .b8 {s}{d}[{d}] = {{", .{ prefix, i, name.length(ip) + 1 });
        for (name.toSlice(ip)) |byte| try builder.print("{d},", .{byte});
        try builder.print("0}};\n", .{});
    }
    const is_errors = target == .error_names;
    try builder.print(".global .align 8 .u64 ", .{});
    try builder.emitSymbol(target);
    try builder.print("[{d}] = {{", .{(names.len + @intFromBool(is_errors)) * 2});
    if (is_errors) try builder.print("0,{d}", .{names.len});
    for (names, 0..) |name, i| {
        try builder.print("{s}generic({s}{d}),{d}", .{ if (i == 0 and !is_errors) "" else ",", prefix, i, name.length(ip) });
    }
    try builder.print("}};\n", .{});
}
