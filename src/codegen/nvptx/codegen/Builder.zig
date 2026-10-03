//! This file builds the MIR of one function in SSA form. Each emit function returns new registers,
//! so each register has one definition. The builder also collects PTX text and its relocations.

const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const Mir = @import("../Mir.zig");
const Kind = @import("Kind.zig");
const Layout = @import("Layout.zig");
const Builder = @This();

arena: Allocator,
text: std.ArrayList(u8) = .empty,
relocs: std.ArrayList(Mir.Reloc) = .empty,
insts: std.MultiArrayList(Mir.Inst) = .empty,
extra: std.ArrayList(u32) = .empty,
strings: std.ArrayList(u8) = .empty,
string_offsets: std.StringHashMapUnmanaged(u32) = .empty,
symbols: std.ArrayList(Mir.Symbol) = .empty,
locals: std.ArrayList(Mir.Local) = .empty,
reg_counts: std.enums.EnumArray(Mir.Class, u32) = .initFill(0),
next_label: u32 = 0,
ptx_version: u16 = 0,
is_kernel: bool = false,
/// The builder clears this flag after an instruction that never passes control to the next one.
reachable: bool = true,
trampolines: std.ArrayList(Trampoline) = .empty,

pub const SymbolPointer = struct {
    target: Mir.Target,
    offset: u64,
    space: Layout.Space,
    /// Only a `.generic` pointer needs `cvta` or `generic()` to reach the symbol.
    address_space: std.lang.AddressSpace,
};

pub const Part = union(enum) {
    reg: Mir.Reg,
    imm: u64,
    symbol: SymbolPointer,
    undef,
};

/// Each branch to a block passes one argument for each parameter of the block.
pub const Block = struct {
    label: u32,
    params: []const Mir.Reg,
};

/// A guarded branch cannot pass arguments. A guarded branch to a block with parameters therefore
/// jumps to `label`. The code at `label` passes `args` to `block` with an unguarded branch.
const Trampoline = struct {
    label: u32,
    block: Block,
    args: []const Mir.Reg,
};

/// The base address of a memory access is a register or the name of a `.param` variable.
pub const Base = union(enum) {
    reg: Mir.Reg,
    name: []const u8,
};

pub fn print(builder: *Builder, comptime format: []const u8, args: anytype) !void {
    try builder.text.print(builder.arena, format, args);
}

pub fn emitSymbol(builder: *Builder, target: Mir.Target) !void {
    try builder.relocs.append(builder.arena, .{ .offset = @intCast(builder.text.items.len), .target = target });
}

pub fn needPtx(builder: *Builder, version: u16) void {
    builder.ptx_version = @max(builder.ptx_version, version);
}

pub fn here(builder: *Builder) u32 {
    return @intCast(builder.extra.items.len);
}

fn newReg(builder: *Builder, class: Mir.Class) Mir.Reg {
    const count = builder.reg_counts.getPtr(class);
    defer count.* += 1;
    return .{ .class = class, .index = count.* };
}

/// Returns a register without a definition. A branch argument without a definition moves nothing.
fn undefinedReg(builder: *Builder, class: Mir.Class) Mir.Reg {
    return builder.newReg(class);
}

pub fn newLabel(builder: *Builder) u32 {
    defer builder.next_label += 1;
    return builder.next_label;
}

pub fn emit(builder: *Builder, class: Mir.Class, tag: Mir.Inst.Tag, ptx_type: Mir.Type, modifiers: Mir.Inst.Modifiers, args: anytype) Layout.Error!Mir.Reg {
    assert(modifiers.guard() == .none);
    const start = builder.here();
    const result = builder.newReg(class);
    try builder.appendArg(result);
    inline for (args) |arg| try builder.appendArg(arg);
    try builder.insts.append(builder.arena, .{ .tag = tag, .ty = ptx_type, .modifiers = modifiers, .data = start });
    return result;
}

/// The new register holds the result when the guard of `modifiers` holds, and `otherwise` if not.
pub fn emitGuarded(builder: *Builder, pred: Mir.Reg, otherwise: Mir.Reg, tag: Mir.Inst.Tag, ptx_type: Mir.Type, modifiers: Mir.Inst.Modifiers, args: anytype) Layout.Error!Mir.Reg {
    assert(modifiers.guard() != .none);
    const start = builder.here();
    try builder.appendArg(pred);
    const result = builder.newReg(otherwise.class);
    try builder.appendArg(result);
    inline for (args) |arg| try builder.appendArg(arg);
    try builder.appendArg(otherwise);
    try builder.insts.append(builder.arena, .{ .tag = tag, .ty = ptx_type, .modifiers = modifiers, .data = start });
    return result;
}

pub fn emitList(builder: *Builder, classes: []const Mir.Class, tag: Mir.Inst.Tag, ptx_type: Mir.Type, modifiers: Mir.Inst.Modifiers, args: anytype) Layout.Error![]const Mir.Reg {
    const results = try builder.newRegs(classes);
    const start = builder.here();
    try builder.appendRegList(results);
    inline for (args) |arg| try builder.appendArg(arg);
    try builder.insts.append(builder.arena, .{ .tag = tag, .ty = ptx_type, .modifiers = modifiers, .data = start });
    return results;
}

pub fn emitEffect(builder: *Builder, tag: Mir.Inst.Tag, ptx_type: Mir.Type, modifiers: Mir.Inst.Modifiers, args: anytype) Layout.Error!void {
    const start = builder.here();
    inline for (args) |arg| try builder.appendArg(arg);
    try builder.insts.append(builder.arena, .{ .tag = tag, .ty = ptx_type, .modifiers = modifiers, .data = start });
}

pub fn addInst(builder: *Builder, tag: Mir.Inst.Tag, data: u32) !void {
    try builder.insts.append(builder.arena, .{ .tag = tag, .data = data });
}

pub fn newRegs(builder: *Builder, classes: []const Mir.Class) ![]const Mir.Reg {
    const regs = try builder.arena.alloc(Mir.Reg, classes.len);
    for (regs, classes) |*reg, class| reg.* = builder.newReg(class);
    return regs;
}

pub fn paramLoad(builder: *Builder, name: []const u8, offset: u64, mem: Mir.MemType, classes: []const Mir.Class) ![]const Mir.Reg {
    const regs = try builder.newRegs(classes);
    try builder.paramAccess(.param_load, name, offset, mem, regs);
    return regs;
}

pub fn paramStore(builder: *Builder, name: []const u8, offset: u64, mem: Mir.MemType, regs: []const Mir.Reg) !void {
    try builder.paramAccess(.param_store, name, offset, mem, regs);
}

fn paramAccess(builder: *Builder, tag: Mir.Inst.Tag, name: []const u8, offset: u64, mem: Mir.MemType, regs: []const Mir.Reg) !void {
    const start = builder.here();
    const param = paramOperand(name);
    try builder.extra.append(builder.arena, @bitCast(Mir.ParamAccess{
        .offset = @intCast(offset),
        .mem = mem,
        .len = @intCast(regs.len),
        .invariant = builder.is_kernel and tag == .param_load and param.paramRef() == .own,
    }));
    try builder.extra.append(builder.arena, @bitCast(param));
    for (regs) |reg| try builder.extra.append(builder.arena, @bitCast(try regOperand(reg)));
    try builder.addInst(tag, start);
}

pub fn appendArg(builder: *Builder, arg: anytype) Layout.Error!void {
    const T = @TypeOf(arg);
    if (T == Mir.Reg) return builder.extra.append(builder.arena, @bitCast(try regOperand(arg)));
    if (T == Mir.Label) return builder.extra.append(builder.arena, @bitCast(Mir.Operand{ .kind = .label, .payload = @intCast(arg.index) }));
    if (T == Mir.LocalRef) return builder.extra.append(builder.arena, @bitCast(Mir.Operand{ .kind = .local, .payload = @intCast(arg.index) }));
    if (T == Mir.SpecialRegRef) return builder.extra.append(builder.arena, @bitCast(Mir.Operand{
        .kind = .special_reg,
        .payload = (@as(u28, @backingInt(arg.reg)) << 2) | arg.dimension,
    }));
    if (T == Mir.Source) return switch (arg) {
        .reg => |reg| builder.appendArg(reg),
        .imm => |imm| builder.appendArg(imm),
    };
    if (T == Mir.Imm) {
        if (arg.format == .unsigned) return builder.appendImm(arg.value);
        return builder.extra.appendSlice(builder.arena, &.{
            @bitCast(Mir.Operand{ .kind = .imm64, .payload = @backingInt(arg.format) }),
            @truncate(arg.value),
            @truncate(arg.value >> 32),
        });
    }
    if (T == Base) return switch (arg) {
        .reg => |reg| builder.appendArg(reg),
        .name => |name| builder.extra.append(builder.arena, @bitCast(paramOperand(name))),
    };
    if (T == Mir.RegList) return builder.appendRegList(arg.regs);
    if (T == Mir.MemOperand) return builder.extra.append(builder.arena, @bitCast(arg));
    if (T == Mir.Symbol) {
        const index: u28 = @intCast(builder.symbols.items.len);
        try builder.symbols.append(builder.arena, .{ .target = arg.target, .offset = arg.offset });
        return builder.extra.append(builder.arena, @bitCast(Mir.Operand{ .kind = .symbol, .payload = index }));
    }
    return builder.appendImm(arg);
}

pub fn appendImm(builder: *Builder, value: anytype) !void {
    if (value >= 0 and value < (1 << 28)) {
        return builder.extra.append(builder.arena, @bitCast(Mir.Operand{ .kind = .imm, .payload = @intCast(value) }));
    }
    const bits: u64 = if (value < 0) @bitCast(@as(i64, value)) else @intCast(value);
    try builder.extra.appendSlice(builder.arena, &.{
        @bitCast(Mir.Operand{ .kind = .imm64, .payload = @intFromBool(value < 0) }),
        @truncate(bits),
        @truncate(bits >> 32),
    });
}

pub fn appendString(builder: *Builder, bytes: []const u8) !void {
    const gop = try builder.string_offsets.getOrPut(builder.arena, bytes);
    if (!gop.found_existing) {
        gop.value_ptr.* = @intCast(builder.strings.items.len);
        try builder.strings.appendSlice(builder.arena, bytes);
        gop.key_ptr.* = try builder.arena.dupe(u8, bytes);
    }
    try builder.extra.appendSlice(builder.arena, &.{
        @bitCast(Mir.Operand{ .kind = .string, .payload = @intCast(gop.value_ptr.*) }),
        @intCast(bytes.len),
    });
}

pub fn appendRegList(builder: *Builder, regs: []const Mir.Reg) !void {
    try builder.extra.append(builder.arena, @bitCast(Mir.Operand{ .kind = .list, .payload = @intCast(regs.len) }));
    for (regs) |reg| try builder.extra.append(builder.arena, @bitCast(try regOperand(reg)));
}

fn regOperand(reg: Mir.Reg) error{TooManyRegisters}!Mir.Operand {
    const index = std.math.cast(u16, reg.index) orelse return error.TooManyRegisters;
    return .reg(reg.class, index);
}

fn paramOperand(name: []const u8) Mir.Operand {
    if (std.mem.eql(u8, name, "func_retval0")) return .param(.own_ret, 0);
    if (std.mem.eql(u8, name, "retval0")) return .param(.arg_ret, 0);
    if (std.mem.startsWith(u8, name, "param_")) return .param(.own, std.fmt.parseInt(u16, name["param_".len..], 10) catch unreachable);
    return .param(.arg, std.fmt.parseInt(u16, name["param".len..], 10) catch unreachable);
}

/// The label instruction of the new block defines its parameters.
pub fn newBlock(builder: *Builder, classes: []const Mir.Class) !Block {
    return .{ .label = builder.newLabel(), .params = try builder.newRegs(classes) };
}

pub fn labelBlock(builder: *Builder) Block {
    return .{ .label = builder.newLabel(), .params = &.{} };
}

/// Control falls into a block only when the block has no parameters.
pub fn startBlock(builder: *Builder, block: Block) ![]const Mir.Reg {
    assert(!builder.reachable or block.params.len == 0);
    const start = builder.here();
    try builder.appendArg(Mir.Label{ .index = block.label });
    try builder.appendRegList(block.params);
    try builder.addInst(.label, start);
    builder.reachable = true;
    return block.params;
}

pub fn branch(builder: *Builder, block: Block, args: []const Mir.Reg) !void {
    assert(args.len == block.params.len);
    const start = builder.here();
    try builder.appendArg(Mir.Label{ .index = block.label });
    try builder.appendRegList(args);
    try builder.addInst(.bra, start);
    try builder.endBlock();
}

/// Branches to `block` when `pred` is true. When `negate` is set, branches when `pred` is false.
pub fn branchIf(builder: *Builder, pred: Mir.Reg, negate: bool, block: Block, args: []const Mir.Reg) !void {
    assert(args.len == block.params.len);
    const target = if (block.params.len == 0) block.label else target: {
        const label = builder.newLabel();
        try builder.trampolines.append(builder.arena, .{ .label = label, .block = block, .args = try builder.arena.dupe(Mir.Reg, args) });
        break :target label;
    };
    try builder.emitEffect(.bra, .none, .{ .none = .{ .guard = if (negate) .negative else .positive } }, .{ pred, Mir.Label{ .index = target } });
}

pub fn ret(builder: *Builder) !void {
    try builder.addInst(.ret, 0);
    try builder.endBlock();
}

pub fn trap(builder: *Builder) !void {
    try builder.emitEffect(.trap, .none, .{ .none = .{} }, .{});
    try builder.endBlock();
}

/// Places the code of each pending `Trampoline` after an unconditional branch, a `ret` or a `trap`.
fn endBlock(builder: *Builder) !void {
    builder.reachable = false;
    while (builder.trampolines.pop()) |pending| {
        const label_start = builder.here();
        try builder.appendArg(Mir.Label{ .index = pending.label });
        try builder.appendRegList(&.{});
        try builder.addInst(.label, label_start);
        const branch_start = builder.here();
        try builder.appendArg(Mir.Label{ .index = pending.block.label });
        try builder.appendRegList(pending.args);
        try builder.addInst(.bra, branch_start);
    }
}

pub fn toReg(builder: *Builder, part: Part, kind: Kind) Layout.Error!Mir.Reg {
    return switch (part) {
        .reg => |reg| reg,
        .imm => |imm| builder.movImm(kind.class, imm),
        .undef => builder.movImm(kind.class, 0),
        .symbol => |sym| {
            // `cvta` converts a state space address to a generic one. `mov` yields a function
            // address with no conversion (PTX ISA 9.4, section 6.4.4).
            const ref: Mir.Symbol = .{ .target = sym.target, .offset = sym.offset };
            if (sym.address_space == .generic and sym.space != .func) {
                const space = std.meta.stringToEnum(Mir.Space, @tagName(sym.space)).?;
                return builder.emit(.b64, .cvta, .u64, .{ .cvta = .{ .space = space, .uncommitted = isInferable(space) } }, .{ref});
            }
            return builder.emit(kind.class, .mov, kind.class.unsignedType(), .{ .none = .{} }, .{ref});
        },
    };
}

/// Returns `part` as a branch argument. An undefined operand needs no instruction.
pub fn toArg(builder: *Builder, part: Part, kind: Kind) Layout.Error!Mir.Reg {
    return if (part == .undef) builder.undefinedReg(kind.class) else builder.toReg(part, kind);
}

pub fn toRegs(builder: *Builder, parts: []const Part, scalars: []const Layout.Scalar) Layout.Error![]const Mir.Reg {
    const regs = try builder.arena.alloc(Mir.Reg, parts.len);
    for (regs, parts, scalars) |*reg, part, scalar| reg.* = try builder.toReg(part, scalar.kind);
    return regs;
}

pub fn movImm(builder: *Builder, class: Mir.Class, imm: u64) Layout.Error!Mir.Reg {
    return switch (class) {
        .pred => builder.emit(.pred, .setp, .u32, .{ .setp = .{ .cmp = .ne } }, .{ imm, 0 }),
        .f32 => builder.emit(.f32, .mov, .b32, .{ .none = .{} }, .{Mir.Imm{ .value = @as(u32, @truncate(imm)), .format = .f32 }}),
        .f64 => builder.emit(.f64, .mov, .b64, .{ .none = .{} }, .{Mir.Imm{ .value = imm, .format = .f64 }}),
        .b16 => builder.emit(.b16, .mov, .b16, .{ .none = .{} }, .{@as(u16, @truncate(imm))}),
        .b32 => builder.emit(.b32, .mov, .b32, .{ .none = .{} }, .{@as(u32, @truncate(imm))}),
        .b64 => builder.emit(.b64, .mov, .b64, .{ .none = .{} }, .{imm}),
    };
}

pub fn toSource(builder: *Builder, part: Part, kind: Kind) Layout.Error!Mir.Source {
    const value: u64 = switch (part) {
        .imm => |imm| imm,
        .undef => 0,
        .reg, .symbol => return .{ .reg = try builder.toReg(part, kind) },
    };
    if (kind.class == .pred or (kind.repr == .float and kind.class == .b16)) return .{ .reg = try builder.toReg(part, kind) };
    if (kind.repr == .float) return .{ .imm = switch (kind.class) {
        .f32 => .{ .value = @as(u32, @truncate(value)), .format = .f32 },
        else => .{ .value = value, .format = .f64 },
    } };
    const bits = @as(u32, kind.size) * 8;
    const truncated = if (bits >= 64) value else value & ((@as(u64, 1) << @intCast(bits)) - 1);
    if (kind.repr != .signed) return .{ .imm = .{ .value = truncated, .format = .unsigned } };
    const shift: u6 = @intCast(64 - bits);
    const extended: i64 = @as(i64, @bitCast(truncated << shift)) >> shift;
    return .{ .imm = .{ .value = @bitCast(extended), .format = .signed } };
}

pub fn sourceReg(builder: *Builder, src: Mir.Source, kind: Kind) Layout.Error!Mir.Reg {
    return switch (src) {
        .reg => |reg| reg,
        .imm => |imm| builder.movImm(kind.class, imm.value),
    };
}

pub fn regParts(builder: *Builder, regs: []const Mir.Reg) ![]const Part {
    const parts = try builder.arena.alloc(Part, regs.len);
    for (parts, regs) |*part, reg| part.* = .{ .reg = reg };
    return parts;
}

/// `kind` is the kind of a pointer into `address_space`.
pub fn newLocal(builder: *Builder, size: u64, alignment: u64, address_space: std.lang.AddressSpace, kind: Kind) !Mir.Reg {
    const index: u32 = @intCast(builder.locals.items.len);
    try builder.locals.append(builder.arena, .{ .size = @max(size, 1), .alignment = alignment });
    if (address_space == .local) return builder.emit(kind.class, .mov, kind.class.unsignedType(), .{ .none = .{} }, .{Mir.LocalRef{ .index = index }});
    return builder.emit(.b64, .cvta, .u64, .{ .cvta = .{ .space = .local, .uncommitted = true } }, .{Mir.LocalRef{ .index = index }});
}

/// Returns true when `transforms/infer_addrspace.zig` can give this state space to generic
/// accesses.
pub fn isInferable(space: Mir.Space) bool {
    return space == .global or space == .shared or space == .local;
}

/// `name` is the relocation of the function name in `prefix`.
pub fn finish(builder: *Builder, gpa: Allocator, kind: Mir.Kind, prototype_len: u32, prefix: []const u8, name: ?Mir.Reloc) Allocator.Error!Mir {
    const text = try gpa.alloc(u8, prefix.len + builder.text.items.len);
    errdefer gpa.free(text);
    @memcpy(text[0..prefix.len], prefix);
    @memcpy(text[prefix.len..], builder.text.items);
    const first = @intFromBool(name != null);
    const relocs = try gpa.alloc(Mir.Reloc, first + builder.relocs.items.len);
    errdefer gpa.free(relocs);
    if (name) |reloc| relocs[0] = reloc;
    for (relocs[first..], builder.relocs.items) |*dst, src| dst.* = .{
        .offset = src.offset + @as(u32, @intCast(prefix.len)),
        .target = src.target,
    };
    var mir: Mir = .{
        .kind = kind,
        .text = text,
        .prototype_len = prototype_len,
        .relocs = relocs,
        .ptx_version = builder.ptx_version,
    };
    errdefer mir.deinit(gpa);
    if (kind == .data) return mir;
    var instructions: std.MultiArrayList(Mir.Inst) = .empty;
    try instructions.ensureTotalCapacity(gpa, builder.insts.len);
    for (0..builder.insts.len) |i| instructions.appendAssumeCapacity(builder.insts.get(i));
    mir.instructions = instructions.toOwnedSlice();
    mir.extra = try gpa.dupe(u32, builder.extra.items);
    mir.strings = try gpa.dupe(u8, builder.strings.items);
    mir.symbols = try gpa.dupe(Mir.Symbol, builder.symbols.items);
    mir.locals = try gpa.dupe(Mir.Local, builder.locals.items);
    for (std.enums.values(Mir.Class)) |class| mir.reg_counts[@backingInt(class)] = builder.reg_counts.get(class);
    mir.label_count = builder.next_label;
    return mir;
}
