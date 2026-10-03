const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const InternPool = @import("../../InternPool.zig");
const Mir = @This();

kind: Kind,
text: []u8,
prototype_len: u32,
relocs: []Reloc,
ptx_version: u16,
is_noinline: bool = false,
instructions: std.MultiArrayList(Inst).Slice = .empty,
extra: []u32 = &.{},
strings: []u8 = &.{},
symbols: []Symbol = &.{},
locals: []Local = &.{},
reg_counts: [class_count]u32 = @splat(0),
label_count: u32 = 0,

pub const Kind = enum { entry, func, data };

pub const Reloc = struct {
    offset: u32,
    target: Target,
};

pub const Target = union(enum) {
    nav: InternPool.Nav.Index,
    uav: InternPool.Index,
    /// The linker writes a `.uav` to `.global` memory and a `.const_uav` to `.const` memory.
    const_uav: InternPool.Index,
    tag_names: InternPool.Index,
    error_names,
    /// `Object.read` gives each name of an object an index into `Object.names`. The linker converts
    /// the index to an index into its own names when it loads the object.
    name: u32,
    /// The linker knows this function only by a symbol name and a type. Examples are a compiler_rt
    /// function that the backend calls and a second export name of a function.
    extern_func: ExternFunc,
};

pub const ExternFunc = struct {
    name: InternPool.NullTerminatedString,
    /// The linker declares the function with this type when no input defines the function.
    ty: InternPool.Index,
};

pub const empty: Mir = .{
    .kind = .data,
    .text = &.{},
    .prototype_len = 0,
    .relocs = &.{},
    .ptx_version = 0,
};

pub fn deinit(mir: *Mir, gpa: Allocator) void {
    gpa.free(mir.text);
    gpa.free(mir.relocs);
    mir.instructions.deinit(gpa);
    gpa.free(mir.extra);
    gpa.free(mir.strings);
    gpa.free(mir.symbols);
    gpa.free(mir.locals);
    mir.* = undefined;
}

pub const Class = enum(u3) {
    pred,
    b16,
    b32,
    b64,
    f32,
    f64,

    pub fn prefix(class: Class) []const u8 {
        return switch (class) {
            .pred => "%p",
            .b16 => "%rs",
            .b32 => "%r",
            .b64 => "%rd",
            .f32 => "%f",
            .f64 => "%fd",
        };
    }

    pub fn bits(class: Class) u16 {
        return switch (class) {
            .pred => 1,
            .b16 => 16,
            .b32, .f32 => 32,
            .b64, .f64 => 64,
        };
    }

    pub fn bitType(class: Class) Type {
        return switch (class) {
            .pred => .pred,
            .b16 => .b16,
            .b32, .f32 => .b32,
            .b64, .f64 => .b64,
        };
    }

    pub fn unsignedType(class: Class) Type {
        return switch (class.bits()) {
            16 => .u16,
            32 => .u32,
            else => .u64,
        };
    }
};

pub const class_count = @typeInfo(Class).@"enum".field_names.len;

/// Before `Lower.run`, one instruction defines each register.
pub const Reg = struct {
    class: Class,
    index: u32,
};

pub const RegList = struct { regs: []const Reg };

pub const Imm = struct { value: u64, format: ImmFormat };

pub const Source = union(enum) {
    reg: Reg,
    imm: Imm,
};

pub const Label = struct { index: u32 };

pub const LocalRef = struct { index: u32 };

pub const SpecialRegRef = struct { reg: SpecialReg, dimension: u2 };

pub const Local = struct {
    size: u64,
    alignment: u64,
};

/// A local variable that holds a copy of a kernel parameter. When promotion cannot move the
/// variable to registers, it uses the address of the parameter in place of the copy.
pub const ParamCopy = struct { local: u32, param: u32 };

pub const Symbol = struct {
    target: Target,
    offset: u64,
};

pub const Inst = struct {
    tag: Tag,
    ty: Type = .none,
    modifiers: Modifiers = .{ .none = .{} },
    data: u32,

    pub const Tag = enum(u8) {
        label,
        param_load,
        param_store,
        call_begin,
        param_decl,
        call,
        call_end,
        ret,
        @"asm",
        callprototype,

        abs,
        add,
        addc,
        @"and",
        atom,
        bra,
        brev,
        brkpt,
        clz,
        cvt,
        cvta,
        div,
        fence,
        fma,
        ld,
        mad,
        madc,
        max,
        min,
        mov,
        mul,
        neg,
        not,
        @"or",
        popc,
        prmt,
        rem,
        selp,
        setp,
        shl,
        shr,
        sqrt,
        st,
        sub,
        subc,
        trap,
        xor,
        /// `transforms/expand.zig` expands this memory intrinsic into loads and stores at link
        /// time.
        copy,
        /// Keeps ptxas from unrolling the loop whose header block holds the instruction. `Emit`
        /// writes `.pragma "nounroll";` right after the label of the block, where the pragma must
        /// stand (PTX ISA 9.4, section 12.1).
        nounroll,
    };

    pub const Modifiers = packed union {
        none: packed struct(u16) {
            guard: Guard = .none,
            unused: u14 = 0,
        },
        arith: packed struct(u16) {
            guard: Guard = .none,
            mode: MulMode = .none,
            rounding: Rounding = .none,
            ftz: bool = false,
            sat: bool = false,
            /// Writes the carry-out to the condition code register.
            cc: bool = false,
            /// Pointer arithmetic: the result is an address in the object of the pointer operand,
            /// as with `getelementptr` of LLVM. `add` and `sub` take the pointer first, and `mad`
            /// takes it as the addend.
            address: bool = false,
            unused: u4 = 0,
        },
        setp: packed struct(u16) {
            guard: Guard = .none,
            cmp: Cmp,
            bool_op: BoolOp = .none,
            ftz: bool = false,
            unused: u6 = 0,
        },
        cvt: packed struct(u16) {
            guard: Guard = .none,
            rounding: Rounding = .none,
            ftz: bool = false,
            sat: bool = false,
            from: Type,
            unused: u3 = 0,
        },
        cvta: packed struct(u16) {
            guard: Guard = .none,
            space: Space,
            to: bool = false,
            /// `transforms/infer_addrspace.zig` has not resolved this conversion yet.
            uncommitted: bool = false,
            unused: u9 = 0,
        },
        mem: packed struct(u16) {
            guard: Guard = .none,
            vec: Vec = .scalar,
            offset: bool = false,
            unused: u11 = 0,
        },
        atom: packed struct(u16) {
            guard: Guard = .none,
            op: AtomOp,
            noftz: bool = false,
            unused: u9 = 0,
        },
        fence: packed struct(u16) {
            guard: Guard = .none,
            sem: Sem,
            scope: Scope,
            unused: u9 = 0,
        },
        /// The operands are the destination, the source address or the stored value, and the count.
        copy: packed struct(u16) {
            guard: Guard = .none,
            op: CopyOp = .copy,
            unused: u12 = 0,
        },
        /// The assembly writes the first `outputs` operands of `asm`. The next `outputs` operands
        /// give the input that shares a register with each output. An immediate marks an output
        /// without such an input.
        @"asm": packed struct(u16) {
            guard: Guard = .none,
            outputs: u11 = 0,
            is_volatile: bool = false,
            /// The assembly names clobbers, for example `memory`.
            clobbers: bool = false,
            /// At least one input is a pointer, so the assembly can read memory through it.
            pointer_inputs: bool = false,
        },

        pub fn guard(modifiers: Modifiers) Guard {
            return modifiers.none.guard;
        }
    };
};

pub const Type = enum(u5) {
    none,
    pred,
    b8,
    b16,
    b32,
    b64,
    u8,
    u16,
    u32,
    u64,
    s8,
    s16,
    s32,
    s64,
    f16,
    f32,
    f64,
    bf16,
    f16x2,
    bf16x2,
};

pub const Guard = enum(u2) { none, positive, negative };

pub const MulMode = enum(u2) { none, lo, hi, wide };

/// `div`, `rcp` and `sqrt` take `.approx` in the place of a rounding modifier, and `div` also takes
/// `.full` (PTX ISA 9.4, sections 9.7.3.8, 9.7.3.13 and 9.7.3.15).
pub const Rounding = enum(u4) { none, rn, rz, rm, rp, rni, rzi, rmi, rpi, approx, full };

pub const Cmp = enum(u5) { eq, ne, lt, le, gt, ge, lo, ls, hi, hs, equ, neu, ltu, leu, gtu, geu, num, nan };

pub const BoolOp = enum(u2) { none, @"and", @"or", xor };

pub const Space = enum(u3) { generic, global, shared, @"const", local, param };

pub const Sem = enum(u3) { none, @"volatile", relaxed, acquire, release, acq_rel, sc };

pub const Scope = enum(u2) { none, cta, gpu, sys };

pub const Vec = enum(u2) { scalar, v2, v4 };

pub const AtomOp = enum(u4) { add, inc, dec, min, max, @"and", @"or", xor, exch, cas };

/// `move` allows overlapping ranges as `@memmove` does, and `set` stores one value in each unit.
pub const CopyOp = enum(u2) { copy, move, set };

/// The facts of one memory access. Each `ld`, `st` and `atom` keeps one `MemOperand` in `extra`
/// after its operands. A `copy` keeps one for its destination and, unless it is a `set`, one more
/// for its source.
pub const MemOperand = packed struct(u32) {
    space: Space = .generic,
    sem: Sem = .none,
    scope: Scope = .none,
    /// Each access moves `1 << log2_size` bytes. A vector access moves all its elements.
    log2_size: u3,
    /// The address is a multiple of `1 << log2_align` bytes and of the access size.
    log2_align: u3 = 0,
    /// No thread changes the memory while the kernel runs, as for `.const` memory.
    invariant: bool = false,
    unused: u17 = 0,

    pub fn size(mem: MemOperand) u64 {
        return @as(u64, 1) << mem.log2_size;
    }

    pub fn alignment(mem: MemOperand) u64 {
        return @max(mem.size(), @as(u64, 1) << mem.log2_align);
    }

    pub fn sized(mem: MemOperand, bytes: u64) MemOperand {
        var result = mem;
        result.log2_size = @intCast(std.math.log2_int(u64, bytes));
        return result;
    }
};

pub const SpecialReg = enum(u4) { tid, ntid, ctaid, nctaid };

pub const ImmFormat = enum(u28) { unsigned, signed, f32, f64, prmt };

pub const Operand = packed struct(u32) {
    payload: u28,
    kind: OperandKind,

    pub fn reg(class: Class, index: u16) Operand {
        return .{ .kind = .reg, .payload = (@as(u28, @backingInt(class)) << 16) | index };
    }

    pub fn regClass(op: Operand) Class {
        return @fromBackingInt(@intCast(@as(u3, @truncate(op.payload >> 16))));
    }

    pub fn regIndex(op: Operand) u16 {
        return @truncate(op.payload);
    }

    pub fn param(ref: ParamRef, index: u16) Operand {
        return .{ .kind = .param, .payload = (@as(u28, @backingInt(ref)) << 16) | index };
    }

    pub fn paramRef(op: Operand) ParamRef {
        return @fromBackingInt(@intCast(@as(u2, @truncate(op.payload >> 16))));
    }

    pub fn paramIndex(op: Operand) u16 {
        return @truncate(op.payload);
    }

    pub fn len(op: Operand) u32 {
        return switch (op.kind) {
            .imm64 => 3,
            .string => 2,
            .list => 1 + op.payload,
            else => 1,
        };
    }
};

pub const OperandKind = enum(u4) {
    reg,
    label,
    local,
    param,
    symbol,
    imm,
    imm64,
    string,
    list,
    special_reg,
};

pub const ParamRef = enum(u2) {
    own,
    own_ret,
    arg,
    arg_ret,
};

pub const MemType = enum(u3) { u8, s8, b16, b32, b64, f32, f64 };

pub const ParamAccess = packed struct(u32) {
    offset: u24,
    mem: MemType,
    len: u3,
    /// No thread changes the parameter while the kernel runs.
    invariant: bool = false,
    unused: u1 = 0,
};

pub const Call = packed struct(u32) {
    arg_count: u16,
    has_ret: bool,
    indirect: bool,
    never_inline: bool,
    unused: u13 = 0,
};

pub const Builder = struct {
    gpa: Allocator,
    instructions: std.MultiArrayList(Inst) = .empty,
    extra: std.ArrayList(u32) = .empty,
    strings: std.ArrayList(u8) = .empty,
    symbols: std.ArrayList(Symbol) = .empty,
    locals: std.ArrayList(Local) = .empty,
    reg_counts: [class_count]u32,
    label_count: u32,

    pub fn deinit(b: *Builder) void {
        b.instructions.deinit(b.gpa);
        b.extra.deinit(b.gpa);
        b.strings.deinit(b.gpa);
        b.symbols.deinit(b.gpa);
        b.locals.deinit(b.gpa);
    }

    pub fn add(b: *Builder, tag: Inst.Tag, data: u32) !void {
        try b.instructions.append(b.gpa, .{ .tag = tag, .data = data });
    }

    pub fn addTyped(b: *Builder, tag: Inst.Tag, ty: Type, modifiers: Inst.Modifiers, words: []const u32) !void {
        const start = b.here();
        try b.extra.appendSlice(b.gpa, words);
        try b.instructions.append(b.gpa, .{ .tag = tag, .ty = ty, .modifiers = modifiers, .data = start });
    }

    pub fn here(b: *const Builder) u32 {
        return @intCast(b.extra.items.len);
    }

    /// Returns the word of a new register of `class`.
    pub fn newReg(b: *Builder, class: Class) error{TooManyRegisters}!u32 {
        const index = std.math.cast(u16, b.reg_counts[@backingInt(class)]) orelse return error.TooManyRegisters;
        b.reg_counts[@backingInt(class)] += 1;
        return @bitCast(Operand.reg(class, index));
    }

    pub fn addLabel(b: *Builder, label: u32, params: []const u32) Allocator.Error!void {
        const at = b.here();
        try b.extra.appendSlice(b.gpa, &.{
            @bitCast(Operand{ .kind = .label, .payload = @intCast(label) }),
            @bitCast(Operand{ .kind = .list, .payload = @intCast(params.len) }),
        });
        try b.extra.appendSlice(b.gpa, params);
        try b.add(.label, at);
    }

    pub fn addBranch(b: *Builder, label: u32, args: []const u32) Allocator.Error!void {
        const at = b.here();
        try b.extra.appendSlice(b.gpa, &.{
            @bitCast(Operand{ .kind = .label, .payload = @intCast(label) }),
            @bitCast(Operand{ .kind = .list, .payload = @intCast(args.len) }),
        });
        try b.extra.appendSlice(b.gpa, args);
        try b.add(.bra, at);
    }
};

pub const OperandRange = struct { first: u32, count: u32 };

/// Returns log2 of the power of two `alignment`, capped at 7, the largest value of `log2_align`.
pub fn log2Align(alignment: u64) u3 {
    return @intCast(@min(std.math.maxInt(u3), std.math.log2_int(u64, alignment)));
}

pub fn operands(mir: *const Mir, inst: Inst) ?OperandRange {
    const data = inst.data;
    const guard: u32 = @intFromBool(inst.modifiers.guard() != .none);
    const count: u32 = switch (inst.tag) {
        .call => return .{ .first = data + 1, .count = 1 },
        .param_decl => return .{ .first = data, .count = 1 },
        .@"asm" => return .{ .first = data + 1, .count = mir.extra[data] },
        .callprototype => return .{ .first = data, .count = 3 },
        // A `label` holds the label and then the list of block parameters.
        .label => return .{ .first = data, .count = 2 },
        .param_load, .param_store, .call_begin, .call_end, .ret => return null,
        .brkpt, .trap, .fence, .nounroll => 0,
        // A `bra` holds the label. Without a guard, a list of block arguments follows.
        .bra => if (guard == 0) 2 else 1,
        .abs, .brev, .clz, .cvt, .cvta, .mov, .neg, .not, .popc, .sqrt => 2,
        .add, .addc, .@"and", .div, .max, .min, .mul, .@"or", .rem, .shl, .shr, .sub, .subc, .xor => 3,
        .fma, .mad, .madc, .prmt, .selp => 4,
        .setp => if (inst.modifiers.setp.bool_op == .none) 3 else 4,
        .ld, .st => if (inst.modifiers.mem.offset) 3 else 2,
        .copy => 3,
        .atom => if (inst.modifiers.atom.op == .cas) 4 else 3,
    };
    return .{ .first = data, .count = guard + count + elseCount(inst) };
}

/// Returns whether the instruction writes its first operand after the guard. That operand is a
/// register or a register list.
pub fn hasResult(tag: Inst.Tag) bool {
    return switch (tag) {
        .label,
        .param_load,
        .param_store,
        .call_begin,
        .param_decl,
        .call,
        .call_end,
        .ret,
        .@"asm",
        .callprototype,
        .bra,
        .brkpt,
        .fence,
        .st,
        .copy,
        .nounroll,
        .trap,
        => false,
        else => true,
    };
}

/// Returns 1 for a guarded instruction that writes a register. Its last operand gives the value of
/// the result when the guard is false.
pub fn elseCount(inst: Inst) u32 {
    return @intFromBool(inst.modifiers.guard() != .none and hasResult(inst.tag));
}

/// Returns whether removing the instruction changes nothing but its results. Such an instruction
/// writes no memory, orders no memory access, joins no carry chain and transfers no control.
pub fn isRemovable(mir: *const Mir, inst: Inst) bool {
    return switch (inst.tag) {
        .label,
        .bra,
        .ret,
        .trap,
        .brkpt,
        .fence,
        .st,
        .copy,
        .nounroll,
        .atom,
        .call_begin,
        .param_decl,
        .call,
        .call_end,
        .callprototype,
        .param_store,
        .addc,
        .subc,
        .madc,
        => false,
        .param_load => true,
        .ld => !mir.hasOrderedMemoryRef(inst),
        .@"asm" => !inst.modifiers.@"asm".is_volatile and !inst.modifiers.@"asm".clobbers and inst.modifiers.@"asm".outputs > 0,
        .add, .sub, .mad => !inst.modifiers.arith.cc,
        else => true,
    };
}

pub fn labelOf(mir: *const Mir, inst: Inst) u32 {
    return @as(Operand, @bitCast(mir.extra[inst.data])).payload;
}

pub fn branchTarget(mir: *const Mir, inst: Inst) u32 {
    const guard: u32 = @intFromBool(inst.modifiers.guard() != .none);
    return @as(Operand, @bitCast(mir.extra[inst.data + guard])).payload;
}

/// Returns the parameters of a `label`, or no registers for another instruction.
pub fn blockParams(mir: *const Mir, inst: Inst) []const u32 {
    if (inst.tag != .label) return &.{};
    const list: Operand = @bitCast(mir.extra[inst.data + 1]);
    return mir.extra[inst.data + 2 ..][0..list.payload];
}

/// Returns the arguments of a `bra` without a guard.
pub fn branchArgs(mir: *const Mir, inst: Inst) []const u32 {
    const list: Operand = @bitCast(mir.extra[inst.data + 1]);
    return mir.extra[inst.data + 2 ..][0..list.payload];
}

pub fn span(mir: *const Mir, inst: Inst) []const u32 {
    const data = inst.data;
    const len: u32 = switch (inst.tag) {
        .call_begin, .call_end, .ret => return &.{},
        .param_load, .param_store => 2 + @as(ParamAccess, @bitCast(mir.extra[data])).len,
        .call => len: {
            const call: Call = @bitCast(mir.extra[data]);
            break :len 1 + @as(Operand, @bitCast(mir.extra[data + 1])).len() + @intFromBool(call.indirect);
        },
        else => mir.memOperandAt(inst) + memOperandCount(inst) - data,
    };
    return mir.extra[data..][0..len];
}

pub fn memOperandCount(inst: Inst) u32 {
    return switch (inst.tag) {
        .ld, .st, .atom => 1,
        .copy => if (inst.modifiers.copy.op == .set) 1 else 2,
        else => 0,
    };
}

/// Returns the position in `extra` of the first `MemOperand` of `inst`, right after its operands.
pub fn memOperandAt(mir: *const Mir, inst: Inst) u32 {
    const range = mir.operands(inst).?;
    var at = range.first;
    for (0..range.count) |_| at += @as(Operand, @bitCast(mir.extra[at])).len();
    return at;
}

/// `side` 1 is the source of a `copy`.
pub fn memOperand(mir: *const Mir, inst: Inst, side: u1) MemOperand {
    assert(side < memOperandCount(inst));
    return @bitCast(mir.extra[mir.memOperandAt(inst) + side]);
}

pub fn addressSpace(mir: *const Mir, inst: Inst, side: u1) Space {
    return switch (inst.tag) {
        .param_load, .param_store => .param,
        else => mir.memOperand(inst, side).space,
    };
}

/// Inline assembly has no `MemOperand`. The queries assume an access when its flags allow one.
fn asmAccessesMemory(inst: Inst) bool {
    const assembly = inst.modifiers.@"asm";
    return assembly.is_volatile or assembly.clobbers or assembly.pointer_inputs;
}

pub fn mayLoad(inst: Inst) bool {
    return switch (inst.tag) {
        .ld, .atom, .param_load, .call => true,
        .copy => inst.modifiers.copy.op != .set,
        .@"asm" => asmAccessesMemory(inst),
        else => false,
    };
}

pub fn mayStore(inst: Inst) bool {
    return switch (inst.tag) {
        .st, .atom, .copy, .param_store, .call => true,
        .@"asm" => asmAccessesMemory(inst),
        else => false,
    };
}

/// Returns whether `inst` makes a volatile or ordered access, or orders accesses as a fence does.
pub fn hasOrderedMemoryRef(mir: *const Mir, inst: Inst) bool {
    return switch (inst.tag) {
        .ld, .st, .atom, .copy => for (0..memOperandCount(inst)) |side| {
            if (mir.memOperand(inst, @intCast(side)).sem != .none) break true;
        } else false,
        .fence, .call => true,
        .@"asm" => asmAccessesMemory(inst),
        else => false,
    };
}

/// Returns whether `inst` is a load whose value stays the same while the kernel runs.
pub fn isDereferenceableInvariantLoad(mir: *const Mir, inst: Inst) bool {
    return switch (inst.tag) {
        .param_load => @as(ParamAccess, @bitCast(mir.extra[inst.data])).invariant,
        .ld => mem: {
            const mem = mir.memOperand(inst, 0);
            break :mem mem.invariant and mem.sem == .none;
        },
        else => false,
    };
}

pub const RegRole = enum {
    use,
    def,
    /// The assembly writes one of its outputs at this place in its template.
    def_ref,
};

/// Calls `visitor.reg(at, role)` once for each register of `inst`. `at` is the position in `extra`.
pub fn visitRegs(mir: *const Mir, inst: Inst, visitor: anytype) void {
    const extra = mir.extra;
    const data = inst.data;
    switch (inst.tag) {
        .param_load, .param_store => {
            const access: ParamAccess = @bitCast(extra[data]);
            const role: RegRole = if (inst.tag == .param_load) .def else .use;
            for (0..access.len) |i| visitor.reg(data + 2 + @as(u32, @intCast(i)), role);
        },
        .call => if (@as(Operand, @bitCast(extra[data + 1])).kind == .reg) visitor.reg(data + 1, .use),
        .call_begin, .call_end, .ret, .param_decl, .callprototype => {},
        .@"asm" => {
            const outputs = inst.modifiers.@"asm".outputs;
            const output_words = extra[data + 1 ..][0..outputs];
            var at = data + 1;
            for (0..extra[data]) |k| {
                const op: Operand = @bitCast(extra[at]);
                if (k < outputs) {
                    visitor.reg(at, .def);
                } else if (k < 2 * outputs or op.kind == .reg) {
                    if (op.kind == .reg) visitor.reg(at, asmRole(output_words, k, outputs, extra[at]));
                } else if (op.kind == .list) {
                    for (at + 1..at + 1 + op.payload) |e| visitor.reg(@intCast(e), asmRole(output_words, k, outputs, extra[e]));
                }
                at += op.len();
            }
        },
        else => {
            const range = mir.operands(inst).?;
            const def_index: ?usize = if (inst.tag == .label)
                1
            else if (hasResult(inst.tag))
                @intFromBool(inst.modifiers.guard() != .none)
            else
                null;
            var at = range.first;
            for (0..range.count) |k| {
                const op: Operand = @bitCast(extra[at]);
                const role: RegRole = if (def_index == k) .def else .use;
                switch (op.kind) {
                    .reg => visitor.reg(at, role),
                    .list => for (at + 1..at + 1 + op.payload) |e| visitor.reg(@intCast(e), role),
                    else => {},
                }
                at += op.len();
            }
        },
    }
}

/// A register in the template can also be an output of the `asm`. The assembly then writes that
/// output at the position of the register in the template.
fn asmRole(output_words: []const u32, k: usize, outputs: u32, word: u32) RegRole {
    if (k < 2 * outputs) return .use;
    return if (std.mem.findScalar(u32, output_words, word) != null) .def_ref else .use;
}

pub fn usedSymbols(mir: *const Mir, gpa: Allocator, list: *std.ArrayList(Target)) !void {
    for (0..mir.instructions.len) |i| {
        const range = mir.operands(mir.instructions.get(i)) orelse continue;
        var at = range.first;
        for (0..range.count) |_| {
            const op: Operand = @bitCast(mir.extra[at]);
            if (op.kind == .symbol) try list.append(gpa, mir.symbols[op.payload].target);
            at += op.len();
        }
    }
}
