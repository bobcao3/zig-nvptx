//! Generates random, well-typed Zig functions and shrinks them.
//! Each test function `t<index>` takes four `u64` input words and returns a `u32` checksum of
//! every value that it computes.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Writer = std.Io.Writer;
const Smith = std.testing.Smith;
const generator = @This();

const none = std.math.maxInt(u32);
const max_bits = 256;
const PrintError = error{ WriteFailed, OutOfMemory, NoSpaceLeft };
pub const input_words = 4;

/// Returns a value less than `n`. With byte input, `Smith` returns the minimum of a range for every
/// input word outside of the range, so this function reduces a whole word modulo `n` instead.
pub fn below(smith: *Smith, n: u32) u32 {
    std.debug.assert(n > 0);
    if (smith.in == null) return smith.valueRangeLessThan(u32, 0, n);
    return @intCast(smith.value(u64) % n);
}

const Source = struct {
    smith: *Smith,

    fn below(src: Source, n: u32) u32 {
        return generator.below(src.smith, n);
    }

    fn percent(src: Source, p: u32) bool {
        return src.below(100) < p;
    }

    fn word(src: Source) u64 {
        return src.smith.value(u64);
    }
};

const TypeId = u32;
const NodeId = u32;
const StmtId = u32;
const VarId = u32;

const Int = struct { signed: bool, bits: u16 };
const Seq = struct { len: u8, elem: TypeId };
const Ptr = struct { child: TypeId, mutable: bool };

const Type = union(enum) {
    int: Int,
    float: u16,
    bool,
    vector: Seq,
    array: Seq,
    @"struct": []const TypeId,
    optional: TypeId,
    error_union: TypeId,
    pointer: Ptr,

    fn eql(a: Type, b: Type) bool {
        if (std.meta.activeTag(a) != std.meta.activeTag(b)) return false;
        return switch (a) {
            .int => |x| x.signed == b.int.signed and x.bits == b.int.bits,
            .float => |x| x == b.float,
            .bool => true,
            .vector => |x| x.len == b.vector.len and x.elem == b.vector.elem,
            .array => |x| x.len == b.array.len and x.elem == b.array.elem,
            .@"struct" => false,
            .optional => |x| x == b.optional,
            .error_union => |x| x == b.error_union,
            .pointer => |x| x.child == b.pointer.child and x.mutable == b.pointer.mutable,
        };
    }
};

const via_space_variants = 7;

const Op = enum {
    var_ref,
    lit,
    input,
    bin,
    shift,
    sat_shl,
    div,
    un,
    abs_signed,
    min_max,
    cmp,
    not,
    and_or,
    select,
    splat,
    init,
    elem,
    elem_runtime,
    field,
    reduce,
    conv,
    bitcount,
    overflow,
    mul_add,
    call,
    cond,
    @"orelse",
    @"catch",
    opt_wrap,
    opt_cond,
    eu_wrap,
    eu_err,
    eu_cond,
    is_null,
    eu_ok,
    switch_expr,
    div_const,
    call_ptr,
    addr_of,
    deref,
    /// `(if (cond) &a else &b).*` for two variables `a` and `b`, which reads through a pointer
    /// that holds either address.
    pick_ref,
    /// `@bitCast` of an aggregate, or a read of one member of an extern union that another member
    /// of a different size initialized. The field `b` selects among a bit cast, an extern union and
    /// a pointer cast.
    reinterpret,
    /// A write of one element of an array member of an extern union, followed by a read of the
    /// whole union.
    patch,
    /// `patch` at an index that the program computes. The written element is the exclusive or of
    /// the new element and the next element of the array.
    patch_at,
    /// `viaSpaceN(T, x, flag)` returns `x` at compile time. At run time, the helper stores `x` to
    /// shared or local memory, reads it back through a generic pointer and returns the value. `b`
    /// selects how the pointer is used.
    via_space,
};

const BinOp = enum {
    add_wrap,
    sub_wrap,
    mul_wrap,
    add_sat,
    sub_sat,
    mul_sat,
    bit_and,
    bit_or,
    bit_xor,
    add,
    sub,
    mul,
    div,

    const text = [_][]const u8{ "+%", "-%", "*%", "+|", "-|", "*|", "&", "|", "^", "+", "-", "*", "/" };
};

const DivOp = enum {
    div_trunc,
    div_floor,
    div_ceil,
    mod,
    rem,
    slash,
    percent,
    div_exact,

    const text = [_][]const u8{ "@divTrunc", "@divFloor", "@divCeil", "@mod", "@rem", "/", "%", "@divExact" };
};

const UnOp = enum {
    bit_not,
    neg_wrap,
    neg,
    abs,
    sqrt,
    floor,
    ceil,
    trunc,
    round,
    bit_reverse,
    byte_swap,
    bool_not,

    const text = [_][]const u8{ "~", "-%", "-", "@abs", "@sqrt", "@floor", "@ceil", "@trunc", "@round", "@bitReverse", "@byteSwap", "!" };
};

const CmpOp = enum {
    eq,
    ne,
    lt,
    le,
    gt,
    ge,

    const text = [_][]const u8{ "==", "!=", "<", "<=", ">", ">=" };
};

const CallKind = enum { plain, never_inline, auto, always_inline };

const ReduceOp = enum { Add, Mul, And, Or, Xor, Min, Max };

const Conv = enum {
    int_int,
    int_cast,
    float_float,
    int_to_float,
    float_to_int,
    bits,
    bool_to_int,
    vec_int,
    vec_int_to_float,
    vec_float_float,
};

const Node = struct {
    ty: TypeId,
    op: Op,
    kids: []NodeId = &.{},
    a: u64 = 0,
    b: u32 = 0,
    /// The bits of a literal.
    value: u256 = 0,
};

const Var = struct {
    ty: TypeId,
    mutable: bool = false,
    read_only: bool = false,
    deleted: bool = false,
    referenced: bool = false,
    assigned: bool = false,
};

const Item = struct { lo: i512, hi: i512 };
const Prong = []const Item;

const Stmt = struct {
    tag: enum { decl, assign, if_else, @"switch", loop, break_if, continue_if, return_if },
    deleted: bool = false,
    flatten: u32 = 0,
    v: VarId = none,
    expr: NodeId = none,
    value: NodeId = none,
    blocks: []const []const StmtId = &.{},
    prongs: []const Prong = &.{},
    count: u32 = 0,
    /// A loop with this flag is a `for` loop over a `usize` range, which casts its capture to the
    /// `u32` counter. Its `expr`, when present, is a runtime part of the bound.
    for_range: bool = false,
};

const Func = struct {
    params: []const VarId,
    ret: TypeId,
    body: []const StmtId,
    ret_expr: NodeId,
    deleted: bool = false,
    is_noinline: bool = false,
    /// The helpers of one group call each other recursively. Every call within the group passes
    /// the `bound` parameter minus one, and a helper returns `base` when its bound is zero.
    group: u32 = none,
    bound: VarId = none,
    base: NodeId = none,
    /// The sum of the sizes of the parameters and the variables of the helper in bytes.
    var_bytes: u32 = 0,
};

pub const Edit = union(enum) {
    delete_func: u32,
    delete_stmt: StmtId,
    flatten: struct { stmt: StmtId, block: u32 },
    zero: NodeId,
    hoist: struct { node: NodeId, kid: u32 },
};

pub const Unit = struct {
    arena: Allocator,
    src: Source,
    index: u32,
    types: std.ArrayList(Type) = .empty,
    pool: std.ArrayList(TypeId) = .empty,
    vars: std.ArrayList(Var) = .empty,
    nodes: std.ArrayList(Node) = .empty,
    stmts: std.ArrayList(Stmt) = .empty,
    funcs: std.ArrayList(Func) = .empty,
    switches: std.ArrayList([]const Prong) = .empty,
    fn_sets: std.ArrayList([]const u32) = .empty,
    scope: std.ArrayList(VarId) = .empty,
    cur: u32 = 0,
    in_base: bool = false,
    is_test: bool = false,
    captures: u32 = 0,
    groups: u32 = 0,
    /// The index of the first variable of the function that the generator fills in.
    first_var: usize = 0,
    /// Bit `k` is set when the kernel passes input word `k` as a kernel parameter. The parameter
    /// holds the word of the first row for every row.
    by_value: u4 = 0,

    pub fn init(arena: Allocator, smith: *Smith, index: u32) Unit {
        return .{ .arena = arena, .src = .{ .smith = smith }, .index = index };
    }

    fn intern(u: *Unit, t: Type) !TypeId {
        for (u.types.items, 0..) |existing, i| if (existing.eql(t)) return @intCast(i);
        try u.types.append(u.arena, t);
        return @intCast(u.types.items.len - 1);
    }

    fn ty(u: *const Unit, t: TypeId) Type {
        return u.types.items[t];
    }

    fn intType(u: *Unit, signed: bool, bits: u16) !TypeId {
        return u.intern(.{ .int = .{ .signed = signed, .bits = bits } });
    }

    fn randBits(u: *Unit) u16 {
        if (u.src.percent(12)) return u.wideBits();
        return u.narrowBits();
    }

    fn narrowBits(u: *Unit) u16 {
        const common = [_]u16{ 8, 16, 32, 64 };
        if (u.src.percent(40)) return common[u.src.below(common.len)];
        return @intCast(1 + u.src.below(64));
    }

    fn wideBits(u: *Unit) u16 {
        const common = [_]u16{ 128, 256 };
        if (u.src.percent(40)) return common[u.src.below(common.len)];
        return @intCast(65 + u.src.below(max_bits - 64));
    }

    fn narrowInt(u: *Unit, signed: bool) !TypeId {
        return u.intType(signed, u.narrowBits());
    }

    fn randInt(u: *Unit) !TypeId {
        return u.intType(u.src.percent(50), u.randBits());
    }

    fn randFloat(u: *Unit) !TypeId {
        const widths = [_]u16{ 16, 32, 64 };
        return u.intern(.{ .float = widths[u.src.below(widths.len)] });
    }

    fn randScalar(u: *Unit) !TypeId {
        const r = u.src.below(10);
        if (r < 6) return u.randInt();
        if (r < 9) return u.randFloat();
        return u.intern(.bool);
    }

    fn randVector(u: *Unit) !TypeId {
        const len: u8 = @intCast(2 + u.src.below(7));
        const elem = switch (u.src.below(6)) {
            0, 1, 2 => try u.randInt(),
            3, 4 => try u.randFloat(),
            else => try u.intern(.bool),
        };
        return u.withElem(try u.intern(.{ .vector = .{ .len = len, .elem = try u.intern(.bool) } }), elem);
    }

    fn randType(u: *Unit) !TypeId {
        return switch (u.src.below(20)) {
            0...7 => u.randInt(),
            8, 9, 10 => u.randFloat(),
            11 => u.intern(.bool),
            12, 13, 14 => u.randVector(),
            15 => {
                const elem = if (u.src.percent(70)) try u.randScalar() else try u.randVector();
                // An array of more than 128 bytes stays in memory when the code loads it.
                const large = u.byteSize(elem) >= 4 and u.byteSize(elem) <= 8 and u.src.percent(15);
                return u.intern(.{ .array = .{ .len = @intCast(if (large) 33 + u.src.below(8) else 1 + u.src.below(4)), .elem = elem } });
            },
            16, 17 => u.newStruct(),
            18 => u.intern(.{ .optional = try u.randScalar() }),
            else => u.intern(.{ .error_union = try u.randScalar() }),
        };
    }

    fn newStruct(u: *Unit) !TypeId {
        const fields = try u.arena.alloc(TypeId, 1 + u.src.below(4));
        for (fields) |*f| f.* = if (u.src.percent(75)) try u.randScalar() else try u.randVector();
        try u.types.append(u.arena, .{ .@"struct" = fields });
        return @intCast(u.types.items.len - 1);
    }

    /// Returns the type of a new local variable. Once the variables of a function take
    /// `local_bytes`, the function declares only `u32` variables. Large variables make large stack
    /// frames, and a CUDA thread has 1 KiB of stack by default.
    fn pickLocalType(u: *Unit) !TypeId {
        const local_bytes = 512;
        var bytes: u32 = 0;
        for (u.vars.items[u.first_var..]) |v| bytes += u.byteSize(v.ty);
        if (bytes > local_bytes) return u.intType(false, 32);
        return u.pickType();
    }

    fn pickType(u: *Unit) !TypeId {
        if (u.src.percent(75)) return u.pool.items[u.src.below(@intCast(u.pool.items.len))];
        return u.randType();
    }

    fn addNode(u: *Unit, n: Node) !NodeId {
        try u.nodes.append(u.arena, n);
        return @intCast(u.nodes.items.len - 1);
    }

    fn addVar(u: *Unit, v: Var) !VarId {
        try u.vars.append(u.arena, v);
        return @intCast(u.vars.items.len - 1);
    }

    fn addStmt(u: *Unit, s: Stmt) !StmtId {
        try u.stmts.append(u.arena, s);
        return @intCast(u.stmts.items.len - 1);
    }

    fn kids(u: *Unit, ids: []const NodeId) ![]NodeId {
        return u.arena.dupe(NodeId, ids);
    }

    fn intMask(bits: u16) u256 {
        return if (bits == max_bits) std.math.maxInt(u256) else (@as(u256, 1) << @intCast(bits)) - 1;
    }

    fn intLit(u: *Unit, i: Int) u256 {
        const mask = intMask(i.bits);
        const top: u256 = @as(u256, 1) << @intCast(i.bits - 1);
        const v: u256 = switch (u.src.below(9)) {
            0 => 0,
            1 => 1,
            2 => mask,
            3 => top,
            4 => top -% 1,
            5 => u.src.below(16),
            6 => mask -% u.src.below(4),
            7 => @as(u256, 1) << @intCast(u.src.below(i.bits)),
            else => u.wideWord(),
        };
        return v & mask;
    }

    fn floatLit(u: *Unit, bits: u16) u64 {
        const value: f64 = switch (u.src.below(11)) {
            0 => 0.0,
            1 => -0.0,
            2 => 1.0,
            3 => -1.5,
            4 => std.math.inf(f64),
            5 => -std.math.inf(f64),
            6 => std.math.nan(f64),
            7 => @floatFromInt(@as(i32, @intCast(u.src.below(200))) - 100),
            8 => 0.1,
            9 => 2.5,
            10 => -0.5,
            else => return u.src.word() & @as(u64, @truncate(intMask(bits))),
        };
        return switch (bits) {
            16 => @as(u16, @bitCast(@as(f16, @floatCast(value)))),
            32 => @as(u32, @bitCast(@as(f32, @floatCast(value)))),
            else => @bitCast(value),
        };
    }

    fn wideWord(u: *Unit) u256 {
        var v: u256 = 0;
        for (0..max_bits / 64) |_| v = (v << 64) | u.src.word();
        return v;
    }

    fn scalarLit(u: *Unit, t: TypeId) u256 {
        return switch (u.ty(t)) {
            .int => |i| u.intLit(i),
            .float => |f| u.floatLit(f),
            .bool => u.src.below(2),
            else => 0,
        };
    }

    fn visible(u: *Unit, t: TypeId, buf: []VarId) []VarId {
        var n: usize = 0;
        for (u.scope.items) |v| {
            if (u.valueType(v) == t and n < buf.len) {
                buf[n] = v;
                n += 1;
            }
        }
        return buf[0..n];
    }

    /// Returns the type of the value that variable `v` names, which is the pointee type when the
    /// variable is a pointer parameter.
    fn valueType(u: *const Unit, v: VarId) TypeId {
        const t = u.vars.items[v].ty;
        return switch (u.ty(t)) {
            .pointer => |p| p.child,
            else => t,
        };
    }

    fn genLeaf(u: *Unit, t: TypeId) error{OutOfMemory}!NodeId {
        var buf: [64]VarId = undefined;
        const vs = u.visible(t, &buf);
        if (vs.len > 0 and u.src.percent(65)) {
            const v = vs[u.src.below(@intCast(vs.len))];
            const other = vs[u.src.below(@intCast(vs.len))];
            if (other != v and u.vars.items[v].ty == t and u.vars.items[other].ty == t and u.src.percent(20)) {
                const cond = try u.genLeaf(try u.intern(.bool));
                return u.addNode(.{ .ty = t, .op = .pick_ref, .a = v, .b = other, .kids = try u.kids(&.{cond}) });
            }
            return u.addNode(.{ .ty = t, .op = if (u.vars.items[v].ty == t) .var_ref else .deref, .a = v });
        }
        if (u.isScalar(t) and u.ty(t) != .bool and u.src.percent(50)) {
            var n: usize = 0;
            for (u.scope.items) |v| {
                const vt = u.vars.items[v].ty;
                if (vt == t or n == buf.len) continue;
                if (std.meta.activeTag(u.ty(vt)) != std.meta.activeTag(u.ty(t)) and (u.isWideInt(vt) or u.isWideInt(t))) continue;
                switch (u.ty(vt)) {
                    .int, .float => {
                        buf[n] = v;
                        n += 1;
                    },
                    else => {},
                }
            }
            if (n > 0) {
                const v = buf[u.src.below(@intCast(n))];
                const vt = u.vars.items[v].ty;
                const kid = try u.addNode(.{ .ty = vt, .op = .var_ref, .a = v });
                const conv: Conv = switch (u.ty(t)) {
                    .int => if (u.ty(vt) == .int) .int_int else .float_to_int,
                    else => if (u.ty(vt) == .int) .int_to_float else .float_float,
                };
                return u.unary(t, .conv, @backingInt(conv), kid);
            }
        }
        switch (u.ty(t)) {
            .int, .float, .bool => if (u.is_test and u.src.percent(60)) return u.addNode(.{
                .ty = t,
                .op = .input,
                .a = u.src.below(input_words),
                .b = if (u.src.percent(70)) 0 else u.src.below(64),
            }),
            .vector => |s| if (u.src.percent(60)) {
                if (u.src.percent(50)) return u.addNode(.{ .ty = t, .op = .splat, .kids = try u.kids(&.{try u.genLeaf(s.elem)}) });
                const ks = try u.arena.alloc(NodeId, s.len);
                for (ks) |*k| k.* = try u.genLeaf(s.elem);
                return u.addNode(.{ .ty = t, .op = .init, .kids = ks });
            },
            .array => |s| if (u.src.percent(60)) {
                const ks = try u.arena.alloc(NodeId, s.len);
                for (ks) |*k| k.* = try u.genLeaf(s.elem);
                return u.addNode(.{ .ty = t, .op = .init, .kids = ks });
            },
            .@"struct" => |fields| if (u.src.percent(60)) {
                const ks = try u.arena.alloc(NodeId, fields.len);
                for (ks, fields) |*k, f| k.* = try u.genLeaf(f);
                return u.addNode(.{ .ty = t, .op = .init, .kids = ks });
            },
            .optional => |p| if (u.src.percent(60)) return u.addNode(.{ .ty = t, .op = .opt_wrap, .kids = try u.kids(&.{try u.genLeaf(p)}) }),
            .error_union => |p| if (u.src.percent(60)) {
                if (u.src.percent(30)) return u.addNode(.{ .ty = t, .op = .eu_err, .a = u.src.below(3) });
                return u.addNode(.{ .ty = t, .op = .eu_wrap, .kids = try u.kids(&.{try u.genLeaf(p)}) });
            },
            .pointer => unreachable,
        }
        return u.addNode(.{ .ty = t, .op = .lit, .value = u.scalarLit(t) });
    }

    fn genExpr(u: *Unit, t: TypeId, depth: u32) error{OutOfMemory}!NodeId {
        if (depth == 0 or u.src.percent(20)) return u.genLeaf(t);
        for (0..8) |_| if (try u.genOp(t, depth)) |n| return n;
        return u.genLeaf(t);
    }

    fn unary(u: *Unit, t: TypeId, op: Op, b: u32, kid: NodeId) !NodeId {
        return u.addNode(.{ .ty = t, .op = op, .b = b, .kids = try u.kids(&.{kid}) });
    }

    fn binary(u: *Unit, t: TypeId, op: Op, b: u32, k0: NodeId, k1: NodeId) !NodeId {
        return u.addNode(.{ .ty = t, .op = op, .b = b, .kids = try u.kids(&.{ k0, k1 }) });
    }

    fn scalarOf(u: *const Unit, t: TypeId) TypeId {
        return switch (u.ty(t)) {
            .vector => |s| s.elem,
            else => t,
        };
    }

    /// Returns the type `t` with the elements `elem` when `t` is a vector, and `elem` otherwise.
    /// Replaces an element type that `vectorElem` rejects with a 128-bit integer.
    fn withElem(u: *Unit, t: TypeId, elem: TypeId) !TypeId {
        return switch (u.ty(t)) {
            .vector => |s| u.intern(.{ .vector = .{
                .len = s.len,
                .elem = if (u.vectorElem(elem)) elem else try u.intType(u.ty(elem).int.signed, 128),
            } }),
            else => elem,
        };
    }

    /// The backend does not lay out a vector of integers whose ABI size holds a 64-bit limb
    /// beyond the bits of the integer.
    fn vectorElem(u: *const Unit, elem: TypeId) bool {
        return switch (u.ty(elem)) {
            .int => |i| i.bits <= 128 or i.bits > 192,
            else => true,
        };
    }

    /// The backend does not convert between floats and integers wider than 64 bits.
    fn isWideInt(u: *const Unit, t: TypeId) bool {
        return switch (u.ty(u.scalarOf(t))) {
            .int => |i| i.bits > 64,
            else => false,
        };
    }

    fn canShare(u: *const Unit, t: TypeId) bool {
        return switch (u.ty(t)) {
            .int => |i| i.bits <= 64,
            .float => |bits| bits <= 64,
            .bool => true,
            else => false,
        };
    }

    fn genViaSpace(u: *Unit, t: TypeId, d: u32) !NodeId {
        const flag = try u.genExpr(try u.intern(.bool), d);
        return u.addNode(.{ .ty = t, .op = .via_space, .b = u.src.below(via_space_variants), .kids = try u.kids(&.{ try u.genExpr(t, d), flag }) });
    }

    fn genOp(u: *Unit, t: TypeId, depth: u32) !?NodeId {
        const d = depth - 1;
        const r = u.src.below(100);
        if (r < 4) return u.genCommon(t, depth);
        if (r < 12 and u.canShare(t)) return try u.genViaSpace(t, d);
        const st = u.ty(u.scalarOf(t));
        const is_vec = u.ty(t) == .vector;
        switch (st) {
            .int => |i| switch (u.src.below(18)) {
                0, 1, 2 => {
                    const op = u.src.below(9);
                    return try u.binary(t, .bin, op, try u.genExpr(t, d), try u.genExpr(t, d));
                },
                3 => {
                    const amount = try u.genLeaf(try u.narrowInt(false));
                    return try u.binary(t, .shift, u.src.below(2), try u.genExpr(t, d), amount);
                },
                4 => {
                    if (is_vec) return null;
                    // Sema saturates a negative integer to its maximum when the shift amount does
                    // not fit in a `usize`.
                    const amount = try u.genLeaf(try u.narrowInt(false));
                    return try u.binary(t, .sat_shl, 0, try u.genExpr(t, d), amount);
                },
                5 => {
                    if (i.signed and i.bits == 1) return null;
                    const ops: []const DivOp = if (i.signed) &.{ .div_trunc, .div_floor, .div_ceil, .mod, .rem } else &.{ .div_trunc, .div_floor, .div_ceil, .mod, .rem, .slash, .percent };
                    const op = ops[u.src.below(@intCast(ops.len))];
                    return try u.binary(t, .div, @backingInt(op), try u.genLeaf(t), try u.genLeaf(t));
                },
                6 => {
                    const ops: []const UnOp = if (i.bits % 8 == 0) &.{ .bit_not, .neg_wrap, .bit_reverse, .byte_swap } else &.{ .bit_not, .neg_wrap, .bit_reverse };
                    return try u.unary(t, .un, @backingInt(ops[u.src.below(@intCast(ops.len))]), try u.genExpr(t, d));
                },
                7 => {
                    if (i.signed) return null;
                    const src_t = try u.withElem(t, try u.intType(true, i.bits));
                    return try u.unary(t, .abs_signed, 0, try u.genExpr(src_t, d));
                },
                8 => return try u.binary(t, .min_max, u.src.below(2), try u.genExpr(t, d), try u.genExpr(t, d)),
                9, 10 => return u.genConv(t, d),
                11 => {
                    if (i.signed or i.bits > 9) return null;
                    const lo: u32 = @as(u32, 1) << @intCast(i.bits - 1);
                    const hi: u32 = @min(max_bits, (@as(u32, 1) << @intCast(i.bits)) - 1);
                    if (lo > hi) return null;
                    const n: u16 = @intCast(lo + u.src.below(hi - lo + 1));
                    const src_t = try u.withElem(t, try u.intType(u.src.percent(50), n));
                    return try u.unary(t, .bitcount, u.src.below(3), try u.genExpr(src_t, d));
                },
                12 => {
                    if (is_vec or !u.vectorElem(t)) return null;
                    const vt = try u.intern(.{ .vector = .{ .len = @intCast(2 + u.src.below(7)), .elem = t } });
                    var op: ReduceOp = @fromBackingInt(@intCast(u.src.below(7)));
                    if (op == .Or and i.signed and i.bits > 64) op = .Xor;
                    return try u.unary(t, .reduce, @backingInt(op), try u.genExpr(vt, d));
                },
                13 => {
                    if (is_vec) return null;
                    if (i.bits == 1 and !i.signed and u.src.percent(50)) {
                        const src_t = try u.randInt();
                        const op = u.src.below(4);
                        const rhs = if (op == 3) try u.genLeaf(try u.narrowInt(false)) else try u.genExpr(src_t, d);
                        return try u.addNode(.{ .ty = t, .op = .overflow, .a = 1, .b = op, .kids = try u.kids(&.{ try u.genExpr(src_t, d), rhs }) });
                    }
                    const op = u.src.below(4);
                    const rhs = if (op == 3) try u.genLeaf(try u.narrowInt(false)) else try u.genExpr(t, d);
                    return try u.addNode(.{ .ty = t, .op = .overflow, .a = 0, .b = op, .kids = try u.kids(&.{ try u.genExpr(t, d), rhs }) });
                },
                15, 16 => {
                    if (i.signed and i.bits == 1) return null;
                    const ops: []const DivOp = if (i.signed) &.{ .div_trunc, .div_floor, .div_ceil, .mod, .rem, .div_exact } else &.{ .div_trunc, .div_floor, .div_ceil, .mod, .rem, .slash, .percent, .div_exact };
                    const op = ops[u.src.below(@intCast(ops.len))];
                    const pick: u8 = @truncate(u.src.word());
                    const picks = if (is_vec and u.src.percent(50)) u.src.word() else @as(u64, pick) * 0x0101010101010101;
                    const dividend = if (u.src.percent(30)) try u.genLeaf(t) else try u.genExpr(t, d);
                    return try u.addNode(.{ .ty = t, .op = .div_const, .a = picks, .b = @backingInt(op), .kids = try u.kids(&.{dividend}) });
                },
                else => return u.genCommon(t, depth),
            },
            .float => switch (u.src.below(10)) {
                0, 1, 2 => return try u.binary(t, .bin, @backingInt(BinOp.add) + u.src.below(4), try u.genExpr(t, d), try u.genExpr(t, d)),
                3 => return try u.binary(t, .min_max, u.src.below(2), try u.genExpr(t, d), try u.genExpr(t, d)),
                4 => return try u.unary(t, .un, @backingInt(UnOp.neg) + u.src.below(7), try u.genExpr(t, d)),
                5 => return try u.addNode(.{ .ty = t, .op = .mul_add, .kids = try u.kids(&.{ try u.genExpr(t, d), try u.genExpr(t, d), try u.genExpr(t, d) }) }),
                6 => {
                    const ops = [_]DivOp{ .div_trunc, .div_floor, .mod, .rem };
                    return try u.binary(t, .div, @backingInt(ops[u.src.below(ops.len)]), try u.genExpr(t, d), try u.genExpr(t, d));
                },
                7, 8 => return u.genConv(t, d),
                else => {
                    if (is_vec) return u.genCommon(t, depth);
                    const vt = try u.intern(.{ .vector = .{ .len = @intCast(2 + u.src.below(7)), .elem = t } });
                    const ops = [_]ReduceOp{ .Add, .Mul, .Min, .Max };
                    return try u.unary(t, .reduce, @backingInt(ops[u.src.below(ops.len)]), try u.genExpr(vt, d));
                },
            },
            .bool => switch (u.src.below(8)) {
                0, 1, 2 => {
                    const s = if (is_vec) try u.withElem(t, try u.randScalar()) else try u.randScalar();
                    if (u.ty(u.scalarOf(s)) == .bool) return null;
                    return try u.binary(t, .cmp, u.src.below(6), try u.genExpr(s, d), try u.genExpr(s, d));
                },
                3 => return try u.unary(t, .not, 0, try u.genExpr(t, d)),
                4 => {
                    if (is_vec) {
                        const op = @backingInt(BinOp.bit_and) + u.src.below(3);
                        return try u.binary(t, .bin, op, try u.genExpr(t, d), try u.genExpr(t, d));
                    }
                    return try u.binary(t, .and_or, u.src.below(2), try u.genExpr(t, d), try u.genExpr(t, d));
                },
                5 => {
                    if (is_vec) return null;
                    const vt = try u.intern(.{ .vector = .{ .len = @intCast(2 + u.src.below(7)), .elem = t } });
                    return try u.unary(t, .reduce, @backingInt(ReduceOp.And) + u.src.below(3), try u.genExpr(vt, d));
                },
                6 => {
                    if (is_vec) return null;
                    if (u.src.percent(50)) {
                        const ot = try u.intern(.{ .optional = try u.randScalar() });
                        return try u.unary(t, .is_null, 0, try u.genExpr(ot, d));
                    }
                    const et = try u.intern(.{ .error_union = try u.randScalar() });
                    return try u.unary(t, .eu_ok, 0, try u.genExpr(et, d));
                },
                else => {
                    if (is_vec) return u.genConv(t, d);
                    return u.genCommon(t, depth);
                },
            },
            else => {},
        }
        return switch (u.ty(t)) {
            .array => |s| {
                if (u.src.percent(50)) return u.genCommon(t, depth);
                const ks = try u.arena.alloc(NodeId, s.len);
                for (ks) |*k| k.* = try u.genExpr(s.elem, d);
                return try u.addNode(.{ .ty = t, .op = .init, .kids = ks });
            },
            .@"struct" => |fields| {
                if (u.src.percent(50)) return u.genCommon(t, depth);
                const ks = try u.arena.alloc(NodeId, fields.len);
                for (ks, fields) |*k, f| k.* = try u.genExpr(f, d);
                return try u.addNode(.{ .ty = t, .op = .init, .kids = ks });
            },
            .optional => |p| switch (u.src.below(3)) {
                0 => try u.unary(t, .opt_wrap, 0, try u.genExpr(p, d)),
                1 => try u.binary(t, .opt_cond, 0, try u.genExpr(try u.intern(.bool), d), try u.genExpr(p, d)),
                else => u.genCommon(t, depth),
            },
            .error_union => |p| switch (u.src.below(3)) {
                0 => try u.unary(t, .eu_wrap, 0, try u.genExpr(p, d)),
                1 => try u.addNode(.{ .ty = t, .op = .eu_cond, .a = u.src.below(3), .kids = try u.kids(&.{ try u.genExpr(try u.intern(.bool), d), try u.genExpr(p, d) }) }),
                else => u.genCommon(t, depth),
            },
            else => null,
        };
    }

    fn genVectorOnly(u: *Unit, t: TypeId, d: u32) !?NodeId {
        const s = u.ty(t).vector;
        return switch (u.src.below(4)) {
            0 => try u.unary(t, .splat, 0, try u.genExpr(s.elem, d)),
            1 => {
                const ks = try u.arena.alloc(NodeId, s.len);
                for (ks) |*k| k.* = try u.genExpr(s.elem, d);
                return try u.addNode(.{ .ty = t, .op = .init, .kids = ks });
            },
            else => {
                const mt = try u.withElem(t, try u.intern(.bool));
                return try u.addNode(.{ .ty = t, .op = .select, .kids = try u.kids(&.{ try u.genExpr(mt, d), try u.genExpr(t, d), try u.genExpr(t, d) }) });
            },
        };
    }

    fn genCommon(u: *Unit, t: TypeId, depth: u32) !?NodeId {
        const d = depth - 1;
        if (u.ty(t) == .vector and u.src.percent(50)) return u.genVectorOnly(t, d);
        switch (u.src.below(11)) {
            0, 1 => {
                var options: [16]u32 = undefined;
                const n = u.callees(t, false, &options);
                if (n == 0) return null;
                return u.genCall(options[u.src.below(n)], d);
            },
            9 => return u.genCallPtr(t, d),
            10 => return u.genView(t, d),
            2 => return try u.addNode(.{ .ty = t, .op = .cond, .kids = try u.kids(&.{ try u.genExpr(try u.intern(.bool), d), try u.genExpr(t, d), try u.genExpr(t, d) }) }),
            3 => {
                const ot = try u.intern(.{ .optional = t });
                if (!u.isScalar(t)) return null;
                return try u.binary(t, .@"orelse", 0, try u.genExpr(ot, d), try u.genExpr(t, d));
            },
            4 => {
                if (!u.isScalar(t)) return null;
                const et = try u.intern(.{ .error_union = t });
                return try u.binary(t, .@"catch", 0, try u.genExpr(et, d), try u.genExpr(t, d));
            },
            5 => {
                const bits: u16 = if (u.src.percent(15)) u.wideBits() else @intCast(1 + u.src.below(16));
                const subject_t = try u.intType(u.src.percent(50), bits);
                const prongs = try u.genProngs(subject_t);
                try u.switches.append(u.arena, prongs);
                const switch_index = u.switches.items.len - 1;
                const ks = try u.arena.alloc(NodeId, prongs.len + 2);
                ks[0] = try u.genExpr(subject_t, d);
                for (ks[1..]) |*k| k.* = try u.genExpr(t, d);
                return try u.addNode(.{ .ty = t, .op = .switch_expr, .a = switch_index, .kids = ks });
            },
            6 => {
                const len: u8 = @intCast(if (u.src.percent(10)) 5 + u.src.below(36) else 1 + u.src.below(4));
                const at = try u.intern(.{ .array = .{ .len = len, .elem = t } });
                if (u.src.percent(50)) return try u.addNode(.{ .ty = t, .op = .elem, .a = u.src.below(len), .kids = try u.kids(&.{try u.genExpr(at, d)}) });
                const index = try u.genLeaf(try u.narrowInt(u.src.percent(50)));
                return try u.binary(t, .elem_runtime, 0, try u.genExpr(at, d), index);
            },
            7 => {
                if (!u.isScalar(t) or !u.vectorElem(t) or (u.ty(t) == .bool and u.src.percent(50))) return null;
                const len: u8 = @intCast(2 + u.src.below(7));
                const vt = try u.intern(.{ .vector = .{ .len = len, .elem = t } });
                if (u.src.percent(50)) return try u.addNode(.{ .ty = t, .op = .elem, .a = u.src.below(len), .kids = try u.kids(&.{try u.genExpr(vt, d)}) });
                const index = try u.genLeaf(try u.narrowInt(u.src.percent(50)));
                return try u.binary(t, .elem_runtime, 0, try u.genExpr(vt, d), index);
            },
            8 => {
                for (u.pool.items) |p| switch (u.ty(p)) {
                    .@"struct" => |fields| for (fields, 0..) |f, fi| {
                        if (f == t and u.src.percent(70)) return try u.addNode(.{ .ty = t, .op = .field, .a = fi, .kids = try u.kids(&.{try u.genExpr(p, d)}) });
                    },
                    else => {},
                };
                return null;
            },
            else => unreachable,
        }
    }

    /// Returns the bit size of `t` when `t` is a 32-bit or 64-bit integer or float, or an array of
    /// 8-bit, 16-bit or 32-bit integers with that many bits.
    fn viewBits(u: *const Unit, t: TypeId) ?u16 {
        const bits: u16 = switch (u.ty(t)) {
            .int => |i| i.bits,
            .float => |f| f,
            .array => |s| switch (u.ty(s.elem)) {
                .int => |i| if (i.bits == 8 or i.bits == 16 or i.bits == 32) i.bits * s.len else return null,
                else => return null,
            },
            else => return null,
        };
        return if (bits == 32 or bits == 64) bits else null;
    }

    /// Returns the integer types and arrays of integers with `bits` bits. A NaN result has
    /// unspecified bits, so no float is a source of a reinterpretation.
    fn viewFamily(u: *Unit, bits: u16, buf: *[8]TypeId) ![]const TypeId {
        var n: usize = 0;
        for ([_]bool{ false, true }) |signed| {
            buf[n] = try u.intType(signed, bits);
            n += 1;
            for ([_]u16{ 8, 16, 32 }) |elem_bits| {
                if (elem_bits >= bits) continue;
                buf[n] = try u.intern(.{ .array = .{ .len = @intCast(bits / elem_bits), .elem = try u.intType(signed, elem_bits) } });
                n += 1;
            }
        }
        return buf[0..n];
    }

    fn genView(u: *Unit, t: TypeId, d: u32) !?NodeId {
        const bits = u.viewBits(t) orelse return null;
        var buf: [8]TypeId = undefined;
        const members = try u.viewFamily(bits, &buf);
        const source = members[u.src.below(@intCast(members.len))];
        if (source == t) return null;
        if (u.ty(t) != .float and u.src.percent(30)) {
            const part_bits: u16 = @as(u16, 8) << @intCast(u.src.below(3));
            if (part_bits >= bits) return null;
            const part = try u.intType(u.src.percent(50), part_bits);
            if (u.src.percent(50)) {
                const index = try u.genExpr(try u.intType(false, 32), d);
                return try u.addNode(.{ .ty = t, .op = .patch_at, .kids = try u.kids(&.{ try u.genExpr(t, d), try u.genExpr(part, d), index }) });
            }
            const at = u.src.below(bits / part_bits);
            return try u.addNode(.{ .ty = t, .op = .patch, .a = at, .kids = try u.kids(&.{ try u.genExpr(t, d), try u.genExpr(part, d) }) });
        }
        return try u.unary(t, .reinterpret, u.src.below(3), try u.genExpr(source, d));
    }

    /// Returns whether the function that the generator fills in may call helper `fi`. A helper
    /// calls only earlier helpers and the helpers of its own group.
    fn canCall(u: *const Unit, fi: u32) bool {
        const callee = u.funcs.items[fi];
        const same_group = callee.group != none and callee.group == u.funcs.items[u.cur].group;
        if (same_group) return !u.in_base;
        return fi < u.cur;
    }

    /// Writes to `buf` the helpers that return `t` and that the current function may call, and
    /// returns their count. When `t` is `none`, the return type does not matter. A call through a
    /// function pointer cannot pass the bound of a recursive helper, so `by_pointer` leaves the
    /// recursive helpers out.
    fn callees(u: *const Unit, t: TypeId, by_pointer: bool, buf: []u32) u32 {
        var n: u32 = 0;
        for (u.funcs.items, 0..) |f, fi| {
            if (t != none and f.ret != t) continue;
            if (by_pointer and f.group != none) continue;
            if (!u.canCall(@intCast(fi)) or n == buf.len) continue;
            buf[n] = @intCast(fi);
            n += 1;
        }
        return n;
    }

    fn genCall(u: *Unit, fi: u32, d: u32) !?NodeId {
        const callee = u.funcs.items[fi];
        const caller = u.funcs.items[u.cur];
        const inner = callee.group != none and callee.group == caller.group;
        const ks = try u.arena.alloc(NodeId, callee.params.len);
        for (ks, callee.params) |*k, p| {
            const pt = u.vars.items[p].ty;
            k.* = if (p != callee.bound)
                try u.genArg(pt, d) orelse return null
            else if (inner)
                try u.addNode(.{ .ty = pt, .op = .var_ref, .a = caller.bound })
            else
                try u.genExpr(pt, d);
        }
        const can_inline = !callee.is_noinline and callee.group == none;
        const kind: CallKind = @fromBackingInt(@intCast(u.src.below(if (can_inline) 4 else 3)));
        const b = @as(u32, @backingInt(kind)) | (@as(u32, @intFromBool(inner)) << 2);
        return try u.addNode(.{ .ty = callee.ret, .op = .call, .a = fi, .b = b, .kids = ks });
    }

    /// Returns an argument for a parameter of type `t`. A pointer argument points at a variable in
    /// scope or passes a pointer parameter on. Returns null when no variable fits.
    fn genArg(u: *Unit, t: TypeId, d: u32) !?NodeId {
        const p = switch (u.ty(t)) {
            .pointer => |p| p,
            else => return try u.genExpr(t, d),
        };
        var buf: [64]VarId = undefined;
        var n: u32 = 0;
        for (u.scope.items) |v| {
            const candidate = u.vars.items[v];
            const fits = switch (u.ty(candidate.ty)) {
                .pointer => |q| q.child == p.child and (q.mutable or !p.mutable),
                else => candidate.ty == p.child and (!p.mutable or (candidate.mutable and !candidate.read_only)),
            };
            if (!fits or n == buf.len) continue;
            buf[n] = v;
            n += 1;
        }
        if (n == 0) return null;
        const v = buf[u.src.below(n)];
        return try u.addNode(.{ .ty = t, .op = if (u.vars.items[v].ty == p.child) .addr_of else .var_ref, .a = v });
    }

    fn sameSignature(u: *const Unit, a: u32, b: u32) bool {
        const fa = u.funcs.items[a];
        const fb = u.funcs.items[b];
        if (fa.ret != fb.ret or fa.params.len != fb.params.len) return false;
        for (fa.params, fb.params) |pa, pb| if (u.vars.items[pa].ty != u.vars.items[pb].ty) return false;
        return true;
    }

    /// Generates a call through a function pointer that a runtime value selects among helpers with
    /// the same signature.
    fn genCallPtr(u: *Unit, t: TypeId, d: u32) !?NodeId {
        var options: [16]u32 = undefined;
        const n = u.callees(t, true, &options);
        if (n == 0) return null;
        const first = options[u.src.below(n)];
        var set: std.ArrayList(u32) = .empty;
        for (options[0..n]) |fi| {
            if (u.sameSignature(first, fi) and (fi == first or u.src.percent(80))) try set.append(u.arena, fi);
        }
        if (set.items.len < 2) return null;
        if (u.src.percent(30)) try set.append(u.arena, first);
        const params = u.funcs.items[first].params;
        const ks = try u.arena.alloc(NodeId, params.len + 1);
        ks[0] = try u.genExpr(try u.intType(false, 32), d);
        for (ks[1..], params) |*k, p| k.* = try u.genArg(u.vars.items[p].ty, d) orelse return null;
        try u.fn_sets.append(u.arena, set.items);
        const form = if (set.items.len == 2) u.src.below(2) else 1;
        return try u.addNode(.{ .ty = t, .op = .call_ptr, .a = u.fn_sets.items.len - 1, .b = form, .kids = ks });
    }

    fn isScalar(u: *const Unit, t: TypeId) bool {
        return switch (u.ty(t)) {
            .int, .float, .bool => true,
            else => false,
        };
    }

    /// Returns another type with the bit size of `t` or null. The backend does not bit-cast
    /// between an integer wider than 64 bits and a vector whose elements are not 8, 16, 32 or 64
    /// bits wide.
    fn sameBits(u: *Unit, t: TypeId) !?TypeId {
        const bits = u.bitSize(t) orelse return null;
        if (bits > 64 and u.ty(t) == .vector and !byteSized(bits / u.ty(t).vector.len)) return null;
        var options: [8]TypeId = undefined;
        var n: usize = 0;
        options[n] = try u.intType(u.src.percent(50), bits);
        n += 1;
        if (bits == 16 or bits == 32 or bits == 64) {
            options[n] = try u.intern(.{ .float = bits });
            n += 1;
        }
        for ([_]u16{ 2, 4, 8 }) |len| {
            if (bits % len != 0) continue;
            const eb = bits / len;
            if (bits > 64 and !byteSized(eb)) continue;
            if (eb == 1) {
                options[n] = try u.intern(.{ .vector = .{ .len = @intCast(len), .elem = try u.intern(.bool) } });
            } else if (eb == 16 or eb == 32 or eb == 64) {
                options[n] = try u.intern(.{ .vector = .{ .len = @intCast(len), .elem = if (u.src.percent(50)) try u.intern(.{ .float = eb }) else try u.intType(u.src.percent(50), eb) } });
            } else {
                options[n] = try u.intern(.{ .vector = .{ .len = @intCast(len), .elem = try u.intType(u.src.percent(50), eb) } });
            }
            n += 1;
        }
        const pick = options[u.src.below(@intCast(n))];
        if (pick == t) return null;
        return pick;
    }

    fn byteSized(bits: u16) bool {
        return bits == 8 or bits == 16 or bits == 32 or bits == 64;
    }

    fn bitSize(u: *const Unit, t: TypeId) ?u16 {
        return switch (u.ty(t)) {
            .int => |i| i.bits,
            .float => |f| f,
            .vector => |s| switch (u.ty(s.elem)) {
                .int => |i| if (i.bits <= 64 and i.bits * s.len <= max_bits) i.bits * s.len else null,
                .float => |f| if (f * s.len <= max_bits) f * s.len else null,
                .bool => s.len,
                else => null,
            },
            else => null,
        };
    }

    fn genConv(u: *Unit, t: TypeId, d: u32) !?NodeId {
        if (u.src.percent(25)) {
            const s = try u.sameBits(t) orelse return null;
            return try u.unary(t, .conv, @backingInt(Conv.bits), try u.genExpr(s, d));
        }
        switch (u.ty(t)) {
            .int => |i| switch (u.src.below(5)) {
                0, 1 => {
                    const s = try u.randInt();
                    if (s == t) return null;
                    return try u.unary(t, .conv, @backingInt(Conv.int_int), try u.genExpr(s, d));
                },
                2 => {
                    const s = try u.intType(i.signed, @intCast(i.bits + u.src.below(max_bits + 1 - i.bits)));
                    if (s == t) return null;
                    return try u.unary(t, .conv, @backingInt(Conv.int_cast), try u.genExpr(s, d));
                },
                3 => {
                    if (i.bits > 64) return null;
                    return try u.unary(t, .conv, @backingInt(Conv.float_to_int), try u.genLeaf(try u.randFloat()));
                },
                else => {
                    if (i.bits == 1 and i.signed) return null;
                    return try u.unary(t, .conv, @backingInt(Conv.bool_to_int), try u.genExpr(try u.intern(.bool), d));
                },
            },
            .float => {
                if (u.src.percent(50)) return try u.unary(t, .conv, @backingInt(Conv.int_to_float), try u.genExpr(try u.narrowInt(u.src.percent(50)), d));
                const s = try u.randFloat();
                if (s == t) return null;
                return try u.unary(t, .conv, @backingInt(Conv.float_float), try u.genExpr(s, d));
            },
            .vector => |v| switch (u.ty(v.elem)) {
                .int => |i| {
                    const s = try u.withElem(t, try u.intType(i.signed, u.randBits()));
                    if (s == t) return null;
                    return try u.unary(t, .conv, @backingInt(Conv.vec_int), try u.genExpr(s, d));
                },
                .float => {
                    if (u.src.percent(50)) return try u.unary(t, .conv, @backingInt(Conv.vec_int_to_float), try u.genExpr(try u.withElem(t, try u.narrowInt(u.src.percent(50))), d));
                    const s = try u.withElem(t, try u.randFloat());
                    if (s == t) return null;
                    return try u.unary(t, .conv, @backingInt(Conv.vec_float_float), try u.genExpr(s, d));
                },
                else => return null,
            },
            else => return null,
        }
    }

    fn genProngs(u: *Unit, t: TypeId) ![]const Prong {
        const i = u.ty(t).int;
        const min: i512 = if (i.signed) -(@as(i512, 1) << @intCast(i.bits - 1)) else 0;
        const max: i512 = if (i.signed) (@as(i512, 1) << @intCast(i.bits - 1)) - 1 else (@as(i512, 1) << @intCast(i.bits)) - 1;
        const total = max - min + 1;
        var used: std.ArrayList(Item) = .empty;
        var covered: i512 = 0;
        const prongs = try u.arena.alloc(Prong, 1 + u.src.below(4));
        var count: usize = 0;
        for (prongs) |*p| {
            var items: std.ArrayList(Item) = .empty;
            for (0..1 + u.src.below(3)) |_| {
                const base: i512 = switch (u.src.below(4)) {
                    0 => min,
                    1 => max,
                    2 => @as(i512, u.src.below(12)) - (if (i.signed) @as(i512, 4) else 0),
                    else => min + @mod(@as(i512, u.wideWord()), total),
                };
                const lo = std.math.clamp(base, min, max);
                const hi = if (u.src.percent(30)) @min(max, lo + u.src.below(6)) else lo;
                if (covered + (hi - lo + 1) >= total) continue;
                const overlaps = for (used.items) |it| {
                    if (lo <= it.hi and it.lo <= hi) break true;
                } else false;
                if (overlaps) continue;
                try used.append(u.arena, .{ .lo = lo, .hi = hi });
                try items.append(u.arena, .{ .lo = lo, .hi = hi });
                covered += hi - lo + 1;
            }
            if (items.items.len == 0) continue;
            prongs[count] = items.items;
            count += 1;
            _ = p;
        }
        if (count == 0) {
            const one = try u.arena.alloc(Item, 1);
            one[0] = .{ .lo = min, .hi = min };
            prongs[0] = one;
            count = 1;
        }
        return prongs[0..count];
    }

    fn genBlock(u: *Unit, depth: u32, max: u32, in_loop: bool) error{OutOfMemory}![]const StmtId {
        const mark = u.scope.items.len;
        defer u.scope.shrinkRetainingCapacity(mark);
        var list: std.ArrayList(StmtId) = .empty;
        for (0..1 + u.src.below(max)) |_| try u.genStmt(&list, depth, in_loop);
        return list.items;
    }

    fn mutableInScope(u: *Unit) ?VarId {
        var buf: [64]VarId = undefined;
        var n: usize = 0;
        for (u.scope.items) |v| if (u.vars.items[v].mutable and n < buf.len) {
            buf[n] = v;
            n += 1;
        };
        if (n == 0) return null;
        return buf[u.src.below(@intCast(n))];
    }

    fn genAssign(u: *Unit, target: VarId) !StmtId {
        return u.addStmt(.{ .tag = .assign, .v = target, .expr = try u.genExpr(u.valueType(target), 2) });
    }

    fn genStmt(u: *Unit, list: *std.ArrayList(StmtId), depth: u32, in_loop: bool) !void {
        const r = u.src.below(100);
        if (r >= 82 and r < 90) {
            var options: [16]u32 = undefined;
            const n = u.callees(none, false, &options);
            const call = if (n == 0) null else try u.genCall(options[u.src.below(n)], 2);
            const expr = call orelse return u.genStmt(list, 0, in_loop);
            const v = try u.addVar(.{ .ty = u.nodes.items[expr].ty });
            try list.append(u.arena, try u.addStmt(.{ .tag = .decl, .v = v, .expr = expr }));
            try u.scope.append(u.arena, v);
        } else if (r < 34 or depth == 0) {
            const t = try u.pickLocalType();
            const expr = try u.genExpr(t, 3);
            const v = try u.addVar(.{ .ty = t });
            try list.append(u.arena, try u.addStmt(.{ .tag = .decl, .v = v, .expr = expr }));
            try u.scope.append(u.arena, v);
        } else if (r < 50) {
            const t = try u.pickLocalType();
            const expr = try u.genExpr(t, 2);
            const v = try u.addVar(.{ .ty = t, .mutable = true });
            try list.append(u.arena, try u.addStmt(.{ .tag = .decl, .v = v, .expr = expr }));
            try u.scope.append(u.arena, v);
            try u.genStmt(list, depth, in_loop);
        } else if (r < 58) {
            const target = u.mutableInScope() orelse return u.genStmt(list, 0, in_loop);
            try list.append(u.arena, try u.genAssign(target));
        } else if (r < 67) {
            const cond = try u.genExpr(try u.intern(.bool), 2);
            const blocks = try u.arena.alloc([]const StmtId, 2);
            blocks[0] = try u.genBlockAssigning(depth - 1, in_loop);
            blocks[1] = if (u.src.percent(60)) try u.genBlockAssigning(depth - 1, in_loop) else &.{};
            try list.append(u.arena, try u.addStmt(.{ .tag = .if_else, .expr = cond, .blocks = blocks }));
        } else if (r < 74) {
            const subject_t = if (u.src.percent(50)) try u.randInt() else try u.intType(u.src.percent(50), @intCast(1 + u.src.below(12)));
            const subject = try u.genExpr(subject_t, 2);
            const prongs = try u.genProngs(subject_t);
            const blocks = try u.arena.alloc([]const StmtId, prongs.len + 1);
            for (blocks) |*b| b.* = try u.genBlockAssigning(depth - 1, in_loop);
            try list.append(u.arena, try u.addStmt(.{ .tag = .@"switch", .expr = subject, .blocks = blocks, .prongs = prongs }));
        } else if (r < 82) {
            const for_range = u.src.percent(60);
            const bound = if (for_range and u.src.percent(50)) try u.genExpr(try u.intType(false, 32), 2) else none;
            const counter = try u.addVar(.{ .ty = try u.intType(false, 32), .read_only = true });
            try u.scope.append(u.arena, counter);
            const blocks = try u.arena.alloc([]const StmtId, 1);
            blocks[0] = try u.genBlockAssigning(depth - 1, true);
            u.scope.items.len -= 1;
            try list.append(u.arena, try u.addStmt(.{
                .tag = .loop,
                .v = counter,
                .expr = bound,
                .blocks = blocks,
                .count = 1 + u.src.below(8),
                .for_range = for_range,
            }));
        } else if (r < 95 and !u.is_test) {
            const cond = try u.genExpr(try u.intern(.bool), 2);
            const value = try u.genExpr(u.funcs.items[u.cur].ret, 2);
            try list.append(u.arena, try u.addStmt(.{ .tag = .return_if, .expr = cond, .value = value }));
        } else if (in_loop) {
            const cond = try u.genExpr(try u.intern(.bool), 2);
            try list.append(u.arena, try u.addStmt(.{ .tag = if (u.src.percent(50)) .break_if else .continue_if, .expr = cond }));
        } else {
            try u.genStmt(list, 0, in_loop);
        }
    }

    fn genBlockAssigning(u: *Unit, depth: u32, in_loop: bool) ![]const StmtId {
        const body = try u.genBlock(depth, 3, in_loop);
        const target = u.mutableInScope() orelse return body;
        if (!u.src.percent(80)) return body;
        const result = try u.arena.alloc(StmtId, body.len + 1);
        @memcpy(result[0..body.len], body);
        result[body.len] = try u.genAssign(target);
        return result;
    }

    pub fn generate(u: *Unit) !void {
        try u.pool.append(u.arena, try u.intType(false, 32));
        try u.pool.append(u.arena, try u.intern(.bool));
        for (0..4 + u.src.below(6)) |_| try u.pool.append(u.arena, try u.randType());

        const helper_count = u.src.below(7);
        while (u.funcs.items.len < helper_count) {
            if (u.src.percent(25)) {
                for (0..@min(1 + u.src.below(2), helper_count - u.funcs.items.len)) |_| try u.declareHelper(u.groups);
                u.groups += 1;
            } else {
                try u.declareHelper(none);
            }
        }
        const params = try u.arena.alloc(VarId, input_words);
        for (params) |*p| p.* = try u.addVar(.{ .ty = try u.intType(false, 64), .read_only = true });
        try u.funcs.append(u.arena, .{ .params = params, .ret = try u.intType(false, 32), .body = &.{}, .ret_expr = none });

        for (u.funcs.items, 0..) |f, fi| {
            u.cur = @intCast(fi);
            u.is_test = fi == u.testFunc();
            u.scope.clearRetainingCapacity();
            u.first_var = u.vars.items.len;
            try u.scope.appendSlice(u.arena, f.params);
            if (u.is_test) {
                u.funcs.items[fi].body = try u.genBlock(2, 10, false);
                continue;
            }
            const first_var = u.vars.items.len;
            if (f.group != none) {
                u.in_base = true;
                u.funcs.items[fi].base = try u.genExpr(f.ret, 2);
                u.in_base = false;
            }
            u.funcs.items[fi].body = if (u.src.percent(75)) try u.genBlock(2, 4, false) else &.{};
            u.funcs.items[fi].ret_expr = try u.genExpr(f.ret, 3);
            var var_bytes: u32 = 0;
            for (f.params) |v| var_bytes += u.byteSize(u.vars.items[v].ty);
            for (u.vars.items[first_var..]) |v| var_bytes += u.byteSize(v.ty);
            u.funcs.items[fi].var_bytes = var_bytes;
        }
        if (u.src.percent(50)) u.by_value = @intCast(u.src.below(1 << input_words));
    }

    /// Returns an estimate of the size of a value of type `t` in bytes.
    fn byteSize(u: *const Unit, t: TypeId) u32 {
        return switch (u.ty(t)) {
            .int => |i| if (i.bits <= 64) std.math.ceilPowerOfTwoAssert(u32, @max(8, i.bits)) / 8 else 8 * ((i.bits + 63) / 64),
            .float => |f| f / 8,
            .bool => 1,
            .vector, .array => |s| s.len * u.byteSize(s.elem),
            .@"struct" => |fields| sum: {
                var bytes: u32 = 0;
                for (fields) |f| bytes += u.byteSize(f);
                break :sum bytes;
            },
            .optional, .error_union => |p| u.byteSize(p) + 8,
            .pointer => 8,
        };
    }

    /// Returns the number of stack frames of group `group` that a call from outside of the group
    /// may create. A CUDA thread has 1 KiB of stack by default, and the helpers of all groups
    /// together get half of the stack. The function assumes that a helper needs at most 64 bytes
    /// plus twice the size of its variables for its stack frame.
    fn boundLimit(u: *const Unit, group: u32) u32 {
        const stack_bytes = 512;
        var frame_bytes: u32 = 64;
        for (u.funcs.items) |f| {
            if (f.group == group) frame_bytes = @max(frame_bytes, 64 + 2 * f.var_bytes);
        }
        return std.math.clamp(stack_bytes / u.groups / frame_bytes, 1, 4);
    }

    /// Adds the signature of a helper. Some helpers copy the signature of an earlier helper so that
    /// a function pointer can select between them.
    fn declareHelper(u: *Unit, group: u32) !void {
        var params: std.ArrayList(VarId) = .empty;
        var ret: TypeId = undefined;
        var bound: VarId = none;
        if (group != none) {
            bound = try u.addVar(.{ .ty = try u.intType(false, 32), .read_only = true });
            try params.append(u.arena, bound);
        }
        const sibling = if (u.funcs.items.len == 0) none else u.src.below(@intCast(u.funcs.items.len));
        if (group == none and sibling != none and u.funcs.items[sibling].group == none and u.src.percent(40)) {
            const s = u.funcs.items[sibling];
            for (s.params) |p| try params.append(u.arena, try u.addVar(u.vars.items[p]));
            ret = s.ret;
        } else {
            for (0..u.src.below(5)) |_| {
                const t = try u.pickType();
                if (u.src.percent(75)) {
                    try params.append(u.arena, try u.addVar(.{ .ty = t, .read_only = true }));
                    continue;
                }
                const mutable = u.src.percent(60);
                const pt = try u.intern(.{ .pointer = .{ .child = t, .mutable = mutable } });
                try params.append(u.arena, try u.addVar(.{ .ty = pt, .read_only = true, .mutable = mutable }));
            }
            ret = try u.pickType();
        }
        try u.funcs.append(u.arena, .{
            .params = params.items,
            .ret = ret,
            .body = &.{},
            .ret_expr = none,
            .is_noinline = u.src.percent(40),
            .group = group,
            .bound = bound,
        });
    }

    fn testFunc(u: *const Unit) u32 {
        return @intCast(u.funcs.items.len - 1);
    }

    fn liveBlock(u: *const Unit, stmt: Stmt, block: usize) bool {
        _ = u;
        return stmt.flatten == 0 or stmt.flatten == block + 1;
    }

    pub fn collectEdits(u: *const Unit, gpa: Allocator, edits: *std.ArrayList(Edit)) !void {
        for (u.funcs.items[0..u.testFunc()], 0..) |f, i| if (!f.deleted) try edits.append(gpa, .{ .delete_func = @intCast(i) });
        for (u.funcs.items) |f| {
            if (f.deleted) continue;
            try u.collectBlockStmts(gpa, f.body, edits);
        }
        for (u.funcs.items) |f| {
            if (f.deleted) continue;
            try u.collectBlockNodes(gpa, f.body, edits);
            if (f.ret_expr != none) try u.collectNode(gpa, f.ret_expr, edits);
            if (f.base != none) try u.collectNode(gpa, f.base, edits);
        }
    }

    fn collectBlockStmts(u: *const Unit, gpa: Allocator, block: []const StmtId, edits: *std.ArrayList(Edit)) error{OutOfMemory}!void {
        for (block) |si| {
            const s = u.stmts.items[si];
            if (s.deleted) continue;
            try edits.append(gpa, .{ .delete_stmt = si });
            if (s.flatten == 0) for (s.blocks, 0..) |_, bi| try edits.append(gpa, .{ .flatten = .{ .stmt = si, .block = @intCast(bi) } });
            for (s.blocks, 0..) |b, bi| if (u.liveBlock(s, bi)) try u.collectBlockStmts(gpa, b, edits);
        }
    }

    fn collectBlockNodes(u: *const Unit, gpa: Allocator, block: []const StmtId, edits: *std.ArrayList(Edit)) error{OutOfMemory}!void {
        for (block) |si| {
            const s = u.stmts.items[si];
            if (s.deleted) continue;
            if (s.expr != none and s.flatten == 0) try u.collectNode(gpa, s.expr, edits);
            if (s.value != none) try u.collectNode(gpa, s.value, edits);
            for (s.blocks, 0..) |b, bi| if (u.liveBlock(s, bi)) try u.collectBlockNodes(gpa, b, edits);
        }
    }

    fn collectNode(u: *const Unit, gpa: Allocator, ni: NodeId, edits: *std.ArrayList(Edit)) error{OutOfMemory}!void {
        const n = u.nodes.items[ni];
        if (n.op == .var_ref or n.op == .input or n.op == .addr_of) {
            if (n.op == .input and n.b != 0) try edits.append(gpa, .{ .zero = ni });
            return;
        }
        if (n.op != .lit or n.value != 0) try edits.append(gpa, .{ .zero = ni });
        if (u.callsDeleted(n)) return;
        for (n.kids, 0..) |k, ki| if (u.nodes.items[k].ty == n.ty) try edits.append(gpa, .{ .hoist = .{ .node = ni, .kid = @intCast(ki) } });
        for (n.kids) |k| try u.collectNode(gpa, k, edits);
    }

    fn callsDeleted(u: *const Unit, n: Node) bool {
        return switch (n.op) {
            .call => u.funcs.items[n.a].deleted,
            .call_ptr => for (u.fn_sets.items[n.a]) |fi| {
                if (u.funcs.items[fi].deleted) break true;
            } else false,
            else => false,
        };
    }

    /// Replaces each call that passes a pointer to a deleted variable with a zero.
    fn dropDanglingCalls(u: *Unit) void {
        for (u.nodes.items) |*n| {
            if (n.op != .call and n.op != .call_ptr) continue;
            for (n.kids) |k| {
                const kid = u.nodes.items[k];
                if (kid.op != .addr_of or !u.vars.items[kid.a].deleted) continue;
                n.* = .{ .ty = n.ty, .op = .lit };
                break;
            }
        }
    }

    fn deleteBlockVars(u: *Unit, block: []const StmtId) void {
        for (block) |si| {
            const s = u.stmts.items[si];
            if (s.v != none and (s.tag == .decl or s.tag == .loop)) u.vars.items[s.v].deleted = true;
            for (s.blocks) |b| u.deleteBlockVars(b);
        }
    }

    pub fn apply(u: *Unit, edit: Edit) void {
        switch (edit) {
            .delete_func => |fi| u.funcs.items[fi].deleted = true,
            .delete_stmt => |si| {
                const s = &u.stmts.items[si];
                s.deleted = true;
                if (s.tag == .decl or s.tag == .loop) u.vars.items[s.v].deleted = true;
                for (s.blocks) |b| u.deleteBlockVars(b);
            },
            .flatten => |f| {
                const s = &u.stmts.items[f.stmt];
                s.flatten = f.block + 1;
                if (s.tag == .loop) u.vars.items[s.v].deleted = true;
                for (s.blocks, 0..) |b, bi| if (bi != f.block) u.deleteBlockVars(b);
            },
            .zero => |ni| {
                const n = &u.nodes.items[ni];
                if (n.op == .input) {
                    n.b = 0;
                } else {
                    n.* = .{ .ty = n.ty, .op = .lit };
                }
            },
            .hoist => |h| u.nodes.items[h.node] = u.nodes.items[u.nodes.items[h.node].kids[h.kid]],
        }
        u.dropDanglingCalls();
    }

    /// Holds a copy of every list that `apply` changes.
    pub const Snapshot = struct {
        vars: []const Var,
        nodes: []const Node,
        stmts: []const Stmt,
        funcs: []const Func,
    };

    pub fn snapshot(u: *const Unit) !Snapshot {
        return .{
            .vars = try u.arena.dupe(Var, u.vars.items),
            .nodes = try u.arena.dupe(Node, u.nodes.items),
            .stmts = try u.arena.dupe(Stmt, u.stmts.items),
            .funcs = try u.arena.dupe(Func, u.funcs.items),
        };
    }

    pub fn restore(u: *Unit, s: Snapshot) void {
        @memcpy(u.vars.items, s.vars);
        @memcpy(u.nodes.items, s.nodes);
        @memcpy(u.stmts.items, s.stmts);
        @memcpy(u.funcs.items, s.funcs);
    }

    fn markBlock(u: *Unit, block: []const StmtId) void {
        for (block) |si| {
            const s = u.stmts.items[si];
            if (s.deleted) continue;
            if (s.tag == .assign) {
                if (u.vars.items[s.v].deleted) continue;
                u.vars.items[s.v].assigned = true;
                u.vars.items[s.v].referenced = true;
            }
            if (s.expr != none and s.flatten == 0) u.markNode(s.expr);
            if (s.value != none) u.markNode(s.value);
            for (s.blocks, 0..) |b, bi| if (u.liveBlock(s, bi)) u.markBlock(b);
        }
    }

    fn markNode(u: *Unit, ni: NodeId) void {
        const n = u.nodes.items[ni];
        switch (n.op) {
            .var_ref, .deref => u.vars.items[n.a].referenced = true,
            .pick_ref => {
                u.vars.items[n.a].referenced = true;
                u.vars.items[n.b].referenced = true;
            },
            .addr_of => {
                u.vars.items[n.a].referenced = true;
                if (u.ty(n.ty).pointer.mutable) u.vars.items[n.a].assigned = true;
            },
            .input => u.vars.items[u.funcs.items[u.testFunc()].params[n.a]].referenced = true,
            .call, .call_ptr => if (u.callsDeleted(n)) return,
            else => {},
        }
        for (n.kids) |k| u.markNode(k);
    }

    pub fn print(u: *Unit, w: *Writer) !void {
        for (u.vars.items) |*v| {
            v.referenced = false;
            v.assigned = false;
        }
        for (u.funcs.items) |f| {
            if (f.deleted) continue;
            u.markBlock(f.body);
            if (f.ret_expr != none) u.markNode(f.ret_expr);
            if (f.base != none) u.markNode(f.base);
            if (f.bound != none) u.vars.items[f.bound].referenced = true;
        }
        for (u.types.items, 0..) |t, ti| switch (t) {
            .@"struct" => |fields| {
                try w.print("const S{d}_{d} = struct {{", .{ u.index, ti });
                for (fields, 0..) |f, fi| {
                    try w.print(" f{d}: ", .{fi});
                    try u.printType(w, f);
                    try w.writeAll(",");
                }
                try w.writeAll(" };\n");
            },
            else => {},
        };
        for (u.funcs.items, 0..) |f, fi| {
            if (f.deleted) continue;
            u.is_test = fi == u.testFunc();
            if (u.is_test) {
                try w.print("pub fn t{d}(", .{u.index});
            } else {
                try w.print("{s}fn t{d}_h{d}(", .{ if (f.is_noinline) "noinline " else "", u.index, fi });
            }
            for (f.params, 0..) |p, pi| {
                if (pi != 0) try w.writeAll(", ");
                try w.print("v{d}: ", .{p});
                try u.printType(w, u.vars.items[p].ty);
            }
            try w.writeAll(") ");
            try u.printType(w, f.ret);
            try w.writeAll(" {\n");
            if (u.is_test) try w.print("    var cs: u32 = {d};\n", .{u.index});
            for (f.params) |p| if (!u.vars.items[p].referenced) try w.print("    _ = &v{d};\n", .{p});
            if (f.bound != none) {
                try w.print("    if (v{d} == 0) return ", .{f.bound});
                try u.printNode(w, f.base);
                try w.writeAll(";\n");
            }
            try u.printBlockBody(w, f.body, 1);
            if (u.is_test) {
                try w.writeAll("    cs = mix(cs, 1);\n    return cs;\n}\n");
            } else {
                try w.writeAll("    return ");
                try u.printNode(w, f.ret_expr);
                try w.writeAll(";\n}\n");
            }
        }
    }

    fn indent(w: *Writer, level: u32) !void {
        for (0..level) |_| try w.writeAll("    ");
    }

    fn printBlockBody(u: *Unit, w: *Writer, block: []const StmtId, level: u32) !void {
        for (block) |si| try u.printStmt(w, si, level);
        for (block) |si| {
            const s = u.stmts.items[si];
            if (s.deleted or s.tag != .decl) continue;
            const v = u.vars.items[s.v];
            if (!v.mutable or !v.assigned) continue;
            if (u.is_test) {
                var name_buf: [32]u8 = undefined;
                try u.printHash(w, try std.fmt.bufPrint(&name_buf, "v{d}", .{s.v}), v.ty, level);
            }
        }
    }

    fn printStmt(u: *Unit, w: *Writer, si: StmtId, level: u32) PrintError!void {
        const s = u.stmts.items[si];
        if (s.deleted) return;
        if (s.flatten != 0) {
            try indent(w, level);
            try w.writeAll("{\n");
            try u.printBlockBody(w, s.blocks[s.flatten - 1], level + 1);
            try indent(w, level);
            try w.writeAll("}\n");
            return;
        }
        switch (s.tag) {
            .decl => {
                const v = u.vars.items[s.v];
                try indent(w, level);
                try w.print("{s} v{d}: ", .{ if (v.mutable and v.assigned) "var" else "const", s.v });
                try u.printType(w, v.ty);
                try w.writeAll(" = ");
                try u.printNode(w, s.expr);
                try w.writeAll(";\n");
                if (v.mutable and v.assigned) return;
                if (u.is_test) {
                    var name_buf: [32]u8 = undefined;
                    try u.printHash(w, try std.fmt.bufPrint(&name_buf, "v{d}", .{s.v}), v.ty, level);
                } else if (!v.referenced) {
                    try indent(w, level);
                    try w.print("_ = &v{d};\n", .{s.v});
                }
            },
            .assign => {
                if (u.vars.items[s.v].deleted) return;
                try indent(w, level);
                try w.print("v{d}{s} = ", .{ s.v, if (u.valueType(s.v) == u.vars.items[s.v].ty) "" else ".*" });
                try u.printNode(w, s.expr);
                try w.writeAll(";\n");
            },
            .return_if => {
                try indent(w, level);
                try w.writeAll("if (");
                try u.printNode(w, s.expr);
                try w.writeAll(") return ");
                try u.printNode(w, s.value);
                try w.writeAll(";\n");
            },
            .if_else => {
                try indent(w, level);
                try w.writeAll("if (");
                try u.printNode(w, s.expr);
                try w.writeAll(") {\n");
                try u.printBlockBody(w, s.blocks[0], level + 1);
                try indent(w, level);
                if (s.blocks[1].len > 0) {
                    try w.writeAll("} else {\n");
                    try u.printBlockBody(w, s.blocks[1], level + 1);
                    try indent(w, level);
                }
                try w.writeAll("}\n");
            },
            .@"switch" => {
                try indent(w, level);
                try w.writeAll("switch (");
                try u.printNode(w, s.expr);
                try w.writeAll(") {\n");
                for (s.prongs, 0..) |p, pi| {
                    try indent(w, level + 1);
                    try printItems(w, p);
                    try w.writeAll(" => {\n");
                    try u.printBlockBody(w, s.blocks[pi], level + 2);
                    try indent(w, level + 1);
                    try w.writeAll("},\n");
                }
                try indent(w, level + 1);
                try w.writeAll("else => {\n");
                try u.printBlockBody(w, s.blocks[s.prongs.len], level + 2);
                try indent(w, level + 1);
                try w.writeAll("},\n");
                try indent(w, level);
                try w.writeAll("}\n");
            },
            .loop => if (s.for_range) {
                try indent(w, level);
                try w.writeAll("for (0..");
                if (s.expr != none) {
                    try w.writeAll("@as(u32, ");
                    try u.printNode(w, s.expr);
                    try w.print(") % {d}", .{s.count + 1});
                } else try w.print("{d}", .{s.count});
                // Names such as `i9` are types, and other captures use `c{d}` with these numbers.
                try w.print(") |range{d}| {{\n", .{s.v});
                try indent(w, level + 1);
                try w.print("const v{d}: u32 = @intCast(range{d});\n", .{ s.v, s.v });
                try indent(w, level + 1);
                try w.print("_ = &v{d};\n", .{s.v});
                try u.printBlockBody(w, s.blocks[0], level + 1);
                try indent(w, level);
                try w.writeAll("}\n");
            } else {
                try indent(w, level);
                try w.print("var v{d}: u32 = 0;\n", .{s.v});
                try indent(w, level);
                try w.print("while (v{d} < {d}) : (v{d} += 1) {{\n", .{ s.v, s.count, s.v });
                try u.printBlockBody(w, s.blocks[0], level + 1);
                try indent(w, level);
                try w.writeAll("}\n");
            },
            .break_if, .continue_if => {
                try indent(w, level);
                try w.writeAll("if (");
                try u.printNode(w, s.expr);
                try w.print(") {s};\n", .{if (s.tag == .break_if) "break" else "continue"});
            },
        }
    }

    fn printItems(w: *Writer, p: Prong) !void {
        for (p, 0..) |it, ii| {
            if (ii != 0) try w.writeAll(", ");
            if (it.lo == it.hi) {
                try w.print("{d}", .{it.lo});
            } else {
                try w.print("{d}...{d}", .{ it.lo, it.hi });
            }
        }
    }

    fn printHash(u: *Unit, w: *Writer, name: []const u8, t: TypeId, level: u32) !void {
        var buf: [64]u8 = undefined;
        switch (u.ty(t)) {
            .int => |i| if (i.bits > 64) {
                var shift: u16 = 0;
                while (shift < i.bits) : (shift += 64) {
                    try indent(w, level);
                    try w.print("cs = mix(cs, @as(u64, @truncate(@as(u{d}, @bitCast({s})) >> {d})));\n", .{ i.bits, name, shift });
                }
                return;
            },
            else => {},
        }
        switch (u.ty(t)) {
            .int, .float, .bool => {
                try indent(w, level);
                try w.writeAll("cs = mix(cs, ");
                try u.printScalarHash(w, name, t);
                try w.writeAll(");\n");
            },
            .vector => |s| for (0..s.len) |i| try u.printHash(w, try std.fmt.bufPrint(&buf, "{s}[{d}]", .{ name, i }), s.elem, level),
            .array => |s| for (0..s.len) |i| try u.printHash(w, try std.fmt.bufPrint(&buf, "{s}[{d}]", .{ name, i }), s.elem, level),
            .@"struct" => |fields| for (fields, 0..) |f, i| try u.printHash(w, try std.fmt.bufPrint(&buf, "{s}.f{d}", .{ name, i }), f, level),
            .optional => |p| {
                const c = u.captures;
                u.captures += 1;
                try indent(w, level);
                try w.print("if ({s}) |c{d}| {{\n", .{ name, c });
                try u.printHash(w, try std.fmt.bufPrint(&buf, "c{d}", .{c}), p, level + 1);
                try indent(w, level);
                try w.writeAll("} else {\n");
                try indent(w, level + 1);
                try w.writeAll("cs = mix(cs, 0x5a5a);\n");
                try indent(w, level);
                try w.writeAll("}\n");
            },
            .error_union => |p| {
                const c = u.captures;
                u.captures += 1;
                try indent(w, level);
                try w.print("if ({s}) |c{d}| {{\n", .{ name, c });
                try u.printHash(w, try std.fmt.bufPrint(&buf, "c{d}", .{c}), p, level + 1);
                try indent(w, level);
                try w.print("}} else |e{d}| {{\n", .{c});
                try indent(w, level + 1);
                try w.print("cs = mix(cs, errorCode(e{d}));\n", .{c});
                try indent(w, level);
                try w.writeAll("}\n");
            },
            .pointer => unreachable,
        }
    }

    fn printScalarHash(u: *Unit, w: *Writer, name: []const u8, t: TypeId) !void {
        switch (u.ty(t)) {
            .int => |i| if (i.signed) {
                try w.print("@as(u64, @as(u{d}, @bitCast({s})))", .{ i.bits, name });
            } else {
                try w.print("@as(u64, {s})", .{name});
            },
            .float => |f| try w.print("floatBits{d}({s})", .{ f, name }),
            .bool => try w.print("@as(u64, @intFromBool({s}))", .{name}),
            else => unreachable,
        }
    }

    fn printType(u: *Unit, w: *Writer, t: TypeId) PrintError!void {
        switch (u.ty(t)) {
            .int => |i| try w.print("{c}{d}", .{ @as(u8, if (i.signed) 'i' else 'u'), i.bits }),
            .float => |f| try w.print("f{d}", .{f}),
            .bool => try w.writeAll("bool"),
            .vector => |s| {
                try w.print("@Vector({d}, ", .{s.len});
                try u.printType(w, s.elem);
                try w.writeAll(")");
            },
            .array => |s| {
                try w.print("[{d}]", .{s.len});
                try u.printType(w, s.elem);
            },
            .@"struct" => try w.print("S{d}_{d}", .{ u.index, t }),
            .optional => |p| {
                try w.writeAll("?");
                try u.printType(w, p);
            },
            .error_union => |p| {
                try w.writeAll("Error!");
                try u.printType(w, p);
            },
            .pointer => |p| {
                try w.writeAll(if (p.mutable) "*" else "*const ");
                try u.printType(w, p.child);
            },
        }
    }

    fn printIntValue(w: *Writer, i: Int, bits: u256) !void {
        if (i.signed and (bits >> @intCast(i.bits - 1)) & 1 == 1) {
            try w.print("{d}", .{@as(i512, bits) - (@as(i512, 1) << @intCast(i.bits))});
        } else {
            try w.print("{d}", .{bits});
        }
    }

    fn printLit(u: *Unit, w: *Writer, t: TypeId, value: u256) PrintError!void {
        switch (u.ty(t)) {
            .int => |i| {
                try w.print("@as({c}{d}, ", .{ @as(u8, if (i.signed) 'i' else 'u'), i.bits });
                try printIntValue(w, i, value);
                try w.writeAll(")");
            },
            .float => |f| try w.print("@as(f{d}, @bitCast(@as(u{d}, 0x{x})))", .{ f, f, value }),
            .bool => try w.writeAll(if (value != 0) "true" else "false"),
            .vector, .array => |s| {
                try w.writeAll("@as(");
                try u.printType(w, t);
                try w.writeAll(", @splat(");
                try u.printLit(w, s.elem, 0);
                try w.writeAll("))");
            },
            .@"struct" => |fields| {
                try u.printType(w, t);
                try w.writeAll("{");
                for (fields, 0..) |f, fi| {
                    try w.print(" .f{d} = ", .{fi});
                    try u.printLit(w, f, 0);
                    try w.writeAll(",");
                }
                try w.writeAll(" }");
            },
            .optional => {
                try w.writeAll("@as(");
                try u.printType(w, t);
                try w.writeAll(", null)");
            },
            .error_union => |p| {
                try w.writeAll("@as(");
                try u.printType(w, t);
                try w.writeAll(", ");
                try u.printLit(w, p, 0);
                try w.writeAll(")");
            },
            .pointer => unreachable,
        }
    }

    fn kidText(u: *Unit, w: *Writer, n: Node, i: usize) !void {
        try u.printNode(w, n.kids[i]);
    }

    fn printCast(u: *Unit, w: *Writer, t: TypeId, builtin: []const u8, n: Node) !void {
        try w.writeAll("@as(");
        try u.printType(w, t);
        try w.print(", {s}(", .{builtin});
        try u.kidText(w, n, 0);
        try w.writeAll("))");
    }

    fn printIntConv(u: *Unit, w: *Writer, n: Node) !void {
        const to = u.ty(n.ty).int;
        const from = u.ty(u.nodes.items[n.kids[0]].ty).int;
        if ((from.signed == to.signed and from.bits <= to.bits) or (!from.signed and to.signed and from.bits < to.bits)) {
            try w.writeAll("@as(");
            try u.printType(w, n.ty);
            try w.writeAll(", ");
            try u.kidText(w, n, 0);
            try w.writeAll(")");
            return;
        }
        if (to.signed) try w.print("@as(i{d}, @bitCast(", .{to.bits});
        if (from.bits >= to.bits) {
            try w.print("@as(u{d}, @truncate(", .{to.bits});
        } else {
            try w.print("@as(u{d}, ", .{to.bits});
        }
        if (from.signed) try w.print("@as(u{d}, @bitCast(", .{from.bits});
        try u.kidText(w, n, 0);
        if (from.signed) try w.writeAll("))");
        try w.writeAll(if (from.bits >= to.bits) "))" else ")");
        if (to.signed) try w.writeAll("))");
    }

    fn printNode(u: *Unit, w: *Writer, ni: NodeId) PrintError!void {
        const n = u.nodes.items[ni];
        switch (n.op) {
            .var_ref => if (u.vars.items[n.a].deleted) try u.printLit(w, n.ty, 0) else try w.print("v{d}", .{n.a}),
            .lit => try u.printLit(w, n.ty, n.value),
            .input => {
                const p = u.funcs.items[u.testFunc()].params[n.a];
                switch (u.ty(n.ty)) {
                    .int => |i| if (i.bits > 64) {
                        const ps = u.funcs.items[u.testFunc()].params;
                        try w.print("@as({c}{d}, @bitCast(@as(u{d}, @truncate(wideInput(", .{ @as(u8, if (i.signed) 'i' else 'u'), i.bits, i.bits });
                        for (0..input_words) |k| try w.print("{s}v{d}", .{ if (k == 0) "" else ", ", ps[(n.a + k) % input_words] });
                        try w.print(") >> {d}))))", .{n.b});
                    } else if (i.signed) {
                        try w.print("@as(i{d}, @bitCast(@as(u{d}, @truncate(v{d} >> {d}))))", .{ i.bits, i.bits, p, n.b });
                    } else {
                        try w.print("@as(u{d}, @truncate(v{d} >> {d}))", .{ i.bits, p, n.b });
                    },
                    .float => |f| try w.print("@as(f{d}, @bitCast(@as(u{d}, @truncate(v{d} >> {d}))))", .{ f, f, p, n.b }),
                    .bool => try w.print("((v{d} >> {d}) & 1 == 1)", .{ p, n.b }),
                    else => unreachable,
                }
            },
            .bin => if (n.b == @backingInt(BinOp.bit_or) and u.isWideSigned(n.ty)) {
                try u.printWideSignedOr(w, n);
            } else {
                try w.writeAll("(");
                try u.kidText(w, n, 0);
                try w.print(" {s} ", .{BinOp.text[n.b]});
                try u.kidText(w, n, 1);
                try w.writeAll(")");
            },
            .shift => {
                const bits = u.ty(u.scalarOf(n.ty)).int.bits;
                const log2 = std.math.log2_int_ceil(u16, bits);
                const amount_t = u.ty(u.nodes.items[n.kids[1]].ty).int;
                try w.writeAll("(");
                try u.kidText(w, n, 0);
                try w.writeAll(if (n.b == 0) " << " else " >> ");
                if (u.ty(n.ty) == .vector) {
                    try w.writeAll("@as(");
                    try u.printType(w, try u.withElem(n.ty, try u.intType(false, log2)));
                    try w.writeAll(", @splat(");
                } else {
                    try w.print("@as(u{d}, ", .{log2});
                }
                try w.writeAll("@intCast(@as(u64, ");
                _ = amount_t;
                try u.kidText(w, n, 1);
                try w.print(") % {d}))", .{bits});
                if (u.ty(n.ty) == .vector) try w.writeAll(")");
                try w.writeAll(")");
            },
            .sat_shl => {
                try w.writeAll("(");
                try u.kidText(w, n, 0);
                try w.writeAll(" <<| ");
                try u.kidText(w, n, 1);
                try w.writeAll(")");
            },
            .div => try u.printDiv(w, n),
            .un => {
                const op: UnOp = @fromBackingInt(@intCast(n.b));
                switch (op) {
                    .bit_not, .neg_wrap, .neg, .bool_not => {
                        try w.print("({s}", .{UnOp.text[n.b]});
                        try u.kidText(w, n, 0);
                        try w.writeAll(")");
                    },
                    else => {
                        try w.print("{s}(", .{UnOp.text[n.b]});
                        try u.kidText(w, n, 0);
                        try w.writeAll(")");
                    },
                }
            },
            .abs_signed => {
                try w.writeAll("@abs(");
                try u.kidText(w, n, 0);
                try w.writeAll(")");
            },
            .min_max => {
                const float = u.ty(u.scalarOf(n.ty)) == .float;
                if (float) try w.writeAll("(");
                try w.writeAll("@as(");
                try u.printType(w, n.ty);
                try w.writeAll(if (n.b == 0) ", @min(" else ", @max(");
                try u.kidText(w, n, 0);
                try w.writeAll(", ");
                try u.kidText(w, n, 1);
                try w.writeAll("))");
                if (float) try u.printPlusZero(w, n.ty);
            },
            .cmp => {
                try w.writeAll("(");
                try u.kidText(w, n, 0);
                try w.print(" {s} ", .{CmpOp.text[n.b]});
                try u.kidText(w, n, 1);
                try w.writeAll(")");
            },
            .not => {
                try w.writeAll("(!");
                try u.kidText(w, n, 0);
                try w.writeAll(")");
            },
            .and_or => {
                try w.writeAll("(");
                try u.kidText(w, n, 0);
                try w.writeAll(if (n.b == 0) " and " else " or ");
                try u.kidText(w, n, 1);
                try w.writeAll(")");
            },
            .select => {
                try w.writeAll("@select(");
                try u.printType(w, u.scalarOf(n.ty));
                try w.writeAll(", ");
                try u.kidText(w, n, 0);
                try w.writeAll(", ");
                try u.kidText(w, n, 1);
                try w.writeAll(", ");
                try u.kidText(w, n, 2);
                try w.writeAll(")");
            },
            .splat => try u.printCast(w, n.ty, "@splat", n),
            .init => {
                try u.printType(w, n.ty);
                const t = u.ty(n.ty);
                if (t == .array) {
                    try w.writeAll("{ ");
                } else {
                    try w.writeAll("{");
                }
                for (n.kids, 0..) |_, ki| {
                    if (t == .@"struct") {
                        try w.print(" .f{d} = ", .{ki});
                    } else if (ki != 0) {
                        try w.writeAll(", ");
                    }
                    try u.kidText(w, n, ki);
                    if (t == .@"struct") try w.writeAll(",");
                }
                try w.writeAll(" }");
            },
            .elem => {
                try w.writeAll("(");
                try u.kidText(w, n, 0);
                try w.print(")[{d}]", .{n.a});
            },
            .elem_runtime => {
                const aggregate_t = u.ty(u.nodes.items[n.kids[0]].ty);
                const len = switch (aggregate_t) {
                    .vector, .array => |s| s.len,
                    else => unreachable,
                };
                const it = u.ty(u.nodes.items[n.kids[1]].ty).int;
                if (aggregate_t == .vector) {
                    // The inner `@as` keeps the array type from becoming the result type of the
                    // vector expression.
                    try w.print("@as([{d}]", .{len});
                    try u.printType(w, n.ty);
                    try w.writeAll(", @as(");
                    try u.printType(w, u.nodes.items[n.kids[0]].ty);
                    try w.writeAll(", ");
                    try u.kidText(w, n, 0);
                    try w.writeAll(")");
                } else {
                    try w.writeAll("(");
                    try u.kidText(w, n, 0);
                }
                try w.writeAll(")[@as(usize, @intCast(@as(u64, ");
                if (it.signed) try w.print("@as(u{d}, @bitCast(", .{it.bits});
                try u.kidText(w, n, 1);
                if (it.signed) try w.writeAll("))");
                try w.print(") % {d}))]", .{len});
            },
            .field => {
                try w.writeAll("(");
                try u.kidText(w, n, 0);
                try w.print(").f{d}", .{n.a});
            },
            .reduce => {
                const op: ReduceOp = @fromBackingInt(@intCast(n.b));
                const float_min_max = u.ty(n.ty) == .float and (op == .Min or op == .Max);
                if (float_min_max) try w.writeAll("(");
                try w.print("@reduce(.{t}, ", .{op});
                try u.kidText(w, n, 0);
                try w.writeAll(")");
                if (float_min_max) try u.printPlusZero(w, n.ty);
            },
            .conv => try u.printConv(w, n),
            .bitcount => {
                const names = [_][]const u8{ "@popCount", "@clz", "@ctz" };
                try w.writeAll("@as(");
                try u.printType(w, n.ty);
                try w.print(", {s}(", .{names[n.b]});
                try u.kidText(w, n, 0);
                try w.writeAll("))");
            },
            .overflow => {
                const names = [_][]const u8{ "@addWithOverflow", "@subWithOverflow", "@mulWithOverflow", "@shlWithOverflow" };
                try w.print("{s}(", .{names[n.b]});
                try u.kidText(w, n, 0);
                try w.writeAll(", ");
                if (n.b == 3) {
                    const bits = u.ty(u.nodes.items[n.kids[0]].ty).int.bits;
                    try w.print("@as(u{d}, @intCast(@as(u64, ", .{std.math.log2_int_ceil(u16, bits)});
                    try u.kidText(w, n, 1);
                    try w.print(") % {d}))", .{bits});
                } else {
                    try u.kidText(w, n, 1);
                }
                try w.print(")[{d}]", .{n.a});
            },
            .mul_add => {
                try w.writeAll("@mulAdd(");
                try u.printType(w, n.ty);
                for (0..3) |i| {
                    try w.writeAll(", ");
                    try u.kidText(w, n, i);
                }
                try w.writeAll(")");
            },
            .call => {
                if (u.callsDeleted(n)) return u.printLit(w, n.ty, 0);
                var name_buf: [32]u8 = undefined;
                const name = try std.fmt.bufPrint(&name_buf, "t{d}_h{d}", .{ u.index, n.a });
                switch (@as(CallKind, @fromBackingInt(@intCast(n.b & 3)))) {
                    .plain => try u.printCall(w, n, name, "(", ")"),
                    .never_inline => {
                        // Sema rejects a `never_inline` call at compile time, and the checksums
                        // that Sema computes at compile time call the same functions.
                        try w.writeAll("(if (@inComptime()) ");
                        try u.printCall(w, n, name, "(", ")");
                        try w.print(" else @call(.never_inline, {s}, ", .{name});
                        try u.printCall(w, n, "", ".{", "}))");
                    },
                    .auto, .always_inline => |kind| {
                        try w.print("@call(.{t}, {s}, ", .{ kind, name });
                        try u.printCall(w, n, "", ".{", "})");
                    },
                }
            },
            .call_ptr => {
                if (u.callsDeleted(n)) return u.printLit(w, n.ty, 0);
                const set = u.fn_sets.items[n.a];
                if (n.b == 0) {
                    try w.writeAll("(if ((");
                    try u.kidText(w, n, 0);
                    try w.print(") & 1 == 0) &t{d}_h{d} else &t{d}_h{d})", .{ u.index, set[0], u.index, set[1] });
                } else {
                    const f = u.funcs.items[set[0]];
                    try w.writeAll("([_]*const fn (");
                    for (f.params, 0..) |p, pi| {
                        if (pi != 0) try w.writeAll(", ");
                        try u.printType(w, u.vars.items[p].ty);
                    }
                    try w.writeAll(") ");
                    try u.printType(w, f.ret);
                    try w.writeAll("{");
                    for (set) |fi| try w.print(" &t{d}_h{d},", .{ u.index, fi });
                    try w.writeAll(" })[");
                    try u.kidText(w, n, 0);
                    try w.print(" % {d}]", .{set.len});
                }
                try w.writeAll("(");
                for (n.kids[1..], 0..) |_, ki| {
                    if (ki != 0) try w.writeAll(", ");
                    try u.kidText(w, n, ki + 1);
                }
                try w.writeAll(")");
            },
            .addr_of => try w.print("&v{d}", .{n.a}),
            .via_space => {
                try w.print("viaSpace{d}(", .{n.b});
                try u.printType(w, n.ty);
                try w.writeAll(", ");
                try u.printNode(w, n.kids[0]);
                try w.writeAll(", ");
                try u.printNode(w, n.kids[1]);
                try w.writeAll(")");
            },
            .deref => try w.print("v{d}.*", .{n.a}),
            .pick_ref => if (u.vars.items[n.a].deleted or u.vars.items[n.b].deleted) {
                try u.printLit(w, n.ty, 0);
            } else {
                try w.writeAll("(if (");
                try u.kidText(w, n, 0);
                try w.print(") &v{d} else &v{d}).*", .{ n.a, n.b });
            },
            .div_const => try u.printDivConst(w, n),
            .reinterpret => switch (n.b) {
                0 => {
                    try w.writeAll("@as(");
                    try u.printType(w, n.ty);
                    try w.writeAll(", @bitCast(");
                    try u.kidText(w, n, 0);
                    try w.writeAll("))");
                },
                else => {
                    try w.writeAll(if (n.b == 1) "viewAs(" else "viewThroughPointer(");
                    try u.printType(w, n.ty);
                    try w.writeAll(", ");
                    try u.kidText(w, n, 0);
                    try w.writeAll(")");
                },
            },
            .patch => {
                try w.writeAll("patchWith(");
                try u.kidText(w, n, 0);
                try w.writeAll(", ");
                try u.kidText(w, n, 1);
                try w.print(", {d})", .{n.a});
            },
            .patch_at => {
                try w.writeAll("patchAt(");
                try u.kidText(w, n, 0);
                try w.writeAll(", ");
                try u.kidText(w, n, 1);
                try w.writeAll(", ");
                try u.kidText(w, n, 2);
                try w.writeAll(")");
            },
            .cond => {
                try w.writeAll("(if (");
                try u.kidText(w, n, 0);
                try w.writeAll(") ");
                try u.kidText(w, n, 1);
                try w.writeAll(" else ");
                try u.kidText(w, n, 2);
                try w.writeAll(")");
            },
            .@"orelse", .@"catch" => {
                try w.writeAll("(");
                try u.kidText(w, n, 0);
                try w.writeAll(if (n.op == .@"orelse") " orelse " else " catch ");
                try u.kidText(w, n, 1);
                try w.writeAll(")");
            },
            .opt_wrap, .eu_wrap => {
                try w.writeAll("@as(");
                try u.printType(w, n.ty);
                try w.writeAll(", ");
                try u.kidText(w, n, 0);
                try w.writeAll(")");
            },
            .opt_cond => {
                try w.writeAll("@as(");
                try u.printType(w, n.ty);
                try w.writeAll(", if (");
                try u.kidText(w, n, 0);
                try w.writeAll(") ");
                try u.kidText(w, n, 1);
                try w.writeAll(" else null)");
            },
            .eu_err => {
                try w.writeAll("@as(");
                try u.printType(w, n.ty);
                try w.print(", error.{c})", .{@as(u8, 'A') + @as(u8, @intCast(n.a))});
            },
            .eu_cond => {
                try w.writeAll("@as(");
                try u.printType(w, n.ty);
                try w.writeAll(", if (");
                try u.kidText(w, n, 0);
                try w.writeAll(") ");
                try u.kidText(w, n, 1);
                try w.print(" else error.{c})", .{@as(u8, 'A') + @as(u8, @intCast(n.a))});
            },
            .is_null => {
                try w.writeAll("(");
                try u.kidText(w, n, 0);
                try w.writeAll(" == null)");
            },
            .eu_ok => {
                try w.writeAll("(if (");
                try u.kidText(w, n, 0);
                try w.writeAll(") |_| true else |_| false)");
            },
            .switch_expr => {
                const prongs = u.switches.items[n.a];
                try w.writeAll("@as(");
                try u.printType(w, n.ty);
                try w.writeAll(", switch (");
                try u.kidText(w, n, 0);
                try w.writeAll(") {");
                for (prongs, 0..) |p, pi| {
                    try w.writeAll(" ");
                    try printItems(w, p);
                    try w.writeAll(" => ");
                    try u.kidText(w, n, pi + 1);
                    try w.writeAll(",");
                }
                try w.writeAll(" else => ");
                try u.kidText(w, n, prongs.len + 1);
                try w.writeAll(", })");
            },
        }
    }

    fn isWideSigned(u: *const Unit, t: TypeId) bool {
        return switch (u.ty(u.scalarOf(t))) {
            .int => |i| i.signed and i.bits > 64,
            else => false,
        };
    }

    /// Sema fails an assertion when it evaluates `|` on signed integers wider than 64 bits at
    /// compile time, so the test functions apply `|` to the unsigned bits of such integers.
    fn printWideSignedOr(u: *Unit, w: *Writer, n: Node) !void {
        const unsigned_t = try u.withElem(n.ty, try u.intType(false, u.ty(u.scalarOf(n.ty)).int.bits));
        try w.writeAll("@as(");
        try u.printType(w, n.ty);
        try w.writeAll(", @bitCast(");
        for (0..2) |ki| {
            if (ki != 0) try w.writeAll(" | ");
            try w.writeAll("@as(");
            try u.printType(w, unsigned_t);
            try w.writeAll(", @bitCast(");
            try u.kidText(w, n, ki);
            try w.writeAll("))");
        }
        try w.writeAll("))");
    }

    /// Prints the arguments of call `n` between `open` and `close`, after `name`. A call within a
    /// group passes the bound of the caller minus one, and a call from outside passes a bound
    /// below `boundLimit`.
    fn printCall(u: *Unit, w: *Writer, n: Node, name: []const u8, open: []const u8, close: []const u8) !void {
        const callee = u.funcs.items[n.a];
        try w.print("{s}{s}", .{ name, open });
        for (n.kids, callee.params, 0..) |_, p, ki| {
            if (ki != 0) try w.writeAll(", ");
            if (p != callee.bound) {
                try u.kidText(w, n, ki);
                continue;
            }
            try w.writeAll("(");
            try u.kidText(w, n, ki);
            if (n.b & 4 != 0) {
                try w.writeAll(" - 1)");
            } else {
                try w.print(" % {d})", .{u.boundLimit(callee.group)});
            }
        }
        try w.writeAll(close);
    }

    /// Returns divisor number `pick` for a division of type `i`. The divisors include the powers of
    /// two from one to the largest positive one, values near the maximum and negative values. A
    /// quotient never divides by minus one, which overflows for the minimum dividend.
    fn divisor(i: Int, pick: u8, quotient: bool) u256 {
        const mask = intMask(i.bits);
        const max_shift: u8 = @intCast(i.bits - 1 - @intFromBool(i.signed));
        const top = @as(u256, 1) << max_shift;
        const max = if (i.signed) top +% (top -% 1) else mask;
        const shift: u8 = @intCast((pick >> 4) % (@as(u32, max_shift) + 1));
        const negative_shift: u8 = @intCast((pick >> 4) % i.bits);
        const v: u256 = switch (pick % 12) {
            0 => 1,
            1 => 2,
            2 => top,
            3 => @max(top >> 1, 1),
            4, 5 => @as(u256, 1) << shift,
            6 => 3,
            7 => if (max_shift > 64) (@as(u256, 1) << 64) + 7 else 7,
            8 => max,
            9 => max -% 1,
            10 => if (i.signed) 0 -% (@as(u256, 1) << negative_shift) else (@as(u256, 1) << shift) +% 1,
            else => if (!i.signed) 10 else if (quotient) @as(u256, 1) << @intCast(i.bits - 1) else mask,
        } & mask;
        if (v == 0 or (i.signed and quotient and v == mask)) return 1;
        return v;
    }

    fn printDivConst(u: *Unit, w: *Writer, n: Node) !void {
        const op: DivOp = @fromBackingInt(@intCast(n.b));
        switch (op) {
            .slash, .percent => {
                try w.writeAll("(");
                try u.kidText(w, n, 0);
                try w.print(" {s} ", .{DivOp.text[n.b]});
                try u.printDivisor(w, n);
            },
            .div_exact => {
                try w.writeAll("@divExact(multipleOf(");
                try u.kidText(w, n, 0);
                try w.writeAll(", ");
                try u.printDivisor(w, n);
                try w.writeAll("), ");
                try u.printDivisor(w, n);
            },
            else => {
                try w.print("{s}(", .{DivOp.text[n.b]});
                try u.kidText(w, n, 0);
                try w.writeAll(", ");
                try u.printDivisor(w, n);
            },
        }
        try w.writeAll(")");
    }

    fn printDivisor(u: *Unit, w: *Writer, n: Node) !void {
        const op: DivOp = @fromBackingInt(@intCast(n.b));
        const quotient = op != .mod and op != .rem and op != .percent;
        const i = u.ty(u.scalarOf(n.ty)).int;
        try w.writeAll("@as(");
        try u.printType(w, n.ty);
        switch (u.ty(n.ty)) {
            .vector => |s| {
                try w.writeAll(", .{");
                for (0..s.len) |lane| {
                    try w.writeAll(" ");
                    try printIntValue(w, i, divisor(i, @truncate(n.a >> @intCast(8 * lane)), quotient));
                    try w.writeAll(",");
                }
                try w.writeAll(" })");
            },
            else => {
                try w.writeAll(", ");
                try printIntValue(w, i, divisor(i, @truncate(n.a), quotient));
                try w.writeAll(")");
            },
        }
    }

    fn printDiv(u: *Unit, w: *Writer, n: Node) !void {
        const op: DivOp = @fromBackingInt(@intCast(n.b));
        const st = u.ty(u.scalarOf(n.ty));
        const is_vec = u.ty(n.ty) == .vector;
        if (op == .slash or op == .percent) {
            try w.writeAll("(");
            try u.kidText(w, n, 0);
            try w.print(" {s} ", .{DivOp.text[n.b]});
        } else {
            try w.print("{s}(", .{if (st == .float and op == .mod) "floatMod" else DivOp.text[n.b]});
            try u.kidText(w, n, 0);
            try w.writeAll(", ");
        }
        if (st == .int and !is_vec) {
            const i = st.int;
            try w.writeAll("(if (");
            try u.kidText(w, n, 1);
            try w.writeAll(" == 0");
            if (i.signed) {
                try w.writeAll(" or (");
                try u.kidText(w, n, 1);
                try w.writeAll(" == -1 and ");
                try u.kidText(w, n, 0);
                try w.writeAll(" == ");
                try printIntValue(w, i, @as(u256, 1) << @intCast(i.bits - 1));
                try w.writeAll(")");
            }
            try w.writeAll(") ");
            try u.printLit(w, n.ty, 1);
            try w.writeAll(" else ");
            try u.kidText(w, n, 1);
            try w.writeAll(")");
        } else if (st == .int) {
            const i = st.int;
            const e = u.scalarOf(n.ty);
            try w.writeAll("@select(");
            try u.printType(w, e);
            try w.writeAll(", ");
            try u.kidText(w, n, 1);
            try w.writeAll(" == ");
            try u.printSplat(w, n.ty, 0);
            try w.writeAll(", ");
            try u.printSplat(w, n.ty, 1);
            try w.writeAll(", ");
            if (i.signed) {
                try w.writeAll("@select(");
                try u.printType(w, e);
                try w.writeAll(", ");
                try u.kidText(w, n, 1);
                try w.writeAll(" == ");
                try u.printSplat(w, n.ty, intMask(i.bits));
                try w.writeAll(", @select(");
                try u.printType(w, e);
                try w.writeAll(", ");
                try u.kidText(w, n, 0);
                try w.writeAll(" == ");
                try u.printSplat(w, n.ty, @as(u256, 1) << @intCast(i.bits - 1));
                try w.writeAll(", ");
                try u.printSplat(w, n.ty, 1);
                try w.writeAll(", ");
                try u.kidText(w, n, 1);
                try w.writeAll("), ");
                try u.kidText(w, n, 1);
                try w.writeAll(")");
            } else {
                try u.kidText(w, n, 1);
            }
            try w.writeAll(")");
        } else {
            try u.kidText(w, n, 1);
        }
        try w.writeAll(")");
    }

    /// Zig does not specify which zero `@min` and `@max` return for a positive and a negative
    /// zero. Adding a positive zero turns a negative zero result into a positive zero and keeps
    /// every other result.
    fn printPlusZero(u: *Unit, w: *Writer, t: TypeId) !void {
        try w.writeAll(" + @as(");
        try u.printType(w, t);
        try w.writeAll(if (u.ty(t) == .vector) ", @splat(0.0)))" else ", 0.0))");
    }

    fn printSplat(u: *Unit, w: *Writer, t: TypeId, value: u256) !void {
        try w.writeAll("@as(");
        try u.printType(w, t);
        try w.writeAll(", @splat(");
        try u.printLit(w, u.scalarOf(t), value & intMask(u.ty(u.scalarOf(t)).int.bits));
        try w.writeAll("))");
    }

    fn printConv(u: *Unit, w: *Writer, n: Node) !void {
        const kind: Conv = @fromBackingInt(@intCast(n.b));
        const from_t = u.nodes.items[n.kids[0]].ty;
        switch (kind) {
            .int_int => try u.printIntConv(w, n),
            .int_cast => {
                const to = u.ty(n.ty).int;
                const from = u.ty(from_t).int;
                try w.writeAll("@as(");
                try u.printType(w, n.ty);
                try w.writeAll(", @intCast(");
                try u.kidText(w, n, 0);
                try w.print(" >> {d}))", .{from.bits - to.bits});
            },
            .float_float, .vec_float_float => try u.printCast(w, n.ty, "@floatCast", n),
            .int_to_float, .vec_int_to_float => try u.printCast(w, n.ty, "@floatFromInt", n),
            .float_to_int => {
                const to = u.ty(n.ty).int;
                const lo: f64 = if (to.signed) -std.math.pow(f64, 2, @floatFromInt(to.bits - 1)) - 1 else -1;
                const hi: f64 = std.math.pow(f64, 2, @floatFromInt(if (to.signed) to.bits - 1 else to.bits));
                try w.writeAll("@as(");
                try u.printType(w, n.ty);
                try w.writeAll(", @intFromFloat(if (@as(f64, @floatCast(");
                try u.kidText(w, n, 0);
                try w.print(")) > {d:.1} and @as(f64, @floatCast(", .{lo});
                try u.kidText(w, n, 0);
                try w.print(")) < {d:.1}) @trunc(@as(f64, @floatCast(", .{hi});
                try u.kidText(w, n, 0);
                try w.writeAll("))) else @as(f64, 0)))");
            },
            .bits => if (u.ty(u.scalarOf(from_t)) == .float) {
                try w.writeAll("@as(");
                try u.printType(w, n.ty);
                try w.writeAll(", @bitCast(quietNan(");
                try u.kidText(w, n, 0);
                try w.writeAll(")))");
            } else {
                try u.printCast(w, n.ty, "@bitCast", n);
            },
            .bool_to_int => {
                try w.writeAll("@as(");
                try u.printType(w, n.ty);
                try w.writeAll(", @intFromBool(");
                try u.kidText(w, n, 0);
                try w.writeAll("))");
            },
            .vec_int => {
                const to = u.ty(u.scalarOf(n.ty)).int;
                const from = u.ty(u.scalarOf(from_t)).int;
                try u.printCast(w, n.ty, if (from.bits > to.bits) "@truncate" else "@intCast", n);
            },
        }
    }
};

pub const header =
    \\const Error = error{ A, B, C };
    \\
    \\fn mix(cs: u32, x: u64) u32 {
    \\    const a = (cs ^ @as(u32, @truncate(x))) *% 0x01000193;
    \\    return (a ^ @as(u32, @truncate(x >> 32))) *% 0x01000193;
    \\}
    \\
    \\fn floatBits16(x: f16) u64 {
    \\    return if (x != x) 0x7e00 else @as(u16, @bitCast(x));
    \\}
    \\
    \\fn floatBits32(x: f32) u64 {
    \\    return if (x != x) 0x7fc00000 else @as(u32, @bitCast(x));
    \\}
    \\
    \\fn floatBits64(x: f64) u64 {
    \\    return if (x != x) 0x7ff8000000000000 else @as(u64, @bitCast(x));
    \\}
    \\
    \\/// Zig does not specify the bits of a NaN result, so the test functions replace every NaN with
    \\/// one quiet NaN before they reinterpret the bits of a float.
    \\fn quietNan(x: anytype) @TypeOf(x) {
    \\    const T = @TypeOf(x);
    \\    const E = if (@typeInfo(T) == .vector) @typeInfo(T).vector.child else T;
    \\    const nan: E = switch (E) {
    \\        f16 => @bitCast(@as(u16, 0x7e00)),
    \\        f32 => @bitCast(@as(u32, 0x7fc00000)),
    \\        else => @bitCast(@as(u64, 0x7ff8000000000000)),
    \\    };
    \\    if (T == E) return if (x != x) nan else x;
    \\    var y = x;
    \\    inline for (0..@typeInfo(T).vector.len) |i| {
    \\        if (y[i] != y[i]) y[i] = nan;
    \\    }
    \\    return y;
    \\}
    \\
    \\/// The x86_64 and LLVM backends of Zig disagree on `@mod` of floats when the divisor is
    \\/// infinite or negative, when a negative remainder plus the divisor rounds to the divisor, and
    \\/// on the sign of a zero result. `floatMod` returns `@rem` in the first two cases and adds a
    \\/// positive zero to every other result.
    \\fn floatMod(x: anytype, y: @TypeOf(x)) @TypeOf(x) {
    \\    const T = @TypeOf(x);
    \\    const r = @rem(x, y);
    \\    if (@typeInfo(T) != .vector) {
    \\        const defined = y > 0 and y - y == 0 and (r >= 0 or r + y != y);
    \\        return if (defined) @mod(x, y) + 0.0 else r;
    \\    }
    \\    const zero: T = @splat(0);
    \\    const defined = (y > zero) & (y - y == zero) & ((r >= zero) | (r + y != y));
    \\    return @select(@typeInfo(T).vector.child, defined, @mod(x, y) + zero, r);
    \\}
    \\
    \\inline fn multipleOf(x: anytype, comptime d: @TypeOf(x)) @TypeOf(x) {
    \\    return x - @rem(x, d);
    \\}
    \\
    \\fn wideInput(a: u64, b: u64, c: u64, d: u64) u256 {
    \\    return @as(u256, a) | (@as(u256, b) << 64) | (@as(u256, c) << 128) | (@as(u256, d) << 192);
    \\}
    \\
    \\fn viewAs(comptime T: type, x: anytype) T {
    \\    const u: extern union { from: @TypeOf(x), to: T } = .{ .from = x };
    \\    return u.to;
    \\}
    \\
    \\fn viewThroughPointer(comptime T: type, x: anytype) T {
    \\    const v: @TypeOf(x) align(@alignOf(T)) = x;
    \\    const p: *const T = @ptrCast(&v);
    \\    return p.*;
    \\}
    \\
    \\fn patchWith(x: anytype, element: anytype, comptime at: usize) @TypeOf(x) {
    \\    const Part = @TypeOf(element);
    \\    var u: extern union { whole: @TypeOf(x), parts: [@sizeOf(@TypeOf(x)) / @sizeOf(Part)]Part } = .{ .whole = x };
    \\    u.parts[at] = element;
    \\    return u.whole;
    \\}
    \\
    \\fn patchAt(x: anytype, element: anytype, at: u32) @TypeOf(x) {
    \\    const Part = @TypeOf(element);
    \\    var u: extern union { whole: @TypeOf(x), parts: [@sizeOf(@TypeOf(x)) / @sizeOf(Part)]Part } = .{ .whole = x };
    \\    const i = at % u.parts.len;
    \\    u.parts[i] = element ^ u.parts[(i + 1) % u.parts.len];
    \\    return u.whole;
    \\}
    \\
    \\var scratch: [32 * 128]u8 align(16) addrspace(.shared) = undefined;
    \\
    \\inline fn sharedSlot(comptime T: type) *addrspace(.shared) T {
    \\    return @ptrCast(@alignCast(&scratch[(@workItemId(0) & 31) * 128]));
    \\}
    \\
    \\noinline fn readThrough(comptime T: type, p: *const T) T {
    \\    return p.*;
    \\}
    \\
    \\fn readThroughInlined(comptime T: type, p: *const T) T {
    \\    return p.*;
    \\}
    \\
    \\fn viaSpace0(comptime T: type, x: T, flag: bool) T {
    \\    _ = flag;
    \\    if (@inComptime()) return x;
    \\    const p: [*]T = @addrSpaceCast(@as([*]addrspace(.shared) T, @ptrCast(sharedSlot(T))));
    \\    const i = @workItemId(0) >> 5;
    \\    p[i] = x;
    \\    return p[i];
    \\}
    \\
    \\fn viaSpace1(comptime T: type, x: T, flag: bool) T {
    \\    if (@inComptime()) return x;
    \\    var local: T = x;
    \\    const shared: *T = @addrSpaceCast(sharedSlot(T));
    \\    const p = if (flag) shared else &local;
    \\    p.* = x;
    \\    return p.*;
    \\}
    \\
    \\fn viaSpace2(comptime T: type, x: T, flag: bool) T {
    \\    _ = flag;
    \\    if (@inComptime()) return x;
    \\    const p: *T = @addrSpaceCast(sharedSlot(T));
    \\    const q: *T = @addrSpaceCast(sharedSlot(T));
    \\    p.* = x;
    \\    return if (p == q and @intFromPtr(p) == @intFromPtr(q)) q.* else other(T, x);
    \\}
    \\
    \\fn viaSpace3(comptime T: type, x: T, flag: bool) T {
    \\    if (@inComptime()) return x;
    \\    const p: *T = @addrSpaceCast(sharedSlot(T));
    \\    p.* = x;
    \\    return if (flag) readThrough(T, p) else readThroughInlined(T, p);
    \\}
    \\
    \\fn viaSpace4(comptime T: type, x: T, flag: bool) T {
    \\    if (@inComptime()) return x;
    \\    const wide = comptime @typeInfo(T) == .int and (@typeInfo(T).int.bits == 32 or @typeInfo(T).int.bits == 64);
    \\    if (!wide) return viaSpace0(T, x, flag);
    \\    const p: *T = @addrSpaceCast(sharedSlot(T));
    \\    p.* = x;
    \\    return @atomicRmw(T, p, .Add, 0, .monotonic);
    \\}
    \\
    \\noinline fn keepAddress(comptime T: type, p: *T) void {
    \\    const v = p.*;
    \\    p.* = v;
    \\}
    \\
    \\fn exchangeThrough(comptime T: type, p: *T, x: T) T {
    \\    return @atomicRmw(T, p, .Xchg, x, .monotonic);
    \\}
    \\
    \\fn loadThrough(comptime T: type, p: *const T) T {
    \\    return @atomicLoad(T, p, .acquire);
    \\}
    \\
    \\fn viaSpace5(comptime T: type, x: T, flag: bool) T {
    \\    if (@inComptime()) return x;
    \\    const wide = comptime @typeInfo(T) == .int and (@typeInfo(T).int.bits == 32 or @typeInfo(T).int.bits == 64);
    \\    if (!wide or @import("builtin").mode == .debug) return viaSpace1(T, x, flag);
    \\    var local: T = ~x;
    \\    keepAddress(T, &local);
    \\    const old = exchangeThrough(T, &local, x);
    \\    return if (flag) loadThrough(T, &local) else ~old;
    \\}
    \\
    \\/// Returns a value that differs from `x`, so that a failed check changes the checksum.
    \\fn other(comptime T: type, x: T) T {
    \\    return switch (@typeInfo(T)) {
    \\        .bool => !x,
    \\        .float => -x,
    \\        else => ~x,
    \\    };
    \\}
    \\
    \\fn viaSpace6(comptime T: type, x: T, flag: bool) T {
    \\    if (@inComptime() or @sizeOf(T) == 0) return x;
    \\    const n = 128 / @sizeOf(T);
    \\    const items: *[n]T = @addrSpaceCast(@as(*addrspace(.shared) [n]T, @ptrCast(sharedSlot(T))));
    \\    const len: usize = if (flag) 3 else 5;
    \\    @memset(items[0..len], x);
    \\    @memmove(items[1 .. len + 1], items[0..len]);
    \\    var local: [6]T = undefined;
    \\    @memcpy(local[0..len], items[1 .. len + 1]);
    \\    const words: *[8]u32 = @addrSpaceCast(sharedSlot([8]u32));
    \\    for (words, 0..) |*w, i| w.* = @intCast(i);
    \\    @memmove(words[1 .. len + 2], words[0 .. len + 1]);
    \\    const backward = words[1] == 0 and words[len + 1] == len;
    \\    @memmove(words[0..len], words[2 .. len + 2]);
    \\    const forward = words[0] == 1 and words[len - 1] == len;
    \\    return if (backward and forward) local[len - 1] else other(T, local[len - 1]);
    \\}
    \\
    \\fn errorCode(e: Error) u64 {
    \\    return switch (e) {
    \\        error.A => 1,
    \\        error.B => 2,
    \\        error.C => 3,
    \\    };
    \\}
    \\
;

/// Returns `rows * input_words` words from a fixed seed. Five in six words start as zero, a value
/// below 8, a value near the maximum, a power of two or a power of two minus one. The function then
/// zero-extends or sign-extends each word from a random width.
pub fn inputWords(gpa: Allocator, rows: u32) ![]u64 {
    var prng: std.Random.DefaultPrng = .init(0x5eed);
    const r = prng.random();
    const words = try gpa.alloc(u64, rows * input_words);
    for (words) |*w| {
        const bits = 1 + r.uintLessThan(u32, 64);
        const v: u64 = switch (r.uintLessThan(u32, 6)) {
            0 => 0,
            1 => r.uintLessThan(u64, 8),
            2 => std.math.maxInt(u64) - r.uintLessThan(u64, 4),
            3 => @as(u64, 1) << @intCast(bits - 1),
            4 => (@as(u64, 1) << @intCast(bits - 1)) -% 1,
            else => r.int(u64),
        };
        const shift: u6 = @intCast(64 - bits);
        w.* = if (r.boolean()) v else @bitCast(@as(i64, @bitCast(v << shift)) >> shift);
    }
    return words;
}
