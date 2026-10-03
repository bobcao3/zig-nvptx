//! Promotion moves local variables, the entries of `Mir.locals`, to registers. A root of a local
//! is a `mov` or `cvta` that names the local. The address tree of the local holds the registers
//! that follow from a root through moves, conversions and additions. The loads and stores of a
//! local cover byte ranges, called partitions. When each access has a known offset and each
//! partition has one size, each load takes the value of the latest store. Accesses of different
//! sizes that overlap form a window of 4 or 8 bytes in one register.
//!
//! A local of at most 8 bytes whose loads and stores add a register to its address can form one
//! window. Input: no `cvt`, `copy`, branch argument or stored value takes an address of the
//! local. Rule: each root becomes `mov d, 0` and each other `cvta` of the address tree becomes a
//! `mov`, so that each tree register holds an offset in the local. An access at offset o reads or
//! writes the bytes at bit 8o of the window register. Output: no instruction names the local.
//! Proof: by induction over the tree, each register holds its former value minus the address of
//! the local in the state space of the register, because `add`, `sub` and `mad.lo` take a tree
//! register as the addend and `cvta` changes only the state space (PTX ISA 9.4, section
//! 9.7.10.23). Zig makes an access past the bounds of an object through a pointer Illegal Behavior
//! (language reference, sections Pointers and Index out of Bounds), so in a defined execution each
//! access covers bytes of the local, and the window holds these bytes as memory did.
//!
//! A `copy` into a promoted local becomes loads, a `copy` out of one becomes stores, and a `copy`
//! between two promoted locals disappears. A local that stays in memory keeps its accesses, and
//! address space inference gives them `.local` later. The pass follows `PromoteMemToReg` and SROA
//! of LLVM, which do not change address spaces either.

const std = @import("std");
const Allocator = std.mem.Allocator;

const Mir = @import("../Mir.zig");
const Cfg = @import("../Cfg.zig");
const Layout = @import("../codegen/Layout.zig");

const none = Cfg.none;
const undef = std.math.maxInt(u32);
const unknown_offset = std.math.minInt(i64);

/// `param_copies` names the local variables that hold copies of kernel parameters.
pub fn run(gpa: Allocator, mir: *Mir, param_copies: []const Mir.ParamCopy) Allocator.Error!void {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const owned_copies = try gpa.dupe(Mir.ParamCopy, param_copies);
    defer gpa.free(owned_copies);
    var copies = owned_copies;
    // When an address of a local goes only into promoted locals, the next round can promote it.
    for (0..mir.locals.len + 1) |_| {
        _ = arena_state.reset(.retain_capacity);
        var p: Promotion = try .init(arena_state.allocator(), mir, copies);
        try p.analyze();
        if (!p.changes()) return;
        if (!try p.rewrite(gpa, mir)) return;
        if (!p.again) return;
        var kept: usize = 0;
        for (copies) |copy| {
            const local = p.local_map[copy.local];
            if (local == none) continue;
            copies[kept] = .{ .local = local, .param = copy.param };
            kept += 1;
        }
        copies = copies[0..kept];
    }
}

/// `promote` moves a local to registers. `contained` and `memory` leave it in memory. Only accesses
/// and address arithmetic use the address of a `contained` local. A local in `param_addressing` or
/// `param_generic` holds a copy of a kernel parameter. Zig parameters are immutable, so the address
/// of the parameter replaces the copy. The transform removes each write of the local and assumes
/// that only the copy writes it. `.param` loads read the parameter at a register address, and
/// `cvta.param` gives the generic address, which needs PTX ISA 7.7 (PTX ISA 9.4, sections 5.1.6.1
/// and 9.7.10.23).
const State = enum { promote, contained, param_addressing, memory, param_generic };

const Local = struct {
    state: State = .promote,
    /// The index of the kernel parameter that the local copies, or `none`.
    param: u32 = none,
    /// `none` before the first root, the result of the only root if that root is a 64-bit
    /// `cvta`, or else `undef`.
    root: u32 = none,
    /// The address tree has a `cvta` other than the root.
    casts: bool = false,
    /// The address tree has a `cvt`.
    converts: bool = false,
    /// A load or store adds a register to the address of the local.
    indexed: bool = false,
    /// The size of the one window that holds a promoted `indexed` local.
    window_size: u8 = 0,
    /// The address escapes only as the value of stores into the locals in `holders`.
    stored_only: bool = true,
    holders: std.ArrayList(u32) = .empty,
};

const Address = struct {
    local: u32 = none,
    /// The offset from the start of the local, or null when a register changes it.
    offset: ?i64 = 0,
};

/// The order of the tags is the order of the accesses of one instruction.
const AccessKind = enum { load, store, copy_src, copy_dest };

const Access = struct {
    inst: u32,
    local: u32,
    /// The offset from the start of the local, or `unknown_offset`. For an `indexed` access, the
    /// offset that the instruction adds to its address register.
    offset: i64,
    /// The address register holds the start of the local plus an offset that a register changes.
    indexed: bool = false,
    kind: AccessKind,
    elem_size: u8,
    /// The elements of a load or a store, or the units of a copy.
    elems: u32,
    /// The predecessor at whose end a load through an `AddressParam` reads, or `none`.
    pred: u32 = none,

    fn bytes(access: Access) i64 {
        return @as(i64, access.elem_size) * access.elems;
    }

    fn writes(access: Access) bool {
        return access.kind == .store or access.kind == .copy_dest;
    }
};

const Partition = struct { local: u32, offset: i64, size: u8, class: Mir.Class };

/// A window is a naturally aligned container of 4 or 8 bytes that holds loads and stores of
/// different sizes. The partition of a window is one register that holds the little-endian bytes.
const Window = struct { offset: i64, size: u8 };

const Range = struct { offset: i64, size: u8, local: u32 };

/// A block parameter whose arguments are addresses of local variables. Each element of each load
/// through the parameter becomes a new parameter of the block, and each predecessor passes the
/// value that the load reads at its end. This follows `speculatePHINodeLoads` of SROA in LLVM.
const AddressParam = struct {
    block: u32,
    index: u32,
    reg: u32,
    incoming: std.ArrayList(struct { pred: u32, address: Address }) = .empty,
    loads: std.ArrayList(u32) = .empty,
    speculates: bool = true,
};

const Placement = struct { block: u32, partition: u32, param: u32 };

const LoadParam = struct { load: u32, elem: u32, reg: u32 };

const Trampoline = struct { label: u32, target: u32, args: []const u32 };

const Promotion = struct {
    arena: Allocator,
    mir: *const Mir,
    cfg: Cfg,
    inst_block: []u32,
    user_starts: []u32,
    users: []u32,
    addresses: []Address,
    /// The local whose address tree holds the instruction, or `none`.
    tree_local: []u32,
    locals: []Local,
    accesses: std.ArrayList(Access) = .empty,
    /// The partition of each element of each load and store in `accesses`, and the partitions that
    /// a copy covers, in the order of their offsets.
    element_partitions: [][]u32 = &.{},
    /// For a copy between two promoted locals, the source partition of each destination partition.
    copy_sources: [][]u32 = &.{},
    partitions: std.ArrayList(Partition) = .empty,
    partition_at: std.AutoHashMapUnmanaged(struct { u32, i64 }, u32) = .empty,
    /// The window of each element offset of a local that has loads and stores of mixed sizes.
    windows: std.AutoHashMapUnmanaged(struct { u32, i64 }, Window) = .empty,
    address_params: std.ArrayList(AddressParam) = .empty,
    address_param_at: std.AutoHashMapUnmanaged(struct { u32, u32 }, u32) = .empty,
    copy_dest_of: []u32 = &.{},
    copy_src_of: []u32 = &.{},
    again: bool = false,
    /// The new index of each local variable after `rewrite`, or `none` for a removed one.
    local_map: []u32 = &.{},

    fn init(arena: Allocator, mir: *const Mir, param_copies: []const Mir.ParamCopy) Allocator.Error!Promotion {
        const cfg: Cfg = try .init(arena, mir);
        const count = mir.instructions.len;
        const inst_block = try arena.alloc(u32, count);
        for (0..cfg.blockCount()) |b| @memset(inst_block[cfg.starts[b]..cfg.starts[b + 1]], @intCast(b));
        const locals = try arena.alloc(Local, mir.locals.len);
        @memset(locals, .{});
        for (param_copies) |copy| locals[copy.local].param = copy.param;
        const tree_local = try arena.alloc(u32, count);
        @memset(tree_local, none);
        const addresses = try arena.alloc(Address, cfg.reg_count);
        @memset(addresses, .{ .local = none });
        var p: Promotion = .{
            .arena = arena,
            .mir = mir,
            .cfg = cfg,
            .inst_block = inst_block,
            .user_starts = undefined,
            .users = undefined,
            .addresses = addresses,
            .tree_local = tree_local,
            .locals = locals,
        };
        try p.collectUsers();
        return p;
    }

    fn collectUsers(p: *Promotion) Allocator.Error!void {
        const Counter = struct {
            p: *Promotion,
            counts: []u32,
            inst: u32,
            fill: bool,
            last: []u32,

            pub fn reg(c: *@This(), at: u32, role: Mir.RegRole) void {
                if (role != .use) return;
                const id = c.p.cfg.regId(@bitCast(c.p.mir.extra[at]));
                if (c.last[id] == c.inst) return;
                c.last[id] = c.inst;
                if (c.fill) {
                    c.p.users[c.counts[id]] = c.inst;
                    c.counts[id] += 1;
                } else c.counts[id] += 1;
            }
        };
        const mir = p.mir;
        const counts = try p.arena.alloc(u32, p.cfg.reg_count + 1);
        @memset(counts, 0);
        const last = try p.arena.alloc(u32, p.cfg.reg_count);
        @memset(last, none);
        var counter: Counter = .{ .p = p, .counts = counts, .inst = 0, .fill = false, .last = last };
        for (0..mir.instructions.len) |i| {
            counter.inst = @intCast(i);
            mir.visitRegs(mir.instructions.get(i), &counter);
        }
        var total: u32 = 0;
        for (counts) |*c| {
            const n = c.*;
            c.* = total;
            total += n;
        }
        p.user_starts = try p.arena.dupe(u32, counts);
        p.users = try p.arena.alloc(u32, total);
        @memset(last, none);
        counter.fill = true;
        for (0..mir.instructions.len) |i| {
            counter.inst = @intCast(i);
            mir.visitRegs(mir.instructions.get(i), &counter);
        }
    }

    fn inst(p: *const Promotion, i: u32) Mir.Inst {
        return p.mir.instructions.get(i);
    }

    fn regId(p: *const Promotion, word: u32) u32 {
        return p.cfg.regId(@bitCast(word));
    }

    /// Returns the positions in `extra` of the operands of `instruction` after its guard.
    fn operandsAt(p: *const Promotion, instruction: Mir.Inst) Allocator.Error![]const u32 {
        const range = p.mir.operands(instruction) orelse return &.{};
        var list: std.ArrayList(u32) = .empty;
        var at = range.first;
        for (0..range.count) |_| {
            try list.append(p.arena, at);
            at += @as(Mir.Operand, @bitCast(p.mir.extra[at])).len();
        }
        const guard: usize = @intFromBool(instruction.modifiers.guard() != .none);
        return list.items[guard..];
    }

    fn operand(p: *const Promotion, at: u32) Mir.Operand {
        return @bitCast(p.mir.extra[at]);
    }

    fn immAt(p: *const Promotion, at: u32) ?i64 {
        const op = p.operand(at);
        return switch (op.kind) {
            .imm => op.payload,
            .imm64 => @bitCast(@as(u64, p.mir.extra[at + 1]) | (@as(u64, p.mir.extra[at + 2]) << 32)),
            else => null,
        };
    }

    fn isReg(p: *const Promotion, at: u32, id: u32) bool {
        const op = p.operand(at);
        return op.kind == .reg and p.regId(p.mir.extra[at]) == id;
    }

    fn escape(p: *Promotion, local: u32) void {
        p.locals[local].state = .memory;
        p.locals[local].stored_only = false;
    }

    fn analyze(p: *Promotion) Allocator.Error!void {
        const mir = p.mir;
        var work: std.ArrayList(u32) = .empty;
        for (0..mir.instructions.len) |index| {
            const i: u32 = @intCast(index);
            const instruction = p.inst(i);
            const ops = try p.operandsAt(instruction);
            for (ops, 0..) |at, k| {
                const op = p.operand(at);
                if (op.kind != .local) continue;
                const local = op.payload;
                const is_root = k == 1 and ops.len == 2 and instruction.modifiers.guard() == .none and switch (instruction.tag) {
                    .mov => true,
                    .cvta => instruction.modifiers.cvta.space == .local and !instruction.modifiers.cvta.to,
                    else => false,
                };
                if (!is_root) {
                    p.escape(local);
                    continue;
                }
                const id = p.regId(mir.extra[ops[0]]);
                p.addresses[id] = .{ .local = local, .offset = 0 };
                p.tree_local[i] = local;
                const first = p.locals[local].root == none;
                p.locals[local].root = if (first and instruction.tag == .cvta and p.operand(ops[0]).regClass() == .b64) mir.extra[ops[0]] else undef;
                try work.append(p.arena, id);
            }
        }
        while (work.pop()) |id| {
            const address = p.addresses[id];
            for (p.users[p.user_starts[id]..p.user_starts[id + 1]]) |u| try p.visitUser(u, id, address, &work);
        }
        try p.findParamLoads();
        p.element_partitions = try p.arena.alloc([]u32, p.accesses.items.len);
        @memset(p.element_partitions, &.{});
        p.copy_sources = try p.arena.alloc([]u32, p.accesses.items.len);
        @memset(p.copy_sources, &.{});
        std.mem.sort(Access, p.accesses.items, {}, struct {
            fn lessThan(_: void, a: Access, b: Access) bool {
                if (a.inst != b.inst) return a.inst < b.inst;
                if (a.kind != b.kind) return @backingInt(a.kind) < @backingInt(b.kind);
                return a.pred < b.pred;
            }
        }.lessThan);
        p.copy_dest_of = try p.arena.alloc(u32, mir.instructions.len);
        p.copy_src_of = try p.arena.alloc(u32, mir.instructions.len);
        @memset(p.copy_dest_of, none);
        @memset(p.copy_src_of, none);
        for (p.accesses.items, 0..) |access, k| switch (access.kind) {
            .copy_dest => p.copy_dest_of[access.inst] = @intCast(k),
            .copy_src => p.copy_src_of[access.inst] = @intCast(k),
            else => {},
        };
        try p.classify();
    }

    fn visitUser(p: *Promotion, u: u32, id: u32, address: Address, work: *std.ArrayList(u32)) Allocator.Error!void {
        const local = address.local;
        if (p.locals[local].state == .memory and !p.locals[local].stored_only) return;
        const instruction = p.inst(u);
        const ops = try p.operandsAt(instruction);
        if (instruction.modifiers.guard() != .none) return p.escape(local);
        switch (instruction.tag) {
            .mov, .cvta, .cvt => {
                if (ops.len != 2 or !p.isReg(ops[1], id)) return p.escape(local);
                const ok = switch (instruction.tag) {
                    .mov => p.operand(ops[0]).regClass() == p.operand(ops[1]).regClass(),
                    .cvta => instruction.modifiers.cvta.space == .local,
                    .cvt => instruction.modifiers.cvt.rounding == .none and !instruction.modifiers.cvt.sat and
                        isAddressWidth(instruction.ty) and isAddressWidth(instruction.modifiers.cvt.from),
                    else => unreachable,
                };
                if (!ok) return p.escape(local);
                if (instruction.tag == .cvta) p.locals[local].casts = true;
                if (instruction.tag == .cvt) p.locals[local].converts = true;
                return p.derive(u, ops[0], .{ .local = local, .offset = address.offset }, work);
            },
            .add, .sub => {
                if (ops.len != 3 or instruction.modifiers.arith.cc or instruction.modifiers.arith.sat) return p.escape(local);
                if (!isAddressWidth(instruction.ty)) return p.escape(local);
                if (!p.isReg(ops[1], id) or p.isReg(ops[2], id)) return p.escape(local);
                const offset: ?i64 = if (p.immAt(ops[2])) |imm|
                    (if (address.offset) |base| (if (instruction.tag == .add) base +% imm else base -% imm) else null)
                else
                    null;
                return p.derive(u, ops[0], .{ .local = local, .offset = offset }, work);
            },
            .mad => {
                if (ops.len != 4 or instruction.modifiers.arith.mode != .lo or instruction.modifiers.arith.cc) return p.escape(local);
                if (p.isReg(ops[1], id) or p.isReg(ops[2], id) or !p.isReg(ops[3], id)) return p.escape(local);
                return p.derive(u, ops[0], .{ .local = local, .offset = null }, work);
            },
            .ld, .st => {
                const form = instruction.modifiers.mem;
                const mem = p.mir.memOperand(instruction, 0);
                if (mem.sem != .none or (mem.space != .generic and mem.space != .local)) return p.escape(local);
                const base_at = if (instruction.tag == .ld) ops[1] else ops[0];
                const value_at = ops[ops.len - 1];
                if (instruction.tag == .st and p.operandHas(value_at, id)) {
                    try p.storedInto(local, base_at);
                    if (!p.isReg(base_at, id)) return;
                }
                if (!p.isReg(base_at, id)) return p.escape(local);
                const elem_size = memorySize(instruction.ty) orelse return p.escape(local);
                const offset_at = if (instruction.tag == .ld) ops[2] else ops[1];
                const imm: i64 = if (form.offset) p.immAt(offset_at) orelse return p.escape(local) else 0;
                const elems: u32 = switch (form.vec) {
                    .scalar => 1,
                    .v2 => 2,
                    .v4 => 4,
                };
                try p.accesses.append(p.arena, .{
                    .inst = u,
                    .local = local,
                    .offset = if (address.offset) |base| base +% imm else imm,
                    .indexed = address.offset == null,
                    .kind = if (instruction.tag == .st) .store else .load,
                    .elem_size = elem_size,
                    .elems = elems,
                });
            },
            .bra => {
                const target = p.cfg.label_blocks[p.mir.branchTarget(instruction)];
                for (p.mir.branchArgs(instruction), 0..) |arg, index| {
                    if (p.regId(arg) == id) try p.addIncoming(target, @intCast(index), p.inst_block[u], address);
                }
            },
            .copy => {
                const is_set = instruction.modifiers.copy.op == .set;
                const is_dest = p.isReg(ops[0], id);
                if (is_dest == p.isReg(ops[1], id) or (is_set and !is_dest)) return p.escape(local);
                // A volatile side keeps the whole copy and each of its accesses.
                if (p.mir.hasOrderedMemoryRef(instruction)) return p.escape(local);
                const mem = p.mir.memOperand(instruction, @intFromBool(!is_dest));
                if (mem.space != .generic and mem.space != .local) return p.escape(local);
                const count = p.immAt(ops[2]) orelse return p.escape(local);
                try p.accesses.append(p.arena, .{
                    .inst = u,
                    .local = local,
                    .offset = address.offset orelse unknown_offset,
                    .kind = if (is_dest) .copy_dest else .copy_src,
                    .elem_size = @intCast(mem.size()),
                    .elems = @intCast(count),
                });
            },
            else => p.escape(local),
        }
    }

    fn addIncoming(p: *Promotion, block: u32, index: u32, pred: u32, address: Address) Allocator.Error!void {
        const entry = try p.address_param_at.getOrPut(p.arena, .{ block, index });
        if (!entry.found_existing) {
            entry.value_ptr.* = @intCast(p.address_params.items.len);
            const params = p.mir.blockParams(p.inst(p.cfg.starts[block]));
            try p.address_params.append(p.arena, .{ .block = block, .index = index, .reg = p.regId(params[index]) });
        }
        try p.address_params.items[entry.value_ptr.*].incoming.append(p.arena, .{ .pred = pred, .address = address });
    }

    /// Keeps the locals of an `AddressParam` in memory unless its users are loads and no write can
    /// come between the start of its block and a load, without a new start of the block.
    fn findParamLoads(p: *Promotion) Allocator.Error!void {
        const cfg = &p.cfg;
        // A marked block runs whole between the start of the parameter block and the load.
        const marks = try p.arena.alloc(u32, cfg.blockCount());
        @memset(marks, none);
        var mark: u32 = 0;
        var work: std.ArrayList(u32) = .empty;
        var writes: std.ArrayList(u32) = .empty;
        for (p.address_params.items) |*param| {
            param.speculates = param.incoming.items.len == cfg.predecessors[param.block].len;
            for (param.incoming.items) |in| {
                if (in.address.offset == null) param.speculates = false;
            }
            writes.clearRetainingCapacity();
            for (p.accesses.items) |access| {
                if (!access.writes()) continue;
                for (param.incoming.items) |in| {
                    if (in.address.local == access.local) {
                        try writes.append(p.arena, access.inst);
                        break;
                    }
                }
            }
            var marked_block: u32 = none;
            for (p.users[p.user_starts[param.reg]..p.user_starts[param.reg + 1]]) |u| {
                if (!param.speculates) break;
                if (!p.isParamLoad(u, param.*)) {
                    param.speculates = false;
                    break;
                }
                const block = p.inst_block[u];
                if (block != marked_block and block != param.block) {
                    marked_block = block;
                    mark += 1;
                    work.clearRetainingCapacity();
                    try work.append(p.arena, block);
                    while (work.pop()) |b| {
                        if (b == param.block and b != block) continue;
                        for (cfg.predecessors[b]) |pred| {
                            if (marks[pred] == mark) continue;
                            marks[pred] = mark;
                            try work.append(p.arena, pred);
                        }
                    }
                }
                for (writes.items) |w| {
                    const runs = if (block == param.block) false else marks[p.inst_block[w]] == mark;
                    if (runs or (p.inst_block[w] == block and w < u)) param.speculates = false;
                }
                if (param.speculates) try param.loads.append(p.arena, u);
            }
            if (!param.speculates) {
                for (param.incoming.items) |in| p.escape(in.address.local);
                continue;
            }
            for (param.loads.items) |u| {
                const instruction = p.inst(u);
                const ops = try p.operandsAt(instruction);
                const imm: i64 = if (instruction.modifiers.mem.offset) p.immAt(ops[2]).? else 0;
                for (param.incoming.items) |in| try p.accesses.append(p.arena, .{
                    .inst = u,
                    .local = in.address.local,
                    .offset = in.address.offset.? +% imm,
                    .kind = .load,
                    .elem_size = memorySize(instruction.ty).?,
                    .elems = switch (instruction.modifiers.mem.vec) {
                        .scalar => 1,
                        .v2 => 2,
                        .v4 => 4,
                    },
                    .pred = in.pred,
                });
            }
        }
    }

    fn isParamLoad(p: *const Promotion, u: u32, param: AddressParam) bool {
        const instruction = p.inst(u);
        if (instruction.tag != .ld or instruction.modifiers.guard() != .none) return false;
        const mem = p.mir.memOperand(instruction, 0);
        if (mem.sem != .none or (mem.space != .generic and mem.space != .local)) return false;
        const ops = p.operandsAt(instruction) catch return false;
        if (!p.isReg(ops[1], param.reg) or p.operandHas(ops[0], param.reg) or memorySize(instruction.ty) == null) return false;
        return !instruction.modifiers.mem.offset or p.immAt(ops[2]) != null;
    }

    fn derive(p: *Promotion, u: u32, dst_at: u32, address: Address, work: *std.ArrayList(u32)) Allocator.Error!void {
        const op = p.operand(dst_at);
        if (op.kind != .reg) return p.escape(address.local);
        const dst = p.regId(p.mir.extra[dst_at]);
        if (p.addresses[dst].local != none) return p.escape(address.local);
        p.addresses[dst] = address;
        p.tree_local[u] = address.local;
        try work.append(p.arena, dst);
    }

    fn operandHas(p: *const Promotion, at: u32, id: u32) bool {
        const op = p.operand(at);
        if (op.kind == .reg) return p.regId(p.mir.extra[at]) == id;
        if (op.kind != .list) return false;
        for (p.mir.extra[at + 1 ..][0..op.payload]) |word| if (p.regId(word) == id) return true;
        return false;
    }

    /// Marks `local` as escaped, and records the local that the store writes as a holder of it.
    fn storedInto(p: *Promotion, local: u32, base_at: u32) Allocator.Error!void {
        const state = &p.locals[local];
        const was_stored_only = state.stored_only;
        p.escape(local);
        const base = p.operand(base_at);
        if (!was_stored_only or base.kind != .reg) return;
        const holder = p.addresses[p.regId(p.mir.extra[base_at])].local;
        if (holder == none) return;
        state.stored_only = true;
        try state.holders.append(p.arena, holder);
    }

    /// Decides the state of each local and numbers the partitions of the promoted locals.
    fn classify(p: *Promotion) Allocator.Error!void {
        for (p.accesses.items) |access| {
            if (access.indexed) p.locals[access.local].indexed = true;
        }
        for (p.locals, 0..) |*state, local| {
            if (!state.indexed or state.state != .promote) continue;
            state.window_size = p.indexedWindowSize(@intCast(local));
            if (state.window_size == 0) state.state = .contained;
        }
        var ranges: std.ArrayList(Range) = .empty;
        for (p.accesses.items) |access| {
            const state = &p.locals[access.local];
            if (state.state != .promote or state.indexed) continue;
            const size: i64 = @intCast(p.mir.locals[access.local].size);
            const is_copy = access.kind == .copy_src or access.kind == .copy_dest;
            if (access.offset == unknown_offset or access.offset < 0 or access.offset + access.bytes() > size or
                (!is_copy and !p.supported(access)))
            {
                state.state = .contained;

                continue;
            }
            if (is_copy) continue;
            for (0..access.elems) |e| try ranges.append(p.arena, .{
                .offset = access.offset + @as(i64, access.elem_size) * @as(i64, @intCast(e)),
                .size = access.elem_size,
                .local = access.local,
            });
        }
        std.mem.sort(Range, ranges.items, {}, struct {
            fn lessThan(_: void, a: Range, b: Range) bool {
                if (a.local != b.local) return a.local < b.local;
                if (a.offset != b.offset) return a.offset < b.offset;
                return a.size < b.size;
            }
        }.lessThan);
        var begin: usize = 0;
        while (begin < ranges.items.len) {
            var end = begin + 1;
            while (end < ranges.items.len and ranges.items[end].local == ranges.items[begin].local) end += 1;
            try p.findWindows(ranges.items[begin..end]);
            begin = end;
        }
        for (p.accesses.items) |access| {
            if (access.pred == none or p.locals[access.local].state != .promote) continue;
            for (0..access.elems) |e| {
                const offset = access.offset + @as(i64, access.elem_size) * @as(i64, @intCast(e));
                const window = p.windows.get(.{ access.local, offset }) orelse continue;
                if (window.size != access.elem_size) p.locals[access.local].state = .contained;
            }
        }
        for (p.accesses.items, p.element_partitions) |access, *element_partitions| {
            if (p.locals[access.local].state != .promote) continue;
            if (access.kind != .load and access.kind != .store) continue;
            element_partitions.* = try p.arena.alloc(u32, access.elems);
            const window_size = p.locals[access.local].window_size;
            for (element_partitions.*, 0..) |*partition, e| {
                const offset = access.offset + @as(i64, access.elem_size) * @as(i64, @intCast(e));
                const mixed = if (window_size != 0) Window{ .offset = 0, .size = window_size } else p.windows.get(.{ access.local, offset });
                const window = mixed orelse Window{ .offset = offset, .size = access.elem_size };
                const entry = try p.partition_at.getOrPut(p.arena, .{ access.local, window.offset });
                if (!entry.found_existing) {
                    entry.value_ptr.* = @intCast(p.partitions.items.len);
                    // The shifts and masks of a window need an integer register.
                    const class: Mir.Class = if (mixed == null) p.elementClass(access, e) else if (window.size == 4) .b32 else .b64;
                    try p.partitions.append(p.arena, .{ .local = access.local, .offset = window.offset, .size = window.size, .class = class });
                }
                partition.* = entry.value_ptr.*;
            }
        }
        try p.classifyCopies();
        for (p.locals) |*state| {
            const waits = state.state == .memory and state.stored_only and for (state.holders.items) |holder| {
                if (p.locals[holder].state != .promote) break false;
            } else true;
            if (waits) p.again = true;
            if (state.param == none or state.casts or state.root == undef or waits) continue;
            state.state = switch (state.state) {
                .contained => .param_addressing,
                .memory => .param_generic,
                else => state.state,
            };
        }
    }

    /// Returns the size of the window that holds a local with `indexed` accesses, or 0 when the
    /// local stays in memory.
    fn indexedWindowSize(p: *const Promotion, local: u32) u8 {
        const size = p.mir.locals[local].size;
        if (p.locals[local].converts or size > 8) return 0;
        for (p.address_params.items) |param| {
            for (param.incoming.items) |in| if (in.address.local == local) return 0;
        }
        for (p.accesses.items) |access| {
            if (access.local != local) continue;
            if (access.kind != .load and access.kind != .store) return 0;
            if (access.pred != none or !p.supported(access)) return 0;
            const start: i64 = if (access.indexed) 0 else access.offset;
            if (start < 0 or start + access.bytes() > @as(i64, @intCast(size))) return 0;
        }
        return if (size <= 4) 4 else 8;
    }

    /// Merges the ranges of one local that overlap without being equal into windows. A window
    /// grows to the aligned container of 4 or 8 bytes that holds it, and then absorbs the ranges
    /// that the container overlaps. The local stays in memory when no container fits.
    fn findWindows(p: *Promotion, ranges: []const Range) Allocator.Error!void {
        const local = ranges[0].local;
        if (p.locals[local].state != .promote) return;
        const Span = struct { start: i64, end: i64, mixed: bool };
        var spans: std.ArrayList(Span) = .empty;
        for (ranges) |range| try spans.append(p.arena, .{ .start = range.offset, .end = range.offset + range.size, .mixed = false });
        var changed = true;
        while (changed) {
            changed = false;
            std.mem.sort(Span, spans.items, {}, struct {
                fn lessThan(_: void, a: Span, b: Span) bool {
                    return if (a.start != b.start) a.start < b.start else a.end < b.end;
                }
            }.lessThan);
            var kept: usize = 0;
            for (spans.items) |span| {
                if (kept > 0 and span.start < spans.items[kept - 1].end) {
                    const last = &spans.items[kept - 1];
                    last.mixed = last.mixed or span.mixed or span.start != last.start or span.end != last.end;
                    last.end = @max(last.end, span.end);
                } else {
                    spans.items[kept] = span;
                    kept += 1;
                }
            }
            spans.shrinkRetainingCapacity(kept);
            for (spans.items) |*span| {
                if (!span.mixed) continue;
                const size: i64 = for ([_]i64{ 4, 8 }) |size| {
                    if (@divFloor(span.start, size) == @divFloor(span.end - 1, size)) break size;
                } else {
                    p.locals[local].state = .contained;
                    return;
                };
                const start = span.start - @mod(span.start, size);
                if (start != span.start or start + size != span.end) changed = true;
                span.* = .{ .start = start, .end = start + size, .mixed = true };
            }
        }
        var next: usize = 0;
        for (ranges) |range| {
            while (spans.items[next].end <= range.offset) next += 1;
            const span = spans.items[next];
            if (span.mixed) try p.windows.put(p.arena, .{ local, range.offset }, .{ .offset = span.start, .size = @intCast(span.end - span.start) });
        }
    }

    /// Keeps a local in memory when a copy cannot become copies of values, stores, or loads of
    /// fewer than `Layout.max_register_bytes` bytes, the limit of `transforms/expand.zig`. The
    /// value of a `set` must fill each partition that it writes.
    fn classifyCopies(p: *Promotion) Allocator.Error!void {
        try p.spreadRanges();
        const local_partitions = try p.arena.alloc(std.ArrayList(u32), p.locals.len);
        for (local_partitions) |*list| list.* = .empty;
        for (p.partitions.items, 0..) |partition, v| try local_partitions[partition.local].append(p.arena, @intCast(v));
        for (local_partitions) |list| std.mem.sort(u32, list.items, p, struct {
            fn lessThan(q: *Promotion, a: u32, b: u32) bool {
                return q.partitions.items[a].offset < q.partitions.items[b].offset;
            }
        }.lessThan);
        for (p.accesses.items, p.element_partitions) |access, *covered| {
            if (access.kind != .copy_src and access.kind != .copy_dest) continue;
            if (p.locals[access.local].state != .promote) continue;
            var list: std.ArrayList(u32) = .empty;
            for (local_partitions[access.local].items) |v| {
                const partition = p.partitions.items[v];
                const inside = partition.offset >= access.offset and partition.offset + partition.size <= access.offset + access.bytes();
                const outside = partition.offset + partition.size <= access.offset or partition.offset >= access.offset + access.bytes();
                if (inside) try list.append(p.arena, v) else if (!outside) p.locals[access.local].state = .contained;
            }
            covered.* = list.items;
        }
        var changed = true;
        while (changed) {
            changed = false;
            for (p.copy_dest_of, p.copy_src_of) |kd, ks| {
                const dest_promoted = kd != none and p.locals[p.accesses.items[kd].local].state == .promote;
                const src_promoted = ks != none and p.locals[p.accesses.items[ks].local].state == .promote;
                if (dest_promoted) {
                    const ok = if (src_promoted)
                        try p.matchCopy(kd, ks)
                    else if (p.isSet(kd))
                        p.setFits(kd)
                    else
                        p.loadsFit(kd);
                    if (!ok) {
                        p.locals[p.accesses.items[kd].local].state = .contained;
                        changed = true;
                    }
                } else if (src_promoted and !p.storesCover(ks)) {
                    p.locals[p.accesses.items[ks].local].state = .contained;
                    changed = true;
                }
            }
            for (p.address_params.items) |*param| {
                if (!param.speculates) continue;
                for (param.incoming.items) |in| {
                    if (p.locals[in.address.local].state != .promote) break;
                } else continue;
                param.speculates = false;
                for (param.incoming.items) |in| p.escape(in.address.local);
                changed = true;
            }
        }
    }

    /// Gives both ends of each copy between two promoted locals the partitions of the other end.
    /// SROA of LLVM forms its partitions from the slices of `memcpy` in this way. A partition that
    /// overlaps a partition of another size keeps both locals in memory.
    fn spreadRanges(p: *Promotion) Allocator.Error!void {
        const local_partitions = try p.arena.alloc(std.ArrayList(u32), p.locals.len);
        for (local_partitions) |*list| list.* = .empty;
        for (p.partitions.items, 0..) |partition, v| try local_partitions[partition.local].append(p.arena, @intCast(v));
        var changed = true;
        while (changed) {
            changed = false;
            for (p.copy_dest_of, p.copy_src_of) |kd, ks| {
                if (kd == none or ks == none) continue;
                const dest = p.accesses.items[kd];
                const src = p.accesses.items[ks];
                if (p.locals[dest.local].state != .promote or p.locals[src.local].state != .promote) continue;
                for ([2]Access{ dest, src }, [2]Access{ src, dest }) |from, to| {
                    var index: usize = 0;
                    while (index < local_partitions[from.local].items.len) : (index += 1) {
                        const partition = p.partitions.items[local_partitions[from.local].items[index]];
                        if (partition.offset < from.offset or partition.offset + partition.size > from.offset + from.bytes()) continue;
                        const offset = to.offset + partition.offset - from.offset;
                        if (p.partition_at.get(.{ to.local, offset })) |existing| {
                            if (p.partitions.items[existing].size == partition.size) continue;
                        }
                        const overlaps = for (local_partitions[to.local].items) |other| {
                            const o = p.partitions.items[other];
                            if (o.offset < offset + partition.size and offset < o.offset + o.size) break true;
                        } else false;
                        if (overlaps) {
                            p.locals[dest.local].state = .contained;
                            p.locals[src.local].state = .contained;
                            break;
                        }
                        const v: u32 = @intCast(p.partitions.items.len);
                        try p.partitions.append(p.arena, .{ .local = to.local, .offset = offset, .size = partition.size, .class = partition.class });
                        try p.partition_at.put(p.arena, .{ to.local, offset }, v);
                        try local_partitions[to.local].append(p.arena, v);
                        changed = true;
                    }
                    if (p.locals[dest.local].state != .promote) break;
                }
            }
        }
    }

    fn matchCopy(p: *Promotion, kd: u32, ks: u32) Allocator.Error!bool {
        const dest = p.accesses.items[kd];
        const src = p.accesses.items[ks];
        const sources = try p.arena.alloc(u32, p.element_partitions[kd].len);
        for (sources, p.element_partitions[kd]) |*source, v| {
            const partition = p.partitions.items[v];
            const at = p.partition_at.get(.{ src.local, src.offset + partition.offset - dest.offset }) orelse return false;
            if (p.partitions.items[at].size != partition.size) return false;
            source.* = at;
        }
        p.copy_sources[kd] = sources;
        return true;
    }

    fn isSet(p: *const Promotion, k: u32) bool {
        return p.inst(p.accesses.items[k].inst).modifiers.copy.op == .set;
    }

    /// Returns whether the value of a `set` fills each partition that it writes.
    fn setFits(p: *const Promotion, kd: u32) bool {
        const dest = p.accesses.items[kd];
        for (p.element_partitions[kd]) |v| {
            const partition = p.partitions.items[v];
            if (partition.size != dest.elem_size or !p.unitAligned(dest, partition)) return false;
        }
        return true;
    }

    /// Returns whether loads of the partitions that a copy writes fit its alignment and size limit.
    fn loadsFit(p: *const Promotion, kd: u32) bool {
        const dest = p.accesses.items[kd];
        var bytes: u64 = 0;
        for (p.element_partitions[kd]) |v| {
            const partition = p.partitions.items[v];
            if (!p.unitAligned(dest, partition)) return false;
            bytes += partition.size;
        }
        return bytes < Layout.max_register_bytes;
    }

    /// Returns whether stores of the partitions that a copy reads can replace the copy.
    fn storesCover(p: *const Promotion, ks: u32) bool {
        const src = p.accesses.items[ks];
        var bytes: i64 = 0;
        for (p.element_partitions[ks]) |v| {
            const partition = p.partitions.items[v];
            if (!p.unitAligned(src, partition)) return false;
            bytes += partition.size;
        }
        return bytes == src.bytes();
    }

    /// The copy unit divides the alignments of both ends. A partition of at most one unit is
    /// aligned at a copy offset that is a multiple of its size (PTX ISA 9.4, section 6.4.1).
    fn unitAligned(p: *const Promotion, access: Access, partition: Partition) bool {
        _ = p;
        const offset: u64 = @intCast(partition.offset - access.offset);
        const alignment = if (offset == 0) access.elem_size else @min(access.elem_size, offset & -%offset);
        return alignment >= partition.size;
    }

    /// Returns whether the transform can replace the access with copies of values.
    fn supported(p: *const Promotion, access: Access) bool {
        const instruction = p.inst(access.inst);
        const ops = p.operandsAt(instruction) catch return false;
        const value_at = if (access.kind == .store) ops[ops.len - 1] else ops[0];
        const value = p.operand(value_at);
        const class = if (value.kind == .reg) value.regClass() else if (value.kind == .list and value.payload > 0)
            @as(Mir.Operand, @bitCast(p.mir.extra[value_at + 1])).regClass()
        else
            return false;
        return switch (access.elem_size) {
            1 => class == .b16 and (access.kind == .store or instruction.ty == .u8 or instruction.ty == .s8),
            2 => class == .b16,
            4 => class == .b32 or class == .f32,
            8 => class == .b64 or class == .f64,
            else => false,
        };
    }

    fn elementClass(p: *const Promotion, access: Access, e: usize) Mir.Class {
        const instruction = p.inst(access.inst);
        const ops = p.operandsAt(instruction) catch unreachable;
        const value_at = if (access.kind == .store) ops[ops.len - 1] else ops[0];
        const value = p.operand(value_at);
        if (value.kind == .reg) return value.regClass();
        return @as(Mir.Operand, @bitCast(p.mir.extra[value_at + 1 + e])).regClass();
    }

    /// Returns whether `instruction` writes a local that the address of a parameter replaces.
    fn writesParamCopy(p: *const Promotion, instruction: Mir.Inst) bool {
        if (instruction.tag != .st and instruction.tag != .copy) return false;
        const range = p.mir.operands(instruction) orelse return false;
        const at = range.first + @intFromBool(instruction.modifiers.guard() != .none);
        const base = p.operand(at);
        if (base.kind != .reg) return false;
        const local = p.addresses[p.regId(p.mir.extra[at])].local;
        if (local == none) return false;
        return p.locals[local].state == .param_addressing or p.locals[local].state == .param_generic;
    }

    /// Replaces the root of a parameter copy with the `.param` or generic address of the parameter.
    fn emitParamRoot(p: *const Promotion, b: *Mir.Builder, instruction: Mir.Inst, local: u32) Allocator.Error!void {
        const state = p.locals[local];
        const dst = p.mir.extra[instruction.data];
        const param: u32 = @bitCast(Mir.Operand.param(.own, @intCast(state.param)));
        if (state.state == .param_addressing) return b.addTyped(.mov, .u64, .{ .none = .{} }, &.{ dst, param });
        try b.addTyped(.cvta, .u64, .{ .cvta = .{ .space = .param } }, &.{ dst, param });
    }

    /// Emits an instruction of the address tree of a local in one window so that the instruction
    /// computes the offset from the start of the local: the root gives 0, and a `cvta` becomes a
    /// `mov`.
    fn emitOffset(p: *const Promotion, b: *Mir.Builder, instruction: Mir.Inst) Allocator.Error!void {
        const ops = try p.operandsAt(instruction);
        const dst = p.mir.extra[ops[0]];
        const ty: Mir.Type = if (p.operand(ops[0]).regClass() == .b64) .u64 else .u32;
        if (p.operand(ops[1]).kind == .local) return b.addTyped(.mov, ty, .{ .none = .{} }, &.{ dst, immOperand(0) });
        if (instruction.tag == .cvta) return b.addTyped(.mov, ty, .{ .none = .{} }, &.{ dst, p.mir.extra[ops[1]] });
        try b.addTyped(instruction.tag, instruction.ty, instruction.modifiers, p.mir.span(instruction));
    }

    /// Returns whether element `e` of a store fills only some bytes of the partition `v`, so that
    /// the store reads the other bytes.
    fn fillsPartial(p: *const Promotion, access: Access, v: u32) bool {
        return access.kind == .store and access.elem_size < p.partitions.items[v].size;
    }

    fn changes(p: *const Promotion) bool {
        for (p.locals) |state| if (state.state != .memory) return true;
        return false;
    }

    fn accessValues(p: *const Promotion, access: Access) []const u32 {
        const instruction = p.inst(access.inst);
        const ops = p.operandsAt(instruction) catch unreachable;
        const value_at = if (access.kind == .store) ops[ops.len - 1] else ops[0];
        const value = p.operand(value_at);
        if (value.kind == .reg) return p.mir.extra[value_at..][0..1];
        return p.mir.extra[value_at + 1 ..][0..value.payload];
    }

    /// Returns false and changes nothing when the new registers do not fit an operand.
    fn rewrite(p: *Promotion, gpa: Allocator, mir: *Mir) Allocator.Error!bool {
        var ssa: Ssa = try .init(p);
        try ssa.placeParams();
        try ssa.rename();
        return ssa.emit(gpa, mir);
    }
};

/// Builds the values of the partitions of the promoted locals.
const Ssa = struct {
    p: *Promotion,
    arena: Allocator,
    block_accesses: [][]const u32,
    placements: std.ArrayList(Placement) = .empty,
    block_placements: [][2]u32,
    /// The value of each partition of each access, in the order of `Promotion.element_partitions`.
    values: [][]u32,
    /// The value of the window before each element of a store that fills only part of a window.
    before: [][]u32,
    /// The arguments of each block for its successors `[0]` and `[1]`.
    edge_args: [][2][]u32,
    load_params: []std.ArrayList(LoadParam),
    load_values: std.AutoHashMapUnmanaged(u32, struct { access: u32, values: []const u32 }) = .empty,
    /// Maps a load through an `AddressParam` and a predecessor to its access in the predecessor.
    edge_loads: std.AutoHashMapUnmanaged(struct { u32, u32 }, u32) = .empty,
    drops_params: []bool,
    next_regs: [Mir.class_count]u32,

    fn init(p: *Promotion) Allocator.Error!Ssa {
        const arena = p.arena;
        const block_count = p.cfg.blockCount();
        var lists = try arena.alloc(std.ArrayList(u32), block_count);
        for (lists) |*list| list.* = .empty;
        for (p.accesses.items, 0..) |access, k| {
            if (p.locals[access.local].state != .promote or access.pred != none) continue;
            try lists[p.inst_block[access.inst]].append(arena, @intCast(k));
        }
        var edge_loads: std.AutoHashMapUnmanaged(struct { u32, u32 }, u32) = .empty;
        for (p.accesses.items, 0..) |access, k| {
            if (p.locals[access.local].state != .promote or access.pred == none) continue;
            try lists[access.pred].append(arena, @intCast(k));
            try edge_loads.put(arena, .{ access.inst, access.pred }, @intCast(k));
        }
        const block_accesses = try arena.alloc([]const u32, block_count);
        for (block_accesses, lists) |*b, list| b.* = list.items;
        const values = try arena.alloc([]u32, p.accesses.items.len);
        for (values, p.element_partitions) |*access_values, partitions| {
            access_values.* = try arena.alloc(u32, partitions.len);
            @memset(access_values.*, undef);
        }
        const before = try arena.alloc([]u32, p.accesses.items.len);
        for (before, p.element_partitions) |*access_before, partitions| {
            access_before.* = try arena.alloc(u32, partitions.len);
            @memset(access_before.*, undef);
        }
        const edge_args = try arena.alloc([2][]u32, block_count);
        @memset(edge_args, .{ &.{}, &.{} });
        const block_placements = try arena.alloc([2]u32, block_count);
        @memset(block_placements, .{ 0, 0 });
        const load_params = try arena.alloc(std.ArrayList(LoadParam), block_count);
        @memset(load_params, .empty);
        const drops_params = try arena.alloc(bool, block_count);
        @memset(drops_params, false);
        var ssa: Ssa = .{
            .p = p,
            .arena = arena,
            .block_accesses = block_accesses,
            .block_placements = block_placements,
            .values = values,
            .before = before,
            .edge_args = edge_args,
            .load_params = load_params,
            .edge_loads = edge_loads,
            .drops_params = drops_params,
            .next_regs = p.mir.reg_counts,
        };
        for (p.address_params.items) |param| {
            if (!param.speculates) continue;
            drops_params[param.block] = true;
            for (param.loads.items) |u| {
                const k = edge_loads.get(.{ u, param.incoming.items[0].pred }).?;
                const results = p.accessValues(p.accesses.items[k]);
                const regs = try arena.alloc(u32, results.len);
                for (regs, results, 0..) |*reg, result, e| {
                    reg.* = ssa.newReg(@as(Mir.Operand, @bitCast(result)).regClass());
                    try load_params[param.block].append(arena, .{ .load = u, .elem = @intCast(e), .reg = reg.* });
                }
                try ssa.load_values.put(arena, u, .{ .access = k, .values = regs });
            }
        }
        return ssa;
    }

    fn changesParams(ssa: *const Ssa, block: u32) bool {
        return ssa.placementsOf(block).len > 0 or ssa.drops_params[block];
    }

    fn dropsParam(ssa: *const Ssa, block: u32, index: usize) bool {
        const p = ssa.p;
        const param = p.address_param_at.get(.{ block, @intCast(index) }) orelse return false;
        return p.address_params.items[param].speculates;
    }

    fn newReg(ssa: *Ssa, class: Mir.Class) u32 {
        const index = ssa.next_regs[@backingInt(class)];
        ssa.next_regs[@backingInt(class)] += 1;
        return @bitCast(Mir.Operand{ .kind = .reg, .payload = (@as(u28, @backingInt(class)) << 16) | @as(u28, @truncate(index)) });
    }

    /// Puts a parameter for each partition into each block of the iterated dominance frontier of
    /// its stores where a load can read the value of a predecessor.
    fn placeParams(ssa: *Ssa) Allocator.Error!void {
        const p = ssa.p;
        const cfg = &p.cfg;
        const arena = ssa.arena;
        const block_count = cfg.blockCount();
        const frontier = try arena.alloc(std.ArrayList(u32), block_count);
        for (frontier) |*f| f.* = .empty;
        for (cfg.order) |b| {
            const preds = cfg.predecessors[b];
            if (preds.len < 2) continue;
            for (preds) |pred| {
                if (cfg.rank[pred] == none) continue;
                var runner = pred;
                while (runner != cfg.idom[b] and runner != none) {
                    if (std.mem.findScalar(u32, frontier[runner].items, b) == null) try frontier[runner].append(arena, b);
                    runner = cfg.idom[runner];
                }
            }
        }
        const Info = struct { defs: std.ArrayList(u32) = .empty, exposed: std.ArrayList(u32) = .empty };
        const infos = try arena.alloc(Info, p.partitions.items.len);
        @memset(infos, .{});
        const seen = try arena.alloc(u32, p.partitions.items.len);
        @memset(seen, none);
        for (cfg.order) |b| {
            for (ssa.block_accesses[b]) |k| {
                const access = p.accesses.items[k];
                for (p.element_partitions[k]) |v| {
                    if (seen[v] == b) continue;
                    seen[v] = b;
                    if (access.writes() and !p.fillsPartial(access, v)) try infos[v].defs.append(arena, b) else try infos[v].exposed.append(arena, b);
                }
            }
            for (ssa.block_accesses[b]) |k| {
                const access = p.accesses.items[k];
                if (!access.writes()) continue;
                for (p.element_partitions[k]) |v| {
                    if (std.mem.findScalarLast(u32, infos[v].defs.items, b) == null) try infos[v].defs.append(arena, b);
                }
            }
        }
        const live_mark = try arena.alloc(u32, block_count);
        const def_mark = try arena.alloc(u32, block_count);
        const frontier_mark = try arena.alloc(u32, block_count);
        @memset(live_mark, none);
        @memset(def_mark, none);
        @memset(frontier_mark, none);
        var work: std.ArrayList(u32) = .empty;
        for (infos, 0..) |info, index| {
            const v: u32 = @intCast(index);
            if (info.defs.items.len == 0 or info.exposed.items.len == 0) continue;
            for (info.defs.items) |b| def_mark[b] = v;
            work.clearRetainingCapacity();
            for (info.exposed.items) |b| {
                if (live_mark[b] == v) continue;
                live_mark[b] = v;
                try work.append(arena, b);
            }
            while (work.pop()) |b| for (cfg.predecessors[b]) |pred| {
                if (cfg.rank[pred] == none or def_mark[pred] == v or live_mark[pred] == v) continue;
                live_mark[pred] = v;
                try work.append(arena, pred);
            };
            work.clearRetainingCapacity();
            try work.appendSlice(arena, info.defs.items);
            while (work.pop()) |b| for (frontier[b].items) |y| {
                if (frontier_mark[y] == v) continue;
                frontier_mark[y] = v;
                if (live_mark[y] == v) try ssa.placements.append(arena, .{ .block = y, .partition = v, .param = ssa.newReg(p.partitions.items[v].class) });
                if (def_mark[y] != v) try work.append(arena, y);
            };
        }
        std.mem.sort(Placement, ssa.placements.items, {}, struct {
            fn lessThan(_: void, a: Placement, b: Placement) bool {
                if (a.block != b.block) return a.block < b.block;
                return a.partition < b.partition;
            }
        }.lessThan);
        var i: u32 = 0;
        while (i < ssa.placements.items.len) {
            const b = ssa.placements.items[i].block;
            const start = i;
            while (i < ssa.placements.items.len and ssa.placements.items[i].block == b) i += 1;
            ssa.block_placements[b] = .{ start, i };
        }
    }

    fn placementsOf(ssa: *const Ssa, b: u32) []const Placement {
        const range = ssa.block_placements[b];
        return ssa.placements.items[range[0]..range[1]];
    }

    /// Gives each load the value of the latest store. This follows the rename pass of LLVM.
    fn rename(ssa: *Ssa) Allocator.Error!void {
        const p = ssa.p;
        const cfg = &p.cfg;
        const arena = ssa.arena;
        const block_count = cfg.blockCount();
        var children = try arena.alloc(std.ArrayList(u32), block_count);
        for (children) |*c| c.* = .empty;
        for (cfg.order[1..]) |b| try children[cfg.idom[b]].append(arena, b);
        const current = try arena.alloc(u32, p.partitions.items.len);
        @memset(current, undef);
        const Undo = struct { partition: u32, value: u32 };
        var undo: std.ArrayList(Undo) = .empty;
        const Frame = struct { block: u32, mark: usize, next: u32 };
        var stack: std.ArrayList(Frame) = .empty;
        try stack.append(arena, .{ .block = 0, .mark = 0, .next = 0 });
        try ssa.enter(0, current, &undo);
        while (stack.items.len > 0) {
            const top = &stack.items[stack.items.len - 1];
            if (top.next < children[top.block].items.len) {
                const child = children[top.block].items[top.next];
                top.next += 1;
                const mark = undo.items.len;
                try stack.append(arena, .{ .block = child, .mark = mark, .next = 0 });
                try ssa.enter(child, current, &undo);
                continue;
            }
            while (undo.items.len > top.mark) {
                const entry = undo.pop().?;
                current[entry.partition] = entry.value;
            }
            _ = stack.pop();
        }
    }

    fn enter(ssa: *Ssa, b: u32, current: []u32, undo: anytype) Allocator.Error!void {
        const p = ssa.p;
        for (ssa.placementsOf(b)) |placement| {
            try undo.append(ssa.arena, .{ .partition = placement.partition, .value = current[placement.partition] });
            current[placement.partition] = placement.param;
        }
        for (ssa.block_accesses[b]) |k| {
            const access = p.accesses.items[k];
            const values = ssa.values[k];
            switch (access.kind) {
                .load, .copy_src => for (p.element_partitions[k], values) |v, *value| {
                    value.* = current[v];
                },
                .store => for (p.element_partitions[k], p.accessValues(access), values, ssa.before[k]) |v, stored, *value, *old| {
                    try undo.append(ssa.arena, .{ .partition = v, .value = current[v] });
                    const partition = p.partitions.items[v];
                    if (access.elem_size == partition.size) {
                        current[v] = stored;
                        value.* = stored;
                    } else {
                        old.* = current[v];
                        value.* = ssa.newReg(partition.class);
                        current[v] = value.*;
                    }
                },
                .copy_dest => {
                    // A copy from memory loads each value into a new register. The copy reads
                    // every source before it writes, which `move` needs when the ends overlap.
                    const ks = p.copy_src_of[access.inst];
                    const from_values = ks != none and p.locals[p.accesses.items[ks].local].state == .promote;
                    const set_value: u32 = if (p.isSet(k)) p.mir.extra[(try p.operandsAt(p.inst(access.inst)))[1]] else undef;
                    for (p.element_partitions[k], values, 0..) |v, *value, e| {
                        value.* = if (set_value != undef)
                            set_value
                        else if (from_values)
                            current[p.copy_sources[k][e]]
                        else
                            ssa.newReg(p.partitions.items[v].class);
                    }
                    for (p.element_partitions[k], values) |v, value| {
                        try undo.append(ssa.arena, .{ .partition = v, .value = current[v] });
                        current[v] = value;
                    }
                },
            }
        }
        for (p.cfg.successors[b], 0..) |s, slot| {
            if (s == none) continue;
            const placements = ssa.placementsOf(s);
            if (placements.len == 0) continue;
            const args = try ssa.arena.alloc(u32, placements.len);
            for (args, placements) |*arg, placement| arg.* = current[placement.partition];
            ssa.edge_args[b][slot] = args;
        }
    }

    /// Returns `value` in a register of `class`, a `mov` copy or, for `undef`, a new undefined one.
    fn argOf(ssa: *Ssa, b: *Mir.Builder, value: u32, class: Mir.Class) Allocator.Error!u32 {
        if (value == undef) return ssa.newReg(class);
        const op: Mir.Operand = @bitCast(value);
        if (op.regClass() == class) return value;
        const moved = ssa.newReg(class);
        try b.addTyped(.mov, class.bitType(), .{ .none = .{} }, &.{ moved, value });
        return moved;
    }

    fn edgeArgs(ssa: *Ssa, b: *Mir.Builder, block: u32, slot: usize, target: u32) Allocator.Error![]const u32 {
        const placements = ssa.placementsOf(target);
        const values = ssa.edge_args[block][slot];
        const args = try ssa.arena.alloc(u32, placements.len);
        for (args, placements, 0..) |*arg, placement, k| {
            const value = if (values.len == placements.len) values[k] else undef;
            arg.* = try ssa.argOf(b, value, ssa.p.partitions.items[placement.partition].class);
        }
        return args;
    }

    fn emit(ssa: *Ssa, gpa: Allocator, mir: *Mir) Allocator.Error!bool {
        const p = ssa.p;
        const cfg = &p.cfg;
        const arena = ssa.arena;
        var local_map = try arena.alloc(u32, mir.locals.len);
        var kept_locals: std.ArrayList(Mir.Local) = .empty;
        for (mir.locals, p.locals, 0..) |local, state, index| {
            if (state.state != .contained and state.state != .memory) {
                local_map[index] = none;
                continue;
            }
            local_map[index] = @intCast(kept_locals.items.len);
            try kept_locals.append(arena, local);
        }
        p.local_map = local_map;
        const access_of = try arena.alloc(u32, mir.instructions.len);
        @memset(access_of, none);
        for (p.accesses.items, 0..) |access, k| {
            if (access.pred == none) access_of[access.inst] = @intCast(k);
        }

        var b: Mir.Builder = .{ .gpa = gpa, .reg_counts = mir.reg_counts, .label_count = mir.label_count };
        defer b.deinit();
        var trampolines: std.ArrayList(Trampoline) = .empty;
        var words: std.ArrayList(u32) = .empty;
        var needs_cvta_param = false;
        for (0..mir.instructions.len) |index| {
            const i: u32 = @intCast(index);
            const instruction = p.inst(i);
            const block = p.inst_block[i];
            if (i == cfg.starts[block] and block > 0 and ssa.placementsOf(block).len > 0) {
                const prev = block - 1;
                const prev_ends_in_branch = cfg.starts[prev] < cfg.starts[block] and p.inst(cfg.starts[block] - 1).tag == .bra;
                for (cfg.successors[prev], 0..) |s, slot| {
                    if (s != block or (prev_ends_in_branch and slot == 0)) continue;
                    try b.addBranch(mir.labelOf(instruction), try ssa.edgeArgs(&b, prev, slot, block));
                }
            }
            const owner = p.tree_local[i];
            if (owner != none and p.locals[owner].state == .promote) {
                if (p.locals[owner].window_size != 0) try p.emitOffset(&b, instruction);
                continue;
            }
            if (p.writesParamCopy(instruction)) continue;
            if (owner != none and p.locals[owner].root == mir.extra[instruction.data] and
                (p.locals[owner].state == .param_addressing or p.locals[owner].state == .param_generic))
            {
                try p.emitParamRoot(&b, instruction, owner);
                if (p.locals[owner].state == .param_generic) needs_cvta_param = true;
                continue;
            }
            if (instruction.tag == .copy) {
                try ssa.emitCopy(&b, i, local_map);
                continue;
            }
            if (ssa.load_values.get(i)) |load| {
                try ssa.emitLoad(&b, load.access, load.values);
                continue;
            }
            const k = access_of[i];
            if (k != none and p.locals[p.accesses.items[k].local].state == .promote) {
                const access = p.accesses.items[k];
                if (access.kind == .load) try ssa.emitLoad(&b, k, ssa.values[k]);
                if (access.kind == .store) try ssa.emitStore(&b, k);
                continue;
            }
            switch (instruction.tag) {
                .label => {
                    words.clearRetainingCapacity();
                    for (mir.blockParams(instruction), 0..) |param, position| {
                        if (!ssa.dropsParam(block, position)) try words.append(arena, param);
                    }
                    for (ssa.placementsOf(block)) |placement| try words.append(arena, placement.param);
                    for (ssa.load_params[block].items) |param| try words.append(arena, param.reg);
                    try b.addLabel(mir.labelOf(instruction), words.items);
                    continue;
                },
                .bra => {
                    const target_label = mir.branchTarget(instruction);
                    const target = cfg.label_blocks[target_label];
                    if (target != none and ssa.changesParams(target)) {
                        if (instruction.modifiers.guard() == .none) {
                            const args = try ssa.edgeArgs(&b, block, 0, target);
                            words.clearRetainingCapacity();
                            for (mir.branchArgs(instruction), 0..) |arg, position| {
                                if (!ssa.dropsParam(target, position)) try words.append(arena, arg);
                            }
                            try words.appendSlice(arena, args);
                            for (ssa.load_params[target].items) |param| {
                                const value = ssa.values[ssa.edge_loads.get(.{ param.load, block }).?][param.elem];
                                try words.append(arena, try ssa.argOf(&b, value, @as(Mir.Operand, @bitCast(param.reg)).regClass()));
                            }
                            try b.addBranch(target_label, words.items);
                        } else {
                            const label = b.label_count;
                            b.label_count += 1;
                            try trampolines.append(arena, .{ .label = label, .target = target, .args = ssa.edge_args[block][0] });
                            words.clearRetainingCapacity();
                            try words.appendSlice(arena, mir.span(instruction));
                            const guard_words: u32 = 1;
                            words.items[guard_words] = @bitCast(Mir.Operand{ .kind = .label, .payload = @intCast(label) });
                            try b.addTyped(.bra, instruction.ty, instruction.modifiers, words.items);
                        }
                        continue;
                    }
                },
                else => {},
            }
            words.clearRetainingCapacity();
            try words.appendSlice(arena, mir.span(instruction));
            if (k != none) {
                const mem_at = mir.memOperandAt(instruction) - instruction.data;
                words.items[mem_at] = @bitCast(accessOperand(@bitCast(words.items[mem_at]), p.locals[p.accesses.items[k].local].state));
            }
            try ssa.renumberLocals(instruction, words.items, local_map);
            try b.addTyped(instruction.tag, instruction.ty, instruction.modifiers, words.items);
        }
        for (trampolines.items) |trampoline| {
            try b.addLabel(trampoline.label, &.{});
            const target_label = mir.labelOf(p.inst(cfg.starts[trampoline.target]));
            const placements = ssa.placementsOf(trampoline.target);
            const args = try arena.alloc(u32, placements.len);
            for (args, placements, 0..) |*arg, placement, a| {
                const value = if (trampoline.args.len == placements.len) trampoline.args[a] else undef;
                arg.* = try ssa.argOf(&b, value, p.partitions.items[placement.partition].class);
            }
            try b.addBranch(target_label, args);
        }
        for (ssa.next_regs) |count| if (count > std.math.maxInt(u16) + 1) return false;
        if (needs_cvta_param) mir.ptx_version = @max(mir.ptx_version, 77);
        var instructions = b.instructions;
        b.instructions = .empty;
        errdefer instructions.deinit(gpa);
        const extra = try b.extra.toOwnedSlice(gpa);
        errdefer gpa.free(extra);
        const locals = try gpa.dupe(Mir.Local, kept_locals.items);
        mir.instructions.deinit(gpa);
        gpa.free(mir.extra);
        gpa.free(mir.locals);
        mir.instructions = instructions.toOwnedSlice();
        mir.extra = extra;
        mir.locals = locals;
        mir.label_count = b.label_count;
        mir.reg_counts = ssa.next_regs;
        return true;
    }

    /// A copy between two promoted locals needs no instruction. A copy into a promoted local
    /// becomes loads, a copy out of one becomes stores, and another copy gets new state spaces.
    fn emitCopy(ssa: *Ssa, b: *Mir.Builder, i: u32, local_map: []const u32) Allocator.Error!void {
        const p = ssa.p;
        const instruction = p.inst(i);
        const ops = try p.operandsAt(instruction);
        const kd = p.copy_dest_of[i];
        const ks = p.copy_src_of[i];
        const dest_state: State = if (kd != none) p.locals[p.accesses.items[kd].local].state else .memory;
        const src_state: State = if (ks != none) p.locals[p.accesses.items[ks].local].state else .memory;
        if (dest_state == .promote and (src_state == .promote or p.isSet(kd))) return;
        if (dest_state == .promote) {
            const mem = accessOperand(p.mir.memOperand(instruction, 1), src_state);
            const dest = p.accesses.items[kd];
            for (p.element_partitions[kd], ssa.values[kd]) |v, value| {
                if (value == undef) continue;
                const offset = p.partitions.items[v].offset - dest.offset;
                try emitUnitAccess(b, .ld, value, p.mir.extra[ops[1]], offset, mem, dest.elem_size, p.partitions.items[v].size);
            }
            return;
        }
        if (src_state == .promote) {
            const mem = accessOperand(p.mir.memOperand(instruction, 0), dest_state);
            const src = p.accesses.items[ks];
            for (p.element_partitions[ks], ssa.values[ks]) |v, value| {
                if (value == undef) continue;
                const offset = p.partitions.items[v].offset - src.offset;
                try emitUnitAccess(b, .st, value, p.mir.extra[ops[0]], offset, mem, src.elem_size, p.partitions.items[v].size);
            }
            return;
        }
        const words = try ssa.arena.dupe(u32, p.mir.span(instruction));
        const mem_at = p.mir.memOperandAt(instruction) - instruction.data;
        words[mem_at] = @bitCast(accessOperand(@bitCast(words[mem_at]), dest_state));
        if (instruction.modifiers.copy.op != .set) words[mem_at + 1] = @bitCast(accessOperand(@bitCast(words[mem_at + 1]), src_state));
        try ssa.renumberLocals(instruction, words, local_map);
        try b.addTyped(.copy, instruction.ty, instruction.modifiers, words);
    }

    fn renumberLocals(ssa: *const Ssa, instruction: Mir.Inst, words: []u32, local_map: []const u32) Allocator.Error!void {
        const p = ssa.p;
        const range = p.mir.operands(instruction) orelse return;
        var at = range.first;
        for (0..range.count) |_| {
            const op: Mir.Operand = @bitCast(p.mir.extra[at]);
            if (op.kind == .local) {
                words[at - instruction.data] = @bitCast(Mir.Operand{ .kind = .local, .payload = @intCast(local_map[op.payload]) });
            }
            at += op.len();
        }
    }

    /// Replaces a load with a copy of each value. A byte loaded as `.u8` or `.s8` gets the
    /// extension that `ld` gives (PTX ISA 9.4, section 9.7.10.8). An element that fills only part
    /// of its window comes out of the window register.
    fn emitLoad(ssa: *Ssa, b: *Mir.Builder, k: u32, values: []const u32) Allocator.Error!void {
        const p = ssa.p;
        const access = p.accesses.items[k];
        const results = p.accessValues(access);
        const ty = p.inst(access.inst).ty;
        for (results, values, p.element_partitions[k], 0..) |result, value, v, e| {
            if (value == undef) continue;
            const partition = p.partitions.items[v];
            if (access.elem_size < partition.size and access.indexed) {
                try ssa.extractAt(b, partition, access.elem_size, try ssa.bitPosition(b, access, e), ty == .s8, result, value);
                continue;
            }
            if (access.elem_size < partition.size) {
                const shift: u6 = @intCast(8 * (access.offset + @as(i64, access.elem_size) * @as(i64, @intCast(e)) - partition.offset));
                try ssa.extract(b, partition, access.elem_size, shift, ty == .s8, result, value);
                continue;
            }
            const class = @as(Mir.Operand, @bitCast(result)).regClass();
            if (access.elem_size == 1 and ty == .s8) {
                try b.addTyped(.cvt, .s16, .{ .cvt = .{ .from = .s8 } }, &.{ result, value });
            } else if (access.elem_size == 1) {
                try b.addTyped(.@"and", .b16, .{ .none = .{} }, &.{ result, value, @bitCast(Mir.Operand{ .kind = .imm, .payload = 0xff }) });
            } else {
                try b.addTyped(.mov, class.bitType(), .{ .none = .{} }, &.{ result, value });
            }
        }
    }

    /// Writes the elements of a store that fill only part of a window into the window register.
    fn emitStore(ssa: *Ssa, b: *Mir.Builder, k: u32) Allocator.Error!void {
        const p = ssa.p;
        const access = p.accesses.items[k];
        for (p.accessValues(access), p.element_partitions[k], ssa.values[k], ssa.before[k], 0..) |stored, v, new, old, e| {
            const partition = p.partitions.items[v];
            if (access.elem_size == partition.size) continue;
            if (access.indexed) {
                try ssa.insertAt(b, partition, access.elem_size, try ssa.bitPosition(b, access, e), new, try ssa.argOf(b, old, partition.class), stored);
                continue;
            }
            const shift: u6 = @intCast(8 * (access.offset + @as(i64, access.elem_size) * @as(i64, @intCast(e)) - partition.offset));
            try ssa.insert(b, partition, access.elem_size, shift, new, try ssa.argOf(b, old, partition.class), stored);
        }
    }

    /// Writes `result`, the `size` bytes at bit `shift` of the little-endian window register
    /// `window`. A signed byte gets sign extension and an unsigned one zero extension, as `ld.s8`
    /// and `ld.u8` give. The halves of a window come from `mov` unpacking (PTX ISA 9.4,
    /// section 9.7.10.4). No float passes through `cvt`, so NaN payloads survive.
    fn extract(ssa: *Ssa, b: *Mir.Builder, partition: Partition, size: u8, shift: u6, signed: bool, result: u32, stored: u32) Allocator.Error!void {
        const bits: u8 = partition.size * 8;
        const class = @as(Mir.Operand, @bitCast(result)).regClass();
        const word: Mir.Type = if (bits == 32) .b32 else .b64;
        const unsigned: Mir.Type = if (bits == 32) .u32 else .u64;
        if (size * 8 == bits / 2) {
            // `mov` unpacks a register of any class of the window size.
            const fits = stored != undef and @as(Mir.Operand, @bitCast(stored)).regClass().bits() == bits;
            const window = if (fits) stored else try ssa.argOf(b, stored, partition.class);
            const half = if (class == .f32) ssa.newReg(.b32) else result;
            const other = ssa.newReg(@as(Mir.Operand, @bitCast(half)).regClass());
            const halves = if (shift == 0) [2]u32{ half, other } else [2]u32{ other, half };
            try b.addTyped(.mov, word, .{ .none = .{} }, &.{ listOperand(2), halves[0], halves[1], window });
            if (half != result) try b.addTyped(.mov, .b32, .{ .none = .{} }, &.{ result, half });
            return;
        }
        const window = try ssa.argOf(b, stored, partition.class);
        var field = window;
        if (size == 1 and signed) {
            if (bits - 8 - shift != 0) {
                field = ssa.newReg(partition.class);
                try b.addTyped(.shl, word, .{ .none = .{} }, &.{ field, window, immOperand(bits - 8 - shift) });
            }
            const extended = ssa.newReg(partition.class);
            try b.addTyped(.shr, if (bits == 32) .s32 else .s64, .{ .none = .{} }, &.{ extended, field, immOperand(bits - 8) });
            field = extended;
        } else {
            if (shift != 0) {
                field = ssa.newReg(partition.class);
                try b.addTyped(.shr, unsigned, .{ .none = .{} }, &.{ field, window, immOperand(shift) });
            }
            if (size == 1) {
                const masked = ssa.newReg(partition.class);
                try b.addTyped(.@"and", word, .{ .none = .{} }, &.{ masked, field, immOperand(0xff) });
                field = masked;
            }
        }
        try b.addTyped(.cvt, .u16, .{ .cvt = .{ .from = unsigned } }, &.{ result, field });
    }

    /// Writes `new`, the window register `old` with the low `size` bytes of `stored` at bit
    /// `shift`. The bytes of `stored` above `size` do not matter.
    fn insert(ssa: *Ssa, b: *Mir.Builder, partition: Partition, size: u8, shift: u6, new: u32, old: u32, stored: u32) Allocator.Error!void {
        const bits: u8 = partition.size * 8;
        const word: Mir.Type = if (bits == 32) .b32 else .b64;
        var field = stored;
        const stored_class = @as(Mir.Operand, @bitCast(stored)).regClass();
        if (stored_class == .f32) {
            field = ssa.newReg(.b32);
            try b.addTyped(.mov, .b32, .{ .none = .{} }, &.{ field, stored });
        }
        if (stored_class.bits() < bits) {
            const widened = ssa.newReg(partition.class);
            const from: Mir.Type = if (stored_class.bits() == 16) .u16 else .u32;
            try b.addTyped(.cvt, if (bits == 32) .u32 else .u64, .{ .cvt = .{ .from = from } }, &.{ widened, field });
            field = widened;
        }
        if (size == 1) {
            const masked = ssa.newReg(partition.class);
            try b.addTyped(.@"and", word, .{ .none = .{} }, &.{ masked, field, immOperand(0xff) });
            field = masked;
        }
        if (shift != 0) {
            const moved = ssa.newReg(partition.class);
            try b.addTyped(.shl, word, .{ .none = .{} }, &.{ moved, field, immOperand(shift) });
            field = moved;
        }
        const field_mask = (@as(u64, 1) << @intCast(size * 8)) - 1;
        const keep_mask = ~(field_mask << shift) & (@as(u64, std.math.maxInt(u64)) >> @as(u6, @intCast(64 - bits)));
        const kept = ssa.newReg(partition.class);
        const keep: [3]u32 = .{ @bitCast(Mir.Operand{ .kind = .imm64, .payload = @backingInt(Mir.ImmFormat.unsigned) }), @truncate(keep_mask), @truncate(keep_mask >> 32) };
        try b.addTyped(.@"and", word, .{ .none = .{} }, &(.{ kept, old } ++ keep));
        try b.addTyped(.@"or", word, .{ .none = .{} }, &.{ new, kept, field });
    }

    /// Returns a `.b32` register with the bit of the window where element `e` of an `indexed`
    /// access starts. The address register of the access holds an offset in the local.
    fn bitPosition(ssa: *Ssa, b: *Mir.Builder, access: Access, e: usize) Allocator.Error!u32 {
        const p = ssa.p;
        const instruction = p.inst(access.inst);
        const ops = try p.operandsAt(instruction);
        const base = p.mir.extra[if (instruction.tag == .ld) ops[1] else ops[0]];
        var byte = base;
        if (@as(Mir.Operand, @bitCast(base)).regClass() == .b64) {
            byte = ssa.newReg(.b32);
            try b.addTyped(.cvt, .u32, .{ .cvt = .{ .from = .u64 } }, &.{ byte, base });
        }
        const offset: u32 = @truncate(@as(u64, @bitCast(access.offset + @as(i64, access.elem_size) * @as(i64, @intCast(e)))));
        if (offset != 0) {
            const sum = ssa.newReg(.b32);
            try b.addTyped(.add, .u32, .{ .arith = .{} }, &(.{ sum, byte } ++ imm64Operand(offset)));
            byte = sum;
        }
        const bit = ssa.newReg(.b32);
        try b.addTyped(.shl, .b32, .{ .none = .{} }, &.{ bit, byte, immOperand(3) });
        return bit;
    }

    /// Writes `result`, the `size` bytes at the bit in the register `position` of the window
    /// register `window`, as `extract` does for a known bit.
    fn extractAt(ssa: *Ssa, b: *Mir.Builder, partition: Partition, size: u8, position: u32, signed: bool, result: u32, stored: u32) Allocator.Error!void {
        const unsigned: Mir.Type = if (partition.size == 4) .u32 else .u64;
        const window = try ssa.argOf(b, stored, partition.class);
        const field = ssa.newReg(partition.class);
        try b.addTyped(.shr, unsigned, .{ .none = .{} }, &.{ field, window, position });
        const class = @as(Mir.Operand, @bitCast(result)).regClass();
        if (size == 4) {
            const low = if (class == .f32) ssa.newReg(.b32) else result;
            try b.addTyped(.cvt, .u32, .{ .cvt = .{ .from = unsigned } }, &.{ low, field });
            if (low != result) try b.addTyped(.mov, .b32, .{ .none = .{} }, &.{ result, low });
            return;
        }
        if (size == 2) return b.addTyped(.cvt, .u16, .{ .cvt = .{ .from = unsigned } }, &.{ result, field });
        const half = ssa.newReg(.b16);
        try b.addTyped(.cvt, .u16, .{ .cvt = .{ .from = unsigned } }, &.{ half, field });
        if (signed) return b.addTyped(.cvt, .s16, .{ .cvt = .{ .from = .s8 } }, &.{ result, half });
        try b.addTyped(.@"and", .b16, .{ .none = .{} }, &.{ result, half, immOperand(0xff) });
    }

    /// Writes `new`, the window register `old` with the low `size` bytes of `stored` at the bit in
    /// the register `position`, as `insert` does for a known bit.
    fn insertAt(ssa: *Ssa, b: *Mir.Builder, partition: Partition, size: u8, position: u32, new: u32, old: u32, stored: u32) Allocator.Error!void {
        const bits: u8 = partition.size * 8;
        const word: Mir.Type = if (bits == 32) .b32 else .b64;
        var field = stored;
        const stored_class = @as(Mir.Operand, @bitCast(stored)).regClass();
        if (stored_class == .f32) {
            field = ssa.newReg(.b32);
            try b.addTyped(.mov, .b32, .{ .none = .{} }, &.{ field, stored });
        }
        if (stored_class.bits() < bits) {
            const widened = ssa.newReg(partition.class);
            const from: Mir.Type = if (stored_class.bits() == 16) .u16 else .u32;
            try b.addTyped(.cvt, if (bits == 32) .u32 else .u64, .{ .cvt = .{ .from = from } }, &.{ widened, field });
            field = widened;
        }
        const field_mask = (@as(u64, 1) << @intCast(size * 8)) - 1;
        if (size == 1) {
            const masked = ssa.newReg(partition.class);
            try b.addTyped(.@"and", word, .{ .none = .{} }, &.{ masked, field, immOperand(0xff) });
            field = masked;
        }
        const moved = ssa.newReg(partition.class);
        try b.addTyped(.shl, word, .{ .none = .{} }, &.{ moved, field, position });
        const mask = ssa.newReg(partition.class);
        try b.addTyped(.mov, word, .{ .none = .{} }, &(.{mask} ++ imm64Operand(field_mask)));
        const shifted_mask = ssa.newReg(partition.class);
        try b.addTyped(.shl, word, .{ .none = .{} }, &.{ shifted_mask, mask, position });
        const keep = ssa.newReg(partition.class);
        try b.addTyped(.not, word, .{ .none = .{} }, &.{ keep, shifted_mask });
        const kept = ssa.newReg(partition.class);
        try b.addTyped(.@"and", word, .{ .none = .{} }, &.{ kept, old, keep });
        try b.addTyped(.@"or", word, .{ .none = .{} }, &.{ new, kept, moved });
    }
};

fn imm64Operand(value: u64) [3]u32 {
    return .{ @bitCast(Mir.Operand{ .kind = .imm64, .payload = @backingInt(Mir.ImmFormat.unsigned) }), @truncate(value), @truncate(value >> 32) };
}

fn listOperand(count: u28) u32 {
    return @bitCast(Mir.Operand{ .kind = .list, .payload = count });
}

fn immOperand(value: u28) u32 {
    return @bitCast(Mir.Operand{ .kind = .imm, .payload = value });
}

/// Returns the memory operand of an access to a local that stays in memory. The address of a kernel
/// parameter replaces a copy in `param_addressing`, and the parameter does not change while the
/// kernel runs.
fn accessOperand(mem: Mir.MemOperand, state: State) Mir.MemOperand {
    if (state != .param_addressing) return mem;
    var result = mem;
    result.space = .param;
    result.invariant = true;
    return result;
}

/// Loads or stores `size` bytes at `base + offset`. Assumes that `base` is a multiple of `unit`.
fn emitUnitAccess(b: *Mir.Builder, tag: Mir.Inst.Tag, value: u32, base: u32, offset: i64, mem: Mir.MemOperand, unit: u8, size: u8) Allocator.Error!void {
    const class = @as(Mir.Operand, @bitCast(value)).regClass();
    const ty: Mir.Type = switch (size) {
        1 => .u8,
        2 => .b16,
        4 => if (class == .f32) .f32 else .b32,
        else => if (class == .f64) .f64 else .b64,
    };
    const at: u64 = @intCast(offset);
    const alignment: u64 = if (at == 0) unit else @min(unit, at & -%at);
    const modifiers: Mir.Inst.Modifiers = .{ .mem = .{ .offset = true } };
    var access = mem.sized(size);
    access.log2_align = Mir.log2Align(alignment);
    const mem_word: u32 = @bitCast(access);
    const imm: [3]u32 = .{ @bitCast(Mir.Operand{ .kind = .imm64, .payload = @backingInt(Mir.ImmFormat.unsigned) }), @truncate(at), @truncate(at >> 32) };
    if (tag == .ld) return b.addTyped(.ld, ty, modifiers, &(.{ value, base } ++ imm ++ .{mem_word}));
    try b.addTyped(.st, ty, modifiers, &(.{base} ++ imm ++ .{ value, mem_word }));
}

fn memorySize(ty: Mir.Type) ?u8 {
    return switch (ty) {
        .b8, .u8, .s8 => 1,
        .b16, .u16, .s16, .f16, .bf16 => 2,
        .b32, .u32, .s32, .f32, .f16x2, .bf16x2 => 4,
        .b64, .u64, .s64, .f64 => 8,
        .none, .pred => null,
    };
}

fn isAddressWidth(ty: Mir.Type) bool {
    return switch (ty) {
        .b32, .u32, .s32, .b64, .u64, .s64 => true,
        else => false,
    };
}
