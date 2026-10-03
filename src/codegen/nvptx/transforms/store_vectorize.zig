//! This file merges adjacent stores into `st.v2` and `st.v4`. ptxas 13.0 does not merge stores
//! through a pointer of unknown alignment. A pointer register in PTX carries no alignment, so the
//! `Mir.MemOperand` of each `st` records the alignment that the Zig types give its address. A
//! group holds stores of one size, space and base register at different offsets. PTX requires each
//! address to be a multiple of the access size (PTX ISA 9.4, section 6.4.1), so these stores never
//! overlap.
//! The merged store takes the position of the last store of its group. Before `Lower.run`, each
//! register has one definition, so each stored value is still in its register there. A group
//! therefore ends at each label and at each instruction that can access memory or branch.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Mir = @import("../Mir.zig");

/// This pass merges at most 16 bytes into one `st` (PTX ISA 9.4, section 5.4.2).
const max_vector_bytes = 16;

const Address = struct {
    root: u32,
    offset: u64,
};

const Store = struct {
    index: u32,
    offset: u64,
    value: u32,
    ty: Mir.Type,
    alignment: u64,
};

const GroupKey = struct {
    root: u32,
    space: Mir.Space,
    size: u64,
};

const StoreVectorizer = struct {
    arena: Allocator,
    mir: *const Mir,
    /// Maps each register that an unguarded `add` of an immediate or `mov` defines to its address.
    addresses: std.AutoHashMapUnmanaged(u32, Address) = .empty,
    group: std.ArrayList(Store) = .empty,
    key: ?GroupKey = null,
    removed: std.AutoHashMapUnmanaged(u32, void) = .empty,
    merged: std.AutoHashMapUnmanaged(u32, MergedStore) = .empty,

    const MergedStore = struct {
        ty: Mir.Type,
        modifiers: Mir.Inst.Modifiers,
        words: []const u32,
    };

    fn resolve(vectorizer: *const StoreVectorizer, word: u32) Address {
        return vectorizer.addresses.get(word) orelse .{ .root = word, .offset = 0 };
    }

    fn recordAddress(vectorizer: *StoreVectorizer, inst: Mir.Inst) Allocator.Error!void {
        // `add` and `mov` define addresses. `ret`, `call_begin` and `call_end` have no operands.
        if (inst.tag != .add and inst.tag != .mov) return;
        const ops = vectorizer.mir.extra[inst.data..];
        switch (inst.tag) {
            .add => {
                if (inst.ty != .u64 and inst.ty != .s64) return;
                if (inst.modifiers.arith.guard != .none or inst.modifiers.arith.cc) return;
                const base: Mir.Operand = @bitCast(ops[1]);
                const imm: Mir.Operand = @bitCast(ops[2]);
                if (base.kind != .reg or imm.kind != .imm) return;
                const address = vectorizer.resolve(ops[1]);
                try vectorizer.addresses.put(vectorizer.arena, ops[0], .{
                    .root = address.root,
                    .offset = address.offset + imm.payload,
                });
            },
            .mov => {
                if (inst.modifiers.none.guard != .none) return;
                const dst: Mir.Operand = @bitCast(ops[0]);
                const src: Mir.Operand = @bitCast(ops[1]);
                if (dst.kind != .reg or src.kind != .reg) return;
                if (dst.regClass() != .b64 or src.regClass() != .b64) return;
                try vectorizer.addresses.put(vectorizer.arena, ops[0], vectorizer.resolve(ops[1]));
            },
            else => unreachable,
        }
    }

    /// Returns the group key and the store of a `st` that this pass can merge.
    fn plainStore(
        vectorizer: *const StoreVectorizer,
        index: u32,
        inst: Mir.Inst,
    ) ?struct { GroupKey, Store } {
        if (inst.tag != .st) return null;
        const form = inst.modifiers.mem;
        if (form.guard != .none or form.vec != .scalar or !form.offset) return null;
        if (vectorizer.mir.hasOrderedMemoryRef(inst)) return null;
        const mem = vectorizer.mir.memOperand(inst, 0);
        const ops = vectorizer.mir.extra[inst.data..];
        const base: Mir.Operand = @bitCast(ops[0]);
        const imm: Mir.Operand = @bitCast(ops[1]);
        const value: Mir.Operand = @bitCast(ops[2]);
        if (base.kind != .reg or imm.kind != .imm) return null;
        if (value.kind != .reg or value.regClass() == .pred) return null;
        const address = vectorizer.resolve(ops[0]);
        return .{
            .{ .root = address.root, .space = mem.space, .size = mem.size() },
            .{
                .index = index,
                .offset = address.offset + imm.payload,
                .value = ops[2],
                .ty = inst.ty,
                .alignment = mem.alignment(),
            },
        };
    }

    fn add(vectorizer: *StoreVectorizer, key: GroupKey, store: Store) Allocator.Error!void {
        const same = if (vectorizer.key) |current| std.meta.eql(current, key) else false;
        const repeated = for (vectorizer.group.items) |member| {
            if (member.offset == store.offset) break true;
        } else false;
        if (!same or repeated) try vectorizer.flush();
        vectorizer.key = key;
        try vectorizer.group.append(vectorizer.arena, store);
    }

    fn flush(vectorizer: *StoreVectorizer) Allocator.Error!void {
        defer {
            vectorizer.group.clearRetainingCapacity();
            vectorizer.key = null;
        }
        const key = vectorizer.key orelse return;
        const stores = vectorizer.group.items;
        if (stores.len < 2) return;
        std.mem.sort(Store, stores, {}, lessThan);
        var k: usize = 0;
        while (k < stores.len) {
            const len = vectorLen(stores[k..], key.size);
            if (len > 1) try vectorizer.merge(key, stores[k..][0..len]);
            k += len;
        }
    }

    fn merge(
        vectorizer: *StoreVectorizer,
        key: GroupKey,
        stores: []const Store,
    ) Allocator.Error!void {
        const first = stores[0];
        if (first.offset > std.math.maxInt(u28)) return;
        var last = first.index;
        var ty = first.ty;
        for (stores) |store| {
            last = @max(last, store.index);
            if (store.ty != first.ty) ty = bitType(key.size);
        }
        const words = try vectorizer.arena.alloc(u32, 4 + stores.len);
        words[0] = key.root;
        words[1] = @bitCast(Mir.Operand{ .kind = .imm, .payload = @intCast(first.offset) });
        words[2] = @bitCast(Mir.Operand{ .kind = .list, .payload = @intCast(stores.len) });
        for (words[3..][0..stores.len], stores) |*word, store| word.* = store.value;
        const mem: Mir.MemOperand = .{
            .space = key.space,
            .log2_size = @intCast(std.math.log2_int(u64, key.size * stores.len)),
            .log2_align = Mir.log2Align(first.alignment),
        };
        words[3 + stores.len] = @bitCast(mem);
        const modifiers: Mir.Inst.Modifiers = .{ .mem = .{
            .vec = if (stores.len == 4) .v4 else .v2,
            .offset = true,
        } };
        try vectorizer.merged.put(vectorizer.arena, last, .{
            .ty = ty,
            .modifiers = modifiers,
            .words = words,
        });
        for (stores) |store| {
            if (store.index != last) try vectorizer.removed.put(vectorizer.arena, store.index, {});
        }
    }
};

pub fn run(gpa: Allocator, mir: *Mir) Allocator.Error!void {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    var vectorizer: StoreVectorizer = .{ .arena = arena_state.allocator(), .mir = mir };
    for (0..mir.instructions.len) |i| {
        const inst = mir.instructions.get(i);
        try vectorizer.recordAddress(inst);
        if (vectorizer.plainStore(@intCast(i), inst)) |key_store| {
            try vectorizer.add(key_store[0], key_store[1]);
        } else if (endsGroup(mir, inst)) {
            try vectorizer.flush();
        }
    }
    try vectorizer.flush();
    if (vectorizer.merged.count() == 0) return;

    var b: Mir.Builder = .{
        .gpa = gpa,
        .reg_counts = mir.reg_counts,
        .label_count = mir.label_count,
    };
    defer b.deinit();
    for (0..mir.instructions.len) |i| {
        if (vectorizer.removed.contains(@intCast(i))) continue;
        if (vectorizer.merged.get(@intCast(i))) |merged| {
            try b.addTyped(.st, merged.ty, merged.modifiers, merged.words);
            continue;
        }
        const inst = mir.instructions.get(i);
        const at = b.here();
        try b.extra.appendSlice(gpa, mir.span(inst));
        try b.instructions.append(gpa, .{
            .tag = inst.tag,
            .ty = inst.ty,
            .modifiers = inst.modifiers,
            .data = at,
        });
    }
    var instructions = b.instructions;
    b.instructions = .empty;
    errdefer instructions.deinit(gpa);
    const extra = try b.extra.toOwnedSlice(gpa);
    mir.instructions.deinit(gpa);
    gpa.free(mir.extra);
    mir.instructions = instructions.toOwnedSlice();
    mir.extra = extra;
}

fn endsGroup(mir: *const Mir, inst: Mir.Inst) bool {
    return switch (inst.tag) {
        .label, .bra, .ret, .trap, .brkpt => true,
        .call_begin, .param_decl, .call_end, .callprototype => true,
        else => Mir.mayLoad(inst) or Mir.mayStore(inst) or mir.hasOrderedMemoryRef(inst),
    };
}

fn lessThan(_: void, a: Store, b: Store) bool {
    return a.offset < b.offset;
}

/// Returns 4, 2 or 1, the number of stores at the start of the sorted `stores` that one `st` holds.
fn vectorLen(stores: []const Store, size: u64) usize {
    var len: usize = 4;
    while (len > 1) : (len /= 2) {
        if (len > stores.len or len * size > max_vector_bytes) continue;
        if (stores[0].alignment % (len * size) != 0) continue;
        const contiguous = for (stores[1..len], 1..) |store, i| {
            if (store.offset != stores[0].offset + i * size) break false;
        } else true;
        if (contiguous) return len;
    }
    return 1;
}

fn bitType(size: u64) Mir.Type {
    return switch (size) {
        1 => .b8,
        2 => .b16,
        4 => .b32,
        else => .b64,
    };
}
