//! This file inlines calls. The pass replaces the call sequence with the body of the callee. Each
//! parameter load of the callee becomes a copy of an argument. Each `ret` becomes a branch to a
//! continuation block. The parameters of the continuation block take the results.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Mir = @import("../Mir.zig");

/// Returns false for a callee that is not a function, is `noinline`, has no instructions or takes
/// the address of one of its parameters.
pub fn isInlinable(callee: *const Mir) bool {
    if (callee.kind != .func or callee.is_noinline or callee.instructions.len == 0) return false;
    for (0..callee.instructions.len) |i| {
        const inst = callee.instructions.get(i);
        switch (inst.tag) {
            .call, .param_decl, .@"asm" => continue,
            else => {},
        }
        const range = callee.operands(inst) orelse continue;
        var at = range.first;
        for (0..range.count) |_| {
            const op: Mir.Operand = @bitCast(callee.extra[at]);
            if (op.kind == .param) return false;
            at += op.len();
        }
    }
    return true;
}

const Offsets = struct {
    regs: [Mir.class_count]u32,
    labels: u32,
    locals: u32,
    symbols: u32,
    strings: u32,
};

fn copyInst(b: *Mir.Builder, src: *const Mir, index: usize, offsets: Offsets) !void {
    const inst = src.instructions.get(index);
    const tag = inst.tag;
    const data = inst.data;
    switch (tag) {
        .param_load, .param_store => {
            const access: Mir.ParamAccess = @bitCast(src.extra[data]);
            const start = b.here();
            try b.extra.appendSlice(b.gpa, src.extra[data..][0..2]);
            for (src.extra[data + 2 ..][0..access.len]) |word| try b.extra.append(b.gpa, moveReg(word, offsets));
            try b.add(tag, start);
        },
        .param_decl => {
            const start = b.here();
            _ = try copyOperand(b, src, data, offsets);
            try b.add(tag, start);
        },
        .call => {
            const call: Mir.Call = @bitCast(src.extra[data]);
            const start = b.here();
            try b.extra.append(b.gpa, src.extra[data]);
            _ = try copyOperand(b, src, data + 1, offsets);
            if (call.indirect) try b.extra.append(b.gpa, src.extra[data + 2] + offsets.labels);
            try b.add(tag, start);
        },
        .@"asm" => {
            const start = b.here();
            const count = src.extra[data];
            try b.extra.append(b.gpa, count);
            var at = data + 1;
            for (0..count) |_| at = try copyOperand(b, src, at, offsets);
            try b.instructions.append(b.gpa, .{ .tag = tag, .modifiers = inst.modifiers, .data = start });
        },
        .call_begin, .call_end, .ret => try b.add(tag, data),
        else => {
            const range = src.operands(inst).?;
            const start = b.here();
            var at = range.first;
            for (0..range.count) |_| at = try copyOperand(b, src, at, offsets);
            try b.extra.appendSlice(b.gpa, src.extra[at..][0..Mir.memOperandCount(inst)]);
            try b.instructions.append(b.gpa, .{ .tag = tag, .ty = inst.ty, .modifiers = inst.modifiers, .data = start });
        },
    }
}

fn moveReg(word: u32, offsets: Offsets) u32 {
    const op: Mir.Operand = @bitCast(word);
    const class = op.regClass();
    return @bitCast(Mir.Operand.reg(class, @intCast(op.regIndex() + offsets.regs[@backingInt(class)])));
}

fn copyOperand(b: *Mir.Builder, src: *const Mir, at: u32, offsets: Offsets) !u32 {
    const op: Mir.Operand = @bitCast(src.extra[at]);
    switch (op.kind) {
        .reg => try b.extra.append(b.gpa, moveReg(src.extra[at], offsets)),
        .label => try b.extra.append(b.gpa, @bitCast(Mir.Operand{ .kind = .label, .payload = @intCast(op.payload + offsets.labels) })),
        .local => try b.extra.append(b.gpa, @bitCast(Mir.Operand{ .kind = .local, .payload = @intCast(op.payload + offsets.locals) })),
        .symbol => try b.extra.append(b.gpa, @bitCast(Mir.Operand{ .kind = .symbol, .payload = @intCast(op.payload + offsets.symbols) })),
        .string => try b.extra.appendSlice(b.gpa, &.{
            @bitCast(Mir.Operand{ .kind = .string, .payload = @intCast(op.payload + offsets.strings) }),
            src.extra[at + 1],
        }),
        .list => {
            try b.extra.append(b.gpa, src.extra[at]);
            for (src.extra[at + 1 ..][0..op.payload]) |word| try b.extra.append(b.gpa, moveReg(word, offsets));
        },
        .param, .imm, .imm64, .special_reg => try b.extra.appendSlice(b.gpa, src.extra[at..][0..op.len()]),
    }
    return at + op.len();
}

fn addMove(b: *Mir.Builder, mem: Mir.MemType, dst: u32, src: u32) !void {
    switch (mem) {
        .u8 => try b.addTyped(.cvt, .u16, .{ .cvt = .{ .from = .u8 } }, &.{ dst, src }),
        .s8 => try b.addTyped(.cvt, .s16, .{ .cvt = .{ .from = .s8 } }, &.{ dst, src }),
        else => try b.addTyped(.mov, switch (@as(Mir.Operand, @bitCast(dst)).regClass()) {
            .pred => .pred,
            .b16 => .b16,
            .b32 => .b32,
            .b64 => .b64,
            .f32 => .f32,
            .f64 => .f64,
        }, .{ .none = .{} }, &.{ dst, src }),
    }
}

/// Returns a new register of the class of `like`. The register has no definition.
fn newReg(b: *Mir.Builder, like: u32) u32 {
    const class = @as(Mir.Operand, @bitCast(like)).regClass();
    defer b.reg_counts[@backingInt(class)] += 1;
    return @bitCast(Mir.Operand.reg(class, @intCast(b.reg_counts[@backingInt(class)])));
}

fn memSize(mem: Mir.MemType) u32 {
    return switch (mem) {
        .u8, .s8 => 1,
        .b16 => 2,
        .b32, .f32 => 4,
        .b64, .f64 => 8,
    };
}

const Element = struct { param: u16, offset: u32 };
const Value = struct { mem: Mir.MemType, reg: u32 };

fn collect(map: *std.AutoHashMapUnmanaged(Element, Value), gpa: Allocator, mir: *const Mir, data: u32) !void {
    const access: Mir.ParamAccess = @bitCast(mir.extra[data]);
    const param: Mir.Operand = @bitCast(mir.extra[data + 1]);
    for (mir.extra[data + 2 ..][0..access.len], 0..) |word, i| {
        try map.put(gpa, .{
            .param = param.paramIndex(),
            .offset = access.offset + @as(u32, @intCast(i)) * memSize(access.mem),
        }, .{ .mem = access.mem, .reg = word });
    }
}

/// `sites.callee(mir, data)` returns the callee to inline at the `call` with the data word `data`.
/// A null result keeps the call.
pub fn run(gpa: Allocator, mir: *Mir, sites: anytype) Allocator.Error!void {
    const tags = mir.instructions.items(.tag);
    const datas = mir.instructions.items(.data);
    const any = for (tags, datas) |tag, data| {
        if (tag == .call and sites.callee(mir, data) != null) break true;
    } else false;
    if (!any) return;

    var b: Mir.Builder = .{ .gpa = gpa, .reg_counts = mir.reg_counts, .label_count = mir.label_count };
    defer b.deinit();
    try b.strings.appendSlice(gpa, mir.strings);
    try b.symbols.appendSlice(gpa, mir.symbols);
    try b.locals.appendSlice(gpa, mir.locals);
    const own: Offsets = .{ .regs = @splat(0), .labels = 0, .locals = 0, .symbols = 0, .strings = 0 };
    var ptx_version = mir.ptx_version;

    var i: usize = 0;
    while (i < tags.len) : (i += 1) {
        if (tags[i] != .call_begin) {
            try copyInst(&b, mir, i, own);
            continue;
        }
        var end = i;
        var call_at: ?usize = null;
        while (tags[end] != .call_end) : (end += 1) {
            if (tags[end] == .call) call_at = end;
        }
        const callee = sites.callee(mir, datas[call_at.?]);
        if (callee == null or !try splice(gpa, &b, mir, i, end, call_at.?, callee.?)) {
            for (i..end + 1) |k| try copyInst(&b, mir, k, own);
        } else {
            ptx_version = @max(ptx_version, callee.?.ptx_version);
        }
        i = end;
    }

    var instructions = b.instructions;
    b.instructions = .empty;
    errdefer instructions.deinit(gpa);
    const extra = try b.extra.toOwnedSlice(gpa);
    errdefer gpa.free(extra);
    const strings = try b.strings.toOwnedSlice(gpa);
    errdefer gpa.free(strings);
    const symbols = try b.symbols.toOwnedSlice(gpa);
    errdefer gpa.free(symbols);
    const locals = try b.locals.toOwnedSlice(gpa);
    const new: Mir = .{
        .kind = mir.kind,
        .text = mir.text,
        .prototype_len = mir.prototype_len,
        .relocs = mir.relocs,
        .ptx_version = ptx_version,
        .is_noinline = mir.is_noinline,
        .instructions = instructions.toOwnedSlice(),
        .extra = extra,
        .strings = strings,
        .symbols = symbols,
        .locals = locals,
        .reg_counts = b.reg_counts,
        .label_count = b.label_count,
    };
    mir.instructions.deinit(gpa);
    gpa.free(mir.extra);
    gpa.free(mir.strings);
    gpa.free(mir.symbols);
    gpa.free(mir.locals);
    mir.* = new;
}

/// Returns false and writes nothing when the parameters of the call and of the callee do not
/// match element by element.
fn splice(gpa: Allocator, b: *Mir.Builder, caller: *const Mir, begin: usize, end: usize, call_at: usize, callee: *const Mir) !bool {
    const tags = caller.instructions.items(.tag)[begin..end];
    const datas = caller.instructions.items(.data)[begin..end];
    var args: std.AutoHashMapUnmanaged(Element, Value) = .empty;
    defer args.deinit(gpa);
    var results: std.AutoHashMapUnmanaged(Element, Value) = .empty;
    defer results.deinit(gpa);
    for (tags, datas) |tag, data| switch (tag) {
        .param_store => try collect(&args, gpa, caller, data),
        .param_load => try collect(&results, gpa, caller, data),
        else => {},
    };
    var loads: std.AutoHashMapUnmanaged(Element, Value) = .empty;
    defer loads.deinit(gpa);
    for (callee.instructions.items(.tag), callee.instructions.items(.data)) |tag, data| {
        if (tag != .param_load) continue;
        const param: Mir.Operand = @bitCast(callee.extra[data + 1]);
        if (param.paramRef() != .own) continue;
        try collect(&loads, gpa, callee, data);
    }
    for (b.reg_counts, callee.reg_counts) |have, add| {
        if (have + add > std.math.maxInt(u16)) return false;
    }
    var it = loads.iterator();
    while (it.next()) |entry| {
        const arg = args.get(entry.key_ptr.*) orelse return false;
        if (memSize(arg.mem) != memSize(entry.value_ptr.mem)) return false;
    }

    const offsets: Offsets = .{
        .regs = b.reg_counts,
        .labels = b.label_count,
        .locals = @intCast(b.locals.items.len),
        .symbols = @intCast(b.symbols.items.len),
        .strings = @intCast(b.strings.items.len),
    };
    for (&b.reg_counts, callee.reg_counts) |*count, add| count.* += add;
    b.label_count += callee.label_count + 1;
    const end_label = b.label_count - 1;
    try b.locals.appendSlice(gpa, callee.locals);
    try b.symbols.appendSlice(gpa, callee.symbols);
    try b.strings.appendSlice(gpa, callee.strings);

    const own: Offsets = .{ .regs = @splat(0), .labels = 0, .locals = 0, .symbols = 0, .strings = 0 };
    for (begin + 1..call_at) |k| switch (tags[k - begin]) {
        .param_decl, .param_store => {},
        else => try copyInst(b, caller, k, own),
    };

    // The parameters of the continuation block follow the order of the result loads in the caller.
    var result_elements: std.ArrayList(Element) = .empty;
    defer result_elements.deinit(gpa);
    var result_regs: std.ArrayList(u32) = .empty;
    defer result_regs.deinit(gpa);
    for (tags, datas) |tag, data| {
        if (tag != .param_load) continue;
        const access: Mir.ParamAccess = @bitCast(caller.extra[data]);
        for (caller.extra[data + 2 ..][0..access.len], 0..) |word, e| {
            try result_elements.append(gpa, .{ .param = 0, .offset = access.offset + @as(u32, @intCast(e)) * memSize(access.mem) });
            try result_regs.append(gpa, word);
        }
    }
    // `stored` maps each result element to its value in the current block of the callee.
    // `Select.airRet` stores the result right before each `ret` and skips undefined elements.
    var stored: std.AutoHashMapUnmanaged(Element, u32) = .empty;
    defer stored.deinit(gpa);
    var ret_args: std.ArrayList(u32) = .empty;
    defer ret_args.deinit(gpa);

    const ctags = callee.instructions.items(.tag);
    const cdatas = callee.instructions.items(.data);
    for (ctags, cdatas, 0..) |tag, data, k| switch (tag) {
        .param_load => {
            const param: Mir.Operand = @bitCast(callee.extra[data + 1]);
            if (param.paramRef() != .own) {
                try copyInst(b, callee, k, offsets);
                continue;
            }
            const access: Mir.ParamAccess = @bitCast(callee.extra[data]);
            for (callee.extra[data + 2 ..][0..access.len], 0..) |word, e| {
                const element: Element = .{ .param = param.paramIndex(), .offset = access.offset + @as(u32, @intCast(e)) * memSize(access.mem) };
                try addMove(b, access.mem, moveReg(word, offsets), args.get(element).?.reg);
            }
        },
        .param_store => {
            const param: Mir.Operand = @bitCast(callee.extra[data + 1]);
            if (param.paramRef() != .own_ret) {
                try copyInst(b, callee, k, offsets);
                continue;
            }
            const access: Mir.ParamAccess = @bitCast(callee.extra[data]);
            for (callee.extra[data + 2 ..][0..access.len], 0..) |word, e| {
                const element: Element = .{ .param = 0, .offset = access.offset + @as(u32, @intCast(e)) * memSize(access.mem) };
                const result = results.get(element) orelse continue;
                var value = moveReg(word, offsets);
                // A 1-byte `st.param` keeps only the low byte (PTX ISA 9.4, section 9.4.1).
                if (result.mem == .u8 or result.mem == .s8) {
                    const narrowed = newReg(b, result.reg);
                    try addMove(b, result.mem, narrowed, value);
                    value = narrowed;
                }
                try stored.put(gpa, element, value);
            }
        },
        .ret => {
            ret_args.clearRetainingCapacity();
            for (result_elements.items, result_regs.items) |element, reg| {
                try ret_args.append(gpa, stored.get(element) orelse newReg(b, reg));
            }
            try b.addBranch(end_label, ret_args.items);
            stored.clearRetainingCapacity();
        },
        .label => {
            stored.clearRetainingCapacity();
            try copyInst(b, callee, k, offsets);
        },
        else => try copyInst(b, callee, k, offsets),
    };
    // A callee can end with an instruction that passes control to the next instruction. The
    // continuation block has parameters, so a branch replaces the fall-through.
    const last = callee.instructions.len - 1;
    if (ctags[last] != .ret and ctags[last] != .trap and !(ctags[last] == .bra and callee.instructions.items(.modifiers)[last].guard() == .none)) {
        ret_args.clearRetainingCapacity();
        for (result_regs.items) |reg| try ret_args.append(gpa, newReg(b, reg));
        try b.addBranch(end_label, ret_args.items);
    }
    try b.addLabel(end_label, result_regs.items);

    for (call_at + 1..end) |k| switch (caller.instructions.items(.tag)[k]) {
        .param_load => {},
        else => try copyInst(b, caller, k, own),
    };
    return true;
}
