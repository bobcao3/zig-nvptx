//! This file runs the link-time code generation (LTCG) passes over the functions of the module.
//! A function inlines its own callees before another function inlines it. A function on a call
//! cycle never inlines another function on a call cycle. Release modes inline and then run
//! promotion, the folding of branches into `selp`, address space inference with the lowering of
//! atomics on `.local` memory, copy expansion, CSE, integer narrowing, `store_vectorize`, DCE and
//! the positive loop-exit guards on each kept function. Debug mode only commits the conversions to
//! generic addresses and expands the copies. Both modes then rewrite the constant operands that
//! ptxas compiles wrongly.

const std = @import("std");
const Allocator = std.mem.Allocator;

const Mir = @import("../../codegen/nvptx/Mir.zig");
const constant_operands = @import("../../codegen/nvptx/transforms/constant_operands.zig");
const cse = @import("../../codegen/nvptx/transforms/cse.zig");
const dce = @import("../../codegen/nvptx/transforms/dce.zig");
const expand = @import("../../codegen/nvptx/transforms/expand.zig");
const infer_addrspace = @import("../../codegen/nvptx/transforms/infer_addrspace.zig");
const inlining = @import("../../codegen/nvptx/transforms/inlining.zig");
const local_atomics = @import("../../codegen/nvptx/transforms/local_atomics.zig");
const narrow = @import("../../codegen/nvptx/transforms/narrow.zig");
const positive_guard = @import("../../codegen/nvptx/transforms/positive_guard.zig");
const promote = @import("../../codegen/nvptx/transforms/promote.zig");
const selp = @import("../../codegen/nvptx/transforms/selp.zig");
const store_vectorize = @import("../../codegen/nvptx/transforms/store_vectorize.zig");
const Linker = @import("../Nvptx.zig");
const tracy = @import("../../tracy.zig");

pub fn run(gpa: Allocator, mirs: *std.array_hash_map.Auto(Mir.Target, Mir), linker: *const Linker) !void {
    const tracy_trace = tracy.traceNamed(@src(), "NVPTX inlining");
    defer tracy_trace.end();
    tracy_trace.setValue(mirs.count());

    if (linker.base.comp.root_mod.optimize_mode == .debug) return;
    var graph: CallGraph = try .init(gpa, mirs, linker);
    defer graph.deinit();
    const traversal_trace = tracy.traceNamed(@src(), "NVPTX inline traversal");
    defer traversal_trace.end();
    for (0..mirs.count()) |index| try graph.visit(@intCast(index));
}

/// Returns the indices in `mirs` of the live functions that the module lists in a table of
/// function addresses: each function on a call cycle, each function whose address a live function
/// or a live global in `data` takes, and each function that these reach through calls. ptxas 13.0
/// and 13.4 crash for sm_90 and newer targets when they remove the last call to such a function
/// and the function is not in such a table.
pub fn addressTable(
    gpa: Allocator,
    mirs: *std.array_hash_map.Auto(Mir.Target, Mir),
    linker: *const Linker,
    live: []const bool,
    data: []const Mir,
) ![]u32 {
    var graph: CallGraph = try .init(gpa, mirs, linker);
    defer graph.deinit();
    var table: Table = .{ .graph = &graph, .listed = try .initEmpty(gpa, mirs.count()) };
    defer table.deinit();
    for (mirs.values(), live, 0..) |*mir, is_live, index| {
        if (!is_live) continue;
        if (graph.in_cycle.isSet(index)) try table.listFunction(@intCast(index));
        try table.listTaken(mir, 1);
    }
    for (data) |*mir| try table.listTaken(mir, 0);
    while (table.work.pop()) |index| {
        const callee_list = try graph.callees(gpa, index);
        defer gpa.free(callee_list);
        for (callee_list) |callee| try table.listFunction(callee);
    }
    var result: std.ArrayList(u32) = .empty;
    errdefer result.deinit(gpa);
    for (live, 0..) |is_live, index| {
        if (is_live and table.listed.isSet(index)) try result.append(gpa, @intCast(index));
    }
    return result.toOwnedSlice(gpa);
}

/// `Table` collects the functions that `addressTable` returns.
const Table = struct {
    graph: *const CallGraph,
    listed: std.DynamicBitSetUnmanaged,
    work: std.ArrayList(u32) = .empty,

    fn deinit(table: *Table) void {
        table.listed.deinit(table.graph.gpa);
        table.work.deinit(table.graph.gpa);
    }

    fn listFunction(table: *Table, index: u32) !void {
        if (table.graph.mirs.values()[index].kind != .func or table.listed.isSet(index)) return;
        table.listed.set(index);
        try table.work.append(table.graph.gpa, index);
    }

    /// Lists each function that `mir` names other than as the callee of a direct call. The first
    /// `skip` relocations of the text name `mir` itself.
    fn listTaken(table: *Table, mir: *const Mir, skip: usize) !void {
        for (0..mir.instructions.len) |i| {
            const inst = mir.instructions.get(i);
            if (inst.tag == .call and table.graph.directCallee(mir, inst.data) != null) continue;
            const range = mir.operands(inst) orelse continue;
            var at = range.first;
            for (0..range.count) |_| {
                const op: Mir.Operand = @bitCast(mir.extra[at]);
                if (op.kind == .symbol) try table.listTarget(mir.symbols[op.payload].target);
                at += op.len();
            }
        }
        for (mir.relocs[@min(skip, mir.relocs.len)..]) |reloc| try table.listTarget(reloc.target);
    }

    fn listTarget(table: *Table, target: Mir.Target) !void {
        const graph = table.graph;
        const index = graph.mirs.getIndex(graph.linker.resolve(target)) orelse return;
        try table.listFunction(@intCast(index));
    }
};

/// Runs the passes after inlining on one kept function. Debug mode commits each conversion to a
/// generic address where it stands, expands the copies and rewrites the constant operands. Release
/// modes run more passes:
/// - Inlining turns pointer arguments into copies, so promotion finds more variables to move to
///   registers.
/// - Promotion does not follow `selp`, so the folding of branches into `selp` runs after it.
/// - Address space inference runs after inlining, so a generic access in a callee finds the
///   conversion in its caller. Inference follows `selp`.
/// - The lowering of atomics on `.local` memory needs the state spaces from inference.
/// - Copy expansion runs after inference, so the loads and stores of a copy get its state spaces.
/// - CSE merges the conversions that inference adds and gives equal addresses one register, so
///   `store_vectorize` finds more stores to merge.
/// - Integer narrowing leaves 64-bit values that nothing reads, and DCE removes them.
/// - The rewrite of constant operands runs last, so every 64-bit addition and every division that
///   reaches ptxas passes through it.
pub fn optimize(gpa: Allocator, mir: *Mir, optimize_mode: std.lang.Optimize) expand.Error!void {
    if (optimize_mode == .debug) {
        try infer_addrspace.run(gpa, mir, false);
        try expand.run(gpa, mir);
        return constant_operands.run(gpa, mir);
    }
    {
        const tracy_trace = tracy.traceNamed(@src(), "NVPTX promotion after inlining");
        defer tracy_trace.end();
        tracy_trace.setValue(mir.instructions.len);
        try promote.run(gpa, mir, &.{});
    }
    {
        const tracy_trace = tracy.traceNamed(@src(), "NVPTX selp folding");
        defer tracy_trace.end();
        tracy_trace.setValue(mir.instructions.len);
        try selp.run(gpa, mir);
    }
    {
        const tracy_trace = tracy.traceNamed(@src(), "NVPTX address space inference");
        defer tracy_trace.end();
        tracy_trace.setValue(mir.instructions.len);
        try infer_addrspace.run(gpa, mir, true);
        try local_atomics.run(gpa, mir);
    }
    {
        const tracy_trace = tracy.traceNamed(@src(), "NVPTX copy expansion");
        defer tracy_trace.end();
        tracy_trace.setValue(mir.instructions.len);
        try expand.run(gpa, mir);
    }
    {
        const tracy_trace = tracy.traceNamed(@src(), "NVPTX CSE after inlining");
        defer tracy_trace.end();
        tracy_trace.setValue(mir.instructions.len);
        try cse.run(gpa, mir);
    }
    {
        const tracy_trace = tracy.traceNamed(@src(), "NVPTX integer narrowing");
        defer tracy_trace.end();
        tracy_trace.setValue(mir.instructions.len);
        try narrow.run(gpa, mir);
    }
    {
        const tracy_trace = tracy.traceNamed(@src(), "NVPTX store vectorization");
        defer tracy_trace.end();
        tracy_trace.setValue(mir.instructions.len);
        try store_vectorize.run(gpa, mir);
    }
    {
        const tracy_trace = tracy.traceNamed(@src(), "NVPTX DCE after inlining");
        defer tracy_trace.end();
        tracy_trace.setValue(mir.instructions.len);
        try dce.run(gpa, mir);
    }
    {
        const tracy_trace = tracy.traceNamed(@src(), "NVPTX positive guards");
        defer tracy_trace.end();
        tracy_trace.setValue(mir.instructions.len);
        try positive_guard.run(gpa, mir);
    }
    const tracy_trace = tracy.traceNamed(@src(), "NVPTX constant operands");
    defer tracy_trace.end();
    tracy_trace.setValue(mir.instructions.len);
    try constant_operands.run(gpa, mir);
}

/// `CallGraph` holds the direct calls between the functions of the module.
const CallGraph = struct {
    gpa: Allocator,
    mirs: *std.array_hash_map.Auto(Mir.Target, Mir),
    linker: *const Linker,
    in_cycle: std.DynamicBitSetUnmanaged,
    done: std.DynamicBitSetUnmanaged,

    fn init(gpa: Allocator, mirs: *std.array_hash_map.Auto(Mir.Target, Mir), linker: *const Linker) !CallGraph {
        var graph: CallGraph = .{ .gpa = gpa, .mirs = mirs, .linker = linker, .in_cycle = .{}, .done = .{} };
        errdefer graph.deinit();
        graph.in_cycle = try .initEmpty(gpa, mirs.count());
        graph.done = try .initEmpty(gpa, mirs.count());
        try graph.findCycles();
        return graph;
    }

    fn deinit(graph: *CallGraph) void {
        graph.in_cycle.deinit(graph.gpa);
        graph.done.deinit(graph.gpa);
    }

    fn directCallee(graph: *const CallGraph, mir: *const Mir, data: u32) ?u32 {
        const call: Mir.Call = @bitCast(mir.extra[data]);
        if (call.indirect) return null;
        const op: Mir.Operand = @bitCast(mir.extra[data + 1]);
        const sym = mir.symbols[op.payload];
        if (sym.offset != 0) return null;
        const index = graph.mirs.getIndex(graph.linker.resolve(sym.target)) orelse return null;
        return @intCast(index);
    }

    fn callees(graph: *const CallGraph, gpa: Allocator, index: u32) ![]u32 {
        const mir = &graph.mirs.values()[index];
        var list: std.ArrayList(u32) = .empty;
        errdefer list.deinit(gpa);
        for (mir.instructions.items(.tag), mir.instructions.items(.data)) |tag, data| {
            if (tag != .call) continue;
            if (graph.directCallee(mir, data)) |callee| try list.append(gpa, callee);
        }
        return list.toOwnedSlice(gpa);
    }

    fn findCycles(graph: *CallGraph) !void {
        const tracy_trace = tracy.traceNamed(@src(), "NVPTX cycle detection");
        defer tracy_trace.end();

        const gpa = graph.gpa;
        const n = graph.mirs.count();
        tracy_trace.setValue(n);
        var arena_state: std.heap.ArenaAllocator = .init(gpa);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        const edges = try arena.alloc([]u32, n);
        {
            const graph_trace = tracy.traceNamed(@src(), "NVPTX call graph");
            defer graph_trace.end();
            for (edges, 0..) |*e, i| e.* = try graph.callees(arena, @intCast(i));
        }
        const order = try arena.alloc(i32, n);
        const low = try arena.alloc(u32, n);
        const on_stack = try arena.alloc(bool, n);
        @memset(order, -1);
        @memset(on_stack, false);
        var stack: std.ArrayList(u32) = .empty;
        var counter: u32 = 0;
        const Frame = struct { node: u32, edge: u32 };
        var frames: std.ArrayList(Frame) = .empty;
        for (0..n) |root| {
            if (order[root] != -1) continue;
            try frames.append(arena, .{ .node = @intCast(root), .edge = 0 });
            order[root] = @intCast(counter);
            low[root] = counter;
            counter += 1;
            try stack.append(arena, @intCast(root));
            on_stack[root] = true;
            while (frames.items.len != 0) {
                const frame = &frames.items[frames.items.len - 1];
                const v = frame.node;
                if (frame.edge < edges[v].len) {
                    const w = edges[v][frame.edge];
                    frame.edge += 1;
                    if (w == v) {
                        graph.in_cycle.set(v);
                    } else if (order[w] == -1) {
                        order[w] = @intCast(counter);
                        low[w] = counter;
                        counter += 1;
                        try stack.append(arena, w);
                        on_stack[w] = true;
                        try frames.append(arena, .{ .node = w, .edge = 0 });
                    } else if (on_stack[w]) {
                        low[v] = @min(low[v], @as(u32, @intCast(order[w])));
                    }
                    continue;
                }
                _ = frames.pop();
                if (frames.items.len != 0) {
                    const parent = frames.items[frames.items.len - 1].node;
                    low[parent] = @min(low[parent], low[v]);
                }
                if (low[v] != @as(u32, @intCast(order[v]))) continue;
                const start = std.mem.lastIndexOfScalar(u32, stack.items, v).?;
                const component = stack.items[start..];
                for (component) |w| on_stack[w] = false;
                if (component.len > 1) for (component) |w| graph.in_cycle.set(w);
                stack.shrinkRetainingCapacity(start);
            }
        }
    }

    fn visit(graph: *CallGraph, index: u32) !void {
        if (graph.done.isSet(index)) return;
        graph.done.set(index);
        const callee_list = try graph.callees(graph.gpa, index);
        defer graph.gpa.free(callee_list);
        for (callee_list) |callee| {
            if (!graph.in_cycle.isSet(callee)) try graph.visit(callee);
        }
        const tracy_trace = tracy.traceNamed(@src(), "NVPTX inline calls");
        defer tracy_trace.end();
        tracy_trace.addTextFmt("function_index={d}", .{index});
        try inlining.run(graph.gpa, &graph.mirs.values()[index], Sites{ .graph = graph, .caller = index });
    }

    /// `Sites` selects the callees that one caller inlines.
    const Sites = struct {
        graph: *const CallGraph,
        caller: u32,

        pub fn callee(sites: Sites, mir: *const Mir, data: u32) ?*const Mir {
            const call: Mir.Call = @bitCast(mir.extra[data]);
            if (call.never_inline) return null;
            const index = sites.graph.directCallee(mir, data) orelse return null;
            if (sites.graph.in_cycle.isSet(sites.caller) and sites.graph.in_cycle.isSet(index)) return null;
            const body = &sites.graph.mirs.values()[index];
            return if (inlining.isInlinable(body)) body else null;
        }
    };
};
