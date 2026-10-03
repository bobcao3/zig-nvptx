const std = @import("std");
const Allocator = std.mem.Allocator;
const build_options = @import("build_options");
const native_endian = @import("builtin").cpu.arch.endian();
const Path = std.Build.Cache.Path;

const Zcu = @import("../Zcu.zig");
const InternPool = @import("../InternPool.zig");
const Compilation = @import("../Compilation.zig");
const link = @import("../link.zig");
const codegen = @import("../codegen.zig");
const CodeGen = @import("../codegen/nvptx/CodeGen.zig");
const Mir = @import("../codegen/nvptx/Mir.zig");
const ltcg = @import("Nvptx/ltcg.zig");
const Object = @import("Nvptx/Object.zig");
const verify = @import("../codegen/nvptx/verify.zig");
const tracy = @import("../tracy.zig");
const Linker = @This();

base: link.File,
/// A Zig function has a `.nav` key. A function from an object input has a `.name` key.
mirs: std.array_hash_map.Auto(Mir.Target, Mir),
externs: std.array_hash_map.Auto(InternPool.Nav.Index, Mir),
/// The value is the first export name of the nav. `flush` writes the later names as aliases.
exports: std.array_hash_map.Auto(InternPool.Nav.Index, InternPool.NullTerminatedString),
export_navs: std.array_hash_map.Auto(InternPool.NullTerminatedString, InternPool.Nav.Index),
inputs: std.ArrayList(Input),
/// This list holds the names of the object inputs. Each `.name` target indexes this list.
names: std.ArrayList(Name),
shared_names: std.StringArrayHashMapUnmanaged(u32),
/// This map holds the variables of the object inputs. The key is the index of the name.
input_data: std.array_hash_map.Auto(u32, Object.Item),
input_externs: std.array_hash_map.Auto(u32, Mir),
input_aliases: std.array_hash_map.Auto(u32, u32),
/// The linker keeps each kernel and each visible item of an object input. The linker keeps an item
/// of a static library only when kept code uses the item.
input_roots: std.ArrayList(Mir.Target),

const Input = struct {
    path: Path,
    is_archive: bool,
};

const Name = struct {
    bytes: []const u8,
    /// A name with this field set is local to that input. The inputs share a name without it.
    local_to: ?u32,
    defined_by: ?u32 = null,
};

pub fn createEmpty(
    arena: Allocator,
    comp: *Compilation,
    emit: Path,
    options: link.File.OpenOptions,
) !*Linker {
    const file = try emit.root_dir.handle.createFile(comp.io, emit.sub_path, .{});
    errdefer file.close(comp.io);
    const linker = try arena.create(Linker);
    linker.* = .{
        .base = .{
            .tag = .nvptx,
            .comp = comp,
            .emit = emit,
            .gc_sections = options.gc_sections orelse false,
            .print_gc_sections = options.print_gc_sections,
            .stack_size = options.stack_size orelse 0,
            .allow_shlib_undefined = options.allow_shlib_undefined orelse false,
            .file = file,
            .build_id = options.build_id,
        },
        .mirs = .empty,
        .externs = .empty,
        .exports = .empty,
        .export_navs = .empty,
        .inputs = .empty,
        .names = .empty,
        .shared_names = .empty,
        .input_data = .empty,
        .input_externs = .empty,
        .input_aliases = .empty,
        .input_roots = .empty,
    };
    return linker;
}

pub fn open(
    arena: Allocator,
    comp: *Compilation,
    emit: Path,
    options: link.File.OpenOptions,
) !*Linker {
    return createEmpty(arena, comp, emit, options);
}

pub fn deinit(linker: *Linker) void {
    const gpa = linker.base.comp.gpa;
    for (linker.mirs.values()) |*mir| mir.deinit(gpa);
    linker.mirs.deinit(gpa);
    for (linker.externs.values()) |*mir| mir.deinit(gpa);
    linker.externs.deinit(gpa);
    linker.exports.deinit(gpa);
    linker.export_navs.deinit(gpa);
    linker.inputs.deinit(gpa);
    for (linker.names.items) |name| gpa.free(name.bytes);
    linker.names.deinit(gpa);
    linker.shared_names.deinit(gpa);
    for (linker.input_data.values()) |*item| item.mir.deinit(gpa);
    linker.input_data.deinit(gpa);
    for (linker.input_externs.values()) |*mir| mir.deinit(gpa);
    linker.input_externs.deinit(gpa);
    linker.input_aliases.deinit(gpa);
    linker.input_roots.deinit(gpa);
}

pub fn loadInput(linker: *Linker, input: link.Input) !void {
    const diags = &linker.base.comp.link_diags;
    const file, const is_archive = switch (input) {
        .object => |object| .{ object, false },
        .archive => |archive| .{ archive, true },
        .res, .dso, .tbd => return diags.fail("linking other files into a PTX module is not supported", .{}),
    };
    const comp = linker.base.comp;
    const gpa = comp.gpa;
    const io = comp.io;
    const size = file.file.length(io) catch |err|
        return diags.failParse(file.path, "failed to get file size: {t}", .{err});
    const source = try gpa.alloc(u8, std.math.cast(usize, size) orelse return error.FileTooBig);
    defer gpa.free(source);
    const read = file.file.readPositionalAll(io, source, 0) catch |err|
        return diags.failParse(file.path, "failed to read: {t}", .{err});
    if (read != source.len) return diags.failParse(file.path, "incomplete read", .{});

    if (native_endian != .little) return diags.failParse(file.path, "reading an NVPTX object needs a little-endian host", .{});
    var object = Object.read(gpa, source) catch |err| switch (err) {
        error.OutOfMemory => |e| return e,
        error.InvalidObject => return diags.failParse(file.path, "the file is not an NVPTX object of format version {d}", .{Object.format_version}),
    };
    defer object.deinit(gpa);
    if (!std.mem.eql(u8, object.compiler, build_options.version)) {
        return diags.failParse(file.path, "Zig {s} wrote the object, but this compiler is Zig {s}; compile the object again", .{ object.compiler, build_options.version });
    }
    const target = &comp.root_mod.resolved_target.result;
    if (!std.mem.eql(u8, object.target, target.cpu.model.name) or object.address_size != target.ptrBitWidth()) {
        return diags.failParse(file.path, "the module targets {s} with {d}-bit addresses, but the output targets {s} with {d}-bit addresses", .{
            object.target, object.address_size, target.cpu.model.name, target.ptrBitWidth(),
        });
    }
    try linker.addObject(&object, .{ .path = file.path, .is_archive = is_archive });
}

fn addObject(linker: *Linker, object: *Object, input: Input) !void {
    const gpa = linker.base.comp.gpa;
    const input_index: u32 = @intCast(linker.inputs.items.len);
    try linker.inputs.append(gpa, input);

    const shared = try gpa.alloc(bool, object.names.len);
    defer gpa.free(shared);
    @memset(shared, false);
    for ([_][]Object.Item{ object.externs, object.data, object.functions }) |items| {
        for (items) |item| {
            if (item.linkage != .local) shared[item.mir.relocs[0].target.name] = true;
        }
    }
    for (object.aliases) |alias| {
        if (object.functions[alias.function].linkage != .local) shared[alias.name.name] = true;
    }
    const ids = try gpa.alloc(u32, object.names.len);
    defer gpa.free(ids);
    for (object.names, shared, ids) |bytes, is_shared, *id| {
        if (is_shared) if (linker.shared_names.get(bytes)) |existing| {
            id.* = existing;
            continue;
        };
        id.* = @intCast(linker.names.items.len);
        const owned = try gpa.dupe(u8, bytes);
        linker.names.append(gpa, .{ .bytes = owned, .local_to = if (is_shared) null else input_index }) catch |err| {
            gpa.free(owned);
            return err;
        };
        if (is_shared) try linker.shared_names.put(gpa, owned, id.*);
    }

    for (object.externs) |*item| {
        const id = renameTargets(&item.mir, ids);
        const gop = try linker.input_externs.getOrPut(gpa, id);
        if (gop.found_existing) continue;
        gop.value_ptr.* = item.mir;
        item.mir = .empty;
    }
    for (object.data) |*item| {
        const id = renameTargets(&item.mir, ids);
        if (item.linkage == .@"extern") {
            if (linker.input_data.contains(id)) continue;
        } else {
            if (!try linker.define(id, input_index)) continue;
            if (linker.input_data.fetchOrderedRemove(id)) |declaration| {
                var mir = declaration.value.mir;
                mir.deinit(gpa);
            }
            if (!input.is_archive and item.linkage == .visible) try linker.input_roots.append(gpa, .{ .name = id });
        }
        try linker.input_data.put(gpa, id, item.*);
        item.mir = .empty;
    }
    const function_ids = try gpa.alloc(u32, object.functions.len);
    defer gpa.free(function_ids);
    for (object.functions, function_ids) |*item, *function_id| {
        const id = renameTargets(&item.mir, ids);
        function_id.* = id;
        if (!try linker.define(id, input_index)) continue;
        if (!input.is_archive and (item.mir.kind == .entry or item.linkage == .visible)) {
            try linker.input_roots.append(gpa, .{ .name = id });
        }
        try linker.mirs.put(gpa, .{ .name = id }, item.mir);
        item.mir = .empty;
    }
    for (object.aliases) |alias| {
        const function_id = function_ids[alias.function];
        // An alias applies only to the definition of its own input.
        if (linker.names.items[function_id].defined_by != input_index) continue;
        const id = ids[alias.name.name];
        if (!try linker.define(id, input_index)) continue;
        try linker.input_aliases.put(gpa, id, function_id);
    }
}

/// Makes the linker use the definition of the name `id` in input `input_index`. Returns false when
/// the linker keeps an earlier definition. A definition in a static library never replaces an
/// earlier definition.
fn define(linker: *Linker, id: u32, input_index: u32) !bool {
    const gpa = linker.base.comp.gpa;
    const name = &linker.names.items[id];
    const earlier = name.defined_by orelse {
        name.defined_by = input_index;
        return true;
    };
    if (linker.inputs.items[input_index].is_archive) return false;
    if (!linker.inputs.items[earlier].is_archive) {
        return linker.base.comp.link_diags.failParse(linker.inputs.items[input_index].path, "duplicate symbol '{s}', which {f} also defines", .{
            name.bytes, linker.inputs.items[earlier].path,
        });
    }
    name.defined_by = input_index;
    if (linker.mirs.fetchOrderedRemove(.{ .name = id })) |entry| {
        var mir = entry.value;
        mir.deinit(gpa);
    }
    if (linker.input_data.fetchOrderedRemove(id)) |entry| {
        var mir = entry.value.mir;
        mir.deinit(gpa);
    }
    _ = linker.input_aliases.orderedRemove(id);
    return true;
}

/// Converts each `.name` target of `mir` from an object name index to a linker name index with
/// `ids`. Returns the new name index of the item from the first relocation.
fn renameTargets(mir: *Mir, ids: []const u32) u32 {
    for (mir.relocs) |*reloc| reloc.target = .{ .name = ids[reloc.target.name] };
    for (mir.symbols) |*symbol| symbol.target = .{ .name = ids[symbol.target.name] };
    return mir.relocs[0].target.name;
}

pub fn updateFunc(
    linker: *Linker,
    pt: Zcu.PerThread,
    func_index: InternPool.Index,
    mir: *codegen.AnyMir,
) !void {
    const tracy_trace = tracy.traceNamed(@src(), "NVPTX updateFunc");
    defer tracy_trace.end();
    tracy_trace.setValue(mir.nvptx.instructions.len);

    const gpa = linker.base.comp.gpa;
    const nav_index = pt.zcu.funcInfo(func_index).owner_nav;
    tracy_trace.addText(pt.zcu.intern_pool.getNav(nav_index).fqn.toSlice(&pt.zcu.intern_pool));
    const gop = try linker.mirs.getOrPut(gpa, .{ .nav = nav_index });
    if (gop.found_existing) gop.value_ptr.deinit(gpa);
    gop.value_ptr.* = mir.nvptx;
    mir.nvptx = .empty;
}

/// Lowers only extern functions. `flush` lowers the variables of functions and exports.
pub fn updateNav(linker: *Linker, pt: Zcu.PerThread, nav_index: InternPool.Nav.Index) link.Error!void {
    const gpa = linker.base.comp.gpa;
    const ip = &pt.zcu.intern_pool;
    switch (ip.indexToKey(ip.getNav(nav_index).resolved.?.value)) {
        .@"extern" => |e| if (ip.isFunctionType(e.ty)) {
            var mir = try CodeGen.generateExternFunc(pt, nav_index, e.ty, .{ .nav = nav_index });
            errdefer mir.deinit(gpa);
            const gop = try linker.externs.getOrPut(gpa, nav_index);
            if (gop.found_existing) gop.value_ptr.deinit(gpa);
            gop.value_ptr.* = mir;
        },
        else => {},
    }
}

pub fn updateExports(
    linker: *Linker,
    pt: Zcu.PerThread,
    export_indices: []const Zcu.Export.Index,
) link.Error!void {
    const zcu = pt.zcu;
    const gpa = zcu.gpa;
    linker.exports.clearRetainingCapacity();
    linker.export_navs.clearRetainingCapacity();
    try linker.exports.ensureUnusedCapacity(gpa, export_indices.len);
    try linker.export_navs.ensureUnusedCapacity(gpa, export_indices.len);
    var strong_names: std.array_hash_map.Auto(InternPool.NullTerminatedString, void) = .empty;
    defer strong_names.deinit(gpa);
    for (export_indices) |exp_index| {
        const exp = exp_index.ptr(zcu);
        if (exp.opts.linkage == .strong) try strong_names.put(gpa, exp.opts.name, {});
    }
    for (export_indices) |exp_index| {
        const exp = exp_index.ptr(zcu);
        switch (exp.exported) {
            .nav => |nav| {
                // A strong export takes the name of a weak export. The first of two weak exports
                // keeps the name.
                const taken = strong_names.contains(exp.opts.name) or linker.export_navs.contains(exp.opts.name);
                if (exp.opts.linkage == .weak and taken) continue;
                linker.export_navs.putAssumeCapacity(exp.opts.name, nav);
                const gop = linker.exports.getOrPutAssumeCapacity(nav);
                if (!gop.found_existing) {
                    gop.value_ptr.* = exp.opts.name;
                    continue;
                }
                // PTX ISA 9.4, section 11.2.3: `.alias` names only a function that is not a kernel.
                const ip = &zcu.intern_pool;
                const ty = ip.getNav(nav).resolved.?.type;
                const kind = if (!ip.isFunctionType(ty))
                    "a variable"
                else if (zcu.typeToFunc(.fromInterned(ty)).?.cc == .nvptx_kernel)
                    "a kernel"
                else
                    continue;
                const first = for (export_indices) |other| {
                    const other_exp = other.ptr(zcu);
                    if (other_exp.exported == .nav and other_exp.exported.nav == nav) break other_exp;
                } else unreachable;
                try zcu.failed_exports.ensureUnusedCapacity(gpa, 1);
                const msg = try Zcu.ErrorMsg.create(gpa, exp.src, "PTX cannot export {s} under a second name", .{kind});
                errdefer msg.destroy(gpa);
                try zcu.errNote(first.src, msg, "first exported here", .{});
                zcu.failed_exports.putAssumeCapacityNoClobber(exp_index, msg);
            },
            .uav => return linker.base.comp.link_diags.fail("cannot export an unnamed constant to PTX", .{}),
        }
    }
}

pub fn flush(
    linker: *Linker,
    arena: Allocator,
    tid: Zcu.PerThread.Id,
    prog_node: std.Progress.Node,
) link.Error!void {
    const tracy_trace = tracy.traceNamed(@src(), "NVPTX flush");
    defer tracy_trace.end();
    tracy_trace.setValue(linker.mirs.count());

    const comp = linker.base.comp;
    const gpa = comp.gpa;
    const zcu = comp.zcu.?;
    const active = zcu.activate(tid);
    defer active.deactivate();
    const pt = active.pt;
    const ip = &zcu.intern_pool;
    const diags = &comp.link_diags;
    const target = &comp.root_mod.resolved_target.result;

    const sub_node = prog_node.start("Flush Module", 0);
    defer sub_node.end();

    for (linker.export_navs.keys()) |name| {
        const id = linker.shared_names.get(name.toSlice(ip)) orelse continue;
        const input = linker.inputs.items[linker.names.items[id].defined_by orelse continue];
        if (input.is_archive) continue;
        return diags.fail("duplicate symbol '{s}', which the Zig code and {f} both define", .{ name.toSlice(ip), input.path });
    }

    // The final link runs the inliner. Therefore an object keeps its calls.
    const relocatable = comp.config.output_mode == .Obj or (comp.config.output_mode == .Lib and comp.config.link_mode == .static);
    if (!relocatable) ltcg.run(gpa, &linker.mirs, linker) catch return diags.fail("out of memory", .{});
    if (std.debug.runtime_safety and !relocatable) for (linker.mirs.values()) |*mir| {
        if (try verify.check(arena, mir)) |message| return diags.fail("internal error: invalid SSA form after inlining: {s}", .{message});
    };
    // `reached` collects the kernels and exports, then everything that the collected items use. A
    // global uses a function when its initial value points to the function.
    var reached: Reached = .{ .live = try gpa.alloc(bool, linker.mirs.count()) };
    defer reached.deinit(gpa);
    const live = reached.live;
    const data = &reached.data;
    var used: std.ArrayList(Mir.Target) = .empty;
    defer used.deinit(gpa);
    {
        const reach_trace = tracy.traceNamed(@src(), "NVPTX reachability");
        defer reach_trace.end();

        @memset(reached.live, false);
        for (linker.mirs.keys(), linker.mirs.values()) |key, mir| {
            if (mir.kind == .entry and key == .nav) try linker.reach(&reached, key);
        }
        // compiler_rt exports every routine. The module keeps a routine of the bundled compiler_rt
        // only when kept code calls the routine.
        const compiler_rt_mod = if (comp.compiler_rt_strat == .zcu) zcu.root_mod.deps.get("compiler_rt") else null;
        for (linker.exports.keys()) |nav| {
            if (compiler_rt_mod) |mod| if (zcu.navFileScope(nav).mod.? == mod) continue;
            try linker.reach(&reached, .{ .nav = nav });
        }
        for (linker.input_roots.items) |root| try linker.reach(&reached, root);
        var i: usize = 0;
        while (true) : (i += 1) {
            while (reached.work.pop()) |index| {
                const mir = &linker.mirs.values()[index];
                if (!relocatable and mir.kind != .data) {
                    ltcg.optimize(gpa, mir, comp.root_mod.optimize_mode) catch |err| switch (err) {
                        error.TooManyRegisters => return diags.fail("a function needs more than 65536 registers of one class after the link-time passes", .{}),
                        error.OutOfMemory => |e| return e,
                    };
                    if (std.debug.runtime_safety) if (try verify.check(arena, mir)) |message| {
                        return diags.fail("internal error: invalid SSA form after the link-time passes: {s}", .{message});
                    };
                    if (std.debug.runtime_safety) if (verify.checkLowered(mir)) |message| {
                        return diags.fail("internal error: {s}", .{message});
                    };
                }
                for (mir.relocs) |reloc| try linker.reach(&reached, reloc.target);
                used.clearRetainingCapacity();
                try mir.usedSymbols(gpa, &used);
                for (used.items) |used_target| try linker.reach(&reached, used_target);
            }
            if (i == data.count()) break;
            const key = data.keys()[i];
            const mir = if (key == .name) linker.input_data.get(key.name).?.mir else switch (key) {
                .nav => |nav| CodeGen.generateNav(pt, nav),
                .uav, .const_uav => CodeGen.generateUav(pt, linker.ownerNav(), key),
                .tag_names, .error_names => CodeGen.generateNames(pt, linker.ownerNav(), key),
                .name, .extern_func => unreachable,
            } catch |err| switch (err) {
                error.AlreadyReported => return diags.fail(
                    "cannot lower a global variable to PTX",
                    .{},
                ),
                else => |e| return e,
            };
            data.values()[i] = mir;
            for (mir.relocs) |reloc| try linker.reach(&reached, reloc.target);
        }
    }

    var ptx_version = std.nvptx.Gpu.fromCpu(target.cpu).ptx;
    for (linker.mirs.values(), live) |mir, is_live| {
        if (is_live) ptx_version = @max(ptx_version, mir.ptx_version);
    }
    for (data.values()) |mir| ptx_version = @max(ptx_version, mir.ptx_version);

    var externs: std.ArrayList(Object.Item) = .empty;
    defer externs.deinit(gpa);
    var extern_names: std.StringArrayHashMapUnmanaged(void) = .empty;
    defer extern_names.deinit(gpa);
    var extern_funcs: std.ArrayList(Mir) = .empty;
    defer {
        for (extern_funcs.items) |*mir| mir.deinit(gpa);
        extern_funcs.deinit(gpa);
    }
    // ptxas rejects an extern declaration when the module also defines the function.
    for (linker.externs.keys(), linker.externs.values()) |nav, mir| {
        if (linker.isDefined(.{ .nav = nav })) continue;
        try extern_names.put(gpa, ip.getNav(nav).getExtern(ip).?.name.toSlice(ip), {});
        try externs.append(gpa, .{ .mir = mir, .linkage = .@"extern" });
    }
    for (reached.extern_funcs.keys(), reached.extern_funcs.values()) |name, ty| {
        const gop = try extern_names.getOrPut(gpa, name.toSlice(ip));
        if (gop.found_existing) continue;
        try extern_funcs.ensureUnusedCapacity(gpa, 1);
        const mir = CodeGen.generateExternFunc(pt, linker.ownerNav(), ty, .{ .extern_func = .{ .name = name, .ty = ty } }) catch |err| switch (err) {
            error.AlreadyReported => return diags.fail("cannot declare the function '{s}' in PTX", .{name.toSlice(ip)}),
            else => |e| return e,
        };
        extern_funcs.appendAssumeCapacity(mir);
        try externs.append(gpa, .{ .mir = mir, .linkage = .@"extern" });
    }
    for (linker.input_externs.keys(), linker.input_externs.values()) |id, mir| {
        if (linker.isDefined(.{ .name = id }) or extern_names.contains(linker.names.items[id].bytes)) continue;
        try externs.append(gpa, .{ .mir = mir, .linkage = .@"extern" });
    }
    var functions: std.ArrayList(Object.Item) = .empty;
    defer functions.deinit(gpa);
    const positions = try gpa.alloc(u32, linker.mirs.count());
    defer gpa.free(positions);
    for (linker.mirs.keys(), linker.mirs.values(), live, positions) |key, mir, is_live, *position| {
        if (!is_live) continue;
        position.* = @intCast(functions.items.len);
        try functions.append(gpa, .{ .mir = mir, .linkage = linker.linkage(key) });
    }
    const listed = if (relocatable) &.{} else ltcg.addressTable(gpa, &linker.mirs, linker, live, data.values()) catch return diags.fail("out of memory", .{});
    defer gpa.free(listed);
    const table = try gpa.alloc(u32, listed.len);
    defer gpa.free(table);
    for (table, listed) |*position, index| position.* = positions[index];
    var aliases: std.ArrayList(Object.Alias) = .empty;
    defer aliases.deinit(gpa);
    for (linker.export_navs.keys(), linker.export_navs.values()) |name, nav| {
        if (linker.exports.get(nav).? == name) continue;
        const index = linker.mirs.getIndex(.{ .nav = nav }) orelse continue;
        if (!live[index] or linker.mirs.values()[index].kind != .func) continue;
        try aliases.append(gpa, .{
            .name = .{ .extern_func = .{ .name = name, .ty = ip.getNav(nav).resolved.?.type } },
            .function = positions[index],
        });
    }
    for (linker.input_aliases.keys(), linker.input_aliases.values()) |id, function_id| {
        const index = linker.mirs.getIndex(.{ .name = function_id }) orelse continue;
        if (live[index]) try aliases.append(gpa, .{ .name = .{ .name = id }, .function = positions[index] });
    }
    // PTX ISA 9.4, section 11.2.3: `.alias` needs PTX ISA 6.3.
    if (aliases.items.len != 0) ptx_version = @max(ptx_version, 63);
    var ordered: std.ArrayList(Object.Item) = .empty;
    defer ordered.deinit(gpa);
    const emitted = try gpa.alloc(bool, data.count());
    defer gpa.free(emitted);
    @memset(emitted, false);
    for (0..data.count()) |index| try linker.orderData(data, &ordered, emitted, index);

    var output: Object = .{
        .ptx_version = ptx_version,
        .target = target.cpu.model.name,
        .address_size = target.ptrBitWidth(),
        .compiler = build_options.version,
        .externs = externs.items,
        .data = ordered.items,
        .functions = functions.items,
        .aliases = aliases.items,
        .names = &.{},
        .table = table,
    };
    var out: std.Io.Writer.Allocating = .init(arena);
    if (relocatable) {
        if (native_endian != .little) return diags.fail("writing an NVPTX object needs a little-endian host", .{});
        if (target.cpu.model.name.len > Object.target_len) return diags.fail("the CPU model name '{s}' does not fit in an NVPTX object", .{target.cpu.model.name});
        const object_trace = tracy.traceNamed(@src(), "NVPTX object");
        defer object_trace.end();
        var names: ObjectNames = .{ .linker = linker, .arena = arena };
        names.addAll(&output) catch return diags.fail("out of memory", .{});
        output.names = names.names.items;
        output.write(&out.writer, &names) catch return diags.fail("out of memory", .{});
    } else {
        const emission_trace = tracy.traceNamed(@src(), "NVPTX emission");
        defer emission_trace.end();
        output.emit(gpa, &out.writer, linker) catch return diags.fail("out of memory", .{});
    }

    const write_trace = tracy.traceNamed(@src(), "NVPTX file write");
    defer write_trace.end();
    write_trace.setValue(out.written().len);
    var file_writer = linker.base.file.?.writer(comp.io, &.{});
    file_writer.interface.writeAll(out.written()) catch {
        return diags.fail("failed to write {f}: {t}", .{ linker.base.emit, file_writer.err.? });
    };
    file_writer.end() catch |err| return diags.fail("failed to write {f}: {t}", .{ linker.base.emit, err });
}

/// Gives each target of an object one name index. Targets with one name share one index. The
/// final link then resolves an extern and its definition to one name.
const ObjectNames = struct {
    linker: *Linker,
    arena: Allocator,
    indices: std.array_hash_map.Auto(Mir.Target, u32) = .empty,
    by_bytes: std.StringHashMapUnmanaged(u32) = .empty,
    names: std.ArrayList([]const u8) = .empty,

    fn addAll(on: *ObjectNames, object: *const Object) !void {
        for ([_][]const Object.Item{ object.externs, object.data, object.functions }) |items| {
            for (items) |item| {
                for (item.mir.relocs) |reloc| try on.add(reloc.target);
                for (item.mir.symbols) |symbol| try on.add(symbol.target);
            }
        }
        for (object.aliases) |alias| try on.add(alias.name);
    }

    fn add(on: *ObjectNames, target: Mir.Target) !void {
        const gop = try on.indices.getOrPut(on.arena, target);
        if (gop.found_existing) return;
        var name: std.Io.Writer.Allocating = .init(on.arena);
        try on.linker.writeName(&name.writer, target);
        const bytes = name.written();
        const by_bytes = try on.by_bytes.getOrPut(on.arena, bytes);
        if (!by_bytes.found_existing) {
            by_bytes.value_ptr.* = @intCast(on.names.items.len);
            try on.names.append(on.arena, bytes);
        }
        gop.value_ptr.* = by_bytes.value_ptr.*;
    }

    pub fn nameIndex(on: *const ObjectNames, target: Mir.Target) u32 {
        return on.indices.get(target).?;
    }
};

const Reached = struct {
    live: []bool,
    work: std.ArrayList(u32) = .empty,
    /// `flush` generates the Zig variables in this map. `input_data` owns the input variables.
    data: std.array_hash_map.Auto(Mir.Target, Mir) = .empty,
    /// This map holds the type of each undefined `.extern_func` target.
    extern_funcs: std.array_hash_map.Auto(InternPool.NullTerminatedString, InternPool.Index) = .empty,

    fn deinit(reached: *Reached, gpa: Allocator) void {
        gpa.free(reached.live);
        reached.work.deinit(gpa);
        for (reached.data.keys(), reached.data.values()) |key, *mir| {
            if (key != .name) mir.deinit(gpa);
        }
        reached.data.deinit(gpa);
        reached.extern_funcs.deinit(gpa);
    }
};

/// Returns the first Zig function, or the first export when the Zig code has no function.
/// `generateUav` and `generateNames` read the compilation target from the module of this nav.
fn ownerNav(linker: *const Linker) InternPool.Nav.Index {
    for (linker.mirs.keys()) |key| if (key == .nav) return key.nav;
    return linker.exports.keys()[0];
}

/// Resolves an extern of the Zig code, an `.extern_func` target or a shared input name to its
/// definition. An export of the Zig code takes precedence. An input alias resolves to its function.
pub fn resolve(linker: *const Linker, target: Mir.Target) Mir.Target {
    const ip = &linker.base.comp.zcu.?.intern_pool;
    const string, const bytes = switch (target) {
        .nav => |nav| name: {
            const name = (ip.getNav(nav).getExtern(ip) orelse return target).name;
            break :name .{ name.toOptional(), name.toSlice(ip) };
        },
        .name => |id| name: {
            const name = linker.names.items[id];
            if (name.local_to != null) return .{ .name = linker.input_aliases.get(id) orelse id };
            break :name .{ ip.getString(name.bytes), name.bytes };
        },
        .extern_func => |extern_func| .{ extern_func.name.toOptional(), extern_func.name.toSlice(ip) },
        .uav, .const_uav, .tag_names, .error_names => return target,
    };
    if (string.unwrap()) |s| if (linker.export_navs.get(s)) |nav| return .{ .nav = nav };
    const id = linker.shared_names.get(bytes) orelse return target;
    if (linker.names.items[id].defined_by == null) return target;
    return .{ .name = linker.input_aliases.get(id) orelse id };
}

fn isDefined(linker: *const Linker, target: Mir.Target) bool {
    const ip = &linker.base.comp.zcu.?.intern_pool;
    return switch (linker.resolve(target)) {
        .nav => |nav| ip.getNav(nav).getExtern(ip) == null,
        .name => |id| linker.names.items[id].defined_by != null,
        .extern_func => false,
        .uav, .const_uav, .tag_names, .error_names => true,
    };
}

fn isExtern(linker: *const Linker, nav: InternPool.Nav.Index) bool {
    const ip = &linker.base.comp.zcu.?.intern_pool;
    return ip.getNav(nav).getExtern(ip) != null;
}

fn linkage(linker: *const Linker, key: Mir.Target) Object.Linkage {
    return switch (key) {
        .nav => |nav| if (linker.exports.contains(nav)) .visible else if (linker.isExtern(nav)) .@"extern" else .local,
        .name => |id| if (linker.input_data.get(id)) |item| item.linkage else if (linker.names.items[id].local_to == null) .visible else .local,
        .uav, .const_uav, .tag_names, .error_names => .local,
        .extern_func => unreachable,
    };
}

fn reach(linker: *Linker, reached: *Reached, target: Mir.Target) !void {
    const resolved = linker.resolve(target);
    if (linker.mirs.getIndex(resolved)) |index| {
        if (reached.live[index]) return;
        reached.live[index] = true;
        return reached.work.append(linker.base.comp.gpa, @intCast(index));
    }
    if (resolved == .extern_func) {
        _ = try reached.extern_funcs.getOrPutValue(linker.base.comp.gpa, resolved.extern_func.name, resolved.extern_func.ty);
        return;
    }
    try linker.addData(&reached.data, resolved);
}

fn addData(linker: *Linker, data: *std.array_hash_map.Auto(Mir.Target, Mir), target: Mir.Target) !void {
    const gpa = linker.base.comp.gpa;
    // The linker writes functions from `mirs` and from the extern declarations, never as data.
    switch (target) {
        .nav => |nav| {
            const ip = &linker.base.comp.zcu.?.intern_pool;
            if (ip.isFunctionType(ip.getNav(nav).resolved.?.type)) return;
        },
        .name => |id| if (!linker.input_data.contains(id)) return,
        .extern_func => unreachable,
        .uav, .const_uav, .tag_names, .error_names => {},
    }
    const gop = try data.getOrPut(gpa, target);
    if (!gop.found_existing) gop.value_ptr.* = .empty;
}

/// Appends the variables of `data[index]` first, then `data[index]` itself. ptxas rejects an
/// initial value that names a later variable.
fn orderData(
    linker: *Linker,
    data: *const std.array_hash_map.Auto(Mir.Target, Mir),
    ordered: *std.ArrayList(Object.Item),
    emitted: []bool,
    index: usize,
) !void {
    if (emitted[index]) return;
    emitted[index] = true;
    const mir = data.values()[index];
    for (mir.relocs) |reloc| {
        const dependency = data.getIndex(linker.resolve(reloc.target)) orelse continue;
        try linker.orderData(data, ordered, emitted, dependency);
    }
    try ordered.append(linker.base.comp.gpa, .{ .mir = mir, .linkage = linker.linkage(data.keys()[index]) });
}

pub fn writeName(linker: *Linker, w: *std.Io.Writer, target: Mir.Target) !void {
    const ip = &linker.base.comp.zcu.?.intern_pool;
    switch (target) {
        .uav => |uav| try w.print("__anon_{d}", .{@backingInt(uav)}),
        .const_uav => |uav| try w.print("__const_anon_{d}", .{@backingInt(uav)}),
        .tag_names => |enum_ty| try w.print("__zig_tag_names_{d}", .{@backingInt(enum_ty)}),
        .error_names => try w.writeAll("__zig_error_names"),
        .nav => |nav| {
            if (linker.exports.get(nav)) |name| return w.writeAll(name.toSlice(ip));
            if (ip.getNav(nav).getExtern(ip)) |e| return w.writeAll(e.name.toSlice(ip));
            const fqn = ip.getNav(nav).fqn.toSlice(ip);
            // A PTX identifier starts with a letter, `_`, `$` or `%` (PTX ISA 9.4, section 4.4).
            if (fqn.len > 0 and std.ascii.isDigit(fqn[0])) try w.writeByte('_');
            for (fqn) |c| try w.writeByte(if (std.ascii.isAlphanumeric(c) or c == '_') c else '_');
            try w.print("${d}", .{@backingInt(nav)});
        },
        .extern_func => |extern_func| try w.writeAll(extern_func.name.toSlice(ip)),
        .name => |id| {
            const name = linker.names.items[id];
            try w.writeAll(name.bytes);
            // Two inputs can use one local name for different items.
            if (name.local_to) |input| try w.print("$i{d}", .{input});
        },
    }
}
