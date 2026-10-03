//! Differential fuzz test of the PTX backend. Each input describes a program of generated test
//! functions. The test compiles the program in this process with a memory limit and a time limit,
//! once in Debug mode, which runs no transform, and once in a release mode, which runs all
//! transforms. Every kernel computes the checksums of one test function at run time and stores the
//! checksums that Sema computes at compile time.
//!
//! When the build option `test-nvptx-fuzz-cmd` names a command, the test runs the command on the
//! PTX of every program. The command assembles the PTX, runs the kernels on a GPU and compares the
//! two kinds of checksums. The usage of the command is in `runCheckCmd`.
//! A compile error that is not a TODO error is a backend bug when the unit compiles without code
//! generation. Running out of memory and each command outcome other than `pass` are backend bugs.
//! The test saves each program with a backend bug in `findings` under the work directory. A crash
//! or hang ends the process and leaves the program in `programs_path` and the step in `step.txt`.
const builtin = @import("builtin");
const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const Smith = std.testing.Smith;

const build_options = @import("build_options");
const options = @import("nvptx_fuzz_options");
const Compilation = @import("../../Compilation.zig");
const Module = @import("../../Module.zig");
const Zcu = @import("../../Zcu.zig");
const dev = @import("../../dev.zig");
const generator = @import("fuzz/generator.zig");

const rows = 32;
const max_functions = 6;
const memory_limit = 2 << 30;
const compile_seconds = 120;
const check_seconds = 120;
const leak_limit = 1 << 30;
const thread_limit = 1;
const reduce_trial_limit = 150;
const reductions_per_process = 2;
const random_input_len = 1 << 20;

/// A process may allocate the thread ids of the compiler only once. With `--fuzz`, the test runner
/// calls the test function again for every batch of inputs when the binary has several fuzz tests.
var thread_ids_allocated = false;

const Outcome = enum {
    pass,
    compiled,
    todo,
    gen_bug,
    compile_error,
    out_of_memory,
    ptxas_error,
    gpu_fault,
    timeout,
    mismatch,
    /// The unit fails in the release build and passes in the Debug build, so a transform can
    /// cause the failure. The message is the release mode.
    release_only,

    fn isBug(outcome: Outcome) bool {
        return switch (outcome) {
            .pass, .compiled, .todo, .gen_bug => false,
            .compile_error, .out_of_memory, .ptxas_error, .gpu_fault, .timeout, .mismatch, .release_only => true,
        };
    }

    /// Returns whether the message of the outcome tells two different causes apart.
    fn hasMessage(outcome: Outcome) bool {
        return switch (outcome) {
            .todo, .gen_bug, .compile_error, .ptxas_error, .release_only => true,
            .pass, .compiled, .out_of_memory, .gpu_fault, .timeout, .mismatch => false,
        };
    }
};

const Harness = struct {
    io: Io,
    compile_threaded: Io.Threaded,
    arena_state: std.heap.ArenaAllocator,
    dirs: std.zig.Directories,
    environ_map: std.process.Environ.Map,
    work: Io.Dir,
    work_path: []const u8,
    /// The directory of the programs that this process checks, relative to `work`. A program
    /// that remains in this directory after the process ends crashed or hung the compiler.
    programs_path: []const u8,
    target: Module.ResolvedTarget,
    arch: []const u8,
    input_path: []const u8,
    records: Io.File,
    records_len: u64,
    counts: [@typeInfo(Outcome).@"enum".field_names.len]u64 = @splat(0),
    programs: u64 = 0,
    leaked_bytes: usize = 0,
    reduced: std.StringHashMapUnmanaged(void) = .empty,
    reductions: u32 = 0,
    /// While the test reduces a program, `record` changes neither `counts` nor `records`.
    reducing: bool = false,

    fn init(h: *Harness, io: Io) !void {
        h.* = .{
            .io = io,
            .compile_threaded = .init(std.heap.page_allocator, .{
                .async_limit = .limited(thread_limit - 1),
                .concurrent_limit = .limited(thread_limit - 1),
            }),
            .arena_state = .init(std.heap.page_allocator),
            .dirs = undefined,
            .environ_map = .init(std.heap.page_allocator),
            .work = undefined,
            .work_path = undefined,
            .programs_path = undefined,
            .target = undefined,
            .arch = "sm_90a",
            .input_path = undefined,
            .records = undefined,
            .records_len = undefined,
        };
        const arena = h.arena_state.allocator();
        const cwd = try std.zig.getResolvedCwd(io, arena);
        h.work_path = try std.fs.path.resolve(arena, &.{ cwd, options.work_dir });
        h.work = try Io.Dir.cwd().createDirPathOpen(io, h.work_path, .{});
        try h.work.createDirPath(io, "findings");
        h.programs_path = try std.fmt.allocPrint(arena, "programs-{x:0>8}", .{std.testing.random_seed});
        try h.work.createDirPath(io, h.programs_path);
        const cache_path = try std.fs.path.join(arena, &.{ h.work_path, "cache" });
        h.dirs = .{
            .cwd = cwd,
            .zig_lib = std.zig.Directories.openUnresolved(arena, io, cwd, options.zig_lib_dir, .@"zig lib"),
            .global_cache = std.zig.Directories.openUnresolved(arena, io, cwd, cache_path, .@"global cache"),
            .local_cache = undefined,
            .build_root = .{ .path = try std.zig.resolvePath(arena, cwd, &.{h.work_path}), .handle = h.work },
        };
        h.dirs.local_cache = h.dirs.global_cache;

        if (options.check_cmd.len != 0) {
            const argv = try std.mem.concat(arena, []const u8, &.{ options.check_cmd, &.{"--arch"} });
            const result = try std.process.run(arena, io, .{ .argv = argv });
            if (!result.term.eql(.success)) {
                std.debug.print("{s} --arch failed:\n{s}\n", .{ options.check_cmd[0], result.stderr });
                return error.CheckCmdFailed;
            }
            h.arch = std.mem.trim(u8, result.stdout, " \r\n");
        }
        const query = try std.Target.Query.parse(.{ .arch_os_abi = "nvptx64-cuda", .cpu_features = h.arch });
        h.target = .{
            .result = try std.zig.system.resolveTargetQuery(io, query),
            .is_native_os = false,
            .is_native_abi = false,
            .is_explicit_dynamic_linker = false,
        };
        if (!thread_ids_allocated) {
            try Zcu.PerThread.Id.allocate(std.heap.page_allocator, @max(thread_limit, 2));
            thread_ids_allocated = true;
        }

        const words = try generator.inputWords(arena, rows);
        h.input_path = try std.fs.path.join(arena, &.{ h.work_path, "input.bin" });
        try h.work.writeFile(io, .{ .sub_path = "input.bin", .data = std.mem.sliceAsBytes(words) });

        const records_name = try std.fmt.allocPrint(arena, "records-{x:0>8}.txt", .{std.testing.random_seed});
        h.records = try h.work.createFile(io, records_name, .{ .truncate = false });
        h.records_len = try h.records.length(io);
        if (h.work.readFileAlloc(io, "findings/reduced.txt", arena, .unlimited)) |reduced| {
            var lines = std.mem.splitScalar(u8, reduced, '\n');
            while (lines.next()) |line| if (line.len != 0) try h.reduced.put(arena, line, {});
        } else |_| {}
    }

    fn deinit(h: *Harness) void {
        h.work.deleteDir(h.io, h.programs_path) catch {};
        h.records.close(h.io);
        h.dirs.global_cache.handle.close(h.io);
        h.dirs.zig_lib.handle.close(h.io);
        h.work.close(h.io);
        h.environ_map.deinit();
        h.compile_threaded.deinit();
        h.arena_state.deinit();
    }

    fn record(h: *Harness, program: *Program, outcome: Outcome, k: u32, message: []const u8) !void {
        if (outcome.isBug()) program.bugs += 1;
        if (outcome == .gen_bug) program.gen_bugs += 1;
        if (outcome != .release_only) program.outcomes[k] = outcome;
        var buffer: [1024]u8 = undefined;
        var w: Io.Writer = .fixed(&buffer);
        w.print("{t}\t{t}\t{s}\t{d}\t", .{ outcome, program.mode, program.name, k }) catch {};
        for (message[0..@min(message.len, 600)]) |c| {
            w.writeByte(if (c == '\n' or c == '\t') ' ' else c) catch {};
        }
        w.writeByte('\n') catch {};
        if (!h.reducing) {
            h.counts[@backingInt(outcome)] += 1;
            try h.records.writePositionalAll(h.io, w.buffered(), h.records_len);
            h.records_len += w.buffered().len;
        }
        if (outcome.isBug() or outcome == .gen_bug) {
            try program.failures.appendSlice(program.arena, w.buffered());
            const sub_path = try std.fmt.allocPrint(program.arena, "repro-k{d}.zig", .{k});
            try program.writeSource(h.io, &.{k}, sub_path);
        }
    }

    fn printSummary(h: *Harness) void {
        std.debug.print("{d} programs:", .{h.programs});
        for (h.counts, 0..) |count, i| {
            if (count != 0) std.debug.print(" {t}={d}", .{ @as(Outcome, @fromBackingInt(@intCast(i))), count });
        }
        std.debug.print("\n", .{});
    }
};

const Program = struct {
    arena: Allocator,
    name: []const u8,
    sub_path: []const u8,
    path: []const u8,
    dir: Io.Dir,
    units: []*generator.Unit,
    mode: std.lang.Optimize,
    outcomes: []?Outcome,
    /// The first and last line of each unit in `prog.zig`, indexed like `units`.
    lines: [][2]u32,
    check_only: bool = false,
    bugs: u32 = 0,
    gen_bugs: u32 = 0,
    failures: std.ArrayList(u8) = .empty,

    fn unitOfLine(program: *const Program, line: u32) ?u32 {
        for (program.lines, 0..) |range, i| {
            if (line >= range[0] and line <= range[1]) return @intCast(i);
        }
        return null;
    }

    fn setStep(program: *Program, io: Io, comptime format: []const u8, args: anytype) !void {
        const text = try std.fmt.allocPrint(program.arena, format, args);
        try program.dir.writeFile(io, .{ .sub_path = "step.txt", .data = text });
    }

    /// Writes every unit and the kernels of the units `ks` to `sub_path`. The line numbers of a
    /// unit are the same in every file.
    fn writeSource(program: *Program, io: Io, ks: []const u32, sub_path: []const u8) !void {
        var w: Io.Writer.Allocating = .init(program.arena);
        const out = &w.writer;
        try out.writeAll(generator.header);
        try out.print("const rows = {d};\n\nconst inputs = [_]u64{{", .{rows});
        const words = try generator.inputWords(program.arena, rows);
        for (words, 0..) |word, i| {
            try out.print("{s}0x{x},", .{ if (i % 4 == 0) "\n    " else " ", word });
        }
        try out.writeAll("\n};\n");
        for (program.units, program.lines) |u, *range| {
            try out.writeAll("\n");
            range[0] = @intCast(std.mem.count(u8, w.written(), "\n") + 1);
            try u.print(out);
            range[1] = @intCast(std.mem.count(u8, w.written(), "\n"));
        }
        for (ks) |k| {
            const unit = program.units[k];
            var kernel_args: [generator.input_words][]const u8 = undefined;
            var expected_args: [generator.input_words][]const u8 = undefined;
            var row_loop: [2][]const u8 = .{ "", "" };
            for (&kernel_args, &expected_args, 0..) |*kernel_arg, *expected_arg, word| {
                const by_value = (unit.by_value >> @intCast(word)) & 1 != 0;
                if (!by_value) row_loop = .{ ", 0..", ", r" };
                kernel_arg.* = if (by_value)
                    try std.fmt.allocPrint(program.arena, "first[{d}]", .{word})
                else
                    try std.fmt.allocPrint(program.arena, "in[r * 4 + {d}]", .{word});
                expected_arg.* = if (by_value)
                    try std.fmt.allocPrint(program.arena, "inputs[{d}]", .{word})
                else
                    try std.fmt.allocPrint(program.arena, "inputs[r * 4 + {d}]", .{word});
            }
            try out.print(
                \\
                \\fn expected{0d}() [rows]u32 {{
                \\    @setEvalBranchQuota(100_000_000);
                \\    var checksums: [rows]u32 = undefined;
                \\    for (&checksums{9s}) |*c{10s}| c.* = t{0d}({1s}, {2s}, {3s}, {4s});
                \\    return checksums;
                \\}}
                \\
                \\export fn k{0d}(out: [*]addrspace(.global) u32, in: [*]addrspace(.global) const u64, first: [4]u64) callconv(.kernel) void {{
                \\    const r = @workItemId(0);
                \\    _ = .{{ &in, &first }};
                \\    const expected = comptime expected{0d}();
                \\    out[r] = t{0d}({5s}, {6s}, {7s}, {8s});
                \\    out[rows + r] = expected[r];
                \\}}
                \\
            , .{ unit.index, expected_args[0], expected_args[1], expected_args[2], expected_args[3], kernel_args[0], kernel_args[1], kernel_args[2], kernel_args[3], row_loop[0], row_loop[1] });
        }
        try program.dir.writeFile(io, .{ .sub_path = sub_path, .data = w.written() });
    }
};

const Diagnostic = struct {
    unit: ?u32,
    message: []const u8,
};

const Compiled = union(enum) {
    ok,
    errors: []const Diagnostic,
    out_of_memory,
};

/// Compiles `root_src_path` in the directory of `program` with a memory limit of `memory_limit`
/// bytes. Ends the process when the compilation takes longer than `compile_seconds`.
fn compile(h: *Harness, program: *Program, root_src_path: []const u8, emit_bin: bool) !Compiled {
    var debug_allocator: std.heap.DebugAllocator(.{
        .enable_memory_limit = true,
        .safety = false,
        .stack_trace_frames = 0,
    }) = .init;
    debug_allocator.requested_memory_limit = memory_limit;
    defer {
        h.leaked_bytes += debug_allocator.total_requested_bytes;
        debug_allocator.deinitWithoutLeakChecks();
    }
    const result = compileWithAllocator(h, program, root_src_path, emit_bin, debug_allocator.allocator()) catch |err| switch (err) {
        error.OutOfMemory => return .out_of_memory,
        else => |e| return e,
    };
    return result;
}

fn compileWithAllocator(h: *Harness, program: *Program, root_src_path: []const u8, emit_bin: bool, gpa: Allocator) !Compiled {
    const io = h.compile_threaded.io();
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const config = try Compilation.Config.resolve(.{
        .output_mode = .Exe,
        .resolved_target = h.target,
        .is_test = false,
        .have_zcu = true,
        .emit_bin = emit_bin,
        .root_optimize_mode = program.mode,
    });
    const root_mod = try Module.create(arena, .{
        .paths = .{
            .root = try .fromUnresolved(arena, h.dirs, &.{program.path}),
            .root_src_path = root_src_path,
        },
        .fully_qualified_name = "root",
        .cc_argv = &.{},
        .inherited = .{ .resolved_target = h.target, .optimize_mode = program.mode },
        .global = config,
        .parent = null,
    });
    const ptx_path = try std.fs.path.join(arena, &.{ program.path, "prog.ptx" });
    var create_diag: Compilation.CreateDiagnostic = undefined;
    const comp = Compilation.create(gpa, arena, io, &create_diag, .{
        .dirs = h.dirs,
        .thread_limit = thread_limit,
        .config = config,
        .root_mod = root_mod,
        .root_name = "prog",
        .cache_mode = .none,
        .emit_bin = if (emit_bin) .{ .yes_path = ptx_path } else .no,
        .environ_map = &h.environ_map,
    }) catch |err| switch (err) {
        error.CreateFail => {
            std.debug.print("unable to create a compilation: {f}\n", .{create_diag});
            return error.CreateFail;
        },
        else => |e| return e,
    };
    defer comp.destroy();

    var done: Io.Event = .unset;
    const watchdog = try std.Thread.spawn(.{}, watch, .{ &done, h.io, program.path });
    defer watchdog.join();
    defer done.set(h.io);
    try comp.update(.none);

    var errors = try comp.getAllErrorsAlloc();
    defer errors.deinit(gpa);
    if (errors.errorMessageCount() == 0) return .ok;
    var diagnostics: std.ArrayList(Diagnostic) = .empty;
    for (errors.getMessages()) |index| {
        const message = errors.getErrorMessage(index);
        const text = errors.nullTerminatedString(message.msg);
        if (std.mem.find(u8, text, "OutOfMemory") != null) return .out_of_memory;
        const unit = if (message.src_loc == .none) null else unit: {
            const location = errors.getSourceLocation(message.src_loc);
            const path = errors.nullTerminatedString(location.src_path);
            if (!std.mem.endsWith(u8, path, root_src_path)) break :unit null;
            break :unit program.unitOfLine(location.line + 1);
        };
        var full: Io.Writer.Allocating = .init(program.arena);
        try full.writer.writeAll(text);
        for (errors.getNotes(index)) |note| {
            try full.writer.print("; {s}", .{errors.nullTerminatedString(errors.getErrorMessage(note).msg)});
        }
        try diagnostics.append(program.arena, .{ .unit = unit, .message = full.written() });
    }
    return .{ .errors = diagnostics.items };
}

fn watch(done: *Io.Event, io: Io, program_path: []const u8) void {
    for (0..compile_seconds) |_| {
        done.waitTimeout(io, .{ .duration = .{ .raw = .fromSeconds(1), .clock = .awake } }) catch continue;
        return;
    }
    if (done.isSet()) return;
    std.debug.print("hang: the compilation of {s} took longer than {d} s\n", .{ program_path, compile_seconds });
    std.process.exit(3);
}

/// Compiles, checks and runs the units `ks` of `program`. When a step fails for a group of units,
/// the function repeats the step for each unit alone to find the unit that fails.
fn check(h: *Harness, program: *Program, all: []const u32) anyerror!void {
    const arena = program.arena;
    var ks: std.ArrayList(u32) = .empty;
    try ks.appendSlice(arena, all);
    while (true) {
        if (ks.items.len == 0) return;
        try program.writeSource(h.io, ks.items, "prog.zig");
        try program.setStep(h.io, "compile {t} units {any}\n", .{ program.mode, ks.items });
        switch (try compile(h, program, "prog.zig", true)) {
            .ok => break,
            .out_of_memory => {
                if (ks.items.len > 1) return checkEachAlone(h, program, ks.items);
                return h.record(program, .out_of_memory, ks.items[0], "");
            },
            .errors => |diagnostics| {
                var failed: std.ArrayList(u32) = .empty;
                for (diagnostics) |d| {
                    const k = if (ks.items.len == 1) ks.items[0] else d.unit orelse continue;
                    if (std.mem.findScalar(u32, failed.items, k) != null) continue;
                    try failed.append(arena, k);
                    if (std.mem.startsWith(u8, d.message, "TODO")) {
                        try h.record(program, .todo, k, d.message);
                    } else if (try failsWithoutCodegen(h, program, k)) {
                        try h.record(program, .gen_bug, k, d.message);
                    } else {
                        try h.record(program, .compile_error, k, d.message);
                    }
                }
                if (failed.items.len == 0) return checkEachAlone(h, program, ks.items);
                for (failed.items) |k| _ = ks.orderedRemove(std.mem.findScalar(u32, ks.items, k).?);
            },
        }
    }

    if (program.check_only or options.check_cmd.len == 0) {
        return recordEach(h, program, ks.items, .compiled);
    }
    try runCheckCmd(h, program, ks.items);
}

fn recordEach(h: *Harness, program: *Program, ks: []const u32, outcome: Outcome) !void {
    for (ks) |k| try h.record(program, outcome, k, "");
}

fn checkEachAlone(h: *Harness, program: *Program, ks: []const u32) !void {
    const alone = try program.arena.dupe(u32, ks);
    for (alone) |k| try check(h, program, &.{k});
}

/// Returns whether unit `k` alone fails to compile when the compiler does not generate code.
fn failsWithoutCodegen(h: *Harness, program: *Program, k: u32) !bool {
    const sub_path = try std.fmt.allocPrint(program.arena, "sema-k{d}.zig", .{k});
    try program.writeSource(h.io, &.{k}, sub_path);
    return switch (try compile(h, program, sub_path, false)) {
        .ok => false,
        .errors, .out_of_memory => true,
    };
}

/// Runs `<check_cmd> <ptx> <input> <rows> k<unit>...`. Kernel `k<unit>` writes `rows` checksums
/// that the GPU computes and then the `rows` checksums of `expected<unit>`. The command prints one
/// line `<outcome>\t<kernel>\t<message>` per result and stops after a `gpu_fault` or `timeout`.
/// The kernel `-` means that the outcome belongs to the whole module. Exit status 0 means that
/// every kernel passed, and 2 means that the command printed another outcome.
fn runCheckCmd(h: *Harness, program: *Program, ks: []const u32) !void {
    const arena = program.arena;
    const ptx_path = try std.fs.path.join(arena, &.{ program.path, "prog.ptx" });
    var first: usize = 0;
    while (first < ks.len) {
        const rest = ks[first..];
        try program.setStep(h.io, "check units {any}\n", .{rest});
        var argv: std.ArrayList([]const u8) = .empty;
        try argv.appendSlice(arena, options.check_cmd);
        try argv.appendSlice(arena, &.{ ptx_path, h.input_path, std.fmt.comptimePrint("{d}", .{rows}) });
        for (rest) |k| try argv.append(arena, try std.fmt.allocPrint(arena, "k{d}", .{k}));
        const timeout: Io.Timeout = .{ .duration = .{ .raw = .fromSeconds(check_seconds), .clock = .awake } };
        const result = std.process.run(arena, h.io, .{
            .argv = argv.items,
            .timeout = timeout.toDeadline(h.io),
        }) catch |err| switch (err) {
            error.Timeout => {
                if (rest.len > 1) return checkEachAlone(h, program, rest);
                return h.record(program, .timeout, rest[0], "");
            },
            else => |e| return e,
        };
        const status = if (result.term == .exited) result.term.exited else 1;
        if (status != 0 and status != 2) {
            std.debug.print("{s} failed:\n{s}\n", .{ options.check_cmd[0], result.stderr });
            return error.CheckCmdFailed;
        }
        var reported: usize = 0;
        var stop: ?usize = null;
        // The path of the PTX file names the program, which would make every message unique.
        const output = try std.mem.replaceOwned(u8, arena, result.stdout, ptx_path, "prog.ptx");
        var lines = std.mem.tokenizeScalar(u8, output, '\n');
        while (lines.next()) |line| {
            var fields = std.mem.splitScalar(u8, line, '\t');
            const outcome = std.meta.stringToEnum(Outcome, fields.first()) orelse return error.CheckCmdOutput;
            const kernel = fields.next() orelse return error.CheckCmdOutput;
            if (std.mem.eql(u8, kernel, "-")) {
                if (rest.len > 1) return checkEachAlone(h, program, rest);
                return h.record(program, outcome, rest[0], fields.rest());
            }
            if (!std.mem.startsWith(u8, kernel, "k")) return error.CheckCmdOutput;
            const k = try std.fmt.parseUnsigned(u32, kernel[1..], 10);
            const i = std.mem.findScalar(u32, rest, k) orelse return error.CheckCmdOutput;
            try h.record(program, outcome, k, fields.rest());
            reported += 1;
            if (outcome == .gpu_fault or outcome == .timeout) stop = i + 1;
        }
        first += stop orelse {
            if (reported != rest.len) return error.CheckCmdOutput;
            return;
        };
    }
}

fn newProgram(h: *Harness, arena: Allocator, units: []*generator.Unit, mode: std.lang.Optimize) !*Program {
    var text: Io.Writer.Allocating = .init(arena);
    for (units) |u| try u.print(&text.writer);
    try text.writer.print("{t}", .{mode});
    const name = try std.fmt.allocPrint(arena, "{x:0>16}", .{std.hash.Wyhash.hash(0, text.written())});
    const sub_path = try std.fmt.allocPrint(arena, "{s}/{s}", .{ h.programs_path, name });
    const program = try arena.create(Program);
    program.* = .{
        .arena = arena,
        .name = name,
        .sub_path = sub_path,
        .path = try std.fs.path.join(arena, &.{ h.work_path, sub_path }),
        .dir = try h.work.createDirPathOpen(h.io, sub_path, .{}),
        .units = units,
        .mode = mode,
        .outcomes = try arena.alloc(?Outcome, units.len),
        .lines = try arena.alloc([2]u32, units.len),
    };
    @memset(program.outcomes, null);
    return program;
}

fn checkProgram(h: *Harness, arena: Allocator, units: []*generator.Unit, mode: std.lang.Optimize, check_only: bool) !*Program {
    const program = try newProgram(h, arena, units, mode);
    program.check_only = check_only;
    const all = try arena.alloc(u32, units.len);
    for (all, 0..) |*k, i| k.* = @intCast(i);
    try check(h, program, all);
    return program;
}

fn testOne(h: *Harness, smith: *Smith) anyerror!void {
    var arena_state: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const mode: std.lang.Optimize = if (generator.below(smith, 2) == 0) .ReleaseSafe else .ReleaseFast;
    const units = try arena.alloc(*generator.Unit, 1 + generator.below(smith, max_functions));
    for (units, 0..) |*u, i| {
        u.* = try arena.create(generator.Unit);
        u.*.* = .init(arena, smith, @intCast(i));
        try u.*.generate();
    }
    const program = try checkProgram(h, arena, units, mode, false);
    defer program.dir.close(h.io);
    const baseline = try checkProgram(h, arena, units, .Debug, false);
    defer baseline.dir.close(h.io);
    try recordReleaseOnly(h, program, baseline);
    h.programs += 1;
    if (h.leaked_bytes > leak_limit) {
        std.debug.print("the compiler leaked {d} bytes\n", .{h.leaked_bytes});
        return error.CompilerLeaked;
    }
    try saveOrDelete(h, arena, baseline);
    try saveOrDelete(h, arena, program);
}

fn recordReleaseOnly(h: *Harness, program: *Program, baseline: *const Program) !void {
    for (program.outcomes, baseline.outcomes, 0..) |outcome, baseline_outcome, k| {
        if (!(outcome orelse continue).isBug() or (baseline_outcome orelse continue).isBug()) continue;
        try h.record(program, .release_only, @intCast(k), @tagName(program.mode));
    }
}

fn saveOrDelete(h: *Harness, arena: Allocator, program: *Program) !void {
    if (program.bugs == 0 and program.gen_bugs == 0) {
        try h.work.deleteTree(h.io, program.sub_path);
        return;
    }
    const prefix = if (program.bugs == 0) "gen-" else "";
    const finding = try std.fmt.allocPrint(arena, "findings/{s}{s}", .{ prefix, program.name });
    try h.work.deleteTree(h.io, finding);
    try h.work.rename(program.sub_path, h.work, finding, h.io);
    try program.dir.writeFile(h.io, .{ .sub_path = "failures.txt", .data = program.failures.items });
    if (program.bugs == 0) return;
    std.debug.print("backend bug in {s}/{s}:\n{s}", .{ h.work_path, finding, program.failures.items });
    if (!builtin.fuzz) try reduceFailures(h, arena, program);
}

/// Shrinks the unit of each failure whose signature is new. Stops after `reductions_per_process`
/// reductions in one process. Saves each result in `findings/reduced-<program>`.
fn reduceFailures(h: *Harness, arena: Allocator, program: *Program) !void {
    var lines = std.mem.splitScalar(u8, program.failures.items, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        if (h.reductions == reductions_per_process) break;
        var fields = std.mem.splitScalar(u8, line, '\t');
        const outcome = std.meta.stringToEnum(Outcome, fields.next().?).?;
        if (!outcome.isBug() or outcome == .release_only) continue;
        _ = fields.next();
        _ = fields.next();
        const k = try std.fmt.parseUnsigned(u32, fields.next().?, 10);
        const signature = try signatureOf(arena, outcome, program.mode, fields.rest());
        if (h.reduced.contains(signature)) continue;
        h.reductions += 1;
        if (outcome.hasMessage()) {
            const persistent = try h.arena_state.allocator().dupe(u8, signature);
            try h.reduced.put(h.arena_state.allocator(), persistent, {});
            const reduced_file = try h.work.createFile(h.io, "findings/reduced.txt", .{ .truncate = false });
            defer reduced_file.close(h.io);
            const reduced_len = try reduced_file.length(h.io);
            try reduced_file.writePositionalAll(h.io, try std.fmt.allocPrint(arena, "{s}\n", .{signature}), reduced_len);
        }
        try reduceOne(h, arena, program.units[k], outcome, program.mode, signature);
    }
}

/// Returns the outcome, the mode and the message with every digit replaced by `N`.
fn signatureOf(arena: Allocator, outcome: Outcome, mode: std.lang.Optimize, message: []const u8) ![]const u8 {
    const kept = if (outcome.hasMessage()) message else "";
    const signature = try std.fmt.allocPrint(arena, "{t} {t} {s}", .{ outcome, mode, kept });
    for (signature) |*c| {
        if (std.ascii.isDigit(c.*)) c.* = 'N';
    }
    return signature;
}

fn reduceOne(h: *Harness, arena: Allocator, unit: *generator.Unit, outcome: Outcome, mode: std.lang.Optimize, signature: []const u8) !void {
    unit.index = 0;
    var units = [_]*generator.Unit{unit};
    var edits: std.ArrayList(generator.Edit) = .empty;
    var trials: u32 = 0;
    var last_failure: ?*Program = null;
    var progress = true;
    while (progress and trials < reduce_trial_limit) {
        progress = false;
        var i: usize = 0;
        while (trials < reduce_trial_limit) : (trials += 1) {
            edits.clearRetainingCapacity();
            try unit.collectEdits(arena, &edits);
            if (i >= edits.items.len) break;
            const before = try unit.snapshot();
            unit.apply(edits.items[i]);
            if (try failsTheSameWay(h, arena, &units, outcome, mode, signature)) |program| {
                if (last_failure) |p| if (!std.mem.eql(u8, p.name, program.name)) {
                    try h.work.deleteTree(h.io, p.sub_path);
                };
                last_failure = program;
                progress = true;
            } else {
                unit.restore(before);
                i += 1;
            }
        }
    }
    const program = last_failure orelse return;
    const target = try std.fmt.allocPrint(arena, "findings/reduced-{s}", .{program.name});
    try h.work.deleteTree(h.io, target);
    try h.work.rename(program.sub_path, h.work, target, h.io);
    std.debug.print("reduced {s} after {d} trials: {s}/{s}\n", .{ signature, trials, h.work_path, target });
}

fn failsTheSameWay(
    h: *Harness,
    arena: Allocator,
    units: []*generator.Unit,
    outcome: Outcome,
    mode: std.lang.Optimize,
    signature: []const u8,
) !?*Program {
    h.reducing = true;
    defer h.reducing = false;
    const check_only = switch (outcome) {
        .compile_error, .out_of_memory => true,
        else => false,
    };
    const program = try checkProgram(h, arena, units, mode, check_only);
    defer program.dir.close(h.io);
    var lines = std.mem.splitScalar(u8, program.failures.items, '\n');
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        var fields = std.mem.splitScalar(u8, line, '\t');
        const got = std.meta.stringToEnum(Outcome, fields.next().?).?;
        _ = fields.next();
        _ = fields.next();
        _ = fields.next();
        if (got != outcome) continue;
        if (std.mem.eql(u8, try signatureOf(arena, got, mode, fields.rest()), signature)) {
            try program.dir.writeFile(h.io, .{ .sub_path = "failures.txt", .data = program.failures.items });
            return program;
        }
    }
    try h.work.deleteTree(h.io, program.sub_path);
    return null;
}

const corpus = [_][]const u8{&corpusEntry(1)};

fn corpusEntry(comptime seed: u64) [4 * 1024]u8 {
    @setEvalBranchQuota(1_000_000);
    comptime {
        var bytes: [4 * 1024]u8 = undefined;
        var prng: std.Random.DefaultPrng = .init(seed);
        prng.random().bytes(&bytes);
        return bytes;
    }
}

test "PTX backend and comptime compute the same checksums" {
    if (!dev.env.supports(.nvptx_backend)) return error.SkipZigTest;
    if (build_options.io_mode != .threaded) return error.SkipZigTest;
    var h: Harness = undefined;
    try h.init(std.testing.io);
    defer h.deinit();
    var bugs: u64 = 0;
    if (!builtin.fuzz and options.random_programs != 0) {
        var prng: std.Random.DefaultPrng = .init(std.testing.random_seed);
        const bytes = try std.testing.allocator.alloc(u8, random_input_len);
        defer std.testing.allocator.free(bytes);
        for (0..options.random_programs) |_| {
            prng.random().bytes(bytes);
            var smith: Smith = .{ .in = bytes };
            try testOne(&h, &smith);
        }
    }
    try std.testing.fuzz(&h, testOne, .{ .corpus = &corpus });
    if (builtin.fuzz) return;
    if (options.random_programs != 0) h.printSummary();
    for (h.counts, 0..) |count, i| {
        if (@as(Outcome, @fromBackingInt(@intCast(i))).isBug()) bugs += count;
    }
    if (bugs != 0) return error.BackendBug;
}

test {
    _ = @import("transforms/constant_operands.zig");
    _ = @import("transforms/positive_guard.zig");
}
