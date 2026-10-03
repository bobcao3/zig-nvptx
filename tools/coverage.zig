//! The `run` command records which coverage points of a `-ffuzz` program ran, in the command and
//! in every process that the command starts. The `report` command writes a Markdown report of the
//! branch coverage of the source paths on the command line. The `overlap` command finds the
//! counter files whose branches the other counter files also cover.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const linux = std.os.linux;
const elf = std.elf;
const fatal = std.process.fatal;
const Coverage = std.debug.Coverage;

comptime {
    if (builtin.os.tag != .linux or builtin.object_format != .elf or @sizeOf(usize) != 8) {
        @compileError("coverage supports only 64-bit ELF executables on Linux");
    }
}

const usage =
    \\usage: coverage run [--exit-zero] [--each <directory>] <counters> -- <command> [argument...]
    \\       coverage report <source root> <report.md> <source path>... -- <executable> <counters>...
    \\       coverage overlap <source root> <source path>... -- <executable> <counters>...
    \\
    \\`--each` writes the counters of each process to `<directory>/<n>` and its command line to
    \\`<directory>/<n>.cmd`. The first executable of a report defines its branches. A further
    \\`-- <executable> <counters>...` adds the counters of another instrumented program for the
    \\branches that the program shares with the first one.
    \\
;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(arena);

    if (args.len < 2) fatal(usage, .{});
    const separator = for (args[2..], 2..) |arg, i| {
        if (std.mem.eql(u8, arg, "--")) break i;
    } else fatal(usage, .{});
    if (std.mem.eql(u8, args[1], "run")) {
        var exit_zero = false;
        var each_dir: ?[]const u8 = null;
        var i: usize = 2;
        while (i < separator) : (i += 1) {
            if (std.mem.eql(u8, args[i], "--exit-zero")) {
                exit_zero = true;
            } else if (std.mem.eql(u8, args[i], "--each") and i + 2 < separator) {
                i += 1;
                each_dir = args[i];
            } else break;
        }
        if (i + 1 != separator or separator + 1 == args.len) fatal(usage, .{});
        const code = try run(gpa, io, init.minimal.environ, args[i], each_dir, args[separator + 1 ..]);
        std.process.exit(if (exit_zero) 0 else code);
    } else if (std.mem.eql(u8, args[1], "report")) {
        if (separator < 5) fatal(usage, .{});
        try report(gpa, arena, io, .{ .root = args[2], .paths = args[4..separator] }, args[3], try groups(arena, args[separator + 1 ..]));
    } else if (std.mem.eql(u8, args[1], "overlap")) {
        if (separator < 4) fatal(usage, .{});
        try overlap(gpa, arena, io, .{ .root = args[2], .paths = args[3..separator] }, try groups(arena, args[separator + 1 ..]));
    } else fatal(usage, .{});
}

/// An instrumented program and the counter files of its runs.
const Group = struct {
    exe_path: []const u8,
    counters_paths: []const []const u8,
};

fn groups(arena: Allocator, args: []const []const u8) ![]const Group {
    var list: std.ArrayList(Group) = .empty;
    var rest = args;
    while (rest.len != 0) {
        const end = for (rest, 0..) |arg, i| {
            if (std.mem.eql(u8, arg, "--")) break i;
        } else rest.len;
        if (end == 0) fatal(usage, .{});
        try list.append(arena, .{ .exe_path = rest[0], .counters_paths = rest[1..end] });
        rest = rest[@min(end + 1, rest.len)..];
    }
    if (list.items.len == 0) fatal(usage, .{});
    return list.items;
}

const Section = struct {
    address: u64,
    offset: u64,
    size: u64,
};

const Sections = struct {
    entry: u64,
    counters: ?Section,
    pcs: ?Section,
};

fn findSections(gpa: Allocator, io: Io, file: Io.File) !Sections {
    var buffer: [4096]u8 = undefined;
    var file_reader = file.reader(io, &buffer);
    const header = try elf.Header.read(&file_reader.interface);
    const section_headers = try gpa.alloc(elf.Elf64_Shdr, header.shnum);
    defer gpa.free(section_headers);
    var it = header.iterateSectionHeaders(&file_reader);
    for (section_headers) |*section_header| section_header.* = (try it.next()).?;

    const names_header = section_headers[header.shstrndx];
    const names = try gpa.alloc(u8, names_header.sh_size);
    defer gpa.free(names);
    if (try file.readPositionalAll(io, names, names_header.sh_offset) != names.len) {
        return error.TruncatedElfFile;
    }

    var sections: Sections = .{ .entry = header.entry, .counters = null, .pcs = null };
    for (section_headers) |section_header| {
        const name = std.mem.sliceTo(names[section_header.sh_name..], 0);
        const section: Section = .{
            .address = section_header.sh_addr,
            .offset = section_header.sh_offset,
            .size = section_header.sh_size,
        };
        if (std.mem.eql(u8, name, "__sancov_cntrs")) sections.counters = section;
        if (std.mem.eql(u8, name, "__sancov_pcs1")) sections.pcs = section;
    }
    return sections;
}

const Process = struct {
    /// The address of the counters, or 0 in a process whose program has no counters.
    address: usize,
    /// The points that ran in this process, when the tracer writes the counters of each process.
    ran: []u8 = &.{},
    command: []const u8 = &.{},
};

const Tracer = struct {
    /// One byte per coverage point of the instrumented program, nonzero if the point ran.
    ran: []u8 = &.{},
    buffer: []u8 = &.{},
    processes: std.array_hash_map.Auto(linux.pid_t, Process) = .empty,
    /// The process of each traced thread.
    threads: std.array_hash_map.Auto(linux.pid_t, linux.pid_t) = .empty,
    each_dir: ?Io.Dir = null,
    written: usize = 0,

    fn processOf(tracer: *Tracer, gpa: Allocator, io: Io, tid: linux.pid_t) !linux.pid_t {
        const thread = try tracer.threads.getOrPut(gpa, tid);
        if (!thread.found_existing) thread.value_ptr.* = readProcessId(io, tid) catch tid;
        const pid = thread.value_ptr.*;
        if (!tracer.processes.contains(pid)) try tracer.addProcess(gpa, io, pid);
        return pid;
    }

    fn addProcess(tracer: *Tracer, gpa: Allocator, io: Io, pid: linux.pid_t) !void {
        // An exec replaces the program of a process that the tracer may know already.
        if (tracer.processes.getPtr(pid)) |old| try tracer.writeProcess(gpa, io, old);
        var path_buffer: [64]u8 = undefined;
        const exe_path = try std.fmt.bufPrint(&path_buffer, "/proc/{d}/exe", .{pid});
        const address = address: {
            const file = Io.Dir.cwd().openFile(io, exe_path, .{}) catch break :address 0;
            defer file.close(io);
            const sections = findSections(gpa, io, file) catch break :address 0;
            const counters = sections.counters orelse break :address 0;
            if (tracer.ran.len == 0) {
                tracer.ran = try gpa.alloc(u8, counters.size);
                tracer.buffer = try gpa.alloc(u8, counters.size);
                @memset(tracer.ran, 0);
            } else if (tracer.ran.len != counters.size) {
                fatal("the command executed two programs with different coverage points", .{});
            }
            break :address counters.address +% try loadBias(io, pid, sections.entry);
        };
        var process: Process = .{ .address = address };
        if (address != 0 and tracer.each_dir != null) {
            process.ran = try gpa.alloc(u8, tracer.ran.len);
            @memset(process.ran, 0);
            process.command = readCommand(gpa, io, pid) catch &.{};
        }
        try tracer.processes.put(gpa, pid, process);
    }

    fn forget(tracer: *Tracer, gpa: Allocator, io: Io, tid: linux.pid_t, pid: linux.pid_t) !void {
        _ = tracer.threads.swapRemove(tid);
        if (tid != pid) return;
        if (tracer.processes.getPtr(pid)) |process| try tracer.writeProcess(gpa, io, process);
        _ = tracer.processes.swapRemove(pid);
    }

    fn writeProcess(tracer: *Tracer, gpa: Allocator, io: Io, process: *Process) !void {
        defer {
            gpa.free(process.ran);
            gpa.free(process.command);
            process.ran = &.{};
            process.command = &.{};
        }
        if (process.ran.len == 0) return;
        for (process.ran) |*r| r.* = @intFromBool(r.* != 0);
        var name_buffer: [32]u8 = undefined;
        try tracer.each_dir.?.writeFile(io, .{
            .sub_path = try std.fmt.bufPrint(&name_buffer, "{d}", .{tracer.written}),
            .data = process.ran,
        });
        try tracer.each_dir.?.writeFile(io, .{
            .sub_path = try std.fmt.bufPrint(&name_buffer, "{d}.cmd", .{tracer.written}),
            .data = process.command,
        });
        tracer.written += 1;
    }

    /// Reads the counters of the process of `tid` through `tid`, which still has the memory of
    /// the process when the process exits.
    fn readCounters(tracer: *Tracer, tid: linux.pid_t, pid: linux.pid_t) void {
        const process = tracer.processes.getPtr(pid) orelse return;
        if (process.address == 0) return;
        const local: [1]std.posix.iovec = .{.{ .base = tracer.buffer.ptr, .len = tracer.buffer.len }};
        const remote: [1]std.posix.iovec_const = .{.{
            .base = @ptrFromInt(process.address),
            .len = tracer.buffer.len,
        }};
        const rc = linux.process_vm_readv(tid, &local, &remote, 0);
        if (linux.errno(rc) != .SUCCESS) return;
        for (tracer.ran[0..rc], tracer.buffer[0..rc]) |*ran, counter| ran.* |= counter;
        if (process.ran.len != 0) {
            for (process.ran[0..rc], tracer.buffer[0..rc]) |*ran, counter| ran.* |= counter;
        }
    }

    fn readAllCounters(tracer: *Tracer) void {
        for (tracer.processes.keys()) |pid| tracer.readCounters(pid, pid);
    }
};

fn readProcessId(io: Io, tid: linux.pid_t) !linux.pid_t {
    var path_buffer: [64]u8 = undefined;
    const status_path = try std.fmt.bufPrint(&path_buffer, "/proc/{d}/status", .{tid});
    var status_buffer: [8192]u8 = undefined;
    const status = try Io.Dir.cwd().readFile(io, status_path, &status_buffer);
    var lines = std.mem.splitScalar(u8, status, '\n');
    while (lines.next()) |line| {
        const value = std.mem.cutPrefix(u8, line, "Tgid:") orelse continue;
        return std.fmt.parseInt(linux.pid_t, std.mem.trim(u8, value, " \t"), 10);
    }
    return error.MissingProcessId;
}

/// Returns the command line of a process with one argument per line.
fn readCommand(gpa: Allocator, io: Io, pid: linux.pid_t) ![]const u8 {
    var path_buffer: [64]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buffer, "/proc/{d}/cmdline", .{pid});
    // procfs reports a size of zero, so the read needs a buffer of its own.
    const buffer = try gpa.alloc(u8, 1 << 20);
    defer gpa.free(buffer);
    const command = try gpa.dupe(u8, try Io.Dir.cwd().readFile(io, path, buffer));
    std.mem.replaceScalar(u8, command, 0, '\n');
    return command;
}

/// Returns the amount that the kernel added to the addresses of a position-independent
/// executable, or zero for other executables.
fn loadBias(io: Io, pid: linux.pid_t, file_entry: u64) !u64 {
    var path_buffer: [64]u8 = undefined;
    const auxv_path = try std.fmt.bufPrint(&path_buffer, "/proc/{d}/auxv", .{pid});
    var auxv_bytes: [4096]u8 align(@alignOf(usize)) = undefined;
    const len = (try Io.Dir.cwd().readFile(io, auxv_path, &auxv_bytes)).len;
    const auxv: []const [2]usize = @ptrCast(auxv_bytes[0 .. len - len % @sizeOf([2]usize)]);
    for (auxv) |entry| {
        if (entry[0] == elf.AT.ENTRY) return entry[1] -% file_entry;
    }
    return error.MissingEntryAddress;
}

/// The waitpid(2) flag `__WALL`. `std.os.linux.W` does not define it.
const wait_all_threads = 0x40000000;
/// A counter wraps to zero after 256 increments, so the tracer also reads the counters while the
/// programs run. The interval timer interrupts the wait for the next process event.
const read_interval_us = 50_000;

const timeval = extern struct {
    sec: isize,
    usec: isize,
};

const itimerval = extern struct {
    interval: timeval,
    value: timeval,
};

fn onAlarm(_: linux.SIG) callconv(.c) void {}

fn startReadTimer() void {
    std.posix.sigaction(.ALRM, &.{
        .handler = .{ .handler = onAlarm },
        .mask = linux.sigemptyset(),
        .flags = 0,
    }, null);
    const timer: itimerval = .{
        .interval = .{ .sec = 0, .usec = read_interval_us },
        .value = .{ .sec = 0, .usec = read_interval_us },
    };
    const rc = linux.syscall3(.setitimer, @backingInt(linux.ITIMER.REAL), @intFromPtr(&timer), 0);
    if (linux.errno(rc) != .SUCCESS) fatal("setitimer failed: {t}", .{linux.errno(rc)});
}

fn run(
    gpa: Allocator,
    io: Io,
    environ: std.process.Environ,
    counters_path: []const u8,
    each_dir: ?[]const u8,
    argv: []const []const u8,
) !u8 {
    var tracer: Tracer = .{};
    if (each_dir) |path| {
        tracer.each_dir = Io.Dir.cwd().createDirPathOpen(io, path, .{}) catch |err|
            fatal("cannot create {s}: {t}", .{ path, err });
    }
    const root = try startTraced(gpa, environ, argv);
    var status: i32 = undefined;
    if (linux.errno(linux.waitpid(root, &status, wait_all_threads)) != .SUCCESS or
        !linux.W.IFSTOPPED(@bitCast(status)))
    {
        fatal("cannot execute {s}", .{argv[0]});
    }
    const options = linux.PTRACE.O.TRACECLONE | linux.PTRACE.O.TRACEFORK |
        linux.PTRACE.O.TRACEVFORK | linux.PTRACE.O.TRACEEXEC | linux.PTRACE.O.TRACEEXIT |
        linux.PTRACE.O.EXITKILL;
    _ = linux.ptrace(linux.PTRACE.SETOPTIONS, root, 0, options, 0);
    try tracer.threads.put(gpa, root, root);
    try tracer.addProcess(gpa, io, root);
    startReadTimer();
    _ = linux.ptrace(linux.PTRACE.CONT, root, 0, 0, 0);

    var exit_code: u8 = 125;
    while (true) {
        const rc = linux.waitpid(-1, &status, wait_all_threads);
        switch (linux.errno(rc)) {
            .SUCCESS => {},
            .INTR => {
                tracer.readAllCounters();
                continue;
            },
            .CHILD => break,
            else => |err| fatal("waitpid failed: {t}", .{err}),
        }
        const tid: linux.pid_t = @intCast(rc);
        const s: u32 = @bitCast(status);
        const pid = try tracer.processOf(gpa, io, tid);
        if (linux.W.IFEXITED(s) or linux.W.IFSIGNALED(s)) {
            if (tid == root) exit_code = if (linux.W.IFEXITED(s))
                linux.W.EXITSTATUS(s)
            else
                128 +| @as(u8, @intCast(@backingInt(linux.W.TERMSIG(s))));
            try tracer.forget(gpa, io, tid, pid);
            continue;
        }
        if (!linux.W.IFSTOPPED(s)) continue;
        var signal: usize = 0;
        switch (s >> 16) {
            // ptrace stops each new thread with a SIGSTOP, which the tracer does not deliver.
            0 => if (linux.W.STOPSIG(s) != .STOP) {
                signal = @backingInt(linux.W.STOPSIG(s));
            },
            // After an exec, the thread that called it is the only thread of its process.
            linux.PTRACE.EVENT.EXEC => {
                try tracer.threads.put(gpa, tid, tid);
                try tracer.addProcess(gpa, io, tid);
            },
            linux.PTRACE.EVENT.EXIT => tracer.readCounters(tid, pid),
            else => {},
        }
        _ = linux.ptrace(linux.PTRACE.CONT, tid, 0, signal, 0);
    }

    if (tracer.ran.len == 0) fatal("{s} has no coverage counters; build it with -ffuzz", .{argv[0]});
    try mergeCounters(gpa, io, counters_path, tracer.ran);
    return exit_code;
}

/// The child stops with SIGTRAP after it executes the command.
fn startTraced(gpa: Allocator, environ: std.process.Environ, argv: []const []const u8) !linux.pid_t {
    const argv_z = try gpa.allocSentinel(?[*:0]const u8, argv.len, null);
    for (argv_z, argv) |*z, arg| z.* = try gpa.dupeSentinel(u8, arg, 0);
    // execve does not search PATH, so the parent builds each candidate path before the fork.
    var programs: std.ArrayList([:0]const u8) = .empty;
    if (std.mem.indexOfScalar(u8, argv[0], '/') != null) {
        try programs.append(gpa, argv_z[0].?[0..argv[0].len :0]);
    } else {
        var it = std.mem.tokenizeScalar(u8, environ.getPosix("PATH") orelse "", ':');
        while (it.next()) |directory| {
            try programs.append(gpa, try std.fmt.allocPrintSentinel(gpa, "{s}/{s}", .{
                directory,
                argv[0],
            }, 0));
        }
    }
    const envp: [*:null]const ?[*:0]const u8 = environ.block.slice.ptr;

    const rc = linux.fork();
    if (linux.errno(rc) != .SUCCESS) fatal("fork failed: {t}", .{linux.errno(rc)});
    if (rc == 0) {
        _ = linux.ptrace(linux.PTRACE.TRACEME, 0, 0, 0, 0);
        for (programs.items) |program| _ = linux.execve(program, argv_z, envp);
        linux.exit(127);
    }
    return @intCast(rc);
}

fn mergeCounters(gpa: Allocator, io: Io, path: []const u8, ran: []u8) !void {
    const file = try Io.Dir.cwd().createFile(io, path, .{
        .read = true,
        .truncate = false,
        .lock = .exclusive,
    });
    defer file.close(io);
    const len = try file.length(io);
    if (len != 0) {
        if (len != ran.len) fatal("{s} belongs to a different program", .{path});
        const old = try gpa.alloc(u8, ran.len);
        defer gpa.free(old);
        _ = try file.readPositionalAll(io, old, 0);
        for (ran, old) |*r, o| r.* |= o;
    }
    for (ran) |*r| r.* = @intFromBool(r.* != 0);
    try file.writePositionalAll(io, ran, 0);
}

const Sources = struct {
    root: []const u8,
    paths: []const []const u8,
};

const Location = struct {
    file: u32,
    line: u32,
    column: u32,
};

/// The source location of a coverage point: the location of the counter and the location of the
/// code after the counter. Every instantiation of a generic function and every inlined copy of
/// a function places a point with the same key. The error paths of the `try`s of a function that
/// return at once share one key, with the location of the function entry.
const BranchKey = struct {
    counter: Location,
    code: Location,
    error_return: bool,
};

const Branch = struct {
    ran: bool,
    file: u32,
    line: u32,
    error_return: bool,
    ignored: bool = false,
};

const Tally = struct {
    ran: usize = 0,
    total: usize = 0,

    fn add(tally: *Tally, ran: bool) void {
        tally.ran += @intFromBool(ran);
        tally.total += 1;
    }

    fn addAll(tally: *Tally, other: Tally) void {
        tally.ran += other.ran;
        tally.total += other.total;
    }

    fn percent(tally: Tally) f64 {
        if (tally.total == 0) return 0;
        return 100 * @as(f64, @floatFromInt(tally.ran)) / @as(f64, @floatFromInt(tally.total));
    }
};

const SourceFile = struct {
    index: u32,
    name: []const u8,
    lines: []const []const u8,
    branches: Tally = .{},
};

const Files = std.array_hash_map.String(SourceFile);

const no_branch = std.math.maxInt(u32);

/// The branches of the source paths, collected from the coverage points of one or more programs.
const Branches = struct {
    files: Files = .empty,
    list: std.array_hash_map.Auto(BranchKey, Branch) = .empty,
    definitions: ?Definitions = null,

    /// Returns the branch of each point of a program, or `no_branch` for a point outside the
    /// source paths. With `extend`, a point at a new source location adds a branch; without it,
    /// such a point has no branch, which keeps the code that only a test program holds out of
    /// the report.
    fn addPoints(branches: *Branches, gpa: Allocator, arena: Allocator, io: Io, exe_path: []const u8, sources: Sources, extend: bool) ![]const u32 {
        if (branches.definitions == null) branches.definitions = try findDefinitions(arena, io, sources);
        var coverage: Coverage = .init;
        defer coverage.deinit(gpa);
        const cache_path: std.Build.Cache.Path = .{ .root_dir = .cwd(), .sub_path = exe_path };
        var info = std.debug.Info.load(gpa, io, cache_path, &coverage, .elf, builtin.cpu.arch) catch |err|
            fatal("failed to load debug info of {s}: {t}", .{ exe_path, err });
        defer info.deinit(gpa);

        const exe_file = try Io.Dir.cwd().openFile(io, exe_path, .{});
        defer exe_file.close(io);
        const sections = try findSections(gpa, io, exe_file);
        const pcs_section = sections.pcs orelse
            fatal("{s} has no coverage points; build it with -ffuzz", .{exe_path});
        const pcs = try arena.alloc(u64, pcs_section.size / @sizeOf(u64));
        _ = try exe_file.readPositionalAll(io, @ptrCast(pcs), pcs_section.offset);
        if (std.mem.allEqual(u64, pcs, 0)) {
            fatal("{s} is position-independent; build it with -fno-pie", .{exe_path});
        }

        const functions = try readFunctions(arena, &info.impl.elf);
        const dwarf = &info.impl.elf.dwarf.?;
        const endian = info.impl.elf.endian;
        const point_branch = try arena.alloc(u32, pcs.len);
        @memset(point_branch, no_branch);
        for (pcs, point_branch) |pc, *point| {
            // LLVM writes the placeholder address 1 for some points (see lib/fuzzer.zig).
            if (pc == 1) continue;
            const function = functionAt(functions, pc) orelse continue;
            const compile_unit = dwarf.findCompileUnit(endian, pc) catch continue;
            dwarf.populateSrcLocCache(gpa, endian, compile_unit) catch continue;
            const line_info = &compile_unit.src_loc_cache.?;
            const is_entry = pc == function.address;
            const location = pointLocation(line_info, pc, function, is_entry) orelse continue;
            const counter = try locate(&branches.files, arena, io, sources, line_info, location.counter);
            const code = try locate(&branches.files, arena, io, sources, line_info, location.code);
            const file = branches.files.values()[counter.file];
            if (file.name.len == 0) continue;
            const same_file = code.file == counter.file;
            const returns_at_once = !is_entry and same_file and
                wordAt(sourceLine(file, code.line), code.column, "try");
            const key: BranchKey = if (returns_at_once) key: {
                const entry = pointLocation(line_info, function.address, function, true) orelse continue;
                const entry_location = try locate(&branches.files, arena, io, sources, line_info, entry.counter);
                break :key .{ .counter = entry_location, .code = entry_location, .error_return = true };
            } else .{ .counter = counter, .code = code, .error_return = false };
            if (failsOnlyByForwarding(file, branches.definitions.?, key, returns_at_once, same_file)) continue;
            if (!extend) {
                point.* = @intCast(branches.list.getIndex(key) orelse continue);
                continue;
            }
            const branch = try branches.list.getOrPut(arena, key);
            if (!branch.found_existing) branch.value_ptr.* = .{
                .ran = false,
                .file = key.counter.file,
                .line = if (returns_at_once) key.counter.line else branchLine(file, counter, code, same_file),
                .error_return = returns_at_once,
            };
            point.* = @intCast(branch.index);
        }
        return point_branch;
    }
};

fn orCounters(gpa: Allocator, io: Io, path: []const u8, ran: []u8) !void {
    const bytes = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited);
    defer gpa.free(bytes);
    if (bytes.len != ran.len) fatal("{s} belongs to a different program", .{path});
    for (ran, bytes) |*r, b| r.* |= b;
}

fn report(gpa: Allocator, arena: Allocator, io: Io, sources: Sources, report_path: []const u8, program_groups: []const Group) !void {
    var branches: Branches = .{};
    for (program_groups, 0..) |group, i| {
        const point_branch = try branches.addPoints(gpa, arena, io, group.exe_path, sources, i == 0);
        const ran = try arena.alloc(u8, point_branch.len);
        @memset(ran, 0);
        for (group.counters_paths) |path| try orCounters(gpa, io, path, ran);
        for (point_branch, ran) |branch, r| {
            if (branch != no_branch and r != 0) branches.list.values()[branch].ran = true;
        }
    }
    ignoreUnrecoverable(&branches);
    ignoreForwardedTries(&branches);
    for (branches.list.values()) |branch| {
        if (!branch.ignored) branches.files.values()[branch.file].branches.add(branch.ran);
    }

    var report_file = try Io.Dir.cwd().createFile(io, report_path, .{});
    defer report_file.close(io);
    var buffer: [4096]u8 = undefined;
    var file_writer = report_file.writer(io, &buffer);
    try writeReport(arena, &file_writer.interface, branches.files.values(), branches.list.values(), sources.paths);
    try file_writer.interface.flush();
}

/// Prints the number of branches that each counter file ran, the number of these branches that no
/// other kept counter file ran, and whether the kept counter files cover the file. Removing one
/// redundant file at a time, smallest first, keeps the union of the branches unchanged.
fn overlap(gpa: Allocator, arena: Allocator, io: Io, sources: Sources, program_groups: []const Group) !void {
    var branches: Branches = .{};
    const point_branches = try arena.alloc([]const u32, program_groups.len);
    for (program_groups, point_branches, 0..) |group, *point_branch, i| {
        point_branch.* = try branches.addPoints(gpa, arena, io, group.exe_path, sources, i == 0);
    }
    const branch_count = branches.list.count();
    const cover = try arena.alloc(u32, branch_count);
    @memset(cover, 0);
    var sets: std.ArrayList(std.DynamicBitSetUnmanaged) = .empty;
    var paths: std.ArrayList([]const u8) = .empty;
    for (program_groups, point_branches) |group, point_branch| {
        const raw = try gpa.alloc(u8, point_branch.len);
        defer gpa.free(raw);
        for (group.counters_paths) |path| {
            var set: std.DynamicBitSetUnmanaged = try .initEmpty(arena, branch_count);
            @memset(raw, 0);
            try orCounters(gpa, io, path, raw);
            for (point_branch, raw) |branch, r| {
                if (branch != no_branch and r != 0 and !set.isSet(branch)) {
                    set.set(branch);
                    cover[branch] += 1;
                }
            }
            try sets.append(arena, set);
            try paths.append(arena, path);
        }
    }

    const removed = try arena.alloc(bool, sets.items.len);
    @memset(removed, false);
    while (true) {
        var pick: ?usize = null;
        for (sets.items, removed, 0..) |set, is_removed, i| {
            if (is_removed or uniqueBranches(set, cover) != 0) continue;
            if (pick == null or set.count() < sets.items[pick.?].count()) pick = i;
        }
        const i = pick orelse break;
        removed[i] = true;
        var it = sets.items[i].iterator(.{});
        while (it.next()) |branch| cover[branch] -= 1;
    }

    var buffer: [4096]u8 = undefined;
    var stdout = Io.File.stdout().writer(io, &buffer);
    const w = &stdout.interface;
    try w.writeAll("branches\tunique\tstatus\tcounters\n");
    for (sets.items, removed, paths.items) |set, is_removed, path| {
        try w.print("{d}\t{d}\t{s}\t{s}\n", .{
            set.count(),
            if (is_removed) 0 else uniqueBranches(set, cover),
            if (is_removed) "redundant" else "kept",
            path,
        });
    }
    try w.flush();
}

fn uniqueBranches(set: std.DynamicBitSetUnmanaged, cover: []const u32) usize {
    var unique: usize = 0;
    var it = set.iterator(.{});
    while (it.next()) |branch| unique += @intFromBool(cover[branch] == 1);
    return unique;
}

const Function = struct {
    address: u64,
    size: u64,
    name: []const u8,

    fn lessThan(_: void, a: Function, b: Function) bool {
        return a.address < b.address;
    }
};

fn readFunctions(arena: Allocator, elf_file: *const std.debug.ElfFile) ![]const Function {
    const symtab = elf_file.symtab orelse return error.MissingSymbolTable;
    const strtab = elf_file.strtab orelse return error.MissingSymbolTable;
    const symbols: []align(1) const elf.Elf64_Sym = @ptrCast(symtab.bytes);
    var functions: std.ArrayList(Function) = .empty;
    for (symbols) |symbol| {
        if (symbol.st_type() != elf.STT_FUNC or symbol.st_size == 0) continue;
        try functions.append(arena, .{
            .address = symbol.st_value,
            .size = symbol.st_size,
            .name = std.mem.sliceTo(strtab[symbol.st_name..], 0),
        });
    }
    std.mem.sortUnstable(Function, functions.items, {}, Function.lessThan);
    return functions.items;
}

fn functionAt(functions: []const Function, pc: u64) ?Function {
    const i = std.sort.partitionPoint(Function, functions, pc, struct {
        fn predicate(target: u64, function: Function) bool {
            return function.address <= target;
        }
    }.predicate);
    if (i == 0) return null;
    const function = functions[i - 1];
    return if (pc < function.address + function.size) function else null;
}

const SrcLocCache = std.debug.Dwarf.CompileUnit.SrcLocCache;
const LineEntry = SrcLocCache.LineEntry;

const PointLocation = struct {
    counter: LineEntry,
    code: LineEntry,
};

/// At a point that is not a function entry, the counter increment carries the location of the
/// preceding code. The counted code starts at the next line table row with a different location.
fn pointLocation(
    line_info: *const SrcLocCache,
    pc: u64,
    function: Function,
    is_entry: bool,
) ?PointLocation {
    const addresses = line_info.line_table.keys();
    const entries = line_info.line_table.values();
    const first = std.sort.lowerBound(u64, addresses, pc, struct {
        fn order(context: u64, item: u64) std.math.Order {
            return std.math.order(context, item);
        }
    }.order);
    if (first == addresses.len) return null;
    const counter = entries[first];
    if (is_entry) return .{ .counter = counter, .code = counter };
    for (addresses[first..], entries[first..]) |address, entry| {
        if (address >= function.address + function.size) break;
        if (entry.file != counter.file or entry.line != counter.line or
            entry.column != counter.column) return .{ .counter = counter, .code = entry };
    }
    return .{ .counter = counter, .code = counter };
}

/// Returns the location of a line table row. The first row of a file adds the file to `files`,
/// with its source lines when one of the source paths of the report names the file.
fn locate(
    files: *Files,
    arena: Allocator,
    io: Io,
    sources: Sources,
    line_info: *const SrcLocCache,
    entry: LineEntry,
) !Location {
    const file_entry = line_info.files[entry.file - @intFromBool(line_info.version < 5)];
    const directory = line_info.directories[file_entry.dir_index].path;
    const path = try std.fs.path.join(arena, &.{ directory, file_entry.path });
    const gop = try files.getOrPut(arena, path);
    if (!gop.found_existing) {
        gop.value_ptr.* = if (matchesAny(path, sources.paths)) .{
            .index = @intCast(gop.index),
            .name = try arena.dupe(u8, file_entry.path),
            .lines = try readLines(arena, io, sources.root, file_entry.path, path),
        } else .{ .index = @intCast(gop.index), .name = "", .lines = &.{} };
    }
    return .{ .file = @intCast(gop.index), .line = entry.line, .column = entry.column };
}

/// A path that starts with `!` removes the files that it matches from the source paths.
fn matchesAny(path: []const u8, source_paths: []const []const u8) bool {
    var matched = false;
    for (source_paths) |source_path| {
        const excluded = std.mem.startsWith(u8, source_path, "!");
        if (!matchesPath(path, if (excluded) source_path[1..] else source_path)) continue;
        if (excluded) return false;
        matched = true;
    }
    return matched;
}

fn matchesPath(path: []const u8, source_path: []const u8) bool {
    if (std.mem.endsWith(u8, source_path, "/")) {
        if (std.mem.startsWith(u8, path, source_path)) return true;
        if (std.mem.indexOf(u8, path, source_path)) |i| {
            if (i > 0 and path[i - 1] == '/') return true;
        }
        return false;
    }
    if (std.mem.eql(u8, path, source_path)) return true;
    return std.mem.endsWith(u8, path, source_path) and path[path.len - source_path.len - 1] == '/';
}

fn readLines(
    arena: Allocator,
    io: Io,
    source_root: []const u8,
    name: []const u8,
    path: []const u8,
) ![]const []const u8 {
    const in_root = try std.fs.path.join(arena, &.{ source_root, name });
    const bytes = Io.Dir.cwd().readFileAlloc(io, in_root, arena, .unlimited) catch
        Io.Dir.cwd().readFileAlloc(io, path, arena, .unlimited) catch return &.{};
    var lines: std.ArrayList([]const u8) = .empty;
    var it = std.mem.splitScalar(u8, bytes, '\n');
    while (it.next()) |line| try lines.append(arena, line);
    return lines.items;
}

fn sourceLine(file: SourceFile, line: u32) []const u8 {
    if (line == 0 or line > file.lines.len) return "";
    return file.lines[line - 1];
}

/// The line of the code identifies a branch best, except when the code is a deferred statement
/// that an earlier `return` or error runs: then the line of the counter names the statement.
fn branchLine(file: SourceFile, counter: Location, code: Location, same_file: bool) u32 {
    if (!same_file) return counter.line;
    if (code.line >= counter.line or counter.line > file.lines.len) return code.line;
    const text = std.mem.trimStart(u8, sourceLine(file, counter.line), " ");
    return if (std.mem.startsWith(u8, text, "}")) code.line else counter.line;
}

const Definitions = std.array_hash_map.String(bool);

fn findDefinitions(arena: Allocator, io: Io, sources: Sources) !Definitions {
    var definitions: Definitions = .empty;
    for (sources.paths) |source_path| {
        if (std.mem.startsWith(u8, source_path, "!")) continue;
        var names: std.ArrayList([]const u8) = .empty;
        if (std.mem.endsWith(u8, source_path, "/")) {
            const dir_path = try std.fs.path.join(arena, &.{ sources.root, source_path });
            var dir = Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true }) catch continue;
            defer dir.close(io);
            var walker = try dir.walk(arena);
            while (try walker.next(io)) |entry| {
                if (entry.kind != .file or !std.mem.endsWith(u8, entry.path, ".zig")) continue;
                try names.append(arena, try std.fs.path.join(arena, &.{ source_path, entry.path }));
            }
        } else try names.append(arena, source_path);
        for (names.items) |name| {
            if (!matchesAny(name, sources.paths)) continue;
            const lines = try readLines(arena, io, sources.root, name, name);
            for (lines, 0..) |line, i| {
                const function = functionName(line) orelse continue;
                const creates = createsError(lines, i);
                const entry = try definitions.getOrPut(arena, function);
                entry.value_ptr.* = creates or (entry.found_existing and entry.value_ptr.*);
            }
        }
    }
    return definitions;
}

fn functionName(line: []const u8) ?[]const u8 {
    var text = std.mem.trimStart(u8, line, " ");
    for ([_][]const u8{ "pub ", "inline ", "noinline ", "export " }) |prefix| {
        while (std.mem.startsWith(u8, text, prefix)) text = text[prefix.len..];
    }
    if (!std.mem.startsWith(u8, text, "fn ")) return null;
    const name = text["fn ".len..];
    const end = std.mem.indexOfScalar(u8, name, '(') orelse return null;
    return name[0..end];
}

fn createsError(lines: []const []const u8, start: usize) bool {
    const indent = lines[start].len - std.mem.trimStart(u8, lines[start], " ").len;
    var i = start;
    while (i < lines.len) : (i += 1) {
        const line = lines[i];
        for ([_][]const u8{ "error.", ".fail(", ".failParse(", ".todo(", "catch |", "return err" }) |origin| {
            if (std.mem.indexOf(u8, line, origin) != null) return true;
        }
        const closes_here = i == start and std.mem.endsWith(u8, line, "}");
        const closes = i > start and line.len > indent and line[indent] == '}' and std.mem.trimStart(u8, line, " ").len == line.len - indent;
        if (closes_here or closes) break;
    }
    return false;
}

/// Returns whether the branch is an error path that only an allocation failure or an error of a
/// callee takes. A function that creates no error itself holds no branch for its error returns.
fn failsOnlyByForwarding(file: SourceFile, definitions: Definitions, key: BranchKey, returns_at_once: bool, same_file: bool) bool {
    if (returns_at_once) return key.counter.line == 0 or !createsError(file.lines, key.counter.line - 1);
    if (tryForwards(definitions, std.mem.trim(u8, sourceLine(file, key.counter.line), " "))) return true;
    return same_file and tryForwards(definitions, std.mem.trim(u8, sourceLine(file, key.code.line), " "));
}

fn tryForwards(definitions: Definitions, text: []const u8) bool {
    const call = for ([_][]const u8{ "try ", " = try ", " => try " }, 0..) |marker, i| {
        if (i == 0 and std.mem.startsWith(u8, text, marker)) break text[marker.len..];
        if (i > 0) if (std.mem.indexOf(u8, text, marker)) |at| break text[at + marker.len ..];
    } else return false;
    return callForwards(definitions, call);
}

fn callForwards(definitions: Definitions, call: []const u8) bool {
    for ([_][]const u8{ "gpa", "arena", "allocator", ".append", ".put", ".alloc", ".dupe", ".ensure", ".toOwnedSlice", ".initEmpty", ".getOrPut", ".resize" }) |allocating| {
        if (std.mem.indexOf(u8, call, allocating) != null) return true;
    }
    const open = std.mem.indexOfScalar(u8, call, '(') orelse return false;
    const name = call[0..open];
    const callee = name[if (std.mem.lastIndexOfScalar(u8, name, '.')) |dot| dot + 1 else 0..];
    return if (definitions.get(callee)) |creates| !creates else false;
}

/// The code of an assertion, a failed allocation or an internal error is no branch to test.
fn ignoreUnrecoverable(branches: *Branches) void {
    const markers = [_][]const u8{ "unreachable", "@panic(", "\"out of memory\"", "internal error" };
    for (branches.list.values()) |*branch| {
        const text = sourceLine(branches.files.values()[branch.file], branch.line);
        for (markers) |marker| {
            if (std.mem.indexOf(u8, text, marker) != null) branch.ignored = true;
        }
    }
}

/// A line with a `try` after other code has a branch for the error path of the `try` and others.
/// When another branch of the line ran, the unrun error path of a forwarding call is no branch.
fn ignoreForwardedTries(branches: *Branches) void {
    var ran_lines: std.array_hash_map.Auto(u64, void) = .empty;
    defer ran_lines.deinit(std.heap.page_allocator);
    const definitions = branches.definitions.?;
    for (branches.list.values()) |branch| {
        if (branch.ran and !branch.error_return) ran_lines.put(std.heap.page_allocator, lineId(branch), {}) catch return;
    }
    for (branches.list.values()) |*branch| {
        if (branch.ran or branch.error_return or !ran_lines.contains(lineId(branch.*))) continue;
        const file = branches.files.values()[branch.file];
        const text = std.mem.trim(u8, sourceLine(file, branch.line), " ");
        const at = std.mem.indexOf(u8, text, "try ") orelse continue;
        if (callForwards(definitions, text[at + "try ".len ..])) branch.ignored = true;
    }
}

fn lineId(branch: Branch) u64 {
    return (@as(u64, branch.file) << 32) | branch.line;
}

fn wordAt(line: []const u8, column: u32, word: []const u8) bool {
    if (column == 0 or column > line.len) return false;
    const rest = line[column - 1 ..];
    return std.mem.startsWith(u8, rest, word) and
        (rest.len == word.len or !isIdentifierByte(rest[word.len]));
}

fn isIdentifierByte(byte: u8) bool {
    return std.ascii.isAlphanumeric(byte) or byte == '_' or byte == '@' or byte == '.';
}

fn writeReport(
    arena: Allocator,
    w: *Io.Writer,
    all_files: []const SourceFile,
    branches: []const Branch,
    source_paths: []const []const u8,
) !void {
    var files: std.ArrayList(SourceFile) = .empty;
    for (all_files) |file| if (file.branches.total != 0) try files.append(arena, file);
    std.mem.sortUnstable(SourceFile, files.items, {}, struct {
        fn lessThan(_: void, a: SourceFile, b: SourceFile) bool {
            return std.mem.lessThan(u8, a.name, b.name);
        }
    }.lessThan);

    try w.writeAll("# Branch coverage of");
    for (source_paths) |source_path| try w.print(" `{s}`", .{source_path});
    try w.writeAll("\n\n");
    try w.writeAll(
        \\The compiler places a coverage point at the entry of each function, at the start of each
        \\body of a runtime `if`, `while`, `for`, `and`, `or`, `catch` and `orelse`, and on the error
        \\path of each `try`. A `switch` prong has no point of its own. The program adds one to an
        \\8-bit counter at each point, and a point ran when the tracer read a nonzero counter. The
        \\tracer misses a point whose counter wrapped to zero between two reads.
        \\
        \\A branch is a point with its source location: the location of the counter and the
        \\location of the first code after the counter in the DWARF line table. Each instantiation
        \\of a generic function and each inlined copy of a function places its own point at the
        \\same source location, and the branch ran when any of these points ran. The file of a
        \\branch is the file of the counter. The line of a branch is the line of the code when the
        \\code is in that file, and the line of the counter otherwise. The error path of a `try`
        \\is a branch of its own when the path runs deferred statements before the return. The
        \\error paths of the `try`s of a function that return at once form one branch of the
        \\function, which ran when any of them ran. The error path of a `try` of a call that only
        \\forwards errors or allocates, and the error returns of a function that creates no error
        \\itself, are no branches. A line with `unreachable`, `@panic(`, an out of memory error or
        \\an internal error holds no branch either.
        \\
        \\| File | Branches run | Branches | % |
        \\|---|---:|---:|---:|
        \\
    );
    var total: Tally = .{};
    for (files.items) |file| {
        try w.print("| {s} | {d} | {d} | {d:.1} |\n", .{
            file.name,
            file.branches.ran,
            file.branches.total,
            file.branches.percent(),
        });
        total.addAll(file.branches);
    }
    try w.print("| **Total** | {d} | {d} | {d:.1} |\n", .{ total.ran, total.total, total.percent() });

    try w.writeAll("\n## Branches that did not run\n");
    for (files.items) |file| {
        var lines: std.ArrayList(u32) = .empty;
        for (branches) |branch| {
            if (branch.file != file.index or branch.ran or branch.ignored) continue;
            // The error return of a function lists the line of the function with the flag set.
            try lines.append(arena, branch.line * 2 + @intFromBool(branch.error_return));
        }
        std.mem.sortUnstable(u32, lines.items, {}, std.sort.asc(u32));
        try w.print("\n### {s}\n\n```\n", .{file.name});
        var previous: u32 = 0;
        for (lines.items) |flagged| {
            if (flagged == previous) continue;
            previous = flagged;
            const line = flagged / 2;
            try w.print("{d:>5}  {s}{s}\n", .{
                line,
                if (flagged % 2 == 1) "error return of " else "",
                sourceLine(file, line),
            });
        }
        try w.writeAll("```\n");
    }
}
