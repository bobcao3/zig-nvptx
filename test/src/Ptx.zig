b: *std.Build,
options: Options,
root_step: *std.Build.Step,
absent_checker: ?*std.Build.Step.Compile = null,
in_order_checker: ?*std.Build.Step.Compile = null,

pub const Options = struct {
    test_filters: []const []const u8,
    test_target_filters: []const []const u8,
    ptxas: ?[]const u8 = null,
};

pub const TestCase = struct {
    name: []const u8,
    source: []const u8 = "",
    /// The name of the root source file that holds `source`.
    file_name: []const u8 = "test.zig",
    /// A root source file in the repository, which the test compiles in place of `source`.
    path: ?[]const u8 = null,
    /// The CPU model and features, for example `sm_80+ptx85`.
    cpu: []const u8,
    matches: []const []const u8 = &.{},
    absent: []const []const u8 = &.{},
    /// Texts that must occur in this order, each after the end of the previous one.
    in_order: []const []const u8 = &.{},
    optimize: std.builtin.Optimize = .debug,
    /// The source of an NVPTX object that the test links with `source`.
    library: ?[]const u8 = null,
    /// The CPU model and features of the objects, when they differ from `cpu`.
    library_cpu: ?[]const u8 = null,
    /// The sources of more NVPTX objects that the test links with `source`.
    more_libraries: []const []const u8 = &.{},
    /// The source of a static library that the test links before the objects.
    archive: ?[]const u8 = null,
    /// The source of a static library that the test links after the objects.
    later_archive: ?[]const u8 = null,
    /// A line of the compile errors, where `/?/` matches any text. The compilation then fails and
    /// the test checks no PTX.
    error_message: ?[]const u8 = null,
};

fn addArchive(b: *std.Build, source: []const u8, name: []const u8, optimize: std.builtin.Optimize, target: std.Build.ResolvedTarget) *std.Build.Step.Compile {
    return b.addLibrary(.{
        .name = name,
        .linkage = .static,
        .root_module = b.createModule(.{
            .root_source_file = b.addWriteFiles().add("library.zig", source),
            .optimize = optimize,
            .target = target,
        }),
    });
}

pub fn addCase(self: *Ptx, case: TestCase) void {
    const b = self.b;
    const target = b.resolveTargetQuery(std.Target.Query.parse(.{
        .arch_os_abi = "nvptx64-cuda",
        .cpu_features = case.cpu,
    }) catch @panic("invalid cpu"));
    if (self.options.test_target_filters.len > 0) {
        const triple_txt = target.query.zigTriple(b.allocator) catch @panic("OOM");
        for (self.options.test_target_filters) |filter| {
            if (std.mem.find(u8, triple_txt, filter) != null) break;
        } else return;
    }

    const name = b.fmt("check ptx {s} ({s})", .{ case.name, case.cpu });
    if (self.options.test_filters.len > 0) {
        for (self.options.test_filters) |filter| {
            if (std.mem.find(u8, name, filter) != null) break;
        } else return;
    }

    const exe = b.addExecutable(.{
        .name = "test",
        .root_module = b.createModule(.{
            .root_source_file = if (case.path) |path| b.path(path) else b.addWriteFiles().add(case.file_name, case.source),
            .optimize = case.optimize,
            .target = target,
        }),
    });
    const library_target = if (case.library_cpu) |cpu| b.resolveTargetQuery(std.Target.Query.parse(.{
        .arch_os_abi = "nvptx64-cuda",
        .cpu_features = cpu,
    }) catch @panic("invalid cpu")) else target;
    if (case.archive) |archive| exe.root_module.linkLibrary(addArchive(b, archive, "archive", case.optimize, library_target));
    var libraries: std.ArrayList([]const u8) = .empty;
    if (case.library) |library| libraries.append(b.allocator, library) catch @panic("OOM");
    libraries.appendSlice(b.allocator, case.more_libraries) catch @panic("OOM");
    for (libraries.items, 0..) |library, i| {
        exe.root_module.addObject(b.addObject(.{
            .name = b.fmt("library{d}", .{i}),
            .root_module = b.createModule(.{
                .root_source_file = b.addWriteFiles().add("library.zig", library),
                .optimize = case.optimize,
                .target = library_target,
            }),
        }));
    }
    if (case.later_archive) |archive| exe.root_module.linkLibrary(addArchive(b, archive, "later_archive", case.optimize, library_target));
    if (case.error_message) |message| {
        exe.expect_errors = .{ .contains = message };
        exe.getEmittedBin().addStepDependencies(self.root_step);
        return;
    }
    const ptx = exe.getEmittedBin();

    const check = b.addCheckFile(ptx, .{ .expected_matches = case.matches });
    check.setName(name);
    self.root_step.dependOn(&check.step);

    if (self.options.ptxas) |ptxas| self.root_step.dependOn(&addPtxasCheck(b, ptxas, target.result, ptx).step);

    if (case.absent.len > 0) {
        const run = b.addRunArtifact(self.absentChecker());
        run.setName(b.fmt("{s} absent", .{name}));
        run.addFileArg(ptx);
        run.addArgs(case.absent);
        run.expectExitCode(0);
        self.root_step.dependOn(&run.step);
    }

    if (case.in_order.len > 0) {
        const run = b.addRunArtifact(self.inOrderChecker());
        run.setName(b.fmt("{s} in order", .{name}));
        run.addFileArg(ptx);
        run.addArgs(case.in_order);
        run.expectExitCode(0);
        self.root_step.dependOn(&run.step);
    }
}

fn inOrderChecker(self: *Ptx) *std.Build.Step.Compile {
    if (self.in_order_checker) |exe| return exe;
    const exe = self.b.addExecutable(.{
        .name = "check-in-order",
        .root_module = self.b.createModule(.{
            .root_source_file = self.b.path("test/src/check_in_order.zig"),
            .target = self.b.graph.host,
        }),
    });
    self.in_order_checker = exe;
    return exe;
}

fn absentChecker(self: *Ptx) *std.Build.Step.Compile {
    if (self.absent_checker) |exe| return exe;
    const exe = self.b.addExecutable(.{
        .name = "check-absent",
        .root_module = self.b.createModule(.{
            .root_source_file = self.b.path("test/src/check_absent.zig"),
            .target = self.b.graph.host,
        }),
    });
    self.absent_checker = exe;
    return exe;
}

/// Runs ptxas on `ptx` for the CPU model of `target`.
pub fn addPtxasCheck(b: *std.Build, ptxas: []const u8, target: std.Target, ptx: std.Build.LazyPath) *std.Build.Step.Run {
    const run = b.addSystemCommand(&.{ ptxas, b.fmt("-arch={s}", .{target.cpu.model.name}) });
    run.setName(b.fmt("ptxas {s}", .{target.cpu.model.name}));
    run.addFileArg(ptx);
    run.addArg("-o");
    _ = run.addOutputFileArg("test.cubin");
    return run;
}

const Ptx = @This();
const std = @import("std");
