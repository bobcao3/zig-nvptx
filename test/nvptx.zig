pub fn addCases(cases: *tests.PtxContext) void {
    // The lowest PTX version that ptxas accepts for each target.
    const versions = [_]struct { []const u8, []const u8 }{
        .{ "sm_75", "6.3" },
        .{ "sm_80", "7.0" },
        .{ "sm_90", "7.8" },
        .{ "sm_90a", "8.0" },
        .{ "sm_100a", "8.6" },
        .{ "sm_100f", "8.8" },
        .{ "sm_120a", "8.7" },
        .{ "sm_121", "8.8" },
    };
    for (versions) |entry| {
        const cpu, const version = entry;
        cases.addCase(.{
            .name = "minimum version",
            .source = empty_kernel,
            .cpu = cpu,
            .matches = &.{
                cases.b.fmt(".version {s}\n.target {s}\n.address_size 64\n", .{ version, cpu }),
            },
        });
    }

    cases.addCase(.{
        .name = "ptx feature raises version",
        .source = empty_kernel,
        .cpu = "sm_80+ptx85",
        .matches = &.{".version 8.5\n.target sm_80\n"},
    });

    cases.addCase(.{
        .name = "kernel directives",
        .source =
        \\export fn bounded(out: *u32) callconv(.{ .nvptx_kernel = .{
        \\    .max_work_group_size = .{ .x = 256 },
        \\    .min_work_groups_per_sm = 2,
        \\    .max_registers_per_work_item = 64,
        \\} }) void {
        \\    out.* = 1;
        \\}
        \\export fn fixed(out: *u32) callconv(.{ .nvptx_kernel = .{
        \\    .required_work_group_size = .{ .x = 32, .y = 4, .z = 2 },
        \\} }) void {
        \\    out.* = 2;
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{
            ".version 7.0\n",
            ".maxntid 256, 1, 1\n.minnctapersm 2\n.maxnreg 64\n{",
            ".reqntid 32, 4, 2\n{",
        },
        .absent = &.{ ".reqnctapercluster", ".maxntid 32" },
    });

    cases.addCase(.{
        .name = "cluster directive",
        .source =
        \\export fn cluster(out: *u32) callconv(.{ .nvptx_kernel = .{
        \\    .cluster_size = .{ .x = 2 },
        \\} }) void {
        \\    out.* = 3;
        \\}
        ,
        .cpu = "sm_90",
        .matches = &.{
            ".version 7.8\n",
            ".reqnctapercluster 2, 1, 1\n{",
        },
        .absent = &.{ ".maxntid", ".reqntid" },
    });

    cases.addCase(.{
        .name = "address of kernel parameter",
        .source =
        \\const TensorMap = extern struct { bytes: [128]u8 align(64) };
        \\export fn k(map: TensorMap, out: *u64) callconv(.kernel) void {
        \\    out.* = @intFromPtr(&map);
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{
            ".version 7.7\n",
            ".param .align 64 .b8 param_0[128],",
            "cvta.param.u64 %rd",
            ", param_0;\n",
        },
        .absent = &.{".local"},
    });

    cases.addCase(.{
        .name = "register list for an array output",
        .source =
        \\export fn k(out: *[64]f32, desc_a: u64, desc_b: u64) callconv(.kernel) void {
        \\    var acc: [64]f32 = @splat(0);
        \\    asm volatile (
        \\        \\wgmma.fence.sync.aligned;
        \\        \\wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 %[d], %[a], %[b],
        \\        \\    1, 1, 1, 0, 0;
        \\        \\wgmma.commit_group.sync.aligned;
        \\        \\wgmma.wait_group.sync.aligned 0;
        \\        : [d] "+f" (acc),
        \\        : [a] "l" (desc_a),
        \\          [b] "l" (desc_b),
        \\    );
        \\    out.* = acc;
        \\}
        ,
        .cpu = "sm_90a",
        .matches = &.{"wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {%f"},
        .absent = &.{".local"},
    });

    cases.addCase(.{
        .name = "register list for a vector operand",
        .source =
        \\export fn k(out: *@Vector(4, u32), in: *const @Vector(4, u32)) callconv(.kernel) void {
        \\    const v = asm volatile ("ld.global.v4.u32 %[r], [%[p]];"
        \\        : [r] "=r" (-> @Vector(4, u32)),
        \\        : [p] "l" (in),
        \\    );
        \\    asm volatile ("st.global.v4.u32 [%[p]], %[v];"
        \\        :
        \\        : [p] "l" (out),
        \\          [v] "r" (v +% @as(@Vector(4, u32), @splat(1))),
        \\    );
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{
            "ld.global.v4.u32 {%r",
            "}, [%rd",
            "st.global.v4.u32 [%rd",
            "], {%r",
        },
        .absent = &.{".local"},
    });

    cases.addCase(.{
        .name = "non-escaping allocas",
        .source =
        \\const Pair = struct { a: u32, b: u32 };
        \\fn helper(x: u32) u32 {
        \\    var pair: Pair = .{ .a = x, .b = x +% 1 };
        \\    var array: [2]u32 = .{ pair.a, pair.b };
        \\    pair.a *%= 3;
        \\    array[1] +%= pair.a;
        \\    return array[0] +% array[1];
        \\}
        \\export fn k(out: *u32) callconv(.kernel) void {
        \\    out.* = @call(.never_inline, helper, .{out.*});
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{".func (.param .b32 func_retval0) test_helper$"},
        .absent = &.{ ".local", "cvta.local" },
    });

    cases.addCase(.{
        .name = "locals that a byte-wise swap indexes stay in registers",
        .source =
        \\const std = @import("std");
        \\export fn k(out: *addrspace(.global) [2]u16, a: u16, b: u16) callconv(.kernel) void {
        \\    var x = a;
        \\    var y = b;
        \\    if (x < y) std.mem.swap(u16, &x, &y);
        \\    out.* = .{ x, y };
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{ "shr.u32 %r", "not.b32 %r" },
        .absent = &.{".local"},
    });

    cases.addCase(.{
        .name = "only exports are visible",
        .source =
        \\fn helper(x: u32) u32 {
        \\    return x *% 3;
        \\}
        \\export fn device(x: u32) u32 {
        \\    return @call(.never_inline, helper, .{x});
        \\}
        \\export fn kernel(out: *u32) callconv(.kernel) void {
        \\    out.* = device(out.*);
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{
            ".visible .func (.param .b32 func_retval0) device(",
            "\n.func (.param .b32 func_retval0) test_helper$",
            ".visible .entry kernel(",
        },
        .absent = &.{".visible .func (.param .b32 func_retval0) test_helper"},
    });

    cases.addCase(.{
        .name = "inlined calls",
        .source =
        \\fn add1(x: u32) u32 {
        \\    return x + 1;
        \\}
        \\fn twice(x: u32) u32 {
        \\    return add1(add1(x));
        \\}
        \\fn firstAbove(x: u32) u32 {
        \\    var acc = x;
        \\    for (0..8) |i| {
        \\        if (acc > 1000) return 7;
        \\        acc = acc *% 3 +% @as(u32, @intCast(i));
        \\    }
        \\    return acc;
        \\}
        \\const Pair = struct { a: u32, b: u64 };
        \\fn pairOf(x: u32) Pair {
        \\    return .{ .a = x + 1, .b = @as(u64, x) * 2 };
        \\}
        \\fn byteOf(b: u8) u8 {
        \\    return b +% 3;
        \\}
        \\fn differs(x: u32, flag: bool) bool {
        \\    return (x & 1 == 1) != flag;
        \\}
        \\export fn k(out: *[6]u32) callconv(.kernel) void {
        \\    const x = out[0];
        \\    out[1] = twice(x);
        \\    out[2] = firstAbove(x);
        \\    const p = pairOf(x);
        \\    out[3] = p.a +% @as(u32, @truncate(p.b));
        \\    out[4] = byteOf(@truncate(x));
        \\    out[5] = @intFromBool(differs(x, true));
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{".visible .entry k("},
        .absent = &.{ "call", ".func", ".param .b32 param" },
    });

    cases.addCase(.{
        .name = "calls that are not inlined",
        .source =
        \\fn fact(n: u32) u32 {
        \\    return if (n <= 1) 1 else n *% fact(n - 1);
        \\}
        \\noinline fn heavy(x: u32) u32 {
        \\    return x ^ 0x55;
        \\}
        \\fn small(x: u32) u32 {
        \\    return x << 1;
        \\}
        \\fn viaPointer(x: u32) callconv(.c) u32 {
        \\    return x + 100;
        \\}
        \\export fn device(x: u32) u32 {
        \\    return x *% 5;
        \\}
        \\export fn k(out: *[5]u32, f: *const fn (u32) callconv(.c) u32) callconv(.kernel) void {
        \\    const x = out[0];
        \\    out[0] = fact(x & 7);
        \\    out[1] = heavy(x);
        \\    out[2] = @call(.never_inline, small, .{x});
        \\    out[3] = f(x) +% viaPointer(x);
        \\    out[4] = device(x);
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{
            "call (retval0), test_fact$",
            "call (retval0), test_heavy$",
            "call (retval0), test_small$",
            ".callprototype (.param .b32 _) _ (.param .b32 _);\n\tcall (retval0), %rd",
            ".visible .func (.param .b32 func_retval0) device(",
        },
        .absent = &.{ "call (retval0), device", "viaPointer" },
    });

    cases.addCase(.{
        .name = "division by a power of two",
        .source =
        \\export fn k(out: *[4]u64, in: *const [2]i32) callconv(.kernel) void {
        \\    out[0] = out[0] % 16;
        \\    out[1] = out[1] / 512;
        \\    out[2] = @bitCast(@as(i64, @divTrunc(in[0], 8)));
        \\    out[3] = @bitCast(@as(i64, @mod(in[1], 4)));
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{ "and.b64 %rd3, %rd2, 15;", "shr.u64 %rd7, %rd6, 9;", "shr.s32 %r1, %r0, 31;\n\tand.b32 %r2, %r1, 7;\n\tadd.s32 %r3, %r0, %r2;\n\tshr.s32 %r4, %r3, 3;", "and.b32 %r6, %r5, 3;" },
        .absent = &.{ "div.", "rem." },
    });

    cases.addCase(.{
        .name = "unreferenced constants",
        .source =
        \\const builtin = @import("builtin");
        \\const table = [_]u32{ 7, 8, 9 };
        \\const unused_table = [_]u32{ 10, 11, 12 };
        \\export fn k(out: *u32) callconv(.kernel) void {
        \\    out.* = if (builtin.os.tag == .cuda) table[1] else 0;
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{ "mov.b32 %r", ", 8;\n" },
        .absent = &.{ "builtin", "table", ".global", ".const" },
    });

    cases.addCase(.{
        .name = "shared global",
        .source =
        \\var tile: [32]f32 addrspace(.shared) = undefined;
        \\export fn k(out: *f32) callconv(.kernel) void {
        \\    const i = @workItemId(0) % 32;
        \\    tile[i] = @floatFromInt(i);
        \\    out.* = tile[31 - i];
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{
            ".shared .align 4 .b8 test_tile$",
            "[128];\n",
            "st.shared.f32",
            "ld.shared.f32",
        },
        .absent = &.{ ".global", ".local" },
    });

    cases.addCase(.{
        .name = "dynamic shared memory",
        .source =
        \\extern var dynamic: [0]f32 align(16) addrspace(.shared);
        \\export fn k(out: *f32) callconv(.kernel) void {
        \\    const tile: [*]addrspace(.shared) f32 = &dynamic;
        \\    tile[@workItemId(0)] = out.*;
        \\    out.* = tile[@workItemId(0) ^ 1];
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{
            ".extern .shared .align 16 .b8 dynamic[];\n",
            "st.shared.f32",
            "ld.shared.f32",
        },
    });

    cases.addCase(.{
        .name = "large kernel parameter raises version",
        .source =
        \\export fn k(p: extern struct { a: [1087]u32 }, out: *u32) callconv(.kernel) void {
        \\    out.* = p.a[1086];
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{".version 8.1\n.target sm_80\n"},
    });

    // The text `%r200,` appears when a kernel defines more than 200 virtual `.b32` registers.
    cases.addCase(.{
        .name = "large kernel parameter stays in param space",
        .source =
        \\const Large = extern struct { values: [8000]i32 };
        \\export fn k(p: Large, out: *addrspace(.global) i32, i: u32) callconv(.kernel) void {
        \\    var sum: i32 = p.values[7999];
        \\    for (p.values) |value| sum +%= value;
        \\    out.* = sum +% p.values[i % 8000];
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{
            "mov.u64 %rd",
            ", param_0;\n",
            ", 31996;\n\tld.param.b32 %r",
        },
        .absent = &.{ ".local", "[param_0+31992]", "%r200," },
    });

    cases.addCase(.{
        .name = "copy of a large kernel parameter",
        .source =
        \\const Large = extern struct { values: [8000]i32 };
        \\export fn k(p: Large, out: *addrspace(.global) i32, i: u32) callconv(.kernel) void {
        \\    var copy = p;
        \\    copy.values[i % 8000] = 1;
        \\    out.* = copy.values[7999];
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{
            ".local .align 4 .b8 local_0[32000];\n",
            "mov.u64 %rd",
            ", param_0;\n",
            "ld.param.b32 %r",
        },
        .absent = &.{ "[param_0+31996]", "%r200," },
    });

    cases.addCase(.{
        .name = "kernel parameter of 132 bytes stays in param space",
        .source =
        \\export fn k(p: extern struct { values: [33]i32 }, out: *addrspace(.global) i32) callconv(.kernel) void {
        \\    var sum: i32 = 0;
        \\    for (p.values) |value| sum +%= value;
        \\    out.* = sum;
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{ "mov.u64 %rd", ", param_0;\n", "ld.param.b32 %r" },
        .absent = &.{ ".local", "[param_0+128]" },
    });

    cases.addCase(.{
        .name = "inline method that changes a struct through a pointer receiver",
        .source =
        \\const Cursor = struct {
        \\    index: u32,
        \\    step: u32,
        \\    inline fn advance(c: *Cursor) void {
        \\        c.index += c.step;
        \\    }
        \\};
        \\export fn k(out: [*]addrspace(.global) u32, n: u32) callconv(.kernel) void {
        \\    var cursor: Cursor = .{ .index = @workItemId(0), .step = n };
        \\    var i: u32 = 0;
        \\    while (i < n) : (i += 1) {
        \\        out[cursor.index] = i;
        \\        cursor.advance();
        \\    }
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{"st.global.b32 [%rd"},
        .absent = &.{".local"},
    });

    cases.addCase(.{
        .name = "element pointers of an inline for capture",
        .source =
        \\export fn k(out: [*]addrspace(.global) f32, x: f32) callconv(.kernel) void {
        \\    var a: [4][2]f32 = undefined;
        \\    inline for (&a, 0..) |*row, i| {
        \\        inline for (row, 0..) |*v, j| v.* = x * @as(f32, i * 2 + j);
        \\    }
        \\    inline for (a, 0..) |row, i| out[@workItemId(0) * 8 + i * 2] = row[0] + row[1];
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{"st.global.f32 [%rd"},
        .absent = &.{".local"},
    });

    cases.addCase(.{
        .name = "address of a local variable passed to an inlined call",
        .source =
        \\const Cursor = struct {
        \\    index: u32,
        \\    step: u32,
        \\    fn advance(c: *Cursor) void {
        \\        c.index += c.step;
        \\    }
        \\};
        \\export fn k(out: [*]addrspace(.global) u32, n: u32) callconv(.kernel) void {
        \\    var cursor: Cursor = .{ .index = @workItemId(0), .step = n };
        \\    var i: u32 = 0;
        \\    while (i < n) : (i += 1) {
        \\        out[cursor.index] = i;
        \\        cursor.advance();
        \\    }
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{"st.global.b32 [%rd"},
        .absent = &.{ ".local", "call" },
    });

    cases.addCase(.{
        .name = "constant array read at a runtime index",
        .source =
        \\const row: [16]u32 = .{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3 };
        \\export fn k(out: [*]addrspace(.global) u32, index: u32) callconv(.kernel) void {
        \\    out[@workItemId(0)] = row[index % 16];
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{ ".const .align", "ld.const.b32" },
        .absent = &.{".local"},
    });

    cases.addCase(.{
        .name = "copy loop of an aggregate above 128 bytes keeps ptxas from unrolling it",
        .source =
        \\const Block = extern struct { values: [256]u32 };
        \\export fn k(in: *addrspace(.global) const Block, out: *addrspace(.global) Block) callconv(.kernel) void {
        \\    const value = in.*;
        \\    out.* = value;
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{ ":\n\t.pragma \"nounroll\";\n\tsetp.ge.u64 %p", "ld.global.b32", "st.local.b32" },
    });

    cases.addCase(.{
        .name = "load through a pointer that chooses between local variables",
        .source =
        \\export fn k(a: [*]addrspace(.global) u32, b: [*]addrspace(.global) u32, c: bool, v: u32) callconv(.kernel) void {
        \\    const pa = a + @workItemId(0);
        \\    const pb = b + @workItemId(0) * 2;
        \\    (if (c) pa else pb)[0] = v;
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{"st.global.b32 [%rd"},
        .absent = &.{".local"},
    });

    const value_choices =
        \\export fn k(out: [*]addrspace(.global) u64, f: *addrspace(.global) f32, a: u64, b: u64, x: f32, c: bool, d: bool) callconv(.kernel) void {
        \\    out[0] = if (c) a else b;
        \\    f.* = if (d) @mulAdd(f32, x, -2.0, 1.0) else 1.0;
        \\    out[1] = @intFromBool(if (c) d else !d);
        \\}
    ;

    cases.addCase(.{
        .name = "a branch that chooses values becomes selp",
        .source = value_choices,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{ "selp.b64 %rd", "selp.b32 %f", "fma.rn.f32 %f", "and.pred %p", "or.pred %p" },
        .absent = &.{"bra "},
    });

    cases.addCase(.{
        .name = "Debug mode keeps the branch that chooses values",
        .source = value_choices,
        .cpu = "sm_80",
        .matches = &.{"bra "},
        .absent = &.{ "selp.b64 %rd", "selp.b32 %f" },
    });

    cases.addCase(.{
        .name = "a choice between variables whose address a call takes becomes selp after inlining",
        .source =
        \\fn bump(p: *u32) void {
        \\    p.* +%= 1;
        \\}
        \\export fn k(out: *addrspace(.global) u32, c: bool, v: u32) callconv(.kernel) void {
        \\    var a: u32 = v;
        \\    var b: u32 = v ^ 5;
        \\    bump(&a);
        \\    const p = if (c) &a else &b;
        \\    out.* = p.*;
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{"selp.b32 %r"},
        .absent = &.{ ".local", "bra " },
    });

    const out_of_bounds =
        \\export fn k(out: [*]addrspace(.global) u32, i: usize) callconv(.kernel) void {
        \\    const table = [4]u32{ 1, 2, 3, 4 };
        \\    out[0] = table[i];
        \\}
    ;

    cases.addCase(.{
        .name = "a failed safety check traps",
        .source = out_of_bounds,
        .cpu = "sm_80",
        .optimize = .safe,
        .matches = &.{"trap;"},
        .absent = &.{"__assertfail"},
    });

    cases.addCase(.{
        .name = "the panic handler of std.nvptx calls the assert function of the driver",
        .source = "pub const panic = @import(\"std\").nvptx.panic;\n" ++ out_of_bounds,
        .cpu = "sm_80",
        .optimize = .safe,
        .matches = &.{ ".extern .func __assertfail(", "call __assertfail, (param0, param1, param2, param3, param4);" },
    });

    cases.addCase(.{
        .name = "a branch whose arm loads stays a branch",
        .source =
        \\export fn k(out: *addrspace(.global) u32, p: [*]addrspace(.global) const u32, c: bool) callconv(.kernel) void {
        \\    out.* = if (c) p[1] else 0;
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{ "bra ", "ld.global.b32 %r" },
        .absent = &.{"selp.b32"},
    });

    cases.addCase(.{
        .name = "nested branches that choose values become selp",
        .source =
        \\export fn k(out: *addrspace(.global) i32, a: i32, b: i32, c: i32) callconv(.kernel) void {
        \\    out.* = if (a < b) (if (b < c) b else c) else (if (a < c) a else c);
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{"selp.b32 %r"},
        .absent = &.{"bra "},
    });

    cases.addCase(.{
        .name = "a division in an arm runs on both paths",
        .source =
        \\export fn k(out: *addrspace(.global) u32, n: u32, d: u32) callconv(.kernel) void {
        \\    out.* = if (d != 0) n / d else 0;
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{ "div.u32 %r", "selp.b32 %r" },
        .absent = &.{"bra "},
    });

    cases.addCase(.{
        .name = "kernel parameter of 128 bytes goes to registers",
        .source =
        \\export fn k(p: extern struct { values: [32]i32 }, out: *addrspace(.global) i32) callconv(.kernel) void {
        \\    var sum: i32 = 0;
        \\    for (p.values) |value| sum +%= value;
        \\    out.* = sum;
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{ "[param_0+124];\n", "selp.b32" },
        .absent = &.{ ".local", "cvta.local" },
    });

    cases.addCase(.{
        .name = "load of an aggregate above 128 bytes",
        .source =
        \\const Table = extern struct { values: [256]i32 };
        \\export fn k(in: *addrspace(.global) const Table, out: *addrspace(.global) i32, i: u32) callconv(.kernel) void {
        \\    const table = in.*;
        \\    out.* = table.values[i % 256] +% table.values[0];
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{ ".local .align 4 .b8 local_0[1024];\n", "ld.global.b32 %r", "cvta.local.u64 %rd2, local_0;", "ld.b32" },
        .absent = &.{ "%r200,", "ld.local" },
    });

    cases.addCase(.{
        .name = "load of an aggregate above 128 bytes in a release mode",
        .source =
        \\const Table = extern struct { values: [256]i32 };
        \\export fn k(in: *addrspace(.global) const Table, out: *addrspace(.global) i32, i: u32) callconv(.kernel) void {
        \\    const table = in.*;
        \\    out.* = table.values[i % 256] +% table.values[0];
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{ ".local .align 4 .b8 local_0[1024];\n", "ld.global.b32 %r", "st.local.b32", "ld.local.b32" },
        .absent = &.{ "%r200,", "cvta" },
    });

    cases.addCase(.{
        .name = "device function receives an aggregate above 128 bytes by reference",
        .source =
        \\const Table = extern struct { values: [256]i32 };
        \\const Small = extern struct { values: [32]i32 };
        \\noinline fn readAt(table: Table, i: u32) i32 {
        \\    return table.values[i % 256];
        \\}
        \\noinline fn readSmall(small: Small, i: u32) i32 {
        \\    return small.values[i % 32];
        \\}
        \\export fn k(p: Table, in: *addrspace(.global) const Small, out: *addrspace(.global) i32, i: u32) callconv(.kernel) void {
        \\    out.* = readAt(p, i) +% readSmall(in.*, i);
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{ ".param .b64 param_0,\n\t.param .b32 param_1", ".param .align 4 .b8 param_0[128],", "cvta.param.u64 %rd" },
        .absent = &.{ "param0[1024]", "%r200," },
    });

    cases.addCase(.{
        .name = "by-value capture of a shared array read at a runtime index",
        .source =
        \\var table: [2][128]u8 addrspace(.shared) = undefined;
        \\export fn k(out: *addrspace(.global) u32, index: u32) callconv(.kernel) void {
        \\    var words: [2]u32 = undefined;
        \\    inline for (&words, table) |*word, row| word.* = row[index % 128];
        \\    out.* = words[0] + words[1];
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{"ld.shared"},
        .absent = &.{ "%rs200,", "%r200," },
    });

    const entry_counts =
        \\const Entry = extern struct { counts: [4]u32, base: u32 };
        \\export fn k(out: [*]addrspace(.global) u32, in: [*]addrspace(.global) const Entry, n: u32) callconv(.kernel) void {
        \\    const entry = in[@workItemId(0)];
        \\    var sum: u32 = entry.base;
        \\    for (entry.counts) |count| {
        \\        var i: u32 = 0;
        \\        while (i < count) : (i += 1) sum += i * n;
        \\    }
        \\    out[@workItemId(0)] = sum;
        \\}
    ;

    inline for (.{ .debug, .fast }) |optimize| {
        cases.addCase(.{
            .name = "by-value array read at a runtime index in a loop takes no local memory in " ++ @tagName(optimize) ++ " mode",
            .source = entry_counts,
            .cpu = "sm_80",
            .optimize = optimize,
            .matches = &.{ "setp.eq.u", "selp.b32 %r" },
            .absent = &.{ ".local", "st.local", "ld.local" },
        });
    }

    cases.addCase(.{
        .name = "by-value array of more than 32 scalars read at a runtime index stays in local memory",
        .source =
        \\const Entry = extern struct { bytes: [33]u8 };
        \\export fn k(out: [*]addrspace(.global) u8, in: [*]addrspace(.global) const Entry, index: u32) callconv(.kernel) void {
        \\    const entry = in[@workItemId(0)];
        \\    out[@workItemId(0)] = for (entry.bytes, 0..) |byte, i| {
        \\        if (i == index) break byte;
        \\    } else 0;
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{".local"},
    });

    const loop_exits =
        \\export fn float_exit(out: [*]addrspace(.global) f32, limit: f32) callconv(.kernel) void {
        \\    var x: f32 = 0;
        \\    while (x < limit) x += 1;
        \\    out[@workItemId(0)] = x;
        \\}
        \\export fn int_exit(out: [*]addrspace(.global) u32, limit: u32) callconv(.kernel) void {
        \\    var x: u32 = 0;
        \\    while (x < limit) x += 3;
        \\    out[@workItemId(0)] = x;
        \\}
    ;

    cases.addCase(.{
        .name = "a loop exit branches on the complement of its comparison",
        .source = loop_exits,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{ "setp.geu.f32 %p0, %f1, %f0;\n\t@%p0 bra", "setp.ge.u32 %p0, %r1, %r0;\n\t@%p0 bra" },
        .absent = &.{ "@!%p", "setp.lt" },
    });

    cases.addCase(.{
        .name = "Debug mode keeps the negated loop exit",
        .source = loop_exits,
        .cpu = "sm_80",
        .matches = &.{ "setp.lt.f32 %p", "setp.lt.u32 %p", "@!%p" },
    });

    cases.addCase(.{
        .name = "a comparison with two users keeps the negated branch",
        .source =
        \\export fn k(out: [*]addrspace(.global) f32, flag: *addrspace(.global) u32, limit: f32) callconv(.kernel) void {
        \\    var x: f32 = 0;
        \\    const below = x < limit;
        \\    flag.* = @intFromBool(below);
        \\    while (x < limit) x += 1;
        \\    out[@workItemId(0)] = x;
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{"setp.geu.f32"},
    });

    cases.addCase(.{
        .name = "unsigned to wider signed integer",
        .source =
        \\export fn k(out: *i64, in: *const u32) callconv(.kernel) void {
        \\    out.* = @as(i38, in.*);
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{"cvt.u64.u32 %rd"},
        .absent = &.{"cvt.s64.u32"},
    });

    // The `and.b32` before the `cvt` avoids a ptxas crash. A comment in `intConvert` in
    // src/codegen/nvptx/codegen/int.zig describes the crash.
    cases.addCase(.{
        .name = "truncation to 16 bits masks the source",
        .source =
        \\export fn k(out: *addrspace(.global) bool, x: u32) callconv(.kernel) void {
        \\    const max = asm ("max.bf16x2 %[d], %[a], %[b];"
        \\        : [d] "=r" (-> u32),
        \\        : [a] "r" (x),
        \\          [b] "r" (@as(u32, 0x3F80_3F80)),
        \\    );
        \\    out.* = @as(u16, @truncate(max)) == 0x3F80;
        \\}
        ,
        .cpu = "sm_90",
        .matches = &.{"max.bf16x2 %r2, %r0, %r1;\n\t// end inline asm\n\tand.b32 %r3, %r2, 65535;\n\tcvt.u16.u32 %rs0, %r3;"},
    });

    cases.addCase(.{
        .name = "float divTrunc",
        .source =
        \\export fn k(out: *f32, in: *const [2]f32) callconv(.kernel) void {
        \\    out.* = @divTrunc(in[0], in[1]);
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{"div.rn.f32 %f2, %f0, %f1;\n\tcvt.rzi.f32.f32 %f3, %f2;\n"},
    });

    cases.addCase(.{
        .name = "bit cast of a vector of 4-bit integers",
        .source =
        \\export fn k(out: *u32, in: *const u32) callconv(.kernel) void {
        \\    const v: @Vector(8, u4) = @splat(@truncate(in.*));
        \\    out.* = @bitCast(v);
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{
            "shl.b64 %rd",
            ", 28;\n\tor.b64 %rd",
        },
    });

    cases.addCase(.{
        .name = "bit cast of a vector of bools",
        .source =
        \\export fn k(out: *u8, in: *const @Vector(8, u32)) callconv(.kernel) void {
        \\    out.* = @bitCast(in.* != @as(@Vector(8, u32), @splat(0)));
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{"selp.b64 %rd"},
        .absent = &.{"cvt.u64.pred"},
    });

    cases.addCase(.{
        .name = "bit cast to a vector of bools",
        .source =
        \\export fn k(out: *u32, in: *const u8) callconv(.kernel) void {
        \\    const mask: @Vector(8, bool) = @bitCast(in.*);
        \\    const one: @Vector(8, u32) = @splat(1);
        \\    out.* = @reduce(.Add, @select(u32, mask, one, one + one));
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{"setp.ne.u64 %p"},
        .absent = &.{"cvt.pred.u64"},
    });

    cases.addCase(.{
        .name = "bit cast of a negative integer to an f32 operand",
        .source =
        \\export fn k(out: *addrspace(.global) f32, a: f32, b: i32) callconv(.kernel) void {
        \\    const v: @Vector(2, f32) = @bitCast(@Vector(2, i32){ b, -1 });
        \\    const m = @max(@as(@Vector(2, f32), @splat(a)), v);
        \\    out.* = m[0] + m[1];
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{", 0fFFFFFFFF;"},
    });

    cases.addCase(.{
        .name = "string constant",
        .source =
        \\export fn k(out: *[*:0]const u8) callconv(.kernel) void {
        \\    out.* = "hello";
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{
            ".b8 __anon_",
            "[6] = {104,101,108,108,111,0};\n",
            "cvta.global.u64 %rd",
            ", __anon_",
        },
        .absent = &.{".visible .global"},
    });

    cases.addCase(.{
        .name = "std.nvptx barriers and fences",
        .source =
        \\const nvptx = @import("std").nvptx;
        \\const Barrier = enum(u4) { work_group = 0, producer = 1 };
        \\export fn k(out: *u32) callconv(.kernel) void {
        \\    out.* = 1;
        \\    nvptx.barrierSyncAligned(.work_group, null);
        \\    nvptx.barrierSyncAligned(Barrier.producer, 64);
        \\    nvptx.barWarpSync(.all);
        \\    nvptx.fence(.seq_cst, .gpu);
        \\    nvptx.fence(.acq_rel, .cta);
        \\    nvptx.fence(.seq_cst, .sys);
        \\    nvptx.fence(.seq_cst, .cluster);
        \\}
        ,
        .cpu = "sm_90",
        .optimize = .fast,
        .matches = &.{
            "bar.warp.sync -1;\n\tbarrier.sync.aligned %r",
            "barrier.sync.aligned %r",
            ", 64;\n",
            "bar.warp.sync %r",
            "fence.sc.gpu;\n",
            "fence.acq_rel.cta;\n",
            "fence.sc.sys;\n",
            "fence.sc.cluster;\n",
        },
    });

    cases.addCase(.{
        .name = "std.nvptx warp functions",
        .source =
        \\const nvptx = @import("std").nvptx;
        \\export fn k(out: [*]u32, f: *f32) callconv(.kernel) void {
        \\    const lane = nvptx.laneid();
        \\    out[0] = nvptx.shflSync(.down, u32, lane, 3, comptime .fromWidth(.down, 16), .all);
        \\    f.* = nvptx.shflSync(.bfly, f32, f.*, 1, comptime .fromWidth(.bfly, 32), .all);
        \\    out[1] = nvptx.voteBallotSync(lane > 3, .all);
        \\    out[2] = @intFromBool(nvptx.voteSync(.uni, lane < 8, .all));
        \\    out[3] = @bitCast(nvptx.reduxSync(.max, i32, @intCast(lane), .all));
        \\    out[4] = nvptx.reduxSync(.xor, u32, lane, .all);
        \\    out[5] = nvptx.activemask();
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{
            "mov.u32 %r0, %laneid;",
            ", 3, 4127, 4294967295;",
            "shfl.sync.down.b32 %r",
            ", 1, 31, 4294967295;",
            "shfl.sync.bfly.b32 %f",
            "vote.sync.ballot.b32 %r",
            "vote.sync.uni.pred %p",
            "redux.sync.max.s32 %r",
            "redux.sync.xor.b32 %r",
            "activemask.b32 %r",
        },
    });

    cases.addCase(.{
        .name = "std.nvptx special registers and approximate math",
        .source =
        \\const nvptx = @import("std").nvptx;
        \\export fn k(out: [*]u32, f: [*]f32) callconv(.kernel) void {
        \\    out[0] = nvptx.nctaid(2) +% nvptx.clock() +% @as(u32, @truncate(nvptx.clock64()));
        \\    f[0] = nvptx.ex2Approx(f[0]) + nvptx.lg2Approx(f[1]) + nvptx.sinApprox(f[2]);
        \\    f[1] = nvptx.cosApprox(f[3]) + nvptx.rcpApprox(f[4]) + nvptx.rsqrtApprox(f[5]);
        \\    f[2] = nvptx.divApprox(f[6], f[7]);
        \\}
        ,
        .cpu = "sm_75",
        .optimize = .fast,
        .matches = &.{
            "mov.u32 %r0, %nctaid.z;",
            "mov.u32 %r1, %clock;",
            "mov.u64 %rd2, %clock64;",
            "ex2.approx.f32 %f",
            "lg2.approx.f32 %f",
            "sin.approx.f32 %f",
            "cos.approx.f32 %f",
            "rcp.approx.f32 %f",
            "rsqrt.approx.f32 %f",
            "div.approx.f32 %f",
        },
    });

    inline for (.{ .debug, .fast }) |optimize| {
        inline for (.{ "sm_89+ptx81", "sm_90", "sm_121a" }) |cpu| {
            cases.addCase(.{
                .name = "std.nvptx FP8 conversions in " ++ @tagName(optimize) ++ " mode",
                .source =
                \\const nv = @import("std").nvptx;
                \\export fn k(out: [*]addrspace(.global) u16, out4: [*]addrspace(.global) u32, x0: f32, x1: f32, x2: f32, x3: f32) callconv(.kernel) void {
                \\    out[0] = @backingInt(nv.E4m3x2.fromF32(x0, x1));
                \\    out[1] = @backingInt(nv.E5m2x2.fromF32(x0, x1));
                \\    out4[0] = @backingInt(nv.E4m3x4.fromF32(x0, x1, x2, x3));
                \\    out4[1] = @backingInt(nv.E5m2x4.fromF32(x0, x1, x2, x3));
                \\}
                ,
                .cpu = cpu,
                .optimize = optimize,
                .matches = &.{
                    "cvt.rn.satfinite.e4m3x2.f32 %rs",
                    "cvt.rn.satfinite.e5m2x2.f32 %rs",
                },
                .absent = &.{"cvt.rs"},
            });
        }
        inline for (.{ "sm_100a+ptx87", "sm_103a" }) |cpu| {
            cases.addCase(.{
                .name = "std.nvptx FP8x4 stochastic conversions in " ++ @tagName(optimize) ++ " mode",
                .source =
                \\const nv = @import("std").nvptx;
                \\export fn k(out: [*]addrspace(.global) u32, x0: f32, x1: f32, x2: f32, x3: f32, random_bits: u32) callconv(.kernel) void {
                \\    out[0] = @backingInt(nv.E4m3x4.fromF32Stochastic(x0, x1, x2, x3, random_bits));
                \\    out[1] = @backingInt(nv.E5m2x4.fromF32Stochastic(x0, x1, x2, x3, random_bits));
                \\}
                ,
                .cpu = cpu,
                .optimize = optimize,
                .matches = &.{
                    "cvt.rs.satfinite.e4m3x4.f32 %r",
                    "cvt.rs.satfinite.e5m2x4.f32 %r",
                },
            });
        }
        inline for (.{ "sm_89+ptx84", "sm_121a" }) |cpu| {
            cases.addCase(.{
                .name = "std.nvptx FP8 mma.sync in " ++ @tagName(optimize) ++ " mode",
                .source = fp8_mma,
                .cpu = cpu,
                .optimize = optimize,
                .matches = &.{
                    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {",
                    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32 {",
                    "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e4m3.f32 {",
                    "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 {",
                },
            });
        }
        cases.addCase(.{
            .name = "std.nvptx FP8 half accumulators and smaller shape in " ++ @tagName(optimize) ++ " mode",
            .source =
            \\const nv = @import("std").nvptx;
            \\export fn small(out: *@Vector(4, f32), a: @Vector(2, u32), b: @Vector(1, u32), c: @Vector(4, f32)) callconv(.kernel) void {
            \\    out.* = nv.mmaSync(.{ .shape = .m16n8k16, .element = .e5m2, .b_element = .e4m3, .accumulator = .f32 }, a, b, c);
            \\}
            \\export fn smallHalf(out: *@Vector(2, u32), a: @Vector(2, u32), b: @Vector(1, u32), c: @Vector(2, u32)) callconv(.kernel) void {
            \\    out.* = nv.mmaSync(.{ .shape = .m16n8k16, .element = .e4m3, .b_element = .e5m2, .accumulator = .f16 }, a, b, c);
            \\}
            \\export fn half(out: *@Vector(2, u32), a: @Vector(4, u32), b: @Vector(2, u32), c: @Vector(2, u32)) callconv(.kernel) void {
            \\    out.* = nv.mmaSync(.{ .shape = .m16n8k32, .element = .e5m2, .accumulator = .f16 }, a, b, c);
            \\}
            ,
            .cpu = "sm_89+ptx87",
            .optimize = optimize,
            .matches = &.{
                "mma.sync.aligned.m16n8k16.row.col.f32.e5m2.e4m3.f32 {",
                "mma.sync.aligned.m16n8k16.row.col.f16.e4m3.e5m2.f16 {",
                "mma.sync.aligned.m16n8k32.row.col.f16.e5m2.e5m2.f16 {",
            },
        });
        cases.addCase(.{
            .name = "std.nvptx FP8 block-scaled unrolled accumulation in " ++ @tagName(optimize) ++ " mode",
            .source =
            \\const nv = @import("std").nvptx;
            \\export fn k(out: *@Vector(4, f32), a: @Vector(4, u32), b: @Vector(2, u32), c: @Vector(4, f32), scale_a: u32, scale_b: u32) callconv(.kernel) void {
            \\    @setEvalBranchQuota(20_000);
            \\    var sum = c;
            \\    inline for (0..256) |i| {
            \\        sum = nv.mmaSyncBlockScale(.{ .shape = .m16n8k32, .element = .e4m3, .accumulator = .f32 }, .{ .a_byte = i % 4, .b_byte = i / 64 }, a, b, sum, scale_a, scale_b);
            \\    }
            \\    out.* = sum;
            \\}
            ,
            .cpu = "sm_120f",
            .optimize = optimize,
            .matches = &.{"mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {"},
        });
        inline for (.{ "sm_120a", "sm_120f", "sm_121a" }) |cpu| {
            cases.addCase(.{
                .name = "std.nvptx FP8 block-scaled mma.sync in " ++ @tagName(optimize) ++ " mode",
                .source = fp8_block_scaled_mma,
                .cpu = cpu,
                .optimize = optimize,
                .matches = &.{
                    "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 {",
                    "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e5m2.f32.ue8m0 {",
                    "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e5m2.e4m3.f32.ue8m0 {",
                    "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e5m2.e5m2.f32.ue8m0 {",
                    ", {3, 1}, ",
                    ", {2, 3};",
                    ", {0, 0};",
                },
            });
        }
    }

    cases.addCase(.{
        .name = "std.nvptx mma.sync",
        .source =
        \\const nv = @import("std").nvptx;
        \\export fn k(out: [*]addrspace(.global) u32) callconv(.kernel) void {
        \\    const bf16: nv.matrix.MmaOptions = .{ .shape = .m16n8k16, .accumulator = .f32, .element = .bf16 };
        \\    const d = nv.mmaSync(bf16, @as(@Vector(4, u32), @splat(out[0])), @as(@Vector(2, u32), @splat(out[1])), @as(@Vector(4, f32), @splat(0)));
        \\    out[2] = @bitCast(d[0]);
        \\    const int8: nv.matrix.MmaOptions = .{
        \\        .shape = nv.matrix.MmaSyncShape.fromMnk(16, 8, 32),
        \\        .accumulator = .s32,
        \\        .element = .s8,
        \\        .b_element = .u8,
        \\        .satfinite = true,
        \\    };
        \\    const e = nv.mmaSync(int8, @as(@Vector(4, u32), @splat(out[0])), @as(@Vector(2, u32), @splat(out[1])), @as(@Vector(4, i32), @splat(0)));
        \\    out[3] = @bitCast(e[0]);
        \\    const f64_options: nv.matrix.MmaOptions = .{ .shape = .m8n8k4, .accumulator = .f64, .element = .f64 };
        \\    const f = nv.mmaSync(f64_options, @as(@Vector(1, f64), .{1}), @as(@Vector(1, f64), .{2}), @as(@Vector(2, f64), @splat(0)));
        \\    out[4] = @intFromFloat(f[0]);
        \\    const bits: nv.matrix.MmaOptions = .{ .shape = .m8n8k128, .accumulator = .s32, .element = .b1, .bit_operation = .xor };
        \\    const g = nv.mmaSync(bits, @as(@Vector(1, u32), .{out[0]}), @as(@Vector(1, u32), .{out[1]}), @as(@Vector(2, i32), @splat(0)));
        \\    out[5] = @bitCast(g[0]);
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {",
            "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.u8.s32 {",
            "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {",
            "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc {",
        },
    });

    cases.addCase(.{
        .name = "std.nvptx ldmatrix and movmatrix",
        .source =
        \\const nv = @import("std").nvptx;
        \\var tile: [128]u32 align(16) addrspace(.shared) = undefined;
        \\export fn k(out: [*]addrspace(.global) u32) callconv(.kernel) void {
        \\    const row = &tile[nv.laneid() * 4];
        \\    const x1 = nv.ldmatrix(.x1, false, row);
        \\    const x2 = nv.ldmatrix(.x2, true, row);
        \\    const x4 = nv.ldmatrix(.x4, false, row);
        \\    out[0] = x1[0] +% x2[1] +% x4[3] +% nv.movmatrix(x4[0]);
        \\}
        ,
        .cpu = "sm_80+ptx78",
        .optimize = .fast,
        .matches = &.{
            "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%r",
            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%r",
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r",
            "movmatrix.sync.aligned.m8n8.trans.b16 %r",
        },
    });

    cases.addCase(.{
        .name = "std.nvptx more special registers",
        .source =
        \\const nvptx = @import("std").nvptx;
        \\export fn k(out: [*]u32, wide: [*]u64) callconv(.kernel) void {
        \\    out[0] = nvptx.smid() +% nvptx.nsmid() +% nvptx.warpid() +% nvptx.nwarpid();
        \\    wide[0] = nvptx.gridid() +% nvptx.globaltimer();
        \\    out[1] = nvptx.lanemaskEq() +% nvptx.lanemaskLe() +% nvptx.lanemaskLt();
        \\    out[2] = nvptx.lanemaskGe() +% nvptx.lanemaskGt() +% nvptx.clockHi();
        \\    out[3] = nvptx.globaltimerLo() +% nvptx.globaltimerHi() +% nvptx.envreg(5);
        \\    out[4] = nvptx.totalSmemSize() +% nvptx.dynamicSmemSize();
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{
            "mov.u32 %r0, %smid;",
            "mov.u32 %r1, %nsmid;",
            "mov.u32 %r3, %warpid;",
            "mov.u32 %r5, %nwarpid;",
            "mov.u64 %rd2, %gridid;",
            "mov.u64 %rd3, %globaltimer;",
            "mov.u32 %r7, %lanemask_eq;",
            "mov.u32 %r8, %lanemask_le;",
            "mov.u32 %r10, %lanemask_lt;",
            "mov.u32 %r12, %lanemask_ge;",
            "mov.u32 %r13, %lanemask_gt;",
            "mov.u32 %r15, %clock_hi;",
            "mov.u32 %r17, %globaltimer_lo;",
            "mov.u32 %r18, %globaltimer_hi;",
            "mov.u32 %r20, %envreg5;",
            "mov.u32 %r22, %total_smem_size;",
            "mov.u32 %r23, %dynamic_smem_size;",
        },
    });

    cases.addCase(.{
        .name = "std.nvptx reserved shared memory registers",
        .source =
        \\const nvptx = @import("std").nvptx;
        \\export fn k(out: [*]u32) callconv(.kernel) void {
        \\    out[0] = nvptx.reservedSmemOffsetBegin() +% nvptx.reservedSmemOffsetEnd();
        \\    out[1] = nvptx.reservedSmemOffsetCap() +% nvptx.reservedSmemOffset(1);
        \\}
        ,
        .cpu = "sm_80+ptx76",
        .optimize = .fast,
        .matches = &.{
            ", %reserved_smem_offset_begin;",
            ", %reserved_smem_offset_end;",
            ", %reserved_smem_offset_cap;",
            ", %reserved_smem_offset_1;",
        },
    });

    cases.addCase(.{
        .name = "std.nvptx execution control and driver functions",
        .source =
        \\const nvptx = @import("std").nvptx;
        \\export fn k(out: [*]u32, trapping: bool) callconv(.kernel) void {
        \\    nvptx.nanosleep(out[0]);
        \\    @breakpoint();
        \\    nvptx.pmevent(3);
        \\    nvptx.pmeventMask(0x5);
        \\    const block: [*]u32 = @ptrCast(@alignCast(nvptx.malloc(64).?));
        \\    block[0] = 1;
        \\    out[1] = block[0];
        \\    nvptx.free(block);
        \\    _ = nvptx.printf("%d %f\n", .{ @as(i32, 1), @as(f32, 2) });
        \\    if (trapping) @trap();
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{
            "nanosleep.u32 %r",
            "pmevent 3;",
            "pmevent.mask 5;",
            ".extern .func (.param .b64 func_retval0) malloc(",
            ".extern .func free(",
            ".extern .func (.param .b32 func_retval0) vprintf(",
        },
    });

    cases.addCase(.{
        .name = "std.nvptx.math conversions",
        .source =
        \\const nvptx = @import("std").nvptx;
        \\const math = nvptx.math;
        \\export fn k(out: [*]u32, f: [*]f32, h: [*]f16) callconv(.kernel) void {
        \\    f[0] = math.cvt(f32, f32, f[1], .{ .rounding = .nearest_even });
        \\    out[0] = @bitCast(math.cvt(i32, f32, f[2], .{
        \\        .rounding = .toward_negative_infinity,
        \\        .saturate = true,
        \\    }));
        \\    f[3] = math.cvt(f32, i32, @as(i32, @bitCast(out[1])), .{ .rounding = .toward_positive_infinity });
        \\    h[0] = math.cvt(f16, f32, f[4], .{ .rounding = .nearest_even, .relu = true });
        \\    f[5] = math.cvt(f32, f32, f[6], .{ .saturate = true });
        \\    out[2] = @backingInt(nvptx.Tf32.fromF32(f[7]));
        \\    out[3] = @bitCast(math.f16x2FromF32Relu(f[8], f[9]));
        \\    out[4] = @backingInt(nvptx.Bf16x2.fromF32(f[10], f[11]));
        \\    out[5] = nvptx.bf16FromF32(f[12]);
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{
            "cvt.rni.f32.f32 %r",
            "cvt.rmi.sat.s32.f32 %r",
            "cvt.rp.f32.s32 %r",
            "cvt.rn.relu.f16.f32 %rs",
            "cvt.sat.f32.f32 %r",
            "cvt.rna.tf32.f32 %r",
            "cvt.rn.relu.f16x2.f32 %r",
            "cvt.rn.bf16x2.f32 %r",
            "cvt.rn.bf16.f32 %rs",
        },
    });

    cases.addCase(.{
        .name = "std.nvptx.math float arithmetic",
        .source =
        \\const math = @import("std").nvptx.math;
        \\export fn k(f: [*]f32, d: [*]f64) callconv(.kernel) void {
        \\    f[0] = math.addF32(.toward_zero, .{ .ftz = true }, f[1], f[2]);
        \\    f[3] = math.fmaF32(.toward_positive_infinity, .{ .saturate = true }, f[4], f[5], f[6]);
        \\    f[7] = math.div(f32, .toward_negative_infinity, f[8], f[9]);
        \\    f[10] = math.sqrtF32(.toward_zero, true, f[11]);
        \\    d[0] = math.rcp(f64, .toward_positive_infinity, d[1]);
        \\    d[2] = math.mul(f64, .nearest_even, d[3], d[4]);
        \\    f[12] = math.tanhApprox(f32, f[13]);
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{
            "add.rz.ftz.f32 %f",
            "fma.rp.sat.f32 %f",
            "div.rm.f32 %f",
            "sqrt.rz.ftz.f32 %f",
            "rcp.rp.f64 %fd",
            "mul.rn.f64 %fd",
            "tanh.approx.f32 %f",
        },
    });

    cases.addCase(.{
        .name = "std.nvptx packed math",
        .source =
        \\const nvptx = @import("std").nvptx;
        \\const math = nvptx.math;
        \\export fn k(out: [*]u32) callconv(.kernel) void {
        \\    const x: nvptx.Bf16x2 = @fromBackingInt(out[0]);
        \\    const y: nvptx.Bf16x2 = @fromBackingInt(out[1]);
        \\    out[2] = @backingInt(x.fma(y, x));
        \\    out[3] = @backingInt(x.minNan(y).maxNan(x).neg().abs());
        \\    const h: @Vector(2, f16) = @bitCast(out[5]);
        \\    out[6] = @bitCast(math.ex2Approx(@Vector(2, f16), math.tanhApprox(@Vector(2, f16), math.fmaRelu(h, h, h))));
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{
            "fma.rn.bf16x2 %r",
            "min.NaN.bf16x2 %r",
            "max.NaN.bf16x2 %r",
            "neg.bf16x2 %r",
            "abs.bf16x2 %r",
            "fma.rn.relu.f16x2 %r",
            "tanh.approx.f16x2 %r",
            "ex2.approx.f16x2 %r",
        },
    });

    cases.addCase(.{
        .name = "std.nvptx.math integer operations",
        .source =
        \\const nvptx = @import("std").nvptx;
        \\const math = nvptx.math;
        \\export fn k(out: [*]u32, wide: [*]u64) callconv(.kernel) void {
        \\    out[0] = math.mul24Lo(u32, out[1], out[2]);
        \\    out[3] = math.mad24Lo(u32, out[1], out[2], out[3]);
        \\    out[4] = math.mulHi(u32, out[1], out[2]);
        \\    wide[0] = math.mulWide(u32, out[1], out[2]);
        \\    out[5] = math.sad(u32, out[1], out[2], out[3]);
        \\    out[6] = math.bfe(u32, out[1], out[2], out[3]);
        \\    out[7] = math.bfi(u32, out[1], out[2], out[3], out[4]);
        \\    out[8] = math.prmt(.replicate_8, out[1], out[2], out[3]);
        \\    out[9] = math.lop3(0x96, out[1], out[2], out[3]);
        \\    out[10] = math.shf(.left, .clamp, out[1], out[2], out[3]);
        \\    out[11] = @bitCast(nvptx.dp4a(.s32, .u32, out[1], out[2], 0));
        \\    out[12] = @bitCast(nvptx.dp2a(.hi, .s32, .s32, out[1], out[2], @bitCast(out[3])));
        \\    out[13] = math.fns(out[1], out[2], @bitCast(out[3]));
        \\    out[14] = @intFromBool(math.testp(.subnormal, f32, @bitCast(out[4])));
        \\    out[15] = @bitCast(math.copysign(f32, @bitCast(out[5]), @bitCast(out[6])));
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{
            "mul24.lo.u32 %r",
            "mad24.lo.u32 %r",
            "mul.hi.u32 %r",
            "mul.wide.u32 %rd",
            "sad.u32 %r",
            "bfe.u32 %r",
            "bfi.b32 %r",
            "prmt.b32.rc8 %r",
            "lop3.b32 %r",
            ", 150;",
            "shf.l.clamp.b32 %r",
            "dp4a.s32.u32 %r",
            "dp2a.hi.s32.s32 %r",
            "fns.b32 %r",
            "testp.subnormal.f32 %p",
            "copysign.f32 %f",
        },
    });

    cases.addCase(.{
        .name = "std.nvptx.math bit mask and extension",
        .source =
        \\const math = @import("std").nvptx.math;
        \\export fn k(out: [*]u32) callconv(.kernel) void {
        \\    out[0] = math.bmsk(.clamp, out[1], out[2]);
        \\    out[3] = @bitCast(math.szext(.wrap, i32, @as(i32, @bitCast(out[1])), out[2]));
        \\}
        ,
        .cpu = "sm_80+ptx76",
        .optimize = .fast,
        .matches = &.{
            "bmsk.clamp.b32 %r",
            "szext.wrap.s32 %r",
        },
    });

    cases.addCase(.{
        .name = "std.nvptx scoped atomics",
        .source =
        \\const nvptx = @import("std").nvptx;
        \\var shared_word: u32 addrspace(.shared) = undefined;
        \\export fn k(out: [*]addrspace(.global) u32, half: *addrspace(.global) f16) callconv(.kernel) void {
        \\    out[0] = nvptx.atom(.cta, .monotonic, .add, &out[1], 1);
        \\    out[2] = nvptx.atom(.sys, .acq_rel, .max, &out[3], 2);
        \\    out[4] = nvptx.atom(.gpu, .acquire, .inc, &out[5], 9);
        \\    out[6] = nvptx.atom(.gpu, .release, .dec, &out[7], 9);
        \\    out[8] = nvptx.atomCas(.cta, .acq_rel, &out[9], 3, 4);
        \\    out[10] = nvptx.atom(.cta, .monotonic, .exch, &shared_word, 5);
        \\    nvptx.red(.gpu, .release, .@"or", &out[11], 6);
        \\    _ = nvptx.atom(.gpu, .monotonic, .add, half, 1);
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{
            "atom.relaxed.cta.global.add.u32 %r",
            "atom.acq_rel.sys.global.max.u32 %r",
            "atom.acquire.gpu.global.inc.u32 %r",
            "atom.release.gpu.global.dec.u32 %r",
            "atom.acq_rel.cta.global.cas.b32 %r",
            "atom.relaxed.cta.shared.exch.b32 %r",
            "red.release.gpu.global.or.b32 [%rd",
            "atom.relaxed.gpu.global.add.noftz.f16 %rs",
        },
    });

    cases.addCase(.{
        .name = "std.nvptx match and barrier reductions",
        .source =
        \\const nvptx = @import("std").nvptx;
        \\export fn k(out: [*]u32) callconv(.kernel) void {
        \\    const lane = nvptx.laneid();
        \\    out[0] = nvptx.matchAnySync(u32, lane, .all);
        \\    out[1] = nvptx.matchAnySync(u64, lane, .all);
        \\    out[2] = nvptx.matchAllSync(u32, lane, .all).mask;
        \\    out[3] = nvptx.barrierRedPopcAligned(.work_group, null, lane > 3);
        \\    out[4] = @intFromBool(nvptx.barrierRedAligned(.@"and", .work_group, 128, lane > 3));
        \\    out[5] = @intFromBool(nvptx.barrierRedAligned(.@"or", .work_group, null, lane > 3));
        \\    nvptx.barrierArriveAligned(@as(nvptx.NamedBarrier, .work_group), 64);
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{
            "match.any.sync.b32 %r",
            "match.any.sync.b64 %r",
            "match.all.sync.b32 %r",
            "|%p",
            "bar.warp.sync -1;\n\tbarrier.red.popc.aligned.u32 %r",
            "barrier.red.and.aligned.pred %p",
            ", 128, %p",
            "barrier.red.or.aligned.pred %p",
            "barrier.arrive.aligned %r",
            ", 64;\n",
        },
    });

    cases.addCase(.{
        .name = "std.nvptx cp.async and mbarrier",
        .source =
        \\const nvptx = @import("std").nvptx;
        \\var tile: [32]u32 align(16) addrspace(.shared) = undefined;
        \\var wide: [4]u32 align(16) addrspace(.shared) = undefined;
        \\var barrier: nvptx.Mbarrier addrspace(.shared) = undefined;
        \\export fn k(src: [*]addrspace(.global) u32, out: [*]u32) callconv(.kernel) void {
        \\    nvptx.cp_async.copy(.cache_all_levels, &tile[0], &src[0]);
        \\    const source: *addrspace(.global) [4]u32 = @ptrCast(src + 4);
        \\    nvptx.cp_async.copy(.cache_global_level, &wide, source);
        \\    nvptx.cp_async.copyZeroFill(.cache_all_levels, &tile[1], &src[1], out[0]);
        \\    nvptx.cp_async.commitGroup();
        \\    nvptx.cp_async.waitGroup(1);
        \\    nvptx.cp_async.waitAll();
        \\    barrier.init(32);
        \\    nvptx.cp_async.mbarrierArrive(&barrier);
        \\    nvptx.cp_async.mbarrierArriveNoinc(&barrier);
        \\    const token = barrier.arrive();
        \\    const other = barrier.arriveNoComplete(2);
        \\    _ = barrier.arriveDrop();
        \\    _ = barrier.arriveDropNoComplete(2);
        \\    out[1] = @intFromBool(barrier.testWait(token)) + @intFromBool(barrier.testWaitParity(1));
        \\    out[2] = other.pendingCount();
        \\    barrier.inval();
        \\}
        ,
        .cpu = "sm_80+ptx71",
        .optimize = .fast,
        .matches = &.{
            "cp.async.ca.shared.global [%r",
            ", 4;\n",
            "cp.async.cg.shared.global [%r",
            ", 16;\n",
            ", 4, %r",
            "cp.async.commit_group;",
            "cp.async.wait_group 1;",
            "cp.async.wait_all;",
            "mbarrier.init.shared.b64 [%r",
            "cp.async.mbarrier.arrive.shared.b64 [%r",
            "cp.async.mbarrier.arrive.noinc.shared.b64 [%r",
            "mbarrier.arrive.shared.b64 %rd",
            "mbarrier.arrive.noComplete.shared.b64 %rd",
            "mbarrier.arrive_drop.shared.b64 %rd",
            "mbarrier.arrive_drop.noComplete.shared.b64 %rd",
            "mbarrier.test_wait.shared.b64 %p",
            "mbarrier.test_wait.parity.shared.b64 %p",
            "mbarrier.pending_count.b64 %r",
            "mbarrier.inval.shared.b64 [%r",
        },
    });

    cases.addCase(.{
        .name = "std.nvptx mbarrier transaction counts and proxy fences",
        .source =
        \\const nvptx = @import("std").nvptx;
        \\var barrier: nvptx.Mbarrier addrspace(.shared) = undefined;
        \\export fn k(out: [*]u32, bytes: u32) callconv(.kernel) void {
        \\    barrier.init(1);
        \\    nvptx.fenceProxyAsync(.shared);
        \\    nvptx.fenceProxyAsync(.global);
        \\    nvptx.fenceProxyAsync(null);
        \\    _ = barrier.arriveExpectTx(bytes);
        \\    barrier.completeTx(bytes);
        \\    out[0] = @intFromBool(barrier.tryWaitParity(1));
        \\}
        ,
        .cpu = "sm_90+ptx80",
        .optimize = .fast,
        .matches = &.{
            "fence.proxy.async.shared::cta;",
            "fence.proxy.async.global;",
            "fence.proxy.async;",
            "mbarrier.arrive.expect_tx.shared.b64 %rd",
            "mbarrier.complete_tx.shared.b64 [%r",
            "mbarrier.try_wait.parity.shared.b64 %p",
        },
    });

    cases.addCase(.{
        .name = "std.nvptx cluster barriers, fences, mapa and grid dependency control",
        .source =
        \\const nvptx = @import("std").nvptx;
        \\var barrier: nvptx.Mbarrier addrspace(.shared) = undefined;
        \\export fn k(out: [*]u32) callconv(.kernel) void {
        \\    nvptx.griddepcontrol(.wait);
        \\    barrier.init(1);
        \\    nvptx.fenceMbarrierInit();
        \\    nvptx.barrierClusterArrive(.relaxed, true);
        \\    nvptx.barrierClusterArrive(.release, true);
        \\    nvptx.barrierClusterArrive(null, false);
        \\    nvptx.barrierClusterWait(false, true);
        \\    nvptx.barrierClusterWait(true, true);
        \\    nvptx.fenceProxyAsyncSyncRestrict(.acquire);
        \\    nvptx.fenceProxyAsyncSyncRestrict(.release);
        \\    const rank = nvptx.clusterCtarank();
        \\    out[0] = nvptx.mapaSharedCluster(@intCast(@intFromPtr(&barrier)), rank ^ 1);
        \\    out[1] = nvptx.clusterNctarank();
        \\    nvptx.griddepcontrol(.launch_dependents);
        \\}
        ,
        .cpu = "sm_90+ptx86",
        .optimize = .fast,
        .matches = &.{
            "griddepcontrol.wait;",
            "fence.mbarrier_init.release.cluster;",
            "barrier.cluster.arrive.relaxed.aligned;",
            "barrier.cluster.arrive.release.aligned;",
            "barrier.cluster.arrive;",
            "barrier.cluster.wait.aligned;",
            "barrier.cluster.wait.acquire.aligned;",
            "fence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster;",
            "fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;",
            "mov.u32 %r",
            "%cluster_ctarank;",
            "mapa.shared::cluster.u32 %r",
            "%cluster_nctarank;",
            "griddepcontrol.launch_dependents;",
        },
    });

    cases.addCase(.{
        .name = "std.nvptx mbarrier phase waits and cluster arrivals",
        .source =
        \\const nvptx = @import("std").nvptx;
        \\const mbarrier = nvptx.mbarrier;
        \\var barrier: nvptx.Mbarrier addrspace(.shared) = undefined;
        \\export fn k(out: [*]u32, phase: u32, bytes: u32) callconv(.kernel) void {
        \\    barrier.init(1);
        \\    const peer = nvptx.mapaSharedCluster(@intCast(@intFromPtr(&barrier)), 1);
        \\    mbarrier.arriveCluster(peer, null);
        \\    mbarrier.arriveCluster(peer, .{ .semantics = .release, .scope = .cluster });
        \\    mbarrier.arriveExpectTxCluster(peer, 1024, null);
        \\    mbarrier.arriveExpectTxCluster(peer, bytes, .{ .semantics = .relaxed, .scope = .cta });
        \\    mbarrier.completeTxCluster(peer, bytes, .cluster);
        \\    var done = barrier.testWaitPhase(phase);
        \\    done = done and barrier.tryWaitPhase(phase, null, null);
        \\    done = done and barrier.tryWaitPhase(phase, .{ .semantics = .acquire, .scope = .cluster }, null);
        \\    done = done and barrier.tryWaitPhase(phase, null, 10000);
        \\    out[0] = @intFromBool(done);
        \\}
        ,
        .cpu = "sm_90+ptx86",
        .optimize = .fast,
        .matches = &.{
            "mbarrier.arrive.shared::cluster.b64 _, [%r",
            "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%r",
            "mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%r",
            "], 1024;",
            "mbarrier.arrive.expect_tx.relaxed.cta.shared::cluster.b64 _, [%r",
            "mbarrier.complete_tx.relaxed.cluster.shared::cluster.b64 [%r",
            "mbarrier.test_wait.parity.shared.b64 %p",
            "mbarrier.try_wait.parity.shared.b64 %p",
            "mbarrier.try_wait.parity.acquire.cluster.shared.b64 %p",
            ", 10000;",
        },
    });

    cases.addCase(.{
        .name = "std.nvptx bulk async groups and tensor map prefetch",
        .source =
        \\const nvptx = @import("std").nvptx;
        \\const TensorMap = extern struct { bytes: [128]u8 align(64) };
        \\export fn k(map: *const TensorMap) callconv(.kernel) void {
        \\    nvptx.cache.prefetchTensormap(map);
        \\    nvptx.cp_async_bulk.commitGroup();
        \\    nvptx.cp_async_bulk.waitGroup(1, true);
        \\    nvptx.cp_async_bulk.waitGroup(0, false);
        \\}
        ,
        .cpu = "sm_90+ptx80",
        .optimize = .fast,
        .matches = &.{
            "prefetch.tensormap [%rd",
            "cp.async.bulk.commit_group;",
            "cp.async.bulk.wait_group.read 1;",
            "cp.async.bulk.wait_group 0;",
        },
    });

    cases.addCase(.{
        .name = "std.nvptx wgmma groups",
        .source =
        \\const wgmma = @import("std").nvptx.wgmma;
        \\export fn k() callconv(.kernel) void {
        \\    wgmma.fenceSyncAligned();
        \\    wgmma.commitGroupSyncAligned();
        \\    wgmma.waitGroupSyncAligned(1);
        \\    wgmma.waitGroupSyncAligned(0);
        \\}
        ,
        .cpu = "sm_90a",
        .optimize = .fast,
        .matches = &.{
            "wgmma.fence.sync.aligned;",
            "wgmma.commit_group.sync.aligned;",
            "wgmma.wait_group.sync.aligned 1;",
            "wgmma.wait_group.sync.aligned 0;",
        },
    });

    cases.addCase(.{
        .name = "std.nvptx tcgen05 fences and waits",
        .source =
        \\const tcgen05 = @import("std").nvptx.tcgen05;
        \\export fn k() callconv(.kernel) void {
        \\    tcgen05.fence(.before_thread_sync);
        \\    tcgen05.fence(.after_thread_sync);
        \\    tcgen05.waitSyncAligned(.ld);
        \\    tcgen05.waitSyncAligned(.st);
        \\}
        ,
        .cpu = "sm_100a",
        .optimize = .fast,
        .matches = &.{
            "tcgen05.fence::before_thread_sync;",
            "tcgen05.fence::after_thread_sync;",
            "tcgen05.wait::ld.sync.aligned;",
            "tcgen05.wait::st.sync.aligned;",
        },
    });

    cases.addCase(.{
        .name = "std.nvptx f32x2 arithmetic and three-input max",
        .source =
        \\const math = @import("std").nvptx.math;
        \\export fn k(p: [*]@Vector(2, f32), out: [*]f32) callconv(.kernel) void {
        \\    const sum = math.addF32x2(.nearest_even, true, p[0], p[1]);
        \\    const product = math.mulF32x2(.toward_zero, false, sum, p[2]);
        \\    p[3] = math.fmaF32x2(.nearest_even, true, product, p[0], p[1]);
        \\    out[0] = math.max3F32(.{}, out[1], out[2], out[3]);
        \\    out[4] = math.max3F32(.{ .ftz = true, .nan = true, .abs = true }, out[5], out[6], out[7]);
        \\}
        ,
        .cpu = "sm_100a+ptx88",
        .optimize = .fast,
        .matches = &.{
            "add.rn.ftz.f32x2 d, a, b;",
            "mul.rz.f32x2 d, a, b;",
            "fma.rn.ftz.f32x2 d, a, b, c;",
            "max.f32 %f",
            "max.ftz.NaN.abs.f32 %f",
        },
    });

    cases.addCase(.{
        .name = "std.nvptx elect.sync",
        .source =
        \\const nvptx = @import("std").nvptx;
        \\export fn k(out: [*]u32) callconv(.kernel) void {
        \\    if (nvptx.electSync(.all)) out[0] = nvptx.laneid();
        \\}
        ,
        .cpu = "sm_90+ptx80",
        .optimize = .fast,
        .matches = &.{"elect.sync _|%p"},
    });

    cases.addCase(.{
        .name = "std.nvptx cache control and cached loads and stores",
        .source =
        \\const nvptx = @import("std").nvptx;
        \\const cache = nvptx.cache;
        \\export fn k(p: [*]addrspace(.global) u32, out: [*]u32) callconv(.kernel) void {
        \\    const policy = cache.createpolicy(.evict_last, .evict_unchanged, 0.5);
        \\    const range = cache.createpolicyRange(.evict_first, null, &p[0], 256, 512);
        \\    out[0] = cache.load(.{ .operator = .cache_streaming }, &p[0]);
        \\    out[1] = cache.load(.{ .l1_eviction_priority = .no_allocate }, &p[1]);
        \\    out[2] = cache.loadWithPolicy(.{ .operator = .cache_global_level }, &p[2], policy);
        \\    out[3] = cache.loadNonCoherent(.cache_all_levels, &p[3]);
        \\    out[4] = cache.loadNonCoherentWithPolicy(.cache_streaming, &p[4], range);
        \\    cache.store(.{ .operator = .write_through }, &p[5], 1);
        \\    cache.store(.{ .l1_eviction_priority = .evict_first }, &p[6], 2);
        \\    cache.storeWithPolicy(.{ .operator = .cache_streaming }, &p[7], 3, policy);
        \\    const pair: *addrspace(.global) @Vector(2, u32) = @ptrCast(@alignCast(p + 8));
        \\    pair.* = cache.load(.{ .operator = .last_use }, pair);
        \\    out[5] = cache.load(.{ .operator = .cache_volatile }, &p[10]);
        \\    cache.prefetch(.l1, &p[0]);
        \\    cache.prefetch(.l2_evict_last, &p[0]);
        \\    cache.prefetchu(@as(*u32, @addrSpaceCast(&p[0])));
        \\    cache.applypriority(@as(*addrspace(.global) [32]u32, @ptrCast(@alignCast(p + 64))));
        \\    cache.discard(@as(*addrspace(.global) [32]u32, @ptrCast(@alignCast(p + 96))));
        \\}
        ,
        .cpu = "sm_80+ptx74",
        .optimize = .fast,
        .matches = &.{
            "createpolicy.fractional.L2::evict_last.L2::evict_unchanged.b64 %rd",
            ", 0f3F000000;",
            "createpolicy.range.global.L2::evict_first.b64 %rd",
            "ld.global.cs.b32 %r",
            "ld.global.L1::no_allocate.b32 %r",
            "ld.global.cg.L2::cache_hint.b32 %r",
            "ld.global.ca.nc.b32 %r",
            "ld.global.cs.nc.L2::cache_hint.b32 %r",
            "st.global.wt.b32 [%rd",
            "st.global.L1::evict_first.b32 [%rd",
            "st.global.cs.L2::cache_hint.b32 [%rd",
            "ld.global.lu.v2.b32 {%r",
            "ld.global.cv.b32 %r",
            "prefetch.global.L1 [%rd",
            "prefetch.global.L2::evict_last [%rd",
            "prefetchu.L1 [%rd",
            "applypriority.global.L2::evict_normal [%rd",
            ", 128;\n",
            "discard.global.L2 [%rd",
        },
    });

    cases.addCase(.{
        .name = "std.nvptx texture fetches",
        .source =
        \\const nvptx = @import("std").nvptx;
        \\export fn k(out: [*]f32, h: u64, i: i32, x: f32) callconv(.kernel) void {
        \\    const t: nvptx.TextureObject = @enumFromInt(h);
        \\    var acc: @Vector(4, f32) = @splat(0);
        \\    acc += t.tex(.@"1d", f32, f32, .{x});
        \\    acc += t.tex(.@"3d", f32, f32, .{ x, x, x });
        \\    acc += t.tex(.a2d, f32, f32, .{ .index = @bitCast(i), .position = .{ x, x } });
        \\    acc += t.tex(.cube, f32, f32, .{ x, x, x });
        \\    acc += @floatFromInt(t.tex(.@"2d", u32, i32, .{ i, i }));
        \\    acc += t.texLevel(.acube, f32, .{ .index = @bitCast(i), .position = .{ x, x, x } }, x);
        \\    acc += t.texGrad(.a1d, f32, .{ .index = @bitCast(i), .position = .{x} }, .{x}, .{x});
        \\    acc += t.texGrad(.@"3d", f32, .{ x, x, x }, .{ x, x, x }, .{ x, x, x });
        \\    acc += t.tld4(.g, .a2d, f32, .{ .index = @bitCast(i), .position = .{ x, x } });
        \\    acc += @floatFromInt(t.tld4(.a, .cube, i32, .{ x, x, x }));
        \\    inline for (0..4) |n| out[n] = acc[n];
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{ "tex.1d.v4.f32.f32 {%r", "}, [%rd1, %r", "tex.3d.v4.f32.f32 {%r", "tex.a2d.v4.f32.f32 {%r", "tex.cube.v4.f32.f32 {%r", "tex.2d.v4.u32.s32 {%r", "tex.level.acube.v4.f32.f32 {%r", "], %f", "tex.grad.a1d.v4.f32.f32 {%r", "}, {%r", "tex.grad.3d.v4.f32.f32 {%r", "tld4.g.a2d.v4.f32.f32 {%r", "tld4.a.cube.v4.s32.f32 {%r" },
    });

    cases.addCase(.{
        .name = "std.nvptx texture queries",
        .source =
        \\const nvptx = @import("std").nvptx;
        \\export fn k(out: [*]u32, h: u64) callconv(.kernel) void {
        \\    const t: nvptx.TextureObject = @enumFromInt(h);
        \\    out[0] = t.txq(.width);
        \\    out[1] = t.txq(.num_mipmap_levels);
        \\    out[2] = t.txq(.channel_data_type);
        \\    out[3] = t.txqLevel(.height, out[0]);
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{
            "txq.width.b32 %r",
            "txq.num_mipmap_levels.b32 %r",
            "txq.channel_data_type.b32 %r",
            "txq.level.height.b32 %r",
            "], %r",
        },
    });

    cases.addCase(.{
        .name = "std.nvptx surface access",
        .source =
        \\const nvptx = @import("std").nvptx;
        \\export fn k(out: [*]u32, h: u64, x: i32) callconv(.kernel) void {
        \\    const s: nvptx.SurfaceObject = @enumFromInt(h);
        \\    const a = s.suldB(.@"1d", u32, 4, .trap, .{x});
        \\    const b = s.suldB(.@"2d", u8, 4, .clamp, .{ x, x });
        \\    const c = s.suldB(.a2d, u64, 2, .zero, .{ .index = @bitCast(x), .position = .{ x, x } });
        \\    out[0] = a[0] +% b[1] +% @as(u32, @truncate(c[1]));
        \\    s.sustB(.@"3d", .trap, .{ x, x, x }, @Vector(2, u16){ 1, 2 });
        \\    s.sustB(.a1d, .zero, .{ .index = @bitCast(x), .position = .{x} }, @Vector(1, u64){3});
        \\    s.suredB(.add, .@"2d", i32, .clamp, .{ x, x }, -1);
        \\    s.suredB(.max, .@"1d", u32, .trap, .{x}, 7);
        \\    s.suredB(.@"or", .@"3d", u32, .zero, .{ x, x, x }, 7);
        \\    s.suredB(.add, .@"1d", u64, .trap, .{x}, 7);
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{ "suld.b.1d.v4.b32.trap {%r", "}, [%rd1, %r", "suld.b.2d.v4.b8.clamp {%rs", "suld.b.a2d.v2.b64.zero {%rd", "sust.b.3d.v2.b16.trap [%rd1, {%r", "}], {%rs", "sust.b.a1d.b64.zero [%rd1, {%r", "sured.b.add.2d.s32.clamp [%rd1, {%r", "sured.b.max.1d.u32.trap [%rd1, %r", "sured.b.or.3d.b32.zero [%rd1, {%r", "sured.b.add.1d.u64.trap [%rd1, %r" },
    });

    cases.addCase(.{
        .name = "vector load and store of a vector",
        .source =
        \\export fn k(
        \\    out: *addrspace(.global) @Vector(4, f32),
        \\    in: *addrspace(.global) const @Vector(4, f32),
        \\) callconv(.kernel) void {
        \\    out.* = in.*;
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .safe,
        .matches = &.{
            "ld.global.v4.f32 {%f0, %f1, %f2, %f3}, [%rd1+0];",
            "st.global.v4.f32 [%rd0+0], {%f0, %f1, %f2, %f3};",
        },
        .absent = &.{ "ld.global.f32", "st.global.f32" },
    });

    cases.addCase(.{
        .name = "vector load and store of an aligned struct",
        .source =
        \\const Quad = extern struct { a: u32, b: u32, c: u32, d: u32 };
        \\export fn k(
        \\    out: *addrspace(.global) align(16) Quad,
        \\    in: *addrspace(.global) align(16) const Quad,
        \\) callconv(.kernel) void {
        \\    out.* = in.*;
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .safe,
        .matches = &.{
            "ld.global.v4.b32 {%r0, %r1, %r2, %r3}, [%rd1+0];",
            "st.global.v4.b32 [%rd0+0], {%r0, %r1, %r2, %r3};",
        },
        .absent = &.{ "ld.global.b32", "st.global.b32" },
    });

    cases.addCase(.{
        .name = "scalar load and store of an array with element alignment",
        .source =
        \\export fn k(out: *addrspace(.global) [4]f32, in: *addrspace(.global) const [4]f32) callconv(.kernel) void {
        \\    out.* = in.*;
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{
            "ld.global.f32 %f3, [%rd1+12];",
            "st.global.f32 [%rd0+12], %f3;",
        },
        .absent = &.{ ".v4", ".v2" },
    });

    cases.addCase(.{
        .name = "vector load and store of two f64",
        .source =
        \\export fn k(
        \\    out: *addrspace(.global) @Vector(2, f64),
        \\    in: *addrspace(.global) const @Vector(2, f64),
        \\) callconv(.kernel) void {
        \\    out.* = in.*;
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .safe,
        .matches = &.{
            "ld.global.v2.f64 {%fd0, %fd1}, [%rd1+0];",
            "st.global.v2.f64 [%rd0+0], {%fd0, %fd1};",
        },
        .absent = &.{ "ld.global.f64", "st.global.f64" },
    });

    cases.addCase(.{
        .name = "vector copy through a shared tile",
        .source =
        \\var tile: [64]@Vector(4, f32) addrspace(.shared) = undefined;
        \\export fn k(
        \\    out: [*]addrspace(.global) @Vector(4, f32),
        \\    in: [*]addrspace(.global) const @Vector(4, f32),
        \\) callconv(.kernel) void {
        \\    const i = @workItemId(0) % 64;
        \\    tile[i] = in[i];
        \\    out[i] = tile[63 - i];
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .safe,
        .matches = &.{
            "ld.global.v4.f32 {%f",
            "mad.lo.s32 %r3, %r1, 16, %r2;",
            "st.shared.v4.f32 [%r3+0]",
            "ld.shared.v4.f32 {%f",
            "st.global.v4.f32 [%rd",
        },
        .absent = &.{ "ld.shared.f32", "st.shared.f32" },
    });

    cases.addCase(.{
        .name = "vector access starts at an aligned offset",
        .source =
        \\const S = extern struct { a: u32, b: [4]f32 };
        \\export fn k(out: *addrspace(.global) align(16) S, in: *addrspace(.global) align(16) const S) callconv(.kernel) void {
        \\    out.* = in.*;
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .safe,
        .matches = &.{ "ld.global.b32 %r0, [%rd1+0];\n\tld.global.f32 %f0, [%rd1+4];\n\tld.global.v2.f32 {%f1, %f2}, [%rd1+8];\n\tld.global.f32 %f3, [%rd1+16];\n", "st.global.v4.b32 [%rd0+0], {%r0, %f0, %f1, %f2};\n\tst.global.f32 [%rd0+16], %f3;\n" },
        .absent = &.{ ".v4.f32", "ld.global.v4" },
    });

    cases.addCase(.{
        .name = "vector load of a kernel parameter",
        .source =
        \\export fn k(out: *addrspace(.global) @Vector(4, f32), v: @Vector(4, f32)) callconv(.kernel) void {
        \\    out.* = v;
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .safe,
        .matches = &.{
            ".param .align 16 .b8 param_1[16]",
            "ld.param.v4.f32 {%f0, %f1, %f2, %f3}, [param_1+0];",
            "st.global.v4.f32 [%rd0+0], {%f0, %f1, %f2, %f3};",
        },
    });

    cases.addCase(.{
        .name = "volatile access stays scalar",
        .source =
        \\export fn k(
        \\    out: *addrspace(.global) volatile @Vector(4, f32),
        \\    in: *addrspace(.global) volatile @Vector(4, f32),
        \\) callconv(.kernel) void {
        \\    out.* = in.*;
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{
            "ld.volatile.global.f32 %f3, [%rd1+12];",
            "st.volatile.global.f32 [%rd0+12], %f3;",
        },
        .absent = &.{ ".v4", ".v2" },
    });

    cases.addCase(.{
        .name = "functions that only a global refers to",
        .source =
        \\fn one() callconv(.c) u32 {
        \\    return 1;
        \\}
        \\fn two() callconv(.c) u32 {
        \\    return 2;
        \\}
        \\var table = [_]*const fn () callconv(.c) u32{ &one, &two };
        \\export fn k(out: *addrspace(.global) u32, i: u32) callconv(.kernel) void {
        \\    out.* = table[i]();
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{
            ".func (.param .b32 func_retval0) test_one$",
            ".func (.param .b32 func_retval0) test_two$",
        },
    });

    cases.addCase(.{
        .name = "extern function that the module exports",
        .source =
        \\const S = struct {
        \\    extern fn shared() u32;
        \\};
        \\export fn shared() u32 {
        \\    return 1;
        \\}
        \\export fn k(out: *addrspace(.global) u32) callconv(.kernel) void {
        \\    out.* = S.shared();
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{".visible .func (.param .b32 func_retval0) shared();\n"},
        .absent = &.{".extern"},
    });

    cases.addCase(.{
        .name = "null check of an optional error set",
        .source =
        \\export fn k(out: *addrspace(.global) bool, in: *addrspace(.global) ?anyerror) callconv(.kernel) void {
        \\    out.* = in.* == null;
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{"setp.eq.u16 %p0, %rs0, 0;"},
    });

    cases.addCase(.{
        .name = "zero-size global",
        .source =
        \\export var empty: [0]u32 = .{};
        \\export fn k(out: *addrspace(.global) usize) callconv(.kernel) void {
        \\    out.* = @intFromPtr(&empty);
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{".visible .global .align 4 .b8 empty[1];\n"},
    });

    cases.addCase(.{
        .name = "atomic read-modify-write of 32-bit and 64-bit integers",
        .source =
        \\export fn k(out: [*]addrspace(.global) u32, word: *addrspace(.global) u32, long: *addrspace(.global) i64) callconv(.kernel) void {
        \\    out[0] = @atomicRmw(u32, word, .Xchg, out[1], .monotonic);
        \\    out[2] = @atomicRmw(u32, word, .Add, out[3], .acquire);
        \\    out[4] = @atomicRmw(u32, word, .Sub, out[5], .release);
        \\    out[6] = @atomicRmw(u32, word, .And, out[7], .acq_rel);
        \\    out[8] = @atomicRmw(u32, word, .Or, out[9], .seq_cst);
        \\    out[10] = @atomicRmw(u32, word, .Xor, out[11], .monotonic);
        \\    out[12] = @atomicRmw(u32, word, .Max, out[13], .monotonic);
        \\    out[14] = @atomicRmw(u32, word, .Min, out[15], .monotonic);
        \\    out[16] = @atomicRmw(u32, word, .Nand, out[17], .monotonic);
        \\    long.* = @atomicRmw(i64, long, .Max, long.*, .monotonic) +% @atomicRmw(i64, long, .Sub, 3, .monotonic);
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{ "atom.relaxed.gpu.global.exch.b32 %r", "atom.acquire.gpu.global.add.u32 %r", "neg.s32 %r5, %r4;\n\tatom.release.gpu.global.add.u32 %r6, [%rd1], %r5;", "atom.acq_rel.gpu.global.and.b32 %r", "fence.sc.gpu;\n\tatom.acq_rel.gpu.global.or.b32 %r", "atom.relaxed.gpu.global.xor.b32 %r", "atom.relaxed.gpu.global.max.u32 %r", "atom.relaxed.gpu.global.min.u32 %r", "not.b32 %r20, %r19;\n\tatom.relaxed.gpu.global.cas.b32 %r21, [%rd1], %r18, %r20;", "atom.relaxed.gpu.global.max.s64 %rd", "atom.relaxed.gpu.global.add.u64 %rd" },
    });

    cases.addCase(.{
        .name = "atomic read-modify-write of bytes, halves, odd widths and bools",
        .source =
        \\var shared_byte: u8 addrspace(.shared) = undefined;
        \\export fn k(
        \\    bytes: [*]addrspace(.global) u8,
        \\    halves: [*]u16,
        \\    odd: *addrspace(.global) u24,
        \\    flag: *addrspace(.global) bool,
        \\) callconv(.kernel) void {
        \\    bytes[0] = @atomicRmw(u8, &bytes[1], .Add, bytes[2], .monotonic);
        \\    bytes[3] = @atomicRmw(u8, &shared_byte, .Min, bytes[4], .acq_rel);
        \\    halves[0] = @atomicRmw(u16, &halves[1], .Xchg, halves[2], .release);
        \\    odd.* = @atomicRmw(u24, odd, .Add, 1, .monotonic);
        \\    flag.* = @atomicRmw(bool, flag, .Xchg, true, .acquire);
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{ "and.b64 %rd6, %rd4, -2;", "atom.relaxed.gpu.global.cas.b16 %rs13, [%rd6], %rs3, %rs12;", "min.u16 %rs21, %rs20, %rs14;", "atom.acq_rel.gpu.shared.cas.b16 %rs", "atom.release.gpu.cas.b16 %rs", "and.b32 %r10, %r9, 16777215;\n\tatom.relaxed.gpu.global.cas.b32 %r11, [%rd2], %r7, %r10;", "atom.acquire.gpu.global.cas.b16 %rs" },
    });

    cases.addCase(.{
        .name = "atomic read-modify-write of floats",
        .source =
        \\var shared_sum: f32 addrspace(.shared) = undefined;
        \\export fn k(
        \\    single: [*]addrspace(.global) f32,
        \\    double: [*]addrspace(.global) f64,
        \\    half: [*]addrspace(.global) f16,
        \\) callconv(.kernel) void {
        \\    single[0] = @atomicRmw(f32, &shared_sum, .Add, single[1], .monotonic);
        \\    single[2] = @atomicRmw(f32, &single[3], .Add, single[4], .monotonic);
        \\    single[5] = @atomicRmw(f32, &single[6], .Xchg, single[7], .monotonic);
        \\    single[8] = @atomicRmw(f32, &single[9], .Max, single[10], .monotonic);
        \\    double[0] = @atomicRmw(f64, &double[1], .Sub, double[2], .monotonic);
        \\    double[3] = @atomicRmw(f64, &double[4], .Min, double[5], .monotonic);
        \\    half[0] = @atomicRmw(f16, &half[1], .Add, half[2], .monotonic);
        \\    half[3] = @atomicRmw(f16, &half[4], .Xchg, half[5], .monotonic);
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{
            "atom.relaxed.gpu.shared.add.f32 %f",
            "atom.relaxed.gpu.global.exch.b32 %r",
            "max.f32 %f9, %f8, %f7;",
            "neg.f64 %fd1, %fd0;\n\tatom.relaxed.gpu.global.add.f64 %fd",
            "min.f64 %fd5, %fd4, %fd3;",
            "atom.relaxed.gpu.global.cas.b64 %rd",
            "atom.relaxed.gpu.global.add.noftz.f16 %rs",
            "atom.relaxed.gpu.global.cas.b16 %rs",
        },
    });

    cases.addCase(.{
        .name = "atomic loads and stores",
        .source =
        \\var shared_flag: bool addrspace(.shared) = undefined;
        \\export fn k(
        \\    flags: [*]addrspace(.global) bool,
        \\    small: [*]addrspace(.global) i7,
        \\    single: [*]f32,
        \\    ptrs: [*]addrspace(.global) *u32,
        \\) callconv(.kernel) void {
        \\    flags[0] = @atomicLoad(bool, &flags[1], .acquire);
        \\    @atomicStore(bool, &shared_flag, flags[2], .release);
        \\    flags[3] = @atomicLoad(bool, &shared_flag, .seq_cst);
        \\    small[0] = @atomicLoad(i7, &small[1], .monotonic);
        \\    @atomicStore(i7, &small[2], small[3], .seq_cst);
        \\    single[0] = @atomicLoad(f32, &single[1], .acquire);
        \\    @atomicStore(f32, &single[2], single[3], .monotonic);
        \\    @atomicStore(*u32, &ptrs[0], @atomicLoad(*u32, &ptrs[1], .acquire), .release);
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{ "ld.acquire.gpu.global.u8 %rs0, [%rd4];\n\tsetp.ne.u16 %p0, %rs0, 0;", "st.release.gpu.shared.u8 [%r0], %rs3;", "fence.sc.gpu;\n\tld.acquire.gpu.shared.u8 %rs", "ld.relaxed.gpu.global.s8 %rs6, [%rd7];\n\tshl.b16 %rs7, %rs6, 9;\n\tshr.s16 %rs8, %rs7, 9;", "fence.sc.gpu;\n\tst.release.gpu.global.s8 [%rd", "ld.acquire.gpu.f32 %f", "st.relaxed.gpu.f32 [%rd", "ld.acquire.gpu.global.b64 %rd14, [%rd13];\n\tst.release.gpu.global.b64 [%rd3], %rd14;" },
    });

    cases.addCase(.{
        .name = "compare and exchange",
        .source =
        \\export fn k(
        \\    words: [*]addrspace(.global) u32,
        \\    longs: [*]addrspace(.global) u64,
        \\    odd: [*]addrspace(.global) i7,
        \\    bytes: [*]addrspace(.global) u8,
        \\    ptrs: [*]addrspace(.global) ?*u32,
        \\    ok: [*]addrspace(.global) bool,
        \\) callconv(.kernel) void {
        \\    ok[0] = @cmpxchgStrong(u32, &words[0], words[1], words[2], .seq_cst, .seq_cst) == null;
        \\    if (@cmpxchgWeak(u64, &longs[0], longs[1], longs[2], .release, .acquire)) |old| longs[3] = old;
        \\    if (@cmpxchgStrong(i7, &odd[0], odd[1], odd[2], .acquire, .monotonic)) |old| odd[3] = old;
        \\    ok[1] = @cmpxchgStrong(bool, &ok[2], ok[3], ok[4], .monotonic, .monotonic) orelse ok[5];
        \\    if (@cmpxchgStrong(u8, &bytes[0], bytes[1], bytes[2], .monotonic, .monotonic)) |old| bytes[3] = old;
        \\    ptrs[0] = @cmpxchgStrong(?*u32, &ptrs[1], ptrs[2], ptrs[3], .acq_rel, .monotonic) orelse ptrs[4];
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{ "fence.sc.gpu;\n\tatom.acq_rel.gpu.global.cas.b32 %r2, [%rd0], %r0, %r1;\n\tsetp.eq.u32 %p0, %r2, %r0;", "atom.acq_rel.gpu.global.cas.b64 %rd12, [%rd1], %rd9, %rd11;", "atom.acquire.gpu.global.cas.b16 %rs", "xor.pred %p11, %p10, %p8;\n\tnot.pred %p12, %p11;", "atom.relaxed.gpu.global.cas.b16 %rs" },
    });

    cases.addCase(.{
        .name = "atomics on registers and local memory",
        .source =
        \\export fn k(out: [*]addrspace(.global) u32, n: u32) callconv(.kernel) void {
        \\    var counter: u32 = n;
        \\    out[0] = @atomicRmw(u32, &counter, .Add, 2, .monotonic);
        \\    out[1] = @cmpxchgStrong(u32, &counter, 3, 4, .monotonic, .monotonic) orelse 0;
        \\    @atomicStore(u32, &counter, @atomicLoad(u32, &counter, .acquire) *% 5, .release);
        \\    out[2] = counter;
        \\    var spilled: [4]u32 = .{ n, n, n, n };
        \\    const private: *addrspace(.local) u32 = @addrSpaceCast(&spilled[n & 3]);
        \\    out[3] = @atomicRmw(u32, private, .Max, 7, .seq_cst);
        \\    out[4] = @cmpxchgWeak(u32, private, 1, 2, .seq_cst, .monotonic) orelse 0;
        \\    @atomicStore(u32, private, @atomicLoad(u32, private, .seq_cst) +% 1, .seq_cst);
        \\    out[5] = spilled[0];
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{ "add.u32 %r2, %r0, %r1;", "selp.b32 %r5, %r4, %r2, %p0;", "cvta.local.u64 %rd3, local_0;\n\tst.b32 [%rd3+0], %r0;", "mad.lo.s64 %rd9, %rd7, 4, %rd3;\n\tcvta.to.local.u64 %rd10, %rd9;\n\tcvt.u32.u64 %r8, %rd10;", "ld.local.b32 %r10, [%r8+0];\n\tmax.u32 %r11, %r10, %r9;\n\tst.local.b32 [%r8+0], %r11;" },
        .absent = &.{ "atom.", "fence." },
    });

    cases.addCase(.{
        .name = "float remainder and modulus",
        .source =
        \\export fn k(half: [*]addrspace(.global) f16, single: [*]addrspace(.global) f32, double: [*]addrspace(.global) f64) callconv(.kernel) void {
        \\    half[0] = @rem(half[1], half[2]);
        \\    half[3] = @mod(half[4], half[5]);
        \\    single[0] = @rem(single[1], single[2]);
        \\    single[3] = @mod(single[4], single[5]);
        \\    double[0] = @rem(double[1], double[2]);
        \\    double[3] = @mod(double[4], double[5]);
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{ "cvt.f32.f16 %f0, %rs0;", "mov.b32 %r5, 2143289344;", "setp.lt.f32 %p37, %f10, 0f00000000;", "add.rn.f32 %f10, %f12, %f11;", "clz.b64 %r", "mov.b64 %rd20, 9221120237041090560;" },
        .absent = &.{"rem.f"},
    });

    cases.addCase(.{
        .name = "tag and error name tables",
        .source =
        \\const Color = enum(u8) { red, green, blue };
        \\const Open = enum(u8) { low = 1, high = 200, _ };
        \\fn check(x: u32) error{ TooSmall, TooLarge }!u32 {
        \\    if (x < 10) return error.TooSmall;
        \\    if (x > 100) return error.TooLarge;
        \\    return x;
        \\}
        \\export fn k(out: [*]addrspace(.global) usize, color: Color, open: Open, x: u32) callconv(.kernel) void {
        \\    out[0] = @tagName(color).len;
        \\    out[1] = @tagName(open).len;
        \\    out[2] = if (check(x)) |v| v else |err| @errorName(err).len;
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{
            "_0[4] = {114,101,100,0};",
            "_1[6] = {103,114,101,101,110,0};",
            "_1[5] = {104,105,103,104,0};",
            ".global .align 1 .b8 __zig_error_name_0[9] = {84,111,111,83,109,97,108,108,0};",
            ".global .align 8 .u64 __zig_error_names[6] = {0,2,generic(__zig_error_name_0),8,generic(__zig_error_name_1),8};",
            "mov.u64 %rd17, __zig_error_names;\n\tmad.lo.u64 %rd18, %rd16, 16, %rd17;",
        },
    });

    cases.addCase(.{
        .name = "null check of an optional slice",
        .source =
        \\export fn k(out: *addrspace(.global) bool, in: *addrspace(.global) ?[]u8) callconv(.kernel) void {
        \\    out.* = in.* == null;
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{
            "ld.global.b64 %rd2, [%rd1+0];",
            "setp.eq.u64 %p0, %rd2, 0;",
        },
    });

    cases.addCase(.{
        .name = "global initial values",
        .source =
        \\const Packed = packed struct(u16) { low: u4, mid: u8, high: u4 };
        \\const Tagged = union(enum) { int: u32, float: f32 };
        \\const Bare = extern union { int: u32, float: f32 };
        \\const Pair = struct { a: u32, b: u32 };
        \\fn one() callconv(.c) u32 {
        \\    return 1;
        \\}
        \\var half: f16 = 1.5;
        \\var single: f32 = -2.25;
        \\var double: f64 = 0.1;
        \\var bits: Packed = .{ .low = 1, .mid = 0xab, .high = 7 };
        \\var tagged: Tagged = .{ .float = 0.5 };
        \\var bare: Bare = .{ .int = 0xdeadbeef };
        \\var maybe: ?u32 = 7;
        \\var nothing: ?u32 = null;
        \\var nothing_slice: ?[]const u8 = null;
        \\var failure: anyerror!u16 = error.Overflow;
        \\var success: anyerror!u16 = 5;
        \\var pair: Pair = .{ .a = 1, .b = 2 };
        \\var field_ptr: *u32 = &pair.b;
        \\var ptrs: [2]*const f32 = .{ &single, &single };
        \\var func: *const fn () callconv(.c) u32 = &one;
        \\var text: []const u8 = "abc";
        \\export fn k(out: [*]addrspace(.global) usize) callconv(.kernel) void {
        \\    const addresses = [_]usize{
        \\        @intFromPtr(&half),          @intFromPtr(&double),  @intFromPtr(&bits),
        \\        @intFromPtr(&tagged),        @intFromPtr(&bare),    @intFromPtr(&maybe),
        \\        @intFromPtr(&nothing),       @intFromPtr(&failure), @intFromPtr(&success),
        \\        @intFromPtr(&nothing_slice), @intFromPtr(&ptrs),    @intFromPtr(&text),
        \\        @intFromPtr(&func),          @intFromPtr(&field_ptr),
        \\    };
        \\    for (addresses, 0..) |a, i| out[i] = a;
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{
            "_half$",
            "[2] = {0,62};",
            "[8] = {154,153,153,153,153,153,185,63};",
            "_bits$",
            "[2] = {177,122};",
            "_tagged$",
            "[8] = {0,0,0,63,1,0,0,0};",
            "[4] = {239,190,173,222};",
            "[8] = {7,0,0,0,1,0,0,0};",
            "_failure$",
            "[4] = {0,0,1,0};",
            "_nothing_slice$",
            "[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};",
            "[2] = {generic(test_single$",
            "[1] = {test_one$",
            "[1] = {generic(test_pair$",
            ")+4};",
        },
    });

    cases.addCase(.{
        .name = "address space casts and inline assembly operands",
        .source =
        \\var tile: [32]u32 addrspace(.shared) = undefined;
        \\export fn k(out: [*]addrspace(.global) u32, generic: *u32, flag: bool, h: f16, d: f64) callconv(.kernel) void {
        \\    const shared: *u32 = @addrSpaceCast(&tile[out[0] & 31]);
        \\    shared.* = 1;
        \\    const back: *addrspace(.shared) u32 = @addrSpaceCast(shared);
        \\    const global: *addrspace(.global) u32 = @addrSpaceCast(generic);
        \\    global.* = back.*;
        \\    var counter: u32 = out[1];
        \\    asm volatile ("add.u32 %[c], %[c], %[n];"
        \\        : [c] "+r" (counter),
        \\        : [n] "n" (5),
        \\    );
        \\    out[2] = counter;
        \\    asm volatile ("st.global.u32 [%[p]], %[v];"
        \\        :
        \\        : [p] "l" (&out[3]),
        \\          [v] "r" (@as(u32, 9)),
        \\        : .{ .memory = true });
        \\    const not_flag = asm ("not.pred %[o], %[i];"
        \\        : [o] "=b" (-> bool),
        \\        : [i] "b" (flag),
        \\    );
        \\    out[4] = @intFromBool(not_flag);
        \\    const twice = asm ("add.f16 %[o], %[i], %[i];"
        \\        : [o] "=h" (-> f16),
        \\        : [i] "h" (h),
        \\    );
        \\    out[5] = @as(u16, @bitCast(twice));
        \\    const neg = asm ("neg.f64 %[o], %[i];"
        \\        : [o] "=d" (-> f64),
        \\        : [i] "d" (d),
        \\    );
        \\    out[6] = @intFromFloat(neg);
        \\    asm volatile ("st.shared.u32 [%[p]], 3;"
        \\        :
        \\        : [p] "r" (&tile[1]),
        \\    );
        \\    var tail: u32 = undefined;
        \\    asm volatile (
        \\        \\skip%=:
        \\        \\mov.u32 %[t], %%tid.x;
        \\        : [t] "=r" (tail),
        \\    );
        \\    out[7] = tail;
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{ "mad.lo.s32 %r4, %r3, 4, %r2;", "cvt.u64.u32 %rd4, %r4;\n\tcvta.shared.u64 %rd5, %rd4;", "cvta.to.shared.u64 %rd6, %rd5;\n\tcvt.u32.u64 %r6, %rd6;", "cvta.to.global.u64 %rd7, %rd1;", "ld.shared.b32 %r7, [%r6+0];", "add.u32 %r8, %r8, 5;", "not.pred %p2, %p0;", "add.f16 %rs3, %rs1, %rs1;", "neg.f64 %fd1, %fd0;", "+4;\n\t// begin inline asm\n\tst.shared.u32 [%r13], 3;", ":\n\tmov.u32 %r15, %tid.x;", "\tskip$L__" },
    });

    cases.addCase(.{
        .name = "bit-packed vector field",
        .source =
        \\const S = struct { f0: @Vector(5, u6), f1: i32 };
        \\export fn k(out: *addrspace(.global) u32, x: u32) callconv(.kernel) void {
        \\    var s: S = .{ .f0 = @splat(0), .f1 = @bitCast(x) };
        \\    _ = &s;
        \\    out.* = s.f0[4];
        \\}
        \\export fn store(out: *addrspace(.global) S, v: @Vector(5, u6), x: i32) callconv(.kernel) void {
        \\    out.* = .{ .f0 = v, .f1 = x };
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .safe,
        .matches = &.{ "shl.b64 %rd17, %rd16, 24;", "cvt.u32.u64 %r2, %rd18;\n\tand.b32 %r3, %r2, 1073741823;\n\tst.global.b32 [%rd0+0], %r3;", "add.u64 %rd19, %rd0, 4;\n\tst.global.b32 [%rd19+0], %r1;", "and.b16 %rs1, %rs0, 63;\n\tcvt.u32.u16 %r2, %rs1;\n\tst.global.b32 [%rd0+0], %r2;" },
    });

    cases.addCase(.{
        .name = "recursive and mutually recursive functions",
        .source =
        \\fn factorial(n: u32) u32 {
        \\    return if (n <= 1) 1 else n *% factorial(n - 1);
        \\}
        \\fn isEven(n: u32) bool {
        \\    return if (n == 0) true else isOdd(n - 1);
        \\}
        \\fn isOdd(n: u32) bool {
        \\    return if (n == 0) false else isEven(n - 1);
        \\}
        \\fn classify(n: u32) u32 {
        \\    if (n == 0) return 10;
        \\    if (n == 1) return 20;
        \\    if (n < 8) return n *% 3;
        \\    if (n < 64) return n +% 7;
        \\    return n;
        \\}
        \\noinline fn kept(n: u32) u32 {
        \\    return classify(n) +% 1;
        \\}
        \\export fn k(out: [*]addrspace(.global) u32, n: u32) callconv(.kernel) void {
        \\    out[0] = factorial(n);
        \\    out[1] = @intFromBool(isEven(n));
        \\    out[2] = classify(n);
        \\    out[3] = kept(n);
        \\    out[4] = @call(.always_inline, classify, .{n +% 1});
        \\    out[5] = @call(.never_inline, classify, .{n +% 2});
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .safe,
        .matches = &.{
            "call (retval0), test_factorial$",
            "call (retval0), test_isOdd$",
            "call (retval0), test_kept$",
        },
        .in_order = &.{ ".u64 __zig_function_table[3] = {test_factorial$", ", test_isEven$", ", test_isOdd$" },
        .absent = &.{".calltargets"},
    });

    cases.addCase(.{
        .name = "unreachable call of a recursive function",
        .source =
        \\fn foo(rt_false: bool) u8 {
        \\    if (rt_false) _ = foo(rt_false);
        \\    return 123;
        \\}
        \\export fn k(out: *addrspace(.global) u8) callconv(.kernel) void {
        \\    out.* = foo(false);
        \\}
        ,
        .cpu = "sm_90",
        .matches = &.{
            ".u64 __zig_function_table[1] = {test_foo$",
            "call (retval0), test_foo$",
        },
        .absent = &.{".calltargets"},
    });

    cases.addCase(.{
        .name = "the functions of a function-pointer table get into the address table",
        .source =
        \\fn scaled(x: u32) u32 {
        \\    return x *% 2;
        \\}
        \\fn negated(x: u32) u32 {
        \\    return 0 -% x;
        \\}
        \\const table = [_]*const fn (u32) u32{ scaled, negated };
        \\export fn k(out: [*]addrspace(.global) u32) callconv(.kernel) void {
        \\    const i = @workItemId(0);
        \\    out[i] = table[i & 1](21);
        \\}
        ,
        .cpu = "sm_90",
        .optimize = .fast,
        .in_order = &.{ ".u64 __zig_function_table[2] = {test_scaled$", ", test_negated$" },
    });

    inline for (.{ .debug, .fast }) |optimize| {
        cases.addCase(.{
            .name = "a 64-bit sum with a constant beyond 32 bits uses the 32-bit halves in " ++ @tagName(optimize) ++ " mode",
            .source =
            \\noinline fn hash(x: u64) u64 {
            \\    return x *% 0x9e37_79b9_7f4a_7c15;
            \\}
            \\export fn k(w: [*]addrspace(.global) u64) callconv(.kernel) void {
            \\    w[1] = (1 << 40) +% hash(w[1]);
            \\    w[2] = hash(w[0]) -% 0x1234_5678_9abc;
            \\}
            \\noinline fn wide(x: u128) u128 {
            \\    return x *% 3;
            \\}
            \\export fn k2(w: [*]addrspace(.global) u128) callconv(.kernel) void {
            \\    w[1] = wide(w[0]) +% (1 << 40);
            \\}
            ,
            .cpu = "sm_80",
            .optimize = optimize,
            .matches = &.{ "addc.u32 %r3, %r1, 256;", "sub.cc.u32 %r6, %r4, 1450744508;\n\tsubc.u32 %r7, %r5, 4660;" },
            .in_order = &.{ "add.cc.u32", "addc.cc.u32", "mov.b64", "addc.u64" },
            .absent = &.{ "add.cc.u64", "sub.u64", "20015998343868" },
        });
        cases.addCase(.{
            .name = "a division by a constant is a multiply-high sequence in " ++ @tagName(optimize) ++ " mode",
            .source =
            \\export fn k(out: [*]addrspace(.global) u32, x: u32, y: i32) callconv(.kernel) void {
            \\    out[0] = x / 1073741822;
            \\    out[1] = x % 7;
            \\    out[2] = @bitCast(@divTrunc(y, -1000));
            \\    out[3] = @bitCast(@rem(y, 1000));
            \\}
            ,
            .cpu = "sm_80",
            .optimize = optimize,
            .matches = &.{
                "shr.u32 %r2, %r0, 1;\n\tmul.hi.u32 %r3, %r2, 2147483653;\n\tshr.u32 %r4, %r3, 28;",
                "mul.hi.s32 %r12, %r1, -274877907;",
                "mul.lo.s32 %r20, %r19, 1000;\n\tsub.s32 %r21, %r1, %r20;",
            },
            .absent = &.{ "div.", "rem." },
        });
    }

    cases.addCase(.{
        .name = "a divisor that inlining makes constant gives a multiply-high sequence",
        .source =
        \\fn divide(x: u32, d: u32) u32 {
        \\    return x / d;
        \\}
        \\export fn k(out: [*]addrspace(.global) u32, x: u32) callconv(.kernel) void {
        \\    out[0] = divide(x, 1073741822);
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{"mul.hi.u32"},
        .absent = &.{"div.u32"},
    });

    cases.addCase(.{
        .name = "add, subtract and compare integers wider than 64 bits",
        .source =
        \\export fn add(out: *u128, a: u128, b: u128) callconv(.kernel) void {
        \\    out.* = a +% b;
        \\}
        \\export fn sub(out: *i100, a: i100, b: i100) callconv(.kernel) void {
        \\    out.* = a -% b;
        \\}
        \\export fn less(out: *bool, a: i128, b: i128) callconv(.kernel) void {
        \\    out.* = a < b;
        \\}
        \\export fn equal(out: *bool, a: u192, b: u192) callconv(.kernel) void {
        \\    out.* = a == b;
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .safe,
        .matches = &.{ "add.cc.u64 %rd5, %rd1, %rd3;\n\taddc.u64 %rd6, %rd2, %rd4;\n\tst.v2.b64", "sub.cc.u64 %rd5, %rd1, %rd3;\n\tsubc.u64 %rd6, %rd2, %rd4;\n\tshl.b64 %rd7, %rd6, 28;\n\tshr.s64 %rd8, %rd7, 28;", "setp.lt.u64 %p0, %rd1, %rd3;\n\tsetp.eq.and.s64 %p1, %rd2, %rd4, %p0;\n\tsetp.lt.or.s64 %p2, %rd2, %rd4, %p1;", "setp.eq.u64 %p0, %rd1, %rd5;\n\tsetp.eq.and.u64 %p1, %rd2, %rd6, %p0;\n\tsetp.eq.and.u64 %p2, %rd3, %rd7, %p1;\n\tsetp.eq.and.u64 %p3, %rd4, %rd8, %p2;" },
    });

    cases.addCase(.{
        .name = "multiply integers wider than 64 bits",
        .source =
        \\export fn mul(out: *u128, a: u128, b: u128) callconv(.kernel) void {
        \\    out.* = a *% b;
        \\}
        \\export fn mulOverflow(out: *struct { u128, u1 }, a: u128, b: u128) callconv(.kernel) void {
        \\    out.* = @mulWithOverflow(a, b);
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{ "mad.lo.cc.u64 %rd6, %rd1, %rd3, %rd5;\n\tmadc.lo.u64 %rd7, %rd1, %rd4, %rd5;\n\tmad.hi.u64 %rd8, %rd1, %rd3, %rd7;\n\tmad.lo.u64 %rd9, %rd2, %rd3, %rd8;", "madc.lo.cc.u64 %rd7, %rd1, %rd4, %rd5;\n\taddc.u64 %rd8, 0, 0;\n\tmad.hi.cc.u64 %rd9, %rd1, %rd3, %rd7;\n\tmadc.hi.u64 %rd10, %rd1, %rd4, %rd8;", "setp.ne.or.u64 %p1, %rd14, %rd16, %p0;\n\tsetp.ne.or.u64 %p2, %rd15, %rd16, %p1;" },
    });

    cases.addCase(.{
        .name = "f16 floor and truncating division round the quotient to f16",
        .source =
        \\export fn k(out: [*]addrspace(.global) f16, a: f16, b: f16) callconv(.kernel) void {
        \\    out[0] = @divFloor(a, b);
        \\    out[1] = @divTrunc(a, b);
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{
            "div.rn.f32 %f2, %f0, %f1;\n\tcvt.rn.f16.f32 %rs2, %f2;\n\tcvt.f32.f16 %f3, %rs2;\n\tcvt.rmi.f32.f32 %f4, %f3;",
            "cvt.rzi.f32.f32 %f5, %f3;\n\tcvt.rn.f16.f32 %rs4, %f5;",
        },
    });

    cases.addCase(.{
        .name = "shift integers wider than 64 bits",
        .source =
        \\export fn shl(out: *u128, a: u128, s: u7) callconv(.kernel) void {
        \\    out.* = a << s;
        \\}
        \\export fn sar(out: *i128, a: i128) callconv(.kernel) void {
        \\    out.* = a >> 70;
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{ "shr.u32 %r1, %r0, 6;\n\tand.b32 %r2, %r1, 1;\n\tsetp.ne.u32 %p0, %r2, 0;\n\tselp.b64 %rd4, %rd3, %rd1, %p0;\n\tselp.b64 %rd5, %rd1, %rd2, %p0;", "and.b32 %r3, %r0, 63;\n\tsub.u32 %r4, 64, %r3;\n\tshl.b64 %rd6, %rd4, %r3;\n\tshl.b64 %rd7, %rd5, %r3;\n\tshr.u64 %rd8, %rd4, %r4;\n\tor.b64 %rd9, %rd7, %rd8;", "shr.s64 %rd3, %rd2, 63;\n\tshr.u64 %rd4, %rd2, 6;\n\tshl.b64 %rd5, %rd3, 58;" },
    });

    cases.addCase(.{
        .name = "divide integers wider than 64 bits",
        .source =
        \\export fn div(out: *u128, a: u128, b: u128) callconv(.kernel) void {
        \\    out.* = a / b;
        \\}
        \\export fn divTrunc(out: *i128, a: i128) callconv(.kernel) void {
        \\    out.* = @divTrunc(a, 16);
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{ "shr.s64 %rd3, %rd2, 63;\n\tand.b64 %rd4, %rd3, 15;\n\tand.b64 %rd5, %rd3, 0;\n\tadd.cc.u64 %rd6, %rd1, %rd4;\n\taddc.u64 %rd7, %rd2, %rd5;", "add.cc.u64 %rd8, %rd8, %rd8;\n\taddc.cc.u64 %rd9, %rd9, %rd9;\n\taddc.cc.u64 %rd10, %rd10, %rd10;\n\taddc.u64 %rd11, %rd11, %rd11;\n\tsub.cc.u64 %rd12, %rd10, %rd3;\n\tsubc.cc.u64 %rd13, %rd11, %rd4;\n\tsubc.u64 %rd14, %rd7, %rd7;\n\tsetp.ge.s64 %p2, %rd14, 0;\n\t@%p2 mov.b64 %rd10, %rd12;\n\t@%p2 mov.b64 %rd11, %rd13;\n\t@%p2 or.b64 %rd8, %rd8, 1;" },
    });

    cases.addCase(.{
        .name = "count and swap the bits of integers wider than 64 bits",
        .source =
        \\export fn clz(out: *u8, a: u128) callconv(.kernel) void {
        \\    out.* = @clz(a);
        \\}
        \\export fn popCount(out: *u8, a: i100) callconv(.kernel) void {
        \\    out.* = @popCount(a);
        \\}
        \\export fn byteSwap(out: *u128, a: u128) callconv(.kernel) void {
        \\    out.* = @byteSwap(a);
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .safe,
        .matches = &.{ "clz.b64 %r0, %rd2;\n\tsetp.ne.u64 %p0, %rd2, 0;\n\tclz.b64 %r1, %rd1;\n\t@!%p0 add.u32 %r0, %r0, %r1;", "and.b64 %rd3, %rd2, 68719476735;\n\tpopc.b64 %r0, %rd1;\n\tpopc.b64 %r1, %rd3;\n\tadd.u32 %r2, %r0, %r1;", "mov.b64 %rd3, {%r2, %r3};", "mov.b64 %rd4, {%r6, %r7};\n\tst.v2.b64 [%rd0+0], {%rd3, %rd4};" },
    });

    cases.addCase(.{
        .name = "parameter aligned to more than 128 bytes",
        .source =
        \\noinline fn h(v: @Vector(8, u256)) @Vector(8, u256) {
        \\    return v +% v;
        \\}
        \\export fn k(out: *@Vector(8, u256), in: *const @Vector(8, u256)) callconv(.kernel) void {
        \\    out.* = h(in.*);
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{
            ".func (.param .align 128 .b8 func_retval0[256])",
            ".param .align 128 .b8 param_0[256]",
            ".param .align 128 .b8 param0[256];",
        },
        .absent = &.{".align 256"},
    });

    cases.addCase(.{
        .name = "math builtins lower to compiler_rt routines",
        .source =
        \\export fn k(out: *[3]f64, in: *const [3]f64) callconv(.kernel) void {
        \\    out[0] = @sin(in[0]);
        \\    out[1] = @floatCast(@exp(@as(f32, @floatCast(in[1]))));
        \\    out[2] = @floatCast(@log2(@as(f16, @floatCast(in[2]))));
        \\}
        ,
        .cpu = "sm_80",
        // `pio2_1` of the argument reduction in lib/compiler_rt/rem_pio2.zig.
        .matches = &.{"0d3FF921FB54400000"},
        .absent = &.{ "sin.approx", "ex2.approx", "lg2.approx" },
    });

    cases.addCase(.{
        .name = "1-byte and 2-byte parameters of a function",
        .source =
        \\noinline fn f(a: u8, b: i16, c: bool) u32 {
        \\    return a + @as(u32, @bitCast(@as(i32, b))) + @intFromBool(c);
        \\}
        \\export fn k(out: *u32, a: u8, b: i16, c: bool) callconv(.kernel) void {
        \\    out.* = f(a, b, c);
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{
            "\t.param .align 1 .b8 param_0[1],\n\t.param .align 2 .b8 param_1[2],\n\t.param .align 1 .b8 param_2[1]);",
            ".param .align 2 .b8 param1[2];",
        },
        .absent = &.{ ".param .u8", ".param .b16", ".param .pred" },
    });

    cases.addCase(.{
        .name = "NVPTX object input",
        .library =
        \\var bias: u32 = 3;
        \\export fn twice(x: u32) u32 {
        \\    return x *% 2 +% bias;
        \\}
        \\export fn thrice(x: u32) u32 {
        \\    return x *% 3;
        \\}
        \\noinline fn kept(x: u32) callconv(.c) u32 {
        \\    return x ^ 5;
        \\}
        \\comptime {
        \\    @export(&kept, .{ .name = "kept" });
        \\}
        ,
        .source =
        \\extern fn twice(x: u32) u32;
        \\extern fn thrice(x: u32) u32;
        \\extern fn kept(x: u32) u32;
        \\export fn k(out: *addrspace(.global) u32) callconv(.kernel) void {
        \\    out.* = twice(out.*) +% kept(out.*) +% @call(.never_inline, thrice, .{1});
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .safe,
        .matches = &.{
            ".visible .func (.param .b32 func_retval0) twice(",
            ".visible .func (.param .b32 func_retval0) kept(",
            "call (retval0), kept, (param0);\n",
            "call (retval0), thrice, (param0);\n",
            "$i0[4] = {3,0,0,0};",
        },
        .absent = &.{ ".extern", "call (retval0), twice, (param0);\n" },
    });

    cases.addCase(.{
        .name = "function of a file whose name starts with a digit",
        .file_name = "2d.zig",
        .source =
        \\noinline fn area(w: u32, h: u32) u32 {
        \\    return w *% h;
        \\}
        \\export fn k(out: *addrspace(.global) u32) callconv(.kernel) void {
        \\    out.* = area(out.*, 3);
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{".func (.param .b32 func_retval0) _2d_area$"},
    });

    cases.addCase(.{
        .name = "second export name of a function",
        .library =
        \\fn twice(x: u32) callconv(.c) u32 {
        \\    return x *% 2;
        \\}
        \\comptime {
        \\    @export(&twice, .{ .name = "twice" });
        \\    @export(&twice, .{ .name = "double" });
        \\}
        ,
        .source =
        \\extern fn twice(x: u32) u32;
        \\extern fn double(x: u32) u32;
        \\export fn k(out: *addrspace(.global) u32) callconv(.kernel) void {
        \\    out.* = twice(out.*) +% @call(.never_inline, double, .{1});
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .safe,
        .matches = &.{
            ".visible .func (.param .b32 func_retval0) double(",
            ".alias double, twice;\n",
            "call (retval0), double, (param0);\n",
        },
        .absent = &.{ ".extern", "call (retval0), twice, (param0);\n" },
    });

    cases.addCase(.{
        .name = "f80 and f128 values",
        .source =
        \\noinline fn pass(x: f128) f128 {
        \\    return x;
        \\}
        \\export fn k(out: *addrspace(.global) f128, bits: *addrspace(.global) u80, in: *addrspace(.global) const f80) callconv(.kernel) void {
        \\    out.* = pass(-2.25);
        \\    bits.* = @bitCast(-in.*);
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{
            // The high limb of -2.25.
            "13835093239654252544",
            ".param .align 8 .b8 param0[16];",
            // Negation flips bit 15 of the 16-bit sign and exponent word of an f80.
            ", 32768;",
            // The top limb of a u80 keeps only its low 16 bits.
            ", 65535;",
        },
        .absent = &.{".extern"},
    });

    cases.addCase(.{
        .name = "compiler_rt calls",
        .library =
        \\const F128 = extern struct { lo: u64, hi: u64 };
        \\noinline fn mul(a: F128, b: F128) callconv(.c) F128 {
        \\    return .{ .lo = a.lo ^ b.lo, .hi = a.hi ^ b.hi };
        \\}
        \\noinline fn lessThan(a: F128, b: F128) callconv(.c) c_long {
        \\    return if (a.hi < b.hi) -1 else 0;
        \\}
        \\noinline fn fromU128(a: u128) callconv(.c) f64 {
        \\    return @bitCast(@as(u64, @truncate(a)));
        \\}
        \\comptime {
        \\    @export(&mul, .{ .name = "__multf3" });
        \\    @export(&lessThan, .{ .name = "__lttf2" });
        \\    @export(&fromU128, .{ .name = "__floatuntidf" });
        \\}
        ,
        .source =
        \\export fn k(
        \\    out: *addrspace(.global) f128,
        \\    less: *addrspace(.global) bool,
        \\    float: *addrspace(.global) f64,
        \\    a: *addrspace(.global) const f128,
        \\    wide: *addrspace(.global) const u96,
        \\) callconv(.kernel) void {
        \\    out.* = a.* * a.*;
        \\    less.* = a.* < out.*;
        \\    float.* = @floatFromInt(wide.*);
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{
            "call (retval0), __multf3, (param0, param1);\n",
            ".param .b64 retval0;\n\tcall (retval0), __lttf2, (param0, param1);\n",
            "cvt.s32.s64",
            ".param .align 16 .b8 param0[16];",
            "call (retval0), __floatuntidf, (param0);\n",
        },
        .absent = &.{".extern"},
    });

    cases.addCase(.{
        .name = "math builtins call compiler_rt",
        .library =
        \\noinline fn expf(x: f32) callconv(.c) f32 {
        \\    return x + 1;
        \\}
        \\noinline fn sin(x: f64) callconv(.c) f64 {
        \\    return x;
        \\}
        \\noinline fn logh(x: f16) callconv(.c) f16 {
        \\    return -x;
        \\}
        \\comptime {
        \\    @export(&expf, .{ .name = "expf" });
        \\    @export(&sin, .{ .name = "sin" });
        \\    @export(&logh, .{ .name = "__logh" });
        \\}
        ,
        .source =
        \\export fn k(o: *addrspace(.global) f32, d: *addrspace(.global) f64, h: *addrspace(.global) f16) callconv(.kernel) void {
        \\    o.* = @exp(o.*);
        \\    d.* = @sin(d.*);
        \\    h.* = @log(h.*);
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{
            "call (retval0), expf, (param0);\n",
            "call (retval0), sin, (param0);\n",
            "call (retval0), __logh, (param0);\n",
        },
        .absent = &.{".extern"},
    });

    cases.addCase(.{
        .name = "optimized float mode",
        .source =
        \\export fn fast(o: [*]addrspace(.global) f32, d: *addrspace(.global) f64) callconv(.kernel) void {
        \\    @setFloatMode(.optimized);
        \\    o[0] = o[1] / o[2];
        \\    o[3] = o[1] * o[2] + o[3];
        \\    d.* = d.* / 3.0;
        \\}
        \\export fn strict(o: [*]addrspace(.global) f32) callconv(.kernel) void {
        \\    o[0] = o[1] / o[2];
        \\    o[3] = o[1] * o[2] + o[3];
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{
            "div.approx.ftz.f32",
            "mul.ftz.f32",
            "add.ftz.f32",
            "div.rn.f64",
            "div.rn.f32",
            "mul.rn.f32",
        },
        .absent = &.{"div.approx.ftz.f64"},
    });

    cases.addCase(.{
        .name = "std.nvptx.math approximate f32 and f64",
        .source =
        \\const math = @import("std").nvptx.math;
        \\export fn k(f: [*]f32, d: [*]f64) callconv(.kernel) void {
        \\    f[0] = math.ex2ApproxF32(true, f[1]) + math.ex2ApproxF32(false, f[2]);
        \\    f[3] = math.lg2ApproxF32(true, f[4]) + math.lg2ApproxF32(false, f[5]);
        \\    f[6] = math.sinApproxF32(true, f[7]) + math.sinApproxF32(false, f[8]);
        \\    f[9] = math.cosApproxF32(true, f[10]) + math.cosApproxF32(false, f[11]);
        \\    f[12] = math.rcpApproxF32(true, f[13]) + math.rcpApproxF32(false, f[14]);
        \\    f[15] = math.rsqrtApproxF32(true, f[16]) + math.rsqrtApproxF32(false, f[17]);
        \\    f[18] = math.sqrtApproxF32(true, f[19]) + math.sqrtApproxF32(false, f[20]);
        \\    f[21] = math.divApproxF32(true, f[22], f[23]) + math.divApproxF32(false, f[24], f[25]);
        \\    f[26] = math.divFullF32(true, f[27], f[28]) + math.divFullF32(false, f[29], f[30]);
        \\    d[0] = math.rcpApproxFtzF64(d[1]) + math.rsqrtApproxF64(d[2]) + math.rsqrtApproxFtzF64(d[3]);
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{
            "ex2.approx.ftz.f32 %f",
            "ex2.approx.f32 %f",
            "lg2.approx.ftz.f32 %f",
            "lg2.approx.f32 %f",
            "sin.approx.ftz.f32 %f",
            "sin.approx.f32 %f",
            "cos.approx.ftz.f32 %f",
            "cos.approx.f32 %f",
            "rcp.approx.ftz.f32 %f",
            "rcp.approx.f32 %f",
            "rsqrt.approx.ftz.f32 %f",
            "rsqrt.approx.f32 %f",
            "\tsqrt.approx.ftz.f32 %f",
            "\tsqrt.approx.f32 %f",
            "div.approx.ftz.f32 %f",
            "div.approx.f32 %f",
            "div.full.ftz.f32 %f",
            "div.full.f32 %f",
            "rcp.approx.ftz.f64 %fd",
            "rsqrt.approx.f64 %fd",
            "rsqrt.approx.ftz.f64 %fd",
        },
    });

    cases.addCase(.{
        .name = "std.nvptx.math approximate bf16",
        .source =
        \\const nvptx = @import("std").nvptx;
        \\const math = nvptx.math;
        \\export fn k(out: [*]u32, h: [*]u16) callconv(.kernel) void {
        \\    out[0] = @backingInt(math.ex2Approx(nvptx.Bf16x2, @fromBackingInt(out[1])));
        \\    h[0] = math.ex2ApproxBf16(h[1]);
        \\}
        ,
        .cpu = "sm_90",
        .optimize = .fast,
        .matches = &.{
            "ex2.approx.ftz.bf16x2 %r",
            "ex2.approx.ftz.bf16 %rs",
        },
    });

    cases.addCase(.{
        .name = "std.nvptx.ptx_device_rt exponentials",
        .source =
        \\const rt = @import("std").nvptx.ptx_device_rt;
        \\export fn k(f: [*]f32) callconv(.kernel) void {
        \\    f[0] = rt.expf(f[1]);
        \\    f[2] = rt.exp2f(f[3]);
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{"fma.rn.f32"},
        .absent = &.{ ".f64", "ex2.approx", "call", ".func" },
    });

    cases.addCase(.{
        .name = "strong export takes the name of a weak export",
        .source =
        \\fn weak(x: u32) callconv(.c) u32 {
        \\    return x ^ 0x1111;
        \\}
        \\fn strong(x: u32) callconv(.c) u32 {
        \\    return x ^ 0x2222;
        \\}
        \\comptime {
        \\    @export(&weak, .{ .name = "pick", .linkage = .weak });
        \\    @export(&strong, .{ .name = "pick" });
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{ ".visible .func (.param .b32 func_retval0) pick(", "8738" },
        .absent = &.{"4369"},
    });

    cases.addCase(.{
        .name = "adjacent stores share vector stores",
        .source =
        \\const Mixed = extern struct { a: f32, b: u32, c: [2]u16, d: f32 };
        \\export fn literal(c: [*]addrspace(.global) f32, a: f32, b: f32) callconv(.kernel) void {
        \\    const p: *addrspace(.global) @Vector(4, f32) = @ptrCast(@alignCast(&c[@workItemId(0) * 4]));
        \\    p.* = .{ a, b, b, a };
        \\}
        \\export fn halves(c: [*]addrspace(.global) u16, x: [8]u16) callconv(.kernel) void {
        \\    const p: *align(16) addrspace(.global) [8]u16 = @ptrCast(@alignCast(&c[@workItemId(0) * 8]));
        \\    inline for (0..8) |i| p[i] = x[i] +% 1;
        \\}
        \\export fn mixed(c: [*]align(16) addrspace(.global) Mixed, a: f32, b: u32, h: u16) callconv(.kernel) void {
        \\    c[@workItemId(0)] = .{ .a = a, .b = b, .c = .{ h, h }, .d = a };
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{ "st.global.v4.f32 [", "st.global.v4.b16 [", "st.global.v2.b32 [", "st.global.v2.b16 [" },
        .absent = &.{"st.global.b16 ["},
    });

    cases.addCase(.{
        .name = "stores stay apart across a load of their memory or without alignment",
        .source =
        \\export fn reads(p: *align(16) addrspace(.global) [4]f32, a: f32) callconv(.kernel) void {
        \\    p.* = .{ a, p[0], a, a };
        \\}
        \\export fn unaligned(p: *addrspace(.global) [8]u16, a: u16) callconv(.kernel) void {
        \\    p.* = .{ a, a, a, a, a, a, a, a };
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .safe,
        .matches = &.{ "st.global.f32 [", "ld.global.f32", "st.global.v2.f32 [", "st.global.b16 [" },
        .absent = &.{ "st.global.v4", "st.global.v2.b16" },
    });

    cases.addCase(.{
        .name = "saturating shift of a narrow integer by a constant takes a u32 amount",
        .source =
        \\noinline fn sat(x: i38) i38 {
        \\    return x <<| @as(u32, 32768);
        \\}
        \\export fn k(out: *addrspace(.global) i64, x: i64) callconv(.kernel) void {
        \\    out.* = sat(@truncate(x));
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{"mov.b32 %r0, 38;\n\tshl.b64 %rd1, %rd0, %r0;"},
    });

    cases.addCase(.{
        .name = "dead code removal drops an unused result of an inlined function",
        .source = unused_result,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{"7777"},
        .absent = &.{"9999"},
    });

    cases.addCase(.{
        .name = "CSE in release modes gives two inlined addresses one register",
        .source = inlined_addresses,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{"st.global.v2.b32 ["},
        .absent = &.{"st.global.b32 ["},
    });

    cases.addCase(.{
        .name = "CSE keeps the zero moves of two loop accumulators apart",
        .source =
        \\export fn sums(out: *addrspace(.global) [2]f32, x: [*]addrspace(.global) const f32, n: u32) callconv(.kernel) void {
        \\    var a: f32 = 0;
        \\    var b: f32 = 0;
        \\    var i: u32 = 0;
        \\    while (i < n) : (i += 1) {
        \\        a += x[i];
        \\        b += x[i] * x[i];
        \\    }
        \\    out.* = .{ a, b };
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{"mov.b32 %f0, 0f00000000;\n\tmov.b32 %f1, 0f00000000;"},
        .absent = &.{"mov.f32 %f"},
    });

    cases.addCase(.{
        .name = "Debug mode runs only promotion and copy expansion",
        .source = inlined_addresses ++ unused_result,
        .cpu = "sm_80",
        .matches = &.{ "call (retval0), ", "mul.lo.u32 %r2, %r0, 9999;" },
        .absent = &.{"st.global.v2"},
    });

    const counted_loop =
        \\export fn k(out: [*]addrspace(.global) u32, n: u32) callconv(.kernel) void {
        \\    for (0..n) |i| {
        \\        const j: u32 = @intCast(i);
        \\        out[j] = j * 3;
        \\    }
        \\}
    ;

    cases.addCase(.{
        .name = "a loop counter with a 32-bit bound uses 32-bit registers",
        .source = counted_loop,
        .cpu = "sm_80",
        .optimize = .safe,
        .matches = &.{ "setp.ge.u32 %p", "add.u32 %r" },
        .absent = &.{ "setp.ge.u64", "setp.le.u64" },
    });

    cases.addCase(.{
        .name = "an increment under a test of a 64-bit value that does not fit stays 64-bit",
        .source =
        \\export fn k(out: [*]u32, a: [*]const u64) callconv(.kernel) void {
        \\    const v = a[0] +% a[1];
        \\    if (v < 100) out[0] = @truncate(v +% 1);
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{"setp.ge.u64 %p"},
    });

    cases.addCase(.{
        .name = "Debug mode keeps the 64-bit loop counter",
        .source = counted_loop,
        .cpu = "sm_80",
        .matches = &.{ "setp.lt.u64 %p", "setp.gt.u64 %p" },
    });

    cases.addCase(.{
        .name = "load of an aggregate above 128 bytes reads global memory in place",
        .source =
        \\export fn k(in: *addrspace(.global) const [64]u32, out: *addrspace(.global) u32) callconv(.kernel) void {
        \\    var sum: u32 = 0;
        \\    for (in.*) |v| sum +%= v;
        \\    out.* = sum;
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{"ld.global.b32 %r"},
        .absent = &.{".local"},
    });

    cases.addCase(.{
        .name = "a store to the source keeps the copy of a load above 128 bytes",
        .source =
        \\export fn k(p: *addrspace(.global) [64]u32, out: *addrspace(.global) u32) callconv(.kernel) void {
        \\    const before = p.*;
        \\    p[0] = 7;
        \\    var sum: u32 = 0;
        \\    for (before) |v| sum +%= v;
        \\    out.* = sum;
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{ ".local .align 4 .b8 local_0[256];", "ld.local.b32" },
    });

    cases.addCase(.{
        .name = "a barrier keeps the copy of a shared array above 128 bytes",
        .source =
        \\var tile: [64]u32 addrspace(.shared) = undefined;
        \\export fn k(out: *addrspace(.global) u32) callconv(.kernel) void {
        \\    const before = tile;
        \\    asm volatile ("bar.sync 0;" ::: .{ .memory = true });
        \\    var sum: u32 = 0;
        \\    for (before) |v| sum +%= v;
        \\    out.* = sum;
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{ ".local .align 4 .b8 local_0[256];", "bar.sync 0;", "ld.local.b32" },
    });

    cases.addCase(.{
        .name = "load of a local variable above 128 bytes reads the variable in place",
        .source =
        \\export fn k(in: *addrspace(.global) const [64]u32, out: *addrspace(.global) u32, i: u32) callconv(.kernel) void {
        \\    var a: [64]u32 = in.*;
        \\    a[i % 64] = 0;
        \\    const b = a;
        \\    var sum: u32 = 0;
        \\    for (b) |v| sum +%= v;
        \\    out.* = sum;
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{".local .align 4 .b8 local_0[256];"},
        .absent = &.{"local_1"},
    });

    cases.addCase(.{
        .name = "a store to a local variable keeps the copy of a load above 128 bytes",
        .source =
        \\export fn k(in: *addrspace(.global) const [64]u32, out: *addrspace(.global) u32, i: u32) callconv(.kernel) void {
        \\    var a: [64]u32 = in.*;
        \\    const b = a;
        \\    a[i % 64] = 0;
        \\    var sum: u32 = 0;
        \\    for (b) |v| sum +%= v;
        \\    out.* = sum +% a[0];
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{".local .align 4 .b8 local_1[256];"},
    });

    cases.addCase(.{
        .name = "an escaped local variable keeps the copy of a load above 128 bytes",
        .source =
        \\noinline fn touch(a: *[64]u32) void {
        \\    a[0] = 9;
        \\}
        \\export fn k(in: *addrspace(.global) const [64]u32, out: *addrspace(.global) u32) callconv(.kernel) void {
        \\    var a: [64]u32 = in.*;
        \\    const b = a;
        \\    touch(&a);
        \\    var sum: u32 = 0;
        \\    for (b) |v| sum +%= v;
        \\    out.* = sum +% a[0];
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{ ".local .align 4 .b8 local_1[256];", "call " },
    });

    cases.addCase(.{
        .name = "device function reads its parameter copy above 128 bytes in place",
        .source =
        \\const Medium = extern struct { values: [33]i32 };
        \\noinline fn total(p: Medium, i: u32) i32 {
        \\    var sum: i32 = p.values[i % 33];
        \\    for (p.values) |value| sum +%= value;
        \\    return sum;
        \\}
        \\export fn k(in: *addrspace(.global) const Medium, out: *addrspace(.global) i32, i: u32) callconv(.kernel) void {
        \\    out.* = total(in.*, i);
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{".local .align 4 .b8 local_0[132];"},
        .absent = &.{"local_1"},
    });

    cases.addCase(.{
        .name = "comptime array read at a runtime index goes to const memory",
        .source =
        \\const row: [16]u32 = .{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3 };
        \\const Pair = extern struct { a: u32, b: u16 };
        \\const pairs: [4]Pair = .{ .{ .a = 1, .b = 2 }, .{ .a = 3, .b = 4 }, .{ .a = 5, .b = 6 }, .{ .a = 7, .b = 8 } };
        \\export fn k(out: *addrspace(.global) [3]u32, i: u32) callconv(.kernel) void {
        \\    out[0] = row[i % 16];
        \\    const pair = pairs[i % 4];
        \\    out[1] = pair.a;
        \\    out[2] = pair.b;
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{ ".const .align 8 .b8 __const_anon_", "ld.const.b32 %r", "ld.const.b16 %rs" },
        .absent = &.{".local"},
    });

    cases.addCase(.{
        .name = "comptime array above 4096 bytes read at a runtime index goes to global memory",
        .source =
        \\const big: [1100]u32 = values: {
        \\    @setEvalBranchQuota(10000);
        \\    var values: [1100]u32 = undefined;
        \\    for (&values, 0..) |*v, i| v.* = @truncate(i *% 2654435761);
        \\    break :values values;
        \\};
        \\export fn k(out: *addrspace(.global) u32, i: u32) callconv(.kernel) void {
        \\    out.* = big[i % 1100];
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{ ".global .align 8 .b8 __anon_", "[4400] = {", "ld.global.b32 %r" },
        .absent = &.{ ".local", ".const" },
    });

    cases.addCase(.{
        .name = "comptime array of optionals read at a runtime index is chosen with selp",
        .source =
        \\const maybe: [4]?u32 = .{ 1, null, 3, null };
        \\export fn k(out: *addrspace(.global) u32, i: u32) callconv(.kernel) void {
        \\    out.* = maybe[i % 4] orelse 9;
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{"selp.b32"},
        .absent = &.{ ".const", ".local" },
    });

    // Kernels of an image generation model, in the modes that reach backend code that the other
    // cases do not reach. Each module exports its kernels when it is the root module.
    const kernel_modules = [_]struct { []const u8, []const u8, std.builtin.Optimize }{
        .{ "fp8", "sm_120f+ptx90", .debug },
        .{ "fp8", "sm_120f+ptx90", .safe },
        .{ "fp8", "sm_120f+ptx90", .fast },
        .{ "attention", "sm_80+ptx90", .debug },
        .{ "blit", "sm_80+ptx90", .safe },
        .{ "blit", "sm_80+ptx90", .fast },
        .{ "gemm_tma", "sm_120+ptx90", .fast },
    };
    inline for (kernel_modules) |module| {
        const stem, const cpu, const optimize = module;
        cases.addCase(.{
            .name = "kernel module " ++ stem ++ " in " ++ @tagName(optimize) ++ " mode",
            .path = "test/nvptx_kernels/" ++ stem ++ ".zig",
            .cpu = cpu,
            .optimize = optimize,
            .matches = &.{".visible .entry "},
        });
    }

    // Integer narrowing reads the loop test of each increment: a test through `and`, a compare
    // with a greater-than, a compare with an immediate above 2^32 and a test in a nested block.
    inline for (.{ .safe, .fast }) |optimize| {
        cases.addCase(.{
            .name = "loop tests that integer narrowing reads in " ++ @tagName(optimize) ++ " mode",
            .source =
            \\export fn k(out: [*]addrspace(.global) u32, n32: u32, m32: u32, x32: u32) callconv(.kernel) void {
            \\    const n: usize = n32;
            \\    const m: usize = m32;
            \\    var i: usize = 0;
            \\    while (i < n and i < m) : (i += 1) out[i] = 1;
            \\    var j: usize = 0;
            \\    while (n > j) : (j += 1) out[j] +%= 2;
            \\    var t: usize = 0;
            \\    while (t < n) {
            \\        if (out[t] > 3) {
            \\            if (out[t] > 9) out[t] = 0;
            \\        }
            \\        t += 1;
            \\    }
            \\    const wide: u64 = x32;
            \\    if (wide < 0x1_0000_0005) out[0] = 5;
            \\}
            ,
            .cpu = "sm_80",
            .optimize = optimize,
            .matches = &.{ "setp.le.u32 %p", "and.pred %p" },
        });
    }

    cases.addCase(.{
        .name = "an increment under a greater-than test keeps a 64-bit counter",
        .source =
        \\export fn k(out: [*]addrspace(.global) u32, n32: u32, lim32: u32) callconv(.kernel) void {
        \\    const n: usize = n32;
        \\    const lim: usize = lim32;
        \\    var v: usize = 0;
        \\    while (v < n) {
        \\        if (v > lim) {
        \\            out[v] = 1;
        \\            v += 1;
        \\        } else {
        \\            v += 2;
        \\        }
        \\    }
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{ "setp.le.u64 %p", "setp.ge.u64 %p" },
    });

    cases.addCase(.{
        .name = "a loop that exits on a compare from an earlier block",
        .source =
        \\export fn k(out: [*]addrspace(.global) u32, n32: u32, x32: u32) callconv(.kernel) void {
        \\    const n: usize = n32;
        \\    var i: usize = 0;
        \\    while (true) {
        \\        const more = i < n;
        \\        if (out[i] == 0) out[i] = 7;
        \\        if (more) {
        \\            i += 1;
        \\            continue;
        \\        }
        \\        break;
        \\    }
        \\    var count: usize = 0;
        \\    var x = x32;
        \\    while (x != 0) : (x >>= 1) count += 1;
        \\    out[0] = @intCast(count);
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{"setp.ge.u64 %p"},
    });

    cases.addCase(.{
        .name = "a loop with a conditional body and a barrier",
        .source =
        \\const std = @import("std");
        \\var scratch: [256]f32 addrspace(.shared) = undefined;
        \\export fn k(out: [*]addrspace(.global) f32, levels: u32) callconv(.kernel) void {
        \\    const tid: u32 = @intCast(@workItemId(0));
        \\    var active: u32 = @intCast(@workGroupSize(0) >> 1);
        \\    var stride: u32 = 1;
        \\    scratch[tid] = out[tid];
        \\    std.nvptx.barrierSyncAligned(.work_group, null);
        \\    for (1..levels) |_| {
        \\        if (tid < active) {
        \\            scratch[tid * 2] += scratch[tid * 2 + stride];
        \\            active >>= 1;
        \\            stride <<= 1;
        \\        }
        \\        std.nvptx.barrierSyncAligned(.work_group, null);
        \\    }
        \\    if (tid == 0) out[0] = scratch[0];
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{ "barrier.sync.aligned", "st.shared.f32" },
    });

    cases.addCase(.{
        .name = "a loop with a recursive call and a call through a function pointer",
        .source =
        \\fn depth(n: u32, acc: u64) u64 {
        \\    if (n == 0) return acc;
        \\    return depth(n - 1, acc *% 3 +% n) +% 1;
        \\}
        \\fn twice(x: u64) u64 {
        \\    return x *% 2;
        \\}
        \\fn thrice(x: u64) u64 {
        \\    return x *% 3;
        \\}
        \\export fn k(out: [*]addrspace(.global) u64, n: u32, pick: bool) callconv(.kernel) void {
        \\    const f: *const fn (u64) u64 = if (pick) &twice else &thrice;
        \\    for (0..n) |i| out[i] = f(depth(@intCast(i % 8), i));
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{ ".callprototype (.param .b64 _) _ (.param .b64 _);", "call (retval0), %rd" },
    });

    cases.addCase(.{
        .name = "a bit cast of each f64 in a loop",
        .source =
        \\export fn k(p: [*]addrspace(.global) const f64, out: [*]addrspace(.global) u64, n: u32) callconv(.kernel) void {
        \\    for (0..n) |i| out[i] = @as(u64, @bitCast(p[i])) ^ 0x8000_0000_0000_0000;
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{ "ld.global.f64 %fd", "xor.b64 %rd" },
    });

    const tied_loop =
        \\export fn k(out: *addrspace(.global) u32, n: u32) callconv(.kernel) void {
        \\    var acc: u32 = n;
        \\    for (0..n) |_| {
        \\        asm volatile ("add.u32 %[acc], %[acc], %[x];"
        \\            : [acc] "+r" (acc),
        \\            : [x] "r" (acc),
        \\        );
        \\    }
        \\    out.* = acc;
        \\}
    ;
    inline for (.{ .debug, .fast }) |optimize| {
        cases.addCase(.{
            .name = "an asm output tied to a value that the asm also reads in " ++ @tagName(optimize) ++ " mode",
            .source = tied_loop,
            .cpu = "sm_80",
            .optimize = optimize,
            .matches = &.{"add.u32 %r"},
        });
    }

    cases.addCase(.{
        .name = "a load of a struct with a bit-packed vector field",
        .source =
        \\const S = struct { v: @Vector(4, u4), w: u32 };
        \\export fn k(p: *addrspace(.global) const S, out: *addrspace(.global) u32) callconv(.kernel) void {
        \\    const s = p.*;
        \\    out.* = s.w +% @reduce(.Add, @as(@Vector(4, u32), s.v));
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{ "ld.global.b16 %rs", "ld.global.b32 %r" },
    });

    cases.addCase(.{
        .name = "an address space cast of the address of a global variable",
        .source =
        \\var total: u32 addrspace(.global) = 0;
        \\export fn k(v: u32) callconv(.kernel) void {
        \\    const p: *u32 = @addrSpaceCast(&total);
        \\    p.* +%= v;
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{"cvta.global.u64 %rd"},
    });

    cases.addCase(.{
        .name = "a volatile store and a release store stay apart",
        .source =
        \\export fn k(out: *addrspace(.global) volatile u32, flag: *addrspace(.global) u32, v: u32) callconv(.kernel) void {
        \\    out.* = v;
        \\    @atomicStore(u32, flag, v +% 1, .release);
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{ "st.volatile.global.b32", "st.release.gpu.global.b32" },
        .absent = &.{"st.global.v2"},
    });

    const volatile_forms = [_]struct {
        name: []const u8,
        source: []const u8,
        matches: []const []const u8,
        absent: []const []const u8,
    }{
        .{
            .name = "volatile memset of an array, unrolled",
            .source =
            \\export fn k(out: [*]addrspace(.global) u32) callconv(.kernel) void {
            \\    const dest: *volatile addrspace(.global) [4]u32 = @ptrCast(out);
            \\    @memset(dest, 1);
            \\    @memset(dest, 2);
            \\}
            ,
            .matches = &.{ "st.volatile.global.b32 [%rd0+12], %r0;", "st.volatile.global.b32 [%rd0+12], %r1;" },
            .absent = &.{"st.global"},
        },
        .{
            .name = "volatile memset of an array, loop",
            .source =
            \\export fn k(out: [*]addrspace(.global) u32) callconv(.kernel) void {
            \\    const dest: *volatile addrspace(.global) [2048]u32 = @ptrCast(out);
            \\    @memset(dest, 1);
            \\}
            ,
            .matches = &.{"st.volatile.global.b32"},
            .absent = &.{"st.global"},
        },
        .{
            .name = "volatile memset of a slice",
            .source =
            \\export fn k(out: [*]addrspace(.global) u32, n: usize) callconv(.kernel) void {
            \\    const dest: [*]volatile addrspace(.global) u32 = out;
            \\    @memset(dest[0..n], 1);
            \\}
            ,
            .matches = &.{"st.volatile.global.b32"},
            .absent = &.{"st.global"},
        },
        .{
            .name = "volatile memcpy of arrays",
            .source =
            \\export fn k(out: [*]addrspace(.global) u32, in: [*]addrspace(.global) u32) callconv(.kernel) void {
            \\    const dest: *volatile addrspace(.global) [4]u32 = @ptrCast(out);
            \\    const src: *volatile addrspace(.global) [4]u32 = @ptrCast(in);
            \\    @memcpy(dest, src);
            \\}
            ,
            .matches = &.{ "ld.volatile.global.b32", "st.volatile.global.b32" },
            .absent = &.{ "ld.global", "st.global" },
        },
        .{
            .name = "volatile memcpy of slices",
            .source =
            \\export fn k(out: [*]addrspace(.global) u32, in: [*]addrspace(.global) u32, n: usize) callconv(.kernel) void {
            \\    const dest: [*]volatile addrspace(.global) u32 = out;
            \\    const src: [*]volatile addrspace(.global) u32 = in;
            \\    @memcpy(dest[0..n], src[0..n]);
            \\}
            ,
            .matches = &.{ "ld.volatile.global.b32", "st.volatile.global.b32" },
            .absent = &.{ "ld.global", "st.global" },
        },
        .{
            .name = "volatile memmove",
            .source =
            \\export fn k(out: [*]addrspace(.global) u32, in: [*]addrspace(.global) u32, n: usize) callconv(.kernel) void {
            \\    const dest: [*]volatile addrspace(.global) u32 = out;
            \\    const src: [*]volatile addrspace(.global) u32 = in;
            \\    @memmove(dest[0..n], src[0..n]);
            \\}
            ,
            .matches = &.{ "ld.volatile.global.b32", "st.volatile.global.b32" },
            .absent = &.{ "ld.global", "st.global" },
        },
        .{
            .name = "memcpy from a volatile source to a plain destination",
            .source =
            \\export fn k(out: [*]addrspace(.global) u32, in: [*]addrspace(.global) u32) callconv(.kernel) void {
            \\    const dest: *addrspace(.global) [4]u32 = @ptrCast(out);
            \\    const src: *volatile addrspace(.global) [4]u32 = @ptrCast(in);
            \\    @memcpy(dest, src);
            \\}
            ,
            .matches = &.{ "ld.volatile.global.b32", "st.global.b32" },
            .absent = &.{ "ld.global", "st.volatile" },
        },
        .{
            .name = "memcpy from a plain source to a volatile destination",
            .source =
            \\export fn k(out: [*]addrspace(.global) u32, in: [*]addrspace(.global) u32) callconv(.kernel) void {
            \\    const dest: *volatile addrspace(.global) [4]u32 = @ptrCast(out);
            \\    const src: *addrspace(.global) [4]u32 = @ptrCast(in);
            \\    @memcpy(dest, src);
            \\}
            ,
            .matches = &.{ "ld.global.b32", "st.volatile.global.b32" },
            .absent = &.{ "ld.volatile", "st.global" },
        },
    };
    for (volatile_forms) |form| {
        for ([_]std.builtin.Optimize{ .debug, .fast }) |optimize| {
            cases.addCase(.{
                .name = form.name,
                .source = form.source,
                .cpu = "sm_80",
                .optimize = optimize,
                .matches = form.matches,
                .absent = form.absent,
            });
        }
    }

    const merge_prefix =
        \\const nvptx = @import("std").nvptx;
        \\var buf: [256]u32 align(16) addrspace(.shared) = undefined;
        \\export fn k(out: [*]addrspace(.global) u32) callconv(.kernel) void {
        \\    const t = @workItemId(0);
        \\    const ptr = &buf[(t + 1) % 256];
        \\    buf[t] = t;
        \\    nvptx.barrierSyncAligned(.work_group, null);
        \\    const before =
    ;
    const merge_middle =
        \\;
        \\    nvptx.barrierSyncAligned(.work_group, null);
        \\    buf[t] = t * 7;
        \\    nvptx.barrierSyncAligned(.work_group, null);
        \\    const after =
    ;
    const merge_suffix =
        \\;
        \\    out[2 * t] = before;
        \\    out[2 * t + 1] = after;
        \\}
    ;
    const merge_forms = [_]struct { []const u8, []const u8 }{
        .{ "a volatile asm", "asm volatile (\"ld.shared.u32 %[d], [%[a]];\" : [d] \"=r\" (-> u32) : [a] \"r\" (@as(u32, @intCast(@intFromPtr(ptr)))))" },
        .{ "an asm with a memory clobber", "asm (\"ld.shared.u32 %[d], [%[a]];\" : [d] \"=r\" (-> u32) : [a] \"r\" (@as(u32, @intCast(@intFromPtr(ptr)))) : .{ .memory = true })" },
        .{ "an asm with a pointer input", "asm (\"ld.shared.u32 %[d], [%[a]];\" : [d] \"=r\" (-> u32) : [a] \"r\" (ptr))" },
    };
    for (merge_forms) |entry| {
        const name, const load = entry;
        cases.addCase(.{
            .name = cases.b.fmt("{s} that reads shared memory is not merged across a barrier", .{name}),
            .source = cases.b.fmt("{s} {s} {s} {s} {s}", .{ merge_prefix, load, merge_middle, load, merge_suffix }),
            .cpu = "sm_80",
            .optimize = .fast,
            .in_order = &.{ "barrier.sync.aligned", "ld.shared.u32", "barrier.sync.aligned", "barrier.sync.aligned", "ld.shared.u32" },
        });
    }

    const parameter_phi_definitions = [_][]const u8{
        "mov.b64 %rd0,", "mov.b64 %rd1,", "mov.b64 %rd2,",
        "mov.b32 %r0,",  "mov.b32 %r1,",  "mov.b32 %r2,",
        "mov.b32 %r3,",  "mov.b32 %r4,",  "add.u32 %r3,",
        "add.u32 %r4,",
    };
    cases.addCase(.{
        .name = "a block parameter does not share the register of a kernel parameter",
        .source =
        \\const Arguments = extern struct { q: u64, k: u64, v: u64, q_stride: u32, k_stride: u32, v_stride: u32, q_heads: u32, kv_heads: u32 };
        \\const Head = enum { query, key, value };
        \\export fn k(arguments: Arguments, rows: u32) callconv(.kernel) void {
        \\    const row = @workItemId(0);
        \\    if (row >= rows) return;
        \\    const head = @workGroupId(0);
        \\    const kind: Head = if (head < arguments.q_heads) .query else if (head < arguments.q_heads + arguments.kv_heads) .key else .value;
        \\    const Target = struct { base: u64, stride: u32, first_head: u32 };
        \\    const target: Target = switch (kind) {
        \\        .query => .{ .base = arguments.q, .stride = arguments.q_stride, .first_head = 0 },
        \\        .key => .{ .base = arguments.k, .stride = arguments.k_stride, .first_head = arguments.q_heads },
        \\        .value => .{ .base = arguments.v, .stride = arguments.v_stride, .first_head = arguments.q_heads + arguments.kv_heads },
        \\    };
        \\    const element = @as(u64, row) * target.stride + (head - target.first_head);
        \\    const out: *addrspace(.global) u16 = @ptrFromInt(target.base + element * 2);
        \\    out.* = 1;
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{
            "ld.param.b64 %rd0, [param_0+0];",
            "ld.param.b64 %rd2, [param_0+16];",
            "ld.param.v2.b32 {%r2, %r3}, [param_0+32];",
            "ld.param.b32 %r4, [param_0+40];",
        },
        .absent = &parameter_phi_definitions,
    });

    cases.addCase(.{
        .name = "a union with members of different sizes lives in registers",
        .source =
        \\const U = extern union { w: u64, h: [2]u32, b: [8]u8 };
        \\export fn k(out: [*]addrspace(.global) u64, x: u64, y: u32, z: u8) callconv(.kernel) void {
        \\    var u: U = .{ .w = x };
        \\    u.h[1] = y;
        \\    u.b[2] = z;
        \\    out[0] = u.w;
        \\    out[1] = u.h[0];
        \\    out[2] = @bitCast(@as(i64, @as(i8, @bitCast(u.b[7]))));
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{ "shl.b64", "or.b64", "cvt.u16.u64", "mov.b64 {%r" },
        .absent = &.{ ".local", "st.u8", "ld.u8" },
    });

    cases.addCase(.{
        .name = "a bitcast between a double and two words uses no local variable",
        .source =
        \\export fn k(out: [*]addrspace(.global) u32, x: f64) callconv(.kernel) void {
        \\    const parts: [2]u32 = @bitCast(x);
        \\    out[0] = parts[0];
        \\    out[1] = parts[1];
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{"mov.b64 {%r0, %r1}, %fd0;"},
        .absent = &.{".local"},
    });

    cases.addCase(.{
        .name = "a pointer cast view of a local variable lives in registers in Debug mode",
        .source =
        \\export fn k(out: [*]addrspace(.global) u32, x: u64) callconv(.kernel) void {
        \\    var whole: u64 = x;
        \\    const words: *[2]u32 = @ptrCast(&whole);
        \\    words[1] +%= 7;
        \\    out[0] = @truncate(whole);
        \\    out[1] = words[1];
        \\}
        ,
        .cpu = "sm_80",
        .absent = &.{".local"},
    });

    cases.addCase(.{
        .name = "a generic pointer to a shared array gets shared accesses",
        .source =
        \\var buf: [2]f32 addrspace(.shared) = undefined;
        \\export fn viaManyPtr(out: *addrspace(.global) f32) callconv(.kernel) void {
        \\    const p: [*]addrspace(.shared) f32 = &buf;
        \\    out.* = p[1];
        \\}
        \\export fn viaArrayPtr(out: *addrspace(.global) f32) callconv(.kernel) void {
        \\    const p: *addrspace(.shared) [2]f32 = &buf;
        \\    out.* = p[1];
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{ "ld.shared.f32 %f0, [%r0+0];", ".visible .entry viaArrayPtr" },
        .absent = &.{ "cvta", "ld.f32", "ld.b32" },
    });

    cases.addCase(.{
        .name = "Debug mode keeps a generic pointer to a shared array generic",
        .source =
        \\var buf: [2]f32 addrspace(.shared) = undefined;
        \\export fn viaManyPtr(out: *addrspace(.global) f32) callconv(.kernel) void {
        \\    const p: [*]addrspace(.shared) f32 = &buf;
        \\    out.* = p[1];
        \\}
        \\export fn viaArrayPtr(out: *addrspace(.global) f32) callconv(.kernel) void {
        \\    const p: *addrspace(.shared) [2]f32 = &buf;
        \\    out.* = p[1];
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{ "cvta.shared.u64 %rd1, ", "ld.f32 %f0, [%rd1+0];", "ld.shared.f32 %f0, [%r0+0];" },
    });

    cases.addCase(.{
        .name = "a run-time chain of address space casts gets shared accesses",
        .source =
        \\export fn k(out: *addrspace(.global) u32, base: [*]addrspace(.shared) u32, i: usize) callconv(.kernel) void {
        \\    const p: [*]u32 = @addrSpaceCast(base);
        \\    p[i] = 7;
        \\    out.* = p[i + 1];
        \\    _ = @atomicRmw(u32, &p[i], .Add, 1, .monotonic);
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{ "st.shared.b32", "ld.shared.b32", "atom.relaxed.gpu.shared.add.u32" },
        .absent = &.{ "cvta", "ld.b32", "st.b32", "atom.relaxed.gpu.add" },
    });

    cases.addCase(.{
        .name = "a shared pointer that escapes gets one cvta at the use",
        .source =
        \\var buf: [8]u32 addrspace(.shared) = undefined;
        \\var sink: usize addrspace(.global) = 0;
        \\export fn k(out: *addrspace(.global) u32, i: u32) callconv(.kernel) void {
        \\    const p: *u32 = @addrSpaceCast(&buf[i & 7]);
        \\    p.* = 5;
        \\    sink = @intFromPtr(p);
        \\    out.* = p.*;
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{ "st.shared.b32 [%rd1+0], %r4;", "cvta.shared.u64 %rd3, %rd1;\n\tst.global.b64", "ld.shared.b32" },
        .absent = &.{ "cvta.shared.u64 %rd4", "ld.b32", "st.b32" },
    });

    cases.addCase(.{
        .name = "a merge of a shared and a global pointer stays generic",
        .source =
        \\var buf: [8]u32 addrspace(.shared) = undefined;
        \\var gbuf: [8]u32 addrspace(.global) = undefined;
        \\export fn k(out: *addrspace(.global) u32, i: u32, flag: bool) callconv(.kernel) void {
        \\    const a: *u32 = @addrSpaceCast(&buf[i & 7]);
        \\    const b: *u32 = @addrSpaceCast(&gbuf[i & 7]);
        \\    const p = if (flag) a else b;
        \\    out.* = p.*;
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{ "cvta.shared.u64", "cvta.global.u64", "selp.b64", "ld.b32 %r4, [%rd7+0];" },
        .absent = &.{"ld.shared"},
    });

    cases.addCase(.{
        .name = "an inlined generic callee gets shared accesses and a called one gets a cvta",
        .source =
        \\var buf: [64]f32 addrspace(.shared) = undefined;
        \\fn get(p: *const f32) f32 {
        \\    return p.*;
        \\}
        \\noinline fn getNoinline(p: *const f32) f32 {
        \\    return p.*;
        \\}
        \\export fn inlined(out: *addrspace(.global) f32, i: u32) callconv(.kernel) void {
        \\    out.* = get(@addrSpaceCast(&buf[i]));
        \\}
        \\export fn called(out: *addrspace(.global) f32, i: u32) callconv(.kernel) void {
        \\    out.* = getNoinline(@addrSpaceCast(&buf[i]));
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{ "ld.shared.f32 %f0, [%rd1+0];", "cvta.shared.u64 %rd2, %rd1;\n\t{\n\t.param .b64 param0;" },
    });

    cases.addCase(.{
        .name = "a local variable that escapes keeps local accesses and gets one cvta",
        .source =
        \\noinline fn bump(p: *f32) void {
        \\    p.* += 1;
        \\}
        \\export fn k(out: *addrspace(.global) f32, i: usize) callconv(.kernel) void {
        \\    var x: [4]f32 = .{ 1, 2, 3, 4 };
        \\    const p: [*]f32 = &x;
        \\    bump(&x[0]);
        \\    out.* = p[i & 3];
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{ "st.local.f32", "ld.local.f32", "cvta.local.u64 %rd6, %rd2;\n\tst.param.b64 [param0+0], %rd6;" },
        .absent = &.{ "cvta.local.u64 %rd7", "ld.f32 %f4" },
    });

    cases.addCase(.{
        .name = "Debug mode keeps the accesses of a local variable that escapes generic",
        .source =
        \\noinline fn bump(p: *f32) void {
        \\    p.* += 1;
        \\}
        \\export fn k(out: *addrspace(.global) f32, i: usize) callconv(.kernel) void {
        \\    var x: [4]f32 = .{ 1, 2, 3, 4 };
        \\    bump(&x[0]);
        \\    out.* = x[i & 3];
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{ "cvta.local.u64", "st.f32", "ld.f32" },
        .absent = &.{ "ld.local", "st.local" },
    });

    cases.addCase(.{
        .name = "an inlined atomic on a local variable becomes a local load and store",
        .source =
        \\fn bump(p: *u32, v: u32) u32 {
        \\    return @atomicRmw(u32, p, .Max, v, .seq_cst);
        \\}
        \\fn swap(p: *u64, expected: u64, new: u64) ?u64 {
        \\    return @cmpxchgStrong(u64, p, expected, new, .acq_rel, .acquire);
        \\}
        \\fn load(p: *const u32) u32 {
        \\    return @atomicLoad(u32, p, .acquire);
        \\}
        \\noinline fn sink(p: *u32, q: *u64) void {
        \\    p.* +%= 1;
        \\    q.* +%= 1;
        \\}
        \\export fn k(out: *addrspace(.global) u64, i: u32) callconv(.kernel) void {
        \\    var x: u32 = i;
        \\    var y: u64 = i;
        \\    sink(&x, &y);
        \\    const old = bump(&x, 7);
        \\    const prior = swap(&y, 3, 4) orelse 9;
        \\    out.* = old + load(&x) + prior;
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{ "max.u32", "setp.eq.b64", "selp.b64", "st.local.u32", "st.local.b64" },
        .absent = &.{ "atom", "ld.acquire", "ld.relaxed" },
    });

    cases.addCase(.{
        .name = "a copy out of shared memory through a generic pointer gets shared loads",
        .source =
        \\var tile: [16]u32 addrspace(.shared) = undefined;
        \\fn copyOut(out: *addrspace(.global) [16]u32, src: *const [16]u32) void {
        \\    out.* = src.*;
        \\}
        \\export fn k(out: *addrspace(.global) [16]u32) callconv(.kernel) void {
        \\    copyOut(out, @addrSpaceCast(&tile));
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{ "ld.shared.b32", "st.global.b32" },
        .absent = &.{ "cvta", "ld.b32" },
    });

    cases.addCase(.{
        .name = "a copy out of shared memory stays generic in Debug mode",
        .source =
        \\var tile: [16]u32 addrspace(.shared) = undefined;
        \\fn copyOut(out: *addrspace(.global) [16]u32, src: *const [16]u32) void {
        \\    out.* = src.*;
        \\}
        \\export fn k(out: *addrspace(.global) [16]u32) callconv(.kernel) void {
        \\    copyOut(out, @addrSpaceCast(&tile));
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{ "cvta.shared.u64", "ld.b32", "st.global.b32" },
        .absent = &.{"ld.shared"},
    });

    cases.addCase(.{
        .name = "an inlined function from an object infers the state space of its access",
        .library =
        \\export fn first(p: *const f32) f32 {
        \\    return p.*;
        \\}
        ,
        .source =
        \\extern fn first(p: *const f32) f32;
        \\var tile: [64]f32 addrspace(.shared) = undefined;
        \\export fn k(out: *addrspace(.global) f32, i: u32) callconv(.kernel) void {
        \\    out.* = first(@addrSpaceCast(&tile[i & 63]));
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{"ld.shared.f32"},
        .absent = &.{ "call", "cvta" },
    });

    cases.addCase(.{
        .name = "a memset of a slice is one loop without unrolling hints",
        .source =
        \\export fn k(out: [*]addrspace(.global) f32, n: usize) callconv(.kernel) void {
        \\    @memset(out[0..n], 1.5);
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{ "setp.ge.u64", "st.global.f32" },
        .absent = &.{"nounroll"},
    });

    cases.addCase(.{
        .name = "a long memset of an array is a loop that ptxas does not unroll",
        .source =
        \\export fn k(out: *addrspace(.global) [64]u32) callconv(.kernel) void {
        \\    @memset(out, 0);
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .in_order = &.{ ".pragma \"nounroll\";", "setp.ge.u64", "st.global.b32" },
    });

    cases.addCase(.{
        .name = "a memset of a local array leaves no local memory",
        .source =
        \\export fn k(out: *addrspace(.global) u32, x: u32) callconv(.kernel) void {
        \\    var a: [4]u32 = undefined;
        \\    @memset(&a, 7);
        \\    a[1] = x;
        \\    out.* = a[0] + a[1] + a[3];
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{"st.global.b32"},
        .absent = &.{".local"},
    });

    cases.addCase(.{
        .name = "a memmove of slices checks the direction and has a loop for each direction",
        .source =
        \\export fn k(out: [*]addrspace(.global) u32, n: usize) callconv(.kernel) void {
        \\    @memmove(out[1 .. n + 1], out[0..n]);
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .in_order = &.{ "setp.hi.u64", "ld.global.b32", "st.global.b32", "setp.eq.u64", "sub.u64", "ld.global.b32", "st.global.b32" },
    });

    cases.addCase(.{
        .name = "a short memmove loads every unit before the first store",
        .source =
        \\export fn k(out: *addrspace(.global) [6]u16) callconv(.kernel) void {
        \\    @memmove(out[1..5], out[0..4]);
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{"ld.global.b16 %rs3, [%rd0+6];\n\tst.global.b16 [%rd1+0], %rs0;"},
        .absent = &.{"setp"},
    });

    cases.addCase(.{
        .name = "integer arithmetic on a shared address leaves the generic value",
        .source =
        \\var tile: [16]u32 addrspace(.shared) = undefined;
        \\noinline fn tileBase() [*]u32 {
        \\    return @addrSpaceCast(@as([*]addrspace(.shared) u32, &tile));
        \\}
        \\export fn k(out: [*]addrspace(.global) u32) callconv(.kernel) void {
        \\    const p: *u32 = @addrSpaceCast(&tile[out[0] & 15]);
        \\    p.* = 42;
        \\    const offset = @intFromPtr(p) - @intFromPtr(tileBase());
        \\    out[1] = @truncate(offset);
        \\    out[2] = @truncate(offset >> 32);
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{ "st.shared.b32 [%rd1+0]", "cvta.shared.u64 %rd3, %rd1;\n\tsub.u64 %rd4, %rd3, %rd2;" },
    });

    cases.addCase(.{
        .name = "a memmove between a shared pointer and its generic alias compares 64-bit addresses",
        .source =
        \\var bytes: [256]u8 addrspace(.shared) = undefined;
        \\export fn k(out: [*]addrspace(.global) u32, n: usize) callconv(.kernel) void {
        \\    const s: [*]addrspace(.shared) u8 = &bytes;
        \\    const g: [*]u8 = @addrSpaceCast(s);
        \\    @memmove(s[1 .. n + 1], g[0..n]);
        \\    out[0] = bytes[n];
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{"cvt.u64.u32 %rd5, %r0;\n\tsetp.hi.u64 %p0, %rd5, %rd2;"},
    });

    for ([_]std.builtin.Optimize{ .debug, .fast }) |optimize| {
        cases.addCase(.{
            .name = "a volatile copy into a local variable keeps each volatile load",
            .source =
            \\export fn k(out: *addrspace(.global) u32, in: *addrspace(.global) volatile [4]u32) callconv(.kernel) void {
            \\    var local: [4]u32 = undefined;
            \\    @memcpy(&local, in);
            \\    out.* = local[1];
            \\}
            ,
            .cpu = "sm_80",
            .optimize = optimize,
            .in_order = &.{ "ld.volatile.global.b32", "ld.volatile.global.b32", "ld.volatile.global.b32", "ld.volatile.global.b32" },
        });
    }

    cases.addCase(.{
        .name = "a memmove between shared and global memory needs no direction check",
        .source =
        \\var tile: [256]u32 addrspace(.shared) = undefined;
        \\export fn k(out: [*]addrspace(.global) u32, n: usize) callconv(.kernel) void {
        \\    @memmove(out[0..n], tile[0..n]);
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{ "ld.shared.b32", "st.global.b32" },
        .absent = &.{"setp.hi"},
    });

    cases.addCase(.{
        .name = "a write between a load above 128 bytes and its use keeps the copy",
        .source =
        \\const Big = extern struct { v: [64]u32 };
        \\var tile: Big addrspace(.shared) = undefined;
        \\export fn rmw_between(out: *addrspace(.global) Big, in: *addrspace(.global) const Big, flag: *addrspace(.global) u32) callconv(.kernel) void {
        \\    const v = in.*;
        \\    _ = @atomicRmw(u32, flag, .Add, 1, .monotonic);
        \\    out.* = v;
        \\}
        \\export fn store_between(out: *addrspace(.global) Big, in: *addrspace(.global) const Big, flag: *addrspace(.global) u32) callconv(.kernel) void {
        \\    const v = in.*;
        \\    @atomicStore(u32, flag, 1, .monotonic);
        \\    out.* = v;
        \\}
        \\export fn exchange_between(out: *addrspace(.global) Big, in: *addrspace(.global) const Big, flag: *addrspace(.global) u32) callconv(.kernel) void {
        \\    const v = in.*;
        \\    _ = @cmpxchgStrong(u32, flag, 0, 1, .monotonic, .monotonic);
        \\    out.* = v;
        \\}
        \\export fn local_source(out: *addrspace(.global) Big, flag: *addrspace(.global) u32, x: u32) callconv(.kernel) void {
        \\    var local: Big = undefined;
        \\    for (&local.v, 0..) |*e, i| e.* = x +% @as(u32, @intCast(i));
        \\    const v = local;
        \\    _ = @atomicRmw(u32, flag, .Add, 1, .monotonic);
        \\    @atomicStore(u32, flag, 1, .monotonic);
        \\    _ = @cmpxchgStrong(u32, flag, 0, 1, .monotonic, .monotonic);
        \\    out.* = v;
        \\}
        \\export fn generic_store(out: *Big, flag: *u32) callconv(.kernel) void {
        \\    const v = tile;
        \\    flag.* = 1;
        \\    out.* = v;
        \\}
        \\export fn asm_clobber(out: *addrspace(.global) Big, in: *addrspace(.global) const Big) callconv(.kernel) void {
        \\    const v = in.*;
        \\    _ = asm ("mov.u32 %[r], 1;" : [r] "=r" (-> u32) :: .{ .memory = true });
        \\    out.* = v;
        \\}
        \\export fn asm_output(out: *addrspace(.global) Big, in: *addrspace(.global) const Big, flag: *addrspace(.global) u32) callconv(.kernel) void {
        \\    const v = in.*;
        \\    var r: u32 = undefined;
        \\    asm ("mov.u32 %[r], 1;" : [r] "=r" (r));
        \\    flag.* = r;
        \\    out.* = v;
        \\}
        \\export fn asm_pointer(out: *addrspace(.global) Big, in: *addrspace(.global) const Big, flag: *addrspace(.global) u32) callconv(.kernel) void {
        \\    const v = in.*;
        \\    _ = asm ("mov.u32 %[r], 1;" : [r] "=r" (-> u32) : [p] "l" (flag));
        \\    out.* = v;
        \\}
        \\export fn asm_plain(out: *addrspace(.global) Big, in: *addrspace(.global) const Big, x: u32) callconv(.kernel) void {
        \\    const v = in.*;
        \\    _ = asm ("mov.u32 %[r], %[x];" : [r] "=r" (-> u32) : [x] "r" (x));
        \\    out.* = v;
        \\}
        \\export fn switch_between(out: *addrspace(.global) Big, in: *addrspace(.global) const Big, flag: *addrspace(.global) u32, x: u32) callconv(.kernel) void {
        \\    const v = in.*;
        \\    switch (x) {
        \\        0 => flag.* = 1,
        \\        1 => flag.* = 2,
        \\        else => {},
        \\    }
        \\    if (x > 7) flag.* = 3;
        \\    out.* = v;
        \\}
        ,
        .cpu = "sm_80",
        .in_order = &.{
            ".entry switch_between(",   ".local .align 4 .b8 local_0[256];",
            ".entry asm_plain(",        ".local .align 4 .b8 local_0[256];",
            ".entry asm_pointer(",      ".local .align 4 .b8 local_0[256];",
            ".entry asm_output(",       ".local .align 4 .b8 local_0[256];",
            ".entry asm_clobber(",      ".local .align 4 .b8 local_0[256];",
            ".entry generic_store(",    ".local .align 4 .b8 local_0[256];",
            ".entry local_source(",     ".local .align 4 .b8 local_0[256];",
            ".entry exchange_between(", ".local .align 4 .b8 local_0[256];",
            ".entry store_between(",    ".local .align 4 .b8 local_0[256];",
            ".entry rmw_between(",      ".local .align 4 .b8 local_0[256];",
        },
        .absent = &.{"local_1"},
    });

    cases.addCase(.{
        .name = "a store to global memory between a load from shared memory and its use keeps no copy",
        .source =
        \\const Big = extern struct { v: [64]u32 };
        \\var tile: Big addrspace(.shared) = undefined;
        \\export fn k(out: *addrspace(.global) Big, flag: *addrspace(.global) u32) callconv(.kernel) void {
        \\    const v = tile;
        \\    flag.* = 1;
        \\    out.* = v;
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{ "ld.shared.b32", "st.global.b32" },
        .absent = &.{".local"},
    });

    cases.addCase(.{
        .name = "a memset of elements with two scalars stores each scalar",
        .source =
        \\const Pair = extern struct { a: u32, b: u32 };
        \\export fn fill_slice(out: [*]addrspace(.global) Pair, n: u32) callconv(.kernel) void {
        \\    @memset(out[0..n], .{ .a = 1, .b = 2 });
        \\}
        \\export fn fill_array(out: *addrspace(.global) [3]Pair) callconv(.kernel) void {
        \\    @memset(out, .{ .a = 3, .b = 4 });
        \\}
        ,
        .cpu = "sm_80",
        .in_order = &.{
            ".entry fill_array(",
            "st.global.b32 [%rd0+0], %r",
            "st.global.b32 [%rd0+20], %r",
            ".entry fill_slice(",
            "setp.ge.u64",
            "st.global.b32 [%rd",
            "+0], %r",
            "st.global.b32 [%rd",
            "+4], %r",
            "add.u64",
            "bra",
        },
        .absent = &.{".local"},
    });

    cases.addCase(.{
        .name = "a memset with undefined stores nothing in a release mode",
        .source =
        \\export fn k(out: *addrspace(.global) [8]u32) callconv(.kernel) void {
        \\    @memset(out, undefined);
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{".entry k("},
        .absent = &.{ "st.global", "bra" },
    });

    cases.addCase(.{
        .name = "an atomic store of undefined stores nothing in a release mode",
        .source =
        \\export fn k(out: *addrspace(.global) u32) callconv(.kernel) void {
        \\    @atomicStore(u32, out, undefined, .release);
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{".entry k("},
        .absent = &.{ "st.", "fence" },
    });

    cases.addCase(.{
        .name = "an address space cast to the same space converts nothing",
        .source =
        \\export fn k(out: *u32) callconv(.kernel) void {
        \\    const same: *u32 = @addrSpaceCast(out);
        \\    same.* = 1;
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{ ".entry k(", "st.b32 [%rd0+0], %r0;" },
        .absent = &.{"cvta"},
    });

    cases.addCase(.{
        .name = "a copy of 16-byte elements aligned to 8 bytes moves two units per element",
        .source =
        \\const Pair = extern struct { a: u64, b: u64 };
        \\export fn k(out: [*]addrspace(.global) Pair, in: [*]addrspace(.global) const Pair, n: u32) callconv(.kernel) void {
        \\    @memcpy(out[0..n], in[0..n]);
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .matches = &.{ "mul.lo.u64 %rd3, %rd2, 2;", "ld.global.b64", "st.global.b64" },
        .absent = &.{".v2"},
    });

    cases.addCase(.{
        .name = "stores beyond the immediate offset limit stay scalar",
        .source =
        \\export fn far(out: [*]align(16) addrspace(.global) u32, x: u32) callconv(.kernel) void {
        \\    const p = out + 60_000_000;
        \\    const q = p + 60_000_000;
        \\    q[0] = x;
        \\    q[1] = x +% 1;
        \\}
        \\export fn near(out: [*]align(16) addrspace(.global) u32, x: u32) callconv(.kernel) void {
        \\    out[0] = x;
        \\    out[1] = x +% 1;
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .in_order = &.{
            ".entry near(",
            "st.global.v2.b32",
            ".entry far(",
            ", 240000000;",
            ", 240000000;",
            "st.global.b32 [%rd",
            "st.global.b32 [%rd",
        },
    });

    cases.addCase(.{
        .name = "an exported variable without a function",
        .source =
        \\export var total: u32 addrspace(.global) = 7;
        ,
        .cpu = "sm_80",
        .matches = &.{".visible .global .align 4 .b8 total[4] = {7,0,0,0};"},
        .absent = &.{ ".entry", ".func" },
    });

    cases.addCase(.{
        .name = "an element of a vector parameter above 128 bytes is read from param space",
        .source =
        \\export fn k(out: *addrspace(.global) u32, v: @Vector(64, u32)) callconv(.kernel) void {
        \\    out.* = v[3];
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{ "mov.u64 %rd1, param_1;", "ld.param.b32 %r0, [%rd1+12];" },
        .absent = &.{ ".local", "ld.param.v4" },
    });

    cases.addCase(.{
        .name = "division and remainder by a negative or positive power of two in a release mode",
        .source =
        \\export fn k(out: [*]addrspace(.global) i32, x: i32) callconv(.kernel) void {
        \\    out[0] = @divTrunc(x, -8);
        \\    out[1] = @rem(x, 16);
        \\}
        ,
        .cpu = "sm_80",
        .optimize = .fast,
        .in_order = &.{
            "shr.s32 %r1, %r0, 31;",
            "shr.u32 %r2, %r1, 29;",
            "add.s32 %r3, %r0, %r2;",
            "shr.s32 %r4, %r3, 3;",
            "neg.s32 %r5, %r4;",
            "and.b32 %r7, %r6, 15;",
            "shr.s32 %r9, %r8, 4;",
            "shl.b32 %r10, %r9, 4;",
            "sub.s32 %r11, %r0, %r10;",
        },
        .absent = &.{ "div.", "rem." },
    });

    cases.addCase(.{
        .name = "a library calls a function that the main source exports",
        .library =
        \\extern fn helper(x: u32) u32;
        \\export fn lib_kernel(out: *addrspace(.global) u32) callconv(.kernel) void {
        \\    out.* = helper(out.*);
        \\}
        ,
        .source =
        \\export fn helper(x: u32) u32 {
        \\    return x +% 1;
        \\}
        \\export fn k(out: *addrspace(.global) u32) callconv(.kernel) void {
        \\    out.* = helper(out.*);
        \\}
        ,
        .cpu = "sm_80",
        .in_order = &.{
            ".visible .func (.param .b32 func_retval0) helper(\n\t.param .b32 param_0);",
            ".entry lib_kernel(",
            "call (retval0), helper, (param0);",
            ".entry k(",
            "call (retval0), helper, (param0);",
            ".visible .func (.param .b32 func_retval0) helper(\n\t.param .b32 param_0)\n{",
        },
        .absent = &.{".extern"},
    });

    cases.addCase(.{
        .name = "two objects define the same function",
        .library =
        \\export fn helper(x: u32) u32 {
        \\    return x +% 1;
        \\}
        ,
        .more_libraries = &.{
            \\export fn helper(x: u32) u32 {
            \\    return x +% 2;
            \\}
        },
        .source =
        \\extern fn helper(x: u32) u32;
        \\export fn k(out: *addrspace(.global) u32) callconv(.kernel) void {
        \\    out.* = helper(out.*);
        \\}
        ,
        .cpu = "sm_80",
        .error_message = "error: duplicate symbol 'helper', which /?/ also defines",
    });

    cases.addCase(.{
        .name = "the main source and an object define the same function",
        .library =
        \\export fn helper(x: u32) u32 {
        \\    return x +% 1;
        \\}
        ,
        .source =
        \\export fn helper(x: u32) u32 {
        \\    return x +% 2;
        \\}
        \\export fn k(out: *addrspace(.global) u32) callconv(.kernel) void {
        \\    out.* = helper(out.*);
        \\}
        ,
        .cpu = "sm_80",
        .error_message = "error: duplicate symbol 'helper', which the Zig code and /?/ both define",
    });

    cases.addCase(.{
        .name = "an object replaces the function of an earlier archive",
        .archive =
        \\export fn helper(x: u32) u32 {
        \\    return x +% 1;
        \\}
        ,
        .library =
        \\export fn helper(x: u32) u32 {
        \\    return x +% 5;
        \\}
        ,
        .source =
        \\extern fn helper(x: u32) u32;
        \\export fn k(out: *addrspace(.global) u32) callconv(.kernel) void {
        \\    out.* = helper(out.*);
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{"add.u32 %r1, %r0, 5;"},
        .absent = &.{"add.u32 %r1, %r0, 1;"},
    });

    cases.addCase(.{
        .name = "an archive does not replace the function of an earlier object",
        .library =
        \\export fn helper(x: u32) u32 {
        \\    return x +% 5;
        \\}
        ,
        .later_archive =
        \\export fn helper(x: u32) u32 {
        \\    return x +% 1;
        \\}
        ,
        .source =
        \\extern fn helper(x: u32) u32;
        \\export fn k(out: *addrspace(.global) u32) callconv(.kernel) void {
        \\    out.* = helper(out.*);
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{"add.u32 %r1, %r0, 5;"},
        .absent = &.{"add.u32 %r1, %r0, 1;"},
    });

    cases.addCase(.{
        .name = "an object defines the variable that an earlier object declares",
        .library =
        \\extern var counter: u32;
        \\export fn bump() u32 {
        \\    counter += 1;
        \\    return counter;
        \\}
        ,
        .more_libraries = &.{
            \\export var counter: u32 = 7;
        },
        .source =
        \\extern fn bump() u32;
        \\export fn k(out: *addrspace(.global) u32) callconv(.kernel) void {
        \\    out.* = bump();
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{".visible .global .align 4 .b8 counter[4] = {7,0,0,0};"},
        .absent = &.{".extern"},
    });

    cases.addCase(.{
        .name = "an object replaces the variable of an earlier archive",
        .archive =
        \\export var counter: u32 = 9;
        ,
        .library =
        \\extern var counter: u32;
        \\export fn bump() u32 {
        \\    counter += 1;
        \\    return counter;
        \\}
        ,
        .more_libraries = &.{
            \\export var counter: u32 = 7;
        },
        .source =
        \\extern fn bump() u32;
        \\export fn k(out: *addrspace(.global) u32) callconv(.kernel) void {
        \\    out.* = bump();
        \\}
        ,
        .cpu = "sm_80",
        .matches = &.{".visible .global .align 4 .b8 counter[4] = {7,0,0,0};"},
        .absent = &.{"{9,0,0,0}"},
    });

    cases.addCase(.{
        .name = "an object for another CPU model cannot link",
        .library =
        \\export fn helper(x: u32) u32 {
        \\    return x +% 1;
        \\}
        ,
        .library_cpu = "sm_75",
        .source =
        \\extern fn helper(x: u32) u32;
        \\export fn k(out: *addrspace(.global) u32) callconv(.kernel) void {
        \\    out.* = helper(out.*);
        \\}
        ,
        .cpu = "sm_80",
        .error_message = "error: the module targets sm_75 with 64-bit addresses, but the output targets sm_80 with 64-bit addresses",
    });

    cases.addCase(.{
        .name = "a library exports a function under two names",
        .library =
        \\fn f(x: u32) callconv(.c) u32 {
        \\    return x *% 3;
        \\}
        \\comptime {
        \\    @export(&f, .{ .name = "f" });
        \\    @export(&f, .{ .name = "g" });
        \\}
        ,
        .source =
        \\extern fn g(x: u32) u32;
        \\export fn k(out: *addrspace(.global) u32) callconv(.kernel) void {
        \\    out.* = g(out.*);
        \\}
        ,
        .cpu = "sm_80",
        .in_order = &.{
            ".visible .func (.param .b32 func_retval0) f(\n\t.param .b32 param_0);",
            ".visible .func (.param .b32 func_retval0) g(\n\t.param .b32 param_0);",
            ".alias g, f;",
            "call (retval0), g, (param0);",
        },
        .absent = &.{".extern"},
    });
}

const unused_result =
    \\fn both(a: u32) [2]u32 {
    \\    return .{ a *% 7777, a *% 9999 };
    \\}
    \\export fn first(out: *addrspace(.global) u32, a: u32) callconv(.kernel) void {
    \\    out.* = both(a)[0];
    \\}
    \\
;

const inlined_addresses =
    \\fn elem(c: [*]align(8) addrspace(.global) [2]u32, i: u32) *align(8) addrspace(.global) [2]u32 {
    \\    return &c[i];
    \\}
    \\export fn pair(c: [*]align(8) addrspace(.global) [2]u32, a: u32, b: u32) callconv(.kernel) void {
    \\    const i = @workItemId(0);
    \\    elem(c, i)[0] = a;
    \\    elem(c, i)[1] = b;
    \\}
;

const fp8_mma =
    \\const nv = @import("std").nvptx;
    \\export fn k(out: [*]addrspace(.global) @Vector(4, f32), a: @Vector(4, u32), b: @Vector(2, u32), c: @Vector(4, f32)) callconv(.kernel) void {
    \\    inline for (.{ .e4m3, .e5m2 }, 0..) |a_element, i| {
    \\        inline for (.{ .e4m3, .e5m2 }, 0..) |b_element, j| {
    \\            out[i * 2 + j] = nv.mmaSync(.{
    \\                .shape = .m16n8k32,
    \\                .element = a_element,
    \\                .b_element = if (i == j) null else b_element,
    \\                .accumulator = .f32,
    \\            }, a, b, c);
    \\        }
    \\    }
    \\}
;

const fp8_block_scaled_mma =
    \\const nv = @import("std").nvptx;
    \\export fn k(out: [*]addrspace(.global) @Vector(4, f32), a: @Vector(4, u32), b: @Vector(2, u32), c: @Vector(4, f32), scale_a: u32, scale_b: u32) callconv(.kernel) void {
    \\    inline for (.{ .e4m3, .e5m2 }, 0..) |a_element, i| {
    \\        inline for (.{ .e4m3, .e5m2 }, 0..) |b_element, j| {
    \\            out[i * 2 + j] = nv.mmaSyncBlockScale(.{
    \\                .shape = .m16n8k32,
    \\                .element = a_element,
    \\                .b_element = if (i == j) null else b_element,
    \\                .accumulator = .f32,
    \\            }, if (i == 0 and j == 0) .{} else .{ .a_byte = 3, .a_thread = 1, .b_byte = 2, .b_thread = 3 }, a, b, c, scale_a, scale_b);
    \\        }
    \\    }
    \\}
;

const empty_kernel =
    \\export fn k() callconv(.kernel) void {}
;

const std = @import("std");
const tests = @import("tests.zig");
