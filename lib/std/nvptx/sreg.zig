//! Special registers, execution control instructions and the device functions that the driver
//! provides.

const std = @import("../std.zig");
const nvptx = @import("../nvptx.zig");

/// Reads `%smid`, the index of the multiprocessor that runs the work item (section 10.8). The
/// index is not contiguous and can change while the kernel runs.
pub inline fn smid() u32 {
    return asm volatile ("mov.u32 %[d], %smid;"
        : [d] "=r" (-> u32),
    );
}

pub inline fn nsmid() u32 {
    nvptx.require("nsmid", .{ .ptx = 20, .sm = .{ .base = 20 }, .threads = .thread });
    return asm ("mov.u32 %[d], %nsmid;"
        : [d] "=r" (-> u32),
    );
}

/// Reads `%warpid`, the index of the hardware warp slot that runs the work item (section 10.4).
/// The index can change while the kernel runs.
pub inline fn warpid() u32 {
    return asm volatile ("mov.u32 %[d], %warpid;"
        : [d] "=r" (-> u32),
    );
}

pub inline fn nwarpid() u32 {
    nvptx.require("nwarpid", .{ .ptx = 20, .sm = .{ .base = 20 }, .threads = .thread });
    return asm ("mov.u32 %[d], %nwarpid;"
        : [d] "=r" (-> u32),
    );
}

/// Reads `%cluster_ctarank`, the rank of the CTA in its cluster (section 10.16).
pub inline fn clusterCtarank() u32 {
    nvptx.require("cluster_ctarank", .{ .ptx = 78, .sm = .{ .base = 90 }, .threads = .thread });
    return asm ("mov.u32 %[d], %cluster_ctarank;"
        : [d] "=r" (-> u32),
    );
}

/// Reads `%cluster_nctarank`, the number of CTAs in the cluster (section 10.17).
pub inline fn clusterNctarank() u32 {
    nvptx.require("cluster_nctarank", .{ .ptx = 78, .sm = .{ .base = 90 }, .threads = .thread });
    return asm ("mov.u32 %[d], %cluster_nctarank;"
        : [d] "=r" (-> u32),
    );
}

pub inline fn gridid() u64 {
    return asm ("mov.u64 %[d], %gridid;"
        : [d] "=l" (-> u64),
    );
}

pub inline fn lanemaskEq() u32 {
    nvptx.require("lanemaskEq", .{ .ptx = 20, .sm = .{ .base = 20 }, .threads = .thread });
    return asm ("mov.u32 %[d], %lanemask_eq;"
        : [d] "=r" (-> u32),
    );
}

pub inline fn lanemaskLe() u32 {
    nvptx.require("lanemaskLe", .{ .ptx = 20, .sm = .{ .base = 20 }, .threads = .thread });
    return asm ("mov.u32 %[d], %lanemask_le;"
        : [d] "=r" (-> u32),
    );
}

pub inline fn lanemaskLt() u32 {
    nvptx.require("lanemaskLt", .{ .ptx = 20, .sm = .{ .base = 20 }, .threads = .thread });
    return asm ("mov.u32 %[d], %lanemask_lt;"
        : [d] "=r" (-> u32),
    );
}

pub inline fn lanemaskGe() u32 {
    nvptx.require("lanemaskGe", .{ .ptx = 20, .sm = .{ .base = 20 }, .threads = .thread });
    return asm ("mov.u32 %[d], %lanemask_ge;"
        : [d] "=r" (-> u32),
    );
}

pub inline fn lanemaskGt() u32 {
    nvptx.require("lanemaskGt", .{ .ptx = 20, .sm = .{ .base = 20 }, .threads = .thread });
    return asm ("mov.u32 %[d], %lanemask_gt;"
        : [d] "=r" (-> u32),
    );
}

/// Reads `%clock_hi`, the upper 32 bits of `clock64` (section 10.23).
pub inline fn clockHi() u32 {
    nvptx.require("clockHi", .{ .ptx = 50, .sm = .{ .base = 20 }, .threads = .thread });
    return asm volatile ("mov.u32 %[d], %clock_hi;"
        : [d] "=r" (-> u32),
    );
}

/// Reads `%globaltimer`, a 64-bit nanosecond timer (section 10.28). The behavior is specific to the target
/// and the value is unspecified when the PTX is compiled for another target.
pub inline fn globaltimer() u64 {
    nvptx.require("globaltimer", .{ .ptx = 31, .sm = .{ .base = 30 }, .threads = .thread });
    return asm volatile ("mov.u64 %[d], %globaltimer;"
        : [d] "=l" (-> u64),
    );
}

pub inline fn globaltimerLo() u32 {
    nvptx.require("globaltimerLo", .{ .ptx = 31, .sm = .{ .base = 30 }, .threads = .thread });
    return asm volatile ("mov.u32 %[d], %globaltimer_lo;"
        : [d] "=r" (-> u32),
    );
}

pub inline fn globaltimerHi() u32 {
    nvptx.require("globaltimerHi", .{ .ptx = 31, .sm = .{ .base = 30 }, .threads = .thread });
    return asm volatile ("mov.u32 %[d], %globaltimer_hi;"
        : [d] "=r" (-> u32),
    );
}

/// Reads `%total_smem_size`. The register excludes the reserved region and counts in
/// multiples of the allocation unit of the target (section 10.30).
pub inline fn totalSmemSize() u32 {
    nvptx.require("totalSmemSize", .{ .ptx = 41, .sm = .{ .base = 20 }, .threads = .thread });
    return asm ("mov.u32 %[d], %total_smem_size;"
        : [d] "=r" (-> u32),
    );
}

/// Reads `%dynamic_smem_size` (section 10.32).
pub inline fn dynamicSmemSize() u32 {
    nvptx.require("dynamicSmemSize", .{ .ptx = 41, .sm = .{ .base = 20 }, .threads = .thread });
    return asm ("mov.u32 %[d], %dynamic_smem_size;"
        : [d] "=r" (-> u32),
    );
}

/// Reads `%reserved_smem_offset_begin` (section 10.29).
pub inline fn reservedSmemOffsetBegin() u32 {
    nvptx.require("reservedSmemOffsetBegin", .{ .ptx = 76, .sm = .{ .base = 80 }, .threads = .thread });
    return asm ("mov.u32 %[d], %reserved_smem_offset_begin;"
        : [d] "=r" (-> u32),
    );
}

pub inline fn reservedSmemOffsetEnd() u32 {
    nvptx.require("reservedSmemOffsetEnd", .{ .ptx = 76, .sm = .{ .base = 80 }, .threads = .thread });
    return asm ("mov.u32 %[d], %reserved_smem_offset_end;"
        : [d] "=r" (-> u32),
    );
}

pub inline fn reservedSmemOffsetCap() u32 {
    nvptx.require("reservedSmemOffsetCap", .{ .ptx = 76, .sm = .{ .base = 80 }, .threads = .thread });
    return asm ("mov.u32 %[d], %reserved_smem_offset_cap;"
        : [d] "=r" (-> u32),
    );
}

pub inline fn reservedSmemOffset(comptime index: u1) u32 {
    nvptx.require("reservedSmemOffset", .{ .ptx = 76, .sm = .{ .base = 80 }, .threads = .thread });
    return switch (index) {
        0 => asm ("mov.u32 %[d], %reserved_smem_offset_0;"
            : [d] "=r" (-> u32),
        ),
        1 => asm ("mov.u32 %[d], %reserved_smem_offset_1;"
            : [d] "=r" (-> u32),
        ),
    };
}

/// Reads `%envreg<index>` (section 10.27).
pub inline fn envreg(comptime index: u5) u32 {
    nvptx.require("envreg", .{ .ptx = 21, .sm = .{ .base = 20 }, .threads = .thread });
    return switch (index) {
        0 => asm volatile ("mov.u32 %[d], %envreg0;"
            : [d] "=r" (-> u32),
        ),
        1 => asm volatile ("mov.u32 %[d], %envreg1;"
            : [d] "=r" (-> u32),
        ),
        2 => asm volatile ("mov.u32 %[d], %envreg2;"
            : [d] "=r" (-> u32),
        ),
        3 => asm volatile ("mov.u32 %[d], %envreg3;"
            : [d] "=r" (-> u32),
        ),
        4 => asm volatile ("mov.u32 %[d], %envreg4;"
            : [d] "=r" (-> u32),
        ),
        5 => asm volatile ("mov.u32 %[d], %envreg5;"
            : [d] "=r" (-> u32),
        ),
        6 => asm volatile ("mov.u32 %[d], %envreg6;"
            : [d] "=r" (-> u32),
        ),
        7 => asm volatile ("mov.u32 %[d], %envreg7;"
            : [d] "=r" (-> u32),
        ),
        8 => asm volatile ("mov.u32 %[d], %envreg8;"
            : [d] "=r" (-> u32),
        ),
        9 => asm volatile ("mov.u32 %[d], %envreg9;"
            : [d] "=r" (-> u32),
        ),
        10 => asm volatile ("mov.u32 %[d], %envreg10;"
            : [d] "=r" (-> u32),
        ),
        11 => asm volatile ("mov.u32 %[d], %envreg11;"
            : [d] "=r" (-> u32),
        ),
        12 => asm volatile ("mov.u32 %[d], %envreg12;"
            : [d] "=r" (-> u32),
        ),
        13 => asm volatile ("mov.u32 %[d], %envreg13;"
            : [d] "=r" (-> u32),
        ),
        14 => asm volatile ("mov.u32 %[d], %envreg14;"
            : [d] "=r" (-> u32),
        ),
        15 => asm volatile ("mov.u32 %[d], %envreg15;"
            : [d] "=r" (-> u32),
        ),
        16 => asm volatile ("mov.u32 %[d], %envreg16;"
            : [d] "=r" (-> u32),
        ),
        17 => asm volatile ("mov.u32 %[d], %envreg17;"
            : [d] "=r" (-> u32),
        ),
        18 => asm volatile ("mov.u32 %[d], %envreg18;"
            : [d] "=r" (-> u32),
        ),
        19 => asm volatile ("mov.u32 %[d], %envreg19;"
            : [d] "=r" (-> u32),
        ),
        20 => asm volatile ("mov.u32 %[d], %envreg20;"
            : [d] "=r" (-> u32),
        ),
        21 => asm volatile ("mov.u32 %[d], %envreg21;"
            : [d] "=r" (-> u32),
        ),
        22 => asm volatile ("mov.u32 %[d], %envreg22;"
            : [d] "=r" (-> u32),
        ),
        23 => asm volatile ("mov.u32 %[d], %envreg23;"
            : [d] "=r" (-> u32),
        ),
        24 => asm volatile ("mov.u32 %[d], %envreg24;"
            : [d] "=r" (-> u32),
        ),
        25 => asm volatile ("mov.u32 %[d], %envreg25;"
            : [d] "=r" (-> u32),
        ),
        26 => asm volatile ("mov.u32 %[d], %envreg26;"
            : [d] "=r" (-> u32),
        ),
        27 => asm volatile ("mov.u32 %[d], %envreg27;"
            : [d] "=r" (-> u32),
        ),
        28 => asm volatile ("mov.u32 %[d], %envreg28;"
            : [d] "=r" (-> u32),
        ),
        29 => asm volatile ("mov.u32 %[d], %envreg29;"
            : [d] "=r" (-> u32),
        ),
        30 => asm volatile ("mov.u32 %[d], %envreg30;"
            : [d] "=r" (-> u32),
        ),
        31 => asm volatile ("mov.u32 %[d], %envreg31;"
            : [d] "=r" (-> u32),
        ),
    };
}

/// Suspends the work item for about `nanoseconds` with `nanosleep.u32` (section 9.7.21.2). The
/// duration is approximate.
pub inline fn nanosleep(nanoseconds: u32) void {
    nvptx.require("nanosleep", .{ .ptx = 63, .sm = .{ .base = 70 }, .threads = .thread });
    asm volatile ("nanosleep.u32 %[t];"
        :
        : [t] "r" (nanoseconds),
    );
}

pub inline fn pmevent(comptime event: u4) void {
    asm volatile ("pmevent %[e];"
        :
        : [e] "n" (event),
    );
}

pub inline fn pmeventMask(comptime mask: u16) void {
    nvptx.require("pmeventMask", .{ .ptx = 30, .sm = .{ .base = 20 }, .threads = .thread });
    asm volatile ("pmevent.mask %[m];"
        :
        : [m] "n" (mask),
    );
}

/// The driver implements these functions. A kernel that calls one of them gets an
/// `.extern .func` declaration.
pub extern fn vprintf(format: [*:0]const u8, arguments: ?*const anyopaque) callconv(.c) i32;
pub extern fn malloc(size: usize) callconv(.c) ?*anyopaque;
pub extern fn free(pointer: ?*anyopaque) callconv(.c) void;

/// The `assert` of CUDA C++ calls this function of the driver. The driver prints the message with
/// the block and the thread, and the launch fails with `CUDA_ERROR_ASSERT`.
pub const assertFail = @extern(*const fn (message: [*:0]const u8, file: [*:0]const u8, line: u32, function: [*:0]const u8, char_size: usize) callconv(.c) void, .{ .name = "__assertfail" });

fn ArgumentBuffer(comptime Args: type) type {
    const info = @typeInfo(Args).@"struct";
    var types: [info.field_names.len]type = undefined;
    for (info.field_types, 0..) |T, i| types[i] = if (T == f32) f64 else T;
    const attrs: [types.len]std.lang.Type.Struct.FieldAttributes = @splat(.{});
    return @Struct(.@"extern", null, info.field_names, &types, &attrs);
}

/// Prints through `vprintf` and returns its result. `args` is a tuple of integers of 32 or 64
/// bits, pointers and floats. The function widens `f32` to `f64`, as C does for variadic
/// arguments. The driver prints the text at the next synchronization of the host.
pub inline fn printf(comptime format: [:0]const u8, args: anytype) i32 {
    var buffer: ArgumentBuffer(@TypeOf(args)) = undefined;
    inline for (@typeInfo(@TypeOf(args)).@"struct".field_names) |name| {
        @field(buffer, name) = @field(args, name);
    }
    return vprintf(format, &buffer);
}
