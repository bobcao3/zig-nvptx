//! Each method emits the `mbarrier` instruction of its name (sections 9.7.15.16.12 to
//! 9.7.15.16.20). Arrivals release and waits acquire at CTA scope. A wait returns true when its
//! phase has completed. A system time limit can end `tryWaitParity` before the phase completes. The
//! methods assume that the counts of the arguments and of the barrier stay in the ranges of table
//! 43, that a token or parity names the current or the preceding phase, that `init` gets memory
//! without a valid barrier and `inval` a valid one, and that a `noComplete` arrival does not
//! complete the phase. `pendingCount` takes only the token of a `noComplete` arrival.

const nvptx = @import("../nvptx.zig");
const require = nvptx.require;

const supported: nvptx.Requirement = .{ .ptx = 70, .sm = .{ .base = 80 }, .threads = .thread };
const tx_count_requirement: nvptx.Requirement = .{
    .ptx = 80,
    .sm = .{ .base = 90 },
    .threads = .thread,
};

pub const Mbarrier = extern struct {
    state: u64 align(8),

    pub const ArrivalToken = enum(u64) {
        _,

        pub inline fn pendingCount(token: ArrivalToken) u32 {
            require("mbarrier.pending_count", supported);
            return asm ("mbarrier.pending_count.b64 %[d], %[token];"
                : [d] "=r" (-> u32),
                : [token] "l" (@backingInt(token)),
            );
        }
    };

    pub inline fn init(m: *addrspace(.shared) Mbarrier, count: u32) void {
        require("mbarrier.init", supported);
        asm volatile ("mbarrier.init.shared.b64 [%[m]], %[count];"
            :
            : [m] "r" (m),
              [count] "r" (count),
            : .{ .memory = true });
    }

    pub inline fn inval(m: *addrspace(.shared) Mbarrier) void {
        require("mbarrier.inval", supported);
        asm volatile ("mbarrier.inval.shared.b64 [%[m]];"
            :
            : [m] "r" (m),
            : .{ .memory = true });
    }

    pub inline fn arrive(m: *addrspace(.shared) Mbarrier) ArrivalToken {
        require("mbarrier.arrive", supported);
        const token = asm volatile ("mbarrier.arrive.shared.b64 %[token], [%[m]];"
            : [token] "=l" (-> u64),
            : [m] "r" (m),
            : .{ .memory = true });
        return @fromBackingInt(@intCast(token));
    }

    pub inline fn arriveNoComplete(m: *addrspace(.shared) Mbarrier, count: u32) ArrivalToken {
        require("mbarrier.arrive.noComplete", supported);
        const token = asm volatile ("mbarrier.arrive.noComplete.shared.b64 %[token], [%[m]], %[count];"
            : [token] "=l" (-> u64),
            : [m] "r" (m),
              [count] "r" (count),
            : .{ .memory = true });
        return @fromBackingInt(@intCast(token));
    }

    pub inline fn arriveDrop(m: *addrspace(.shared) Mbarrier) ArrivalToken {
        require("mbarrier.arrive_drop", supported);
        const token = asm volatile ("mbarrier.arrive_drop.shared.b64 %[token], [%[m]];"
            : [token] "=l" (-> u64),
            : [m] "r" (m),
            : .{ .memory = true });
        return @fromBackingInt(@intCast(token));
    }

    pub inline fn arriveDropNoComplete(m: *addrspace(.shared) Mbarrier, count: u32) ArrivalToken {
        require("mbarrier.arrive_drop.noComplete", supported);
        const token = asm volatile ("mbarrier.arrive_drop.noComplete.shared.b64 %[token], [%[m]], %[count];"
            : [token] "=l" (-> u64),
            : [m] "r" (m),
              [count] "r" (count),
            : .{ .memory = true });
        return @fromBackingInt(@intCast(token));
    }

    pub inline fn testWait(m: *addrspace(.shared) Mbarrier, token: ArrivalToken) bool {
        require("mbarrier.test_wait", supported);
        return asm volatile ("mbarrier.test_wait.shared.b64 %[complete], [%[m]], %[token];"
            : [complete] "=b" (-> bool),
            : [m] "r" (m),
              [token] "l" (@backingInt(token)),
            : .{ .memory = true });
    }

    pub inline fn testWaitParity(m: *addrspace(.shared) Mbarrier, parity: u1) bool {
        require("mbarrier.test_wait.parity", .{
            .ptx = 71,
            .sm = .{ .base = 80 },
            .threads = .thread,
        });
        return asm volatile ("mbarrier.test_wait.parity.shared.b64 %[complete], [%[m]], %[parity];"
            : [complete] "=b" (-> bool),
            : [m] "r" (m),
              [parity] "r" (@as(u32, parity)),
            : .{ .memory = true });
    }

    pub inline fn arriveExpectTx(m: *addrspace(.shared) Mbarrier, tx_count: u32) ArrivalToken {
        require("mbarrier.arrive.expect_tx", tx_count_requirement);
        const token = asm volatile ("mbarrier.arrive.expect_tx.shared.b64 %[token], [%[m]], %[tx_count];"
            : [token] "=l" (-> u64),
            : [m] "r" (m),
              [tx_count] "r" (tx_count),
            : .{ .memory = true });
        return @fromBackingInt(@intCast(token));
    }

    pub inline fn completeTx(m: *addrspace(.shared) Mbarrier, tx_count: u32) void {
        require("mbarrier.complete_tx", tx_count_requirement);
        asm volatile ("mbarrier.complete_tx.shared.b64 [%[m]], %[tx_count];"
            :
            : [m] "r" (m),
              [tx_count] "r" (tx_count),
            : .{ .memory = true });
    }

    /// The integer parity of `phase` selects the phase, so `phase` can be a count of phases.
    pub inline fn testWaitPhase(m: *addrspace(.shared) Mbarrier, phase: u32) bool {
        require("mbarrier.test_wait.parity", .{
            .ptx = 71,
            .sm = .{ .base = 80 },
            .threads = .thread,
        });
        return asm volatile ("mbarrier.test_wait.parity.shared.b64 %[complete], [%[m]], %[phase];"
            : [complete] "=b" (-> bool),
            : [m] "r" (m),
              [phase] "r" (phase),
            : .{ .memory = true });
    }

    /// The integer parity of `phase` selects the phase. `order` adds the `.sem.scope` qualifiers
    /// and `time_hint` the suspend time limit in nanoseconds.
    pub inline fn tryWaitPhase(
        m: *addrspace(.shared) Mbarrier,
        phase: u32,
        comptime order: ?WaitOrder,
        comptime time_hint: ?u32,
    ) bool {
        const ptx: u16 = if (order) |o| (if (o.semantics == .relaxed) 86 else 80) else 78;
        require("mbarrier.try_wait.parity", .{ .ptx = ptx, .sm = .{ .base = 90 }, .threads = .thread });
        const qualifiers = if (order) |o| "." ++ @tagName(o.semantics) ++ "." ++ @tagName(o.scope) else "";
        const text = "mbarrier.try_wait.parity" ++ qualifiers ++
            ".shared.b64 %[complete], [%[m]], %[phase]";
        if (time_hint) |hint| return asm volatile (text ++ ", %[hint];"
            : [complete] "=b" (-> bool),
            : [m] "r" (m),
              [phase] "r" (phase),
              [hint] "n" (hint),
            : .{ .memory = true });
        return asm volatile (text ++ ";"
            : [complete] "=b" (-> bool),
            : [m] "r" (m),
              [phase] "r" (phase),
            : .{ .memory = true });
    }

    pub inline fn tryWaitParity(m: *addrspace(.shared) Mbarrier, parity: u1) bool {
        require("mbarrier.try_wait.parity", .{
            .ptx = 78,
            .sm = .{ .base = 90 },
            .threads = .thread,
        });
        return asm volatile ("mbarrier.try_wait.parity.shared.b64 %[complete], [%[m]], %[parity];"
            : [complete] "=b" (-> bool),
            : [m] "r" (m),
              [parity] "r" (@as(u32, parity)),
            : .{ .memory = true });
    }
};

pub const BarrierScope = enum { cta, cluster };

pub const WaitOrder = struct {
    semantics: enum { acquire, relaxed },
    scope: BarrierScope,
};

pub const ArriveOrder = struct {
    semantics: enum { release, relaxed },
    scope: BarrierScope,
};

fn arriveQualifiers(comptime order: ?ArriveOrder) []const u8 {
    return if (order) |o| "." ++ @tagName(o.semantics) ++ "." ++ @tagName(o.scope) else "";
}

fn arriveRequirement(comptime order: ?ArriveOrder) nvptx.Requirement {
    const relaxed = if (order) |o| o.semantics == .relaxed else false;
    return .{ .ptx = if (relaxed) 86 else 80, .sm = .{ .base = 90 }, .threads = .thread };
}

/// Emits `mbarrier.arrive.shared::cluster` (section 9.7.15.16.16) on the barrier at the
/// `.shared::cluster` address `address`. Without `order`, PTX assumes `.release.cta`.
pub inline fn arriveCluster(address: u32, comptime order: ?ArriveOrder) void {
    require("mbarrier.arrive.shared::cluster", arriveRequirement(order));
    asm volatile ("mbarrier.arrive" ++ arriveQualifiers(order) ++ ".shared::cluster.b64 _, [%[a]];"
        :
        : [a] "r" (address),
        : .{ .memory = true });
}

/// Emits `mbarrier.arrive.expect_tx.shared::cluster` (section 9.7.15.16.16) on the barrier at the
/// `.shared::cluster` address `address`. Without `order`, PTX assumes `.release.cta`.
pub inline fn arriveExpectTxCluster(
    address: u32,
    tx_count: u32,
    comptime order: ?ArriveOrder,
) void {
    require("mbarrier.arrive.expect_tx.shared::cluster", arriveRequirement(order));
    asm volatile ("mbarrier.arrive.expect_tx" ++ arriveQualifiers(order) ++
            ".shared::cluster.b64 _, [%[a]], %[tx_count];"
        :
        : [a] "r" (address),
          [tx_count] "rn" (tx_count),
        : .{ .memory = true });
}

/// Emits `mbarrier.complete_tx.shared::cluster` (section 9.7.15.16.15) on the barrier at the
/// `.shared::cluster` address `address`. `scope` adds `.relaxed` with that scope.
pub inline fn completeTxCluster(address: u32, tx_count: u32, comptime scope: ?BarrierScope) void {
    require("mbarrier.complete_tx.shared::cluster", tx_count_requirement);
    const qualifiers = if (scope) |s| ".relaxed." ++ @tagName(s) else "";
    const text = "mbarrier.complete_tx" ++ qualifiers ++ ".shared::cluster.b64 [%[a]], %[tx_count];";
    asm volatile (text
        :
        : [a] "r" (address),
          [tx_count] "rn" (tx_count),
        : .{ .memory = true });
}
