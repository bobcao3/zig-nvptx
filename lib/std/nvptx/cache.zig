//! Cache control and cached loads and stores on global memory (section 9.7.10).

const std = @import("../std.zig");
const nvptx = @import("../nvptx.zig");
const require = nvptx.require;

/// The 64-bit value that `createpolicy` returns and that the `WithPolicy` functions take.
pub const CachePolicy = enum(u64) { _ };

pub const L2EvictionPriority = enum { evict_last, evict_normal, evict_first, evict_unchanged };

pub const L1EvictionPriority = enum {
    evict_normal,
    evict_unchanged,
    evict_first,
    evict_last,
    no_allocate,
};

pub const LoadOperator = enum {
    /// `.ca`: cache at all levels.
    cache_all_levels,
    /// `.cg`: cache in L2 only.
    cache_global_level,
    /// `.cs`: cache streaming.
    cache_streaming,
    /// `.lu`: last use.
    last_use,
    /// `.cv`: cache as volatile, so the load fetches the line again.
    cache_volatile,
};

pub const NonCoherentOperator = enum {
    /// `.ca`
    cache_all_levels,
    /// `.cg`
    cache_global_level,
    /// `.cs`
    cache_streaming,
};

pub const StoreOperator = enum {
    /// `.wb`: write back at all coherent levels.
    write_back,
    /// `.cg`: cache in L2 only.
    cache_global_level,
    /// `.cs`: cache streaming.
    cache_streaming,
    /// `.wt`: write through to system memory.
    write_through,
};

/// The caching qualifier of `ld`, either a cache operator or a priority for L1.
pub const LoadCaching = union(enum) {
    operator: LoadOperator,
    l1_eviction_priority: L1EvictionPriority,
};

/// The caching qualifier of `st`, either a cache operator or a priority for L1.
pub const StoreCaching = union(enum) {
    operator: StoreOperator,
    l1_eviction_priority: L1EvictionPriority,
};

inline fn loadB8(comptime caching: LoadCaching, address: usize) u8 {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .cache_all_levels => "ld.global.ca.b8 %[d], [%[a]];",
            .cache_global_level => "ld.global.cg.b8 %[d], [%[a]];",
            .cache_streaming => "ld.global.cs.b8 %[d], [%[a]];",
            .last_use => "ld.global.lu.b8 %[d], [%[a]];",
            .cache_volatile => "ld.global.cv.b8 %[d], [%[a]];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "ld.global.L1::evict_normal.b8 %[d], [%[a]];",
            .evict_unchanged => "ld.global.L1::evict_unchanged.b8 %[d], [%[a]];",
            .evict_first => "ld.global.L1::evict_first.b8 %[d], [%[a]];",
            .evict_last => "ld.global.L1::evict_last.b8 %[d], [%[a]];",
            .no_allocate => "ld.global.L1::no_allocate.b8 %[d], [%[a]];",
        },
    };
    const registers = asm volatile (text
        : [d] "=r" (-> u32),
        : [a] "l" (address),
        : .{ .memory = true });
    return @truncate(registers);
}

inline fn loadB8WithPolicy(comptime caching: LoadCaching, address: usize, policy: CachePolicy) u8 {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .cache_all_levels => "ld.global.ca.L2::cache_hint.b8 %[d], [%[a]], %[policy];",
            .cache_global_level => "ld.global.cg.L2::cache_hint.b8 %[d], [%[a]], %[policy];",
            .cache_streaming => "ld.global.cs.L2::cache_hint.b8 %[d], [%[a]], %[policy];",
            .last_use => "ld.global.lu.L2::cache_hint.b8 %[d], [%[a]], %[policy];",
            .cache_volatile => "ld.global.cv.L2::cache_hint.b8 %[d], [%[a]], %[policy];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "ld.global.L1::evict_normal.L2::cache_hint.b8 %[d], [%[a]], %[policy];",
            .evict_unchanged => "ld.global.L1::evict_unchanged.L2::cache_hint.b8 %[d], [%[a]], %[policy];",
            .evict_first => "ld.global.L1::evict_first.L2::cache_hint.b8 %[d], [%[a]], %[policy];",
            .evict_last => "ld.global.L1::evict_last.L2::cache_hint.b8 %[d], [%[a]], %[policy];",
            .no_allocate => "ld.global.L1::no_allocate.L2::cache_hint.b8 %[d], [%[a]], %[policy];",
        },
    };
    const registers = asm volatile (text
        : [d] "=r" (-> u32),
        : [a] "l" (address),
          [policy] "l" (@backingInt(policy)),
        : .{ .memory = true });
    return @truncate(registers);
}

inline fn loadNonCoherentB8(comptime operator: NonCoherentOperator, address: usize) u8 {
    const text = switch (operator) {
        .cache_all_levels => "ld.global.ca.nc.b8 %[d], [%[a]];",
        .cache_global_level => "ld.global.cg.nc.b8 %[d], [%[a]];",
        .cache_streaming => "ld.global.cs.nc.b8 %[d], [%[a]];",
    };
    const registers = asm volatile (text
        : [d] "=r" (-> u32),
        : [a] "l" (address),
        : .{ .memory = true });
    return @truncate(registers);
}

inline fn loadNonCoherentB8WithPolicy(comptime operator: NonCoherentOperator, address: usize, policy: CachePolicy) u8 {
    const text = switch (operator) {
        .cache_all_levels => "ld.global.ca.nc.L2::cache_hint.b8 %[d], [%[a]], %[policy];",
        .cache_global_level => "ld.global.cg.nc.L2::cache_hint.b8 %[d], [%[a]], %[policy];",
        .cache_streaming => "ld.global.cs.nc.L2::cache_hint.b8 %[d], [%[a]], %[policy];",
    };
    const registers = asm volatile (text
        : [d] "=r" (-> u32),
        : [a] "l" (address),
          [policy] "l" (@backingInt(policy)),
        : .{ .memory = true });
    return @truncate(registers);
}

inline fn storeB8(comptime caching: StoreCaching, address: usize, value: u8) void {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .write_back => "st.global.wb.b8 [%[a]], %[b];",
            .cache_global_level => "st.global.cg.b8 [%[a]], %[b];",
            .cache_streaming => "st.global.cs.b8 [%[a]], %[b];",
            .write_through => "st.global.wt.b8 [%[a]], %[b];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "st.global.L1::evict_normal.b8 [%[a]], %[b];",
            .evict_unchanged => "st.global.L1::evict_unchanged.b8 [%[a]], %[b];",
            .evict_first => "st.global.L1::evict_first.b8 [%[a]], %[b];",
            .evict_last => "st.global.L1::evict_last.b8 [%[a]], %[b];",
            .no_allocate => "st.global.L1::no_allocate.b8 [%[a]], %[b];",
        },
    };
    asm volatile (text
        :
        : [a] "l" (address),
          [b] "r" (@as(u32, value)),
        : .{ .memory = true });
}

inline fn storeB8WithPolicy(comptime caching: StoreCaching, address: usize, value: u8, policy: CachePolicy) void {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .write_back => "st.global.wb.L2::cache_hint.b8 [%[a]], %[b], %[policy];",
            .cache_global_level => "st.global.cg.L2::cache_hint.b8 [%[a]], %[b], %[policy];",
            .cache_streaming => "st.global.cs.L2::cache_hint.b8 [%[a]], %[b], %[policy];",
            .write_through => "st.global.wt.L2::cache_hint.b8 [%[a]], %[b], %[policy];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "st.global.L1::evict_normal.L2::cache_hint.b8 [%[a]], %[b], %[policy];",
            .evict_unchanged => "st.global.L1::evict_unchanged.L2::cache_hint.b8 [%[a]], %[b], %[policy];",
            .evict_first => "st.global.L1::evict_first.L2::cache_hint.b8 [%[a]], %[b], %[policy];",
            .evict_last => "st.global.L1::evict_last.L2::cache_hint.b8 [%[a]], %[b], %[policy];",
            .no_allocate => "st.global.L1::no_allocate.L2::cache_hint.b8 [%[a]], %[b], %[policy];",
        },
    };
    asm volatile (text
        :
        : [a] "l" (address),
          [b] "r" (@as(u32, value)),
          [policy] "l" (@backingInt(policy)),
        : .{ .memory = true });
}

inline fn loadB16(comptime caching: LoadCaching, address: usize) u16 {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .cache_all_levels => "ld.global.ca.b16 %[d], [%[a]];",
            .cache_global_level => "ld.global.cg.b16 %[d], [%[a]];",
            .cache_streaming => "ld.global.cs.b16 %[d], [%[a]];",
            .last_use => "ld.global.lu.b16 %[d], [%[a]];",
            .cache_volatile => "ld.global.cv.b16 %[d], [%[a]];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "ld.global.L1::evict_normal.b16 %[d], [%[a]];",
            .evict_unchanged => "ld.global.L1::evict_unchanged.b16 %[d], [%[a]];",
            .evict_first => "ld.global.L1::evict_first.b16 %[d], [%[a]];",
            .evict_last => "ld.global.L1::evict_last.b16 %[d], [%[a]];",
            .no_allocate => "ld.global.L1::no_allocate.b16 %[d], [%[a]];",
        },
    };
    const registers = asm volatile (text
        : [d] "=r" (-> u32),
        : [a] "l" (address),
        : .{ .memory = true });
    return @truncate(registers);
}

inline fn loadB16WithPolicy(comptime caching: LoadCaching, address: usize, policy: CachePolicy) u16 {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .cache_all_levels => "ld.global.ca.L2::cache_hint.b16 %[d], [%[a]], %[policy];",
            .cache_global_level => "ld.global.cg.L2::cache_hint.b16 %[d], [%[a]], %[policy];",
            .cache_streaming => "ld.global.cs.L2::cache_hint.b16 %[d], [%[a]], %[policy];",
            .last_use => "ld.global.lu.L2::cache_hint.b16 %[d], [%[a]], %[policy];",
            .cache_volatile => "ld.global.cv.L2::cache_hint.b16 %[d], [%[a]], %[policy];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "ld.global.L1::evict_normal.L2::cache_hint.b16 %[d], [%[a]], %[policy];",
            .evict_unchanged => "ld.global.L1::evict_unchanged.L2::cache_hint.b16 %[d], [%[a]], %[policy];",
            .evict_first => "ld.global.L1::evict_first.L2::cache_hint.b16 %[d], [%[a]], %[policy];",
            .evict_last => "ld.global.L1::evict_last.L2::cache_hint.b16 %[d], [%[a]], %[policy];",
            .no_allocate => "ld.global.L1::no_allocate.L2::cache_hint.b16 %[d], [%[a]], %[policy];",
        },
    };
    const registers = asm volatile (text
        : [d] "=r" (-> u32),
        : [a] "l" (address),
          [policy] "l" (@backingInt(policy)),
        : .{ .memory = true });
    return @truncate(registers);
}

inline fn loadNonCoherentB16(comptime operator: NonCoherentOperator, address: usize) u16 {
    const text = switch (operator) {
        .cache_all_levels => "ld.global.ca.nc.b16 %[d], [%[a]];",
        .cache_global_level => "ld.global.cg.nc.b16 %[d], [%[a]];",
        .cache_streaming => "ld.global.cs.nc.b16 %[d], [%[a]];",
    };
    const registers = asm volatile (text
        : [d] "=r" (-> u32),
        : [a] "l" (address),
        : .{ .memory = true });
    return @truncate(registers);
}

inline fn loadNonCoherentB16WithPolicy(comptime operator: NonCoherentOperator, address: usize, policy: CachePolicy) u16 {
    const text = switch (operator) {
        .cache_all_levels => "ld.global.ca.nc.L2::cache_hint.b16 %[d], [%[a]], %[policy];",
        .cache_global_level => "ld.global.cg.nc.L2::cache_hint.b16 %[d], [%[a]], %[policy];",
        .cache_streaming => "ld.global.cs.nc.L2::cache_hint.b16 %[d], [%[a]], %[policy];",
    };
    const registers = asm volatile (text
        : [d] "=r" (-> u32),
        : [a] "l" (address),
          [policy] "l" (@backingInt(policy)),
        : .{ .memory = true });
    return @truncate(registers);
}

inline fn storeB16(comptime caching: StoreCaching, address: usize, value: u16) void {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .write_back => "st.global.wb.b16 [%[a]], %[b];",
            .cache_global_level => "st.global.cg.b16 [%[a]], %[b];",
            .cache_streaming => "st.global.cs.b16 [%[a]], %[b];",
            .write_through => "st.global.wt.b16 [%[a]], %[b];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "st.global.L1::evict_normal.b16 [%[a]], %[b];",
            .evict_unchanged => "st.global.L1::evict_unchanged.b16 [%[a]], %[b];",
            .evict_first => "st.global.L1::evict_first.b16 [%[a]], %[b];",
            .evict_last => "st.global.L1::evict_last.b16 [%[a]], %[b];",
            .no_allocate => "st.global.L1::no_allocate.b16 [%[a]], %[b];",
        },
    };
    asm volatile (text
        :
        : [a] "l" (address),
          [b] "r" (@as(u32, value)),
        : .{ .memory = true });
}

inline fn storeB16WithPolicy(comptime caching: StoreCaching, address: usize, value: u16, policy: CachePolicy) void {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .write_back => "st.global.wb.L2::cache_hint.b16 [%[a]], %[b], %[policy];",
            .cache_global_level => "st.global.cg.L2::cache_hint.b16 [%[a]], %[b], %[policy];",
            .cache_streaming => "st.global.cs.L2::cache_hint.b16 [%[a]], %[b], %[policy];",
            .write_through => "st.global.wt.L2::cache_hint.b16 [%[a]], %[b], %[policy];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "st.global.L1::evict_normal.L2::cache_hint.b16 [%[a]], %[b], %[policy];",
            .evict_unchanged => "st.global.L1::evict_unchanged.L2::cache_hint.b16 [%[a]], %[b], %[policy];",
            .evict_first => "st.global.L1::evict_first.L2::cache_hint.b16 [%[a]], %[b], %[policy];",
            .evict_last => "st.global.L1::evict_last.L2::cache_hint.b16 [%[a]], %[b], %[policy];",
            .no_allocate => "st.global.L1::no_allocate.L2::cache_hint.b16 [%[a]], %[b], %[policy];",
        },
    };
    asm volatile (text
        :
        : [a] "l" (address),
          [b] "r" (@as(u32, value)),
          [policy] "l" (@backingInt(policy)),
        : .{ .memory = true });
}

inline fn loadB32(comptime caching: LoadCaching, address: usize) u32 {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .cache_all_levels => "ld.global.ca.b32 %[d], [%[a]];",
            .cache_global_level => "ld.global.cg.b32 %[d], [%[a]];",
            .cache_streaming => "ld.global.cs.b32 %[d], [%[a]];",
            .last_use => "ld.global.lu.b32 %[d], [%[a]];",
            .cache_volatile => "ld.global.cv.b32 %[d], [%[a]];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "ld.global.L1::evict_normal.b32 %[d], [%[a]];",
            .evict_unchanged => "ld.global.L1::evict_unchanged.b32 %[d], [%[a]];",
            .evict_first => "ld.global.L1::evict_first.b32 %[d], [%[a]];",
            .evict_last => "ld.global.L1::evict_last.b32 %[d], [%[a]];",
            .no_allocate => "ld.global.L1::no_allocate.b32 %[d], [%[a]];",
        },
    };
    const registers = asm volatile (text
        : [d] "=r" (-> u32),
        : [a] "l" (address),
        : .{ .memory = true });
    return registers;
}

inline fn loadB32WithPolicy(comptime caching: LoadCaching, address: usize, policy: CachePolicy) u32 {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .cache_all_levels => "ld.global.ca.L2::cache_hint.b32 %[d], [%[a]], %[policy];",
            .cache_global_level => "ld.global.cg.L2::cache_hint.b32 %[d], [%[a]], %[policy];",
            .cache_streaming => "ld.global.cs.L2::cache_hint.b32 %[d], [%[a]], %[policy];",
            .last_use => "ld.global.lu.L2::cache_hint.b32 %[d], [%[a]], %[policy];",
            .cache_volatile => "ld.global.cv.L2::cache_hint.b32 %[d], [%[a]], %[policy];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "ld.global.L1::evict_normal.L2::cache_hint.b32 %[d], [%[a]], %[policy];",
            .evict_unchanged => "ld.global.L1::evict_unchanged.L2::cache_hint.b32 %[d], [%[a]], %[policy];",
            .evict_first => "ld.global.L1::evict_first.L2::cache_hint.b32 %[d], [%[a]], %[policy];",
            .evict_last => "ld.global.L1::evict_last.L2::cache_hint.b32 %[d], [%[a]], %[policy];",
            .no_allocate => "ld.global.L1::no_allocate.L2::cache_hint.b32 %[d], [%[a]], %[policy];",
        },
    };
    const registers = asm volatile (text
        : [d] "=r" (-> u32),
        : [a] "l" (address),
          [policy] "l" (@backingInt(policy)),
        : .{ .memory = true });
    return registers;
}

inline fn loadNonCoherentB32(comptime operator: NonCoherentOperator, address: usize) u32 {
    const text = switch (operator) {
        .cache_all_levels => "ld.global.ca.nc.b32 %[d], [%[a]];",
        .cache_global_level => "ld.global.cg.nc.b32 %[d], [%[a]];",
        .cache_streaming => "ld.global.cs.nc.b32 %[d], [%[a]];",
    };
    const registers = asm volatile (text
        : [d] "=r" (-> u32),
        : [a] "l" (address),
        : .{ .memory = true });
    return registers;
}

inline fn loadNonCoherentB32WithPolicy(comptime operator: NonCoherentOperator, address: usize, policy: CachePolicy) u32 {
    const text = switch (operator) {
        .cache_all_levels => "ld.global.ca.nc.L2::cache_hint.b32 %[d], [%[a]], %[policy];",
        .cache_global_level => "ld.global.cg.nc.L2::cache_hint.b32 %[d], [%[a]], %[policy];",
        .cache_streaming => "ld.global.cs.nc.L2::cache_hint.b32 %[d], [%[a]], %[policy];",
    };
    const registers = asm volatile (text
        : [d] "=r" (-> u32),
        : [a] "l" (address),
          [policy] "l" (@backingInt(policy)),
        : .{ .memory = true });
    return registers;
}

inline fn storeB32(comptime caching: StoreCaching, address: usize, value: u32) void {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .write_back => "st.global.wb.b32 [%[a]], %[b];",
            .cache_global_level => "st.global.cg.b32 [%[a]], %[b];",
            .cache_streaming => "st.global.cs.b32 [%[a]], %[b];",
            .write_through => "st.global.wt.b32 [%[a]], %[b];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "st.global.L1::evict_normal.b32 [%[a]], %[b];",
            .evict_unchanged => "st.global.L1::evict_unchanged.b32 [%[a]], %[b];",
            .evict_first => "st.global.L1::evict_first.b32 [%[a]], %[b];",
            .evict_last => "st.global.L1::evict_last.b32 [%[a]], %[b];",
            .no_allocate => "st.global.L1::no_allocate.b32 [%[a]], %[b];",
        },
    };
    asm volatile (text
        :
        : [a] "l" (address),
          [b] "r" (@as(u32, value)),
        : .{ .memory = true });
}

inline fn storeB32WithPolicy(comptime caching: StoreCaching, address: usize, value: u32, policy: CachePolicy) void {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .write_back => "st.global.wb.L2::cache_hint.b32 [%[a]], %[b], %[policy];",
            .cache_global_level => "st.global.cg.L2::cache_hint.b32 [%[a]], %[b], %[policy];",
            .cache_streaming => "st.global.cs.L2::cache_hint.b32 [%[a]], %[b], %[policy];",
            .write_through => "st.global.wt.L2::cache_hint.b32 [%[a]], %[b], %[policy];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "st.global.L1::evict_normal.L2::cache_hint.b32 [%[a]], %[b], %[policy];",
            .evict_unchanged => "st.global.L1::evict_unchanged.L2::cache_hint.b32 [%[a]], %[b], %[policy];",
            .evict_first => "st.global.L1::evict_first.L2::cache_hint.b32 [%[a]], %[b], %[policy];",
            .evict_last => "st.global.L1::evict_last.L2::cache_hint.b32 [%[a]], %[b], %[policy];",
            .no_allocate => "st.global.L1::no_allocate.L2::cache_hint.b32 [%[a]], %[b], %[policy];",
        },
    };
    asm volatile (text
        :
        : [a] "l" (address),
          [b] "r" (@as(u32, value)),
          [policy] "l" (@backingInt(policy)),
        : .{ .memory = true });
}

inline fn loadB64(comptime caching: LoadCaching, address: usize) u64 {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .cache_all_levels => "ld.global.ca.b64 %[d], [%[a]];",
            .cache_global_level => "ld.global.cg.b64 %[d], [%[a]];",
            .cache_streaming => "ld.global.cs.b64 %[d], [%[a]];",
            .last_use => "ld.global.lu.b64 %[d], [%[a]];",
            .cache_volatile => "ld.global.cv.b64 %[d], [%[a]];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "ld.global.L1::evict_normal.b64 %[d], [%[a]];",
            .evict_unchanged => "ld.global.L1::evict_unchanged.b64 %[d], [%[a]];",
            .evict_first => "ld.global.L1::evict_first.b64 %[d], [%[a]];",
            .evict_last => "ld.global.L1::evict_last.b64 %[d], [%[a]];",
            .no_allocate => "ld.global.L1::no_allocate.b64 %[d], [%[a]];",
        },
    };
    const registers = asm volatile (text
        : [d] "=l" (-> u64),
        : [a] "l" (address),
        : .{ .memory = true });
    return registers;
}

inline fn loadB64WithPolicy(comptime caching: LoadCaching, address: usize, policy: CachePolicy) u64 {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .cache_all_levels => "ld.global.ca.L2::cache_hint.b64 %[d], [%[a]], %[policy];",
            .cache_global_level => "ld.global.cg.L2::cache_hint.b64 %[d], [%[a]], %[policy];",
            .cache_streaming => "ld.global.cs.L2::cache_hint.b64 %[d], [%[a]], %[policy];",
            .last_use => "ld.global.lu.L2::cache_hint.b64 %[d], [%[a]], %[policy];",
            .cache_volatile => "ld.global.cv.L2::cache_hint.b64 %[d], [%[a]], %[policy];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "ld.global.L1::evict_normal.L2::cache_hint.b64 %[d], [%[a]], %[policy];",
            .evict_unchanged => "ld.global.L1::evict_unchanged.L2::cache_hint.b64 %[d], [%[a]], %[policy];",
            .evict_first => "ld.global.L1::evict_first.L2::cache_hint.b64 %[d], [%[a]], %[policy];",
            .evict_last => "ld.global.L1::evict_last.L2::cache_hint.b64 %[d], [%[a]], %[policy];",
            .no_allocate => "ld.global.L1::no_allocate.L2::cache_hint.b64 %[d], [%[a]], %[policy];",
        },
    };
    const registers = asm volatile (text
        : [d] "=l" (-> u64),
        : [a] "l" (address),
          [policy] "l" (@backingInt(policy)),
        : .{ .memory = true });
    return registers;
}

inline fn loadNonCoherentB64(comptime operator: NonCoherentOperator, address: usize) u64 {
    const text = switch (operator) {
        .cache_all_levels => "ld.global.ca.nc.b64 %[d], [%[a]];",
        .cache_global_level => "ld.global.cg.nc.b64 %[d], [%[a]];",
        .cache_streaming => "ld.global.cs.nc.b64 %[d], [%[a]];",
    };
    const registers = asm volatile (text
        : [d] "=l" (-> u64),
        : [a] "l" (address),
        : .{ .memory = true });
    return registers;
}

inline fn loadNonCoherentB64WithPolicy(comptime operator: NonCoherentOperator, address: usize, policy: CachePolicy) u64 {
    const text = switch (operator) {
        .cache_all_levels => "ld.global.ca.nc.L2::cache_hint.b64 %[d], [%[a]], %[policy];",
        .cache_global_level => "ld.global.cg.nc.L2::cache_hint.b64 %[d], [%[a]], %[policy];",
        .cache_streaming => "ld.global.cs.nc.L2::cache_hint.b64 %[d], [%[a]], %[policy];",
    };
    const registers = asm volatile (text
        : [d] "=l" (-> u64),
        : [a] "l" (address),
          [policy] "l" (@backingInt(policy)),
        : .{ .memory = true });
    return registers;
}

inline fn storeB64(comptime caching: StoreCaching, address: usize, value: u64) void {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .write_back => "st.global.wb.b64 [%[a]], %[b];",
            .cache_global_level => "st.global.cg.b64 [%[a]], %[b];",
            .cache_streaming => "st.global.cs.b64 [%[a]], %[b];",
            .write_through => "st.global.wt.b64 [%[a]], %[b];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "st.global.L1::evict_normal.b64 [%[a]], %[b];",
            .evict_unchanged => "st.global.L1::evict_unchanged.b64 [%[a]], %[b];",
            .evict_first => "st.global.L1::evict_first.b64 [%[a]], %[b];",
            .evict_last => "st.global.L1::evict_last.b64 [%[a]], %[b];",
            .no_allocate => "st.global.L1::no_allocate.b64 [%[a]], %[b];",
        },
    };
    asm volatile (text
        :
        : [a] "l" (address),
          [b] "l" (value),
        : .{ .memory = true });
}

inline fn storeB64WithPolicy(comptime caching: StoreCaching, address: usize, value: u64, policy: CachePolicy) void {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .write_back => "st.global.wb.L2::cache_hint.b64 [%[a]], %[b], %[policy];",
            .cache_global_level => "st.global.cg.L2::cache_hint.b64 [%[a]], %[b], %[policy];",
            .cache_streaming => "st.global.cs.L2::cache_hint.b64 [%[a]], %[b], %[policy];",
            .write_through => "st.global.wt.L2::cache_hint.b64 [%[a]], %[b], %[policy];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "st.global.L1::evict_normal.L2::cache_hint.b64 [%[a]], %[b], %[policy];",
            .evict_unchanged => "st.global.L1::evict_unchanged.L2::cache_hint.b64 [%[a]], %[b], %[policy];",
            .evict_first => "st.global.L1::evict_first.L2::cache_hint.b64 [%[a]], %[b], %[policy];",
            .evict_last => "st.global.L1::evict_last.L2::cache_hint.b64 [%[a]], %[b], %[policy];",
            .no_allocate => "st.global.L1::no_allocate.L2::cache_hint.b64 [%[a]], %[b], %[policy];",
        },
    };
    asm volatile (text
        :
        : [a] "l" (address),
          [b] "l" (value),
          [policy] "l" (@backingInt(policy)),
        : .{ .memory = true });
}

inline fn loadV2B8(comptime caching: LoadCaching, address: usize) @Vector(2, u8) {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .cache_all_levels => "ld.global.ca.v2.b8 %[d], [%[a]];",
            .cache_global_level => "ld.global.cg.v2.b8 %[d], [%[a]];",
            .cache_streaming => "ld.global.cs.v2.b8 %[d], [%[a]];",
            .last_use => "ld.global.lu.v2.b8 %[d], [%[a]];",
            .cache_volatile => "ld.global.cv.v2.b8 %[d], [%[a]];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "ld.global.L1::evict_normal.v2.b8 %[d], [%[a]];",
            .evict_unchanged => "ld.global.L1::evict_unchanged.v2.b8 %[d], [%[a]];",
            .evict_first => "ld.global.L1::evict_first.v2.b8 %[d], [%[a]];",
            .evict_last => "ld.global.L1::evict_last.v2.b8 %[d], [%[a]];",
            .no_allocate => "ld.global.L1::no_allocate.v2.b8 %[d], [%[a]];",
        },
    };
    const registers = asm volatile (text
        : [d] "=r" (-> [2]u32),
        : [a] "l" (address),
        : .{ .memory = true });
    return @truncate(@as(@Vector(2, u32), registers));
}

inline fn loadV2B8WithPolicy(comptime caching: LoadCaching, address: usize, policy: CachePolicy) @Vector(2, u8) {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .cache_all_levels => "ld.global.ca.L2::cache_hint.v2.b8 %[d], [%[a]], %[policy];",
            .cache_global_level => "ld.global.cg.L2::cache_hint.v2.b8 %[d], [%[a]], %[policy];",
            .cache_streaming => "ld.global.cs.L2::cache_hint.v2.b8 %[d], [%[a]], %[policy];",
            .last_use => "ld.global.lu.L2::cache_hint.v2.b8 %[d], [%[a]], %[policy];",
            .cache_volatile => "ld.global.cv.L2::cache_hint.v2.b8 %[d], [%[a]], %[policy];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "ld.global.L1::evict_normal.L2::cache_hint.v2.b8 %[d], [%[a]], %[policy];",
            .evict_unchanged => "ld.global.L1::evict_unchanged.L2::cache_hint.v2.b8 %[d], [%[a]], %[policy];",
            .evict_first => "ld.global.L1::evict_first.L2::cache_hint.v2.b8 %[d], [%[a]], %[policy];",
            .evict_last => "ld.global.L1::evict_last.L2::cache_hint.v2.b8 %[d], [%[a]], %[policy];",
            .no_allocate => "ld.global.L1::no_allocate.L2::cache_hint.v2.b8 %[d], [%[a]], %[policy];",
        },
    };
    const registers = asm volatile (text
        : [d] "=r" (-> [2]u32),
        : [a] "l" (address),
          [policy] "l" (@backingInt(policy)),
        : .{ .memory = true });
    return @truncate(@as(@Vector(2, u32), registers));
}

inline fn loadNonCoherentV2B8(comptime operator: NonCoherentOperator, address: usize) @Vector(2, u8) {
    const text = switch (operator) {
        .cache_all_levels => "ld.global.ca.nc.v2.b8 %[d], [%[a]];",
        .cache_global_level => "ld.global.cg.nc.v2.b8 %[d], [%[a]];",
        .cache_streaming => "ld.global.cs.nc.v2.b8 %[d], [%[a]];",
    };
    const registers = asm volatile (text
        : [d] "=r" (-> [2]u32),
        : [a] "l" (address),
        : .{ .memory = true });
    return @truncate(@as(@Vector(2, u32), registers));
}

inline fn loadNonCoherentV2B8WithPolicy(comptime operator: NonCoherentOperator, address: usize, policy: CachePolicy) @Vector(2, u8) {
    const text = switch (operator) {
        .cache_all_levels => "ld.global.ca.nc.L2::cache_hint.v2.b8 %[d], [%[a]], %[policy];",
        .cache_global_level => "ld.global.cg.nc.L2::cache_hint.v2.b8 %[d], [%[a]], %[policy];",
        .cache_streaming => "ld.global.cs.nc.L2::cache_hint.v2.b8 %[d], [%[a]], %[policy];",
    };
    const registers = asm volatile (text
        : [d] "=r" (-> [2]u32),
        : [a] "l" (address),
          [policy] "l" (@backingInt(policy)),
        : .{ .memory = true });
    return @truncate(@as(@Vector(2, u32), registers));
}

inline fn storeV2B8(comptime caching: StoreCaching, address: usize, value: @Vector(2, u8)) void {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .write_back => "st.global.wb.v2.b8 [%[a]], %[b];",
            .cache_global_level => "st.global.cg.v2.b8 [%[a]], %[b];",
            .cache_streaming => "st.global.cs.v2.b8 [%[a]], %[b];",
            .write_through => "st.global.wt.v2.b8 [%[a]], %[b];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "st.global.L1::evict_normal.v2.b8 [%[a]], %[b];",
            .evict_unchanged => "st.global.L1::evict_unchanged.v2.b8 [%[a]], %[b];",
            .evict_first => "st.global.L1::evict_first.v2.b8 [%[a]], %[b];",
            .evict_last => "st.global.L1::evict_last.v2.b8 [%[a]], %[b];",
            .no_allocate => "st.global.L1::no_allocate.v2.b8 [%[a]], %[b];",
        },
    };
    asm volatile (text
        :
        : [a] "l" (address),
          [b] "r" (@as([2]u32, @as(@Vector(2, u32), value))),
        : .{ .memory = true });
}

inline fn storeV2B8WithPolicy(comptime caching: StoreCaching, address: usize, value: @Vector(2, u8), policy: CachePolicy) void {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .write_back => "st.global.wb.L2::cache_hint.v2.b8 [%[a]], %[b], %[policy];",
            .cache_global_level => "st.global.cg.L2::cache_hint.v2.b8 [%[a]], %[b], %[policy];",
            .cache_streaming => "st.global.cs.L2::cache_hint.v2.b8 [%[a]], %[b], %[policy];",
            .write_through => "st.global.wt.L2::cache_hint.v2.b8 [%[a]], %[b], %[policy];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "st.global.L1::evict_normal.L2::cache_hint.v2.b8 [%[a]], %[b], %[policy];",
            .evict_unchanged => "st.global.L1::evict_unchanged.L2::cache_hint.v2.b8 [%[a]], %[b], %[policy];",
            .evict_first => "st.global.L1::evict_first.L2::cache_hint.v2.b8 [%[a]], %[b], %[policy];",
            .evict_last => "st.global.L1::evict_last.L2::cache_hint.v2.b8 [%[a]], %[b], %[policy];",
            .no_allocate => "st.global.L1::no_allocate.L2::cache_hint.v2.b8 [%[a]], %[b], %[policy];",
        },
    };
    asm volatile (text
        :
        : [a] "l" (address),
          [b] "r" (@as([2]u32, @as(@Vector(2, u32), value))),
          [policy] "l" (@backingInt(policy)),
        : .{ .memory = true });
}

inline fn loadV2B16(comptime caching: LoadCaching, address: usize) @Vector(2, u16) {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .cache_all_levels => "ld.global.ca.v2.b16 %[d], [%[a]];",
            .cache_global_level => "ld.global.cg.v2.b16 %[d], [%[a]];",
            .cache_streaming => "ld.global.cs.v2.b16 %[d], [%[a]];",
            .last_use => "ld.global.lu.v2.b16 %[d], [%[a]];",
            .cache_volatile => "ld.global.cv.v2.b16 %[d], [%[a]];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "ld.global.L1::evict_normal.v2.b16 %[d], [%[a]];",
            .evict_unchanged => "ld.global.L1::evict_unchanged.v2.b16 %[d], [%[a]];",
            .evict_first => "ld.global.L1::evict_first.v2.b16 %[d], [%[a]];",
            .evict_last => "ld.global.L1::evict_last.v2.b16 %[d], [%[a]];",
            .no_allocate => "ld.global.L1::no_allocate.v2.b16 %[d], [%[a]];",
        },
    };
    const registers = asm volatile (text
        : [d] "=r" (-> [2]u32),
        : [a] "l" (address),
        : .{ .memory = true });
    return @truncate(@as(@Vector(2, u32), registers));
}

inline fn loadV2B16WithPolicy(comptime caching: LoadCaching, address: usize, policy: CachePolicy) @Vector(2, u16) {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .cache_all_levels => "ld.global.ca.L2::cache_hint.v2.b16 %[d], [%[a]], %[policy];",
            .cache_global_level => "ld.global.cg.L2::cache_hint.v2.b16 %[d], [%[a]], %[policy];",
            .cache_streaming => "ld.global.cs.L2::cache_hint.v2.b16 %[d], [%[a]], %[policy];",
            .last_use => "ld.global.lu.L2::cache_hint.v2.b16 %[d], [%[a]], %[policy];",
            .cache_volatile => "ld.global.cv.L2::cache_hint.v2.b16 %[d], [%[a]], %[policy];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "ld.global.L1::evict_normal.L2::cache_hint.v2.b16 %[d], [%[a]], %[policy];",
            .evict_unchanged => "ld.global.L1::evict_unchanged.L2::cache_hint.v2.b16 %[d], [%[a]], %[policy];",
            .evict_first => "ld.global.L1::evict_first.L2::cache_hint.v2.b16 %[d], [%[a]], %[policy];",
            .evict_last => "ld.global.L1::evict_last.L2::cache_hint.v2.b16 %[d], [%[a]], %[policy];",
            .no_allocate => "ld.global.L1::no_allocate.L2::cache_hint.v2.b16 %[d], [%[a]], %[policy];",
        },
    };
    const registers = asm volatile (text
        : [d] "=r" (-> [2]u32),
        : [a] "l" (address),
          [policy] "l" (@backingInt(policy)),
        : .{ .memory = true });
    return @truncate(@as(@Vector(2, u32), registers));
}

inline fn loadNonCoherentV2B16(comptime operator: NonCoherentOperator, address: usize) @Vector(2, u16) {
    const text = switch (operator) {
        .cache_all_levels => "ld.global.ca.nc.v2.b16 %[d], [%[a]];",
        .cache_global_level => "ld.global.cg.nc.v2.b16 %[d], [%[a]];",
        .cache_streaming => "ld.global.cs.nc.v2.b16 %[d], [%[a]];",
    };
    const registers = asm volatile (text
        : [d] "=r" (-> [2]u32),
        : [a] "l" (address),
        : .{ .memory = true });
    return @truncate(@as(@Vector(2, u32), registers));
}

inline fn loadNonCoherentV2B16WithPolicy(comptime operator: NonCoherentOperator, address: usize, policy: CachePolicy) @Vector(2, u16) {
    const text = switch (operator) {
        .cache_all_levels => "ld.global.ca.nc.L2::cache_hint.v2.b16 %[d], [%[a]], %[policy];",
        .cache_global_level => "ld.global.cg.nc.L2::cache_hint.v2.b16 %[d], [%[a]], %[policy];",
        .cache_streaming => "ld.global.cs.nc.L2::cache_hint.v2.b16 %[d], [%[a]], %[policy];",
    };
    const registers = asm volatile (text
        : [d] "=r" (-> [2]u32),
        : [a] "l" (address),
          [policy] "l" (@backingInt(policy)),
        : .{ .memory = true });
    return @truncate(@as(@Vector(2, u32), registers));
}

inline fn storeV2B16(comptime caching: StoreCaching, address: usize, value: @Vector(2, u16)) void {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .write_back => "st.global.wb.v2.b16 [%[a]], %[b];",
            .cache_global_level => "st.global.cg.v2.b16 [%[a]], %[b];",
            .cache_streaming => "st.global.cs.v2.b16 [%[a]], %[b];",
            .write_through => "st.global.wt.v2.b16 [%[a]], %[b];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "st.global.L1::evict_normal.v2.b16 [%[a]], %[b];",
            .evict_unchanged => "st.global.L1::evict_unchanged.v2.b16 [%[a]], %[b];",
            .evict_first => "st.global.L1::evict_first.v2.b16 [%[a]], %[b];",
            .evict_last => "st.global.L1::evict_last.v2.b16 [%[a]], %[b];",
            .no_allocate => "st.global.L1::no_allocate.v2.b16 [%[a]], %[b];",
        },
    };
    asm volatile (text
        :
        : [a] "l" (address),
          [b] "r" (@as([2]u32, @as(@Vector(2, u32), value))),
        : .{ .memory = true });
}

inline fn storeV2B16WithPolicy(comptime caching: StoreCaching, address: usize, value: @Vector(2, u16), policy: CachePolicy) void {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .write_back => "st.global.wb.L2::cache_hint.v2.b16 [%[a]], %[b], %[policy];",
            .cache_global_level => "st.global.cg.L2::cache_hint.v2.b16 [%[a]], %[b], %[policy];",
            .cache_streaming => "st.global.cs.L2::cache_hint.v2.b16 [%[a]], %[b], %[policy];",
            .write_through => "st.global.wt.L2::cache_hint.v2.b16 [%[a]], %[b], %[policy];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "st.global.L1::evict_normal.L2::cache_hint.v2.b16 [%[a]], %[b], %[policy];",
            .evict_unchanged => "st.global.L1::evict_unchanged.L2::cache_hint.v2.b16 [%[a]], %[b], %[policy];",
            .evict_first => "st.global.L1::evict_first.L2::cache_hint.v2.b16 [%[a]], %[b], %[policy];",
            .evict_last => "st.global.L1::evict_last.L2::cache_hint.v2.b16 [%[a]], %[b], %[policy];",
            .no_allocate => "st.global.L1::no_allocate.L2::cache_hint.v2.b16 [%[a]], %[b], %[policy];",
        },
    };
    asm volatile (text
        :
        : [a] "l" (address),
          [b] "r" (@as([2]u32, @as(@Vector(2, u32), value))),
          [policy] "l" (@backingInt(policy)),
        : .{ .memory = true });
}

inline fn loadV2B32(comptime caching: LoadCaching, address: usize) @Vector(2, u32) {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .cache_all_levels => "ld.global.ca.v2.b32 %[d], [%[a]];",
            .cache_global_level => "ld.global.cg.v2.b32 %[d], [%[a]];",
            .cache_streaming => "ld.global.cs.v2.b32 %[d], [%[a]];",
            .last_use => "ld.global.lu.v2.b32 %[d], [%[a]];",
            .cache_volatile => "ld.global.cv.v2.b32 %[d], [%[a]];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "ld.global.L1::evict_normal.v2.b32 %[d], [%[a]];",
            .evict_unchanged => "ld.global.L1::evict_unchanged.v2.b32 %[d], [%[a]];",
            .evict_first => "ld.global.L1::evict_first.v2.b32 %[d], [%[a]];",
            .evict_last => "ld.global.L1::evict_last.v2.b32 %[d], [%[a]];",
            .no_allocate => "ld.global.L1::no_allocate.v2.b32 %[d], [%[a]];",
        },
    };
    const registers = asm volatile (text
        : [d] "=r" (-> [2]u32),
        : [a] "l" (address),
        : .{ .memory = true });
    return registers;
}

inline fn loadV2B32WithPolicy(comptime caching: LoadCaching, address: usize, policy: CachePolicy) @Vector(2, u32) {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .cache_all_levels => "ld.global.ca.L2::cache_hint.v2.b32 %[d], [%[a]], %[policy];",
            .cache_global_level => "ld.global.cg.L2::cache_hint.v2.b32 %[d], [%[a]], %[policy];",
            .cache_streaming => "ld.global.cs.L2::cache_hint.v2.b32 %[d], [%[a]], %[policy];",
            .last_use => "ld.global.lu.L2::cache_hint.v2.b32 %[d], [%[a]], %[policy];",
            .cache_volatile => "ld.global.cv.L2::cache_hint.v2.b32 %[d], [%[a]], %[policy];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "ld.global.L1::evict_normal.L2::cache_hint.v2.b32 %[d], [%[a]], %[policy];",
            .evict_unchanged => "ld.global.L1::evict_unchanged.L2::cache_hint.v2.b32 %[d], [%[a]], %[policy];",
            .evict_first => "ld.global.L1::evict_first.L2::cache_hint.v2.b32 %[d], [%[a]], %[policy];",
            .evict_last => "ld.global.L1::evict_last.L2::cache_hint.v2.b32 %[d], [%[a]], %[policy];",
            .no_allocate => "ld.global.L1::no_allocate.L2::cache_hint.v2.b32 %[d], [%[a]], %[policy];",
        },
    };
    const registers = asm volatile (text
        : [d] "=r" (-> [2]u32),
        : [a] "l" (address),
          [policy] "l" (@backingInt(policy)),
        : .{ .memory = true });
    return registers;
}

inline fn loadNonCoherentV2B32(comptime operator: NonCoherentOperator, address: usize) @Vector(2, u32) {
    const text = switch (operator) {
        .cache_all_levels => "ld.global.ca.nc.v2.b32 %[d], [%[a]];",
        .cache_global_level => "ld.global.cg.nc.v2.b32 %[d], [%[a]];",
        .cache_streaming => "ld.global.cs.nc.v2.b32 %[d], [%[a]];",
    };
    const registers = asm volatile (text
        : [d] "=r" (-> [2]u32),
        : [a] "l" (address),
        : .{ .memory = true });
    return registers;
}

inline fn loadNonCoherentV2B32WithPolicy(comptime operator: NonCoherentOperator, address: usize, policy: CachePolicy) @Vector(2, u32) {
    const text = switch (operator) {
        .cache_all_levels => "ld.global.ca.nc.L2::cache_hint.v2.b32 %[d], [%[a]], %[policy];",
        .cache_global_level => "ld.global.cg.nc.L2::cache_hint.v2.b32 %[d], [%[a]], %[policy];",
        .cache_streaming => "ld.global.cs.nc.L2::cache_hint.v2.b32 %[d], [%[a]], %[policy];",
    };
    const registers = asm volatile (text
        : [d] "=r" (-> [2]u32),
        : [a] "l" (address),
          [policy] "l" (@backingInt(policy)),
        : .{ .memory = true });
    return registers;
}

inline fn storeV2B32(comptime caching: StoreCaching, address: usize, value: @Vector(2, u32)) void {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .write_back => "st.global.wb.v2.b32 [%[a]], %[b];",
            .cache_global_level => "st.global.cg.v2.b32 [%[a]], %[b];",
            .cache_streaming => "st.global.cs.v2.b32 [%[a]], %[b];",
            .write_through => "st.global.wt.v2.b32 [%[a]], %[b];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "st.global.L1::evict_normal.v2.b32 [%[a]], %[b];",
            .evict_unchanged => "st.global.L1::evict_unchanged.v2.b32 [%[a]], %[b];",
            .evict_first => "st.global.L1::evict_first.v2.b32 [%[a]], %[b];",
            .evict_last => "st.global.L1::evict_last.v2.b32 [%[a]], %[b];",
            .no_allocate => "st.global.L1::no_allocate.v2.b32 [%[a]], %[b];",
        },
    };
    asm volatile (text
        :
        : [a] "l" (address),
          [b] "r" (@as([2]u32, value)),
        : .{ .memory = true });
}

inline fn storeV2B32WithPolicy(comptime caching: StoreCaching, address: usize, value: @Vector(2, u32), policy: CachePolicy) void {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .write_back => "st.global.wb.L2::cache_hint.v2.b32 [%[a]], %[b], %[policy];",
            .cache_global_level => "st.global.cg.L2::cache_hint.v2.b32 [%[a]], %[b], %[policy];",
            .cache_streaming => "st.global.cs.L2::cache_hint.v2.b32 [%[a]], %[b], %[policy];",
            .write_through => "st.global.wt.L2::cache_hint.v2.b32 [%[a]], %[b], %[policy];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "st.global.L1::evict_normal.L2::cache_hint.v2.b32 [%[a]], %[b], %[policy];",
            .evict_unchanged => "st.global.L1::evict_unchanged.L2::cache_hint.v2.b32 [%[a]], %[b], %[policy];",
            .evict_first => "st.global.L1::evict_first.L2::cache_hint.v2.b32 [%[a]], %[b], %[policy];",
            .evict_last => "st.global.L1::evict_last.L2::cache_hint.v2.b32 [%[a]], %[b], %[policy];",
            .no_allocate => "st.global.L1::no_allocate.L2::cache_hint.v2.b32 [%[a]], %[b], %[policy];",
        },
    };
    asm volatile (text
        :
        : [a] "l" (address),
          [b] "r" (@as([2]u32, value)),
          [policy] "l" (@backingInt(policy)),
        : .{ .memory = true });
}

inline fn loadV2B64(comptime caching: LoadCaching, address: usize) @Vector(2, u64) {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .cache_all_levels => "ld.global.ca.v2.b64 %[d], [%[a]];",
            .cache_global_level => "ld.global.cg.v2.b64 %[d], [%[a]];",
            .cache_streaming => "ld.global.cs.v2.b64 %[d], [%[a]];",
            .last_use => "ld.global.lu.v2.b64 %[d], [%[a]];",
            .cache_volatile => "ld.global.cv.v2.b64 %[d], [%[a]];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "ld.global.L1::evict_normal.v2.b64 %[d], [%[a]];",
            .evict_unchanged => "ld.global.L1::evict_unchanged.v2.b64 %[d], [%[a]];",
            .evict_first => "ld.global.L1::evict_first.v2.b64 %[d], [%[a]];",
            .evict_last => "ld.global.L1::evict_last.v2.b64 %[d], [%[a]];",
            .no_allocate => "ld.global.L1::no_allocate.v2.b64 %[d], [%[a]];",
        },
    };
    const registers = asm volatile (text
        : [d] "=l" (-> [2]u64),
        : [a] "l" (address),
        : .{ .memory = true });
    return registers;
}

inline fn loadV2B64WithPolicy(comptime caching: LoadCaching, address: usize, policy: CachePolicy) @Vector(2, u64) {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .cache_all_levels => "ld.global.ca.L2::cache_hint.v2.b64 %[d], [%[a]], %[policy];",
            .cache_global_level => "ld.global.cg.L2::cache_hint.v2.b64 %[d], [%[a]], %[policy];",
            .cache_streaming => "ld.global.cs.L2::cache_hint.v2.b64 %[d], [%[a]], %[policy];",
            .last_use => "ld.global.lu.L2::cache_hint.v2.b64 %[d], [%[a]], %[policy];",
            .cache_volatile => "ld.global.cv.L2::cache_hint.v2.b64 %[d], [%[a]], %[policy];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "ld.global.L1::evict_normal.L2::cache_hint.v2.b64 %[d], [%[a]], %[policy];",
            .evict_unchanged => "ld.global.L1::evict_unchanged.L2::cache_hint.v2.b64 %[d], [%[a]], %[policy];",
            .evict_first => "ld.global.L1::evict_first.L2::cache_hint.v2.b64 %[d], [%[a]], %[policy];",
            .evict_last => "ld.global.L1::evict_last.L2::cache_hint.v2.b64 %[d], [%[a]], %[policy];",
            .no_allocate => "ld.global.L1::no_allocate.L2::cache_hint.v2.b64 %[d], [%[a]], %[policy];",
        },
    };
    const registers = asm volatile (text
        : [d] "=l" (-> [2]u64),
        : [a] "l" (address),
          [policy] "l" (@backingInt(policy)),
        : .{ .memory = true });
    return registers;
}

inline fn loadNonCoherentV2B64(comptime operator: NonCoherentOperator, address: usize) @Vector(2, u64) {
    const text = switch (operator) {
        .cache_all_levels => "ld.global.ca.nc.v2.b64 %[d], [%[a]];",
        .cache_global_level => "ld.global.cg.nc.v2.b64 %[d], [%[a]];",
        .cache_streaming => "ld.global.cs.nc.v2.b64 %[d], [%[a]];",
    };
    const registers = asm volatile (text
        : [d] "=l" (-> [2]u64),
        : [a] "l" (address),
        : .{ .memory = true });
    return registers;
}

inline fn loadNonCoherentV2B64WithPolicy(comptime operator: NonCoherentOperator, address: usize, policy: CachePolicy) @Vector(2, u64) {
    const text = switch (operator) {
        .cache_all_levels => "ld.global.ca.nc.L2::cache_hint.v2.b64 %[d], [%[a]], %[policy];",
        .cache_global_level => "ld.global.cg.nc.L2::cache_hint.v2.b64 %[d], [%[a]], %[policy];",
        .cache_streaming => "ld.global.cs.nc.L2::cache_hint.v2.b64 %[d], [%[a]], %[policy];",
    };
    const registers = asm volatile (text
        : [d] "=l" (-> [2]u64),
        : [a] "l" (address),
          [policy] "l" (@backingInt(policy)),
        : .{ .memory = true });
    return registers;
}

inline fn storeV2B64(comptime caching: StoreCaching, address: usize, value: @Vector(2, u64)) void {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .write_back => "st.global.wb.v2.b64 [%[a]], %[b];",
            .cache_global_level => "st.global.cg.v2.b64 [%[a]], %[b];",
            .cache_streaming => "st.global.cs.v2.b64 [%[a]], %[b];",
            .write_through => "st.global.wt.v2.b64 [%[a]], %[b];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "st.global.L1::evict_normal.v2.b64 [%[a]], %[b];",
            .evict_unchanged => "st.global.L1::evict_unchanged.v2.b64 [%[a]], %[b];",
            .evict_first => "st.global.L1::evict_first.v2.b64 [%[a]], %[b];",
            .evict_last => "st.global.L1::evict_last.v2.b64 [%[a]], %[b];",
            .no_allocate => "st.global.L1::no_allocate.v2.b64 [%[a]], %[b];",
        },
    };
    asm volatile (text
        :
        : [a] "l" (address),
          [b] "l" (@as([2]u64, value)),
        : .{ .memory = true });
}

inline fn storeV2B64WithPolicy(comptime caching: StoreCaching, address: usize, value: @Vector(2, u64), policy: CachePolicy) void {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .write_back => "st.global.wb.L2::cache_hint.v2.b64 [%[a]], %[b], %[policy];",
            .cache_global_level => "st.global.cg.L2::cache_hint.v2.b64 [%[a]], %[b], %[policy];",
            .cache_streaming => "st.global.cs.L2::cache_hint.v2.b64 [%[a]], %[b], %[policy];",
            .write_through => "st.global.wt.L2::cache_hint.v2.b64 [%[a]], %[b], %[policy];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "st.global.L1::evict_normal.L2::cache_hint.v2.b64 [%[a]], %[b], %[policy];",
            .evict_unchanged => "st.global.L1::evict_unchanged.L2::cache_hint.v2.b64 [%[a]], %[b], %[policy];",
            .evict_first => "st.global.L1::evict_first.L2::cache_hint.v2.b64 [%[a]], %[b], %[policy];",
            .evict_last => "st.global.L1::evict_last.L2::cache_hint.v2.b64 [%[a]], %[b], %[policy];",
            .no_allocate => "st.global.L1::no_allocate.L2::cache_hint.v2.b64 [%[a]], %[b], %[policy];",
        },
    };
    asm volatile (text
        :
        : [a] "l" (address),
          [b] "l" (@as([2]u64, value)),
          [policy] "l" (@backingInt(policy)),
        : .{ .memory = true });
}

inline fn loadV4B8(comptime caching: LoadCaching, address: usize) @Vector(4, u8) {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .cache_all_levels => "ld.global.ca.v4.b8 %[d], [%[a]];",
            .cache_global_level => "ld.global.cg.v4.b8 %[d], [%[a]];",
            .cache_streaming => "ld.global.cs.v4.b8 %[d], [%[a]];",
            .last_use => "ld.global.lu.v4.b8 %[d], [%[a]];",
            .cache_volatile => "ld.global.cv.v4.b8 %[d], [%[a]];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "ld.global.L1::evict_normal.v4.b8 %[d], [%[a]];",
            .evict_unchanged => "ld.global.L1::evict_unchanged.v4.b8 %[d], [%[a]];",
            .evict_first => "ld.global.L1::evict_first.v4.b8 %[d], [%[a]];",
            .evict_last => "ld.global.L1::evict_last.v4.b8 %[d], [%[a]];",
            .no_allocate => "ld.global.L1::no_allocate.v4.b8 %[d], [%[a]];",
        },
    };
    const registers = asm volatile (text
        : [d] "=r" (-> [4]u32),
        : [a] "l" (address),
        : .{ .memory = true });
    return @truncate(@as(@Vector(4, u32), registers));
}

inline fn loadV4B8WithPolicy(comptime caching: LoadCaching, address: usize, policy: CachePolicy) @Vector(4, u8) {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .cache_all_levels => "ld.global.ca.L2::cache_hint.v4.b8 %[d], [%[a]], %[policy];",
            .cache_global_level => "ld.global.cg.L2::cache_hint.v4.b8 %[d], [%[a]], %[policy];",
            .cache_streaming => "ld.global.cs.L2::cache_hint.v4.b8 %[d], [%[a]], %[policy];",
            .last_use => "ld.global.lu.L2::cache_hint.v4.b8 %[d], [%[a]], %[policy];",
            .cache_volatile => "ld.global.cv.L2::cache_hint.v4.b8 %[d], [%[a]], %[policy];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "ld.global.L1::evict_normal.L2::cache_hint.v4.b8 %[d], [%[a]], %[policy];",
            .evict_unchanged => "ld.global.L1::evict_unchanged.L2::cache_hint.v4.b8 %[d], [%[a]], %[policy];",
            .evict_first => "ld.global.L1::evict_first.L2::cache_hint.v4.b8 %[d], [%[a]], %[policy];",
            .evict_last => "ld.global.L1::evict_last.L2::cache_hint.v4.b8 %[d], [%[a]], %[policy];",
            .no_allocate => "ld.global.L1::no_allocate.L2::cache_hint.v4.b8 %[d], [%[a]], %[policy];",
        },
    };
    const registers = asm volatile (text
        : [d] "=r" (-> [4]u32),
        : [a] "l" (address),
          [policy] "l" (@backingInt(policy)),
        : .{ .memory = true });
    return @truncate(@as(@Vector(4, u32), registers));
}

inline fn loadNonCoherentV4B8(comptime operator: NonCoherentOperator, address: usize) @Vector(4, u8) {
    const text = switch (operator) {
        .cache_all_levels => "ld.global.ca.nc.v4.b8 %[d], [%[a]];",
        .cache_global_level => "ld.global.cg.nc.v4.b8 %[d], [%[a]];",
        .cache_streaming => "ld.global.cs.nc.v4.b8 %[d], [%[a]];",
    };
    const registers = asm volatile (text
        : [d] "=r" (-> [4]u32),
        : [a] "l" (address),
        : .{ .memory = true });
    return @truncate(@as(@Vector(4, u32), registers));
}

inline fn loadNonCoherentV4B8WithPolicy(comptime operator: NonCoherentOperator, address: usize, policy: CachePolicy) @Vector(4, u8) {
    const text = switch (operator) {
        .cache_all_levels => "ld.global.ca.nc.L2::cache_hint.v4.b8 %[d], [%[a]], %[policy];",
        .cache_global_level => "ld.global.cg.nc.L2::cache_hint.v4.b8 %[d], [%[a]], %[policy];",
        .cache_streaming => "ld.global.cs.nc.L2::cache_hint.v4.b8 %[d], [%[a]], %[policy];",
    };
    const registers = asm volatile (text
        : [d] "=r" (-> [4]u32),
        : [a] "l" (address),
          [policy] "l" (@backingInt(policy)),
        : .{ .memory = true });
    return @truncate(@as(@Vector(4, u32), registers));
}

inline fn storeV4B8(comptime caching: StoreCaching, address: usize, value: @Vector(4, u8)) void {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .write_back => "st.global.wb.v4.b8 [%[a]], %[b];",
            .cache_global_level => "st.global.cg.v4.b8 [%[a]], %[b];",
            .cache_streaming => "st.global.cs.v4.b8 [%[a]], %[b];",
            .write_through => "st.global.wt.v4.b8 [%[a]], %[b];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "st.global.L1::evict_normal.v4.b8 [%[a]], %[b];",
            .evict_unchanged => "st.global.L1::evict_unchanged.v4.b8 [%[a]], %[b];",
            .evict_first => "st.global.L1::evict_first.v4.b8 [%[a]], %[b];",
            .evict_last => "st.global.L1::evict_last.v4.b8 [%[a]], %[b];",
            .no_allocate => "st.global.L1::no_allocate.v4.b8 [%[a]], %[b];",
        },
    };
    asm volatile (text
        :
        : [a] "l" (address),
          [b] "r" (@as([4]u32, @as(@Vector(4, u32), value))),
        : .{ .memory = true });
}

inline fn storeV4B8WithPolicy(comptime caching: StoreCaching, address: usize, value: @Vector(4, u8), policy: CachePolicy) void {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .write_back => "st.global.wb.L2::cache_hint.v4.b8 [%[a]], %[b], %[policy];",
            .cache_global_level => "st.global.cg.L2::cache_hint.v4.b8 [%[a]], %[b], %[policy];",
            .cache_streaming => "st.global.cs.L2::cache_hint.v4.b8 [%[a]], %[b], %[policy];",
            .write_through => "st.global.wt.L2::cache_hint.v4.b8 [%[a]], %[b], %[policy];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "st.global.L1::evict_normal.L2::cache_hint.v4.b8 [%[a]], %[b], %[policy];",
            .evict_unchanged => "st.global.L1::evict_unchanged.L2::cache_hint.v4.b8 [%[a]], %[b], %[policy];",
            .evict_first => "st.global.L1::evict_first.L2::cache_hint.v4.b8 [%[a]], %[b], %[policy];",
            .evict_last => "st.global.L1::evict_last.L2::cache_hint.v4.b8 [%[a]], %[b], %[policy];",
            .no_allocate => "st.global.L1::no_allocate.L2::cache_hint.v4.b8 [%[a]], %[b], %[policy];",
        },
    };
    asm volatile (text
        :
        : [a] "l" (address),
          [b] "r" (@as([4]u32, @as(@Vector(4, u32), value))),
          [policy] "l" (@backingInt(policy)),
        : .{ .memory = true });
}

inline fn loadV4B16(comptime caching: LoadCaching, address: usize) @Vector(4, u16) {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .cache_all_levels => "ld.global.ca.v4.b16 %[d], [%[a]];",
            .cache_global_level => "ld.global.cg.v4.b16 %[d], [%[a]];",
            .cache_streaming => "ld.global.cs.v4.b16 %[d], [%[a]];",
            .last_use => "ld.global.lu.v4.b16 %[d], [%[a]];",
            .cache_volatile => "ld.global.cv.v4.b16 %[d], [%[a]];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "ld.global.L1::evict_normal.v4.b16 %[d], [%[a]];",
            .evict_unchanged => "ld.global.L1::evict_unchanged.v4.b16 %[d], [%[a]];",
            .evict_first => "ld.global.L1::evict_first.v4.b16 %[d], [%[a]];",
            .evict_last => "ld.global.L1::evict_last.v4.b16 %[d], [%[a]];",
            .no_allocate => "ld.global.L1::no_allocate.v4.b16 %[d], [%[a]];",
        },
    };
    const registers = asm volatile (text
        : [d] "=r" (-> [4]u32),
        : [a] "l" (address),
        : .{ .memory = true });
    return @truncate(@as(@Vector(4, u32), registers));
}

inline fn loadV4B16WithPolicy(comptime caching: LoadCaching, address: usize, policy: CachePolicy) @Vector(4, u16) {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .cache_all_levels => "ld.global.ca.L2::cache_hint.v4.b16 %[d], [%[a]], %[policy];",
            .cache_global_level => "ld.global.cg.L2::cache_hint.v4.b16 %[d], [%[a]], %[policy];",
            .cache_streaming => "ld.global.cs.L2::cache_hint.v4.b16 %[d], [%[a]], %[policy];",
            .last_use => "ld.global.lu.L2::cache_hint.v4.b16 %[d], [%[a]], %[policy];",
            .cache_volatile => "ld.global.cv.L2::cache_hint.v4.b16 %[d], [%[a]], %[policy];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "ld.global.L1::evict_normal.L2::cache_hint.v4.b16 %[d], [%[a]], %[policy];",
            .evict_unchanged => "ld.global.L1::evict_unchanged.L2::cache_hint.v4.b16 %[d], [%[a]], %[policy];",
            .evict_first => "ld.global.L1::evict_first.L2::cache_hint.v4.b16 %[d], [%[a]], %[policy];",
            .evict_last => "ld.global.L1::evict_last.L2::cache_hint.v4.b16 %[d], [%[a]], %[policy];",
            .no_allocate => "ld.global.L1::no_allocate.L2::cache_hint.v4.b16 %[d], [%[a]], %[policy];",
        },
    };
    const registers = asm volatile (text
        : [d] "=r" (-> [4]u32),
        : [a] "l" (address),
          [policy] "l" (@backingInt(policy)),
        : .{ .memory = true });
    return @truncate(@as(@Vector(4, u32), registers));
}

inline fn loadNonCoherentV4B16(comptime operator: NonCoherentOperator, address: usize) @Vector(4, u16) {
    const text = switch (operator) {
        .cache_all_levels => "ld.global.ca.nc.v4.b16 %[d], [%[a]];",
        .cache_global_level => "ld.global.cg.nc.v4.b16 %[d], [%[a]];",
        .cache_streaming => "ld.global.cs.nc.v4.b16 %[d], [%[a]];",
    };
    const registers = asm volatile (text
        : [d] "=r" (-> [4]u32),
        : [a] "l" (address),
        : .{ .memory = true });
    return @truncate(@as(@Vector(4, u32), registers));
}

inline fn loadNonCoherentV4B16WithPolicy(comptime operator: NonCoherentOperator, address: usize, policy: CachePolicy) @Vector(4, u16) {
    const text = switch (operator) {
        .cache_all_levels => "ld.global.ca.nc.L2::cache_hint.v4.b16 %[d], [%[a]], %[policy];",
        .cache_global_level => "ld.global.cg.nc.L2::cache_hint.v4.b16 %[d], [%[a]], %[policy];",
        .cache_streaming => "ld.global.cs.nc.L2::cache_hint.v4.b16 %[d], [%[a]], %[policy];",
    };
    const registers = asm volatile (text
        : [d] "=r" (-> [4]u32),
        : [a] "l" (address),
          [policy] "l" (@backingInt(policy)),
        : .{ .memory = true });
    return @truncate(@as(@Vector(4, u32), registers));
}

inline fn storeV4B16(comptime caching: StoreCaching, address: usize, value: @Vector(4, u16)) void {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .write_back => "st.global.wb.v4.b16 [%[a]], %[b];",
            .cache_global_level => "st.global.cg.v4.b16 [%[a]], %[b];",
            .cache_streaming => "st.global.cs.v4.b16 [%[a]], %[b];",
            .write_through => "st.global.wt.v4.b16 [%[a]], %[b];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "st.global.L1::evict_normal.v4.b16 [%[a]], %[b];",
            .evict_unchanged => "st.global.L1::evict_unchanged.v4.b16 [%[a]], %[b];",
            .evict_first => "st.global.L1::evict_first.v4.b16 [%[a]], %[b];",
            .evict_last => "st.global.L1::evict_last.v4.b16 [%[a]], %[b];",
            .no_allocate => "st.global.L1::no_allocate.v4.b16 [%[a]], %[b];",
        },
    };
    asm volatile (text
        :
        : [a] "l" (address),
          [b] "r" (@as([4]u32, @as(@Vector(4, u32), value))),
        : .{ .memory = true });
}

inline fn storeV4B16WithPolicy(comptime caching: StoreCaching, address: usize, value: @Vector(4, u16), policy: CachePolicy) void {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .write_back => "st.global.wb.L2::cache_hint.v4.b16 [%[a]], %[b], %[policy];",
            .cache_global_level => "st.global.cg.L2::cache_hint.v4.b16 [%[a]], %[b], %[policy];",
            .cache_streaming => "st.global.cs.L2::cache_hint.v4.b16 [%[a]], %[b], %[policy];",
            .write_through => "st.global.wt.L2::cache_hint.v4.b16 [%[a]], %[b], %[policy];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "st.global.L1::evict_normal.L2::cache_hint.v4.b16 [%[a]], %[b], %[policy];",
            .evict_unchanged => "st.global.L1::evict_unchanged.L2::cache_hint.v4.b16 [%[a]], %[b], %[policy];",
            .evict_first => "st.global.L1::evict_first.L2::cache_hint.v4.b16 [%[a]], %[b], %[policy];",
            .evict_last => "st.global.L1::evict_last.L2::cache_hint.v4.b16 [%[a]], %[b], %[policy];",
            .no_allocate => "st.global.L1::no_allocate.L2::cache_hint.v4.b16 [%[a]], %[b], %[policy];",
        },
    };
    asm volatile (text
        :
        : [a] "l" (address),
          [b] "r" (@as([4]u32, @as(@Vector(4, u32), value))),
          [policy] "l" (@backingInt(policy)),
        : .{ .memory = true });
}

inline fn loadV4B32(comptime caching: LoadCaching, address: usize) @Vector(4, u32) {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .cache_all_levels => "ld.global.ca.v4.b32 %[d], [%[a]];",
            .cache_global_level => "ld.global.cg.v4.b32 %[d], [%[a]];",
            .cache_streaming => "ld.global.cs.v4.b32 %[d], [%[a]];",
            .last_use => "ld.global.lu.v4.b32 %[d], [%[a]];",
            .cache_volatile => "ld.global.cv.v4.b32 %[d], [%[a]];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "ld.global.L1::evict_normal.v4.b32 %[d], [%[a]];",
            .evict_unchanged => "ld.global.L1::evict_unchanged.v4.b32 %[d], [%[a]];",
            .evict_first => "ld.global.L1::evict_first.v4.b32 %[d], [%[a]];",
            .evict_last => "ld.global.L1::evict_last.v4.b32 %[d], [%[a]];",
            .no_allocate => "ld.global.L1::no_allocate.v4.b32 %[d], [%[a]];",
        },
    };
    const registers = asm volatile (text
        : [d] "=r" (-> [4]u32),
        : [a] "l" (address),
        : .{ .memory = true });
    return registers;
}

inline fn loadV4B32WithPolicy(comptime caching: LoadCaching, address: usize, policy: CachePolicy) @Vector(4, u32) {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .cache_all_levels => "ld.global.ca.L2::cache_hint.v4.b32 %[d], [%[a]], %[policy];",
            .cache_global_level => "ld.global.cg.L2::cache_hint.v4.b32 %[d], [%[a]], %[policy];",
            .cache_streaming => "ld.global.cs.L2::cache_hint.v4.b32 %[d], [%[a]], %[policy];",
            .last_use => "ld.global.lu.L2::cache_hint.v4.b32 %[d], [%[a]], %[policy];",
            .cache_volatile => "ld.global.cv.L2::cache_hint.v4.b32 %[d], [%[a]], %[policy];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "ld.global.L1::evict_normal.L2::cache_hint.v4.b32 %[d], [%[a]], %[policy];",
            .evict_unchanged => "ld.global.L1::evict_unchanged.L2::cache_hint.v4.b32 %[d], [%[a]], %[policy];",
            .evict_first => "ld.global.L1::evict_first.L2::cache_hint.v4.b32 %[d], [%[a]], %[policy];",
            .evict_last => "ld.global.L1::evict_last.L2::cache_hint.v4.b32 %[d], [%[a]], %[policy];",
            .no_allocate => "ld.global.L1::no_allocate.L2::cache_hint.v4.b32 %[d], [%[a]], %[policy];",
        },
    };
    const registers = asm volatile (text
        : [d] "=r" (-> [4]u32),
        : [a] "l" (address),
          [policy] "l" (@backingInt(policy)),
        : .{ .memory = true });
    return registers;
}

inline fn loadNonCoherentV4B32(comptime operator: NonCoherentOperator, address: usize) @Vector(4, u32) {
    const text = switch (operator) {
        .cache_all_levels => "ld.global.ca.nc.v4.b32 %[d], [%[a]];",
        .cache_global_level => "ld.global.cg.nc.v4.b32 %[d], [%[a]];",
        .cache_streaming => "ld.global.cs.nc.v4.b32 %[d], [%[a]];",
    };
    const registers = asm volatile (text
        : [d] "=r" (-> [4]u32),
        : [a] "l" (address),
        : .{ .memory = true });
    return registers;
}

inline fn loadNonCoherentV4B32WithPolicy(comptime operator: NonCoherentOperator, address: usize, policy: CachePolicy) @Vector(4, u32) {
    const text = switch (operator) {
        .cache_all_levels => "ld.global.ca.nc.L2::cache_hint.v4.b32 %[d], [%[a]], %[policy];",
        .cache_global_level => "ld.global.cg.nc.L2::cache_hint.v4.b32 %[d], [%[a]], %[policy];",
        .cache_streaming => "ld.global.cs.nc.L2::cache_hint.v4.b32 %[d], [%[a]], %[policy];",
    };
    const registers = asm volatile (text
        : [d] "=r" (-> [4]u32),
        : [a] "l" (address),
          [policy] "l" (@backingInt(policy)),
        : .{ .memory = true });
    return registers;
}

inline fn storeV4B32(comptime caching: StoreCaching, address: usize, value: @Vector(4, u32)) void {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .write_back => "st.global.wb.v4.b32 [%[a]], %[b];",
            .cache_global_level => "st.global.cg.v4.b32 [%[a]], %[b];",
            .cache_streaming => "st.global.cs.v4.b32 [%[a]], %[b];",
            .write_through => "st.global.wt.v4.b32 [%[a]], %[b];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "st.global.L1::evict_normal.v4.b32 [%[a]], %[b];",
            .evict_unchanged => "st.global.L1::evict_unchanged.v4.b32 [%[a]], %[b];",
            .evict_first => "st.global.L1::evict_first.v4.b32 [%[a]], %[b];",
            .evict_last => "st.global.L1::evict_last.v4.b32 [%[a]], %[b];",
            .no_allocate => "st.global.L1::no_allocate.v4.b32 [%[a]], %[b];",
        },
    };
    asm volatile (text
        :
        : [a] "l" (address),
          [b] "r" (@as([4]u32, value)),
        : .{ .memory = true });
}

inline fn storeV4B32WithPolicy(comptime caching: StoreCaching, address: usize, value: @Vector(4, u32), policy: CachePolicy) void {
    const text = switch (caching) {
        .operator => |operator| switch (operator) {
            .write_back => "st.global.wb.L2::cache_hint.v4.b32 [%[a]], %[b], %[policy];",
            .cache_global_level => "st.global.cg.L2::cache_hint.v4.b32 [%[a]], %[b], %[policy];",
            .cache_streaming => "st.global.cs.L2::cache_hint.v4.b32 [%[a]], %[b], %[policy];",
            .write_through => "st.global.wt.L2::cache_hint.v4.b32 [%[a]], %[b], %[policy];",
        },
        .l1_eviction_priority => |priority| switch (priority) {
            .evict_normal => "st.global.L1::evict_normal.L2::cache_hint.v4.b32 [%[a]], %[b], %[policy];",
            .evict_unchanged => "st.global.L1::evict_unchanged.L2::cache_hint.v4.b32 [%[a]], %[b], %[policy];",
            .evict_first => "st.global.L1::evict_first.L2::cache_hint.v4.b32 [%[a]], %[b], %[policy];",
            .evict_last => "st.global.L1::evict_last.L2::cache_hint.v4.b32 [%[a]], %[b], %[policy];",
            .no_allocate => "st.global.L1::no_allocate.L2::cache_hint.v4.b32 [%[a]], %[b], %[policy];",
        },
    };
    asm volatile (text
        :
        : [a] "l" (address),
          [b] "r" (@as([4]u32, value)),
          [policy] "l" (@backingInt(policy)),
        : .{ .memory = true });
}

/// Emits `ld.global` with a cache operator or an L1 eviction priority (section 9.7.10.8). The
/// type comes from the pointer. It is an integer, a float or a vector of 2 or 4 of them, and
/// has 8, 16, 32 or 64 bits per element.
pub inline fn load(
    comptime caching: LoadCaching,
    ptr: anytype,
) @typeInfo(@TypeOf(ptr)).pointer.child {
    switch (caching) {
        .operator => require("ld.global", .{ .ptx = 20, .sm = .{ .base = 20 }, .threads = .thread }),
        .l1_eviction_priority => require("ld.global.L1::evict", .{
            .ptx = 74,
            .sm = .{ .base = 70 },
            .threads = .thread,
        }),
    }
    if ((@typeInfo(@TypeOf(ptr)).pointer.attrs.@"addrspace" orelse .generic) != .global) {
        @compileError("ld.global needs a pointer in .global, found '" ++ @typeName(@TypeOf(ptr)) ++ "'");
    }
    const address = @intFromPtr(ptr);
    const Element = @typeInfo(@TypeOf(ptr)).pointer.child;
    const lanes = switch (@typeInfo(Element)) {
        .vector => |vector| vector.len,
        else => 1,
    };
    const bits = switch (@typeInfo(Element)) {
        .vector => |vector| @bitSizeOf(vector.child),
        else => @bitSizeOf(Element),
    };
    return switch (lanes) {
        1 => switch (bits) {
            8 => @bitCast(loadB8(caching, address)),
            16 => @bitCast(loadB16(caching, address)),
            32 => @bitCast(loadB32(caching, address)),
            64 => @bitCast(loadB64(caching, address)),
            else => @compileError("ld.global cannot access '" ++ @typeName(Element) ++ "'"),
        },
        2 => switch (bits) {
            8 => @bitCast(loadV2B8(caching, address)),
            16 => @bitCast(loadV2B16(caching, address)),
            32 => @bitCast(loadV2B32(caching, address)),
            64 => @bitCast(loadV2B64(caching, address)),
            else => @compileError("ld.global cannot access '" ++ @typeName(Element) ++ "'"),
        },
        4 => switch (bits) {
            8 => @bitCast(loadV4B8(caching, address)),
            16 => @bitCast(loadV4B16(caching, address)),
            32 => @bitCast(loadV4B32(caching, address)),
            else => @compileError("ld.global cannot access '" ++ @typeName(Element) ++ "'"),
        },
        else => @compileError("ld.global cannot access '" ++ @typeName(Element) ++ "'"),
    };
}

/// Emits `ld.global.L2::cache_hint` (section 9.7.10.8). The L2 cache treats the access as
/// `policy` says.
pub inline fn loadWithPolicy(
    comptime caching: LoadCaching,
    ptr: anytype,
    policy: CachePolicy,
) @typeInfo(@TypeOf(ptr)).pointer.child {
    require("ld.global.L2::cache_hint", .{ .ptx = 74, .sm = .{ .base = 80 }, .threads = .thread });
    if ((@typeInfo(@TypeOf(ptr)).pointer.attrs.@"addrspace" orelse .generic) != .global) {
        @compileError("ld.global needs a pointer in .global, found '" ++ @typeName(@TypeOf(ptr)) ++ "'");
    }
    const address = @intFromPtr(ptr);
    const Element = @typeInfo(@TypeOf(ptr)).pointer.child;
    const lanes = switch (@typeInfo(Element)) {
        .vector => |vector| vector.len,
        else => 1,
    };
    const bits = switch (@typeInfo(Element)) {
        .vector => |vector| @bitSizeOf(vector.child),
        else => @bitSizeOf(Element),
    };
    return switch (lanes) {
        1 => switch (bits) {
            8 => @bitCast(loadB8WithPolicy(caching, address, policy)),
            16 => @bitCast(loadB16WithPolicy(caching, address, policy)),
            32 => @bitCast(loadB32WithPolicy(caching, address, policy)),
            64 => @bitCast(loadB64WithPolicy(caching, address, policy)),
            else => @compileError("ld.global cannot access '" ++ @typeName(Element) ++ "'"),
        },
        2 => switch (bits) {
            8 => @bitCast(loadV2B8WithPolicy(caching, address, policy)),
            16 => @bitCast(loadV2B16WithPolicy(caching, address, policy)),
            32 => @bitCast(loadV2B32WithPolicy(caching, address, policy)),
            64 => @bitCast(loadV2B64WithPolicy(caching, address, policy)),
            else => @compileError("ld.global cannot access '" ++ @typeName(Element) ++ "'"),
        },
        4 => switch (bits) {
            8 => @bitCast(loadV4B8WithPolicy(caching, address, policy)),
            16 => @bitCast(loadV4B16WithPolicy(caching, address, policy)),
            32 => @bitCast(loadV4B32WithPolicy(caching, address, policy)),
            else => @compileError("ld.global cannot access '" ++ @typeName(Element) ++ "'"),
        },
        else => @compileError("ld.global cannot access '" ++ @typeName(Element) ++ "'"),
    };
}

/// Emits `ld.global.nc` (section 9.7.10.9), a load through the non-coherent cache. Assumes that
/// the memory does not change while the kernel runs.
pub inline fn loadNonCoherent(
    comptime op: NonCoherentOperator,
    ptr: anytype,
) @typeInfo(@TypeOf(ptr)).pointer.child {
    require("ld.global.nc", .{ .ptx = 31, .sm = .{ .base = 32 }, .threads = .thread });
    if ((@typeInfo(@TypeOf(ptr)).pointer.attrs.@"addrspace" orelse .generic) != .global) {
        @compileError("ld.global.nc needs a pointer in .global, found '" ++ @typeName(@TypeOf(ptr)) ++ "'");
    }
    const address = @intFromPtr(ptr);
    const Element = @typeInfo(@TypeOf(ptr)).pointer.child;
    const lanes = switch (@typeInfo(Element)) {
        .vector => |vector| vector.len,
        else => 1,
    };
    const bits = switch (@typeInfo(Element)) {
        .vector => |vector| @bitSizeOf(vector.child),
        else => @bitSizeOf(Element),
    };
    return switch (lanes) {
        1 => switch (bits) {
            8 => @bitCast(loadNonCoherentB8(op, address)),
            16 => @bitCast(loadNonCoherentB16(op, address)),
            32 => @bitCast(loadNonCoherentB32(op, address)),
            64 => @bitCast(loadNonCoherentB64(op, address)),
            else => @compileError("ld.global.nc cannot access '" ++ @typeName(Element) ++ "'"),
        },
        2 => switch (bits) {
            8 => @bitCast(loadNonCoherentV2B8(op, address)),
            16 => @bitCast(loadNonCoherentV2B16(op, address)),
            32 => @bitCast(loadNonCoherentV2B32(op, address)),
            64 => @bitCast(loadNonCoherentV2B64(op, address)),
            else => @compileError("ld.global.nc cannot access '" ++ @typeName(Element) ++ "'"),
        },
        4 => switch (bits) {
            8 => @bitCast(loadNonCoherentV4B8(op, address)),
            16 => @bitCast(loadNonCoherentV4B16(op, address)),
            32 => @bitCast(loadNonCoherentV4B32(op, address)),
            else => @compileError("ld.global.nc cannot access '" ++ @typeName(Element) ++ "'"),
        },
        else => @compileError("ld.global.nc cannot access '" ++ @typeName(Element) ++ "'"),
    };
}

/// Emits `ld.global.nc.L2::cache_hint` (section 9.7.10.9). Assumes that the memory does not
/// change while the kernel runs.
pub inline fn loadNonCoherentWithPolicy(
    comptime op: NonCoherentOperator,
    ptr: anytype,
    policy: CachePolicy,
) @typeInfo(@TypeOf(ptr)).pointer.child {
    require("ld.global.nc.L2::cache_hint", .{ .ptx = 74, .sm = .{ .base = 80 }, .threads = .thread });
    if ((@typeInfo(@TypeOf(ptr)).pointer.attrs.@"addrspace" orelse .generic) != .global) {
        @compileError("ld.global.nc needs a pointer in .global, found '" ++ @typeName(@TypeOf(ptr)) ++ "'");
    }
    const address = @intFromPtr(ptr);
    const Element = @typeInfo(@TypeOf(ptr)).pointer.child;
    const lanes = switch (@typeInfo(Element)) {
        .vector => |vector| vector.len,
        else => 1,
    };
    const bits = switch (@typeInfo(Element)) {
        .vector => |vector| @bitSizeOf(vector.child),
        else => @bitSizeOf(Element),
    };
    return switch (lanes) {
        1 => switch (bits) {
            8 => @bitCast(loadNonCoherentB8WithPolicy(op, address, policy)),
            16 => @bitCast(loadNonCoherentB16WithPolicy(op, address, policy)),
            32 => @bitCast(loadNonCoherentB32WithPolicy(op, address, policy)),
            64 => @bitCast(loadNonCoherentB64WithPolicy(op, address, policy)),
            else => @compileError("ld.global.nc cannot access '" ++ @typeName(Element) ++ "'"),
        },
        2 => switch (bits) {
            8 => @bitCast(loadNonCoherentV2B8WithPolicy(op, address, policy)),
            16 => @bitCast(loadNonCoherentV2B16WithPolicy(op, address, policy)),
            32 => @bitCast(loadNonCoherentV2B32WithPolicy(op, address, policy)),
            64 => @bitCast(loadNonCoherentV2B64WithPolicy(op, address, policy)),
            else => @compileError("ld.global.nc cannot access '" ++ @typeName(Element) ++ "'"),
        },
        4 => switch (bits) {
            8 => @bitCast(loadNonCoherentV4B8WithPolicy(op, address, policy)),
            16 => @bitCast(loadNonCoherentV4B16WithPolicy(op, address, policy)),
            32 => @bitCast(loadNonCoherentV4B32WithPolicy(op, address, policy)),
            else => @compileError("ld.global.nc cannot access '" ++ @typeName(Element) ++ "'"),
        },
        else => @compileError("ld.global.nc cannot access '" ++ @typeName(Element) ++ "'"),
    };
}

/// Emits `st.global` with a cache operator or an L1 eviction priority (section 9.7.10.11). The
/// type comes from the pointer and has the same restrictions as in `load`.
pub inline fn store(
    comptime caching: StoreCaching,
    ptr: anytype,
    value: @typeInfo(@TypeOf(ptr)).pointer.child,
) void {
    switch (caching) {
        .operator => require("st.global", .{ .ptx = 20, .sm = .{ .base = 20 }, .threads = .thread }),
        .l1_eviction_priority => require("st.global.L1::evict", .{
            .ptx = 74,
            .sm = .{ .base = 70 },
            .threads = .thread,
        }),
    }
    if ((@typeInfo(@TypeOf(ptr)).pointer.attrs.@"addrspace" orelse .generic) != .global) {
        @compileError("st.global needs a pointer in .global, found '" ++ @typeName(@TypeOf(ptr)) ++ "'");
    }
    const address = @intFromPtr(ptr);
    const Element = @TypeOf(value);
    const lanes = switch (@typeInfo(Element)) {
        .vector => |vector| vector.len,
        else => 1,
    };
    const bits = switch (@typeInfo(Element)) {
        .vector => |vector| @bitSizeOf(vector.child),
        else => @bitSizeOf(Element),
    };
    switch (lanes) {
        1 => switch (bits) {
            8 => storeB8(caching, address, @bitCast(value)),
            16 => storeB16(caching, address, @bitCast(value)),
            32 => storeB32(caching, address, @bitCast(value)),
            64 => storeB64(caching, address, @bitCast(value)),
            else => @compileError("st.global cannot access '" ++ @typeName(Element) ++ "'"),
        },
        2 => switch (bits) {
            8 => storeV2B8(caching, address, @bitCast(value)),
            16 => storeV2B16(caching, address, @bitCast(value)),
            32 => storeV2B32(caching, address, @bitCast(value)),
            64 => storeV2B64(caching, address, @bitCast(value)),
            else => @compileError("st.global cannot access '" ++ @typeName(Element) ++ "'"),
        },
        4 => switch (bits) {
            8 => storeV4B8(caching, address, @bitCast(value)),
            16 => storeV4B16(caching, address, @bitCast(value)),
            32 => storeV4B32(caching, address, @bitCast(value)),
            else => @compileError("st.global cannot access '" ++ @typeName(Element) ++ "'"),
        },
        else => @compileError("st.global cannot access '" ++ @typeName(Element) ++ "'"),
    }
}

/// Emits `st.global.L2::cache_hint` (section 9.7.10.11). The L2 cache treats the access as
/// `policy` says.
pub inline fn storeWithPolicy(
    comptime caching: StoreCaching,
    ptr: anytype,
    value: @typeInfo(@TypeOf(ptr)).pointer.child,
    policy: CachePolicy,
) void {
    require("st.global.L2::cache_hint", .{ .ptx = 74, .sm = .{ .base = 80 }, .threads = .thread });
    if ((@typeInfo(@TypeOf(ptr)).pointer.attrs.@"addrspace" orelse .generic) != .global) {
        @compileError("st.global needs a pointer in .global, found '" ++ @typeName(@TypeOf(ptr)) ++ "'");
    }
    const address = @intFromPtr(ptr);
    const Element = @TypeOf(value);
    const lanes = switch (@typeInfo(Element)) {
        .vector => |vector| vector.len,
        else => 1,
    };
    const bits = switch (@typeInfo(Element)) {
        .vector => |vector| @bitSizeOf(vector.child),
        else => @bitSizeOf(Element),
    };
    switch (lanes) {
        1 => switch (bits) {
            8 => storeB8WithPolicy(caching, address, @bitCast(value), policy),
            16 => storeB16WithPolicy(caching, address, @bitCast(value), policy),
            32 => storeB32WithPolicy(caching, address, @bitCast(value), policy),
            64 => storeB64WithPolicy(caching, address, @bitCast(value), policy),
            else => @compileError("st.global cannot access '" ++ @typeName(Element) ++ "'"),
        },
        2 => switch (bits) {
            8 => storeV2B8WithPolicy(caching, address, @bitCast(value), policy),
            16 => storeV2B16WithPolicy(caching, address, @bitCast(value), policy),
            32 => storeV2B32WithPolicy(caching, address, @bitCast(value), policy),
            64 => storeV2B64WithPolicy(caching, address, @bitCast(value), policy),
            else => @compileError("st.global cannot access '" ++ @typeName(Element) ++ "'"),
        },
        4 => switch (bits) {
            8 => storeV4B8WithPolicy(caching, address, @bitCast(value), policy),
            16 => storeV4B16WithPolicy(caching, address, @bitCast(value), policy),
            32 => storeV4B32WithPolicy(caching, address, @bitCast(value), policy),
            else => @compileError("st.global cannot access '" ++ @typeName(Element) ++ "'"),
        },
        else => @compileError("st.global cannot access '" ++ @typeName(Element) ++ "'"),
    }
}

pub const PrefetchLevel = enum {
    /// `.L1`
    l1,
    /// `.L2`
    l2,
    /// `.L2::evict_last`
    l2_evict_last,
    /// `.L2::evict_normal`
    l2_evict_normal,
};

/// Emits `prefetch` (section 9.7.10.16). Accepts a pointer in `.global` or in `.generic`, and
/// emits the `.global` qualifier for the former. The eviction priority levels need a `.global`
/// pointer.
pub inline fn prefetch(comptime level: PrefetchLevel, ptr: anytype) void {
    switch (level) {
        .l1, .l2 => require("prefetch", .{ .ptx = 20, .sm = .{ .base = 20 }, .threads = .thread }),
        .l2_evict_last, .l2_evict_normal => require("prefetch.L2::evict", .{
            .ptx = 74,
            .sm = .{ .base = 80 },
            .threads = .thread,
        }),
    }
    const text = comptime switch (@typeInfo(@TypeOf(ptr)).pointer.attrs.@"addrspace" orelse .generic) {
        .global => switch (level) {
            .l1 => "prefetch.global.L1 [%[a]];",
            .l2 => "prefetch.global.L2 [%[a]];",
            .l2_evict_last => "prefetch.global.L2::evict_last [%[a]];",
            .l2_evict_normal => "prefetch.global.L2::evict_normal [%[a]];",
        },
        .generic => switch (level) {
            .l1 => "prefetch.L1 [%[a]];",
            .l2 => "prefetch.L2 [%[a]];",
            .l2_evict_last, .l2_evict_normal => @compileError("prefetch with an L2 priority needs .global"),
        },
        else => @compileError("prefetch needs a pointer in .global or .generic"),
    };
    asm volatile (text
        :
        : [a] "l" (ptr),
        : .{ .memory = true });
}

/// Emits `prefetchu.L1` (section 9.7.10.16). The instruction needs a pointer in `.generic`.
/// Emits `prefetch.tensormap` (section 9.7.10.16) on the tensor map that `ptr` points to. Accepts a
/// pointer in `.generic`, `.const` or `.param`.
pub inline fn prefetchTensormap(ptr: anytype) void {
    require("prefetch.tensormap", .{ .ptx = 80, .sm = .{ .base = 90 }, .threads = .thread });
    const text = comptime switch (@typeInfo(@TypeOf(ptr)).pointer.attrs.@"addrspace" orelse .generic) {
        .generic => "prefetch.tensormap [%[a]];",
        .constant => "prefetch.const.tensormap [%[a]];",
        .param => "prefetch.param.tensormap [%[a]];",
        else => @compileError("prefetch.tensormap needs a pointer in .generic, .const or .param"),
    };
    asm volatile (text
        :
        : [a] "l" (ptr),
        : .{ .memory = true });
}

pub inline fn prefetchu(ptr: anytype) void {
    require("prefetchu", .{ .ptx = 20, .sm = .{ .base = 20 }, .threads = .thread });
    if ((@typeInfo(@TypeOf(ptr)).pointer.attrs.@"addrspace" orelse .generic) != .generic) {
        @compileError("prefetchu needs a pointer in .generic");
    }
    asm volatile ("prefetchu.L1 [%[a]];"
        :
        : [a] "l" (ptr),
        : .{ .memory = true });
}

/// Emits `createpolicy.fractional` (section 9.7.10.21). A fraction `fraction` of the accesses
/// get the `primary` priority and the rest get `secondary`, which is `evict_unchanged` when null.
/// Asserts that `fraction` is greater than 0 and at most 1. The secondary priority is
/// `evict_first` or `evict_unchanged`.
pub inline fn createpolicy(
    comptime primary: L2EvictionPriority,
    comptime secondary: ?L2EvictionPriority,
    comptime fraction: f32,
) CachePolicy {
    require("createpolicy", .{ .ptx = 74, .sm = .{ .base = 80 }, .threads = .thread });
    if (!(fraction > 0 and fraction <= 1)) @compileError("the fraction is greater than 0 and at most 1");
    const text = comptime switch (primary) {
        .evict_last => if (secondary) |priority| switch (priority) {
            .evict_first => "createpolicy.fractional.L2::evict_last.L2::evict_first.b64 %[policy], %[fraction];",
            .evict_unchanged => "createpolicy.fractional.L2::evict_last.L2::evict_unchanged.b64 %[policy], %[fraction];",
            else => @compileError("the secondary priority is evict_first or evict_unchanged"),
        } else "createpolicy.fractional.L2::evict_last.b64 %[policy], %[fraction];",
        .evict_normal => if (secondary) |priority| switch (priority) {
            .evict_first => "createpolicy.fractional.L2::evict_normal.L2::evict_first.b64 %[policy], %[fraction];",
            .evict_unchanged => "createpolicy.fractional.L2::evict_normal.L2::evict_unchanged.b64 %[policy], %[fraction];",
            else => @compileError("the secondary priority is evict_first or evict_unchanged"),
        } else "createpolicy.fractional.L2::evict_normal.b64 %[policy], %[fraction];",
        .evict_first => if (secondary) |priority| switch (priority) {
            .evict_first => "createpolicy.fractional.L2::evict_first.L2::evict_first.b64 %[policy], %[fraction];",
            .evict_unchanged => "createpolicy.fractional.L2::evict_first.L2::evict_unchanged.b64 %[policy], %[fraction];",
            else => @compileError("the secondary priority is evict_first or evict_unchanged"),
        } else "createpolicy.fractional.L2::evict_first.b64 %[policy], %[fraction];",
        .evict_unchanged => if (secondary) |priority| switch (priority) {
            .evict_first => "createpolicy.fractional.L2::evict_unchanged.L2::evict_first.b64 %[policy], %[fraction];",
            .evict_unchanged => "createpolicy.fractional.L2::evict_unchanged.L2::evict_unchanged.b64 %[policy], %[fraction];",
            else => @compileError("the secondary priority is evict_first or evict_unchanged"),
        } else "createpolicy.fractional.L2::evict_unchanged.b64 %[policy], %[fraction];",
    };
    const policy = asm (text
        : [policy] "=l" (-> u64),
        : [fraction] "f" (fraction),
    );
    return @fromBackingInt(@intCast(policy));
}

/// Emits `createpolicy.range.global` (section 9.7.10.21). Accesses in the `primary_size` bytes
/// from `ptr` get the `primary` priority. Accesses in the rest of the `total_size` bytes around
/// them get `secondary`, which is `evict_unchanged` when null. Assumes that `primary_size` is at
/// most `total_size`.
pub inline fn createpolicyRange(
    comptime primary: L2EvictionPriority,
    comptime secondary: ?L2EvictionPriority,
    ptr: anytype,
    primary_size: u32,
    total_size: u32,
) CachePolicy {
    require("createpolicy.range", .{ .ptx = 74, .sm = .{ .base = 80 }, .threads = .thread });
    if ((@typeInfo(@TypeOf(ptr)).pointer.attrs.@"addrspace" orelse .generic) != .global) {
        @compileError("createpolicy.range.global needs a pointer in .global, found '" ++ @typeName(@TypeOf(ptr)) ++ "'");
    }
    const text = comptime switch (primary) {
        .evict_last => if (secondary) |priority| switch (priority) {
            .evict_first => "createpolicy.range.global.L2::evict_last.L2::evict_first.b64 %[policy], [%[a]], %[primary_size], %[total_size];",
            .evict_unchanged => "createpolicy.range.global.L2::evict_last.L2::evict_unchanged.b64 %[policy], [%[a]], %[primary_size], %[total_size];",
            else => @compileError("the secondary priority is evict_first or evict_unchanged"),
        } else "createpolicy.range.global.L2::evict_last.b64 %[policy], [%[a]], %[primary_size], %[total_size];",
        .evict_normal => if (secondary) |priority| switch (priority) {
            .evict_first => "createpolicy.range.global.L2::evict_normal.L2::evict_first.b64 %[policy], [%[a]], %[primary_size], %[total_size];",
            .evict_unchanged => "createpolicy.range.global.L2::evict_normal.L2::evict_unchanged.b64 %[policy], [%[a]], %[primary_size], %[total_size];",
            else => @compileError("the secondary priority is evict_first or evict_unchanged"),
        } else "createpolicy.range.global.L2::evict_normal.b64 %[policy], [%[a]], %[primary_size], %[total_size];",
        .evict_first => if (secondary) |priority| switch (priority) {
            .evict_first => "createpolicy.range.global.L2::evict_first.L2::evict_first.b64 %[policy], [%[a]], %[primary_size], %[total_size];",
            .evict_unchanged => "createpolicy.range.global.L2::evict_first.L2::evict_unchanged.b64 %[policy], [%[a]], %[primary_size], %[total_size];",
            else => @compileError("the secondary priority is evict_first or evict_unchanged"),
        } else "createpolicy.range.global.L2::evict_first.b64 %[policy], [%[a]], %[primary_size], %[total_size];",
        .evict_unchanged => if (secondary) |priority| switch (priority) {
            .evict_first => "createpolicy.range.global.L2::evict_unchanged.L2::evict_first.b64 %[policy], [%[a]], %[primary_size], %[total_size];",
            .evict_unchanged => "createpolicy.range.global.L2::evict_unchanged.L2::evict_unchanged.b64 %[policy], [%[a]], %[primary_size], %[total_size];",
            else => @compileError("the secondary priority is evict_first or evict_unchanged"),
        } else "createpolicy.range.global.L2::evict_unchanged.b64 %[policy], [%[a]], %[primary_size], %[total_size];",
    };
    const policy = asm (text
        : [policy] "=l" (-> u64),
        : [a] "l" (ptr),
          [primary_size] "r" (primary_size),
          [total_size] "r" (total_size),
    );
    return @fromBackingInt(@intCast(policy));
}

/// Emits `applypriority.global.L2::evict_normal` for the 128 bytes at `ptr` (section 9.7.10.17).
/// Asserts that `ptr` is aligned to 128 bytes.
pub inline fn applypriority(ptr: anytype) void {
    require("applypriority", .{ .ptx = 74, .sm = .{ .base = 80 }, .threads = .thread });
    if ((@typeInfo(@TypeOf(ptr)).pointer.attrs.@"addrspace" orelse .generic) != .global) {
        @compileError("applypriority.global needs a pointer in .global, found '" ++ @typeName(@TypeOf(ptr)) ++ "'");
    }
    nvptx.assertAligned(ptr, 128);
    asm volatile ("applypriority.global.L2::evict_normal [%[a]], 128;"
        :
        : [a] "l" (ptr),
        : .{ .memory = true });
}

/// Emits `discard.global.L2` for the 128 bytes at `ptr` (section 9.7.10.20). Later reads of the
/// range can return different values until the range is written. Asserts that `ptr` is aligned
/// to 128 bytes.
pub inline fn discard(ptr: anytype) void {
    require("discard", .{ .ptx = 74, .sm = .{ .base = 80 }, .threads = .thread });
    if ((@typeInfo(@TypeOf(ptr)).pointer.attrs.@"addrspace" orelse .generic) != .global) {
        @compileError("discard.global needs a pointer in .global, found '" ++ @typeName(@TypeOf(ptr)) ++ "'");
    }
    nvptx.assertAligned(ptr, 128);
    asm volatile ("discard.global.L2 [%[a]], 128;"
        :
        : [a] "l" (ptr),
        : .{ .memory = true });
}
