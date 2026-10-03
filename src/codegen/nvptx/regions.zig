//! For each block, this file counts the `br` instructions that leave the block. Code generation
//! needs the count before it enters the body.

const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;

const Air = @import("../../Air.zig");

pub const Region = struct {
    brs: u32,
    /// The field is true when the block ends with a `br` to the block itself.
    final_br: bool,
};

/// Maps each `block` and `dbg_inline_block` to its region.
pub const Regions = std.AutoHashMapUnmanaged(Air.Inst.Index, Region);

pub fn find(arena: Allocator, air: *const Air) Allocator.Error!Regions {
    var finder: Finder = .{ .arena = arena, .air = air };
    try finder.body(air.getMainBody());
    return finder.regions;
}

const Finder = struct {
    arena: Allocator,
    air: *const Air,
    regions: Regions = .empty,

    fn body(f: *Finder, insts: []const Air.Inst.Index) Allocator.Error!void {
        for (insts) |inst| try f.instruction(inst);
    }

    fn region(f: *Finder, inst: Air.Inst.Index, insts: []const Air.Inst.Index) Allocator.Error!void {
        try f.regions.putNoClobber(f.arena, inst, .{ .brs = 0, .final_br = false });
        try f.body(insts);
        assert(insts.len != 0);
        const end = insts[insts.len - 1];
        const final_br = f.air.instructions.items(.tag)[@backingInt(end)] == .br and
            f.air.instructions.items(.data)[@backingInt(end)].br.block_inst == inst;
        f.regions.getPtr(inst).?.final_br = final_br;
    }

    fn instruction(f: *Finder, inst: Air.Inst.Index) Allocator.Error!void {
        const air = f.air;
        const data = air.instructions.items(.data)[@backingInt(inst)];
        switch (air.instructions.items(.tag)[@backingInt(inst)]) {
            .block => try f.region(inst, air.unwrapBlock(inst).body),
            .dbg_inline_block => try f.region(inst, air.unwrapDbgBlock(inst).body),
            .loop => try f.body(air.unwrapBlock(inst).body),
            .loop_switch_br, .switch_br => {
                const switch_br = air.unwrapSwitch(inst);
                var cases = switch_br.iterateCases();
                while (cases.next()) |case| try f.body(case.body);
                try f.body(cases.elseBody());
            },
            .cond_br => {
                const cond_br = air.unwrapCondBr(inst);
                try f.body(cond_br.then_body);
                try f.body(cond_br.else_body);
            },
            .@"try", .try_cold => try f.body(air.unwrapTry(inst).else_body),
            .try_ptr, .try_ptr_cold => try f.body(air.unwrapTryPtr(inst).else_body),
            .br => {
                f.regions.getPtr(data.br.block_inst).?.brs += 1;
            },
            else => {},
        }
    }
};
