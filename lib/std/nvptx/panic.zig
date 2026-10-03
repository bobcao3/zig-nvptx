//! A panic handler for kernels: `pub const panic = std.nvptx.panic;` in the root file. Each panic
//! calls the `__assertfail` function of the driver, which prints the message with the block and the
//! thread and makes the launch fail with `CUDA_ERROR_ASSERT`. The messages are string literals, and
//! `call` copies a message into global memory, so no panic needs stack space. The call still adds
//! registers and spills to some kernels, so `std.lang.panic` selects `std.debug.no_panic`.

const std = @import("../std.zig");
const nvptx = std.nvptx;

/// The first `call` claims `call_message`, and a later `call` reports "panic".
var call_claimed: u32 addrspace(.global) = 0;
var call_message: [256:0]u8 addrspace(.global) = undefined;

pub fn call(msg: []const u8, ra: ?usize) noreturn {
    @branchHint(.cold);
    _ = ra;
    if (@atomicRmw(u32, &call_claimed, .Xchg, 1, .monotonic) != 0) fail("panic");
    const len = @min(msg.len, call_message.len);
    for (call_message[0..len], msg[0..len]) |*byte, source| byte.* = source;
    call_message[len] = 0;
    const message: *const [call_message.len:0]u8 = @addrSpaceCast(&call_message);
    fail(message);
}

fn fail(message: [*:0]const u8) noreturn {
    @branchHint(.cold);
    nvptx.assertFail(message, "", 0, "", 1);
    @trap();
}

pub fn sentinelMismatch(expected: anytype, found: @TypeOf(expected)) noreturn {
    @branchHint(.cold);
    _ = found;
    fail("sentinel mismatch");
}

pub fn unwrapError(err: anyerror) noreturn {
    @branchHint(.cold);
    _ = &err;
    fail("attempt to unwrap error");
}

pub fn outOfBounds(index: usize, len: usize) noreturn {
    @branchHint(.cold);
    _ = index;
    _ = len;
    fail("index out of bounds");
}

pub fn startGreaterThanEnd(start: usize, end: usize) noreturn {
    @branchHint(.cold);
    _ = start;
    _ = end;
    fail("start index is larger than end index");
}

pub fn inactiveUnionField(active: anytype, accessed: @TypeOf(active)) noreturn {
    @branchHint(.cold);
    _ = accessed;
    fail("access of inactive union field");
}

pub fn sliceCastLenRemainder(src_len: usize) noreturn {
    @branchHint(.cold);
    _ = src_len;
    fail("slice length does not divide exactly into destination elements");
}

pub fn reachedUnreachable() noreturn {
    @branchHint(.cold);
    fail("reached unreachable code");
}

pub fn unwrapNull() noreturn {
    @branchHint(.cold);
    fail("attempt to use null value");
}

pub fn castToNull() noreturn {
    @branchHint(.cold);
    fail("cast causes pointer to be null");
}

pub fn incorrectAlignment() noreturn {
    @branchHint(.cold);
    fail("incorrect alignment");
}

pub fn invalidErrorCode() noreturn {
    @branchHint(.cold);
    fail("invalid error code");
}

pub fn unexpectedErrorCode(err: anyerror) noreturn {
    @branchHint(.cold);
    _ = &err;
    fail("unexpected error code");
}

pub fn integerOutOfBounds() noreturn {
    @branchHint(.cold);
    fail("integer does not fit in destination type");
}

pub fn integerOverflow() noreturn {
    @branchHint(.cold);
    fail("integer overflow");
}

pub fn shlOverflow() noreturn {
    @branchHint(.cold);
    fail("left shift overflowed bits");
}

pub fn shrOverflow() noreturn {
    @branchHint(.cold);
    fail("right shift overflowed bits");
}

pub fn divideByZero() noreturn {
    @branchHint(.cold);
    fail("division by zero");
}

pub fn exactDivisionRemainder() noreturn {
    @branchHint(.cold);
    fail("exact division produced remainder");
}

pub fn integerPartOutOfBounds() noreturn {
    @branchHint(.cold);
    fail("integer part of floating point value out of bounds");
}

pub fn corruptSwitch() noreturn {
    @branchHint(.cold);
    fail("switch on corrupt value");
}

pub fn shiftRhsTooBig() noreturn {
    @branchHint(.cold);
    fail("shift amount is greater than the type size");
}

pub fn invalidEnumValue() noreturn {
    @branchHint(.cold);
    fail("invalid enum value");
}

pub fn forLenMismatch() noreturn {
    @branchHint(.cold);
    fail("for loop over objects with non-equal lengths");
}

pub fn copyLenMismatch() noreturn {
    @branchHint(.cold);
    fail("source and destination have non-equal lengths");
}

pub fn memcpyAlias() noreturn {
    @branchHint(.cold);
    fail("@memcpy arguments alias");
}

pub fn noreturnReturned() noreturn {
    @branchHint(.cold);
    fail("'noreturn' function returned");
}

pub fn loadUninstantiableType() noreturn {
    @branchHint(.cold);
    fail("attempt to load uninstantiable type");
}
