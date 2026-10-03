//! Rounding, saturation, directed-rounding arithmetic and bit manipulation instructions of PTX
//! that the Zig operators and builtins do not express.

const std = @import("../std.zig");
const nvptx = std.nvptx;

const always: nvptx.Requirement = .{ .ptx = 10, .sm = .{ .base = 10 }, .threads = .thread };
const sm_20: nvptx.Requirement = .{ .ptx = 20, .sm = .{ .base = 20 }, .threads = .thread };
const double: nvptx.Requirement = .{ .ptx = 10, .sm = .{ .base = 13 }, .threads = .thread };
const approximate: nvptx.Requirement = .{ .ptx = 14, .sm = .{ .base = 10 }, .threads = .thread };
const bf16_ex2: nvptx.Requirement = .{ .ptx = 78, .sm = .{ .base = 90 }, .threads = .thread };

pub const Rounding = enum {
    nearest_even,
    toward_zero,
    toward_negative_infinity,
    toward_positive_infinity,

    /// The PTX modifiers `.rn`, `.rz`, `.rm` and `.rp`.
    fn float(comptime rounding: Rounding) []const u8 {
        return switch (rounding) {
            .nearest_even => "rn",
            .toward_zero => "rz",
            .toward_negative_infinity => "rm",
            .toward_positive_infinity => "rp",
        };
    }

    /// The PTX modifiers `.rni`, `.rzi`, `.rmi` and `.rpi`.
    fn integer(comptime rounding: Rounding) []const u8 {
        return switch (rounding) {
            .nearest_even => "rni",
            .toward_zero => "rzi",
            .toward_negative_infinity => "rmi",
            .toward_positive_infinity => "rpi",
        };
    }
};

/// Modifiers of the `f32` forms of `add`, `sub`, `mul` and `fma`.
pub const Modifiers = struct {
    ftz: bool = false,
    saturate: bool = false,
};

pub const ConvertOptions = struct {
    /// Required for a conversion to a float from an integer or from a wider float and for a
    /// conversion from a float to an integer. Optional between floats of the same size, where it
    /// rounds to an integer. Illegal otherwise.
    rounding: ?Rounding = null,
    saturate: bool = false,
    ftz: bool = false,
    relu: bool = false,
};

fn unsupported(comptime name: []const u8, comptime T: type) noreturn {
    @compileError(name ++ " does not support '" ++ @typeName(T) ++ "'");
}

inline fn unary16(comptime text: []const u8, a: u16) u16 {
    return asm (text
        : [d] "=h" (-> u16),
        : [a] "h" (a),
    );
}

inline fn unary32(comptime text: []const u8, a: u32) u32 {
    return asm (text
        : [d] "=r" (-> u32),
        : [a] "r" (a),
    );
}

inline fn unaryF32(comptime text: []const u8, a: f32) f32 {
    return asm (text
        : [d] "=f" (-> f32),
        : [a] "f" (a),
    );
}

inline fn unaryF64(comptime text: []const u8, a: f64) f64 {
    return asm (text
        : [d] "=d" (-> f64),
        : [a] "d" (a),
    );
}

inline fn binary32(comptime text: []const u8, a: u32, b: u32) u32 {
    return asm (text
        : [d] "=r" (-> u32),
        : [a] "r" (a),
          [b] "r" (b),
    );
}

inline fn ternary16(comptime text: []const u8, a: u16, b: u16, c: u16) u16 {
    return asm (text
        : [d] "=h" (-> u16),
        : [a] "h" (a),
          [b] "h" (b),
          [c] "h" (c),
    );
}

inline fn ternary32(comptime text: []const u8, a: u32, b: u32, c: u32) u32 {
    return asm (text
        : [d] "=r" (-> u32),
        : [a] "r" (a),
          [b] "r" (b),
          [c] "r" (c),
    );
}

inline fn ternary64(comptime text: []const u8, a: u64, b: u64, c: u64) u64 {
    return asm (text
        : [d] "=l" (-> u64),
        : [a] "l" (a),
          [b] "l" (b),
          [c] "l" (c),
    );
}

fn typeName(comptime T: type) []const u8 {
    return switch (T) {
        f16 => "f16",
        f32 => "f32",
        f64 => "f64",
        i16 => "s16",
        u16 => "u16",
        i32 => "s32",
        u32 => "u32",
        i64 => "s64",
        u64 => "u64",
        else => unsupported("cvt", T),
    };
}

fn roundingModifier(
    comptime From: type,
    comptime To: type,
    comptime rounding: ?Rounding,
) []const u8 {
    const from_float = @typeInfo(From) == .float;
    const to_float = @typeInfo(To) == .float;
    const required = "this conversion needs a rounding mode";
    if (from_float and !to_float) return "." ++ (rounding orelse @compileError(required)).integer();
    if (!from_float and to_float) return "." ++ (rounding orelse @compileError(required)).float();
    if (!from_float and !to_float) {
        if (rounding != null) @compileError("a conversion between integers has no rounding mode");
        return "";
    }
    const from_bits = @typeInfo(From).float.bits;
    const to_bits = @typeInfo(To).float.bits;
    const mode = rounding orelse {
        if (to_bits < from_bits) @compileError(required);
        return "";
    };
    if (to_bits > from_bits) @compileError("a widening conversion has no rounding mode");
    return "." ++ if (to_bits == from_bits) mode.integer() else mode.float();
}

fn cvtRequirement(comptime To: type, comptime From: type, comptime options: ConvertOptions) nvptx.Requirement {
    if (options.relu and !(From == f32 and To == f16)) {
        @compileError("cvt with relu converts f32 to f16");
    }
    if (options.relu) return .{ .ptx = 70, .sm = .{ .base = 80 }, .threads = .thread };
    if (From == f64 or To == f64) return double;
    return always;
}

fn cvtText(comptime To: type, comptime From: type, comptime options: ConvertOptions) []const u8 {
    return "cvt" ++ roundingModifier(From, To, options.rounding) ++
        (if (options.relu) ".relu" else "") ++
        (if (options.ftz) ".ftz" else "") ++
        (if (options.saturate) ".sat" else "") ++
        "." ++ typeName(To) ++ "." ++ typeName(From) ++ " %[d], %[a];";
}

/// Evaluates the instruction text once for each distinct conversion.
fn Conversion(comptime To: type, comptime From: type, comptime options: ConvertOptions) type {
    return struct {
        const requirement = cvtRequirement(To, From, options);
        const text = cvtText(To, From, options);
    };
}

inline fn convert16From16(comptime text: []const u8, a: u16) u16 {
    return asm (text
        : [d] "=h" (-> u16),
        : [a] "h" (a),
    );
}

inline fn convert16From32(comptime text: []const u8, a: u32) u16 {
    return asm (text
        : [d] "=h" (-> u16),
        : [a] "r" (a),
    );
}

inline fn convert16From64(comptime text: []const u8, a: u64) u16 {
    return asm (text
        : [d] "=h" (-> u16),
        : [a] "l" (a),
    );
}

inline fn convert32From16(comptime text: []const u8, a: u16) u32 {
    return asm (text
        : [d] "=r" (-> u32),
        : [a] "h" (a),
    );
}

inline fn convert32From32(comptime text: []const u8, a: u32) u32 {
    return asm (text
        : [d] "=r" (-> u32),
        : [a] "r" (a),
    );
}

inline fn convert32From64(comptime text: []const u8, a: u64) u32 {
    return asm (text
        : [d] "=r" (-> u32),
        : [a] "l" (a),
    );
}

inline fn convert64From16(comptime text: []const u8, a: u16) u64 {
    return asm (text
        : [d] "=l" (-> u64),
        : [a] "h" (a),
    );
}

inline fn convert64From32(comptime text: []const u8, a: u32) u64 {
    return asm (text
        : [d] "=l" (-> u64),
        : [a] "r" (a),
    );
}

inline fn convert64From64(comptime text: []const u8, a: u64) u64 {
    return asm (text
        : [d] "=l" (-> u64),
        : [a] "l" (a),
    );
}

/// Emits `cvt` (section 9.7.10.24) from `From` to `To`. Both types are one of `f16`, `f32`, `f64`
/// and the signed and unsigned integers of 16, 32 and 64 bits. `options.relu` needs the conversion
/// from `f32` to `f16`.
pub inline fn cvt(comptime To: type, comptime From: type, x: From, comptime options: ConvertOptions) To {
    const instruction = Conversion(To, From, options);
    nvptx.require("cvt", instruction.requirement);
    const text = instruction.text;
    return @bitCast(switch (To) {
        f16, i16, u16 => switch (From) {
            f16, i16, u16 => convert16From16(text, @bitCast(x)),
            f32, i32, u32 => convert16From32(text, @bitCast(x)),
            f64, i64, u64 => convert16From64(text, @bitCast(x)),
            else => unsupported("cvt", From),
        },
        f32, i32, u32 => switch (From) {
            f16, i16, u16 => convert32From16(text, @bitCast(x)),
            f32, i32, u32 => convert32From32(text, @bitCast(x)),
            f64, i64, u64 => convert32From64(text, @bitCast(x)),
            else => unsupported("cvt", From),
        },
        f64, i64, u64 => switch (From) {
            f16, i16, u16 => convert64From16(text, @bitCast(x)),
            f32, i32, u32 => convert64From32(text, @bitCast(x)),
            f64, i64, u64 => convert64From64(text, @bitCast(x)),
            else => unsupported("cvt", From),
        },
        else => unsupported("cvt", To),
    });
}

/// Emits `add` on `f32` (section 9.7.3.3).
pub inline fn addF32(comptime rounding: Rounding, comptime modifiers: Modifiers, a: f32, b: f32) f32 {
    const directed = rounding == .toward_negative_infinity or rounding == .toward_positive_infinity;
    if (nvptx.gpu.ptx < 10 or nvptx.gpu.sm < (if (directed) 20 else 10)) @compileError("add on f32 needs a newer PTX version or target");
    return asm (switch (rounding) {
            .nearest_even => if (modifiers.ftz)
                if (modifiers.saturate)
                    "add.rn.ftz.sat.f32 %[d], %[a], %[b];"
                else
                    "add.rn.ftz.f32 %[d], %[a], %[b];"
            else if (modifiers.saturate)
                "add.rn.sat.f32 %[d], %[a], %[b];"
            else
                "add.rn.f32 %[d], %[a], %[b];",
            .toward_zero => if (modifiers.ftz)
                if (modifiers.saturate)
                    "add.rz.ftz.sat.f32 %[d], %[a], %[b];"
                else
                    "add.rz.ftz.f32 %[d], %[a], %[b];"
            else if (modifiers.saturate)
                "add.rz.sat.f32 %[d], %[a], %[b];"
            else
                "add.rz.f32 %[d], %[a], %[b];",
            .toward_negative_infinity => if (modifiers.ftz)
                if (modifiers.saturate)
                    "add.rm.ftz.sat.f32 %[d], %[a], %[b];"
                else
                    "add.rm.ftz.f32 %[d], %[a], %[b];"
            else if (modifiers.saturate)
                "add.rm.sat.f32 %[d], %[a], %[b];"
            else
                "add.rm.f32 %[d], %[a], %[b];",
            .toward_positive_infinity => if (modifiers.ftz)
                if (modifiers.saturate)
                    "add.rp.ftz.sat.f32 %[d], %[a], %[b];"
                else
                    "add.rp.ftz.f32 %[d], %[a], %[b];"
            else if (modifiers.saturate)
                "add.rp.sat.f32 %[d], %[a], %[b];"
            else
                "add.rp.f32 %[d], %[a], %[b];",
        }
        : [d] "=f" (-> f32),
        : [a] "f" (a),
          [b] "f" (b),
    );
}

/// Emits `add` (section 9.7.3.3). `T` is `f32` or `f64`. `addF32` takes the `f32` modifiers.
pub inline fn add(comptime T: type, comptime rounding: Rounding, a: T, b: T) T {
    switch (T) {
        f32 => return addF32(rounding, .{}, a, b),
        f64 => {
            if (nvptx.gpu.ptx < 10 or nvptx.gpu.sm < 13) @compileError("add on f64 needs a newer PTX version or target");
            return asm (switch (rounding) {
                    .nearest_even => "add.rn.f64 %[d], %[a], %[b];",
                    .toward_zero => "add.rz.f64 %[d], %[a], %[b];",
                    .toward_negative_infinity => "add.rm.f64 %[d], %[a], %[b];",
                    .toward_positive_infinity => "add.rp.f64 %[d], %[a], %[b];",
                }
                : [d] "=d" (-> f64),
                : [a] "d" (a),
                  [b] "d" (b),
            );
        },
        else => unsupported("add", T),
    }
}

/// Emits `sub` on `f32` (section 9.7.3.4).
pub inline fn subF32(comptime rounding: Rounding, comptime modifiers: Modifiers, a: f32, b: f32) f32 {
    const directed = rounding == .toward_negative_infinity or rounding == .toward_positive_infinity;
    if (nvptx.gpu.ptx < 10 or nvptx.gpu.sm < (if (directed) 20 else 10)) @compileError("sub on f32 needs a newer PTX version or target");
    return asm (switch (rounding) {
            .nearest_even => if (modifiers.ftz)
                if (modifiers.saturate)
                    "sub.rn.ftz.sat.f32 %[d], %[a], %[b];"
                else
                    "sub.rn.ftz.f32 %[d], %[a], %[b];"
            else if (modifiers.saturate)
                "sub.rn.sat.f32 %[d], %[a], %[b];"
            else
                "sub.rn.f32 %[d], %[a], %[b];",
            .toward_zero => if (modifiers.ftz)
                if (modifiers.saturate)
                    "sub.rz.ftz.sat.f32 %[d], %[a], %[b];"
                else
                    "sub.rz.ftz.f32 %[d], %[a], %[b];"
            else if (modifiers.saturate)
                "sub.rz.sat.f32 %[d], %[a], %[b];"
            else
                "sub.rz.f32 %[d], %[a], %[b];",
            .toward_negative_infinity => if (modifiers.ftz)
                if (modifiers.saturate)
                    "sub.rm.ftz.sat.f32 %[d], %[a], %[b];"
                else
                    "sub.rm.ftz.f32 %[d], %[a], %[b];"
            else if (modifiers.saturate)
                "sub.rm.sat.f32 %[d], %[a], %[b];"
            else
                "sub.rm.f32 %[d], %[a], %[b];",
            .toward_positive_infinity => if (modifiers.ftz)
                if (modifiers.saturate)
                    "sub.rp.ftz.sat.f32 %[d], %[a], %[b];"
                else
                    "sub.rp.ftz.f32 %[d], %[a], %[b];"
            else if (modifiers.saturate)
                "sub.rp.sat.f32 %[d], %[a], %[b];"
            else
                "sub.rp.f32 %[d], %[a], %[b];",
        }
        : [d] "=f" (-> f32),
        : [a] "f" (a),
          [b] "f" (b),
    );
}

/// Emits `sub` (section 9.7.3.4). `T` is `f32` or `f64`. `subF32` takes the `f32` modifiers.
pub inline fn sub(comptime T: type, comptime rounding: Rounding, a: T, b: T) T {
    switch (T) {
        f32 => return subF32(rounding, .{}, a, b),
        f64 => {
            if (nvptx.gpu.ptx < 10 or nvptx.gpu.sm < 13) @compileError("sub on f64 needs a newer PTX version or target");
            return asm (switch (rounding) {
                    .nearest_even => "sub.rn.f64 %[d], %[a], %[b];",
                    .toward_zero => "sub.rz.f64 %[d], %[a], %[b];",
                    .toward_negative_infinity => "sub.rm.f64 %[d], %[a], %[b];",
                    .toward_positive_infinity => "sub.rp.f64 %[d], %[a], %[b];",
                }
                : [d] "=d" (-> f64),
                : [a] "d" (a),
                  [b] "d" (b),
            );
        },
        else => unsupported("sub", T),
    }
}

/// Emits `mul` on `f32` (section 9.7.3.5).
pub inline fn mulF32(comptime rounding: Rounding, comptime modifiers: Modifiers, a: f32, b: f32) f32 {
    const directed = rounding == .toward_negative_infinity or rounding == .toward_positive_infinity;
    if (nvptx.gpu.ptx < 10 or nvptx.gpu.sm < (if (directed) 20 else 10)) @compileError("mul on f32 needs a newer PTX version or target");
    return asm (switch (rounding) {
            .nearest_even => if (modifiers.ftz)
                if (modifiers.saturate)
                    "mul.rn.ftz.sat.f32 %[d], %[a], %[b];"
                else
                    "mul.rn.ftz.f32 %[d], %[a], %[b];"
            else if (modifiers.saturate)
                "mul.rn.sat.f32 %[d], %[a], %[b];"
            else
                "mul.rn.f32 %[d], %[a], %[b];",
            .toward_zero => if (modifiers.ftz)
                if (modifiers.saturate)
                    "mul.rz.ftz.sat.f32 %[d], %[a], %[b];"
                else
                    "mul.rz.ftz.f32 %[d], %[a], %[b];"
            else if (modifiers.saturate)
                "mul.rz.sat.f32 %[d], %[a], %[b];"
            else
                "mul.rz.f32 %[d], %[a], %[b];",
            .toward_negative_infinity => if (modifiers.ftz)
                if (modifiers.saturate)
                    "mul.rm.ftz.sat.f32 %[d], %[a], %[b];"
                else
                    "mul.rm.ftz.f32 %[d], %[a], %[b];"
            else if (modifiers.saturate)
                "mul.rm.sat.f32 %[d], %[a], %[b];"
            else
                "mul.rm.f32 %[d], %[a], %[b];",
            .toward_positive_infinity => if (modifiers.ftz)
                if (modifiers.saturate)
                    "mul.rp.ftz.sat.f32 %[d], %[a], %[b];"
                else
                    "mul.rp.ftz.f32 %[d], %[a], %[b];"
            else if (modifiers.saturate)
                "mul.rp.sat.f32 %[d], %[a], %[b];"
            else
                "mul.rp.f32 %[d], %[a], %[b];",
        }
        : [d] "=f" (-> f32),
        : [a] "f" (a),
          [b] "f" (b),
    );
}

/// Emits `mul` (section 9.7.3.5). `T` is `f32` or `f64`. `mulF32` takes the `f32` modifiers.
pub inline fn mul(comptime T: type, comptime rounding: Rounding, a: T, b: T) T {
    switch (T) {
        f32 => return mulF32(rounding, .{}, a, b),
        f64 => {
            if (nvptx.gpu.ptx < 10 or nvptx.gpu.sm < 13) @compileError("mul on f64 needs a newer PTX version or target");
            return asm (switch (rounding) {
                    .nearest_even => "mul.rn.f64 %[d], %[a], %[b];",
                    .toward_zero => "mul.rz.f64 %[d], %[a], %[b];",
                    .toward_negative_infinity => "mul.rm.f64 %[d], %[a], %[b];",
                    .toward_positive_infinity => "mul.rp.f64 %[d], %[a], %[b];",
                }
                : [d] "=d" (-> f64),
                : [a] "d" (a),
                  [b] "d" (b),
            );
        },
        else => unsupported("mul", T),
    }
}

/// Emits `fma` on `f32` (section 9.7.3.6).
pub inline fn fmaF32(comptime rounding: Rounding, comptime modifiers: Modifiers, a: f32, b: f32, c: f32) f32 {
    if (nvptx.gpu.ptx < 20 or nvptx.gpu.sm < 20) @compileError("fma on f32 needs a newer PTX version or target");
    // The backend emits `@mulAdd` as `fma.rn` and, unlike assembly, can move it into a `selp`.
    if (rounding == .nearest_even and !modifiers.ftz and !modifiers.saturate) return @mulAdd(f32, a, b, c);
    return asm (switch (rounding) {
            .nearest_even => if (modifiers.ftz)
                if (modifiers.saturate)
                    "fma.rn.ftz.sat.f32 %[d], %[a], %[b], %[c];"
                else
                    "fma.rn.ftz.f32 %[d], %[a], %[b], %[c];"
            else if (modifiers.saturate)
                "fma.rn.sat.f32 %[d], %[a], %[b], %[c];"
            else
                "fma.rn.f32 %[d], %[a], %[b], %[c];",
            .toward_zero => if (modifiers.ftz)
                if (modifiers.saturate)
                    "fma.rz.ftz.sat.f32 %[d], %[a], %[b], %[c];"
                else
                    "fma.rz.ftz.f32 %[d], %[a], %[b], %[c];"
            else if (modifiers.saturate)
                "fma.rz.sat.f32 %[d], %[a], %[b], %[c];"
            else
                "fma.rz.f32 %[d], %[a], %[b], %[c];",
            .toward_negative_infinity => if (modifiers.ftz)
                if (modifiers.saturate)
                    "fma.rm.ftz.sat.f32 %[d], %[a], %[b], %[c];"
                else
                    "fma.rm.ftz.f32 %[d], %[a], %[b], %[c];"
            else if (modifiers.saturate)
                "fma.rm.sat.f32 %[d], %[a], %[b], %[c];"
            else
                "fma.rm.f32 %[d], %[a], %[b], %[c];",
            .toward_positive_infinity => if (modifiers.ftz)
                if (modifiers.saturate)
                    "fma.rp.ftz.sat.f32 %[d], %[a], %[b], %[c];"
                else
                    "fma.rp.ftz.f32 %[d], %[a], %[b], %[c];"
            else if (modifiers.saturate)
                "fma.rp.sat.f32 %[d], %[a], %[b], %[c];"
            else
                "fma.rp.f32 %[d], %[a], %[b], %[c];",
        }
        : [d] "=f" (-> f32),
        : [a] "f" (a),
          [b] "f" (b),
          [c] "f" (c),
    );
}

/// Emits `fma` (section 9.7.3.6). `T` is `f32` or `f64`. `fmaF32` takes the `f32` modifiers.
pub inline fn fma(comptime T: type, comptime rounding: Rounding, a: T, b: T, c: T) T {
    switch (T) {
        f32 => return fmaF32(rounding, .{}, a, b, c),
        f64 => {
            if (nvptx.gpu.ptx < 14 or nvptx.gpu.sm < 13) @compileError("fma on f64 needs a newer PTX version or target");
            if (rounding == .nearest_even) return @mulAdd(f64, a, b, c);
            return asm (switch (rounding) {
                    .nearest_even => "fma.rn.f64 %[d], %[a], %[b], %[c];",
                    .toward_zero => "fma.rz.f64 %[d], %[a], %[b], %[c];",
                    .toward_negative_infinity => "fma.rm.f64 %[d], %[a], %[b], %[c];",
                    .toward_positive_infinity => "fma.rp.f64 %[d], %[a], %[b], %[c];",
                }
                : [d] "=d" (-> f64),
                : [a] "d" (a),
                  [b] "d" (b),
                  [c] "d" (c),
            );
        },
        else => unsupported("fma", T),
    }
}

const f32x2: nvptx.Requirement = .{ .ptx = 86, .sm = .{ .base = 100 }, .threads = .thread };

fn f32x2Suffix(comptime rounding: Rounding, comptime ftz: bool) []const u8 {
    return "." ++ rounding.float() ++ (if (ftz) ".ftz" else "") ++ ".f32x2";
}

/// Emits `add.f32x2` (section 9.7.3.3). The instruction adds the two lanes in parallel.
pub inline fn addF32x2(
    comptime rounding: Rounding,
    comptime ftz: bool,
    a: @Vector(2, f32),
    b: @Vector(2, f32),
) @Vector(2, f32) {
    nvptx.require("add.f32x2", f32x2);
    return asm ("{\n.reg .b64 a, b, d;\nmov.b64 a, %[a];\nmov.b64 b, %[b];\n" ++
            "add" ++ f32x2Suffix(rounding, ftz) ++ " d, a, b;\nmov.b64 %[d], d;\n}"
        : [d] "=f" (-> @Vector(2, f32)),
        : [a] "f" (a),
          [b] "f" (b),
    );
}

/// Emits `mul.f32x2` (section 9.7.3.5). The instruction multiplies the two lanes in parallel.
pub inline fn mulF32x2(
    comptime rounding: Rounding,
    comptime ftz: bool,
    a: @Vector(2, f32),
    b: @Vector(2, f32),
) @Vector(2, f32) {
    nvptx.require("mul.f32x2", f32x2);
    return asm ("{\n.reg .b64 a, b, d;\nmov.b64 a, %[a];\nmov.b64 b, %[b];\n" ++
            "mul" ++ f32x2Suffix(rounding, ftz) ++ " d, a, b;\nmov.b64 %[d], d;\n}"
        : [d] "=f" (-> @Vector(2, f32)),
        : [a] "f" (a),
          [b] "f" (b),
    );
}

/// Emits `fma.f32x2` (section 9.7.3.6). The instruction computes `a * b + c` on the two lanes in
/// parallel.
pub inline fn fmaF32x2(
    comptime rounding: Rounding,
    comptime ftz: bool,
    a: @Vector(2, f32),
    b: @Vector(2, f32),
    c: @Vector(2, f32),
) @Vector(2, f32) {
    nvptx.require("fma.f32x2", f32x2);
    return asm ("{\n.reg .b64 a, b, c, d;\nmov.b64 a, %[a];\n" ++
            "mov.b64 b, %[b];\nmov.b64 c, %[c];\n" ++
            "fma" ++ f32x2Suffix(rounding, ftz) ++ " d, a, b, c;\nmov.b64 %[d], d;\n}"
        : [d] "=f" (-> @Vector(2, f32)),
        : [a] "f" (a),
          [b] "f" (b),
          [c] "f" (c),
    );
}

pub const MaxModifiers = struct {
    ftz: bool = false,
    nan: bool = false,
    abs: bool = false,
};

/// Emits `max.f32` with three inputs (section 9.7.3.12).
pub inline fn max3F32(comptime modifiers: MaxModifiers, a: f32, b: f32, c: f32) f32 {
    nvptx.require("max.f32 with three inputs", .{
        .ptx = 88,
        .sm = .{ .base = 100 },
        .threads = .thread,
    });
    const qualifiers = (if (modifiers.ftz) ".ftz" else "") ++ (if (modifiers.nan) ".NaN" else "") ++
        (if (modifiers.abs) ".abs" else "");
    return asm ("max" ++ qualifiers ++ ".f32 %[d], %[a], %[b], %[c];"
        : [d] "=f" (-> f32),
        : [a] "f" (a),
          [b] "f" (b),
          [c] "f" (c),
    );
}

/// Emits `div` on `f32` (section 9.7.3.8).
pub inline fn divF32(comptime rounding: Rounding, comptime ftz: bool, a: f32, b: f32) f32 {
    if (nvptx.gpu.ptx < 14 or nvptx.gpu.sm < 20) @compileError("div on f32 needs a newer PTX version or target");
    return asm (switch (rounding) {
            .nearest_even => if (ftz)
                "div.rn.ftz.f32 %[d], %[a], %[b];"
            else
                "div.rn.f32 %[d], %[a], %[b];",
            .toward_zero => if (ftz)
                "div.rz.ftz.f32 %[d], %[a], %[b];"
            else
                "div.rz.f32 %[d], %[a], %[b];",
            .toward_negative_infinity => if (ftz)
                "div.rm.ftz.f32 %[d], %[a], %[b];"
            else
                "div.rm.f32 %[d], %[a], %[b];",
            .toward_positive_infinity => if (ftz)
                "div.rp.ftz.f32 %[d], %[a], %[b];"
            else
                "div.rp.f32 %[d], %[a], %[b];",
        }
        : [d] "=f" (-> f32),
        : [a] "f" (a),
          [b] "f" (b),
    );
}

/// Emits `div` (section 9.7.3.8). `T` is `f32` or `f64`. `divF32` takes the `f32` modifiers.
pub inline fn div(comptime T: type, comptime rounding: Rounding, a: T, b: T) T {
    switch (T) {
        f32 => return divF32(rounding, false, a, b),
        f64 => {
            const directed = rounding == .toward_negative_infinity or rounding == .toward_positive_infinity;
            if (nvptx.gpu.ptx < 14 or nvptx.gpu.sm < (if (directed) 20 else 13)) @compileError("div on f64 needs a newer PTX version or target");
            return asm (switch (rounding) {
                    .nearest_even => "div.rn.f64 %[d], %[a], %[b];",
                    .toward_zero => "div.rz.f64 %[d], %[a], %[b];",
                    .toward_negative_infinity => "div.rm.f64 %[d], %[a], %[b];",
                    .toward_positive_infinity => "div.rp.f64 %[d], %[a], %[b];",
                }
                : [d] "=d" (-> f64),
                : [a] "d" (a),
                  [b] "d" (b),
            );
        },
        else => unsupported("div", T),
    }
}

/// Emits `sqrt` on `f32` (section 9.7.3.15).
pub inline fn sqrtF32(comptime rounding: Rounding, comptime ftz: bool, a: f32) f32 {
    if (nvptx.gpu.ptx < 20 or nvptx.gpu.sm < 20) @compileError("sqrt on f32 needs a newer PTX version or target");
    return asm (switch (rounding) {
            .nearest_even => if (ftz)
                "sqrt.rn.ftz.f32 %[d], %[a];"
            else
                "sqrt.rn.f32 %[d], %[a];",
            .toward_zero => if (ftz)
                "sqrt.rz.ftz.f32 %[d], %[a];"
            else
                "sqrt.rz.f32 %[d], %[a];",
            .toward_negative_infinity => if (ftz)
                "sqrt.rm.ftz.f32 %[d], %[a];"
            else
                "sqrt.rm.f32 %[d], %[a];",
            .toward_positive_infinity => if (ftz)
                "sqrt.rp.ftz.f32 %[d], %[a];"
            else
                "sqrt.rp.f32 %[d], %[a];",
        }
        : [d] "=f" (-> f32),
        : [a] "f" (a),
    );
}

/// Emits `sqrt` (section 9.7.3.15). `T` is `f32` or `f64`. `sqrtF32` takes the `f32` modifiers.
pub inline fn sqrt(comptime T: type, comptime rounding: Rounding, a: T) T {
    switch (T) {
        f32 => return sqrtF32(rounding, false, a),
        f64 => {
            const directed = rounding == .toward_negative_infinity or rounding == .toward_positive_infinity;
            if (nvptx.gpu.ptx < 14 or nvptx.gpu.sm < (if (directed) 20 else 13)) @compileError("sqrt on f64 needs a newer PTX version or target");
            return asm (switch (rounding) {
                    .nearest_even => "sqrt.rn.f64 %[d], %[a];",
                    .toward_zero => "sqrt.rz.f64 %[d], %[a];",
                    .toward_negative_infinity => "sqrt.rm.f64 %[d], %[a];",
                    .toward_positive_infinity => "sqrt.rp.f64 %[d], %[a];",
                }
                : [d] "=d" (-> f64),
                : [a] "d" (a),
            );
        },
        else => unsupported("sqrt", T),
    }
}

/// Emits `rcp` on `f32` (section 9.7.3.13).
pub inline fn rcpF32(comptime rounding: Rounding, comptime ftz: bool, a: f32) f32 {
    if (nvptx.gpu.ptx < 20 or nvptx.gpu.sm < 20) @compileError("rcp on f32 needs a newer PTX version or target");
    return asm (switch (rounding) {
            .nearest_even => if (ftz)
                "rcp.rn.ftz.f32 %[d], %[a];"
            else
                "rcp.rn.f32 %[d], %[a];",
            .toward_zero => if (ftz)
                "rcp.rz.ftz.f32 %[d], %[a];"
            else
                "rcp.rz.f32 %[d], %[a];",
            .toward_negative_infinity => if (ftz)
                "rcp.rm.ftz.f32 %[d], %[a];"
            else
                "rcp.rm.f32 %[d], %[a];",
            .toward_positive_infinity => if (ftz)
                "rcp.rp.ftz.f32 %[d], %[a];"
            else
                "rcp.rp.f32 %[d], %[a];",
        }
        : [d] "=f" (-> f32),
        : [a] "f" (a),
    );
}

/// Emits `rcp` (section 9.7.3.13). `T` is `f32` or `f64`. `rcpF32` takes the `f32` modifiers.
pub inline fn rcp(comptime T: type, comptime rounding: Rounding, a: T) T {
    switch (T) {
        f32 => return rcpF32(rounding, false, a),
        f64 => {
            const directed = rounding == .toward_negative_infinity or rounding == .toward_positive_infinity;
            if (nvptx.gpu.ptx < 14 or nvptx.gpu.sm < (if (directed) 20 else 13)) @compileError("rcp on f64 needs a newer PTX version or target");
            return asm (switch (rounding) {
                    .nearest_even => "rcp.rn.f64 %[d], %[a];",
                    .toward_zero => "rcp.rz.f64 %[d], %[a];",
                    .toward_negative_infinity => "rcp.rm.f64 %[d], %[a];",
                    .toward_positive_infinity => "rcp.rp.f64 %[d], %[a];",
                }
                : [d] "=d" (-> f64),
                : [a] "d" (a),
            );
        },
        else => unsupported("rcp", T),
    }
}

/// Emits `div.approx` on `f32` (section 9.7.3.8). `nvptx.divApprox` is the form without `ftz`.
pub inline fn divApproxF32(comptime ftz: bool, a: f32, b: f32) f32 {
    nvptx.require("div.approx", approximate);
    return binaryF32(if (ftz) "div.approx.ftz.f32 %[d], %[a], %[b];" else "div.approx.f32 %[d], %[a], %[b];", a, b);
}

/// Emits `div.full` on `f32` (section 9.7.3.8).
pub inline fn divFullF32(comptime ftz: bool, a: f32, b: f32) f32 {
    nvptx.require("div.full", approximate);
    return binaryF32(if (ftz) "div.full.ftz.f32 %[d], %[a], %[b];" else "div.full.f32 %[d], %[a], %[b];", a, b);
}

/// Emits `rcp.approx` on `f32` (section 9.7.3.13). `nvptx.rcpApprox` is the form without `ftz`.
pub inline fn rcpApproxF32(comptime ftz: bool, x: f32) f32 {
    nvptx.require("rcp.approx", approximate);
    return unaryF32(if (ftz) "rcp.approx.ftz.f32 %[d], %[a];" else "rcp.approx.f32 %[d], %[a];", x);
}

/// Emits `rcp.approx.ftz.f64` (section 9.7.3.14). The instruction ignores the lower 32 bits of
/// `x` and sets the lower 32 bits of the result to zero.
pub inline fn rcpApproxFtzF64(x: f64) f64 {
    nvptx.require("rcp.approx.ftz.f64", .{ .ptx = 21, .sm = .{ .base = 20 }, .threads = .thread });
    return unaryF64("rcp.approx.ftz.f64 %[d], %[a];", x);
}

/// Emits `sqrt.approx` on `f32` (section 9.7.3.15).
pub inline fn sqrtApproxF32(comptime ftz: bool, x: f32) f32 {
    nvptx.require("sqrt.approx", approximate);
    return unaryF32(if (ftz) "sqrt.approx.ftz.f32 %[d], %[a];" else "sqrt.approx.f32 %[d], %[a];", x);
}

/// Emits `rsqrt.approx` on `f32` (section 9.7.3.16). `nvptx.rsqrtApprox` is the form without
/// `ftz`.
pub inline fn rsqrtApproxF32(comptime ftz: bool, x: f32) f32 {
    nvptx.require("rsqrt.approx", approximate);
    return unaryF32(if (ftz) "rsqrt.approx.ftz.f32 %[d], %[a];" else "rsqrt.approx.f32 %[d], %[a];", x);
}

/// Emits `rsqrt.approx.f64` (section 9.7.3.16).
pub inline fn rsqrtApproxF64(x: f64) f64 {
    nvptx.require("rsqrt.approx.f64", .{ .ptx = 14, .sm = .{ .base = 13 }, .threads = .thread });
    return unaryF64("rsqrt.approx.f64 %[d], %[a];", x);
}

/// Emits `rsqrt.approx.ftz.f64` (section 9.7.3.17). The instruction ignores the lower 32 bits of
/// `x` and sets the lower 32 bits of the result to zero.
pub inline fn rsqrtApproxFtzF64(x: f64) f64 {
    nvptx.require("rsqrt.approx.ftz.f64", .{ .ptx = 40, .sm = .{ .base = 20 }, .threads = .thread });
    return unaryF64("rsqrt.approx.ftz.f64 %[d], %[a];", x);
}

/// Emits `sin.approx` on `f32` (section 9.7.3.18). `nvptx.sinApprox` is the form without `ftz`.
pub inline fn sinApproxF32(comptime ftz: bool, x: f32) f32 {
    nvptx.require("sin.approx", approximate);
    return unaryF32(if (ftz) "sin.approx.ftz.f32 %[d], %[a];" else "sin.approx.f32 %[d], %[a];", x);
}

/// Emits `cos.approx` on `f32` (section 9.7.3.19). `nvptx.cosApprox` is the form without `ftz`.
pub inline fn cosApproxF32(comptime ftz: bool, x: f32) f32 {
    nvptx.require("cos.approx", approximate);
    return unaryF32(if (ftz) "cos.approx.ftz.f32 %[d], %[a];" else "cos.approx.f32 %[d], %[a];", x);
}

/// Emits `lg2.approx` on `f32` (section 9.7.3.20). `nvptx.lg2Approx` is the form without `ftz`.
pub inline fn lg2ApproxF32(comptime ftz: bool, x: f32) f32 {
    nvptx.require("lg2.approx", approximate);
    return unaryF32(if (ftz) "lg2.approx.ftz.f32 %[d], %[a];" else "lg2.approx.f32 %[d], %[a];", x);
}

/// Emits `ex2.approx` on `f32` (section 9.7.3.21). `nvptx.ex2Approx` is the form without `ftz`.
pub inline fn ex2ApproxF32(comptime ftz: bool, x: f32) f32 {
    nvptx.require("ex2.approx", approximate);
    return unaryF32(if (ftz) "ex2.approx.ftz.f32 %[d], %[a];" else "ex2.approx.f32 %[d], %[a];", x);
}

/// Emits `tanh.approx` (sections 9.7.3.22 and 9.7.4.9). `T` is `f32`, `f16` or `@Vector(2, f16)`.
pub inline fn tanhApprox(comptime T: type, x: T) T {
    nvptx.require("tanh.approx", .{ .ptx = 70, .sm = .{ .base = 75 }, .threads = .thread });
    return switch (T) {
        f32 => unaryF32("tanh.approx.f32 %[d], %[a];", x),
        f16 => @bitCast(unary16("tanh.approx.f16 %[d], %[a];", @bitCast(x))),
        @Vector(2, f16) => @bitCast(unary32("tanh.approx.f16x2 %[d], %[a];", @bitCast(x))),
        else => unsupported("tanhApprox", T),
    };
}

/// Emits `ex2.approx` (section 9.7.4.10). `T` is `f16`, `@Vector(2, f16)` or `nvptx.Bf16x2`.
/// The form for `nvptx.Bf16x2` is `ex2.approx.ftz.bf16x2`, which flushes subnormal inputs and
/// results to sign-preserving zero. `ex2ApproxF32` covers `f32`.
pub inline fn ex2Approx(comptime T: type, x: T) T {
    switch (T) {
        nvptx.Bf16x2 => nvptx.require("ex2.approx.ftz.bf16x2", bf16_ex2),
        else => nvptx.require("ex2.approx", .{ .ptx = 70, .sm = .{ .base = 75 }, .threads = .thread }),
    }
    return switch (T) {
        f16 => @bitCast(unary16("ex2.approx.f16 %[d], %[a];", @bitCast(x))),
        @Vector(2, f16) => @bitCast(unary32("ex2.approx.f16x2 %[d], %[a];", @bitCast(x))),
        nvptx.Bf16x2 => @fromBackingInt(unary32("ex2.approx.ftz.bf16x2 %[d], %[a];", @backingInt(x))),
        else => unsupported("ex2Approx", T),
    };
}

/// Emits `ex2.approx.ftz.bf16` (section 9.7.4.10) on the bit pattern of a `bf16`. The
/// instruction flushes subnormal inputs and results to sign-preserving zero.
pub inline fn ex2ApproxBf16(x: u16) u16 {
    nvptx.require("ex2.approx.ftz.bf16", bf16_ex2);
    return unary16("ex2.approx.ftz.bf16 %[d], %[a];", x);
}

/// Emits `fma.rn.relu` (section 9.7.4.4) on `@Vector(2, f16)`.
pub inline fn fmaRelu(a: @Vector(2, f16), b: @Vector(2, f16), c: @Vector(2, f16)) @Vector(2, f16) {
    nvptx.require("fma.rn.relu", .{ .ptx = 70, .sm = .{ .base = 80 }, .threads = .thread });
    return @bitCast(ternary32("fma.rn.relu.f16x2 %[d], %[a], %[b], %[c];", @bitCast(a), @bitCast(b), @bitCast(c)));
}

inline fn packF32(comptime text: []const u8, x0: f32, x1: f32) u32 {
    return asm (text
        : [d] "=r" (-> u32),
        : [lo] "f" (x0),
          [hi] "f" (x1),
    );
}

/// Emits `cvt.rn.f16x2.f32` (section 9.7.10.24). Element 0 of the result converts `x0`.
pub inline fn f16x2FromF32(x0: f32, x1: f32) @Vector(2, f16) {
    nvptx.require("cvt.rn.f16x2.f32", .{ .ptx = 70, .sm = .{ .base = 80 }, .threads = .thread });
    return @bitCast(packF32("cvt.rn.f16x2.f32 %[d], %[hi], %[lo];", x0, x1));
}

/// Emits `cvt.rn.relu.f16x2.f32` (section 9.7.10.24). Element 0 of the result converts `x0`.
pub inline fn f16x2FromF32Relu(x0: f32, x1: f32) @Vector(2, f16) {
    nvptx.require("cvt.rn.relu.f16x2.f32", .{ .ptx = 70, .sm = .{ .base = 80 }, .threads = .thread });
    return @bitCast(packF32("cvt.rn.relu.f16x2.f32 %[d], %[hi], %[lo];", x0, x1));
}

/// Emits `mul24.lo` (section 9.7.1.6). `T` is `i32` or `u32`.
pub inline fn mul24Lo(comptime T: type, a: T, b: T) T {
    nvptx.require("mul24.lo", always);
    return @bitCast(switch (T) {
        i32 => binary32("mul24.lo.s32 %[d], %[a], %[b];", @bitCast(a), @bitCast(b)),
        u32 => binary32("mul24.lo.u32 %[d], %[a], %[b];", a, b),
        else => unsupported("mul24Lo", T),
    });
}

/// Emits `mul24.hi` (section 9.7.1.6), which returns bits 47 to 16 of the 48-bit product. `T` is
/// `i32` or `u32`.
pub inline fn mul24Hi(comptime T: type, a: T, b: T) T {
    nvptx.require("mul24.hi", always);
    return @bitCast(switch (T) {
        i32 => binary32("mul24.hi.s32 %[d], %[a], %[b];", @bitCast(a), @bitCast(b)),
        u32 => binary32("mul24.hi.u32 %[d], %[a], %[b];", a, b),
        else => unsupported("mul24Hi", T),
    });
}

/// Emits `mad24.lo` (section 9.7.1.7). `T` is `i32` or `u32`.
pub inline fn mad24Lo(comptime T: type, a: T, b: T, c: T) T {
    nvptx.require("mad24.lo", always);
    return @bitCast(switch (T) {
        i32 => ternary32("mad24.lo.s32 %[d], %[a], %[b], %[c];", @bitCast(a), @bitCast(b), @bitCast(c)),
        u32 => ternary32("mad24.lo.u32 %[d], %[a], %[b], %[c];", a, b, c),
        else => unsupported("mad24Lo", T),
    });
}

/// Emits `mad24.hi` (section 9.7.1.7). `T` is `i32` or `u32`. Only `i32` supports `saturate`.
pub inline fn mad24Hi(comptime T: type, comptime saturate: bool, a: T, b: T, c: T) T {
    nvptx.require("mad24.hi", always);
    return @bitCast(switch (T) {
        i32 => ternary32(
            if (saturate) "mad24.hi.sat.s32 %[d], %[a], %[b], %[c];" else "mad24.hi.s32 %[d], %[a], %[b], %[c];",
            @bitCast(a),
            @bitCast(b),
            @bitCast(c),
        ),
        u32 => if (saturate)
            @compileError("mad24.hi.sat needs i32")
        else
            ternary32("mad24.hi.u32 %[d], %[a], %[b], %[c];", a, b, c),
        else => unsupported("mad24Hi", T),
    });
}

inline fn binary16(comptime text: []const u8, a: u16, b: u16) u16 {
    return asm (text
        : [d] "=h" (-> u16),
        : [a] "h" (a),
          [b] "h" (b),
    );
}

inline fn binary64(comptime text: []const u8, a: u64, b: u64) u64 {
    return asm (text
        : [d] "=l" (-> u64),
        : [a] "l" (a),
          [b] "l" (b),
    );
}

inline fn binaryF32(comptime text: []const u8, a: f32, b: f32) f32 {
    return asm (text
        : [d] "=f" (-> f32),
        : [a] "f" (a),
          [b] "f" (b),
    );
}

inline fn binaryF64(comptime text: []const u8, a: f64, b: f64) f64 {
    return asm (text
        : [d] "=d" (-> f64),
        : [a] "d" (a),
          [b] "d" (b),
    );
}

/// Emits `mul.hi` (section 9.7.1.3), which returns the high half of the full product. `T` is a
/// signed or unsigned integer of 16, 32 or 64 bits.
pub inline fn mulHi(comptime T: type, a: T, b: T) T {
    nvptx.require("mul.hi", always);
    return @bitCast(switch (T) {
        i16 => binary16("mul.hi.s16 %[d], %[a], %[b];", @bitCast(a), @bitCast(b)),
        u16 => binary16("mul.hi.u16 %[d], %[a], %[b];", a, b),
        i32 => binary32("mul.hi.s32 %[d], %[a], %[b];", @bitCast(a), @bitCast(b)),
        u32 => binary32("mul.hi.u32 %[d], %[a], %[b];", a, b),
        i64 => binary64("mul.hi.s64 %[d], %[a], %[b];", @bitCast(a), @bitCast(b)),
        u64 => binary64("mul.hi.u64 %[d], %[a], %[b];", a, b),
        else => unsupported("mulHi", T),
    });
}

fn Wide(comptime T: type) type {
    return switch (T) {
        i16 => i32,
        u16 => u32,
        i32 => i64,
        u32 => u64,
        else => unsupported("mulWide", T),
    };
}

/// Emits `mul.wide` (section 9.7.1.3), which returns the full product of two 16-bit or 32-bit
/// integers. `T` is `i16`, `u16`, `i32` or `u32`.
pub inline fn mulWide(comptime T: type, a: T, b: T) Wide(T) {
    nvptx.require("mul.wide", always);
    return switch (T) {
        i16 => @bitCast(asm ("mul.wide.s16 %[d], %[a], %[b];"
            : [d] "=r" (-> u32),
            : [a] "h" (@as(u16, @bitCast(a))),
              [b] "h" (@as(u16, @bitCast(b))),
        )),
        u16 => asm ("mul.wide.u16 %[d], %[a], %[b];"
            : [d] "=r" (-> u32),
            : [a] "h" (a),
              [b] "h" (b),
        ),
        i32 => @bitCast(asm ("mul.wide.s32 %[d], %[a], %[b];"
            : [d] "=l" (-> u64),
            : [a] "r" (@as(u32, @bitCast(a))),
              [b] "r" (@as(u32, @bitCast(b))),
        )),
        u32 => asm ("mul.wide.u32 %[d], %[a], %[b];"
            : [d] "=l" (-> u64),
            : [a] "r" (a),
              [b] "r" (b),
        ),
        else => unsupported("mulWide", T),
    };
}

/// Emits `sad` (section 9.7.1.8), which returns `|a - b| + c`. `T` is a signed or unsigned
/// integer of 16, 32 or 64 bits.
pub inline fn sad(comptime T: type, a: T, b: T, c: T) T {
    nvptx.require("sad", always);
    return @bitCast(switch (T) {
        i16 => ternary16("sad.s16 %[d], %[a], %[b], %[c];", @bitCast(a), @bitCast(b), @bitCast(c)),
        u16 => ternary16("sad.u16 %[d], %[a], %[b], %[c];", a, b, c),
        i32 => ternary32("sad.s32 %[d], %[a], %[b], %[c];", @bitCast(a), @bitCast(b), @bitCast(c)),
        u32 => ternary32("sad.u32 %[d], %[a], %[b], %[c];", a, b, c),
        i64 => ternary64("sad.s64 %[d], %[a], %[b], %[c];", @bitCast(a), @bitCast(b), @bitCast(c)),
        u64 => ternary64("sad.u64 %[d], %[a], %[b], %[c];", a, b, c),
        else => unsupported("sad", T),
    });
}

/// Emits `bfe` (section 9.7.1.20), which extracts `len` bits of `value` starting at bit `pos`.
/// The function uses the low 8 bits of `pos` and `len`. `T` is `i32`, `u32`, `i64` or `u64`.
pub inline fn bfe(comptime T: type, value: T, pos: u32, len: u32) T {
    nvptx.require("bfe", sm_20);
    return switch (T) {
        i32 => @bitCast(asm ("bfe.s32 %[d], %[a], %[pos], %[len];"
            : [d] "=r" (-> u32),
            : [a] "r" (@as(u32, @bitCast(value))),
              [pos] "r" (pos & 0xFF),
              [len] "r" (len & 0xFF),
        )),
        u32 => asm ("bfe.u32 %[d], %[a], %[pos], %[len];"
            : [d] "=r" (-> u32),
            : [a] "r" (value),
              [pos] "r" (pos & 0xFF),
              [len] "r" (len & 0xFF),
        ),
        i64 => @bitCast(asm ("bfe.s64 %[d], %[a], %[pos], %[len];"
            : [d] "=l" (-> u64),
            : [a] "l" (@as(u64, @bitCast(value))),
              [pos] "r" (pos & 0xFF),
              [len] "r" (len & 0xFF),
        )),
        u64 => asm ("bfe.u64 %[d], %[a], %[pos], %[len];"
            : [d] "=l" (-> u64),
            : [a] "l" (value),
              [pos] "r" (pos & 0xFF),
              [len] "r" (len & 0xFF),
        ),
        else => unsupported("bfe", T),
    };
}

/// Emits `bfi` (section 9.7.1.21), which replaces `len` bits of `base` at `pos` with the low bits
/// of `insert`. The function uses the low 8 bits of `pos` and `len`. `T` is `u32` or `u64`.
pub inline fn bfi(comptime T: type, insert: T, base: T, pos: u32, len: u32) T {
    nvptx.require("bfi", sm_20);
    return switch (T) {
        u32 => asm ("bfi.b32 %[d], %[a], %[b], %[pos], %[len];"
            : [d] "=r" (-> u32),
            : [a] "r" (insert),
              [b] "r" (base),
              [pos] "r" (pos & 0xFF),
              [len] "r" (len & 0xFF),
        ),
        u64 => asm ("bfi.b64 %[d], %[a], %[b], %[pos], %[len];"
            : [d] "=l" (-> u64),
            : [a] "l" (insert),
              [b] "l" (base),
              [pos] "r" (pos & 0xFF),
              [len] "r" (len & 0xFF),
        ),
        else => unsupported("bfi", T),
    };
}

pub const PermuteMode = enum {
    forward_4_extract,
    backward_4_extract,
    replicate_8,
    edge_clamp_left,
    edge_clamp_right,
    replicate_16,
};

/// Emits `prmt.b32` (section 9.7.10.7). Without `mode`, each of the four nibbles of `selector` picks a byte of
/// `{ b, a }` for the result.
pub inline fn prmt(comptime mode: ?PermuteMode, a: u32, b: u32, selector: u32) u32 {
    nvptx.require("prmt", sm_20);
    const text = if (mode) |m| switch (m) {
        .forward_4_extract => "prmt.b32.f4e %[d], %[a], %[b], %[c];",
        .backward_4_extract => "prmt.b32.b4e %[d], %[a], %[b], %[c];",
        .replicate_8 => "prmt.b32.rc8 %[d], %[a], %[b], %[c];",
        .edge_clamp_left => "prmt.b32.ecl %[d], %[a], %[b], %[c];",
        .edge_clamp_right => "prmt.b32.ecr %[d], %[a], %[b], %[c];",
        .replicate_16 => "prmt.b32.rc16 %[d], %[a], %[b], %[c];",
    } else "prmt.b32 %[d], %[a], %[b], %[c];";
    return ternary32(text, a, b, selector);
}

pub const ShiftMode = enum { wrap, clamp };

/// Returns the table that `lop3` expects for `operation`. PTX defines it as the result of the
/// operation on a = 0xF0, b = 0xCC and c = 0xAA.
pub fn lop3Table(comptime operation: fn (u8, u8, u8) u8) u8 {
    return operation(0xF0, 0xCC, 0xAA);
}

/// Emits `lop3.b32` (section 9.7.9.6). Every result bit is the bit of `table` that the three input
/// bits index.
pub inline fn lop3(comptime table: u8, a: u32, b: u32, c: u32) u32 {
    nvptx.require("lop3", .{ .ptx = 43, .sm = .{ .base = 50 }, .threads = .thread });
    return asm ("lop3.b32 %[d], %[a], %[b], %[c], %[table];"
        : [d] "=r" (-> u32),
        : [a] "r" (a),
          [b] "r" (b),
          [c] "r" (c),
          [table] "n" (table),
    );
}

pub const FunnelShift = enum { left, right };

/// Emits `shf.l` or `shf.r` (section 9.7.9.7). The funnel shift treats `{ high, low }` as one
/// 64-bit value. A left shift returns the high 32 bits of the shifted value, a right shift the low
/// 32 bits.
pub inline fn shf(
    comptime direction: FunnelShift,
    comptime mode: ShiftMode,
    low: u32,
    high: u32,
    amount: u32,
) u32 {
    nvptx.require("shf", .{ .ptx = 31, .sm = .{ .base = 32 }, .threads = .thread });
    const text = switch (direction) {
        .left => switch (mode) {
            .wrap => "shf.l.wrap.b32 %[d], %[a], %[b], %[c];",
            .clamp => "shf.l.clamp.b32 %[d], %[a], %[b], %[c];",
        },
        .right => switch (mode) {
            .wrap => "shf.r.wrap.b32 %[d], %[a], %[b], %[c];",
            .clamp => "shf.r.clamp.b32 %[d], %[a], %[b], %[c];",
        },
    };
    return ternary32(text, low, high, amount);
}

/// The PTX type of the operand of `dp4a` and `dp2a`: `.u32` or `.s32`.
pub const DotElement = enum { u32, s32 };

/// Emits `dp4a` (section 9.7.1.24). The instruction splits `a` and `b` into four bytes each, sums the products of
/// the byte pairs and adds `c`.
pub inline fn dp4a(
    comptime a_type: DotElement,
    comptime b_type: DotElement,
    a: u32,
    b: u32,
    c: (if (a_type == .u32 and b_type == .u32) u32 else i32),
) (if (a_type == .u32 and b_type == .u32) u32 else i32) {
    nvptx.require("dp4a", .{ .ptx = 50, .sm = .{ .base = 61 }, .threads = .thread });
    const text = switch (a_type) {
        .u32 => switch (b_type) {
            .u32 => "dp4a.u32.u32 %[d], %[a], %[b], %[c];",
            .s32 => "dp4a.u32.s32 %[d], %[a], %[b], %[c];",
        },
        .s32 => switch (b_type) {
            .u32 => "dp4a.s32.u32 %[d], %[a], %[b], %[c];",
            .s32 => "dp4a.s32.s32 %[d], %[a], %[b], %[c];",
        },
    };
    return @bitCast(ternary32(text, a, b, @bitCast(c)));
}

pub const HalfSelector = enum { lo, hi };

/// Emits `dp2a` (section 9.7.1.25). The instruction splits `a` into two 16-bit elements, takes two bytes of `b`
/// (the low two for `.lo`, the high two for `.hi`), sums the two products and adds `c`.
pub inline fn dp2a(
    comptime half: HalfSelector,
    comptime a_type: DotElement,
    comptime b_type: DotElement,
    a: u32,
    b: u32,
    c: (if (a_type == .u32 and b_type == .u32) u32 else i32),
) (if (a_type == .u32 and b_type == .u32) u32 else i32) {
    nvptx.require("dp2a", .{ .ptx = 50, .sm = .{ .base = 61 }, .threads = .thread });
    const text = switch (half) {
        .lo => switch (a_type) {
            .u32 => switch (b_type) {
                .u32 => "dp2a.lo.u32.u32 %[d], %[a], %[b], %[c];",
                .s32 => "dp2a.lo.u32.s32 %[d], %[a], %[b], %[c];",
            },
            .s32 => switch (b_type) {
                .u32 => "dp2a.lo.s32.u32 %[d], %[a], %[b], %[c];",
                .s32 => "dp2a.lo.s32.s32 %[d], %[a], %[b], %[c];",
            },
        },
        .hi => switch (a_type) {
            .u32 => switch (b_type) {
                .u32 => "dp2a.hi.u32.u32 %[d], %[a], %[b], %[c];",
                .s32 => "dp2a.hi.u32.s32 %[d], %[a], %[b], %[c];",
            },
            .s32 => switch (b_type) {
                .u32 => "dp2a.hi.s32.u32 %[d], %[a], %[b], %[c];",
                .s32 => "dp2a.hi.s32.s32 %[d], %[a], %[b], %[c];",
            },
        },
    };
    return @bitCast(ternary32(text, a, b, @bitCast(c)));
}

/// Emits `fns.b32` (section 9.7.1.18). A nonzero `offset` selects the `|offset|`-th set bit of `mask`,
/// counted from bit `base` (included) upward for a positive and downward for a negative `offset`.
/// A zero `offset` selects `base` when that bit is set. The result is `0xffffffff` when no bit
/// qualifies.
pub inline fn fns(mask: u32, base: u32, offset: i32) u32 {
    nvptx.require("fns", .{ .ptx = 60, .sm = .{ .base = 30 }, .threads = .thread });
    return ternary32("fns.b32 %[d], %[a], %[b], %[c];", mask, base, @bitCast(offset));
}

/// Emits `bmsk.b32` (section 9.7.1.23), which sets `len` bits of the result starting at bit `start`.
pub inline fn bmsk(comptime mode: ShiftMode, start: u32, len: u32) u32 {
    nvptx.require("bmsk", .{ .ptx = 76, .sm = .{ .base = 70 }, .threads = .thread });
    return binary32(switch (mode) {
        .wrap => "bmsk.wrap.b32 %[d], %[a], %[b];",
        .clamp => "bmsk.clamp.b32 %[d], %[a], %[b];",
    }, start, len);
}

/// Emits `szext` (section 9.7.1.22), which sign-extends or zero-extends the low `width` bits of
/// `value`. `T` is `i32` or `u32`.
pub inline fn szext(comptime mode: ShiftMode, comptime T: type, value: T, width: u32) T {
    nvptx.require("szext", .{ .ptx = 76, .sm = .{ .base = 70 }, .threads = .thread });
    const text = switch (mode) {
        .wrap => switch (T) {
            i32 => "szext.wrap.s32 %[d], %[a], %[b];",
            u32 => "szext.wrap.u32 %[d], %[a], %[b];",
            else => unsupported("szext", T),
        },
        .clamp => switch (T) {
            i32 => "szext.clamp.s32 %[d], %[a], %[b];",
            u32 => "szext.clamp.u32 %[d], %[a], %[b];",
            else => unsupported("szext", T),
        },
    };
    return @bitCast(binary32(text, @bitCast(value), width));
}

pub const FloatProperty = enum { finite, infinite, number, notanumber, normal, subnormal };

/// Emits `testp` (section 9.7.3.1). `T` is `f32` or `f64`.
pub inline fn testp(comptime property: FloatProperty, comptime T: type, x: T) bool {
    nvptx.require("testp", sm_20);
    return switch (T) {
        f32 => asm (switch (property) {
                .finite => "testp.finite.f32 %[d], %[a];",
                .infinite => "testp.infinite.f32 %[d], %[a];",
                .number => "testp.number.f32 %[d], %[a];",
                .notanumber => "testp.notanumber.f32 %[d], %[a];",
                .normal => "testp.normal.f32 %[d], %[a];",
                .subnormal => "testp.subnormal.f32 %[d], %[a];",
            }
            : [d] "=b" (-> bool),
            : [a] "f" (x),
        ),
        f64 => asm (switch (property) {
                .finite => "testp.finite.f64 %[d], %[a];",
                .infinite => "testp.infinite.f64 %[d], %[a];",
                .number => "testp.number.f64 %[d], %[a];",
                .notanumber => "testp.notanumber.f64 %[d], %[a];",
                .normal => "testp.normal.f64 %[d], %[a];",
                .subnormal => "testp.subnormal.f64 %[d], %[a];",
            }
            : [d] "=b" (-> bool),
            : [a] "d" (x),
        ),
        else => unsupported("testp", T),
    };
}

/// Emits `copysign` (section 9.7.3.2), which returns `magnitude` with the sign of `sign`. `T` is
/// `f32` or `f64`.
pub inline fn copysign(comptime T: type, sign: T, magnitude: T) T {
    nvptx.require("copysign", sm_20);
    return switch (T) {
        f32 => binaryF32("copysign.f32 %[d], %[a], %[b];", sign, magnitude),
        f64 => binaryF64("copysign.f64 %[d], %[a], %[b];", sign, magnitude),
        else => unsupported("copysign", T),
    };
}

test lop3Table {
    const operations = struct {
        fn xor3(a: u8, b: u8, c: u8) u8 {
            return a ^ b ^ c;
        }
        fn majority(a: u8, b: u8, c: u8) u8 {
            return (a & b) | (a & c) | (b & c);
        }
        fn select(a: u8, b: u8, c: u8) u8 {
            return (a & b) | (~a & c);
        }
    };
    try std.testing.expectEqual(0x96, comptime lop3Table(operations.xor3));
    try std.testing.expectEqual(0xE8, comptime lop3Table(operations.majority));
    try std.testing.expectEqual(0xCA, comptime lop3Table(operations.select));
}

test roundingModifier {
    try std.testing.expectEqualStrings(".rni", comptime roundingModifier(f32, i32, .nearest_even));
    try std.testing.expectEqualStrings(".rm", comptime roundingModifier(i64, f64, .toward_negative_infinity));
    try std.testing.expectEqualStrings(".rzi", comptime roundingModifier(f32, f32, .toward_zero));
    try std.testing.expectEqualStrings(".rp", comptime roundingModifier(f64, f32, .toward_positive_infinity));
    try std.testing.expectEqualStrings("", comptime roundingModifier(f16, f32, null));
    try std.testing.expectEqualStrings("", comptime roundingModifier(f32, f32, null));
    try std.testing.expectEqualStrings("", comptime roundingModifier(i32, i16, null));
}

test typeName {
    try std.testing.expectEqualStrings("s32", comptime typeName(i32));
    try std.testing.expectEqualStrings("u64", comptime typeName(u64));
    try std.testing.expectEqualStrings("f16", comptime typeName(f16));
}
