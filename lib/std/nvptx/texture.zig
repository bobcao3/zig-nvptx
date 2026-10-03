//! Texture and surface instructions in unified mode: the texture or surface is a 64-bit handle,
//! the same value as CUDA's `cudaTextureObject_t` or `cudaSurfaceObject_t`. A driver call
//! creates a handle on the host.

const std = @import("../std.zig");
const nvptx = std.nvptx;

pub const Geometry = enum { @"1d", @"2d", @"3d", a1d, a2d, cube, acube };

/// What a surface does with a coordinate outside the surface. `.trap` raises an error, `.clamp`
/// uses the nearest element inside the surface, and `.zero` makes a load return zero and a store
/// do nothing.
pub const Clamp = enum { trap, clamp, zero };

pub const Query = enum {
    width,
    height,
    depth,
    channel_data_type,
    channel_order,
    normalized_coords,
    array_size,
    num_mipmap_levels,
    num_samples,
};

pub const LevelQuery = enum { width, height, depth };

/// The channel that `tld4` gathers from the four texels around the coordinate.
pub const GatherComponent = enum { r, g, b, a };

pub const ReductionOperation = enum { add, min, max, @"and", @"or" };

const position_lens = [_]usize{ 1, 2, 3, 1, 2, 3, 3 };

/// The number of floating-point or integer coordinates that locate a point, without the array
/// index.
pub fn positionLen(comptime geometry: Geometry) usize {
    return position_lens[@backingInt(geometry)];
}

/// For an array geometry, the coordinates are a struct with the array index and the position.
/// Otherwise they are the position.
pub fn Coordinates(comptime geometry: Geometry, comptime Element: type) type {
    return switch (geometry) {
        .@"1d", .@"2d", .@"3d", .cube => @Vector(position_lens[@backingInt(geometry)], Element),
        .a1d => struct { index: u32, position: @Vector(1, Element) },
        .a2d => struct { index: u32, position: @Vector(2, Element) },
        .acube => struct { index: u32, position: @Vector(3, Element) },
    };
}

inline fn fetch(
    comptime text: []const u8,
    comptime coordinate_len: usize,
    handle: u64,
    list: @Vector(coordinate_len, u32),
) @Vector(4, u32) {
    return asm (text
        : [d] "=r" (-> @Vector(4, u32)),
        : [t] "l" (handle),
          [c] "r" (list),
    );
}

inline fn fetchLevel(
    comptime text: []const u8,
    comptime coordinate_len: usize,
    handle: u64,
    list: @Vector(coordinate_len, u32),
    level_of_detail: f32,
) @Vector(4, u32) {
    return asm (text
        : [d] "=r" (-> @Vector(4, u32)),
        : [t] "l" (handle),
          [c] "r" (list),
          [lod] "f" (level_of_detail),
    );
}

inline fn fetchGrad(
    comptime text: []const u8,
    comptime coordinate_len: usize,
    comptime gradient_len: usize,
    handle: u64,
    list: @Vector(coordinate_len, u32),
    gradient_x: @Vector(gradient_len, u32),
    gradient_y: @Vector(gradient_len, u32),
) @Vector(4, u32) {
    return asm (text
        : [d] "=r" (-> @Vector(4, u32)),
        : [t] "l" (handle),
          [c] "r" (list),
          [dx] "r" (gradient_x),
          [dy] "r" (gradient_y),
    );
}

/// A 64-bit handle of a texture. The instructions below are the `tex`, `tld4` and `txq`
/// instructions of the PTX ISA, which take the handle in unified mode.
pub const TextureObject = enum(u64) {
    _,

    /// Emits `tex` (section 9.7.12.3) with integer or floating-point coordinates. Integer coordinates fetch one
    /// texel without filtering and need a geometry other than `.cube` and `.acube`.
    pub inline fn tex(
        texture: TextureObject,
        comptime geometry: Geometry,
        comptime Texel: type,
        comptime Coordinate: type,
        coordinates: Coordinates(geometry, Coordinate),
    ) @Vector(4, Texel) {
        nvptx.require("tex", .{
            .ptx = switch (geometry) {
                .@"1d", .@"2d", .@"3d" => 10,
                .a1d, .a2d => 23,
                .cube, .acube => 30,
            },
            .sm = .{ .base = if (geometry == .acube) 20 else 0 },
            .threads = .thread,
        });
        const coordinate_suffix = switch (Coordinate) {
            f32 => ".f32",
            i32 => if (geometry == .cube or geometry == .acube)
                @compileError("tex supports integer coordinates only for geometries other than " ++
                    "cube and acube")
            else
                ".s32",
            else => @compileError("tex coordinates are f32 or i32, found '" ++
                @typeName(Coordinate) ++ "'"),
        };
        const texel_suffix = switch (Texel) {
            f32 => ".f32",
            i32 => ".s32",
            u32 => ".u32",
            else => @compileError("a texture fetch returns f32, i32 or u32, found '" ++
                @typeName(Texel) ++ "'"),
        };
        const text = "tex." ++ @tagName(geometry) ++ ".v4" ++ texel_suffix ++
            coordinate_suffix ++ " %[d], [%[t], %[c]];";
        const handle = @backingInt(texture);
        const result = switch (geometry) {
            .@"1d" => fetch(text, 1, handle, @bitCast(coordinates)),
            .@"2d" => fetch(text, 2, handle, @bitCast(coordinates)),
            .@"3d", .cube => point: {
                const b: @Vector(3, u32) = @bitCast(coordinates);
                break :point fetch(text, 4, handle, .{ b[0], b[1], b[2], 0 });
            },
            .a1d => point: {
                const b: @Vector(1, u32) = @bitCast(coordinates.position);
                break :point fetch(text, 2, handle, .{ coordinates.index, b[0] });
            },
            .a2d => point: {
                const b: @Vector(2, u32) = @bitCast(coordinates.position);
                break :point fetch(text, 4, handle, .{ coordinates.index, b[0], b[1], 0 });
            },
            .acube => point: {
                const b: @Vector(3, u32) = @bitCast(coordinates.position);
                break :point fetch(text, 4, handle, .{ coordinates.index, b[0], b[1], b[2] });
            },
        };
        return @bitCast(result);
    }

    /// Emits `tex.level` (section 9.7.12.3), which samples the mipmap level `level_of_detail`.
    pub inline fn texLevel(
        texture: TextureObject,
        comptime geometry: Geometry,
        comptime Texel: type,
        coordinates: Coordinates(geometry, f32),
        level_of_detail: f32,
    ) @Vector(4, Texel) {
        nvptx.require("tex.level", .{ .ptx = 31, .sm = .{ .base = 20 }, .threads = .thread });
        const texel_suffix = switch (Texel) {
            f32 => ".f32",
            i32 => ".s32",
            u32 => ".u32",
            else => @compileError("a texture fetch returns f32, i32 or u32, found '" ++
                @typeName(Texel) ++ "'"),
        };
        const text = "tex.level." ++ @tagName(geometry) ++ ".v4" ++ texel_suffix ++
            ".f32 %[d], [%[t], %[c]], %[lod];";
        const handle = @backingInt(texture);
        const result = switch (geometry) {
            .@"1d" => fetchLevel(text, 1, handle, @bitCast(coordinates), level_of_detail),
            .@"2d" => fetchLevel(text, 2, handle, @bitCast(coordinates), level_of_detail),
            .@"3d", .cube => point: {
                const b: @Vector(3, u32) = @bitCast(coordinates);
                break :point fetchLevel(text, 4, handle, .{ b[0], b[1], b[2], 0 }, level_of_detail);
            },
            .a1d => point: {
                const b: @Vector(1, u32) = @bitCast(coordinates.position);
                break :point fetchLevel(
                    text,
                    2,
                    handle,
                    .{ coordinates.index, b[0] },
                    level_of_detail,
                );
            },
            .a2d => point: {
                const b: @Vector(2, u32) = @bitCast(coordinates.position);
                break :point fetchLevel(
                    text,
                    4,
                    handle,
                    .{ coordinates.index, b[0], b[1], 0 },
                    level_of_detail,
                );
            },
            .acube => point: {
                const b: @Vector(3, u32) = @bitCast(coordinates.position);
                break :point fetchLevel(
                    text,
                    4,
                    handle,
                    .{ coordinates.index, b[0], b[1], b[2] },
                    level_of_detail,
                );
            },
        };
        return @bitCast(result);
    }

    /// Emits `tex.grad` (section 9.7.12.3), which selects the mipmap level from the gradients of the position along
    /// the x and y axes of the screen.
    pub inline fn texGrad(
        texture: TextureObject,
        comptime geometry: Geometry,
        comptime Texel: type,
        coordinates: Coordinates(geometry, f32),
        gradient_x: @Vector(position_lens[@backingInt(geometry)], f32),
        gradient_y: @Vector(position_lens[@backingInt(geometry)], f32),
    ) @Vector(4, Texel) {
        nvptx.require("tex.grad", .{
            .ptx = if (geometry == .cube or geometry == .acube) 43 else 31,
            .sm = .{ .base = 20 },
            .threads = .thread,
        });
        const single = "%[dx], %[dy];";
        const texel_suffix = switch (Texel) {
            f32 => ".f32",
            i32 => ".s32",
            u32 => ".u32",
            else => @compileError("a texture fetch returns f32, i32 or u32, found '" ++
                @typeName(Texel) ++ "'"),
        };
        const text = "tex.grad." ++ @tagName(geometry) ++ ".v4" ++ texel_suffix ++
            ".f32 %[d], [%[t], %[c]], " ++ switch (geometry) {
            .@"1d", .a1d => "{%[dx]}, {%[dy]};",
            else => single,
        };
        const handle = @backingInt(texture);
        const result = switch (geometry) {
            .@"1d" => fetchGrad(
                text,
                1,
                1,
                handle,
                @bitCast(coordinates),
                @bitCast(gradient_x),
                @bitCast(gradient_y),
            ),
            .@"2d" => fetchGrad(
                text,
                2,
                2,
                handle,
                @bitCast(coordinates),
                @bitCast(gradient_x),
                @bitCast(gradient_y),
            ),
            .@"3d", .cube => point: {
                const b: @Vector(3, u32) = @bitCast(coordinates);
                const x: @Vector(3, u32) = @bitCast(gradient_x);
                const y: @Vector(3, u32) = @bitCast(gradient_y);
                break :point fetchGrad(
                    text,
                    4,
                    4,
                    handle,
                    .{ b[0], b[1], b[2], 0 },
                    .{ x[0], x[1], x[2], 0 },
                    .{ y[0], y[1], y[2], 0 },
                );
            },
            .a1d => point: {
                const b: @Vector(1, u32) = @bitCast(coordinates.position);
                break :point fetchGrad(
                    text,
                    2,
                    1,
                    handle,
                    .{ coordinates.index, b[0] },
                    @bitCast(gradient_x),
                    @bitCast(gradient_y),
                );
            },
            .a2d => point: {
                const b: @Vector(2, u32) = @bitCast(coordinates.position);
                break :point fetchGrad(
                    text,
                    4,
                    2,
                    handle,
                    .{ coordinates.index, b[0], b[1], 0 },
                    @bitCast(gradient_x),
                    @bitCast(gradient_y),
                );
            },
            .acube => point: {
                const b: @Vector(3, u32) = @bitCast(coordinates.position);
                const x: @Vector(3, u32) = @bitCast(gradient_x);
                const y: @Vector(3, u32) = @bitCast(gradient_y);
                break :point fetchGrad(
                    text,
                    4,
                    4,
                    handle,
                    .{ coordinates.index, b[0], b[1], b[2] },
                    .{ x[0], x[1], x[2], 0 },
                    .{ y[0], y[1], y[2], 0 },
                );
            },
        };
        return @bitCast(result);
    }

    /// Emits `tld4` (section 9.7.12.4), which returns the `component` of the four texels that bilinear filtering
    /// would combine.
    pub inline fn tld4(
        texture: TextureObject,
        comptime component: GatherComponent,
        comptime geometry: Geometry,
        comptime Texel: type,
        coordinates: Coordinates(geometry, f32),
    ) @Vector(4, Texel) {
        nvptx.require("tld4", .{
            .ptx = if (geometry == .@"2d") 22 else 43,
            .sm = .{ .base = if (geometry == .@"2d") 20 else 30 },
            .threads = .thread,
        });
        const texel_suffix = switch (Texel) {
            f32 => ".f32",
            i32 => ".s32",
            u32 => ".u32",
            else => @compileError("a texture fetch returns f32, i32 or u32, found '" ++
                @typeName(Texel) ++ "'"),
        };
        const text = "tld4." ++ @tagName(component) ++ "." ++ @tagName(geometry) ++ ".v4" ++
            texel_suffix ++ ".f32 %[d], [%[t], %[c]];";
        const handle = @backingInt(texture);
        const result = switch (geometry) {
            .@"2d" => fetch(text, 2, handle, @bitCast(coordinates)),
            .cube => point: {
                const b: @Vector(3, u32) = @bitCast(coordinates);
                break :point fetch(text, 4, handle, .{ b[0], b[1], b[2], 0 });
            },
            .a2d => point: {
                const b: @Vector(2, u32) = @bitCast(coordinates.position);
                break :point fetch(text, 4, handle, .{ coordinates.index, b[0], b[1], 0 });
            },
            .acube => point: {
                const b: @Vector(3, u32) = @bitCast(coordinates.position);
                break :point fetch(text, 4, handle, .{ coordinates.index, b[0], b[1], b[2] });
            },
            .@"1d", .@"3d", .a1d => {
                @compileError("tld4 supports the geometries 2d, a2d, cube and acube");
            },
        };
        return @bitCast(result);
    }

    /// Emits `txq` (section 9.7.12.5).
    pub inline fn txq(texture: TextureObject, comptime query: Query) u32 {
        nvptx.require("txq", .{
            .ptx = switch (query) {
                .width, .height, .depth, .normalized_coords => 15,
                .channel_data_type, .channel_order => 21,
                .array_size, .num_mipmap_levels, .num_samples => 41,
            },
            .sm = .{ .base = switch (query) {
                .num_mipmap_levels => 20,
                .num_samples => 30,
                else => 0,
            } },
            .threads = .thread,
        });
        return asm ("txq." ++ @tagName(query) ++ ".b32 %[d], [%[t]];"
            : [d] "=r" (-> u32),
            : [t] "l" (@backingInt(texture)),
        );
    }

    /// Emits `txq.level` (section 9.7.12.5), which queries one mipmap level.
    pub inline fn txqLevel(texture: TextureObject, comptime query: LevelQuery, level: u32) u32 {
        nvptx.require("txq.level", .{ .ptx = 43, .sm = .{ .base = 30 }, .threads = .thread });
        return asm ("txq.level." ++ @tagName(query) ++ ".b32 %[d], [%[t]], %[lod];"
            : [d] "=r" (-> u32),
            : [t] "l" (@backingInt(texture)),
              [lod] "r" (level),
        );
    }
};

inline fn load(
    comptime Bits: type,
    comptime lanes: usize,
    comptime text: []const u8,
    comptime coordinate_len: usize,
    handle: u64,
    list: @Vector(coordinate_len, u32),
) @Vector(lanes, Bits) {
    return switch (Bits) {
        u16 => asm volatile (text
            : [d] "=h" (-> @Vector(lanes, u16)),
            : [s] "l" (handle),
              [c] "r" (list),
            : .{ .memory = true }),
        u32 => asm volatile (text
            : [d] "=r" (-> @Vector(lanes, u32)),
            : [s] "l" (handle),
              [c] "r" (list),
            : .{ .memory = true }),
        u64 => asm volatile (text
            : [d] "=l" (-> @Vector(lanes, u64)),
            : [s] "l" (handle),
              [c] "r" (list),
            : .{ .memory = true }),
        else => unreachable,
    };
}

inline fn store(
    comptime Bits: type,
    comptime lanes: usize,
    comptime text: []const u8,
    comptime coordinate_len: usize,
    handle: u64,
    list: @Vector(coordinate_len, u32),
    bits: @Vector(lanes, Bits),
) void {
    switch (Bits) {
        u16 => asm volatile (text
            :
            : [s] "l" (handle),
              [c] "r" (list),
              [v] "h" (bits),
            : .{ .memory = true }),
        u32 => asm volatile (text
            :
            : [s] "l" (handle),
              [c] "r" (list),
              [v] "r" (bits),
            : .{ .memory = true }),
        u64 => asm volatile (text
            :
            : [s] "l" (handle),
              [c] "r" (list),
              [v] "l" (bits),
            : .{ .memory = true }),
        else => unreachable,
    }
}

inline fn reduce(
    comptime Bits: type,
    comptime text: []const u8,
    comptime coordinate_len: usize,
    handle: u64,
    list: @Vector(coordinate_len, u32),
    value: Bits,
) void {
    switch (Bits) {
        u32 => asm volatile (text
            :
            : [s] "l" (handle),
              [c] "r" (list),
              [v] "r" (value),
            : .{ .memory = true }),
        u64 => asm volatile (text
            :
            : [s] "l" (handle),
              [c] "r" (list),
              [v] "l" (value),
            : .{ .memory = true }),
        else => unreachable,
    }
}

/// A 64-bit handle of a surface. The x coordinate of every access is in bytes. The instructions
/// below are `suld.b`, `sust.b` and `sured.b` of the PTX ISA in unified mode.
pub const SurfaceObject = enum(u64) {
    _,

    /// Emits `suld.b` (section 9.7.13.1), which loads `lanes` elements of `Texel` without conversion.
    pub inline fn suldB(
        surface: SurfaceObject,
        comptime geometry: Geometry,
        comptime Texel: type,
        comptime lanes: usize,
        comptime clamp: Clamp,
        coordinates: Coordinates(geometry, i32),
    ) @Vector(lanes, Texel) {
        if (geometry == .cube or geometry == .acube) {
            @compileError("suld.b does not support the geometries cube and acube");
        }
        const needs_sm_20 = geometry == .@"3d" or geometry == .a1d or geometry == .a2d or
            clamp != .trap;
        nvptx.require("suld.b", .{
            .ptx = if (geometry == .@"3d" or geometry == .a1d or geometry == .a2d)
                30
            else if (clamp != .trap) 20 else 15,
            .sm = .{ .base = if (needs_sm_20) 20 else 0 },
            .threads = .thread,
        });
        const width_suffix = switch (@bitSizeOf(Texel)) {
            8 => ".b8",
            16 => ".b16",
            32 => ".b32",
            64 => ".b64",
            else => @compileError("a surface element has 8, 16, 32 or 64 bits, found '" ++
                @typeName(Texel) ++ "'"),
        };
        const vector_suffix = switch (lanes) {
            1 => "",
            2 => ".v2",
            4 => ".v4",
            else => @compileError("a surface access moves 1, 2 or 4 elements"),
        };
        if (lanes == 4 and @bitSizeOf(Texel) == 64) {
            @compileError("a surface access moves at most 128 bits");
        }
        const Bits = @Int(.unsigned, @max(16, @bitSizeOf(Texel)));
        const text = "suld.b." ++ @tagName(geometry) ++ vector_suffix ++
            width_suffix ++ "." ++ @tagName(clamp) ++ " %[d], [%[s], %[c]];";
        const handle = @backingInt(surface);
        const loaded = switch (geometry) {
            .@"1d" => load(Bits, lanes, text, 1, handle, @bitCast(coordinates)),
            .@"2d" => load(Bits, lanes, text, 2, handle, @bitCast(coordinates)),
            .@"3d" => load(Bits, lanes, text, 4, handle, .{
                @bitCast(coordinates[0]), @bitCast(coordinates[1]), @bitCast(coordinates[2]), 0,
            }),
            .a1d => load(Bits, lanes, text, 2, handle, .{
                coordinates.index, @bitCast(coordinates.position[0]),
            }),
            .a2d => load(Bits, lanes, text, 4, handle, .{
                coordinates.index,
                @bitCast(coordinates.position[0]),
                @bitCast(coordinates.position[1]),
                0,
            }),
            .cube, .acube => unreachable,
        };
        const Unsigned = @Int(.unsigned, @bitSizeOf(Texel));
        return @bitCast(@as(@Vector(lanes, Unsigned), @truncate(loaded)));
    }

    /// Emits `sust.b` (section 9.7.13.2), which stores the elements of the vector `texels` without conversion.
    pub inline fn sustB(
        surface: SurfaceObject,
        comptime geometry: Geometry,
        comptime clamp: Clamp,
        coordinates: Coordinates(geometry, i32),
        texels: anytype,
    ) void {
        if (geometry == .cube or geometry == .acube) {
            @compileError("sust.b does not support the geometries cube and acube");
        }
        const needs_sm_20 = geometry == .@"3d" or geometry == .a1d or geometry == .a2d or
            clamp != .trap;
        nvptx.require("sust.b", .{
            .ptx = if (geometry == .@"3d" or geometry == .a1d or geometry == .a2d)
                30
            else if (clamp != .trap) 20 else 15,
            .sm = .{ .base = if (needs_sm_20) 20 else 0 },
            .threads = .thread,
        });
        const lanes = @typeInfo(@TypeOf(texels)).vector.len;
        const Texel = @typeInfo(@TypeOf(texels)).vector.child;
        const width_suffix = switch (@bitSizeOf(Texel)) {
            8 => ".b8",
            16 => ".b16",
            32 => ".b32",
            64 => ".b64",
            else => @compileError("a surface element has 8, 16, 32 or 64 bits, found '" ++
                @typeName(Texel) ++ "'"),
        };
        const vector_suffix = switch (lanes) {
            1 => "",
            2 => ".v2",
            4 => ".v4",
            else => @compileError("a surface access moves 1, 2 or 4 elements"),
        };
        if (lanes == 4 and @bitSizeOf(Texel) == 64) {
            @compileError("a surface access moves at most 128 bits");
        }
        const Bits = @Int(.unsigned, @max(16, @bitSizeOf(Texel)));
        const Unsigned = @Int(.unsigned, @bitSizeOf(Texel));
        const bits: @Vector(lanes, Bits) = @as(@Vector(lanes, Unsigned), @bitCast(texels));
        const text = "sust.b." ++ @tagName(geometry) ++ vector_suffix ++
            width_suffix ++ "." ++ @tagName(clamp) ++ " [%[s], %[c]], %[v];";
        const handle = @backingInt(surface);
        switch (geometry) {
            .@"1d" => store(Bits, lanes, text, 1, handle, @bitCast(coordinates), bits),
            .@"2d" => store(Bits, lanes, text, 2, handle, @bitCast(coordinates), bits),
            .@"3d" => store(Bits, lanes, text, 4, handle, .{
                @bitCast(coordinates[0]), @bitCast(coordinates[1]), @bitCast(coordinates[2]), 0,
            }, bits),
            .a1d => store(Bits, lanes, text, 2, handle, .{
                coordinates.index, @bitCast(coordinates.position[0]),
            }, bits),
            .a2d => store(Bits, lanes, text, 4, handle, .{
                coordinates.index,
                @bitCast(coordinates.position[0]),
                @bitCast(coordinates.position[1]),
                0,
            }, bits),
            .cube, .acube => unreachable,
        }
    }

    /// Emits `sured.b` (section 9.7.13.3), which applies `operation` atomically to the element at `coordinates`.
    /// The geometry is `.@"1d"`, `.@"2d"` or `.@"3d"`. `.add` takes u32, i32 or u64, `.min` and
    /// `.max` take u32, i32, u64 or i64, and `.@"and"` and `.@"or"` take u32.
    pub inline fn suredB(
        surface: SurfaceObject,
        comptime operation: ReductionOperation,
        comptime geometry: Geometry,
        comptime T: type,
        comptime clamp: Clamp,
        coordinates: Coordinates(geometry, i32),
        value: T,
    ) void {
        const wide_extremum = (operation == .min or operation == .max) and @bitSizeOf(T) == 64;
        nvptx.require("sured.b", .{
            .ptx = if (wide_extremum) 81 else if (geometry == .@"3d") 30 else 20,
            .sm = .{ .base = if (wide_extremum) 50 else 20 },
            .threads = .thread,
        });
        const type_suffix = switch (operation) {
            .add => switch (T) {
                u32 => ".u32",
                i32 => ".s32",
                u64 => ".u64",
                else => @compileError("sured.b.add needs u32, i32 or u64"),
            },
            .min, .max => switch (T) {
                u32 => ".u32",
                i32 => ".s32",
                u64 => ".u64",
                i64 => ".s64",
                else => @compileError("sured.b.min and sured.b.max need u32, i32, u64 or i64"),
            },
            .@"and", .@"or" => if (T == u32)
                ".b32"
            else
                @compileError("sured.b.and and sured.b.or need u32"),
        };
        const text = "sured.b." ++ @tagName(operation) ++ "." ++ @tagName(geometry) ++
            type_suffix ++ "." ++ @tagName(clamp) ++ " [%[s], %[c]], %[v];";
        const handle = @backingInt(surface);
        const Bits = if (@bitSizeOf(T) == 64) u64 else u32;
        const bits: Bits = @bitCast(value);
        switch (geometry) {
            .@"1d" => reduce(Bits, text, 1, handle, @bitCast(coordinates), bits),
            .@"2d" => reduce(Bits, text, 2, handle, @bitCast(coordinates), bits),
            .@"3d" => reduce(Bits, text, 4, handle, .{
                @bitCast(coordinates[0]), @bitCast(coordinates[1]), @bitCast(coordinates[2]), 0,
            }, bits),
            else => @compileError("sured.b does not support the geometry " ++ @tagName(geometry)),
        }
    }
};

test positionLen {
    try std.testing.expectEqual(1, positionLen(.@"1d"));
    try std.testing.expectEqual(2, positionLen(.@"2d"));
    try std.testing.expectEqual(3, positionLen(.@"3d"));
    try std.testing.expectEqual(1, positionLen(.a1d));
    try std.testing.expectEqual(2, positionLen(.a2d));
    try std.testing.expectEqual(3, positionLen(.cube));
    try std.testing.expectEqual(3, positionLen(.acube));
}

test Coordinates {
    try std.testing.expect(Coordinates(.@"2d", f32) == @Vector(2, f32));
    try std.testing.expect(Coordinates(.cube, f32) == @Vector(3, f32));
    const array = Coordinates(.a2d, f32){ .index = 3, .position = .{ 1, 2 } };
    try std.testing.expectEqual(@Vector(2, f32){ 1, 2 }, array.position);
    const cube_array = Coordinates(.acube, i32){ .index = 1, .position = .{ 4, 5, 6 } };
    try std.testing.expectEqual(@Vector(3, i32){ 4, 5, 6 }, cube_array.position);
}
