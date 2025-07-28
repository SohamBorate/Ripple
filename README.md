# Ripple

Ripple is a raytracer made completely from scratch in the C programming language.

It started by rendering simple spheres but aims to render
complex objects with varying geometry and aesthetic lighting.

## Building

This program uses the `MSVC` compiler with `Make` on `Windows 11 x64-based operating system`.

Before building, make sure there's a directory named `build`.

```
> make
cl /O2 /Wall /c src/bmp.c /Fobuild/bmp.obj
Microsoft (R) C/C++ Optimizing Compiler Version 19.38.33133 for x64
Copyright (C) Microsoft Corporation.  All rights reserved.

bmp.c
src\bmp.c(10) : warning C4710: 'printf': function not inlined
cl /O2 /Wall /c src/ripple.c /Fobuild/ripple.obj
Microsoft (R) C/C++ Optimizing Compiler Version 19.38.33133 for x64
Copyright (C) Microsoft Corporation.  All rights reserved.

ripple.c
src\sphere.h(11): warning C4820: '<unnamed-tag>': '1' bytes padding added after data member 'color'
src\plane.h(11): warning C4820: '<unnamed-tag>': '1' bytes padding added after data member 'color'
src\cube.h(30): warning C4820: '<unnamed-tag>': '1' bytes padding added after data member 'color'
src\base_part.h(15): warning C4201: nonstandard extension used: nameless struct/union
src\raycast.h(14): warning C4820: '<unnamed-tag>': '1' bytes padding added after data member 'color'
src/ripple.c(39): warning C4305: 'function': truncation from 'double' to 'float'
src/ripple.c(68): warning C4020: 'plane_new': too many actual parameters
src\ripple.c(22) : warning C4710: 'printf': function not inlined
src\ripple.c(86) : warning C4710: 'printf': function not inlined
src\ripple.c(89) : warning C4710: 'printf': function not inlined
src\ripple.c(98) : warning C4710: 'printf': function not inlined
src\ripple.c(103) : warning C4710: 'printf': function not inlined
src\ripple.c(104) : warning C4710: 'printf': function not inlined
src\ripple.c(105) : warning C4710: 'printf': function not inlined
src\ripple.c(120) : warning C4710: 'printf': function not inlined
cl /O2 /Wall /c src/sphere.c /Fobuild/sphere.obj
Microsoft (R) C/C++ Optimizing Compiler Version 19.38.33133 for x64
Copyright (C) Microsoft Corporation.  All rights reserved.

sphere.c
src\sphere.h(11): warning C4820: '<unnamed-tag>': '1' bytes padding added after data member 'color'
src\sphere.c(6) : warning C4710: 'printf': function not inlined
cl /O2 /Wall /c src/cube.c /Fobuild/cube.obj
Microsoft (R) C/C++ Optimizing Compiler Version 19.38.33133 for x64
Copyright (C) Microsoft Corporation.  All rights reserved.

cube.c
src\plane.h(11): warning C4820: '<unnamed-tag>': '1' bytes padding added after data member 'color'
src\cube.h(30): warning C4820: '<unnamed-tag>': '1' bytes padding added after data member 'color'
src/cube.c(40): warning C4020: 'plane_new': too many actual parameters
src/cube.c(45): warning C4020: 'plane_new': too many actual parameters
src/cube.c(50): warning C4020: 'plane_new': too many actual parameters
src/cube.c(56): warning C4020: 'plane_new': too many actual parameters
src/cube.c(61): warning C4020: 'plane_new': too many actual parameters
src/cube.c(66): warning C4020: 'plane_new': too many actual parameters
src\cube.c(7) : warning C4710: 'printf': function not inlined
src\cube.c(8) : warning C4710: 'printf': function not inlined
src\cube.c(9) : warning C4710: 'printf': function not inlined
src\cube.c(10) : warning C4710: 'printf': function not inlined
src\cube.c(11) : warning C4710: 'printf': function not inlined
src\cube.c(12) : warning C4710: 'printf': function not inlined
src\cube.c(14) : warning C4710: 'printf': function not inlined
src\cube.c(15) : warning C4710: 'printf': function not inlined
src\cube.c(16) : warning C4710: 'printf': function not inlined
src\cube.c(17) : warning C4710: 'printf': function not inlined
src\cube.c(18) : warning C4710: 'printf': function not inlined
src\cube.c(19) : warning C4710: 'printf': function not inlined
cl /O2 /Wall /c src/plane.c /Fobuild/plane.obj
Microsoft (R) C/C++ Optimizing Compiler Version 19.38.33133 for x64
Copyright (C) Microsoft Corporation.  All rights reserved.

plane.c
src\plane.h(11): warning C4820: '<unnamed-tag>': '1' bytes padding added after data member 'color'
src/plane.c(4): warning C4013: 'printf' undefined; assuming extern returning int
src/plane.c(11): warning C4029: declared formal parameter list different from definition
cl /O2 /Wall /c src/vec3.c /Fobuild/vec3.obj
Microsoft (R) C/C++ Optimizing Compiler Version 19.38.33133 for x64
Copyright (C) Microsoft Corporation.  All rights reserved.

vec3.c
src\vec3.c(37) : warning C4711: function 'vec3_new' selected for automatic inline expansion
src\vec3.c(44) : warning C4711: function 'vec3_new' selected for automatic inline expansion
src\vec3.c(51) : warning C4711: function 'vec3_new' selected for automatic inline expansion
src\vec3.c(58) : warning C4711: function 'vec3_new' selected for automatic inline expansion
src\vec3.c(75) : warning C4711: function 'vec3_new' selected for automatic inline expansion
src\vec3.c(26) : warning C4711: function 'vec3_magnitude' selected for automatic inline expansion
src\vec3.c(30) : warning C4711: function 'vec3_new' selected for automatic inline expansion
src\vec3.c(8) : warning C4710: 'printf': function not inlined
src\vec3.c(8) : warning C4711: function 'vec3_magnitude' selected for automatic inline expansion
cl /O2 /Wall /c src/render.c /Fobuild/render.obj
Microsoft (R) C/C++ Optimizing Compiler Version 19.38.33133 for x64
Copyright (C) Microsoft Corporation.  All rights reserved.

render.c
src\sphere.h(11): warning C4820: '<unnamed-tag>': '1' bytes padding added after data member 'color'
src\plane.h(11): warning C4820: '<unnamed-tag>': '1' bytes padding added after data member 'color'
src\cube.h(30): warning C4820: '<unnamed-tag>': '1' bytes padding added after data member 'color'
src\base_part.h(15): warning C4201: nonstandard extension used: nameless struct/union
src\raycast.h(14): warning C4820: '<unnamed-tag>': '1' bytes padding added after data member 'color'
src/render.c(13): warning C4305: 'initializing': truncation from 'double' to 'const float'
src/render.c(16): warning C4244: 'initializing': conversion from 'double' to 'const float', possible loss of data
src/render.c(17): warning C4244: 'initializing': conversion from 'double' to 'const float', possible loss of data
src/render.c(38): warning C4244: '=': conversion from 'double' to 'uint8_t', possible loss of data
src/render.c(39): warning C4244: '=': conversion from 'double' to 'uint8_t', possible loss of data
src/render.c(40): warning C4244: '=': conversion from 'double' to 'uint8_t', possible loss of data
src/render.c(43): warning C4244: 'function': conversion from 'double' to 'float', possible loss of data
src/render.c(44): warning C4244: 'function': conversion from 'double' to 'float', possible loss of data
src/render.c(81): warning C4244: 'initializing': conversion from 'double' to 'float', possible loss of data
src/render.c(106): warning C4242: 'return': conversion from 'int' to 'uint8_t', possible loss of data
src\render.c(85) : warning C4711: function 'validate_rgb' selected for automatic inline expansion
src\render.c(86) : warning C4711: function 'validate_rgb' selected for automatic inline expansion
src\render.c(87) : warning C4711: function 'validate_rgb' selected for automatic inline expansion
src\render.c(22) : warning C4710: 'printf': function not inlined
src\render.c(23) : warning C4710: 'printf': function not inlined
src\render.c(24) : warning C4710: 'printf': function not inlined
src\render.c(28) : warning C5045: Compiler will insert Spectre mitigation for memory load if /Qspectre switch specified
src\render.c(27) : note: index 'WIDTH' range checked by comparison on this line
src\render.c(28) : note: feeds call on this line
cl /O2 /Wall /c src/raycast.c /Fobuild/raycast.obj
Microsoft (R) C/C++ Optimizing Compiler Version 19.38.33133 for x64
Copyright (C) Microsoft Corporation.  All rights reserved.

raycast.c
src\sphere.h(11): warning C4820: '<unnamed-tag>': '1' bytes padding added after data member 'color'
src\plane.h(11): warning C4820: '<unnamed-tag>': '1' bytes padding added after data member 'color'
src\cube.h(30): warning C4820: '<unnamed-tag>': '1' bytes padding added after data member 'color'
src\base_part.h(15): warning C4201: nonstandard extension used: nameless struct/union
src\raycast.h(14): warning C4820: '<unnamed-tag>': '1' bytes padding added after data member 'color'
src/raycast.c(57): warning C4244: 'initializing': conversion from 'double' to 'float', possible loss of data
src\raycast.c(27) : warning C5045: Compiler will insert Spectre mitigation for memory load if /Qspectre switch specified
src\raycast.c(19) : note: index 'perp_distance' range checked by comparison on this line
src\raycast.c(27) : note: feeds call on this line
src\raycast.c(68) : warning C5045: Compiler will insert Spectre mitigation for memory load if /Qspectre switch specified
src\raycast.c(60) : note: index 'perp_distance' range checked by comparison on this line
src\raycast.c(68) : note: feeds call on this line
src\raycast.c(190) : warning C4711: function 'sphere_raycast' selected for automatic inline expansion
cl /O2 /Wall build/bmp.obj build/ripple.obj build/sphere.obj build/cube.obj build/plane.obj build/vec3.obj build/render.obj build/raycast.obj /Febuild/ripple.exe
Microsoft (R) C/C++ Optimizing Compiler Version 19.38.33133 for x64
Copyright (C) Microsoft Corporation.  All rights reserved.

Microsoft (R) Incremental Linker Version 14.38.33133.0
Copyright (C) Microsoft Corporation.  All rights reserved.

/out:build/ripple.exe
build/bmp.obj
build/ripple.obj
build/sphere.obj
build/cube.obj
build/plane.obj
build/vec3.obj
build/render.obj
build/raycast.obj

```

## Usage

Run the program specifying the width and height.

```
> build\ripple.exe 3840 2160
Sphere:
        Memory Address: 000000056DB1FBF0
        Position: (-1.000000 -0.500000 6.000000)
        Radius: 0.800000
        Color: RGB (200, 30, 30)
Cube:
        Memory address: 000000056DB1FB20
        Position: (2.000000 -1.000000 5.000000)
        Vertex 1 (front top left) (relative to position): (0.500000 0.500000 0.500000)
        Vertex 2 (back bottom right) (relative to position): (-0.500000 -0.500000 -0.500000)
        Color: (146, 112, 45)
        Surfaces:
                Front:
                        Normal: (0.000000 0.000000 1.000000)
                        Point: (2.500000 -0.500000 5.500000)
                Back:
                        Normal: (0.000000 0.000000 -1.000000)
                        Point: (1.500000 -1.500000 4.500000)
                Top:
                        Normal: (0.000000 1.000000 0.000000)
                        Point: (2.500000 -0.500000 5.500000)
                Bottom:
                        Normal: (0.000000 -1.000000 0.000000)
                        Point: (1.500000 -1.500000 4.500000)
                Right:
                        Normal: (-1.000000 0.000000 0.000000)
                        Point: (1.500000 -1.500000 4.500000)
                Left:
                        Normal: (1.000000 0.000000 0.000000)
                        Point: (2.500000 -0.500000 5.500000)
Plane:
        Memory Address: 000000056DB1FBE0
        Normal: (0.000000 1.000000 0.000000)
        Point: (0.000000 -2.000000 0.000000)
        Color: (56, 112, 45)
Sun position: Vector3:
        Memory Address: 000000056DB1FC60
        Value: (0.000000 10.000000 0.000000)
        Magnitude: 10.000000

Width: 3840, Height: 2160
Light Intensity: 5000
Field Of View: 90
Shading Enabled: 1

Aspect Ratio: 1.777778
Horizontal Field Of View in radians: 1.570796
Vertical Field Of View in radians: 0.883573
Scene Width: 2.000000 Scene Height: 0.945930
Least Scene Width: 0.000521 Least Scene Height: 0.000438


Render time: 5.570000 seconds

```

You should see an `output.bmp` file

![image](output.jpg)
