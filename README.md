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
src\sphere.h(8): warning C4820: '<unnamed-tag>': '1' bytes padding added after data member 'color'
src\ripple.c(12) : warning C4710: 'printf': function not inlined
src\ripple.c(56) : warning C4710: 'printf': function not inlined
src\ripple.c(59) : warning C4710: 'printf': function not inlined
src\ripple.c(68) : warning C4710: 'printf': function not inlined
src\ripple.c(73) : warning C4710: 'printf': function not inlined
src\ripple.c(74) : warning C4710: 'printf': function not inlined
src\ripple.c(75) : warning C4710: 'printf': function not inlined
src\ripple.c(89) : warning C4710: 'printf': function not inlined
cl /O2 /Wall /c src/sphere.c /Fobuild/sphere.obj
Microsoft (R) C/C++ Optimizing Compiler Version 19.38.33133 for x64
Copyright (C) Microsoft Corporation.  All rights reserved.

sphere.c
src\sphere.h(8): warning C4820: '<unnamed-tag>': '1' bytes padding added after data member 'color'
src\sphere.c(6) : warning C4710: 'printf': function not inlined
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
src\sphere.h(8): warning C4820: '<unnamed-tag>': '1' bytes padding added after data member 'color'
src/render.c(36): warning C4244: 'function': conversion from 'double' to 'float', possible loss of data
src/render.c(37): warning C4244: 'function': conversion from 'double' to 'float', possible loss of data
src/render.c(71): warning C4244: 'initializing': conversion from 'double' to 'float', possible loss of data
src/render.c(87): warning C4242: 'return': conversion from 'int' to 'uint8_t', possible loss of data
src\render.c(74) : warning C4711: function 'validate_rgb' selected for automatic inline expansion
src\render.c(75) : warning C4711: function 'validate_rgb' selected for automatic inline expansion
src\render.c(76) : warning C4711: function 'validate_rgb' selected for automatic inline expansion
src\render.c(64) : warning C5045: Compiler will insert Spectre mitigation for memory load if /Qspectre switch specified
src\render.c(52) : note: index 'perp_dist' range checked by comparison on this line
src\render.c(64) : note: feeds call on this line
src\render.c(15) : warning C5045: Compiler will insert Spectre mitigation for memory load if /Qspectre switch specified
src\render.c(13) : note: index 'HEIGHT' range checked by comparison on this line
src\render.c(15) : note: feeds call on this line
src\render.c(15) : warning C5045: Compiler will insert Spectre mitigation for memory load if /Qspectre switch specified
src\render.c(14) : note: index 'WIDTH' range checked by comparison on this line
src\render.c(15) : note: feeds call on this line
cl /O2 /Wall build/bmp.obj build/ripple.obj build/sphere.obj build/vec3.obj build/render.obj /Febuild/ripple.exe
Microsoft (R) C/C++ Optimizing Compiler Version 19.38.33133 for x64
Copyright (C) Microsoft Corporation.  All rights reserved.

Microsoft (R) Incremental Linker Version 14.38.33133.0
Copyright (C) Microsoft Corporation.  All rights reserved.

/out:build/ripple.exe
build/bmp.obj
build/ripple.obj
build/sphere.obj
build/vec3.obj
build/render.obj

```

## Usage

Run the program specifying the width and height.

```
> build\ripple.exe 3840 2160
Sphere:
        Memory Address: 0000002F260FFAE0
        Position: (0.000000 0.000000 20.000000)
        Radius: 4.000000
        Color: RGB (112, 1, 185)
Sphere:
        Memory Address: 0000002F260FFAE0
        Position: (3.000000 0.000000 12.000000)
        Radius: 1.000000
        Color: RGB (1, 112, 185)
Sphere:
        Memory Address: 0000002F260FFAE0
        Position: (-15.000000 -2.000000 20.000000)
        Radius: 10.000000
        Color: RGB (146, 112, 185)
Sun position: Vector3:
        Memory Address: 0000002F260FFB50
        Value: (10.000000 0.000000 0.000000)
        Magnitude: 10.000000

Width: 3840, Height: 2160
Light Intensity: 6000
Field Of View: 70
Shading Enabled: 1

Aspect Ratio: 1.777778
Horizontal Field Of View in radians: 1.221730
Vertical Field Of View in radians: 0.687223
Scene Width: 1.400415 Scene Height: 0.715611
Least Scene Width: 0.000365 Least Scene Height: 0.000331


Render time: 4.070000 seconds

```

You should see an `output.bmp` file

![image](output.jpg)