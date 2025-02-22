# Ripple Cuda

This branch is for using GPU acceleration with CUDA.

## Building

This branch uses the `NVCC` and `MSVC` compiler with `Make` on `Windows 11 x64-based operating system`.

Before building, makes sure there's a directory named `build`.

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
src\sphere.h(14): warning C4820: '<unnamed-tag>': '1' bytes padding added after data member 'color'
src\ripple.c(15) : warning C4710: 'printf': function not inlined
src\ripple.c(59) : warning C4710: 'printf': function not inlined
src\ripple.c(62) : warning C4710: 'printf': function not inlined
src\ripple.c(70) : warning C4710: 'printf': function not inlined
src\ripple.c(75) : warning C4710: 'printf': function not inlined
src\ripple.c(76) : warning C4710: 'printf': function not inlined
src\ripple.c(77) : warning C4710: 'printf': function not inlined
src\ripple.c(91) : warning C4710: 'printf': function not inlined
cl /O2 /Wall /c src/sphere.c /Fobuild/sphere.obj
Microsoft (R) C/C++ Optimizing Compiler Version 19.38.33133 for x64
Copyright (C) Microsoft Corporation.  All rights reserved.

sphere.c
src\sphere.h(14): warning C4820: '<unnamed-tag>': '1' bytes padding added after data member 'color'
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
src\vec3.c(8) : warning C4710: 'printf': function not inlined
src\vec3.c(30) : warning C4711: function 'vec3_new' selected for automatic inline expansion
src\vec3.c(8) : warning C4711: function 'vec3_magnitude' selected for automatic inline expansion
nvcc -arch=sm_75 -gencode arch=compute_75,code=sm_75 -O2 -dc -c src/render.cu -o build/render.obj
render.cu
tmpxft_00003d44_00000000-7_render.cudafe1.cpp
nvcc -arch=sm_75 -gencode arch=compute_75,code=sm_75 -O2 -dc -c src/sphere_cuda.cu -o build/sphere_cuda.obj
sphere_cuda.cu
tmpxft_00002844_00000000-7_sphere_cuda.cudafe1.cpp
nvcc -arch=sm_75 -gencode arch=compute_75,code=sm_75 -O2 -dc -c src/vec3_cuda.cu -o build/vec3_cuda.obj
vec3_cuda.cu
tmpxft_0000455c_00000000-7_vec3_cuda.cudafe1.cpp
nvcc -arch=sm_75 -gencode arch=compute_75,code=sm_75 -O2 -dlink build/render.obj build/sphere_cuda.obj build/vec3_cuda.obj -o build/cuda_link.obj
render.obj
sphere_cuda.obj
vec3_cuda.obj
nvcc -arch=sm_75 -gencode arch=compute_75,code=sm_75 -O2 build/bmp.obj build/ripple.obj build/sphere.obj build/vec3.obj build/render.obj build/sphere_cuda.obj build/vec3_cuda.obj build/cuda_link.obj -o build/ripple.exe -lcudart
bmp.obj
ripple.obj
sphere.obj
vec3.obj
render.obj
sphere_cuda.obj
vec3_cuda.obj
cuda_link.obj

```

## Usage

Run the program specifying the width and height.

```
> build\ripple.exe 3840 2160
Sphere:
        Memory Address: 00000090906FFC90
        Position: (0.000000 0.000000 10.000000)
        Radius: 4.000000
        Color: RGB (112, 1, 185)
Sun position: Vector3:
        Memory Address: 00000090906FFCF0
        Value: (0.000000 10.000000 0.000000)
        Magnitude: 10.000000

Width: 3840, Height: 2160
Light Intensity: 3000
Field Of View: 70
Shading Enabled: 1

Reporting from the GPU

Sphere:
        Memory Address: 00000090906FFB10
        Position: (0.000000 0.000000 10.000000)
        Radius: 4.000000
        Color: RGB (112, 1, 185)
Sun position: Vector3:
        Memory Address: 00000090906FFB70
        Value: (0.000000 10.000000 0.000000)
        Magnitude: 10.000000

Width: 3840, Height: 2160
Light Intensity: 3000
Field Of View: 70
Shading Enabled: 1

Render time: 0.243000 seconds

```

You should see an `output.bmp` file

![image](output.jpg)