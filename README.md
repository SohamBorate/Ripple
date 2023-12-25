# Ripple

Ripple is a raytracer made completely from scratch in the C programming language.

It started by rendering simple spheres but aims to render
complex objects with varying geometry and aesthetic lighting.

## Building

This program uses the `GCC` compiler with `Make` on `Windows 11 x64-based operating system`

```
> make ripple
gcc -o ripple ripple.c vec3.c sphere.c bmp.c

```

## Usage

Run the program specifying the height and width.

```
> ripple 1024 1024
Sphere (0x5d5ffbe0) (0.000000 0.000000 10.000000) (4.000000)

Width: 1024, Height: 1024
Sun position: Vector3 (0x5d5ffb30) (0.000000 -10.000000 0.000000) (10.000000)

Render time: 0.366000 seconds

```

You should see an `output.bmp` file

![image](output.jpg)