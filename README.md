# Ripple

Ripple is a raytracer made completely from scratch in the C programming language.

It started by rendering simple spheres but aims to render
complex objects with varying geometry and aesthetic lighting.

## Building

This program uses the `GCC` compiler with `Make` on `Windows 11 x64-based operating system`

```
> make ripple
gcc -o ripple ripple.c vec3.c sphere.c bmp.c render.c

```

## Usage

Run the program specifying the height and width.

```
> ripple 3840 2160
Sphere:
        Memory Address: 000001888ffb7340
        Position: (0.000000 0.000000 10.000000)
        Radius: 4.000000
        Color: RGB (112, 1, 185)
Sun position: Vector3:
        Memory Address: 000001888ffb73e0
        Value: (0.000000 10.000000 0.000000)
        Magnitude: 10.000000

Width: 3840, Height: 2160
Light Intensity: 3000
Field Of View: 70
Shading Enabled: 1

Render time: 42.058000 seconds

```

You should see an `output.bmp` file

![image](output.jpg)