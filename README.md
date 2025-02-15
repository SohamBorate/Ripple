```
This branch is for an animation with a Windows (operating system) window (application).
Will only work on Windows (operating system)
```

# Ripple

Ripple is a raytracer made completely from scratch in the C programming language.

It started by rendering simple spheres but aims to render
complex objects with varying geometry and aesthetic lighting.

## Building

This program uses the `GCC` compiler with `Make` on `Windows 11 x64-based operating system`

```
> make ripplewindow
gcc -o ripplewindow RippleWindow.c vec3.c sphere.c BmpWindow.c -mwindows

```

## Usage

Run the program specifying the height and width.

```
> ripplewindow
```