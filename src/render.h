#ifndef RENDER_H
#define RENDER_H

#ifdef __CUDACC__
    #include <cuda_runtime.h>
    #define CUDA_DEV __device__
#else
    #define CUDA_DEV
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "config.h"
#include "sphere.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

CUDA_DEV uint8_t validate_rgb(int n);

CUDA_DEV RGB render_pixel(vec3 origin, vec3 sun, sphere ball, const int ROW, const int COLUMN, const int HEIGHT, const int WIDTH);

CUDA_DEV void render_scene(vec3 origin, vec3 sun, sphere ball, int HEIGHT, int WIDTH, RGB *pixels);

#endif
