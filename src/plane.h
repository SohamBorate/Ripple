#ifndef PLANE_H
#define PLANE_H

#ifdef __CUDACC__
    #include <cuda_runtime.h>
    #define CUDA_HOSTDEV __host__ __device__
#else
    #define CUDA_HOSTDEV
#endif

#include <stdio.h>

#include "bmp.h"
#include "vec3.h"

typedef struct {
    vec3 normal; // plane perpendicular to this direction vector
    vec3 point; // plane passes through this position vector
    RGB color;
} plane;

CUDA_HOSTDEV void plane_print(plane p);

CUDA_HOSTDEV plane plane_new(vec3 normal, vec3 point, RGB color);

#endif