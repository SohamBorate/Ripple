#ifndef SPHERE_H
#define SPHERE_H

#ifdef __CUDACC__
    #include <cuda_runtime.h>
    #define CUDA_HOSTDEV __host__ __device__
#else
    #define CUDA_HOSTDEV
#endif

#include "bmp.h"
#include "vec3.h"

typedef struct {
    vec3 position;
    float radius;
    RGB color;
} sphere;

CUDA_HOSTDEV void sphere_print(sphere s);

CUDA_HOSTDEV sphere sphere_new(vec3 position, float radius, RGB color);

#endif
