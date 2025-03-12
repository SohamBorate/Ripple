#ifndef CUBE_H
#define CUBE_H

#ifdef __CUDACC__
    #include <cuda_runtime.h>
    #define CUDA_HOSTDEV __host__ __device__
#else
    #define CUDA_HOSTDEV
#endif

#include "bmp.h"
#include "vec3.h"
#include "plane.h"

typedef struct {
    plane front;
    plane back;
    plane top;
    plane bottom;
    plane right;
    plane left;
} cube_surfaces;

// this cube architecture will probably be tweaked a bit in the future

typedef struct {
    vec3 position; // anchor point of base part
    vec3 vertex_1; // front top left, relative to position, not origin
    vec3 vertex_2; // back bottom right, relative to position, not origin

    vec3 world_vertex_1; // front top left, relative to origin, not position
    vec3 world_vertex_2; // back bottom right, relative to origin, not position

    cube_surfaces surfaces; // normal vectors of all the surfaces

    RGB color;
} cube;

CUDA_HOSTDEV void cube_print(cube c);

CUDA_HOSTDEV cube cube_new(vec3 position, vec3 vertex_1, vec3 vertex_2, RGB color);

#endif