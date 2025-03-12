#ifndef RAYCAST_H
#define RAYCAST_H

#ifdef __CUDACC__
    #include <cuda_runtime.h>
    #define CUDA_DEV __device__
#else
    #define CUDA_DEV
#endif

#include "base_part.h"
#include "bmp.h"
#include "vec3.h"

typedef struct {
    int BasePartIndex; // index in "Objects" array
    float hit_distance;
    vec3 hit_position;
    vec3 hit_normal;
    RGB color;
} RaycastResult;

CUDA_DEV RaycastResult sphere_raycast(vec3 origin, vec3 direction, int object_index, sphere ball);

CUDA_DEV RaycastResult plane_raycast(vec3 origin, vec3 direction, int object_index, plane plane);

CUDA_DEV RaycastResult raycast(vec3 origin, vec3 direction, int num_objects, BasePart *objects, int ignore_object_index); // int ignore_object_index is index of the object to be ignore during raytracing

#endif