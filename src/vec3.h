#ifndef VEC3_H
#define VEC3_H

#ifdef __CUDACC__
    #include <cuda_runtime.h>
    #define CUDA_HOSTDEV __host__ __device__
#else
    #define CUDA_HOSTDEV
#endif

typedef struct {
    float x;
    float y;
    float z;
} vec3;

CUDA_HOSTDEV void vec3_print(vec3 v);

CUDA_HOSTDEV vec3 vec3_new(float x, float y, float z);

CUDA_HOSTDEV float vec3_magnitude(vec3 v);

CUDA_HOSTDEV vec3 vec3_unit(vec3 v);

CUDA_HOSTDEV vec3 vec3_sum(vec3 v1, vec3 v2);

CUDA_HOSTDEV vec3 vec3_difference(vec3 v1, vec3 v2);

CUDA_HOSTDEV vec3 vec3_scalar_product(vec3 v, float s);

CUDA_HOSTDEV vec3 vec3_vec3_product(vec3 v1, vec3 v2);

CUDA_HOSTDEV float vec3_dot_scalar(vec3 v, float s);

CUDA_HOSTDEV float vec3_dot_vec3(vec3 v1, vec3 v2);

CUDA_HOSTDEV vec3 vec3_cross_vec3(vec3 v1, vec3 v2);

#endif
