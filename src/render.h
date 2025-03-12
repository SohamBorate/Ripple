#ifndef RENDER_H
#define RENDER_H

#ifdef __CUDACC__
    #include <cuda_runtime.h>
    #define CUDA_DEV __device__
#else
    #define CUDA_DEV
#endif

#include <stdint.h>

#include "base_part.h"
#include "bmp.h"
#include "config.h"
#include "data_types.h"
#include "raycast.h"
#include "vec3.h"

CUDA_DEV void render_scene(vec3 origin, vec3 sun, int num_objects, BasePart *objects, int HEIGHT, int WIDTH, RGB *pixels);

CUDA_DEV RGB render_pixel(vec3 origin, vec3 sun, int num_objects, BasePart *objects, const int ROW, const int COLUMN, const float SCENE_HEIGHT, const float SCENE_WIDTH, const float LEAST_HEIGHT, const float LEAST_WIDTH);

CUDA_DEV uint8_t validate_rgb(int n);

#endif