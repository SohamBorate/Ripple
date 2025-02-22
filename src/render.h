#ifndef RENDER_H
#define RENDER_H

#include <stdint.h>

#include "base_part.h"
#include "bmp.h"
#include "config.h"
#include "data_types.h"
#include "sphere.h"
#include "vec3.h"

void render_scene(vec3 origin, vec3 sun, int num_objects, BasePart *objects, int HEIGHT, int WIDTH, RGB *pixels);

RGB render_pixel(vec3 origin, vec3 sun, int num_objects, BasePart *objects, const int ROW, const int COLUMN, const float SCENE_HEIGHT, const float SCENE_WIDTH, const float LEAST_HEIGHT, const float LEAST_WIDTH);

uint8_t validate_rgb(int n);

#endif