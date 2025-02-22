#ifndef SPHERE_H
#define SPHERE_H

#include "bmp.h"
#include "vec3.h"

typedef struct {
    vec3 position;
    float radius;
    RGB color;
} sphere;

void sphere_print(sphere s);

sphere sphere_new(vec3 position, float radius, RGB color);

#endif