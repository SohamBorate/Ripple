#ifndef PLANE_H
#define PLANE_H

#include "bmp.h"
#include "vec3.h"

typedef struct {
    vec3 normal; // plane perpendicular to this direction vector
    vec3 point; // plane passes through this position vector
    RGB color;
} plane;

void plane_print(plane p);

plane plane_new(vec3 normal, vec3 point);

#endif