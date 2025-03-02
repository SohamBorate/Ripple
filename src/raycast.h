#ifndef RAYCAST_H
#define RAYCAST_H

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

RaycastResult sphere_raycast(vec3 origin, vec3 direction, int object_index, sphere ball);

RaycastResult plane_raycast(vec3 origin, vec3 direction, int object_index, plane plane);

RaycastResult raycast(vec3 origin, vec3 direction, int num_objects, BasePart *objects, int ignore_object_index); // int ignore_object_index is index of the object to be ignore during raytracing

#endif