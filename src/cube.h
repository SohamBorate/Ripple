#ifndef CUBE_H
#define CUBE_H

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

    cube_surfaces surfaces; // normal vectors of all the surfaces

    RGB color;
} cube;

void cube_print(cube c);

cube cube_new(vec3 position, vec3 vertex_1, vec3 vertex_2, RGB color);

#endif