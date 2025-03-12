#include <stdlib.h>
#include <stdio.h>

#include "cube.h"

void cube_print(cube c) {
    printf("Cube:\n");
    printf("\tMemory address: %p\n", &c);
    printf("\tPosition: (%f %f %f)\n", c.position.x, c.position.y, c.position.z);
    printf("\tVertex 1 (front top left) (relative to position): (%f %f %f)\n", c.vertex_1.x, c.vertex_1.y, c.vertex_1.z);
    printf("\tVertex 2 (back bottom right) (relative to position): (%f %f %f)\n", c.vertex_2.x, c.vertex_2.y, c.vertex_2.z);
    printf("\tColor: (%i, %i, %i)\n", c.color.red, c.color.green, c.color.blue);

    printf("\tSurfaces:\n\t\tFront:\n\t\t\tNormal: (%f %f %f)\n\t\t\tPoint: (%f %f %f)\n", c.surfaces.front.normal.x, c.surfaces.front.normal.y, c.surfaces.front.normal.z, c.surfaces.front.point.x, c.surfaces.front.point.y, c.surfaces.front.point.z);
    printf("\t\tBack:\n\t\t\tNormal: (%f %f %f)\n\t\t\tPoint: (%f %f %f)\n", c.surfaces.back.normal.x, c.surfaces.back.normal.y, c.surfaces.back.normal.z, c.surfaces.back.point.x, c.surfaces.back.point.y, c.surfaces.back.point.z);
    printf("\t\tTop:\n\t\t\tNormal: (%f %f %f)\n\t\t\tPoint: (%f %f %f)\n", c.surfaces.top.normal.x, c.surfaces.top.normal.y, c.surfaces.top.normal.z, c.surfaces.top.point.x, c.surfaces.top.point.y, c.surfaces.top.point.z);
    printf("\t\tBottom:\n\t\t\tNormal: (%f %f %f)\n\t\t\tPoint: (%f %f %f)\n", c.surfaces.bottom.normal.x, c.surfaces.bottom.normal.y, c.surfaces.bottom.normal.z, c.surfaces.bottom.point.x, c.surfaces.bottom.point.y, c.surfaces.bottom.point.z);
    printf("\t\tRight:\n\t\t\tNormal: (%f %f %f)\n\t\t\tPoint: (%f %f %f)\n", c.surfaces.right.normal.x, c.surfaces.right.normal.y, c.surfaces.right.normal.z, c.surfaces.right.point.x, c.surfaces.right.point.y, c.surfaces.right.point.z);
    printf("\t\tLeft:\n\t\t\tNormal: (%f %f %f)\n\t\t\tPoint: (%f %f %f)\n", c.surfaces.left.normal.x, c.surfaces.left.normal.y, c.surfaces.left.normal.z, c.surfaces.left.point.x, c.surfaces.left.point.y, c.surfaces.left.point.z);
}

cube cube_new(vec3 position, vec3 vertex_1, vec3 vertex_2, RGB color) {
    cube c;
    c.position = position;
    c.vertex_1 = vertex_1;
    c.vertex_2 = vertex_2;
    c.color = color;

    vec3 world_vertex_1 = vec3_sum(position, vertex_1);
    vec3 world_vertex_2 = vec3_sum(position, vertex_2);

    c.world_vertex_1 = world_vertex_1;
    c.world_vertex_2 = world_vertex_2;

    // surfaces connected to vertex 1 (front top left)
    // RGB color2;
    // color2.red = 200;
    // color2.green = 1;
    // color2.blue = 1;
    c.surfaces.front = plane_new(vec3_new(0.0, 0.0, 1.0), world_vertex_1, color);

    // color2.red = 1;
    // color2.green = 200;
    // color2.blue = 1;
    c.surfaces.top = plane_new(vec3_new(0.0, 1.0, 0.0), world_vertex_1, color);

    // color2.red = 1;
    // color2.green = 1;
    // color2.blue = 200;
    c.surfaces.left = plane_new(vec3_new(1.0, 0.0, 0.0), world_vertex_1, color);
    
    // surfaces connected to vertex 2 (back bottom right)
    // color2.red = 200;
    // color2.green = 200;
    // color2.blue = 1;
    c.surfaces.back = plane_new(vec3_new(0.0, 0.0, -1.0), world_vertex_2, color);

    // color2.red = 1;
    // color2.green = 200;
    // color2.blue = 200;
    c.surfaces.bottom = plane_new(vec3_new(0.0, -1.0, 0.0), world_vertex_2, color);

    // color2.red = 200;
    // color2.green = 1;
    // color2.blue = 200;
    c.surfaces.right = plane_new(vec3_new(-1.0, 0.0, 0.0), world_vertex_2, color);

    return c;
}
