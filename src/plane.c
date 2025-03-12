#include "plane.h"

CUDA_HOSTDEV void plane_print(plane p) {
    printf("Plane:\n");
    printf("\tMemory Address: %p\n", &p);
    printf("\tNormal: (%f %f %f)\n", p.normal.x, p.normal.y, p.normal.z);
    printf("\tPoint: (%f %f %f)\n", p.point.x, p.point.y, p.point.z);
    printf("\tColor: (%i, %i, %i)\n", p.color.red, p.color.green, p.color.blue);
}

CUDA_HOSTDEV plane plane_new(vec3 normal, vec3 point, RGB color) {
    plane p;
    p.normal = normal;
    p.point = point;
    p.color = color;
    return p;
}
