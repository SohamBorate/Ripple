#include <stdlib.h>
#include <stdio.h>
#include "sphere.h"

void sphere_print(sphere *s) {
    printf("Sphere:\n\tMemory Address: %p\n\tPosition: (%f %f %f)\n\tRadius: %f\n\tColor: RGB (%i, %i, %i)\n", (void *)s, s->position->x, s->position->y, s->position->z, s->radius, s->color.red, s->color.green, s->color.blue);
}

sphere* sphere_new(vec3 *position, float radius, RGB color) {
    sphere *s = malloc(sizeof(sphere));
    if (s == NULL) {
        printf("Memory allocation for Sphere failed\n");
        return NULL;
    }
    s->position = position;
    s->radius = radius;
    s->color = color;
    return s;
}

void sphere_free(sphere *s) {
    if (s->position != NULL) {
        vec3_free(s->position);
    }
    free(s);
}