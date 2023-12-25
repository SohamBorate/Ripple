#include <stdlib.h>
#include <stdio.h>
#include "sphere.h"

void sphere_print(sphere *s) {
    printf("Sphere (0x%x) (%f %f %f) (%f)\n", &s, s->position->x, s->position->y, s->position->z, s->radius);
}

sphere* sphere_new(vec3 *position, float radius) {
    sphere *s = malloc(sizeof(sphere));
    if (s == NULL) return NULL;
    s->position = position;
    s->radius = radius;
    return s;
}

void sphere_free(sphere *s) {
    if (s->position != NULL) {
        vec3_free(s->position);
    }
    free(s);
}