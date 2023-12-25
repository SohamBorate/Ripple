#include "vec3.h"

typedef struct {
    vec3 *position;
    float radius;
} sphere;

void sphere_print(sphere *s);

sphere* sphere_new(vec3 *position, float radius);

void sphere_free(sphere *s);