#include "vec3.h"
#include "bmp.h"

typedef struct {
    vec3 *position;
    float radius;
    RGB color;
} sphere;

void sphere_print(sphere *s);

sphere* sphere_new(vec3 *position, float radius, RGB color);

void sphere_free(sphere *s);